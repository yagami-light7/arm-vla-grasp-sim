"""PCT + SCAN 导航安全状态机的依赖无关核心。

本模块故意不导入 ``rclpy``，只处理可确定性测试的状态与安全决策。ROS 2
adapter 负责以下边界：

* 从 ``/body_pose``、``/cloud_registered`` 和 ``/planning/bspline`` 验证
  header、frame、QoS 与消息内容，再以同一 ROS 时钟调用本状态机；
* 把 PCT 全局规划结果、SCAN 连续规划结果、轨迹预测碰撞和 controller
  ``/planning/goal_reached`` 转换为对应的 ``report_*`` 事件；
* 根据 :class:`SupervisorDecision` 发布 ``/navigation/status``，并按
  ``global_replan_request_id`` 幂等请求 PCT 重规划；
* 在唯一的 ``/cmd_vel`` 发布所有者内执行命令门控：只有
  ``allow_tracking_command`` 为真时才转发 controller 命令，否则持续输出
  ``velocity_override``。禁止另起第二个零速 publisher 与 controller 抢写。

SCAN 当前没有规划失败或预测碰撞的标准 ROS 2 消息，因此这些信号必须由
后续 planner/controller adapter 显式提供，不能从“长时间没收到 B-spline”
猜测为碰撞。B-spline、Odometry 或点云超时只会进入安全停车；只有连续
SCAN 失败、预测碰撞或全局规划失败会产生 PCT 重规划请求。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


ZERO_BODY_VELOCITY = (0.0, 0.0, 0.0)


class NavigationState(str, Enum):
    """PCT + SCAN 在线导航状态。"""

    IDLE = "IDLE"
    GLOBAL_PLANNING = "GLOBAL_PLANNING"
    LOCAL_PLANNING = "LOCAL_PLANNING"
    TRACKING = "TRACKING"
    GLOBAL_REPLAN = "GLOBAL_REPLAN"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    GOAL_REACHED = "GOAL_REACHED"


@dataclass(frozen=True)
class NavigationSupervisorConfig:
    """导航状态机的超时与失败阈值。"""

    odometry_timeout_s: float = 0.50
    point_cloud_timeout_s: float = 0.50
    bspline_timeout_s: float = 1.00
    max_consecutive_scan_failures: int = 5

    def __post_init__(self) -> None:
        for name, value in (
            ("odometry_timeout_s", self.odometry_timeout_s),
            ("point_cloud_timeout_s", self.point_cloud_timeout_s),
            ("bspline_timeout_s", self.bspline_timeout_s),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} 必须是有限正数。")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} 必须是有限正数。") from exc
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} 必须是有限正数。")
        if (
            isinstance(self.max_consecutive_scan_failures, bool)
            or not isinstance(self.max_consecutive_scan_failures, int)
            or self.max_consecutive_scan_failures < 1
        ):
            raise ValueError("max_consecutive_scan_failures 必须是正整数。")


@dataclass(frozen=True)
class SupervisorDecision:
    """交给 ROS 2 adapter 的单拍安全决策。

    ``velocity_override`` 为 ``None`` 时，adapter 才能转发 SCAN controller
    的 ``vx、vy、wz``。其余状态始终返回机体系零速度，其中
    ``GOAL_REACHED`` 是锁存状态，只有新目标或显式 ``cancel`` 才会解除。
    """

    state: NavigationState
    allow_tracking_command: bool
    velocity_override: tuple[float, float, float] | None
    global_replan_requested: bool
    global_replan_in_flight: bool
    global_replan_request_id: int
    reason: str
    stale_inputs: tuple[str, ...]
    consecutive_scan_failures: int
    state_revision: int

    @property
    def force_zero_velocity(self) -> bool:
        """当前拍是否必须用零速度覆盖 controller 命令。"""

        return self.velocity_override == ZERO_BODY_VELOCITY

    def to_status_dict(self) -> dict[str, Any]:
        """返回可映射到 ``/navigation/status`` 的纯 Python 字段。"""

        return {
            "state": self.state.value,
            "allow_tracking_command": self.allow_tracking_command,
            "force_zero_velocity": self.force_zero_velocity,
            "velocity_override": self.velocity_override,
            "global_replan_requested": self.global_replan_requested,
            "global_replan_in_flight": self.global_replan_in_flight,
            "global_replan_request_id": self.global_replan_request_id,
            "reason": self.reason,
            "stale_inputs": list(self.stale_inputs),
            "consecutive_scan_failures": self.consecutive_scan_failures,
            "state_revision": self.state_revision,
        }


class NavigationSupervisor:
    """协调 PCT、SCAN 和闭环 controller 的安全状态机。"""

    def __init__(
        self,
        config: NavigationSupervisorConfig | None = None,
    ) -> None:
        self.config = config or NavigationSupervisorConfig()
        self._state = NavigationState.IDLE
        self._reason = "idle"
        self._clock_s: float | None = None
        self._state_revision = 0
        self._consecutive_scan_failures = 0
        self._odometry_observed_at_s: float | None = None
        self._point_cloud_observed_at_s: float | None = None
        self._bspline_observed_at_s: float | None = None
        self._bspline_valid_until_s: float | None = None
        self._global_path_available = False
        self._global_replan_pending = False
        self._global_replan_in_flight = False
        self._global_replan_request_id = 0
        self._emergency_resume_state: NavigationState | None = None

    @property
    def state(self) -> NavigationState:
        """返回当前导航状态。"""

        return self._state

    @property
    def global_replan_request_id(self) -> int:
        """返回单调递增的 PCT 重规划请求编号。"""

        return self._global_replan_request_id

    def start_goal(self, now_s: float) -> SupervisorDecision:
        """开始一个新目标并请求首次 PCT 全局规划。"""

        now = self._advance_clock(now_s)
        self._global_path_available = False
        self._invalidate_bspline()
        self._consecutive_scan_failures = 0
        self._global_replan_pending = False
        self._global_replan_in_flight = False
        self._emergency_resume_state = None
        self._transition(
            NavigationState.GLOBAL_PLANNING,
            reason="new_goal",
        )
        return self._decision(now)

    def cancel(self, now_s: float) -> SupervisorDecision:
        """取消当前目标并锁存零速度的 IDLE 状态。"""

        now = self._advance_clock(now_s)
        self._global_path_available = False
        self._invalidate_bspline()
        self._consecutive_scan_failures = 0
        self._global_replan_pending = False
        self._global_replan_in_flight = False
        self._emergency_resume_state = None
        self._transition(NavigationState.IDLE, reason="cancelled")
        return self._decision(now)

    def report_global_path_available(self, now_s: float) -> SupervisorDecision:
        """报告 PCT 已发布可用全局三维路径。"""

        now = self._advance_clock(now_s)
        allowed = {
            NavigationState.GLOBAL_PLANNING,
            NavigationState.GLOBAL_REPLAN,
            NavigationState.EMERGENCY_STOP,
        }
        if self._state not in allowed:
            raise RuntimeError(
                f"{self._state.value} 状态不能接收新的 PCT 全局路径。"
            )
        if (
            self._state is NavigationState.EMERGENCY_STOP
            and not self._global_replan_pending
        ):
            raise RuntimeError("非重规划型 EMERGENCY_STOP 不能直接接收全局路径。")
        self._global_path_available = True
        self._invalidate_path_generation_sensors()
        self._invalidate_bspline()
        self._consecutive_scan_failures = 0
        self._global_replan_pending = False
        self._global_replan_in_flight = False
        self._emergency_resume_state = None
        self._transition(
            NavigationState.LOCAL_PLANNING,
            reason="global_path_available",
        )
        return self._decision(now)

    def report_global_planning_failed(
        self,
        now_s: float,
        *,
        reason: str = "global_planning_failed",
    ) -> SupervisorDecision:
        """报告 PCT 全局规划失败，并保持零速请求重新规划。"""

        now = self._advance_clock(now_s)
        if self._state is not NavigationState.GLOBAL_PLANNING:
            raise RuntimeError("只有 GLOBAL_PLANNING 能报告全局规划失败。")
        self._global_path_available = False
        self._invalidate_bspline()
        self._request_global_replan()
        self._transition(
            NavigationState.GLOBAL_REPLAN,
            reason=self._normalize_reason(reason, "global_planning_failed"),
        )
        return self._decision(now)

    def report_global_planning_started(
        self,
        now_s: float,
    ) -> SupervisorDecision:
        """确认 PCT 已开始处理当前重规划请求。"""

        now = self._advance_clock(now_s)
        if self._state not in {
            NavigationState.GLOBAL_REPLAN,
            NavigationState.EMERGENCY_STOP,
        }:
            raise RuntimeError("当前没有可开始的 PCT 重规划请求。")
        if not self._global_replan_pending:
            raise RuntimeError("当前状态没有待处理的 PCT 重规划请求。")
        self._global_replan_pending = False
        self._global_replan_in_flight = True
        self._emergency_resume_state = None
        self._transition(
            NavigationState.GLOBAL_PLANNING,
            reason="global_replanning",
        )
        return self._decision(now)

    def report_global_replan_transport_failed(
        self,
        now_s: float,
        *,
        reason: str = "global_replan_transport_failed",
    ) -> SupervisorDecision:
        """在有界服务事务确定失败后锁存急停且终止请求。

        该事件不同于 PCT 返回 ``NO_PATH``：后者表示请求已被 PCT 接收并
        完成，可以分配下一次全局重规划；transport terminal error 表示
        当前请求的结果身份不可信，必须停止自动重试并等待外部恢复或新目标。
        """

        now = self._advance_clock(now_s)
        if self._state not in {
            NavigationState.GLOBAL_REPLAN,
            NavigationState.GLOBAL_PLANNING,
            NavigationState.EMERGENCY_STOP,
        }:
            raise RuntimeError("当前没有可终止的 PCT 重规划事务。")
        if not (
            self._global_replan_pending or self._global_replan_in_flight
        ):
            raise RuntimeError("当前没有待处理或执行中的 PCT 重规划事务。")
        self._global_path_available = False
        self._invalidate_bspline()
        self._global_replan_pending = False
        self._global_replan_in_flight = False
        self._emergency_resume_state = None
        self._transition(
            NavigationState.EMERGENCY_STOP,
            reason=self._normalize_reason(
                reason,
                "global_replan_transport_failed",
            ),
        )
        return self._decision(now)

    def observe_odometry(
        self,
        now_s: float,
        *,
        observed_at_s: float | None = None,
    ) -> SupervisorDecision:
        """记录已通过 adapter 校验的 Odometry 源时间。"""

        now = self._advance_clock(now_s)
        observed = (
            now
            if observed_at_s is None
            else self._finite_time(observed_at_s, name="observed_at_s")
        )
        if observed > now:
            raise ValueError("Odometry 源时间不能晚于 supervisor 当前时间。")
        self._odometry_observed_at_s = observed
        self._maybe_enter_tracking(now)
        return self._decision(now)

    def observe_point_cloud(
        self,
        now_s: float,
        *,
        observed_at_s: float | None = None,
    ) -> SupervisorDecision:
        """记录已通过 adapter 校验的 PointCloud2 源时间。"""

        now = self._advance_clock(now_s)
        observed = (
            now
            if observed_at_s is None
            else self._finite_time(observed_at_s, name="observed_at_s")
        )
        if observed > now:
            raise ValueError("PointCloud2 源时间不能晚于 supervisor 当前时间。")
        self._point_cloud_observed_at_s = observed
        self._maybe_enter_tracking(now)
        return self._decision(now)

    def report_scan_success(
        self,
        now_s: float,
        *,
        valid_until_s: float | None = None,
    ) -> SupervisorDecision:
        """记录 SCAN 发布的有效 B-spline。

        ``valid_until_s`` 应由 adapter 用
        ``knots[pos_pts.size] - knots[order]`` 计算有效时长，再加到
        ``Bspline.start_time``；未提供时使用 ``bspline_timeout_s`` 作为
        保守的接收超时。
        """

        now = self._advance_clock(now_s)
        if self._state not in {
            NavigationState.LOCAL_PLANNING,
            NavigationState.TRACKING,
            NavigationState.EMERGENCY_STOP,
        }:
            raise RuntimeError(
                f"{self._state.value} 状态不能接收 SCAN B-spline。"
            )
        if (
            self._state is NavigationState.EMERGENCY_STOP
            and self._global_replan_pending
        ):
            raise RuntimeError("等待 PCT 重规划时不能恢复旧 SCAN 轨迹。")
        if valid_until_s is None:
            valid_until = now + float(self.config.bspline_timeout_s)
        else:
            valid_until = self._finite_time(
                valid_until_s,
                name="valid_until_s",
            )
            if valid_until < now:
                raise ValueError("valid_until_s 不能早于 B-spline 接收时间。")
        self._bspline_observed_at_s = now
        self._bspline_valid_until_s = valid_until
        self._consecutive_scan_failures = 0
        if self._state is not NavigationState.EMERGENCY_STOP:
            self._maybe_enter_tracking(now)
        return self._decision(now)

    def extend_scan_trajectory_validity(
        self,
        now_s: float,
        *,
        valid_until_s: float,
    ) -> SupervisorDecision:
        """把当前 SCAN 轨迹提升到同一身份预先计算的绝对硬截止。"""

        now = self._advance_clock(now_s)
        if self._state not in {
            NavigationState.LOCAL_PLANNING,
            NavigationState.TRACKING,
            NavigationState.EMERGENCY_STOP,
        }:
            raise RuntimeError(
                f"{self._state.value} 状态不能扩展 SCAN 轨迹有效期。"
            )
        if self._bspline_valid_until_s is None:
            raise RuntimeError("尚未登记 SCAN B-spline，不能扩展有效期。")
        requested = self._finite_time(
            valid_until_s,
            name="valid_until_s",
        )
        if requested < now:
            raise ValueError("扩展后的 valid_until_s 不能早于当前时间。")
        # ControllerStatus 心跳只能把期限提升到 adapter 由不可变身份算出的
        # 同一个绝对上界；重复或状态抖动绝不能滚动续期，也不能缩短期限。
        self._bspline_valid_until_s = max(
            self._bspline_valid_until_s,
            requested,
        )
        return self._decision(now)

    def report_local_trajectory_finished(
        self,
        now_s: float,
    ) -> SupervisorDecision:
        """报告非最终局部轨迹自然结束并等待下一条 SCAN 轨迹。

        滚动局部规划中的一段轨迹结束不是急停，也不应触发 PCT 重规划。
        但在下一条有效 B-spline 被 controller 接受前，supervisor 必须回到
        ``LOCAL_PLANNING`` 并强制零速，不能继续沿用已经结束的轨迹授权。
        """

        now = self._advance_clock(now_s)
        if self._state not in {
            NavigationState.LOCAL_PLANNING,
            NavigationState.TRACKING,
        }:
            raise RuntimeError(
                f"{self._state.value} 状态不能报告局部轨迹自然结束。"
            )
        if not self._global_path_available:
            raise RuntimeError("没有有效全局 Path 时不能等待下一条局部轨迹。")
        if self._state is NavigationState.TRACKING:
            self._invalidate_bspline()
            self._transition(
                NavigationState.LOCAL_PLANNING,
                reason="local_trajectory_finished",
            )
        return self._decision(now)

    def report_scan_failure(
        self,
        now_s: float,
        *,
        reason: str = "scan_planning_failed",
    ) -> SupervisorDecision:
        """累计 SCAN 规划失败，达到阈值后立即停车并请求 PCT 重规划。"""

        now = self._advance_clock(now_s)
        if self._state not in {
            NavigationState.LOCAL_PLANNING,
            NavigationState.TRACKING,
        }:
            raise RuntimeError(
                f"{self._state.value} 状态不能累计 SCAN 规划失败。"
            )
        self._consecutive_scan_failures += 1
        failure_reason = self._normalize_reason(reason, "scan_planning_failed")
        if (
            self._consecutive_scan_failures
            >= self.config.max_consecutive_scan_failures
        ):
            self._global_path_available = False
            self._invalidate_bspline()
            self._request_global_replan()
            self._transition(
                NavigationState.GLOBAL_REPLAN,
                reason=(
                    f"{failure_reason}:"
                    f"{self._consecutive_scan_failures}"
                ),
            )
        else:
            self._reason = (
                f"{failure_reason}:"
                f"{self._consecutive_scan_failures}/"
                f"{self.config.max_consecutive_scan_failures}"
            )
        return self._decision(now)

    def report_predicted_collision(
        self,
        now_s: float,
        *,
        reason: str = "predicted_collision",
    ) -> SupervisorDecision:
        """预测到轨迹碰撞时立即急停并请求 PCT 重规划。"""

        now = self._advance_clock(now_s)
        if self._state in {
            NavigationState.IDLE,
            NavigationState.GOAL_REACHED,
        }:
            return self._decision(now)
        self._global_path_available = False
        self._invalidate_bspline()
        self._request_global_replan()
        self._emergency_resume_state = None
        self._transition(
            NavigationState.EMERGENCY_STOP,
            reason=self._normalize_reason(reason, "predicted_collision"),
        )
        return self._decision(now)

    def report_emergency_stop(
        self,
        now_s: float,
        *,
        reason: str,
        request_global_replan: bool = False,
    ) -> SupervisorDecision:
        """接收外部安全模块的急停事件。"""

        now = self._advance_clock(now_s)
        if self._state in {
            NavigationState.IDLE,
            NavigationState.GOAL_REACHED,
        }:
            return self._decision(now)
        resume_state = (
            self._emergency_resume_state
            if self._state is NavigationState.EMERGENCY_STOP
            else self._state
        )
        if request_global_replan:
            self._global_path_available = False
            self._invalidate_bspline()
            self._request_global_replan()
            resume_state = None
        self._emergency_resume_state = resume_state
        self._transition(
            NavigationState.EMERGENCY_STOP,
            reason=self._normalize_reason(reason, "external_emergency"),
        )
        return self._decision(now)

    def clear_emergency(self, now_s: float) -> SupervisorDecision:
        """在输入恢复且无需 PCT 重规划时解除外部/超时急停。"""

        now = self._advance_clock(now_s)
        if self._state is not NavigationState.EMERGENCY_STOP:
            raise RuntimeError("只有 EMERGENCY_STOP 能执行 clear_emergency。")
        if self._global_replan_pending:
            raise RuntimeError("存在 PCT 重规划请求时不能直接解除急停。")
        resume_state = self._emergency_resume_state
        if resume_state is None:
            raise RuntimeError("该急停没有可直接恢复的导航状态。")
        if (
            resume_state is NavigationState.TRACKING
            and self._stale_inputs(now)
        ):
            raise RuntimeError("Odometry、点云或 B-spline 尚未全部恢复。")
        self._emergency_resume_state = None
        self._transition(resume_state, reason="emergency_cleared")
        self._maybe_enter_tracking(now)
        return self._decision(now)

    def report_goal_reached(self, now_s: float) -> SupervisorDecision:
        """锁存 GOAL_REACHED；后续 tick 永远持续输出零速度。"""

        now = self._advance_clock(now_s)
        if self._state is NavigationState.GOAL_REACHED:
            return self._decision(now)
        if self._state is not NavigationState.TRACKING:
            raise RuntimeError("只有 TRACKING 能报告 GOAL_REACHED。")
        self._global_path_available = False
        self._global_replan_pending = False
        self._global_replan_in_flight = False
        self._consecutive_scan_failures = 0
        self._invalidate_bspline()
        self._emergency_resume_state = None
        self._transition(
            NavigationState.GOAL_REACHED,
            reason="goal_reached",
        )
        return self._decision(now)

    def tick(self, now_s: float) -> SupervisorDecision:
        """推进超时检查，但不推进仿真或轨迹时间。"""

        now = self._advance_clock(now_s)
        self._maybe_enter_tracking(now)
        if self._state is NavigationState.TRACKING:
            stale_inputs = self._stale_inputs(now)
            if stale_inputs:
                self._emergency_resume_state = NavigationState.TRACKING
                self._transition(
                    NavigationState.EMERGENCY_STOP,
                    reason="input_timeout:" + ",".join(stale_inputs),
                )
        return self._decision(now)

    def _maybe_enter_tracking(self, now_s: float) -> None:
        if (
            self._state is NavigationState.LOCAL_PLANNING
            and self._global_path_available
            and not self._stale_inputs(now_s)
        ):
            self._transition(
                NavigationState.TRACKING,
                reason="tracking_inputs_ready",
            )

    def _decision(self, now_s: float) -> SupervisorDecision:
        allow_tracking = (
            self._state is NavigationState.TRACKING
            and not self._stale_inputs(now_s)
        )
        return SupervisorDecision(
            state=self._state,
            allow_tracking_command=allow_tracking,
            velocity_override=None if allow_tracking else ZERO_BODY_VELOCITY,
            global_replan_requested=self._global_replan_pending,
            global_replan_in_flight=self._global_replan_in_flight,
            global_replan_request_id=self._global_replan_request_id,
            reason=self._reason,
            stale_inputs=self._stale_inputs(now_s),
            consecutive_scan_failures=self._consecutive_scan_failures,
            state_revision=self._state_revision,
        )

    def _stale_inputs(self, now_s: float) -> tuple[str, ...]:
        stale: list[str] = []
        if not self._is_recent(
            self._odometry_observed_at_s,
            now_s,
            self.config.odometry_timeout_s,
        ):
            stale.append("odometry")
        if not self._is_recent(
            self._point_cloud_observed_at_s,
            now_s,
            self.config.point_cloud_timeout_s,
        ):
            stale.append("point_cloud")
        if (
            self._bspline_observed_at_s is None
            or self._bspline_valid_until_s is None
            or now_s > self._bspline_valid_until_s
        ):
            stale.append("bspline")
        return tuple(stale)

    @staticmethod
    def _is_recent(
        observed_at_s: float | None,
        now_s: float,
        timeout_s: float,
    ) -> bool:
        return (
            observed_at_s is not None
            and now_s >= observed_at_s
            and now_s - observed_at_s <= float(timeout_s)
        )

    def _request_global_replan(self) -> None:
        if not self._global_replan_pending:
            self._global_replan_request_id += 1
        self._global_replan_pending = True
        self._global_replan_in_flight = False

    def _invalidate_bspline(self) -> None:
        self._bspline_observed_at_s = None
        self._bspline_valid_until_s = None

    def _invalidate_path_generation_sensors(self) -> None:
        """要求新 Path 代次在接受后重新取得 Odometry 与点云证据."""
        self._odometry_observed_at_s = None
        self._point_cloud_observed_at_s = None

    def _transition(
        self,
        state: NavigationState,
        *,
        reason: str,
    ) -> None:
        if self._state is not state:
            self._state_revision += 1
        self._state = state
        self._reason = reason

    def _advance_clock(self, now_s: float) -> float:
        now = self._finite_time(now_s, name="now_s")
        if self._clock_s is not None and now < self._clock_s:
            raise ValueError(
                "supervisor 时钟不能倒退；ROS 2 adapter 必须使用同一连续时钟。"
            )
        self._clock_s = now
        return now

    @staticmethod
    def _finite_time(value: float, *, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} 必须是有限非负秒数。")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是有限非负秒数。") from exc
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} 必须是有限非负秒数。")
        return result

    @staticmethod
    def _normalize_reason(reason: str, fallback: str) -> str:
        normalized = str(reason).strip()
        return normalized or fallback


__all__ = [
    "NavigationState",
    "NavigationSupervisor",
    "NavigationSupervisorConfig",
    "SupervisorDecision",
    "ZERO_BODY_VELOCITY",
]
