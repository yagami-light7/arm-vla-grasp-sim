"""加载并校验标准 Go2 的 MoE-CTS TorchScript locomotion policy。

该模块只负责“单帧观测 -> 关节动作”的模型边界，不负责 ROS 2 Topic、速度
限幅、机器人状态读取或物理推进。这样 ``CmdVelToPolicyAdapter`` 仍然是唯一的
速度安全门，而本类只处理新旧 policy 文件格式不同的问题。

Torch 使用延迟导入：普通 ROS 2/Python 单元测试导入本模块时不需要安装 Torch，
只有真正创建 ``MoeCtsPolicyAdapter`` 时才需要进入 Isaac/Conda 环境。
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MoeCtsPolicyError(RuntimeError):
    """MoE-CTS policy 加载或推理失败的基类。"""


class MoeCtsPolicyLoadError(MoeCtsPolicyError):
    """TorchScript 文件不存在、不可读或无法加载。"""


class MoeCtsPolicyContractError(MoeCtsPolicyError):
    """模型、观测、历史或动作不满足约定维度。"""


class MoeCtsPolicyInferenceError(MoeCtsPolicyError):
    """模型 forward 调用自身抛出异常。"""


@dataclass(frozen=True, slots=True)
class MoeCtsPolicyContract:
    """定义当前预训练 MoE-CTS 模型的不可变接口合同。"""

    observation_group_name: str = "single_obs"
    batch_size: int = 1
    single_observation_dim: int = 45
    history_dim: int = 450
    action_dim: int = 12

    def __post_init__(self) -> None:
        if (
            not isinstance(self.observation_group_name, str)
            or not self.observation_group_name.strip()
        ):
            raise ValueError("observation_group_name 必须是非空字符串。")
        object.__setattr__(
            self,
            "observation_group_name",
            self.observation_group_name.strip(),
        )
        for field_name in (
            "batch_size",
            "single_observation_dim",
            "history_dim",
            "action_dim",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} 必须是正整数。")

    @property
    def observation_shape(self) -> tuple[int, int]:
        """返回模型期望的单帧观测 shape。"""

        return self.batch_size, self.single_observation_dim

    @property
    def history_shape(self) -> tuple[int, int]:
        """返回模型内部 CTS 历史 shape。"""

        return self.batch_size, self.history_dim

    @property
    def action_shape(self) -> tuple[int, int]:
        """返回模型输出动作 shape。"""

        return self.batch_size, self.action_dim


@dataclass(frozen=True, slots=True)
class MoeCtsPolicyVerification:
    """记录一次不污染 episode 历史的接口验证结果。"""

    policy_path: Path
    device: str
    observation_shape: tuple[int, ...]
    history_shape: tuple[int, ...]
    action_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """转换为可写入日志或 ``project.json`` 的纯 Python 字典。"""

        return {
            "policy_path": str(self.policy_path),
            "device": self.device,
            "observation_shape": list(self.observation_shape),
            "history_shape": list(self.history_shape),
            "action_shape": list(self.action_shape),
        }


def _shape_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    """读取 tensor-like 对象的 shape，并给出可定位的合同错误。"""

    shape = getattr(value, "shape", None)
    if shape is None:
        raise MoeCtsPolicyContractError(f"{field_name} 缺少 shape 属性。")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise MoeCtsPolicyContractError(
            f"{field_name} 的 shape 无法转换为整数元组：{shape!r}。"
        ) from exc


class MoeCtsPolicyAdapter:
    """封装部署用 TorchScript，并在每次推理前后执行失败关闭校验。"""

    def __init__(
        self,
        policy_path: str | Path,
        *,
        device: object,
        contract: MoeCtsPolicyContract | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.contract = contract or MoeCtsPolicyContract()
        if not isinstance(self.contract, MoeCtsPolicyContract):
            raise TypeError("contract 必须是 MoeCtsPolicyContract。")

        self.policy_path = Path(policy_path).expanduser().resolve()
        if not self.policy_path.is_file():
            raise MoeCtsPolicyLoadError(
                f"找不到 MoE-CTS TorchScript：{self.policy_path}"
            )
        self.device = device
        self._torch = torch_module or self._import_torch()
        self._policy = self._load_policy()
        self._inference_count = 0
        self._reset_count = 0
        # 构造完成前必须主动清空历史；禁止继承导出或上一次 episode 的缓存。
        self.reset()

    @staticmethod
    def _import_torch() -> Any:
        """延迟导入 Torch，并把环境错误转换为明确提示。"""

        try:
            return importlib.import_module("torch")
        except ImportError as exc:
            raise MoeCtsPolicyLoadError(
                "当前 Python 环境没有 Torch；请在 Isaac Lab 的 Conda 环境中创建 "
                "MoeCtsPolicyAdapter。"
            ) from exc

    def _load_policy(self) -> Any:
        """使用部署协议加载模型，而不是调用 RSL-RL runner.load()。"""

        try:
            policy = self._torch.jit.load(
                str(self.policy_path),
                map_location=self.device,
            )
            policy = policy.eval()
        except Exception as exc:
            raise MoeCtsPolicyLoadError(
                f"加载 MoE-CTS TorchScript 失败：{self.policy_path}"
            ) from exc

        if not callable(policy):
            raise MoeCtsPolicyContractError("加载结果不可调用，缺少 forward 接口。")
        if not callable(getattr(policy, "reset", None)):
            raise MoeCtsPolicyContractError(
                "TorchScript 缺少 reset()，无法隔离不同 episode 的 CTS 历史。"
            )
        if not hasattr(policy, "obs_history"):
            raise MoeCtsPolicyContractError(
                "TorchScript 缺少 obs_history，不能验证 450 维 CTS 历史。"
            )
        self._require_shape(
            policy.obs_history,
            expected=self.contract.history_shape,
            field_name="policy.obs_history",
        )
        self._require_finite(policy.obs_history, field_name="policy.obs_history")
        return policy

    @property
    def inference_count(self) -> int:
        """返回该实例完成的 forward 次数，包含显式接口验证。"""

        return self._inference_count

    @property
    def reset_count(self) -> int:
        """返回该实例完成的历史清零次数。"""

        return self._reset_count

    @property
    def policy(self) -> Any:
        """只读暴露底层 ScriptModule，供诊断读取 schema，不应用于直接推理。"""

        return self._policy

    def _require_shape(
        self,
        value: object,
        *,
        expected: tuple[int, ...],
        field_name: str,
    ) -> tuple[int, ...]:
        actual = _shape_tuple(value, field_name=field_name)
        if actual != expected:
            raise MoeCtsPolicyContractError(
                f"{field_name} shape 不匹配：期望 {expected}，实际 {actual}。"
            )
        return actual

    def _require_finite(self, value: object, *, field_name: str) -> None:
        """拒绝任何 NaN/Inf，防止非有限动作进入 Isaac action manager。"""

        try:
            all_finite = bool(self._torch.isfinite(value).all().item())
        except Exception as exc:
            raise MoeCtsPolicyContractError(
                f"无法检查 {field_name} 是否为有限值。"
            ) from exc
        if not all_finite:
            raise MoeCtsPolicyContractError(f"{field_name} 包含 NaN 或 Inf。")

    def _require_zero_history(self) -> None:
        """确认 reset() 真正清空内部历史，而不只是存在同名方法。"""

        history = self._policy.obs_history
        self._require_shape(
            history,
            expected=self.contract.history_shape,
            field_name="policy.obs_history",
        )
        self._require_finite(history, field_name="policy.obs_history")
        try:
            nonzero_count = int(self._torch.count_nonzero(history).item())
        except Exception as exc:
            raise MoeCtsPolicyContractError(
                "无法确认 policy.reset() 是否清空 obs_history。"
            ) from exc
        if nonzero_count != 0:
            raise MoeCtsPolicyContractError(
                "policy.reset() 后 obs_history 仍包含非零值，episode 历史可能串扰。"
            )

    def reset(self) -> None:
        """清空模型内部 10 帧历史；每次 Isaac episode reset 都必须调用。"""

        try:
            self._policy.reset()
        except Exception as exc:
            raise MoeCtsPolicyContractError(
                "调用 policy.reset() 失败，无法安全开始新 episode。"
            ) from exc
        self._require_zero_history()
        self._reset_count += 1

    def infer(self, single_observation: object) -> Any:
        """对一帧 45 维观测推理，并返回一帧 12 维腿部动作。"""

        self._require_shape(
            single_observation,
            expected=self.contract.observation_shape,
            field_name="single_observation",
        )
        self._require_finite(
            single_observation,
            field_name="single_observation",
        )
        try:
            with self._torch.inference_mode():
                actions = self._policy(single_observation)
        except Exception as exc:
            raise MoeCtsPolicyInferenceError("MoE-CTS forward 推理失败。") from exc

        self._require_shape(
            actions,
            expected=self.contract.action_shape,
            field_name="policy actions",
        )
        self._require_finite(actions, field_name="policy actions")
        self._require_shape(
            self._policy.obs_history,
            expected=self.contract.history_shape,
            field_name="policy.obs_history",
        )
        self._require_finite(
            self._policy.obs_history,
            field_name="policy.obs_history",
        )
        self._inference_count += 1
        return actions

    def infer_from_observations(self, observations: object) -> Any:
        """从 Isaac 环境观测字典中取出 ``single_obs`` 并执行推理。"""

        if not isinstance(observations, Mapping):
            raise MoeCtsPolicyContractError("环境 observations 必须是 Mapping。")
        group_name = self.contract.observation_group_name
        if group_name not in observations:
            raise MoeCtsPolicyContractError(
                f"环境 observations 缺少 {group_name!r} 观测组；"
                f"实际 keys={tuple(observations.keys())!r}。"
            )
        return self.infer(observations[group_name])

    def verify(self, single_observation: object) -> MoeCtsPolicyVerification:
        """执行一次真实 forward 后重新清空历史，生成可审计接口报告。"""

        try:
            actions = self.infer(single_observation)
            action_shape = _shape_tuple(actions, field_name="policy actions")
            history_shape = _shape_tuple(
                self._policy.obs_history,
                field_name="policy.obs_history",
            )
            return MoeCtsPolicyVerification(
                policy_path=self.policy_path,
                device=str(self.device),
                observation_shape=_shape_tuple(
                    single_observation,
                    field_name="single_observation",
                ),
                history_shape=history_shape,
                action_shape=action_shape,
            )
        finally:
            # 验证 forward 已把首帧写入历史；无论成功或失败都不能污染正式 episode。
            self.reset()

    def verify_observations(
        self,
        observations: object,
    ) -> MoeCtsPolicyVerification:
        """验证完整环境观测字典，同时保持 verify() 的历史隔离语义。"""

        if not isinstance(observations, Mapping):
            raise MoeCtsPolicyContractError("环境 observations 必须是 Mapping。")
        group_name = self.contract.observation_group_name
        if group_name not in observations:
            raise MoeCtsPolicyContractError(
                f"环境 observations 缺少 {group_name!r} 观测组。"
            )
        return self.verify(observations[group_name])


__all__ = [
    "MoeCtsPolicyAdapter",
    "MoeCtsPolicyContract",
    "MoeCtsPolicyContractError",
    "MoeCtsPolicyError",
    "MoeCtsPolicyInferenceError",
    "MoeCtsPolicyLoadError",
    "MoeCtsPolicyVerification",
]
