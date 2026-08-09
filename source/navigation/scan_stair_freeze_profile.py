"""加载并审计 Go2-X5 的 SCAN 楼梯底盘冻结生产 profile。

该 profile 只描述楼梯执行控制，不承载 PCT tomography、gateway 或拓扑修补
数据。运行时参数最终仍构造为 :class:`ScanStairFreezeConfig`，从而与冻结协调器
共享同一份数值约束。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from source.navigation.scan_stair_freeze import (
    SCAN_STAIR_FREEZE_DOG_JOINT_NAMES,
    SCAN_STAIR_FREEZE_DOG_STAND_JOINT_POSITIONS,
    ScanStairFreezeConfig,
)


SCAN_STAIR_FREEZE_PROFILE_SCHEMA_VERSION = 1
SCAN_STAIR_FREEZE_PROFILE_ASSET_KIND = (
    "scan_stair_freeze_execution_profile_v1"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "asset_kind",
        "description",
        "scene",
        "robot",
        "controller",
        "baseline_provenance",
        "execution_contract",
        "fixed_navigation_posture",
        "parameters",
        "contract_sha256",
    }
)
_BASELINE_KEYS = frozenset(
    {"source_branch", "behavior", "pct_topology_profile_reused"}
)
_EXECUTION_CONTRACT_KEYS = frozenset(
    {
        "path_height_semantics",
        "root_target_height_semantics",
        "cmd_vel_during_freeze",
        "root_lock",
        "support_joint_lock",
        "full_body_lock_during_active",
        "non_physical_root_lock_workaround",
    }
)
_POSTURE_KEYS = frozenset(
    {
        "arm_posture",
        "arm_fixed_during_navigation",
        "dog_joint_names",
        "dog_stand_joint_positions_rad",
    }
)
_PARAMETER_KEYS = frozenset(
    field.name for field in fields(ScanStairFreezeConfig)
)
_CONTROL_TIMING_PARAMETER_KEYS = frozenset({"default_control_dt_s"})


class ScanStairFreezeProfileError(ValueError):
    """楼梯冻结 profile 缺失、漂移或不满足生产安全约束。"""


@dataclass(frozen=True, slots=True)
class ScanStairFreezeFixedPosture:
    """冻结期间必须保持的 Go2-X5 腿部与机械臂姿态合同。"""

    arm_posture: str
    arm_fixed_during_navigation: bool
    dog_joint_names: tuple[str, ...]
    dog_stand_joint_positions_rad: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ScanStairFreezeExecutionContract:
    """底盘冻结的非物理语义与锁定动作合同。"""

    path_height_semantics: str
    root_target_height_semantics: str
    cmd_vel_during_freeze: tuple[float, float, float]
    root_lock: bool
    support_joint_lock: bool
    full_body_lock_during_active: bool
    non_physical_root_lock_workaround: bool


@dataclass(frozen=True, slots=True)
class ScanStairFreezeProfile:
    """一份经过严格 schema、数值和 digest 校验的冻结执行 profile。"""

    profile_id: str
    scene: str
    robot: str
    controller: str
    description: str
    source_branch: str
    baseline_behavior: str
    pct_topology_profile_reused: bool
    execution_contract: ScanStairFreezeExecutionContract
    fixed_navigation_posture: ScanStairFreezeFixedPosture
    config: ScanStairFreezeConfig
    contract_sha256: str
    source_sha256: str
    source_path: Path

    def pipeline_navigation_overrides(self) -> dict[str, Any]:
        """返回可写入 ``NavigationConfig`` 的冻结参数，不覆盖全局控制周期。"""

        output: dict[str, Any] = {}
        for field in fields(ScanStairFreezeConfig):
            if field.name in _CONTROL_TIMING_PARAMETER_KEYS:
                continue
            # body height 是 PCT、SCAN 与冻结执行共享的唯一导航高度合同，
            # 禁止重新生成历史 scan_stair_freeze_body_height_m 双配置。
            key = (
                "navigation_body_height_m"
                if field.name == "body_height_m"
                else f"scan_stair_freeze_{field.name}"
            )
            output[key] = getattr(self.config, field.name)
        return output

    def audit_report(self) -> dict[str, Any]:
        """返回可直接写入运行 summary 的确定性审计字段。"""

        return {
            "profile_id": self.profile_id,
            "scene": self.scene,
            "robot": self.robot,
            "controller": self.controller,
            "contract_sha256": self.contract_sha256,
            "source_sha256": self.source_sha256,
            "source_path": str(self.source_path),
            "source_branch": self.source_branch,
            "baseline_behavior": self.baseline_behavior,
            "pct_topology_profile_reused": self.pct_topology_profile_reused,
            "non_physical_root_lock_workaround": (
                self.execution_contract.non_physical_root_lock_workaround
            ),
        }

    def validate_runtime_bindings(
        self,
        *,
        navigation_body_height_m: float,
        control_dt_s: float,
    ) -> ScanStairFreezeConfig:
        """确认 profile 合同与唯一导航高度、实际控制周期完全一致。"""

        body_height = _finite_scalar(
            navigation_body_height_m,
            label="navigation_body_height_m",
        )
        control_dt = _finite_scalar(control_dt_s, label="control_dt_s")
        if not math.isclose(
            body_height,
            self.config.body_height_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ScanStairFreezeProfileError(
                "冻结 profile.body_height_m 与唯一 navigation_body_height_m 不一致。"
            )
        if not math.isclose(
            control_dt,
            self.config.default_control_dt_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ScanStairFreezeProfileError(
                "冻结 profile.default_control_dt_s 与实际控制周期不一致。"
            )
        return self.config


def compute_scan_stair_freeze_contract_sha256(
    payload: Mapping[str, Any],
) -> str:
    """计算排除自引用 digest 字段后的规范 JSON SHA256。"""

    if not isinstance(payload, Mapping):
        raise ScanStairFreezeProfileError("冻结 profile 顶层必须是对象。")
    canonical_payload = {
        key: value for key, value in payload.items() if key != "contract_sha256"
    }
    try:
        canonical = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScanStairFreezeProfileError(
            "冻结 profile 不能规范化为有限 JSON。"
        ) from exc
    return hashlib.sha256(canonical).hexdigest()


def load_scan_stair_freeze_profile(
    path: str | Path,
    *,
    expected_scene: str | None = None,
    expected_robot: str | None = None,
) -> ScanStairFreezeProfile:
    """从 JSON 加载楼梯冻结执行 profile，并拒绝任何 schema 漂移。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SCAN 楼梯冻结 profile 不存在：{source}")
    raw_bytes = source.read_bytes()
    try:
        payload = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ScanStairFreezeProfileError(
            f"SCAN 楼梯冻结 profile 不是 UTF-8：{source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ScanStairFreezeProfileError(
            f"无法解析 SCAN 楼梯冻结 profile：{source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ScanStairFreezeProfileError("冻结 profile 顶层必须是对象。")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, label="冻结 profile")
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"]
        != SCAN_STAIR_FREEZE_PROFILE_SCHEMA_VERSION
    ):
        raise ScanStairFreezeProfileError(
            "不支持的冻结 profile schema_version。"
        )
    if payload["asset_kind"] != SCAN_STAIR_FREEZE_PROFILE_ASSET_KIND:
        raise ScanStairFreezeProfileError(
            "冻结 profile asset_kind 不能与 PCT 拓扑 profile 混用。"
        )

    profile_id = _required_text(payload, "profile_id", label="冻结 profile")
    description = _required_text(payload, "description", label="冻结 profile")
    scene = _required_text(payload, "scene", label="冻结 profile")
    robot = _required_text(payload, "robot", label="冻结 profile")
    controller = _required_text(payload, "controller", label="冻结 profile")
    if controller != "scan_stair_freeze":
        raise ScanStairFreezeProfileError(
            "生产冻结 profile.controller 必须是 scan_stair_freeze。"
        )
    if expected_scene is not None and scene != str(expected_scene):
        raise ScanStairFreezeProfileError(
            f"冻结 profile 场景不匹配：期望 {expected_scene!r}，实际 {scene!r}。"
        )
    if expected_robot is not None and robot != str(expected_robot):
        raise ScanStairFreezeProfileError(
            f"冻结 profile 机器人不匹配：期望 {expected_robot!r}，实际 {robot!r}。"
        )

    baseline = _required_mapping(
        payload, "baseline_provenance", label="冻结 profile"
    )
    _require_exact_keys(
        baseline,
        _BASELINE_KEYS,
        label="baseline_provenance",
    )
    source_branch = _required_text(
        baseline, "source_branch", label="baseline_provenance"
    )
    baseline_behavior = _required_text(
        baseline, "behavior", label="baseline_provenance"
    )
    topology_reused = baseline["pct_topology_profile_reused"]
    if topology_reused is not False:
        raise ScanStairFreezeProfileError(
            "冻结执行参数禁止复用 PCT 拓扑 profile。"
        )
    if source_branch != "pct-scene" or baseline_behavior != "chassis_root_lock":
        raise ScanStairFreezeProfileError(
            "冻结 profile 必须明确绑定 pct-scene 的 chassis_root_lock 基线。"
        )

    execution_contract = _load_execution_contract(payload)
    posture = _load_fixed_posture(payload)
    config = _load_parameters(payload)

    declared_digest = payload["contract_sha256"]
    if not _is_sha256(declared_digest):
        raise ScanStairFreezeProfileError(
            "contract_sha256 必须是 64 位小写十六进制 SHA256。"
        )
    computed_digest = compute_scan_stair_freeze_contract_sha256(payload)
    if declared_digest != computed_digest:
        raise ScanStairFreezeProfileError(
            "冻结 profile contract_sha256 与规范内容不匹配。"
        )

    return ScanStairFreezeProfile(
        profile_id=profile_id,
        scene=scene,
        robot=robot,
        controller=controller,
        description=description,
        source_branch=source_branch,
        baseline_behavior=baseline_behavior,
        pct_topology_profile_reused=False,
        execution_contract=execution_contract,
        fixed_navigation_posture=posture,
        config=config,
        contract_sha256=computed_digest,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        source_path=source,
    )


def load_scene_scan_stair_freeze_profile(
    scene_profile: Any,
    project_root: str | Path,
    *,
    expected_robot: str = "go2_x5",
) -> ScanStairFreezeProfile:
    """加载场景显式引用且列入必需资产的冻结执行 profile。"""

    raw_path = getattr(scene_profile, "scan_stair_freeze_profile", None)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ScanStairFreezeProfileError(
            f"场景 {getattr(scene_profile, 'name', '<unknown>')!r} "
            "没有显式引用 scan_stair_freeze_profile。"
        )
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ScanStairFreezeProfileError(
            "场景 scan_stair_freeze_profile 必须是项目内相对路径。"
        )
    required_assets = tuple(getattr(scene_profile, "required_assets", ()))
    if raw_path not in required_assets:
        raise ScanStairFreezeProfileError(
            "场景引用的 scan_stair_freeze_profile 必须列入 required_assets。"
        )
    root = Path(project_root).expanduser().resolve()
    source = (root / relative_path).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ScanStairFreezeProfileError(
            "场景 scan_stair_freeze_profile 解析后超出项目目录。"
        ) from exc
    return load_scan_stair_freeze_profile(
        source,
        expected_scene=getattr(scene_profile, "name", None),
        expected_robot=expected_robot,
    )


def bind_pipeline_navigation_settings(
    navigation: Any,
    profile: ScanStairFreezeProfile,
) -> Any:
    """把生产 profile 原子绑定到不可变 ``NavigationSettings`` 副本。"""

    if not is_dataclass(navigation) or isinstance(navigation, type):
        raise ScanStairFreezeProfileError(
            "冻结 profile 只能绑定到 dataclass navigation settings 实例。"
        )
    overrides = profile.pipeline_navigation_overrides()
    missing = sorted(key for key in overrides if not hasattr(navigation, key))
    if missing:
        raise ScanStairFreezeProfileError(
            f"navigation settings 缺少冻结 profile 字段：{missing}。"
        )
    if not hasattr(navigation, "control_dt"):
        raise ScanStairFreezeProfileError(
            "navigation settings 缺少 control_dt，无法校验冻结控制周期。"
        )
    # 这两个值存在显式 CLI 入口；profile 不得静默覆盖用户输入。其他详细
    # 冻结参数没有 CLI 入口，由本 profile 作为唯一生产来源。
    original_enabled = getattr(navigation, "scan_stair_freeze_enabled")
    if original_enabled is not profile.config.enabled:
        raise ScanStairFreezeProfileError(
            "scan_stair_freeze_enabled 与生产冻结 profile 不一致，拒绝静默覆盖。"
        )
    profile.validate_runtime_bindings(
        navigation_body_height_m=getattr(navigation, "navigation_body_height_m"),
        control_dt_s=getattr(navigation, "control_dt"),
    )
    try:
        bound = replace(navigation, **overrides)
    except TypeError as exc:
        raise ScanStairFreezeProfileError(
            "无法原子绑定冻结 profile 到 navigation settings。"
        ) from exc
    profile.validate_runtime_bindings(
        navigation_body_height_m=getattr(bound, "navigation_body_height_m"),
        control_dt_s=getattr(bound, "control_dt"),
    )
    return bound


def _load_execution_contract(
    payload: Mapping[str, Any],
) -> ScanStairFreezeExecutionContract:
    raw = _required_mapping(payload, "execution_contract", label="冻结 profile")
    _require_exact_keys(raw, _EXECUTION_CONTRACT_KEYS, label="execution_contract")
    expected_text = {
        "path_height_semantics": "ground",
        "root_target_height_semantics": "ground_plus_body_height_once",
    }
    for key, expected in expected_text.items():
        if raw[key] != expected:
            raise ScanStairFreezeProfileError(
                f"execution_contract.{key} 必须是 {expected!r}。"
            )
    for key in (
        "root_lock",
        "support_joint_lock",
        "full_body_lock_during_active",
        "non_physical_root_lock_workaround",
    ):
        if raw[key] is not True:
            raise ScanStairFreezeProfileError(
                f"execution_contract.{key} 必须显式为 true。"
            )
    zero_cmd = _finite_number_sequence(
        raw["cmd_vel_during_freeze"],
        length=3,
        label="execution_contract.cmd_vel_during_freeze",
    )
    if zero_cmd != (0.0, 0.0, 0.0):
        raise ScanStairFreezeProfileError(
            "冻结期间 cmd_vel 必须严格为 [0, 0, 0]。"
        )
    return ScanStairFreezeExecutionContract(
        path_height_semantics="ground",
        root_target_height_semantics="ground_plus_body_height_once",
        cmd_vel_during_freeze=zero_cmd,
        root_lock=True,
        support_joint_lock=True,
        full_body_lock_during_active=True,
        non_physical_root_lock_workaround=True,
    )


def _load_fixed_posture(
    payload: Mapping[str, Any],
) -> ScanStairFreezeFixedPosture:
    raw = _required_mapping(
        payload, "fixed_navigation_posture", label="冻结 profile"
    )
    _require_exact_keys(raw, _POSTURE_KEYS, label="fixed_navigation_posture")
    if raw["arm_posture"] != "stow":
        raise ScanStairFreezeProfileError(
            "fixed_navigation_posture.arm_posture 必须是 stow。"
        )
    if raw["arm_fixed_during_navigation"] is not True:
        raise ScanStairFreezeProfileError(
            "机械臂在冻结导航期间必须保持固定。"
        )
    raw_names = raw["dog_joint_names"]
    if not isinstance(raw_names, list) or any(
        not isinstance(value, str) or not value for value in raw_names
    ):
        raise ScanStairFreezeProfileError("dog_joint_names 必须是非空字符串数组。")
    names = tuple(raw_names)
    expected_names = tuple(SCAN_STAIR_FREEZE_DOG_JOINT_NAMES)
    if names != expected_names:
        raise ScanStairFreezeProfileError(
            "冻结姿态关节顺序与 SCAN runtime 常量不一致。"
        )
    positions = _finite_number_sequence(
        raw["dog_stand_joint_positions_rad"],
        length=len(names),
        label="fixed_navigation_posture.dog_stand_joint_positions_rad",
    )
    if positions != tuple(SCAN_STAIR_FREEZE_DOG_STAND_JOINT_POSITIONS):
        raise ScanStairFreezeProfileError(
            "冻结站立姿态与 SCAN runtime 常量不一致。"
        )
    if any(abs(value) > math.pi for value in positions):
        raise ScanStairFreezeProfileError("冻结站立关节角必须位于 [-pi, pi]。")
    return ScanStairFreezeFixedPosture(
        arm_posture="stow",
        arm_fixed_during_navigation=True,
        dog_joint_names=names,
        dog_stand_joint_positions_rad=positions,
    )


def _load_parameters(payload: Mapping[str, Any]) -> ScanStairFreezeConfig:
    raw = _required_mapping(payload, "parameters", label="冻结 profile")
    _require_exact_keys(raw, _PARAMETER_KEYS, label="parameters")
    try:
        config = ScanStairFreezeConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise ScanStairFreezeProfileError(f"冻结 parameters 非法：{exc}") from exc
    if config.enabled is not True:
        raise ScanStairFreezeProfileError("生产冻结 profile 必须 enabled=true。")
    if config.require_supervisor_sensor_status is not True:
        raise ScanStairFreezeProfileError(
            "生产冻结 profile 必须启用 supervisor 传感器状态门。"
        )
    if config.speed_mps > 0.30:
        raise ScanStairFreezeProfileError(
            "Go2-X5 楼梯冻结 root 速度不能超过 0.30 m/s。"
        )
    if config.activation_lookahead_m < config.approach_distance_m:
        raise ScanStairFreezeProfileError(
            "activation_lookahead_m 不能小于 approach_distance_m。"
        )
    if config.activation_passed_margin_m > config.activation_radius_m:
        raise ScanStairFreezeProfileError(
            "activation_passed_margin_m 不能大于 activation_radius_m。"
        )
    if (
        config.post_release_stabilization_timeout_s
        < config.post_release_stable_time_s
    ):
        raise ScanStairFreezeProfileError(
            "释放稳定超时不能短于要求的连续稳定时间。"
        )
    if config.post_release_max_linear_speed_mps >= config.speed_mps:
        raise ScanStairFreezeProfileError(
            "释放稳定线速度阈值必须小于冻结 root 执行速度。"
        )
    if config.post_release_max_tilt_rad >= math.pi / 2.0:
        raise ScanStairFreezeProfileError(
            "释放稳定最大倾角必须小于 pi/2。"
        )
    if not (
        config.min_measured_body_height_m
        <= config.body_height_m
        <= config.max_measured_body_height_m
    ):
        raise ScanStairFreezeProfileError(
            "body_height_m 必须位于实测高度安全边界内。"
        )
    return config


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ScanStairFreezeProfileError(f"冻结 profile 含重复字段：{key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> Any:
    raise ScanStairFreezeProfileError(f"冻结 profile 禁止非有限 JSON 数值：{value}")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"缺少字段 {missing}")
        if unknown:
            details.append(f"未知字段 {unknown}")
        raise ScanStairFreezeProfileError(f"{label} schema 非法：{'；'.join(details)}。")


def _required_mapping(
    value: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ScanStairFreezeProfileError(f"{label}.{key} 必须是对象。")
    return item


def _required_text(
    value: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ScanStairFreezeProfileError(f"{label}.{key} 必须是非空字符串。")
    if item != item.strip():
        raise ScanStairFreezeProfileError(f"{label}.{key} 不能带首尾空白。")
    return item


def _finite_number_sequence(
    value: Any,
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise ScanStairFreezeProfileError(f"{label} 必须含 {length} 个数值。")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ScanStairFreezeProfileError(f"{label} 只能包含数值。")
        number = float(item)
        if not math.isfinite(number):
            raise ScanStairFreezeProfileError(f"{label} 只能包含有限数值。")
        output.append(number)
    return tuple(output)


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScanStairFreezeProfileError(f"{label} 必须是有限数值。")
    output = float(value)
    if not math.isfinite(output):
        raise ScanStairFreezeProfileError(f"{label} 必须是有限数值。")
    return output


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "SCAN_STAIR_FREEZE_PROFILE_ASSET_KIND",
    "SCAN_STAIR_FREEZE_PROFILE_SCHEMA_VERSION",
    "ScanStairFreezeExecutionContract",
    "ScanStairFreezeFixedPosture",
    "ScanStairFreezeProfile",
    "ScanStairFreezeProfileError",
    "bind_pipeline_navigation_settings",
    "compute_scan_stair_freeze_contract_sha256",
    "load_scan_stair_freeze_profile",
    "load_scene_scan_stair_freeze_profile",
]
