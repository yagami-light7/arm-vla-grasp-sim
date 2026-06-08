"""Read-only contact-grasp monitoring for contact-only carrying."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable


Vector3 = tuple[float, float, float]
StateReader = Callable[[], "RigidBodyState | None"]


@dataclass(frozen=True)
class RigidBodyState:
    """World-frame pose and optional velocity for one rigid body or frame."""

    position: Vector3
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    linear_velocity: Vector3 | None = None
    angular_velocity: Vector3 | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "quat_wxyz": list(self.quat_wxyz),
            "linear_velocity": list(self.linear_velocity) if self.linear_velocity is not None else None,
            "angular_velocity": list(self.angular_velocity) if self.angular_velocity is not None else None,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class ContactGraspMonitorConfig:
    """Thresholds for lift, stable grasp, and carry slip checks."""

    min_lift_height: float = 0.05
    max_slip_distance: float = 0.08
    object_drop_height_threshold: float = 0.05
    max_object_speed: float = 3.0
    max_ee_object_distance: float = 0.30


class ContactGraspMonitor:
    """Monitor a grasp without moving, attaching, or teleporting the object."""

    def __init__(
        self,
        *,
        object_prim_path: str,
        ee_prim_path: str | None = None,
        object_state_reader: StateReader | None = None,
        ee_state_reader: StateReader | None = None,
        config: ContactGraspMonitorConfig | None = None,
    ):
        self.object_prim_path = object_prim_path
        self.ee_prim_path = ee_prim_path
        self.object_state_reader = object_state_reader
        self.ee_state_reader = ee_state_reader
        self.config = config or ContactGraspMonitorConfig()
        self.object_state_before_grasp: RigidBodyState | None = None
        self.relative_object_ee_before_lift: Vector3 | None = None
        self._lift_verified = False
        self._last_slip_report: dict[str, Any] = {}

    def mark_grasp_reference(self) -> dict[str, Any]:
        """Store the object and object-to-EE reference used by later checks."""

        self.object_state_before_grasp = self.get_object_state()
        self.relative_object_ee_before_lift = self.get_relative_object_ee_pose()
        self._lift_verified = False
        return {
            "object_pose_before_grasp": (
                self.object_state_before_grasp.to_dict() if self.object_state_before_grasp is not None else None
            ),
            "relative_object_ee_before_lift": (
                list(self.relative_object_ee_before_lift) if self.relative_object_ee_before_lift is not None else None
            ),
        }

    def get_object_pose_world(self) -> dict[str, Any] | None:
        state = self.get_object_state()
        return state.to_dict() if state is not None else None

    def get_ee_pose_world(self) -> dict[str, Any] | None:
        state = self.get_ee_state()
        return state.to_dict() if state is not None else None

    def get_object_velocity(self) -> dict[str, Any]:
        state = self.get_object_state()
        if state is None:
            return {"linear": None, "angular": None, "speed": None}
        speed = _norm3(state.linear_velocity) if state.linear_velocity is not None else None
        return {
            "linear": list(state.linear_velocity) if state.linear_velocity is not None else None,
            "angular": list(state.angular_velocity) if state.angular_velocity is not None else None,
            "speed": speed,
        }

    def get_relative_object_ee_pose(self) -> Vector3 | None:
        object_state = self.get_object_state()
        ee_state = self.get_ee_state()
        if object_state is None or ee_state is None:
            return None
        return _sub3(object_state.position, ee_state.position)

    def check_object_lifted(self) -> bool:
        report = self.get_lift_report()
        lifted = bool(report.get("object_lifted", False))
        self._lift_verified = self._lift_verified or lifted
        return lifted

    def get_lift_report(self) -> dict[str, Any]:
        current = self.get_object_state()
        before = self.object_state_before_grasp
        if current is None or before is None:
            return {
                "object_lifted": False,
                "failure_reason": "object_state_unavailable",
                "lifted_height": None,
                "min_lift_height": self.config.min_lift_height,
            }
        lifted_height = float(current.position[2] - before.position[2])
        return {
            "object_lifted": lifted_height >= self.config.min_lift_height,
            "lifted_height": lifted_height,
            "min_lift_height": self.config.min_lift_height,
            "object_z_before_grasp": before.position[2],
            "object_z_current": current.position[2],
        }

    def check_grasp_stable(self) -> bool:
        report = self.get_stability_report()
        return bool(report.get("grasp_stable", False))

    def get_stability_report(self) -> dict[str, Any]:
        reference = self.relative_object_ee_before_lift
        current = self.get_relative_object_ee_pose()
        if reference is None or current is None:
            return {
                "grasp_stable": False,
                "failure_reason": "relative_pose_unavailable",
                "translation_drift": None,
                "max_slip_distance": self.config.max_slip_distance,
            }
        drift = _norm3(_sub3(current, reference))
        return {
            "grasp_stable": drift <= self.config.max_slip_distance,
            "translation_drift": drift,
            "max_slip_distance": self.config.max_slip_distance,
            "relative_object_ee": list(current),
            "relative_object_ee_reference": list(reference),
        }

    def check_object_slipped(self) -> bool:
        report = self.get_slip_report()
        self._last_slip_report = report
        return bool(report.get("object_slipped", False))

    def get_slip_report(self) -> dict[str, Any]:
        object_state = self.get_object_state()
        ee_state = self.get_ee_state()
        stability = self.get_stability_report()
        lifted = self.get_lift_report()
        reasons: list[str] = []

        if object_state is None:
            reasons.append("object_state_unavailable")
        else:
            object_z = object_state.position[2]
            if object_z <= self.config.object_drop_height_threshold:
                reasons.append("object_below_drop_height_threshold")
            if self._lift_verified and self.object_state_before_grasp is not None:
                min_carry_z = self.object_state_before_grasp.position[2] + 0.4 * self.config.min_lift_height
                if object_z < min_carry_z:
                    reasons.append("object_lost_lift_height")
            if object_state.linear_velocity is not None:
                speed = _norm3(object_state.linear_velocity)
                if speed > self.config.max_object_speed:
                    reasons.append("object_velocity_too_large")

        drift = stability.get("translation_drift")
        if drift is not None and drift > self.config.max_slip_distance:
            reasons.append("object_ee_drift_too_large")

        if object_state is not None and ee_state is not None:
            ee_distance = _norm3(_sub3(object_state.position, ee_state.position))
            if ee_distance > self.config.max_ee_object_distance:
                reasons.append("object_too_far_from_ee")
        else:
            ee_distance = None

        return {
            "object_slipped": bool(reasons),
            "failure_reasons": reasons,
            "object_lifted": lifted.get("object_lifted", False),
            "lift_report": lifted,
            "stability_report": stability,
            "object_ee_distance": ee_distance,
            "contact_report": self.get_contact_report(),
        }

    def get_contact_report(self) -> dict[str, Any]:
        """Return optional contact data when a runtime-specific reader exists."""

        return {
            "available": False,
            "reason": "contact sensor reader not configured",
        }

    def get_object_state(self) -> RigidBodyState | None:
        if self.object_state_reader is not None:
            return self.object_state_reader()
        return _read_usd_state(self.object_prim_path)

    def get_ee_state(self) -> RigidBodyState | None:
        if self.ee_state_reader is not None:
            return self.ee_state_reader()
        if not self.ee_prim_path:
            return None
        return _read_usd_state(self.ee_prim_path)


def _read_usd_state(prim_path: str) -> RigidBodyState | None:
    """Best-effort USD pose reader that does not mutate the stage."""

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except ImportError:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    linear_velocity = _read_vec3_attr(prim, ("physics:velocity", "velocities"))
    angular_velocity = _read_vec3_attr(prim, ("physics:angularVelocity", "angularVelocities"))
    return RigidBodyState(
        position=(float(translation[0]), float(translation[1]), float(translation[2])),
        quat_wxyz=(float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )


def _read_vec3_attr(prim: Any, names: tuple[str, ...]) -> Vector3 | None:
    for name in names:
        attr = prim.GetAttribute(name)
        if not attr:
            continue
        value = attr.Get()
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            value = value[0]
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError, IndexError):
            continue
    return None


def _sub3(lhs: Vector3, rhs: Vector3) -> Vector3:
    return (float(lhs[0] - rhs[0]), float(lhs[1] - rhs[1]), float(lhs[2] - rhs[2]))


def _norm3(value: Vector3 | None) -> float | None:
    if value is None:
        return None
    return math.sqrt(float(value[0]) ** 2 + float(value[1]) ** 2 + float(value[2]) ** 2)
