"""JSON task schema for navigation, grasping, and optional placement."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pose2D:
    """Planar world-frame pose."""

    x: float
    y: float
    yaw: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pose2D":
        return cls(x=float(data["x"]), y=float(data["y"]), yaw=float(data.get("yaw", 0.0)))

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.yaw]


@dataclass(frozen=True)
class ObjectPoseWorld:
    """World-frame object pose used by randomized pick tasks."""

    x: float
    y: float
    z: float
    yaw: float
    roll: float = 0.0
    pitch: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectPoseWorld":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
            yaw=float(data.get("yaw", 0.0)),
            roll=float(data.get("roll", 0.0)),
            pitch=float(data.get("pitch", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "roll": self.roll,
            "pitch": self.pitch,
        }


@dataclass(frozen=True)
class PickConfig:
    """Navigation goal and object selection for the pick phase."""

    base_goal: Pose2D
    object_prim_path: str | None = None
    object_pose_world: ObjectPoseWorld | None = None
    grasp_mode: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PickConfig":
        object_pose_data = data.get("object_pose_world")
        return cls(
            base_goal=Pose2D.from_dict(data["base_goal"]),
            object_prim_path=data.get("object_prim_path"),
            object_pose_world=ObjectPoseWorld.from_dict(object_pose_data) if object_pose_data is not None else None,
            grasp_mode=str(data.get("grasp_mode", "auto")),
        )


@dataclass(frozen=True)
class PlaceConfig:
    """Optional carry-navigation and placement phase."""

    enabled: bool = False
    base_goal: Pose2D | None = None
    place_pose_world: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlaceConfig":
        data = data or {}
        goal_data = data.get("base_goal")
        return cls(
            enabled=bool(data.get("enabled", False)),
            base_goal=Pose2D.from_dict(goal_data) if goal_data is not None else None,
            place_pose_world=data.get("place_pose_world"),
        )


@dataclass(frozen=True)
class RecordingConfig:
    """Episode recording controls."""

    dataset_dir: str = "episodes"
    front_camera: bool = True
    wrist_camera: bool = False
    save_every_n_steps: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RecordingConfig":
        data = data or {}
        return cls(
            dataset_dir=str(data.get("dataset_dir", "episodes")),
            front_camera=bool(data.get("front_camera", True)),
            wrist_camera=bool(data.get("wrist_camera", False)),
            save_every_n_steps=max(1, int(data.get("save_every_n_steps", 10))),
        )


@dataclass(frozen=True)
class NavPickTask:
    """Complete task description for the two-stage pipeline."""

    task_id: int
    episode_id: int
    instruction: str
    scene_usd: str
    nav_map: str
    start: Pose2D
    pick: PickConfig
    place: PlaceConfig = field(default_factory=PlaceConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    randomization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NavPickTask":
        return cls(
            task_id=int(data["task_id"]),
            episode_id=int(data["episode_id"]),
            instruction=str(data.get("instruction", "")),
            scene_usd=str(data["scene_usd"]),
            nav_map=str(data["nav_map"]),
            start=Pose2D.from_dict(data["start"]),
            pick=PickConfig.from_dict(data["pick"]),
            place=PlaceConfig.from_dict(data.get("place")),
            recording=RecordingConfig.from_dict(data.get("recording")),
            randomization=dict(data.get("randomization") or {}),
        )


def load_task(path: str | Path) -> NavPickTask:
    """Read and validate a JSON navigation-pick task."""

    task_path = Path(path).expanduser().resolve()
    return NavPickTask.from_dict(json.loads(task_path.read_text(encoding="utf-8")))
