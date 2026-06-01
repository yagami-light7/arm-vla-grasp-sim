"""Multi-phase CSV and image recorder for navigation-pick VLA episodes."""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any


PHASES = ("nav", "yaw_align", "grasp", "place")

EPISODE_COLUMNS = (
    "timestamp",
    "phase",
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_yaw",
    "base_quat_w",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "base_lin_vel_x_body",
    "base_lin_vel_y_body",
    "base_ang_vel_z_body",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "ee_pos_x",
    "ee_pos_y",
    "ee_pos_z",
    "ee_roll",
    "ee_pitch",
    "ee_yaw",
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
    "gripper",
    "arm_action_joint1",
    "arm_action_joint2",
    "arm_action_joint3",
    "arm_action_joint4",
    "arm_action_joint5",
    "arm_action_joint6",
    "gripper_action",
    "object_pos_x",
    "object_pos_y",
    "object_pos_z",
    "object_quat_w",
    "object_quat_x",
    "object_quat_y",
    "object_quat_z",
    "front_image",
    "wrist_image",
    "success",
    "failure_reason",
)


class EpisodeRecorder:
    """Append stable-schema phase records and images to one episode directory."""

    def __init__(
        self,
        dataset_dir: str | Path,
        task_id: int | str,
        episode_id: int | str,
        *,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.episode_dir = Path(dataset_dir).expanduser().resolve() / str(task_id) / str(episode_id)
        self._row_counts = {phase: 0 for phase in PHASES}
        if self.enabled:
            self.episode_dir.mkdir(parents=True, exist_ok=True)

    def save_task(self, task: dict[str, Any]) -> Path:
        """Write the immutable task description used for this episode."""

        return self._write_json("task.json", task)

    def write_summary(self, summary: dict[str, Any]) -> Path:
        """Write the latest success/failure summary."""

        payload = dict(summary)
        payload.setdefault("updated_at", time.time())
        payload.setdefault("episode_dir", str(self.episode_dir))
        return self._write_json("summary.json", payload)

    def record(
        self,
        phase: str,
        values: dict[str, Any],
        *,
        front_image: bytes | str | Path | None = None,
        wrist_image: bytes | str | Path | None = None,
        image_extension: str = ".jpg",
    ) -> dict[str, Any]:
        """Append one row and optionally persist front/wrist image payloads."""

        self._require_phase(phase)
        row = {column: "" for column in EPISODE_COLUMNS}
        row.update({key: value for key, value in values.items() if key in row})
        row["phase"] = phase
        row["timestamp"] = values.get("timestamp", time.time())
        index = self._row_counts[phase]
        if front_image is not None:
            row["front_image"] = self._save_image(phase, "front", index, front_image, image_extension)
        if wrist_image is not None:
            row["wrist_image"] = self._save_image(phase, "wrist", index, wrist_image, image_extension)
        if self.enabled:
            csv_path = self.episode_dir / phase / "data.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not csv_path.exists()
            with csv_path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        self._row_counts[phase] += 1
        return row

    def phase_csv(self, phase: str) -> Path:
        """Return the CSV path for a phase."""

        self._require_phase(phase)
        return self.episode_dir / phase / "data.csv"

    def _write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.episode_dir / name
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _save_image(
        self,
        phase: str,
        camera: str,
        index: int,
        payload: bytes | str | Path,
        extension: str,
    ) -> str:
        extension = extension if extension.startswith(".") else f".{extension}"
        relative_path = Path(phase) / "images" / camera / f"{index:06d}{extension}"
        output_path = self.episode_dir / relative_path
        if self.enabled:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                output_path.write_bytes(payload)
            else:
                shutil.copyfile(Path(payload), output_path)
        return relative_path.as_posix()

    @staticmethod
    def _require_phase(phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"unsupported episode phase {phase!r}; expected one of {PHASES}")
