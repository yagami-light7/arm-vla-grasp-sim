"""Continuous JSONL recorder for contact-only VLA expert episodes."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


CAMERA_NAMES = ("front", "wrist", "third")


def _json_safe(value: Any) -> Any:
    """Convert tensor/numpy-like values into JSON-serializable containers."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class VLAEpisodeRecorder:
    """Append per-frame observation/action/state records to ``frames.jsonl``."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        task_id: int | str,
        episode_id: int | str,
        record_every_n_steps: int = 1,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.record_every_n_steps = max(1, int(record_every_n_steps))
        self.episode_dir = Path(dataset_dir).expanduser().resolve()
        self.task_id = task_id
        self.episode_id = episode_id
        self.frames_path = self.episode_dir / "frames.jsonl"
        self.frame_count = 0
        self._step_count = 0
        if self.enabled:
            self.episode_dir.mkdir(parents=True, exist_ok=True)
            (self.episode_dir / "images").mkdir(parents=True, exist_ok=True)
            self.frames_path.write_text("", encoding="utf-8")

    def save_task(self, task: dict[str, Any]) -> Path:
        """Persist the immutable task JSON used to create this episode."""

        return self._write_json("task.json", task)

    def write_summary(self, summary: dict[str, Any]) -> Path:
        """Write the latest episode summary."""

        payload = dict(summary)
        payload.setdefault("task_id", self.task_id)
        payload.setdefault("episode_id", self.episode_id)
        payload.setdefault("frame_count", self.frame_count)
        payload.setdefault("episode_dir", str(self.episode_dir))
        payload.setdefault("updated_at", time.time())
        return self._write_json("summary.json", payload)

    def record(
        self,
        *,
        phase: str,
        observation: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        timestamp: float | None = None,
        images: dict[str, bytes | str | Path | None] | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Append one frame unless skipped by ``record_every_n_steps``."""

        step = self._step_count
        self._step_count += 1
        if not force and step % self.record_every_n_steps != 0:
            return None

        frame_index = self.frame_count
        observation_payload = dict(observation or {})
        image_paths = self._save_images(frame_index, images or {})
        observation_payload.update(image_paths)

        state_payload = {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "object_attached": False,
            "attachment_mode": "contact_only",
            **dict(state or {}),
        }
        state_payload["object_attached"] = False
        state_payload["attachment_mode"] = "contact_only"

        frame = {
            "frame": frame_index,
            "timestamp": float(time.time() if timestamp is None else timestamp),
            "phase": str(phase),
            "observation": _json_safe(observation_payload),
            "action": _json_safe(action or {}),
            "state": _json_safe(state_payload),
        }

        if self.enabled:
            with self.frames_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        self.frame_count += 1
        return frame

    def _write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.episode_dir / name
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _save_images(self, frame_index: int, images: dict[str, bytes | str | Path | None]) -> dict[str, str]:
        output: dict[str, str] = {}
        for camera_name in CAMERA_NAMES:
            payload = images.get(camera_name)
            if payload is None:
                continue
            relative_path = Path("images") / camera_name / f"{frame_index:06d}.jpg"
            output[f"{camera_name}_image"] = relative_path.as_posix()
            if not self.enabled:
                continue
            target = self.episode_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                target.write_bytes(payload)
            else:
                shutil.copyfile(Path(payload), target)
        return output
