"""Structured JSONL recorder used by the new pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from source.interfaces import EpisodeSpec, StepRecord


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class JsonlEpisodeRecorder:
    """Write task, events, frames, export manifest, and final summary."""

    def __init__(self, output_dir: str | Path):
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self._output_dir / "events.jsonl"
        self.frames_path = self._output_dir / "frames.jsonl"
        self.events_path.write_text("", encoding="utf-8")
        self.frames_path.write_text("", encoding="utf-8")
        self.event_count = 0
        self.frame_count = 0

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def save_task(self, episode_spec: EpisodeSpec) -> Path:
        payload = episode_spec.raw_task or asdict(episode_spec)
        return self._write_json("task.json", payload)

    def record_event(self, event: dict[str, Any]) -> None:
        self._append_jsonl(self.events_path, event)
        self.event_count += 1

    def record_step(self, record: StepRecord) -> None:
        self._append_jsonl(self.frames_path, asdict(record))
        self.frame_count += 1

    def prepare_lerobot_export(self) -> dict[str, Any]:
        payload = {
            "lerobot_exported": False,
            "reason": "dry_run_contains_no_trainable_physics_data",
            "source_frames": str(self.frames_path),
            "frame_count": self.frame_count,
        }
        path = self._write_json("lerobot_manifest.json", payload)
        return {**payload, "manifest_path": str(path)}

    def close(self, summary: dict[str, Any]) -> Path:
        payload = {
            **summary,
            "event_count": self.event_count,
            "frame_count": self.frame_count,
            "data_output_path": str(self.output_dir),
        }
        return self._write_json("summary.json", payload)

    def _write_json(self, name: str, payload: Any) -> Path:
        path = self.output_dir / name
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _append_jsonl(path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
