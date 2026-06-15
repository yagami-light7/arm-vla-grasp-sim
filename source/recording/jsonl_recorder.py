"""Structured JSONL recorder used by the new pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from source.interfaces import EpisodeSpec, SimulationState, StepRecord

from .lerobot_dataset import (
    DwaEpisodeWriter,
    LeRobotRecordingConfig,
    materialize_lerobot_dataset,
)


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

    def __init__(
        self,
        output_dir: str | Path,
        *,
        lerobot_config: LeRobotRecordingConfig | None = None,
    ):
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self._output_dir / "events.jsonl"
        self.frames_path = self._output_dir / "frames.jsonl"
        self.events_path.write_text("", encoding="utf-8")
        self.frames_path.write_text("", encoding="utf-8")
        self._lerobot_config = lerobot_config or LeRobotRecordingConfig()
        self._dataset_writer = DwaEpisodeWriter(self._output_dir, self._lerobot_config)
        self._task_payload: dict[str, Any] = {}
        self.event_count = 0
        self.frame_count = 0

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def save_task(self, episode_spec: EpisodeSpec) -> Path:
        payload = episode_spec.raw_task or asdict(episode_spec)
        self._task_payload = _json_safe(payload)
        return self._write_json("task.json", payload)

    def record_event(self, event: dict[str, Any]) -> None:
        self._append_jsonl(self.events_path, event)
        self.event_count += 1

    def record_step(self, record: StepRecord) -> None:
        sample_report = self._dataset_writer.record(record)
        metadata = dict(record.metadata)
        if sample_report is not None:
            metadata["dataset_sample"] = sample_report
        payload = {
            "step_index": record.step_index,
            "timestamp": record.timestamp,
            "pipeline_state": record.pipeline_state,
            "observation": _simulation_state_payload(record.observation),
            "action": _json_safe(record.action),
            "post_step_observation": _simulation_state_payload(record.post_step_observation),
            "metadata": _json_safe(metadata),
        }
        self._append_jsonl(self.frames_path, payload)
        self.frame_count += 1

    def prepare_lerobot_export(self) -> dict[str, Any]:
        raw_payload = {
            "recording_enabled": self._lerobot_config.enabled,
            "raw_episode_ready": self._dataset_writer.frame_count > 0,
            "raw_data_path": str(self._dataset_writer.csv_path),
            "raw_image_dir": str(self._dataset_writer.image_dir),
            "raw_samples_path": str(self._dataset_writer.samples_path),
            "sampled_frame_count": self._dataset_writer.frame_count,
            "fps": self._lerobot_config.fps,
            "capture_every_n_steps": self._lerobot_config.capture_every_n_steps,
            "jpeg_quality": self._lerobot_config.jpeg_quality,
            "image_size": [
                self._lerobot_config.image_height,
                self._lerobot_config.image_width,
            ],
            "frequency_report": dict(self._dataset_writer.frequency_report),
            "source_frames": str(self.frames_path),
            "frame_count": self.frame_count,
        }
        if not self._lerobot_config.enabled:
            payload = {
                **raw_payload,
                "lerobot_exported": False,
                "reason": "lerobot_recording_disabled_for_execution_mode",
            }
        elif self._dataset_writer.frame_count <= 0:
            payload = {
                **raw_payload,
                "lerobot_exported": False,
                "reason": "no_synchronized_front_camera_frames",
            }
        else:
            try:
                conversion = materialize_lerobot_dataset(
                    [self.output_dir],
                    self.output_dir / "lerobot_dataset",
                    fps=self._lerobot_config.fps,
                    chunks_size=self._lerobot_config.chunks_size,
                )
            except Exception as exc:
                conversion = {
                    "lerobot_exported": False,
                    "reason": "lerobot_conversion_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            payload = {**raw_payload, **conversion}
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


def _simulation_state_payload(state: SimulationState) -> dict[str, Any]:
    """序列化数值状态，但只记录相机名称，避免把像素写入 frames.jsonl。"""

    return {
        "step_index": state.step_index,
        "timestamp": state.timestamp,
        "robot_root_pose": state.robot_root_pose,
        "robot_root_velocity": state.robot_root_velocity,
        "joint_positions": state.joint_positions,
        "joint_velocities": state.joint_velocities,
        "tcp_pose": state.tcp_pose,
        "object_pose": state.object_pose,
        "object_velocity": state.object_velocity,
        "camera_images": sorted(str(name) for name in state.camera_images),
        "metadata": state.metadata,
    }
