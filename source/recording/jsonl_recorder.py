"""Structured JSONL recorder used by the new pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from source.interfaces import EpisodeSpec, SimulationState, StepRecord

from .subtask_export import update_subtask_task_gate, write_subtask_task_stub
from .subtask_segmentation import (
    SUBTASK_DIRECTORY_LAYOUT,
    SUBTASK_LABELS,
    SUBTASK_SCHEMA_VERSION,
    TASK_STAGES,
    task_requests_subtask_segmentation,
)
from .lerobot_dataset import (
    ACTION_NAMES,
    BASE_POSE_NAMES,
    BASE_VELOCITY_NAMES,
    CONTROL_ACTION_SCHEMA,
    DwaEpisodeWriter,
    LeRobotRecordingConfig,
    OBJECT_STATE_NAMES,
    SCHEMA_VERSION,
    STATE_NAMES,
    TCP_POSE_NAMES,
    VLA_TRAINING_ACTION_DIMENSION,
    VLA_TRAINING_ACTION_NAMES,
    VLA_TRAINING_ACTION_SCHEMA,
    materialize_lerobot_dataset,
)
from .training_action import (
    VLA_TRAINING_ACTION_ALIGNMENT,
    VLA_TRAINING_TERMINAL_ACTION,
    physical_execution_success_verified,
    task_requests_vla_training_action,
    training_mesh_truth_manipulation_targets_verified,
    training_quality_success_verified,
    training_receptacle_support_verified,
    training_visual_source_verified,
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
        self._training_eligible = True
        self._training_eligibility_reason: str | None = None
        self.event_count = 0
        self.frame_count = 0

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def save_task(self, episode_spec: EpisodeSpec) -> Path:
        payload = episode_spec.raw_task or asdict(episode_spec)
        self._task_payload = _json_safe(payload)
        path = self._write_json("task.json", payload)
        if task_requests_subtask_segmentation(self._task_payload):
            write_subtask_task_stub(
                dataset_root=self.output_dir / "lerobot_dataset",
                task=self._task_payload,
                dataset_schema_version=SCHEMA_VERSION,
            )
        return path

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
            "dataset_frame_index": (
                sample_report.get("frame_index") if sample_report is not None else None
            ),
            "dataset_timestamp": (
                sample_report.get("timestamp") if sample_report is not None else None
            ),
            "video_features": (
                sample_report.get("video_features", {}) if sample_report is not None else {}
            ),
            "metadata": _json_safe(metadata),
        }
        self._append_jsonl(self.frames_path, payload)
        self.frame_count += 1

    def mark_training_eligible(self, eligible: bool, *, reason: str | None = None) -> None:
        """标记当前 episode 是否允许导出训练数据；失败样本只能保留诊断文件。"""

        self._training_eligible = bool(eligible)
        self._training_eligibility_reason = reason

    def prepare_lerobot_export(
        self,
        *,
        training_eligible: bool | None = None,
        training_eligibility_reason: str | None = None,
    ) -> dict[str, Any]:
        if training_eligible is not None:
            self.mark_training_eligible(
                bool(training_eligible),
                reason=training_eligibility_reason,
            )
        writer_report = self._dataset_writer.finalize()
        camera_keys = list(writer_report["camera_keys"])
        vla_training_action_requested = task_requests_vla_training_action(
            self._task_payload
        )
        subtask_segmentation_requested = task_requests_subtask_segmentation(
            self._task_payload
        )
        action_names = (
            VLA_TRAINING_ACTION_NAMES
            if vla_training_action_requested
            else ACTION_NAMES
        )
        feature_keys = [
            "observation.state",
            "observation.base_velocity",
            "observation.base_pose",
            "observation.object_state",
            "observation.tcp_pose",
            "pipeline_state",
            "action",
            "next.done",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            *(f"observation.images.{key}" for key in camera_keys),
        ]
        if vla_training_action_requested:
            feature_keys.insert(feature_keys.index("next.done"), "control.action")
        if subtask_segmentation_requested:
            pipeline_state_index = feature_keys.index("pipeline_state") + 1
            feature_keys[pipeline_state_index:pipeline_state_index] = [
                "task_stage",
                "subtask",
                "subtask_segment_index",
            ]
        raw_payload = {
            "schema_version": SCHEMA_VERSION,
            "recording_enabled": self._lerobot_config.enabled,
            "raw_episode_ready": self._dataset_writer.frame_count > 0,
            "episode_index": 0,
            "episode_dir": str(self.output_dir),
            "dataset_root": str(self.output_dir / "lerobot_dataset"),
            "parquet_path": None,
            "raw_data_path": str(self._dataset_writer.csv_path),
            "raw_image_dir": (
                str(self._dataset_writer.image_root)
                if self._lerobot_config.save_raw_images
                else None
            ),
            "raw_samples_path": str(self._dataset_writer.samples_path),
            "sampled_frame_count": self._dataset_writer.frame_count,
            "fps": self._lerobot_config.dataset_fps,
            "dataset_fps": self._lerobot_config.dataset_fps,
            "capture_every_n_steps": self._lerobot_config.capture_every_n_steps,
            "jpeg_quality": self._lerobot_config.jpeg_quality,
            "camera_keys_requested": list(self._lerobot_config.camera_keys),
            "camera_keys": camera_keys,
            "missing_camera_keys": writer_report["missing_camera_keys"],
            "video_paths": {
                key: str(
                    self._dataset_writer.video_staging_root
                    / f"observation.images.{key}"
                    / "episode.mp4"
                )
                for key in camera_keys
            },
            "feature_keys": feature_keys,
            "observation_state_names": list(STATE_NAMES),
            "action_names": list(action_names),
            "control_action_schema": CONTROL_ACTION_SCHEMA,
            "control_action_dimension": len(ACTION_NAMES),
            "control_action_names": list(ACTION_NAMES),
            "vla_training_action_schema": VLA_TRAINING_ACTION_SCHEMA,
            "vla_training_action_dimension": VLA_TRAINING_ACTION_DIMENSION,
            "vla_training_action_names": list(VLA_TRAINING_ACTION_NAMES),
            "vla_training_action_requested": vla_training_action_requested,
            "vla_training_action_available": False,
            "vla_training_eligible": False,
            "vla_training_ineligibility_reason": (
                "training_action_10d_not_exported"
                if vla_training_action_requested
                else "task_does_not_request_vla_training_action"
            ),
            "training_action_alignment": VLA_TRAINING_ACTION_ALIGNMENT,
            "training_action_horizon_frames": 1,
            "training_action_terminal_action": VLA_TRAINING_TERMINAL_ACTION,
            "training_action_base_pose_frame": "world",
            "training_action_tcp_pose_frame": "base_frame",
            "training_action_tcp_euler_order": "roll_pitch_yaw",
            "training_action_position_unit": "m",
            "training_action_angle_unit": "rad",
            "training_action_gripper_range": [0.0, 1.0],
            "subtask_segmentation_requested": subtask_segmentation_requested,
            "subtask_segmentation_available": False,
            "subtask_directory_export_available": False,
            "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
            "subtask_directory_layout": SUBTASK_DIRECTORY_LAYOUT,
            "task_stages": list(TASK_STAGES),
            "subtask_labels": list(SUBTASK_LABELS),
            "base_pose_names": list(BASE_POSE_NAMES),
            "base_pose_frame": "world",
            "object_state_names": list(OBJECT_STATE_NAMES),
            "tcp_pose_names": list(TCP_POSE_NAMES),
            "tcp_pose_frame": "world",
            "base_velocity_names": list(BASE_VELOCITY_NAMES),
            "task_index": 0,
            "task_text": str(
                self._task_payload.get("instruction")
                or "Complete the navigation pick and place task."
            ),
            "raw_images_saved": self._lerobot_config.save_raw_images,
            "image_size": [
                self._lerobot_config.image_height,
                self._lerobot_config.image_width,
            ],
            "frequency_report": dict(self._dataset_writer.frequency_report),
            "source_frames": str(self.frames_path),
            "frame_count": self.frame_count,
            "num_frames": self._dataset_writer.frame_count,
            "episode_success_verified": bool(self._training_eligible),
            "training_eligible": bool(
                self._training_eligible
                and self._lerobot_config.enabled
                and self._dataset_writer.frame_count > 0
            ),
        }
        if not self._training_eligible:
            payload = {
                **raw_payload,
                "lerobot_exported": False,
                "success": False,
                "failure_reason": "episode_not_training_eligible",
                "reason": self._training_eligibility_reason
                or "episode_not_training_eligible",
            }
            path = self._write_json("lerobot_manifest.json", payload)
            return {**payload, "manifest_path": str(path)}
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
        elif not self._lerobot_config.debug_per_episode_lerobot:
            # batch 模式可只保留原始 episode，结束后统一生成 dataset root。
            unified_root = self.output_dir.parents[1] / "lerobot_dataset"
            payload = {
                **raw_payload,
                "dataset_root": str(unified_root),
                "lerobot_exported": True,
                "success": True,
                "failure_reason": None,
                "export_deferred_to_unified_dataset": True,
                "reason": None,
                "validation_report": None,
            }
        else:
            try:
                conversion = materialize_lerobot_dataset(
                    [self.output_dir],
                    self.output_dir / "lerobot_dataset",
                    fps=self._lerobot_config.dataset_fps,
                    chunks_size=self._lerobot_config.chunks_size,
                    validate=self._lerobot_config.validate_export,
                )
            except Exception as exc:
                conversion = {
                    "lerobot_exported": False,
                    "reason": "lerobot_conversion_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            payload = {**raw_payload, **conversion}
        export_ready = bool(payload.get("lerobot_exported"))
        vla_action_available = bool(payload.get("vla_training_action_available"))
        subtask_export_available = bool(
            payload.get("subtask_directory_export_available")
        )
        training_eligible = bool(
            raw_payload["training_eligible"]
            and export_ready
            and (
                not vla_training_action_requested
                or vla_action_available
            )
            and (
                not subtask_segmentation_requested
                or subtask_export_available
            )
        )
        vla_training_eligible = bool(
            training_eligible
            and vla_training_action_requested
            and vla_action_available
        )
        payload["episode_success_verified"] = bool(self._training_eligible)
        payload["training_eligible"] = training_eligible
        payload["vla_training_eligible"] = vla_training_eligible
        if vla_training_eligible:
            payload["vla_training_ineligibility_reason"] = None
        elif vla_training_action_requested and not vla_action_available:
            payload["vla_training_ineligibility_reason"] = (
                "training_action_10d_not_exported"
            )
        elif vla_training_action_requested and not self._training_eligible:
            payload["vla_training_ineligibility_reason"] = (
                self._training_eligibility_reason
                or "episode_success_gate_not_verified"
            )
        elif subtask_segmentation_requested and not subtask_export_available:
            payload["vla_training_ineligibility_reason"] = (
                "subtask_directory_export_not_available"
            )
        path = self._write_json("lerobot_manifest.json", payload)
        return {**payload, "manifest_path": str(path)}

    def close(self, summary: dict[str, Any]) -> Path:
        self._dataset_writer.finalize()
        success = bool(summary.get("success", False))
        physical_success_verified = physical_execution_success_verified(summary)
        training_quality_verified = training_quality_success_verified(summary)
        if not success:
            failure_reason = str(summary.get("failure_reason") or "episode_failed")
            self.mark_training_eligible(False, reason=failure_reason)
            self._remove_lerobot_artifacts()
        elif not training_quality_verified:
            if not physical_success_verified:
                ineligibility_reason = "physical_execution_provenance_not_verified"
            elif not training_visual_source_verified(summary):
                ineligibility_reason = "vla_rgb_camera_capture_not_verified"
            elif not training_receptacle_support_verified(summary):
                ineligibility_reason = (
                    "task_receptacle_support_runtime_not_verified"
                )
            elif not training_mesh_truth_manipulation_targets_verified(summary):
                ineligibility_reason = (
                    "mesh_truth_manipulation_targets_not_verified"
                )
            else:  # pragma: no cover - 未来新增质量门禁的兜底。
                ineligibility_reason = "training_quality_gate_not_verified"
            self.mark_training_eligible(
                False,
                reason=ineligibility_reason,
            )
        existing_export = dict(summary.get("lerobot_export") or {})
        if not training_quality_verified:
            existing_export.update(
                {
                    "training_eligible": False,
                    "vla_training_eligible": False,
                    "episode_success_verified": False,
                    "vla_training_ineligibility_reason": (
                        self._training_eligibility_reason
                        or "physical_execution_provenance_not_verified"
                    ),
                }
            )
        payload = {
            **summary,
            "lerobot_export": existing_export,
            "event_count": self.event_count,
            "frame_count": self.frame_count,
            "data_output_path": str(self.output_dir),
            "lerobot_training_eligible": bool(
                training_quality_verified
                and existing_export.get("training_eligible")
            ),
            "lerobot_export_skipped": not bool(
                existing_export.get("lerobot_exported")
            ),
            "training_quality_gate_passed": training_quality_verified,
        }
        if not success:
            payload["lerobot_export_skip_reason"] = (
                self._training_eligibility_reason or "episode_failed"
            )
            payload["lerobot_export"] = {
                **existing_export,
                "lerobot_exported": False,
                "training_eligible": False,
                "reason": payload["lerobot_export_skip_reason"],
                "manifest_path": None,
            }
            if existing_export:
                payload["lerobot_export"]["original_lerobot_exported"] = existing_export.get(
                    "lerobot_exported"
                )
                payload["lerobot_export"]["original_reason"] = existing_export.get(
                    "reason"
                )
        elif not training_quality_verified:
            payload["lerobot_export_skip_reason"] = (
                self._training_eligibility_reason
                or "physical_execution_provenance_not_verified"
            )
            gate_eligible = False
            gate_reason = payload["lerobot_export_skip_reason"]
            self._write_export_training_gate(
                eligible=False,
                reason=gate_reason,
            )
        else:
            gate_eligible = bool(payload["lerobot_training_eligible"])
            gate_reason = (
                None
                if gate_eligible
                else "lerobot_export_not_training_ready"
            )
            self._write_export_training_gate(
                eligible=gate_eligible,
                reason=gate_reason,
            )
        if success:
            self._apply_export_training_gate(
                payload["lerobot_export"],
                eligible=gate_eligible,
                reason=gate_reason,
            )
        return self._write_json("summary.json", payload)

    def _write_export_training_gate(
        self,
        *,
        eligible: bool,
        reason: str | None,
    ) -> None:
        """把最终物理证据门禁同步到 manifest 与 LeRobot info。"""

        for path in (
            self.output_dir / "lerobot_manifest.json",
            self.output_dir / "lerobot_dataset/meta/info.json",
            self.output_dir / "lerobot_dataset/validation_report.json",
        ):
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            self._apply_export_training_gate(
                payload,
                eligible=eligible,
                reason=reason,
            )
            path.write_text(
                json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        update_subtask_task_gate(
            self.output_dir / "lerobot_dataset",
            eligible=eligible,
            reason=reason,
        )

    @classmethod
    def _apply_export_training_gate(
        cls,
        payload: dict[str, Any],
        *,
        eligible: bool,
        reason: str | None,
    ) -> None:
        """同步最终成功门禁，并修正导出阶段内嵌的预终态快照。"""

        vla_available = bool(payload.get("vla_training_action_available"))
        if not vla_available:
            details = payload.get("details")
            info = details.get("info") if isinstance(details, dict) else None
            if isinstance(info, dict):
                vla_available = bool(info.get("vla_training_action_available"))
        vla_reason = (
            None
            if eligible and vla_available
            else (
                reason
                or payload.get("vla_training_ineligibility_reason")
                or "task_does_not_request_vla_training_action"
            )
        )
        gate = {
            "episode_success_verified": bool(eligible),
            "training_eligible": bool(eligible),
            "vla_training_eligible": bool(eligible and vla_available),
            "vla_training_ineligibility_reason": vla_reason,
        }
        payload.update(gate)
        if "source_episode_success_verified" in payload:
            payload["source_episode_success_verified"] = bool(eligible)

        validation_report = payload.get("validation_report")
        if isinstance(validation_report, dict):
            cls._apply_export_training_gate(
                validation_report,
                eligible=eligible,
                reason=reason,
            )
        details = payload.get("details")
        info = details.get("info") if isinstance(details, dict) else None
        if isinstance(info, dict):
            info_vla_available = bool(info.get("vla_training_action_available"))
            info_reason = (
                None
                if eligible and info_vla_available
                else (
                    reason
                    or info.get("vla_training_ineligibility_reason")
                    or "task_does_not_request_vla_training_action"
                )
            )
            info.update(
                {
                    "episode_success_verified": bool(eligible),
                    "training_eligible": bool(eligible),
                    "vla_training_eligible": bool(eligible and info_vla_available),
                    "vla_training_ineligibility_reason": info_reason,
                }
            )

    def _remove_lerobot_artifacts(self) -> None:
        """失败 episode 不留下 LeRobot manifest/dataset，避免被后续训练误收集。"""

        manifest_path = self.output_dir / "lerobot_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        dataset_path = self.output_dir / "lerobot_dataset"
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        if task_requests_subtask_segmentation(self._task_payload):
            write_subtask_task_stub(
                dataset_root=dataset_path,
                task=self._task_payload,
                dataset_schema_version=SCHEMA_VERSION,
            )
            update_subtask_task_gate(
                dataset_path,
                eligible=False,
                reason=self._training_eligibility_reason or "episode_failed",
            )

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
