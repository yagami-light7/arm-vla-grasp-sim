"""Structured JSONL recorder used by the new pipeline."""

from __future__ import annotations

import hashlib
import json
import math
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
    training_wrist_camera_object_clearance_verified,
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
    if isinstance(value, float) and not math.isfinite(value):
        # JSON 标准没有 NaN/Infinity；诊断统计已单独保留 nonfinite_count，
        # 原始非有限采样统一写 null，保证 jq 和严格解析器都能读取结果。
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


class JsonlEpisodeRecorder:
    """Write task, events, frames, export manifest, and final summary."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        lerobot_config: LeRobotRecordingConfig | None = None,
        diagnostic_frame_stride: int = 1,
    ):
        if (
            isinstance(diagnostic_frame_stride, bool)
            or not isinstance(diagnostic_frame_stride, int)
            or diagnostic_frame_stride < 1
        ):
            raise ValueError("diagnostic_frame_stride 必须是正整数")
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self._output_dir / "events.jsonl"
        self.frames_path = self._output_dir / "frames.jsonl"
        self.navigation_path_snapshots_path = (
            self._output_dir / "navigation_path_snapshots.jsonl"
        )
        self.events_path.write_text("", encoding="utf-8")
        self.frames_path.write_text("", encoding="utf-8")
        self.navigation_path_snapshots_path.write_text("", encoding="utf-8")
        self._lerobot_config = lerobot_config or LeRobotRecordingConfig()
        self._diagnostic_frame_stride = diagnostic_frame_stride
        self._dataset_writer = DwaEpisodeWriter(self._output_dir, self._lerobot_config)
        self._task_payload: dict[str, Any] = {}
        self._training_eligible = True
        self._training_eligibility_reason: str | None = None
        self._performance_profiler: Any | None = None
        self.event_count = 0
        self.frame_count = 0
        self.control_step_count = 0
        self.navigation_path_snapshot_count = 0
        self._navigation_path_snapshot_source_identities: set[str] = set()
        self._navigation_path_snapshot_fingerprints: set[str] = set()
        self._last_logged_pipeline_state: str | None = None

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def set_performance_profiler(self, profiler: Any | None) -> None:
        self._performance_profiler = profiler
        self._dataset_writer.set_performance_profiler(profiler)

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
        profiler = self._performance_profiler
        if profiler is None:
            sample_report = self._dataset_writer.record(record)
        else:
            with profiler.measure("recorder.dataset_record"):
                sample_report = self._dataset_writer.record(record)
        self._record_navigation_path_snapshots(record)
        metadata = dict(record.metadata)
        if sample_report is not None:
            metadata["dataset_sample"] = sample_report
        state_changed = record.pipeline_state != self._last_logged_pipeline_state
        self.control_step_count += 1
        stair_probe_frame = bool(
            record.action.metadata.get("stair_fixed_command_probe") is True
        )
        diagnostic_stride_frame = bool(
            not self._lerobot_config.enabled
            and (
                self.control_step_count == 1
                or self.control_step_count % self._diagnostic_frame_stride == 0
            )
        )
        # Full-physics diagnostics follow the 5 Hz dataset grid and always keep
        # state transitions. Non-recording smoke 默认 stride=1 保留每 tick；live
        # 验收可显式降采样笨重 JSON 诊断。固定命令 probe 始终逐 control tick 留证。
        should_log_frame = bool(
            diagnostic_stride_frame
            or sample_report is not None
            or state_changed
            or stair_probe_frame
        )
        self._last_logged_pipeline_state = record.pipeline_state
        if not should_log_frame:
            return
        stair_probe_telemetry = _aligned_stair_probe_telemetry(record)
        payload = {
            "step_index": record.step_index,
            "timestamp": record.timestamp,
            "pipeline_state": record.pipeline_state,
            "observation": _simulation_state_payload(
                record.observation,
                include_full_metadata=state_changed,
            ),
            "action": _json_safe(record.action),
            "post_step_observation": _simulation_state_payload(
                record.post_step_observation,
                include_full_metadata=state_changed,
            ),
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
            "diagnostic_log_reason": (
                "state_transition"
                if state_changed
                else (
                    "stair_probe_low_level_telemetry"
                    if stair_probe_frame
                    else (
                        "dataset_sample"
                        if sample_report is not None
                        else "diagnostic_stride"
                    )
                )
            ),
        }
        if stair_probe_telemetry is not None:
            # probe 专用大字段只在对应 action 的 frame 保存，避免普通导航帧
            # 重复最后一份 height scan/contact 遥测。
            payload["stair_probe_low_level_telemetry"] = (
                stair_probe_telemetry
            )
        if profiler is None:
            self._append_jsonl(self.frames_path, payload)
        else:
            with profiler.measure("recorder.frames_jsonl_write"):
                self._append_jsonl(self.frames_path, payload)
        self.frame_count += 1

    def _record_navigation_path_snapshots(self, record: StepRecord) -> None:
        """每个 ROS Path 代际只保存一次完整点列，避免逐帧日志重复膨胀。"""

        for captured_from, state in (
            ("observation", record.observation),
            ("post_step_observation", record.post_step_observation),
        ):
            report = state.metadata.get("scan_reference_path_last_report")
            if not isinstance(report, dict):
                continue
            # compact frame 中只有哈希；只有运行时原始报告带该键，才有资格
            # 成为后续 SCAN/DWA 复用的精确 Path 输入证据。
            if "points_ground_xyz" not in report:
                continue
            points = report.get("points_ground_xyz")
            try:
                point_count: int | None = len(points)  # type: ignore[arg-type]
            except TypeError:
                point_count = None
            # 运行时在新 Path 到达前会一直复用同一代际。先用小型身份跳过
            # 逐 tick 重复项，避免为一条百点路径反复做 JSON 序列化和 SHA。
            source_identity = json.dumps(
                _json_safe(
                    {
                        "source": report.get("source"),
                        "topic": report.get("topic"),
                        "frame_id": report.get("frame_id"),
                        "stamp": report.get("stamp"),
                        "sequence": report.get("sequence"),
                        "points_sha256": report.get("points_sha256"),
                        "point_count": point_count,
                        "terminal_yaw": report.get("terminal_yaw"),
                        "cleared": report.get("cleared"),
                    }
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if (
                source_identity
                in self._navigation_path_snapshot_source_identities
            ):
                continue
            report_payload = _json_safe(report)
            canonical_report = json.dumps(
                report_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            fingerprint = hashlib.sha256(canonical_report).hexdigest()
            if fingerprint in self._navigation_path_snapshot_fingerprints:
                continue
            payload = {
                "schema": "navigation_path_snapshot_v1",
                "snapshot_index": self.navigation_path_snapshot_count + 1,
                "captured_from": captured_from,
                "step_index": record.step_index,
                "timestamp": record.timestamp,
                "pipeline_state": record.pipeline_state,
                "state_step_index": state.step_index,
                "state_timestamp": state.timestamp,
                "report_payload_sha256": fingerprint,
                "report": report_payload,
            }
            profiler = self._performance_profiler
            if profiler is None:
                self._append_jsonl(self.navigation_path_snapshots_path, payload)
            else:
                with profiler.measure("recorder.navigation_path_snapshot_write"):
                    self._append_jsonl(
                        self.navigation_path_snapshots_path,
                        payload,
                    )
            self._navigation_path_snapshot_fingerprints.add(fingerprint)
            self._navigation_path_snapshot_source_identities.add(source_identity)
            self.navigation_path_snapshot_count += 1

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
            "camera_capture_transient_missing_keys": writer_report[
                "missing_camera_keys"
            ],
            "camera_state_synchronization": writer_report[
                "camera_state_synchronization"
            ],
            "sampling_coverage": writer_report["sampling_coverage"],
            "async_encoding_and_write": writer_report[
                "async_encoding_and_write"
            ],
            "async_queue_size": writer_report["async_queue_size"],
            "async_max_queue_depth": writer_report["async_max_queue_depth"],
            "async_queue_block_seconds": writer_report[
                "async_queue_block_seconds"
            ],
            "committed_frame_count": writer_report["committed_frame_count"],
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
            "control_step_count": self.control_step_count,
            "num_frames": self._dataset_writer.frame_count,
            "episode_success_verified": bool(self._training_eligible),
            "training_eligible": bool(
                self._training_eligible
                and self._lerobot_config.enabled
                and self._dataset_writer.frame_count > 0
                and writer_report["sampling_coverage"]["verified"] is True
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
        elif writer_report["sampling_coverage"]["verified"] is not True:
            payload = {
                **raw_payload,
                "lerobot_exported": False,
                "success": False,
                "failure_reason": "episode_sampling_coverage_incomplete",
                "reason": "episode_sampling_coverage_incomplete",
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
        actual_camera_keys = set(payload.get("camera_keys") or ())
        requested_camera_keys = set(raw_payload["camera_keys_requested"])
        payload["camera_capture_transient_missing_keys"] = list(
            raw_payload["camera_capture_transient_missing_keys"]
        )
        payload["missing_camera_keys"] = sorted(
            requested_camera_keys - actual_camera_keys
        )
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
            elif not training_wrist_camera_object_clearance_verified(summary):
                ineligibility_reason = (
                    "wrist_camera_object_clearance_not_verified"
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
            "control_step_count": self.control_step_count,
            "navigation_path_snapshot_count": (
                self.navigation_path_snapshot_count
            ),
            "navigation_path_snapshots_path": str(
                self.navigation_path_snapshots_path
            ),
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
                json.dumps(
                    _json_safe(payload),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
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
            json.dumps(
                _json_safe(payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _append_jsonl(path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _json_safe(payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            stream.write("\n")


def _simulation_state_payload(
    state: SimulationState,
    *,
    include_full_metadata: bool = True,
) -> dict[str, Any]:
    """序列化数值状态，但只记录相机名称，避免把像素写入 frames.jsonl。"""

    metadata = (
        dict(state.metadata)
        if include_full_metadata
        else _compact_simulation_metadata(state.metadata)
    )
    # probe 大字段只允许出现在 frame 顶层；从嵌套状态中移除可避免同一帧
    # 重复两次，也避免 probe 结束后的状态切换帧误带上一控制步证据。
    metadata.pop("stair_probe_low_level_telemetry", None)
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
        "metadata": metadata,
    }


def _aligned_stair_probe_telemetry(
    record: StepRecord,
) -> dict[str, Any] | None:
    """取出与当前 action 同拍的 probe 遥测，并验证 pre/post step 对齐。"""

    if record.action.metadata.get("stair_fixed_command_probe") is not True:
        return None
    candidate = record.post_step_observation.metadata.get(
        "stair_probe_low_level_telemetry"
    )
    if not isinstance(candidate, dict):
        return {
            "available": False,
            "unavailable_reason": "runtime_probe_telemetry_missing",
            "expected_pre_step_index": int(record.observation.step_index),
            "expected_post_step_index": int(
                record.post_step_observation.step_index
            ),
        }
    alignment = candidate.get("alignment")
    if not isinstance(alignment, dict):
        return {
            "available": False,
            "unavailable_reason": "runtime_probe_alignment_missing",
            "runtime_report": candidate,
        }
    expected_pre = int(record.observation.step_index)
    expected_post = int(record.post_step_observation.step_index)
    actual_pre = alignment.get("pre_step_state_step_index")
    actual_post = alignment.get("post_step_state_step_index")
    if (
        actual_pre != expected_pre
        or actual_post != expected_post
        or candidate.get("complete") is not True
    ):
        return {
            "available": False,
            "unavailable_reason": "runtime_probe_step_alignment_mismatch",
            "expected_pre_step_index": expected_pre,
            "actual_pre_step_index": actual_pre,
            "expected_post_step_index": expected_post,
            "actual_post_step_index": actual_post,
            "runtime_report": candidate,
        }
    return candidate


def _compact_simulation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """保留逐拍关键值，并移除只需在终态汇总中保存的历史数组。"""

    keys = (
        "physics_dt",
        "control_dt",
        "decimation",
        "camera_render_interval_control_steps",
        "camera_render_hz",
        "camera_capture_report",
        "environment_terminated",
        "joint_names",
        "base_pose_xyyaw",
        "body_velocity",
        "body_linear_velocity",
        "last_action_source",
        "used_base_teleport",
        "used_direct_joint_state",
        "used_object_teleport",
        "used_kinematic_object_follow",
        "used_visual_replay",
        "used_manipulation_base_lock",
        "used_manipulation_support_joint_lock",
        # ROS 2 导航联调必须逐 tick 保留“收到什么、实际写了什么”以及传感器
        # 发布新鲜度；否则超时后的终态只能看到零速，无法还原真实控制链。
        "scan_cmd_vel_last_write_report",
        "navigation_policy_gate_lifecycle_report",
        "grid_map_observation_diagnostics_last_report",
        "grid_map_observation_lifecycle_report",
        "bspline_diagnostics_last_report",
        "bspline_diagnostics_lifecycle_report",
        "active_sensing_lifecycle_report",
        "dynamic_navigation_evidence_report",
        "dynamic_obstacle_runtime_report",
        "dynamic_obstacle_lifecycle_report",
        "dynamic_obstacle_raw_cloud_last_report",
        "dynamic_obstacle_raw_cloud_lifecycle_report",
        "scan_controller_status_last_report",
        "scan_controller_status_lifecycle_report",
        "scan_goal_reached_last_sample",
        "navigation_stair_execution_frozen_last_publish_report",
        "navigation_ros2_last_publish_report",
    )
    compact = {key: metadata[key] for key in keys if key in metadata}
    lifecycle_history_keys = {
        "navigation_policy_gate_lifecycle_report": {
            "identity_verified_tracking_write_reports",
        },
        "grid_map_observation_lifecycle_report": {
            "diagnostic_reports",
        },
        "bspline_diagnostics_lifecycle_report": {
            "diagnostic_reports",
            "trajectory_identities",
        },
        "active_sensing_lifecycle_report": {
            "attempts",
        },
        "scan_controller_status_lifecycle_report": {
            "accepted_status_reports",
            "tracking_status_reports",
            "accepted_trajectory_identities",
        },
    }
    for report_key, omitted_keys in lifecycle_history_keys.items():
        report = compact.get(report_key)
        if not isinstance(report, dict):
            continue
        # 完整历史仍由 episode summary 原样保存；逐拍只需计数、首末样本
        # 与最近状态，否则同一有界历史会在数百帧中重复写入数 GB JSONL。
        compact[report_key] = {
            key: value
            for key, value in report.items()
            if key not in omitted_keys
        }
    path_report = metadata.get("scan_reference_path_last_report")
    if isinstance(path_report, dict):
        # 点数组已由 executor 用同一几何哈希验证；逐帧只保留代际证据，避免
        # 长路径在数千控制 tick 中重复膨胀 JSONL。
        compact["scan_reference_path_last_report"] = {
            key: value
            for key, value in path_report.items()
            if key != "points_ground_xyz"
        }
    return compact
