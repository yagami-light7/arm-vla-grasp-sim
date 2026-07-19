"""Full-physics 原始数据记录与 LeRobot v2.1 数据集生成。"""

from __future__ import annotations

import csv
import json
import math
import queue
import shutil
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from source.interfaces import StepRecord

from .subtask_segmentation import (
    INSTRUCTION_ANNOTATION_SCHEMA,
    RELATIVE_DIRECTION_LABELS,
    SUBTASK_DIRECTORY_LAYOUT,
    SUBTASK_LABELS,
    SUBTASK_SCHEMA_VERSION,
    TASK_STAGES,
    extract_action_semantics,
    hydrate_sample_action_semantics,
    segment_episode_samples,
    task_requests_subtask_segmentation,
    validate_subtask_segmentation_config,
)
from .subtask_export import materialize_subtask_episode, update_subtask_task_gate
from .training_action import (
    BASE_POSE_NAMES,
    VLA_TRAINING_ACTION_ALIGNMENT,
    VLA_TRAINING_ACTION_DIMENSION,
    VLA_TRAINING_ACTION_NAMES,
    VLA_TRAINING_ACTION_SCHEMA,
    VLA_TRAINING_TERMINAL_ACTION,
    build_vla_training_actions,
    task_requests_vla_training_action,
    training_quality_success_verified,
    validate_vla_training_action_config,
)


LEGACY_DWA_CSV_COLUMNS = (
    "时间戳(秒)",
    "位置X",
    "位置Y",
    "位置Z",
    "偏航角",
    "线速度X",
    "线速度Y",
    "线速度Z",
    "末端X",
    "末端Y",
    "末端Z",
    "末端Roll",
    "末端Pitch",
    "末端Yaw",
    "关节1",
    "关节2",
    "关节3",
    "关节4",
    "关节5",
    "关节6",
    "夹爪",
    "前摄像头图像",
)

# 保留 DWA 数值列，同时显式记录 pipeline state，便于离线按阶段筛选。
# wrist 图像本来已经进入 samples.jsonl/LeRobot video feature；这里补齐 raw CSV 的人工可读列。
DWA_CSV_COLUMNS_WITHOUT_WRIST = (*LEGACY_DWA_CSV_COLUMNS, "pipeline_state")
DWA_CSV_COLUMNS = (*LEGACY_DWA_CSV_COLUMNS, "腕部摄像头图像", "pipeline_state")
SUPPORTED_DWA_CSV_COLUMNS = {
    LEGACY_DWA_CSV_COLUMNS,
    DWA_CSV_COLUMNS_WITHOUT_WRIST,
    DWA_CSV_COLUMNS,
}

STATE_COLUMNS = (
    "位置X",
    "位置Y",
    "位置Z",
    "偏航角",
    "末端X",
    "末端Y",
    "末端Z",
    "末端Roll",
    "末端Pitch",
    "末端Yaw",
    "关节1",
    "关节2",
    "关节3",
    "关节4",
    "关节5",
    "关节6",
    "夹爪",
)

STATE_NAMES = (
    "base_x",
    "base_y",
    "base_z",
    "base_yaw",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_roll",
    "tcp_pitch",
    "tcp_yaw",
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
    "gripper_joint7_joint8_mean",
)

BASE_VELOCITY_NAMES = (
    "vx_body",
    "vy_body",
    "wz_body",
)

ACTION_NAMES = (
    "base_cmd_vx",
    "base_cmd_vy",
    "base_cmd_wz",
    "arm_joint1_target",
    "arm_joint2_target",
    "arm_joint3_target",
    "arm_joint4_target",
    "arm_joint5_target",
    "arm_joint6_target",
    "gripper_joint7_target",
    "gripper_joint8_target",
)

OBJECT_STATE_NAMES = (
    "object_x",
    "object_y",
    "object_z",
    "object_quat_w",
    "object_quat_x",
    "object_quat_y",
    "object_quat_z",
    "object_vx",
    "object_vy",
    "object_vz",
    "object_wx",
    "object_wy",
    "object_wz",
)

TCP_POSE_NAMES = (
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_quat_w",
    "tcp_quat_x",
    "tcp_quat_y",
    "tcp_quat_z",
)

SCHEMA_VERSION = "full_physics_lerobot_v2.2.0"
CAMERA_STATE_TIMESTAMP_TOLERANCE_SECONDS = 1.0e-9
CONTROL_ACTION_SCHEMA = "base_velocity_arm_joint_gripper_targets_v1"


@dataclass(frozen=True)
class LeRobotRecordingConfig:
    """LeRobot 采样、相机和本地调试输出配置。"""

    enabled: bool = False
    control_dt: float = 0.02
    dataset_fps: float = 5.0
    image_height: int = 480
    image_width: int = 640
    jpeg_quality: int = 90
    chunks_size: int = 1000
    camera_keys: tuple[str, ...] = ("front", "wrist", "overview")
    primary_camera_key: str = "front"
    save_raw_images: bool = True
    debug_per_episode_lerobot: bool = True
    unified_dataset: bool = True
    validate_export: bool = True
    # Low-level tests and library callers remain synchronous unless requested;
    # the real pipeline enables this through RecordingSettings.
    async_encoding_and_write: bool = False
    async_queue_size: int = 16

    @property
    def capture_every_n_steps(self) -> float:
        """返回平均控制步间隔；15 FPS 等非整数比率由时间栅格调度。"""

        return (1.0 / float(self.dataset_fps)) / float(self.control_dt)


@dataclass(frozen=True)
class SynchronizedSamplePacket:
    """Immutable transaction handed from the simulation thread to the I/O worker."""

    episode_id: str
    frame_index: int
    simulation_step: int
    simulation_timestamp: float
    pipeline_state: str
    state_snapshot: dict[str, Any]
    action_snapshot: tuple[float, ...]
    images: tuple[tuple[str, np.ndarray], ...]
    csv_row: dict[str, Any]
    sample_payload: dict[str, Any]


def _quat_wxyz_to_rpy(quat: tuple[float, ...]) -> tuple[float, float, float]:
    if len(quat) != 4:
        return 0.0, 0.0, 0.0
    w, x, y, z = (float(value) for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _image_to_rgb_uint8(image: Any, *, camera_key: str) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"{camera_key} camera image must be HxWx3/4, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _joint_values(record: StepRecord) -> tuple[tuple[float, ...], tuple[float, float]]:
    state = record.observation
    joint_names = tuple(str(name) for name in state.metadata.get("joint_names", ()))
    positions = tuple(float(value) for value in state.joint_positions)
    by_name = dict(zip(joint_names, positions))
    arm = tuple(float(by_name.get(f"arm_joint{index}", 0.0)) for index in range(1, 7))
    gripper_values = [
        float(by_name[name])
        for name in ("arm_joint7", "arm_joint8")
        if name in by_name
    ]
    if len(gripper_values) == 2:
        gripper = (gripper_values[0], gripper_values[1])
    elif len(positions) >= 8 and not joint_names:
        arm = tuple(positions[-8:-2])
        gripper = (float(positions[-2]), float(positions[-1]))
    else:
        gripper = (0.0, 0.0)
    return arm, gripper


def _measured_base_velocity(record: StepRecord) -> tuple[float, float, float]:
    """优先读取 adapter 已转换到 body frame 的 vx/vy/wz。"""

    state = record.observation
    body_velocity = state.metadata.get("body_velocity")
    if isinstance(body_velocity, (list, tuple)) and len(body_velocity) >= 3:
        return tuple(float(value) for value in body_velocity[:3])
    return (
        float(state.robot_root_velocity[0]),
        float(state.robot_root_velocity[1]),
        float(state.robot_root_velocity[5]),
    )


class DwaEpisodeWriter:
    """按固定数据时间栅格记录一个连续 full-physics episode。"""

    def __init__(self, episode_dir: str | Path, config: LeRobotRecordingConfig):
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.config = config
        self.csv_path = self.episode_dir / "data.csv"
        self.image_root = self.episode_dir / "images"
        self.image_dir = self.image_root / config.primary_camera_key
        self.video_staging_root = self.episode_dir / "recording_videos"
        self.samples_path = self.episode_dir / "samples.jsonl"
        self.frame_count = 0
        self._last_sampled_sim_step = -1
        self._next_sample_timestamp: float | None = None
        self._max_observed_sim_step = -1
        self._max_observed_sim_timestamp = 0.0
        self._last_arm_target: tuple[float, ...] | None = None
        self._last_gripper_target: tuple[float, float] | None = None
        self._video_writers: dict[str, Any] = {}
        self._camera_frame_counts: dict[str, int] = {}
        self._camera_shapes: dict[str, tuple[int, int, int]] = {}
        self._missing_camera_keys: set[str] = set()
        self._performance_profiler: Any | None = None
        self._packet_queue: queue.Queue[SynchronizedSamplePacket | None] | None = None
        self._packet_worker: threading.Thread | None = None
        self._packet_worker_stopped = False
        self._worker_error: BaseException | None = None
        self._worker_error_lock = threading.Lock()
        self._committed_frame_indices: list[int] = []
        self._max_queue_depth = 0
        self._queue_block_seconds = 0.0
        self._synchronization_errors: list[dict[str, Any]] = []
        self._finalized = False
        self.frequency_report: dict[str, Any] = {
            "physics_dt": None,
            "physics_hz": None,
            "control_dt": self.config.control_dt,
            "control_hz": 1.0 / self.config.control_dt,
            "dataset_fps": self.config.dataset_fps,
            "capture_every_n_control_steps": self.config.capture_every_n_steps,
            "sampling_mode": "fixed_dataset_time_grid",
        }
        if not self.config.enabled:
            return
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_raw_images:
            for camera_key in self.config.camera_keys:
                image_dir = self.image_root / camera_key
                image_dir.mkdir(parents=True, exist_ok=True)
                for stale_image in image_dir.glob("*.jpg"):
                    stale_image.unlink()
        if self.video_staging_root.exists():
            shutil.rmtree(self.video_staging_root)
        self.video_staging_root.mkdir(parents=True, exist_ok=True)
        self.samples_path.write_text("", encoding="utf-8")
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=DWA_CSV_COLUMNS).writeheader()
        if self.config.async_encoding_and_write:
            self._packet_queue = queue.Queue(maxsize=self.config.async_queue_size)
            self._packet_worker = threading.Thread(
                target=self._packet_worker_main,
                name=f"lerobot-writer-{self.episode_dir.name}",
                daemon=True,
            )
            self._packet_worker.start()

    @property
    def actual_camera_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.config.camera_keys
            if self._camera_frame_counts.get(key) == self.frame_count and self.frame_count > 0
        )

    @property
    def missing_camera_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._missing_camera_keys))

    @property
    def raw_images_saved(self) -> bool:
        return bool(self.config.save_raw_images)

    def set_performance_profiler(self, profiler: Any | None) -> None:
        self._performance_profiler = profiler

    def _measure(self, operation: str):
        if self._performance_profiler is None:
            return nullcontext()
        return self._performance_profiler.measure(operation)

    def record(self, record: StepRecord) -> dict[str, Any] | None:
        if not self.config.enabled or self._finalized:
            return None
        self._raise_worker_error()
        # During live stage reuse, the pre-tick observation of BUILD_STAGE still
        # belongs to the previous episode.  It is useful in frames.jsonl for
        # diagnostics but must never establish the new dataset sampling grid.
        if record.pipeline_state == "build_stage":
            return None
        full_action = self._update_full_action(record)
        state = record.observation
        sim_step = int(state.step_index)
        sim_timestamp = float(state.timestamp)
        self._max_observed_sim_step = max(self._max_observed_sim_step, sim_step)
        self._max_observed_sim_timestamp = max(
            self._max_observed_sim_timestamp,
            sim_timestamp,
        )
        if sim_step < self._last_sampled_sim_step:
            raise RuntimeError(
                "simulation step regressed within one recorded episode: "
                f"last_sampled={self._last_sampled_sim_step} current={sim_step}"
            )
        if sim_step == self._last_sampled_sim_step:
            return None

        # 第一个有效相机帧立即采样；后续按 dataset timestamp 选择最近控制帧。
        if self._next_sample_timestamp is not None:
            half_control_dt = 0.5 * float(self.config.control_dt)
            if sim_timestamp + half_control_dt < self._next_sample_timestamp:
                return None
        primary_image = state.camera_images.get(self.config.primary_camera_key)
        if primary_image is None:
            self._missing_camera_keys.add(self.config.primary_camera_key)
            return None

        self._update_frequency_report(state.metadata)
        capture_report = state.metadata.get("camera_capture_report")
        if isinstance(capture_report, dict):
            capture_step = int(capture_report.get("capture_step_index", sim_step))
            capture_timestamp = float(
                capture_report.get("capture_timestamp", sim_timestamp)
            )
            synchronization_source = str(
                capture_report.get("synchronization_source", "runtime_capture_report")
            )
        else:
            if self.config.async_encoding_and_write:
                raise RuntimeError(
                    "asynchronous camera export requires camera_capture_report timestamps"
                )
            capture_step = sim_step
            capture_timestamp = sim_timestamp
            synchronization_source = "state_snapshot_fallback"
        timestamp_error = abs(capture_timestamp - sim_timestamp)
        synchronization_error = None
        if capture_step != sim_step:
            synchronization_error = {
                "reason": "camera_state_step_mismatch",
                "frame_index": self.frame_count,
                "camera_capture_step": capture_step,
                "state_step": sim_step,
            }
        elif timestamp_error > CAMERA_STATE_TIMESTAMP_TOLERANCE_SECONDS:
            synchronization_error = {
                "reason": "camera_state_timestamp_mismatch",
                "frame_index": self.frame_count,
                "camera_capture_timestamp": capture_timestamp,
                "state_timestamp": sim_timestamp,
                "absolute_error_seconds": timestamp_error,
            }
        if synchronization_error is not None:
            self._synchronization_errors.append(synchronization_error)
            raise RuntimeError(
                "camera/state synchronization failed: "
                f"{synchronization_error['reason']}"
            )

        frame_index = self.frame_count
        camera_frames: dict[str, dict[str, Any]] = {}
        prepared_images: list[tuple[str, np.ndarray]] = []
        for camera_key in self.config.camera_keys:
            camera_image = state.camera_images.get(camera_key)
            if camera_image is None:
                self._missing_camera_keys.add(camera_key)
                if self.config.async_encoding_and_write:
                    raise RuntimeError(
                        f"asynchronous sample packet is missing camera: {camera_key}"
                    )
                continue
            with self._measure("recorder.image_freeze_and_prepare"):
                image = self._prepare_image(camera_image, camera_key=camera_key).copy()
                image.setflags(write=False)
            prepared_images.append((camera_key, image))
            raw_path = self._raw_image_relative_path(camera_key, frame_index)
            camera_frames[camera_key] = {
                "feature_key": f"observation.images.{camera_key}",
                "frame_index": frame_index,
                "timestamp": float(frame_index) / float(self.config.dataset_fps),
                "simulation_step": sim_step,
                "simulation_timestamp": sim_timestamp,
                "camera_capture_step": capture_step,
                "camera_capture_timestamp": capture_timestamp,
                "state_step": sim_step,
                "state_timestamp": sim_timestamp,
                "timestamp_alignment_error_seconds": timestamp_error,
                "synchronization_source": synchronization_source,
                "raw_image_path": raw_path,
            }

        primary_raw_path = camera_frames[self.config.primary_camera_key]["raw_image_path"]
        wrist_raw_path = camera_frames.get("wrist", {}).get("raw_image_path")
        row = self._build_row(
            record,
            frame_index=frame_index,
            image_name=Path(primary_raw_path).name if primary_raw_path else "",
            wrist_image_name=Path(wrist_raw_path).name if wrist_raw_path else "",
        )
        sample = {
            "frame_index": frame_index,
            "timestamp": float(frame_index) / float(self.config.dataset_fps),
            "simulation_step": sim_step,
            "simulation_timestamp": sim_timestamp,
            "camera_capture_step": capture_step,
            "camera_capture_timestamp": capture_timestamp,
            "camera_state_timestamp_error_seconds": timestamp_error,
            "camera_state_synchronized": True,
            "pipeline_state": record.pipeline_state,
            "base_velocity": list(_measured_base_velocity(record)),
            "action": list(full_action),
            "base_pose": list(state.robot_root_pose),
            "object_state": self._object_state(record),
            "tcp_pose": list(
                state.tcp_pose or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            ),
            "tcp_pose_valid": state.tcp_pose is not None,
            "gripper_position": float(sum(_joint_values(record)[1]) / 2.0),
            "camera_frames": camera_frames,
            **extract_action_semantics(
                {
                    "source": record.action.source,
                    "gripper_command": record.action.gripper_command,
                    "metadata": record.action.metadata,
                }
            ),
        }
        packet = SynchronizedSamplePacket(
            episode_id=self.episode_dir.name,
            frame_index=frame_index,
            simulation_step=sim_step,
            simulation_timestamp=sim_timestamp,
            pipeline_state=record.pipeline_state,
            state_snapshot={
                "step_index": sim_step,
                "timestamp": sim_timestamp,
                "robot_root_pose": tuple(state.robot_root_pose),
                "robot_root_velocity": tuple(state.robot_root_velocity),
                "joint_positions": tuple(state.joint_positions),
                "joint_velocities": tuple(state.joint_velocities),
                "tcp_pose": state.tcp_pose,
                "object_pose": state.object_pose,
                "object_velocity": state.object_velocity,
            },
            action_snapshot=tuple(full_action),
            images=tuple(prepared_images),
            csv_row=row,
            sample_payload=sample,
        )
        if self._packet_queue is None:
            self._commit_packet(packet)
        else:
            self._enqueue_packet(packet)

        report = {
            "frame_index": frame_index,
            "timestamp": sample["timestamp"],
            "simulation_step": sim_step,
            "simulation_timestamp": sim_timestamp,
            "camera_capture_step": capture_step,
            "camera_capture_timestamp": capture_timestamp,
            "camera_state_synchronized": True,
            "pipeline_state": record.pipeline_state,
            "async_queued": self._packet_queue is not None,
            "video_features": {
                key: value["feature_key"] for key, value in camera_frames.items()
            },
            "raw_images": {
                key: value["raw_image_path"] for key, value in camera_frames.items()
            },
        }
        self.frame_count += 1
        self._last_sampled_sim_step = sim_step
        if self._next_sample_timestamp is None:
            self._next_sample_timestamp = sim_timestamp
        self._next_sample_timestamp += 1.0 / float(self.config.dataset_fps)
        return report

    def _enqueue_packet(self, packet: SynchronizedSamplePacket) -> None:
        assert self._packet_queue is not None
        started_at = time.perf_counter()
        self._packet_queue.put(packet)
        blocked = time.perf_counter() - started_at
        self._queue_block_seconds += blocked
        self._max_queue_depth = max(self._max_queue_depth, self._packet_queue.qsize())
        if self._performance_profiler is not None:
            self._performance_profiler.record("recorder.async_queue_put", blocked)
        self._raise_worker_error()

    def _packet_worker_main(self) -> None:
        assert self._packet_queue is not None
        while True:
            packet = self._packet_queue.get()
            try:
                if packet is None:
                    return
                if self._worker_error is None:
                    self._commit_packet(packet)
            except BaseException as exc:  # pragma: no cover - exercised via injected failure test.
                with self._worker_error_lock:
                    if self._worker_error is None:
                        self._worker_error = exc
            finally:
                self._packet_queue.task_done()

    def _commit_packet(self, packet: SynchronizedSamplePacket) -> None:
        if packet.frame_index != len(self._committed_frame_indices):
            raise RuntimeError(
                "sample packet commit order mismatch: "
                f"expected={len(self._committed_frame_indices)} "
                f"actual={packet.frame_index}"
            )
        for camera_key, image in packet.images:
            with self._measure("recorder.raw_jpeg_write"):
                self._write_raw_image(camera_key, image, packet.frame_index)
            with self._measure("recorder.staged_video_write"):
                self._write_video_frame(camera_key, image)
        with self._measure("recorder.csv_write"):
            with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=DWA_CSV_COLUMNS).writerow(
                    packet.csv_row
                )
        with self._measure("recorder.samples_jsonl_write"):
            with self.samples_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        packet.sample_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
        self._committed_frame_indices.append(packet.frame_index)

    def _raise_worker_error(self) -> None:
        with self._worker_error_lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError(
                "asynchronous LeRobot writer failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _stop_packet_worker(self) -> None:
        if self._packet_queue is None or self._packet_worker_stopped:
            return
        self._packet_queue.put(None)
        assert self._packet_worker is not None
        self._packet_worker.join()
        self._packet_worker_stopped = True
        self._raise_worker_error()

    def finalize(self) -> dict[str, Any]:
        """结束 MP4 编码，保证转换器读取到完整 moov/frame metadata。"""

        if self._finalized:
            return self.report()
        self._stop_packet_worker()
        with self._measure("recorder.staged_video_finalize"):
            for writer in self._video_writers.values():
                writer.release()
        self._video_writers.clear()
        expected_indices = list(range(self.frame_count))
        if self._committed_frame_indices != expected_indices:
            raise RuntimeError(
                "sample packet commit sequence is incomplete: "
                f"expected={len(expected_indices)} "
                f"committed={len(self._committed_frame_indices)}"
            )
        if self._synchronization_errors:
            raise RuntimeError(
                "camera/state synchronization errors were recorded: "
                f"{len(self._synchronization_errors)}"
            )
        self._finalized = True
        return self.report()

    def report(self) -> dict[str, Any]:
        expected_sample_count = (
            int(
                math.floor(
                    self._max_observed_sim_timestamp
                    * float(self.config.dataset_fps)
                    + 1.0e-9
                )
            )
            + 1
            if self._max_observed_sim_step >= 0
            else 0
        )
        allowed_missing_count = (
            max(2, int(math.ceil(expected_sample_count * 0.02)))
            if expected_sample_count > 0
            else 0
        )
        minimum_sample_count = max(
            0,
            expected_sample_count - allowed_missing_count,
        )
        coverage_verified = bool(
            expected_sample_count == 0
            or self.frame_count >= minimum_sample_count
        )
        return {
            "sampled_frame_count": self.frame_count,
            "camera_keys": list(self.actual_camera_keys),
            "missing_camera_keys": list(self.missing_camera_keys),
            "camera_frame_counts": dict(self._camera_frame_counts),
            "camera_shapes": {
                key: list(shape) for key, shape in self._camera_shapes.items()
            },
            "raw_images_saved": self.raw_images_saved,
            "video_staging_root": str(self.video_staging_root),
            "async_encoding_and_write": self.config.async_encoding_and_write,
            "async_queue_size": self.config.async_queue_size,
            "async_max_queue_depth": self._max_queue_depth,
            "async_queue_block_seconds": self._queue_block_seconds,
            "committed_frame_count": len(self._committed_frame_indices),
            "committed_frame_indices_contiguous": (
                self._committed_frame_indices == list(range(self.frame_count))
            ),
            "sampling_coverage": {
                "verified": coverage_verified,
                "sampled_frame_count": self.frame_count,
                "expected_sample_count": expected_sample_count,
                "minimum_sample_count": minimum_sample_count,
                "allowed_missing_count": allowed_missing_count,
                "max_observed_sim_step": self._max_observed_sim_step,
                "max_observed_sim_timestamp": self._max_observed_sim_timestamp,
                "dataset_fps": float(self.config.dataset_fps),
                "rule": "sampled_frames_cover_observed_episode_time_grid",
            },
            "camera_state_synchronization": {
                "verified": bool(
                    not self._synchronization_errors
                    and self._committed_frame_indices == list(range(self.frame_count))
                ),
                "error_count": len(self._synchronization_errors),
                "errors": list(self._synchronization_errors),
                "timestamp_tolerance_seconds": (
                    CAMERA_STATE_TIMESTAMP_TOLERANCE_SECONDS
                ),
                "rule": "camera_capture_step_equals_state_step",
            },
        }

    def _prepare_image(self, image: Any, *, camera_key: str) -> np.ndarray:
        image = _image_to_rgb_uint8(image, camera_key=camera_key)
        expected_size = (self.config.image_width, self.config.image_height)
        if (image.shape[1], image.shape[0]) != expected_size:
            image = np.asarray(
                Image.fromarray(image).resize(expected_size, Image.Resampling.BILINEAR)
            )
        return np.ascontiguousarray(image)

    def _raw_image_relative_path(
        self,
        camera_key: str,
        frame_index: int,
    ) -> str | None:
        if not self.config.save_raw_images:
            return None
        prefix = "camera0" if camera_key == "front" else camera_key
        image_name = f"{prefix}_{frame_index:05d}.jpg"
        image_path = self.image_root / camera_key / image_name
        return str(image_path.relative_to(self.episode_dir))

    def _write_raw_image(
        self,
        camera_key: str,
        image: np.ndarray,
        frame_index: int,
    ) -> str | None:
        relative_path = self._raw_image_relative_path(camera_key, frame_index)
        if relative_path is None:
            return None
        image_path = self.episode_dir / relative_path
        temporary_path = image_path.with_suffix(image_path.suffix + ".tmp")
        Image.fromarray(image).save(
            temporary_path,
            format="JPEG",
            quality=self.config.jpeg_quality,
        )
        temporary_path.replace(image_path)
        return relative_path

    def _write_video_frame(self, camera_key: str, image: np.ndarray) -> None:
        import cv2

        writer = self._video_writers.get(camera_key)
        if writer is None:
            video_path = (
                self.video_staging_root
                / f"observation.images.{camera_key}"
                / "episode.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.config.dataset_fps),
                (image.shape[1], image.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to create staged camera video: {video_path}")
            self._video_writers[camera_key] = writer
            self._camera_shapes[camera_key] = tuple(int(value) for value in image.shape)
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self._camera_frame_counts[camera_key] = self._camera_frame_counts.get(camera_key, 0) + 1

    def _update_frequency_report(self, metadata: dict[str, Any]) -> None:
        physics_dt = metadata.get("physics_dt")
        control_dt = metadata.get("control_dt")
        if isinstance(physics_dt, (int, float)) and float(physics_dt) > 0.0:
            self.frequency_report["physics_dt"] = float(physics_dt)
            self.frequency_report["physics_hz"] = 1.0 / float(physics_dt)
        if isinstance(control_dt, (int, float)) and float(control_dt) > 0.0:
            self.frequency_report["control_dt"] = float(control_dt)
            self.frequency_report["control_hz"] = 1.0 / float(control_dt)

    def _update_full_action(self, record: StepRecord) -> tuple[float, ...]:
        arm_actual, gripper_actual = _joint_values(record)
        if record.action.arm_joint_positions is not None:
            self._last_arm_target = tuple(
                float(value) for value in record.action.arm_joint_positions[:6]
            )
        elif self._last_arm_target is None:
            self._last_arm_target = arm_actual

        metadata = record.action.metadata or {}
        gripper_positions = metadata.get("gripper_joint_positions")
        if isinstance(gripper_positions, (list, tuple)) and gripper_positions:
            values = tuple(float(value) for value in gripper_positions)
            self._last_gripper_target = (
                values[0],
                values[1] if len(values) > 1 else values[0],
            )
        elif record.action.gripper_command == "open":
            self._last_gripper_target = (0.04, 0.04)
        elif record.action.gripper_command == "close":
            self._last_gripper_target = (0.0, 0.0)
        elif self._last_gripper_target is None:
            self._last_gripper_target = gripper_actual

        return (
            *(float(value) for value in record.action.base_velocity),
            *self._last_arm_target,
            *self._last_gripper_target,
        )

    @staticmethod
    def _object_state(record: StepRecord) -> list[float]:
        state = record.observation
        pose = state.object_pose or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        velocity = state.object_velocity or (0.0,) * 6
        return [*(float(value) for value in pose), *(float(value) for value in velocity)]

    def _build_row(
        self,
        record: StepRecord,
        *,
        frame_index: int,
        image_name: str,
        wrist_image_name: str = "",
    ) -> dict[str, Any]:
        state = record.observation
        root_pose = state.robot_root_pose
        root_yaw = _quat_wxyz_to_rpy(tuple(root_pose[3:7]))[2]
        # CSV 历史列继续保存线速度 XYZ；训练用 vx/vy/wz 单独写入 samples/parquet。
        body_linear_velocity = tuple(
            float(value) for value in state.metadata.get("body_linear_velocity", ())
        )
        if len(body_linear_velocity) < 3:
            body_linear_velocity = tuple(float(value) for value in state.robot_root_velocity[:3])

        tcp_pose = state.tcp_pose or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        tcp_rpy = _quat_wxyz_to_rpy(tuple(tcp_pose[3:7]))
        arm, gripper_positions = _joint_values(record)
        gripper = sum(gripper_positions) / len(gripper_positions)
        row = {
            "时间戳(秒)": f"{float(frame_index) / float(self.config.dataset_fps):.6f}",
            "位置X": f"{float(root_pose[0]):.6f}",
            "位置Y": f"{float(root_pose[1]):.6f}",
            "位置Z": f"{float(root_pose[2]):.6f}",
            "偏航角": f"{root_yaw:.6f}",
            "线速度X": f"{body_linear_velocity[0]:.6f}",
            "线速度Y": f"{body_linear_velocity[1]:.6f}",
            "线速度Z": f"{body_linear_velocity[2]:.6f}",
            "末端X": f"{float(tcp_pose[0]):.6f}",
            "末端Y": f"{float(tcp_pose[1]):.6f}",
            "末端Z": f"{float(tcp_pose[2]):.6f}",
            "末端Roll": f"{tcp_rpy[0]:.6f}",
            "末端Pitch": f"{tcp_rpy[1]:.6f}",
            "末端Yaw": f"{tcp_rpy[2]:.6f}",
            "夹爪": f"{gripper:.6f}",
            "前摄像头图像": image_name,
            "腕部摄像头图像": wrist_image_name,
            "pipeline_state": record.pipeline_state,
        }
        for index, value in enumerate(arm, start=1):
            row[f"关节{index}"] = f"{value:.6f}"
        return row


def _read_task(task_path: Path) -> dict[str, Any]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise ValueError(f"{task_path} must contain a JSON object")
    return task


def _read_instruction(task_path: Path) -> str:
    task = _read_task(task_path)
    return str(task.get("instruction") or "Complete the navigation pick and place task.")


def _read_episode_rows(episode_dir: Path) -> list[dict[str, str]]:
    csv_path = episode_dir / "data.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames not in SUPPORTED_DWA_CSV_COLUMNS:
            raise ValueError(f"{csv_path} has unexpected DWA columns")
        rows = list(reader)
    for row in rows:
        # 兼容旧 raw CSV：历史版本只有 front 图像列，wrist 在 samples.jsonl 中。
        row.setdefault("腕部摄像头图像", "")
        row.setdefault("pipeline_state", "")
    return rows


def _read_episode_samples(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "samples.jsonl"
    if not path.is_file():
        return []
    samples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError(f"{path} contains a non-object JSONL record")
    body_velocity_by_step = _read_body_velocity_by_sim_step(episode_dir)
    for sample in samples:
        action = sample.get("action")
        if isinstance(action, list) and len(action) == 10:
            # 第一版把双指夹爪平均成1维；离线迁移时恢复为两个相同目标。
            sample["action"] = [*action[:9], action[9], action[9]]
        if "base_velocity" not in sample:
            simulation_step = sample.get("simulation_step")
            if isinstance(simulation_step, int) and simulation_step in body_velocity_by_step:
                sample["base_velocity"] = list(body_velocity_by_step[simulation_step])
    return samples


def _sample_base_poses(
    rows: list[dict[str, str]],
    samples: list[dict[str, Any]],
) -> np.ndarray:
    """读取完整 world pose；旧 episode 仅能按 CSV yaw 恢复水平姿态。"""

    poses: list[list[float]] = []
    for row, sample in zip(rows, samples):
        raw_pose = sample.get("base_pose")
        if isinstance(raw_pose, list) and len(raw_pose) == len(BASE_POSE_NAMES):
            pose = [float(value) for value in raw_pose]
        else:
            yaw = float(row["偏航角"])
            pose = [
                float(row["位置X"]),
                float(row["位置Y"]),
                float(row["位置Z"]),
                math.cos(0.5 * yaw),
                0.0,
                0.0,
                math.sin(0.5 * yaw),
            ]
        if not all(math.isfinite(value) for value in pose):
            raise ValueError("episode contains a non-finite base pose")
        poses.append(pose)
    return np.asarray(poses, dtype=np.float32)


def _read_body_velocity_by_sim_step(
    episode_dir: Path,
) -> dict[int, tuple[float, float, float]]:
    """从 50 Hz frames 恢复旧样本缺失的 body-frame vx/vy/wz。"""

    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.is_file():
        return {}
    result: dict[int, tuple[float, float, float]] = {}
    with frames_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            for observation_key in ("observation", "post_step_observation"):
                observation = payload.get(observation_key)
                if not isinstance(observation, dict):
                    continue
                simulation_step = observation.get("step_index")
                metadata = observation.get("metadata")
                body_velocity = (
                    metadata.get("body_velocity") if isinstance(metadata, dict) else None
                )
                if (
                    isinstance(simulation_step, int)
                    and isinstance(body_velocity, list)
                    and len(body_velocity) >= 3
                ):
                    result[simulation_step] = tuple(
                        float(value) for value in body_velocity[:3]
                    )
    return result


def _source_episode_metadata(episode_dir: Path) -> dict[str, Any]:
    """保留 batch 子进程的原 episode/seed，避免统一重编号后丢失来源。"""

    summary_path = episode_dir / "summary.json"
    payload: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("episode_id", "episode_index", "seed"):
            value = summary.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                payload[f"source_{key}"] = int(value)
    parent_name = episode_dir.parent.name
    if parent_name.startswith("episode_"):
        try:
            payload.setdefault("source_episode_index", int(parent_name.split("_", 1)[1]))
        except ValueError:
            pass
    return payload


def _source_episode_success_verified(episode_dir: Path) -> bool:
    """只接受最终 summary 的成功标记作为离线批量训练资格证据。"""

    summary_path = episode_dir / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return bool(
        isinstance(summary, dict)
        and training_quality_success_verified(summary)
    )


def discover_recorded_episodes(
    episodes_root: str | Path,
    *,
    require_success: bool = True,
    require_training_quality: bool = False,
) -> list[Path]:
    """发现原始 episode；训练转换可额外要求最终物理来源门禁。"""

    root = Path(episodes_root).expanduser().resolve()
    episodes: list[Path] = []
    for csv_path in sorted(root.rglob("data.csv")):
        episode_dir = csv_path.parent
        if "lerobot_dataset" in episode_dir.parts:
            continue
        if not (episode_dir / "task.json").is_file():
            continue
        if require_success:
            summary_path = episode_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not bool(summary.get("success")):
                continue
            if require_training_quality and not _source_episode_success_verified(
                episode_dir
            ):
                continue
        episodes.append(episode_dir)
    return episodes


def _compute_stats(array: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def _infer_dataset_fps(episode_dirs: list[Path], requested_fps: float | None) -> float:
    if requested_fps is not None:
        return float(requested_fps)
    detected: list[float] = []
    for episode_dir in episode_dirs:
        manifest_path = episode_dir / "lerobot_manifest.json"
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = payload.get("fps") or payload.get("dataset_fps")
        if isinstance(value, (int, float)) and float(value) > 0.0:
            detected.append(float(value))
    if detected and any(abs(value - detected[0]) > 1.0e-6 for value in detected[1:]):
        raise ValueError(f"recorded episodes use inconsistent dataset fps: {detected}")
    return detected[0] if detected else 5.0


def _sample_camera_keys(
    rows: list[dict[str, str]],
    samples: list[dict[str, Any]],
) -> tuple[str, ...]:
    complete_keys: set[str] | None = None
    for sample in samples:
        camera_frames = sample.get("camera_frames")
        if not isinstance(camera_frames, dict):
            continue
        keys = {str(key) for key in camera_frames}
        complete_keys = keys if complete_keys is None else complete_keys & keys
    if complete_keys:
        return tuple(sorted(complete_keys))
    if rows and all(row.get("前摄像头图像") for row in rows):
        return ("front",)
    return ()


def _copy_or_encode_video(
    *,
    episode_dir: Path,
    output_path: Path,
    camera_key: str,
    episode_index: int,
    chunk_index: int,
    rows: list[dict[str, str]],
    samples: list[dict[str, Any]],
    fps: float,
) -> tuple[Path, tuple[int, int, int]]:
    import cv2

    video_path = (
        output_path
        / "videos"
        / f"chunk-{chunk_index:03d}"
        / f"observation.images.{camera_key}"
        / f"episode_{episode_index:06d}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        episode_dir
        / "recording_videos"
        / f"observation.images.{camera_key}"
        / "episode.mp4",
        episode_dir
        / "lerobot_dataset"
        / "videos/chunk-000"
        / f"observation.images.{camera_key}"
        / "episode_000000.mp4",
    ]
    source_video = next((path for path in candidates if path.is_file()), None)
    if source_video is not None and not _video_matches(
        source_video,
        expected_frames=len(rows),
        expected_fps=fps,
    ):
        source_video = None
    if source_video is not None and source_video.resolve() != video_path.resolve():
        shutil.copyfile(source_video, video_path)
    elif source_video is None:
        image_paths: list[Path] = []
        for row, sample in zip(rows, samples):
            camera_frames = sample.get("camera_frames")
            raw_path = None
            if isinstance(camera_frames, dict):
                frame_info = camera_frames.get(camera_key)
                if isinstance(frame_info, dict):
                    raw_path = frame_info.get("raw_image_path")
            if not raw_path and camera_key == "front":
                image_name = row.get("前摄像头图像")
                raw_path = f"images/front/{image_name}" if image_name else None
            if not raw_path and camera_key == "wrist":
                image_name = row.get("腕部摄像头图像")
                raw_path = f"images/wrist/{image_name}" if image_name else None
            if not raw_path:
                raise RuntimeError(
                    f"{episode_dir} camera {camera_key} has no staged video or raw image"
                )
            image_paths.append(episode_dir / str(raw_path))
        first_image = cv2.imread(str(image_paths[0]))
        if first_image is None:
            raise RuntimeError(f"failed to read episode image: {image_paths[0]}")
        image_height, image_width = first_image.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (image_width, image_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to create MP4 video: {video_path}")
        try:
            for image_path in image_paths:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise RuntimeError(f"failed to read episode image: {image_path}")
                writer.write(frame)
        finally:
            writer.release()

    capture = cv2.VideoCapture(str(video_path))
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    return video_path, (height, width, 3)


def _video_matches(
    video_path: Path,
    *,
    expected_frames: int,
    expected_fps: float,
) -> bool:
    """只复用帧数和 FPS 均正确的旧视频，防止历史少帧继续传播。"""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return False
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return frame_count == expected_frames and math.isclose(
        video_fps,
        expected_fps,
        rel_tol=1.0e-3,
        abs_tol=1.0e-2,
    )


def _feature_metadata(
    *,
    camera_shapes: dict[str, tuple[int, int, int]],
    fps: float,
    action_names: tuple[str, ...],
    include_control_action: bool,
    include_subtasks: bool,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": list(STATE_NAMES),
        },
        "observation.base_velocity": {
            "dtype": "float32",
            "shape": [len(BASE_VELOCITY_NAMES)],
            "names": list(BASE_VELOCITY_NAMES),
        },
        "observation.base_pose": {
            "dtype": "float32",
            "shape": [len(BASE_POSE_NAMES)],
            "names": list(BASE_POSE_NAMES),
        },
        "observation.object_state": {
            "dtype": "float32",
            "shape": [len(OBJECT_STATE_NAMES)],
            "names": list(OBJECT_STATE_NAMES),
        },
        "observation.tcp_pose": {
            "dtype": "float32",
            "shape": [len(TCP_POSE_NAMES)],
            "names": list(TCP_POSE_NAMES),
        },
        "pipeline_state": {"dtype": "string", "shape": [1], "names": None},
        "action": {
            "dtype": "float32",
            "shape": [len(action_names)],
            "names": list(action_names),
        },
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if include_control_action:
        features["control.action"] = {
            "dtype": "float32",
            "shape": [len(ACTION_NAMES)],
            "names": list(ACTION_NAMES),
        }
    if include_subtasks:
        features.update(
            {
                "task_stage": {"dtype": "string", "shape": [1], "names": None},
                "subtask": {"dtype": "string", "shape": [1], "names": None},
                "subtask_segment_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
                "instruction": {
                    "dtype": "string",
                    "shape": [1],
                    "names": None,
                },
                "instruction_id": {
                    "dtype": "string",
                    "shape": [1],
                    "names": None,
                },
                "instruction_target_id": {
                    "dtype": "string",
                    "shape": [1],
                    "names": None,
                },
                "instruction_direction": {
                    "dtype": "string",
                    "shape": [1],
                    "names": None,
                },
                "instruction_relative_bearing_rad": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None,
                },
                "instruction_pose_source": {
                    "dtype": "string",
                    "shape": [1],
                    "names": None,
                },
                "instruction_annotation_schema": {
                    "dtype": "string",
                    "shape": [1],
                    "names": None,
                },
            }
        )
    for camera_key, shape in sorted(camera_shapes.items()):
        features[f"observation.images.{camera_key}"] = {
            "dtype": "video",
            "shape": list(shape),
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
            },
        }
    return features


def materialize_lerobot_dataset(
    episode_dirs: Iterable[str | Path],
    output_root: str | Path,
    *,
    fps: float | None = None,
    chunks_size: int = 1000,
    validate: bool = True,
) -> dict[str, Any]:
    """把一个或多个原始 episode 合并为统一 LeRobot 数据集。"""

    episodes = [Path(path).expanduser().resolve() for path in episode_dirs]
    valid_episodes: list[
        tuple[
            Path,
            str,
            list[dict[str, str]],
            list[dict[str, Any]],
            dict[str, Any],
            dict[str, Any] | None,
            Any,
        ]
    ] = []
    for episode_dir in episodes:
        task_path = episode_dir / "task.json"
        csv_path = episode_dir / "data.csv"
        if not task_path.is_file() or not csv_path.is_file():
            continue
        task = _read_task(task_path)
        training_action_config = validate_vla_training_action_config(task)
        subtask_config = validate_subtask_segmentation_config(task)
        rows = _read_episode_rows(episode_dir)
        if not rows:
            continue
        samples = _read_episode_samples(episode_dir)
        if len(samples) != len(rows):
            raise ValueError(
                f"{episode_dir} data.csv/samples.jsonl length mismatch: "
                f"{len(rows)} != {len(samples)}"
            )
        if subtask_config is not None:
            samples = hydrate_sample_action_semantics(episode_dir, samples)
        valid_episodes.append(
            (
                episode_dir,
                str(
                    task.get("instruction")
                    or "Complete the navigation pick and place task."
                ),
                rows,
                samples,
                task,
                training_action_config,
                subtask_config,
            )
        )

    output_path = Path(output_root).expanduser().resolve()
    dataset_fps = _infer_dataset_fps(
        [episode_dir for episode_dir, *_rest in valid_episodes],
        fps,
    )
    requested_modes = {
        task_requests_vla_training_action(task)
        for _episode_dir, _instruction, _rows, _samples, task, _config, _subtask
        in valid_episodes
    }
    vla_training_action_enabled = requested_modes == {True}
    requested_subtask_modes = {
        task_requests_subtask_segmentation(task)
        for _episode_dir, _instruction, _rows, _samples, task, _config, _subtask
        in valid_episodes
    }
    subtask_segmentation_enabled = requested_subtask_modes == {True}
    instruction_annotation_languages = sorted(
        {
            str(annotation.get("language") or "").strip()
            for *_prefix, task, _config, _subtask in valid_episodes
            for annotation in [task.get("instruction_annotation")]
            if isinstance(annotation, dict)
            and bool(annotation.get("enabled", True))
            and str(annotation.get("language") or "").strip()
        }
    )
    action_names = (
        VLA_TRAINING_ACTION_NAMES if vla_training_action_enabled else ACTION_NAMES
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "lerobot_exported": False,
        "success": False,
        "failure_reason": None,
        "dataset_root": str(output_path),
        "dataset_path": str(output_path),
        "episode_count": len(valid_episodes),
        "fps": dataset_fps,
        "chunks_size": int(chunks_size),
        "format": "lerobot_v2.1_full_physics",
        "observation_state_names": list(STATE_NAMES),
        "action_names": list(action_names),
        "control_action_schema": CONTROL_ACTION_SCHEMA,
        "control_action_dimension": len(ACTION_NAMES),
        "control_action_names": list(ACTION_NAMES),
        "vla_training_action_schema": VLA_TRAINING_ACTION_SCHEMA,
        "vla_training_action_dimension": VLA_TRAINING_ACTION_DIMENSION,
        "vla_training_action_names": list(VLA_TRAINING_ACTION_NAMES),
        "vla_training_action_requested": vla_training_action_enabled,
        "vla_training_action_available": False,
        "vla_training_eligible": False,
        "vla_training_ineligibility_reason": (
            "training_action_conversion_not_completed"
            if vla_training_action_enabled
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
        "base_pose_names": list(BASE_POSE_NAMES),
        "base_pose_frame": "world",
        "tcp_pose_frame": "world",
        "object_state_names": list(OBJECT_STATE_NAMES),
        "tcp_pose_names": list(TCP_POSE_NAMES),
        "base_velocity_names": list(BASE_VELOCITY_NAMES),
        "subtask_segmentation_requested": subtask_segmentation_enabled,
        "subtask_segmentation_available": False,
        "subtask_directory_export_available": False,
        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
        "subtask_directory_layout": SUBTASK_DIRECTORY_LAYOUT,
        "task_stages": list(TASK_STAGES),
        "subtask_labels": list(SUBTASK_LABELS),
        "instruction_annotation_available": False,
        "instruction_annotation_schema": INSTRUCTION_ANNOTATION_SCHEMA,
        "instruction_direction_labels": list(RELATIVE_DIRECTION_LABELS),
        "instruction_annotation_languages": instruction_annotation_languages,
    }
    if not valid_episodes:
        return {**report, "failure_reason": "no_recorded_episode_frames"}
    if len(requested_modes) != 1:
        return {
            **report,
            "failure_reason": "mixed_training_action_schemas",
            "vla_training_ineligibility_reason": "mixed_training_action_schemas",
        }
    if len(requested_subtask_modes) != 1:
        return {
            **report,
            "failure_reason": "mixed_subtask_segmentation_schemas",
            "vla_training_ineligibility_reason": (
                "mixed_subtask_segmentation_schemas"
            ),
        }
    if subtask_segmentation_enabled and not vla_training_action_enabled:
        return {
            **report,
            "failure_reason": "subtask_export_requires_vla_training_action",
            "vla_training_ineligibility_reason": (
                "subtask_export_requires_vla_training_action"
            ),
        }
    if vla_training_action_enabled:
        training_configs = [
            config
            for *_prefix, config, _subtask_config in valid_episodes
            if config is not None
        ]
        if not training_configs or any(
            config != training_configs[0] for config in training_configs[1:]
        ):
            return {
                **report,
                "failure_reason": "inconsistent_training_action_config",
                "vla_training_ineligibility_reason": (
                    "inconsistent_training_action_config"
                ),
            }
    if subtask_segmentation_enabled:
        subtask_configs = [
            subtask_config
            for *_prefix, subtask_config in valid_episodes
            if subtask_config is not None
        ]
        comparable_subtask_configs = [
            {
                key: value
                for key, value in config.metadata().items()
                if key != "config_source"
            }
            for config in subtask_configs
        ]
        if not subtask_configs or any(
            config != comparable_subtask_configs[0]
            for config in comparable_subtask_configs[1:]
        ):
            return {
                **report,
                "failure_reason": "inconsistent_subtask_segmentation_config",
                "vla_training_ineligibility_reason": (
                    "inconsistent_subtask_segmentation_config"
                ),
            }
    resolved_subtask_episode_ids = (
        _resolve_subtask_episode_ids(
            [task for *_prefix, task, _config, _subtask in valid_episodes]
        )
        if subtask_segmentation_enabled
        else []
    )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        return {
            **report,
            "failure_reason": "missing_conversion_dependency",
            "missing_dependency": exc.name,
        }

    if output_path.exists():
        shutil.rmtree(output_path)
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    task_index_map: dict[str, int] = {}
    for (
        _episode_dir,
        instruction,
        _rows,
        _samples,
        _task,
        _config,
        _subtask_config,
    ) in valid_episodes:
        task_index_map.setdefault(instruction, len(task_index_map))
    (meta_dir / "task_index_map.json").write_text(
        json.dumps(task_index_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    schema_fields = [
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field("task_index", pa.int64()),
            pa.field("observation.state", pa.list_(pa.float32(), len(STATE_NAMES))),
            pa.field(
                "observation.base_velocity",
                pa.list_(pa.float32(), len(BASE_VELOCITY_NAMES)),
            ),
            pa.field(
                "observation.base_pose",
                pa.list_(pa.float32(), len(BASE_POSE_NAMES)),
            ),
            pa.field(
                "observation.object_state",
                pa.list_(pa.float32(), len(OBJECT_STATE_NAMES)),
            ),
            pa.field(
                "observation.tcp_pose",
                pa.list_(pa.float32(), len(TCP_POSE_NAMES)),
            ),
            pa.field("pipeline_state", pa.string()),
            pa.field("action", pa.list_(pa.float32(), len(action_names))),
            pa.field("next.done", pa.bool_()),
        ]
    if vla_training_action_enabled:
        schema_fields.insert(
            -1,
            pa.field("control.action", pa.list_(pa.float32(), len(ACTION_NAMES))),
        )
    if subtask_segmentation_enabled:
        action_field_index = next(
            index
            for index, field in enumerate(schema_fields)
            if field.name == "action"
        )
        schema_fields[action_field_index:action_field_index] = [
            pa.field("task_stage", pa.string()),
            pa.field("subtask", pa.string()),
            pa.field("subtask_segment_index", pa.int64()),
            pa.field("instruction", pa.string()),
            pa.field("instruction_id", pa.string()),
            pa.field("instruction_target_id", pa.string()),
            pa.field("instruction_direction", pa.string()),
            pa.field("instruction_relative_bearing_rad", pa.float32()),
            pa.field("instruction_pose_source", pa.string()),
            pa.field("instruction_annotation_schema", pa.string()),
        ]
    schema = pa.schema(schema_fields)
    all_feature_arrays: dict[str, list[np.ndarray]] = {
        "observation.state": [],
        "observation.base_velocity": [],
        "observation.base_pose": [],
        "observation.object_state": [],
        "observation.tcp_pose": [],
        "action": [],
    }
    if vla_training_action_enabled:
        all_feature_arrays["control.action"] = []
    episodes_meta: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    episode_exports: list[dict[str, Any]] = []
    subtask_exports: list[dict[str, Any]] = []
    global_frame_index = 0
    shared_camera_keys: set[str] | None = None
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    all_raw_images_saved = True

    for episode_index, (
        episode_dir,
        instruction,
        rows,
        samples,
        task,
        training_action_config,
        subtask_config,
    ) in enumerate(valid_episodes):
        chunk_index = episode_index // chunks_size
        states = np.asarray(
            [[float(row[column]) for column in STATE_COLUMNS] for row in rows],
            dtype=np.float32,
        )
        base_velocities = np.asarray(
            [
                sample.get(
                    "base_velocity",
                    [
                        float(row["线速度X"]),
                        float(row["线速度Y"]),
                        0.0,
                    ],
                )
                for row, sample in zip(rows, samples)
            ],
            dtype=np.float32,
        )
        base_poses = _sample_base_poses(rows, samples)
        control_actions = np.asarray(
            [sample["action"] for sample in samples],
            dtype=np.float32,
        )
        if vla_training_action_enabled:
            assert training_action_config is not None
            source_gripper_range = tuple(
                float(value)
                for value in training_action_config[
                    "source_gripper_joint_range_m"
                ]
            )
            actions = build_vla_training_actions(
                samples,
                source_gripper_joint_range_m=(
                    source_gripper_range[0],
                    source_gripper_range[1],
                ),
            )
        else:
            actions = control_actions
        object_states = np.asarray(
            [sample["object_state"] for sample in samples],
            dtype=np.float32,
        )
        tcp_poses = np.asarray([sample["tcp_pose"] for sample in samples], dtype=np.float32)
        pipeline_states = [
            str(sample.get("pipeline_state") or row.get("pipeline_state") or "unknown")
            for row, sample in zip(rows, samples)
        ]
        timestamps = [
            float(sample.get("timestamp", frame_index / dataset_fps))
            for frame_index, sample in enumerate(samples)
        ]
        frame_count = len(rows)
        task_index = task_index_map[instruction]
        segmentation: Any | None = None
        if subtask_segmentation_enabled:
            assert subtask_config is not None
            segmentation = segment_episode_samples(
                samples,
                task,
                config=subtask_config,
                fps=dataset_fps,
            )
        table_payload: dict[str, Any] = {
                "index": pa.array(
                    range(global_frame_index, global_frame_index + frame_count),
                    type=pa.int64(),
                ),
                "episode_index": pa.array([episode_index] * frame_count, type=pa.int64()),
                "frame_index": pa.array(range(frame_count), type=pa.int64()),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "task_index": pa.array([task_index] * frame_count, type=pa.int64()),
                "observation.state": pa.array(
                    states.tolist(),
                    type=pa.list_(pa.float32(), len(STATE_NAMES)),
                ),
                "observation.base_velocity": pa.array(
                    base_velocities.tolist(),
                    type=pa.list_(pa.float32(), len(BASE_VELOCITY_NAMES)),
                ),
                "observation.base_pose": pa.array(
                    base_poses.tolist(),
                    type=pa.list_(pa.float32(), len(BASE_POSE_NAMES)),
                ),
                "observation.object_state": pa.array(
                    object_states.tolist(),
                    type=pa.list_(pa.float32(), len(OBJECT_STATE_NAMES)),
                ),
                "observation.tcp_pose": pa.array(
                    tcp_poses.tolist(),
                    type=pa.list_(pa.float32(), len(TCP_POSE_NAMES)),
                ),
                "pipeline_state": pa.array(pipeline_states, type=pa.string()),
                "action": pa.array(
                    actions.tolist(),
                    type=pa.list_(pa.float32(), len(action_names)),
                ),
                "next.done": pa.array(
                    [False] * (frame_count - 1) + [True],
                    type=pa.bool_(),
                ),
            }
        if vla_training_action_enabled:
            table_payload["control.action"] = pa.array(
                control_actions.tolist(),
                type=pa.list_(pa.float32(), len(ACTION_NAMES)),
            )
        if segmentation is not None:
            table_payload.update(
                {
                    "task_stage": pa.array(
                        [frame["task_stage"] for frame in segmentation["frames"]],
                        type=pa.string(),
                    ),
                    "subtask": pa.array(
                        [frame["subtask"] for frame in segmentation["frames"]],
                        type=pa.string(),
                    ),
                    "subtask_segment_index": pa.array(
                        [
                            int(frame["segment_index"])
                            for frame in segmentation["frames"]
                        ],
                        type=pa.int64(),
                    ),
                    "instruction": pa.array(
                        [frame["instruction"] for frame in segmentation["frames"]],
                        type=pa.string(),
                    ),
                    "instruction_id": pa.array(
                        [frame["instruction_id"] for frame in segmentation["frames"]],
                        type=pa.string(),
                    ),
                    "instruction_target_id": pa.array(
                        [
                            frame["instruction_target_id"]
                            for frame in segmentation["frames"]
                        ],
                        type=pa.string(),
                    ),
                    "instruction_direction": pa.array(
                        [
                            frame["instruction_direction"] or ""
                            for frame in segmentation["frames"]
                        ],
                        type=pa.string(),
                    ),
                    "instruction_relative_bearing_rad": pa.array(
                        [
                            frame["instruction_relative_bearing_rad"]
                            for frame in segmentation["frames"]
                        ],
                        type=pa.float32(),
                    ),
                    "instruction_pose_source": pa.array(
                        [
                            frame["instruction_pose_source"]
                            for frame in segmentation["frames"]
                        ],
                        type=pa.string(),
                    ),
                    "instruction_annotation_schema": pa.array(
                        [
                            frame["instruction_annotation_schema"]
                            for frame in segmentation["frames"]
                        ],
                        type=pa.string(),
                    ),
                }
            )
        table = pa.table(
            table_payload,
            schema=schema,
        )
        data_dir = output_path / "data" / f"chunk-{chunk_index:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, parquet_path)

        camera_keys = _sample_camera_keys(rows, samples)
        if not camera_keys:
            raise RuntimeError(f"{episode_dir} has no complete camera stream")
        shared_camera_keys = (
            set(camera_keys)
            if shared_camera_keys is None
            else shared_camera_keys & set(camera_keys)
        )
        video_paths: dict[str, str] = {}
        for camera_key in camera_keys:
            video_path, shape = _copy_or_encode_video(
                episode_dir=episode_dir,
                output_path=output_path,
                camera_key=camera_key,
                episode_index=episode_index,
                chunk_index=chunk_index,
                rows=rows,
                samples=samples,
                fps=dataset_fps,
            )
            camera_shapes[camera_key] = shape
            video_paths[camera_key] = str(video_path)

        raw_images_saved = any(
            isinstance(sample.get("camera_frames"), dict)
            and any(
                isinstance(value, dict) and bool(value.get("raw_image_path"))
                for value in sample["camera_frames"].values()
            )
            for sample in samples
        ) or any((episode_dir / "images").rglob("*.jpg"))
        all_raw_images_saved = all_raw_images_saved and raw_images_saved
        subtask_export: dict[str, Any] | None = None
        if segmentation is not None:
            missing_subtask_cameras = {
                "front",
                "wrist",
            }.difference(camera_keys)
            if missing_subtask_cameras:
                raise RuntimeError(
                    f"{episode_dir} subtask directory export requires front and "
                    f"wrist cameras; missing={sorted(missing_subtask_cameras)}"
                )
            assert training_action_config is not None
            assert subtask_config is not None
            source_range_values = tuple(
                float(value)
                for value in training_action_config[
                    "source_gripper_joint_range_m"
                ]
            )
            subtask_task = dict(task)
            subtask_task["_source_task_episode_id"] = task.get("episode_id")
            subtask_task["episode_id"] = resolved_subtask_episode_ids[
                episode_index
            ]
            subtask_export = materialize_subtask_episode(
                source_episode_dir=episode_dir,
                dataset_root=output_path,
                task=subtask_task,
                episode_index=episode_index,
                global_frame_offset=global_frame_index,
                rows=rows,
                samples=samples,
                training_actions=actions,
                control_actions=control_actions,
                segmentation=segmentation,
                dataset_schema_version=SCHEMA_VERSION,
                source_gripper_joint_range_m=(
                    source_range_values[0],
                    source_range_values[1],
                ),
            )
            subtask_exports.append(subtask_export)
        feature_stats = {
            "observation.state": _compute_stats(states),
            "observation.base_velocity": _compute_stats(base_velocities),
            "observation.base_pose": _compute_stats(base_poses),
            "observation.object_state": _compute_stats(object_states),
            "observation.tcp_pose": _compute_stats(tcp_poses),
            "action": _compute_stats(actions),
        }
        if vla_training_action_enabled:
            feature_stats["control.action"] = _compute_stats(control_actions)
        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": frame_count,
                **(
                    {
                        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
                        "subtask_segment_count": segmentation["segment_count"],
                        "subtask_frame_counts": segmentation["frame_counts"],
                        "subtask_segments": segmentation["segments"],
                        "instruction_annotation": segmentation[
                            "instruction_annotation"
                        ],
                    }
                    if segmentation is not None
                    else {}
                ),
                **_source_episode_metadata(episode_dir),
            }
        )
        episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": feature_stats,
            }
        )
        episode_exports.append(
            {
                "episode_index": episode_index,
                "episode_dir": str(episode_dir),
                **_source_episode_metadata(episode_dir),
                "parquet_path": str(parquet_path),
                "video_paths": video_paths,
                "camera_keys": list(camera_keys),
                "num_frames": frame_count,
                "task_index": task_index,
                "task_text": instruction,
                "raw_images_saved": raw_images_saved,
                **(
                    {"subtask_export": subtask_export}
                    if subtask_export is not None
                    else {}
                ),
            }
        )
        for key, array in (
            ("observation.state", states),
            ("observation.base_velocity", base_velocities),
            ("observation.base_pose", base_poses),
            ("observation.object_state", object_states),
            ("observation.tcp_pose", tcp_poses),
            ("action", actions),
        ):
            all_feature_arrays[key].append(array)
        if vla_training_action_enabled:
            all_feature_arrays["control.action"].append(control_actions)
        global_frame_index += frame_count

    actual_camera_keys = tuple(sorted(shared_camera_keys or ()))
    if not actual_camera_keys:
        raise RuntimeError("episodes do not share any complete camera stream")
    # 统一数据集只能声明所有 episode 都存在的 camera feature。
    for episode_export in episode_exports:
        episode_export["camera_keys"] = list(actual_camera_keys)
        episode_export["video_paths"] = {
            key: value
            for key, value in episode_export["video_paths"].items()
            if key in actual_camera_keys
        }
    camera_shapes = {
        key: shape for key, shape in camera_shapes.items() if key in actual_camera_keys
    }

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for payload in episodes_meta:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task, task_index in sorted(task_index_map.items(), key=lambda item: item[1]):
            stream.write(
                json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False)
                + "\n"
            )
    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as stream:
        for payload in episode_stats:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if subtask_segmentation_enabled:
        with (meta_dir / "subtasks.jsonl").open("w", encoding="utf-8") as stream:
            for payload in subtask_exports:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    stats = {
        key: _compute_stats(np.concatenate(arrays, axis=0))
        for key, arrays in all_feature_arrays.items()
    }
    for camera_key in actual_camera_keys:
        stats[f"observation.images.{camera_key}"] = {
            "min": [[[0.0, 0.0, 0.0]]],
            "max": [[[1.0, 1.0, 1.0]]],
            "mean": [[[0.5, 0.5, 0.5]]],
            "std": [[[0.5, 0.5, 0.5]]],
            "count": [global_frame_index],
        }
    # 保留旧 stats.jsonl，避免已有下游脚本突然失效。
    with (meta_dir / "stats.jsonl").open("w", encoding="utf-8") as stream:
        for key, value in stats.items():
            stream.write(json.dumps({key: value}, ensure_ascii=False) + "\n")

    total_episodes = len(valid_episodes)
    features = _feature_metadata(
        camera_shapes=camera_shapes,
        fps=dataset_fps,
        action_names=tuple(action_names),
        include_control_action=vla_training_action_enabled,
        include_subtasks=subtask_segmentation_enabled,
    )
    source_success_verified = all(
        _source_episode_success_verified(episode_dir)
        for episode_dir, *_rest in valid_episodes
    )
    subtask_export_available = bool(
        subtask_segmentation_enabled
        and len(subtask_exports) == len(valid_episodes)
    )
    vla_training_eligible = bool(
        vla_training_action_enabled
        and source_success_verified
        and (
            not subtask_segmentation_enabled
            or subtask_export_available
        )
    )
    vla_ineligibility_reason = None
    if not vla_training_action_enabled:
        vla_ineligibility_reason = "task_does_not_request_vla_training_action"
    elif not source_success_verified:
        vla_ineligibility_reason = "episode_success_gate_not_verified"
    elif subtask_segmentation_enabled and not subtask_export_available:
        vla_ineligibility_reason = "subtask_directory_export_not_available"
    info = {
        "codebase_version": "v2.1",
        "schema_version": SCHEMA_VERSION,
        "robot_type": "go2_x5_mobile_manipulator",
        "total_episodes": total_episodes,
        "total_frames": global_frame_index,
        "total_tasks": len(task_index_map),
        "total_videos": total_episodes * len(actual_camera_keys),
        "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "splits": {"train": f"0:{total_episodes}"},
        "fps": dataset_fps,
        "video": True,
        "camera_keys": list(actual_camera_keys),
        "features": features,
        "observation_state_names": list(STATE_NAMES),
        "action_names": list(action_names),
        "control_action_schema": CONTROL_ACTION_SCHEMA,
        "control_action_names": list(ACTION_NAMES),
        "vla_training_action_schema": VLA_TRAINING_ACTION_SCHEMA,
        "vla_training_action_names": list(VLA_TRAINING_ACTION_NAMES),
        "vla_training_action_available": vla_training_action_enabled,
        "vla_training_eligible": vla_training_eligible,
        "vla_training_ineligibility_reason": vla_ineligibility_reason,
        "training_action_alignment": VLA_TRAINING_ACTION_ALIGNMENT,
        "training_action_horizon_frames": 1,
        "training_action_terminal_action": VLA_TRAINING_TERMINAL_ACTION,
        "training_action_base_pose_frame": "world",
        "training_action_tcp_pose_frame": "base_frame",
        "training_action_tcp_euler_order": "roll_pitch_yaw",
        "training_action_position_unit": "m",
        "training_action_angle_unit": "rad",
        "training_action_gripper_range": [0.0, 1.0],
        "base_pose_names": list(BASE_POSE_NAMES),
        "base_pose_frame": "world",
        "object_state_names": list(OBJECT_STATE_NAMES),
        "tcp_pose_names": list(TCP_POSE_NAMES),
        "tcp_pose_frame": "world",
        "base_velocity_names": list(BASE_VELOCITY_NAMES),
        "subtask_segmentation_requested": subtask_segmentation_enabled,
        "subtask_segmentation_available": subtask_segmentation_enabled,
        "subtask_directory_export_available": subtask_export_available,
        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
        "subtask_directory_layout": SUBTASK_DIRECTORY_LAYOUT,
        "subtask_directory_root": "episodes",
        "subtask_metadata_path": "meta/subtasks.jsonl",
        "subtask_duplicate_episode_id_policy": (
            "renumber_per_task_from_1_in_dataset_order"
        ),
        "task_stages": list(TASK_STAGES),
        "subtask_labels": list(SUBTASK_LABELS),
        "instruction_annotation_available": subtask_segmentation_enabled,
        "instruction_annotation_schema": INSTRUCTION_ANNOTATION_SCHEMA,
        "instruction_annotation_feature": "instruction",
        "instruction_direction_labels": list(RELATIVE_DIRECTION_LABELS),
        "instruction_direction_frame": "robot_base_at_segment_first_frame",
        "instruction_relative_bearing_unit": "rad",
        "instruction_annotation_languages": instruction_annotation_languages,
        "task_index_semantics": "episode_level_instruction",
        "subtask_min_segment_frames": (
            subtask_configs[0].min_segment_frames
            if subtask_segmentation_enabled
            else None
        ),
        "subtask_hysteresis_frames": (
            subtask_configs[0].hysteresis_frames
            if subtask_segmentation_enabled
            else None
        ),
        "contact_label_source": (
            subtask_configs[0].contact_label_source
            if subtask_segmentation_enabled
            else None
        ),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
    }
    info_path = meta_dir / "info.json"
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        **report,
        "lerobot_exported": True,
        "success": True,
        "frame_count": global_frame_index,
        "num_frames": global_frame_index,
        "task_count": len(task_index_map),
        "camera_keys": list(actual_camera_keys),
        "feature_keys": list(features),
        "raw_images_saved": all_raw_images_saved,
        "info_path": str(info_path),
        "episodes": episode_exports,
        "vla_training_action_available": vla_training_action_enabled,
        "vla_training_eligible": vla_training_eligible,
        "vla_training_ineligibility_reason": vla_ineligibility_reason,
        "source_episode_success_verified": source_success_verified,
        "subtask_segmentation_available": subtask_segmentation_enabled,
        "subtask_directory_export_available": subtask_export_available,
        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
        "subtask_directory_layout": SUBTASK_DIRECTORY_LAYOUT,
        "subtask_exports": subtask_exports,
        "instruction_annotation_available": subtask_segmentation_enabled,
        "instruction_annotation_schema": INSTRUCTION_ANNOTATION_SCHEMA,
        "instruction_direction_labels": list(RELATIVE_DIRECTION_LABELS),
        "instruction_annotation_languages": instruction_annotation_languages,
    }
    if len(episode_exports) == 1:
        result.update(episode_exports[0])

    if validate:
        from .lerobot_validator import validate_lerobot_dataset

        validation_report = validate_lerobot_dataset(output_path)
        validation_success = bool(
            validation_report.get("success", validation_report.get("valid", False))
        )
        result["validation_report"] = validation_report
        result["success"] = validation_success
        result["lerobot_exported"] = validation_success
        result["failure_reason"] = (
            None
            if validation_success
            else validation_report.get("failure_reason", "lerobot_validation_failed")
        )
        if subtask_segmentation_enabled:
            if not validation_success:
                update_subtask_task_gate(
                    output_path,
                    eligible=False,
                    reason=result["failure_reason"],
                )
            elif source_success_verified:
                update_subtask_task_gate(
                    output_path,
                    eligible=vla_training_eligible,
                    reason=vla_ineligibility_reason,
                )
    return result


def _resolve_subtask_episode_ids(tasks: list[dict[str, Any]]) -> list[Any]:
    """避免 batch 子进程重复的 task episode_id 覆盖已有目录。"""

    resolved: list[Any] = [
        task.get("episode_id", index + 1) for index, task in enumerate(tasks)
    ]
    indices_by_task: dict[str, list[int]] = {}
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id", 0))
        indices_by_task.setdefault(task_id, []).append(index)
    for indices in indices_by_task.values():
        path_ids = [str(resolved[index]) for index in indices]
        if len(path_ids) == len(set(path_ids)):
            continue
        for output_episode_id, index in enumerate(indices, start=1):
            resolved[index] = output_episode_id
    return resolved


__all__ = [
    "ACTION_NAMES",
    "BASE_POSE_NAMES",
    "BASE_VELOCITY_NAMES",
    "CONTROL_ACTION_SCHEMA",
    "DWA_CSV_COLUMNS",
    "DwaEpisodeWriter",
    "LEGACY_DWA_CSV_COLUMNS",
    "LeRobotRecordingConfig",
    "OBJECT_STATE_NAMES",
    "SCHEMA_VERSION",
    "STATE_COLUMNS",
    "STATE_NAMES",
    "TCP_POSE_NAMES",
    "VLA_TRAINING_ACTION_DIMENSION",
    "VLA_TRAINING_ACTION_NAMES",
    "VLA_TRAINING_ACTION_SCHEMA",
    "discover_recorded_episodes",
    "materialize_lerobot_dataset",
]
