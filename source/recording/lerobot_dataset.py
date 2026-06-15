"""DWA-compatible raw episode recording and LeRobot v2.1 materialization."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from source.interfaces import StepRecord


DWA_CSV_COLUMNS = (
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

ACTION_COLUMNS = ("线速度X", "线速度Y", "线速度Z")

STATE_NAMES = (
    "pos_x",
    "pos_y",
    "pos_z",
    "yaw",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_roll",
    "ee_pitch",
    "ee_yaw",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper",
)

ACTION_NAMES = ("vel_x", "vel_y", "vel_z")
FULL_ACTION_NAMES = (
    "base_cmd_vx",
    "base_cmd_vy",
    "base_cmd_wz",
    "arm_target_joint1",
    "arm_target_joint2",
    "arm_target_joint3",
    "arm_target_joint4",
    "arm_target_joint5",
    "arm_target_joint6",
    "gripper_target",
)

OBJECT_STATE_NAMES = (
    "object_x",
    "object_y",
    "object_z",
    "object_quat_w",
    "object_quat_x",
    "object_quat_y",
    "object_quat_z",
    "object_vel_x",
    "object_vel_y",
    "object_vel_z",
    "object_ang_vel_x",
    "object_ang_vel_y",
    "object_ang_vel_z",
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


@dataclass(frozen=True)
class LeRobotRecordingConfig:
    """固定为 DWA 数据合同的采样和编码参数。"""

    enabled: bool = False
    control_dt: float = 0.02
    fps: int = 5
    image_height: int = 480
    image_width: int = 640
    jpeg_quality: int = 90
    chunks_size: int = 1000

    @property
    def capture_every_n_steps(self) -> int:
        return max(1, round((1.0 / float(self.fps)) / float(self.control_dt)))


def _quat_wxyz_to_rpy(quat: tuple[float, ...]) -> tuple[float, float, float]:
    if len(quat) != 4:
        return 0.0, 0.0, 0.0
    w, x, y, z = (float(value) for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _image_to_rgb_uint8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"front camera image must be HxWx3/4, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _joint_values(record: StepRecord) -> tuple[tuple[float, ...], float]:
    state = record.post_step_observation
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
        gripper = sum(gripper_values) / 2.0
    elif len(positions) >= 8 and not joint_names:
        arm = tuple(positions[-8:-2])
        gripper = sum(positions[-2:]) / 2.0
    else:
        gripper = 0.0
    return arm, gripper


class DwaEpisodeWriter:
    """按 DWA 的 5 Hz CSV/JPEG 约定记录一个连续 full-physics episode。"""

    def __init__(self, episode_dir: str | Path, config: LeRobotRecordingConfig):
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.config = config
        self.csv_path = self.episode_dir / "data.csv"
        self.image_dir = self.episode_dir / "images" / "front"
        self.samples_path = self.episode_dir / "samples.jsonl"
        self.frame_count = 0
        self._last_sampled_sim_step = -1
        self._next_sample_sim_step: int | None = None
        self._last_arm_target: tuple[float, ...] | None = None
        self._last_gripper_target: float | None = None
        self.frequency_report: dict[str, Any] = {
            "physics_dt": None,
            "physics_hz": None,
            "control_dt": self.config.control_dt,
            "control_hz": 1.0 / self.config.control_dt,
            "capture_fps": self.config.fps,
            "capture_every_n_control_steps": self.config.capture_every_n_steps,
        }
        if not self.config.enabled:
            return
        self.image_dir.mkdir(parents=True, exist_ok=True)
        for stale_image in self.image_dir.glob("camera0_*.jpg"):
            stale_image.unlink()
        self.samples_path.write_text("", encoding="utf-8")
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=DWA_CSV_COLUMNS).writeheader()

    def record(self, record: StepRecord) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None
        full_action = self._update_full_action(record)
        state = record.post_step_observation
        sim_step = int(state.step_index)
        if sim_step <= self._last_sampled_sim_step:
            return None
        if (
            self._next_sample_sim_step is not None
            and sim_step < self._next_sample_sim_step
        ):
            return None
        front_image = state.camera_images.get("front")
        if front_image is None:
            return None
        physics_dt = state.metadata.get("physics_dt")
        control_dt = state.metadata.get("control_dt")
        if isinstance(physics_dt, (int, float)) and float(physics_dt) > 0.0:
            self.frequency_report["physics_dt"] = float(physics_dt)
            self.frequency_report["physics_hz"] = 1.0 / float(physics_dt)
        if isinstance(control_dt, (int, float)) and float(control_dt) > 0.0:
            self.frequency_report["control_dt"] = float(control_dt)
            self.frequency_report["control_hz"] = 1.0 / float(control_dt)

        image = _image_to_rgb_uint8(front_image)
        expected_size = (self.config.image_width, self.config.image_height)
        if (image.shape[1], image.shape[0]) != expected_size:
            image = np.asarray(Image.fromarray(image).resize(expected_size, Image.Resampling.BILINEAR))

        image_name = f"camera0_{self.frame_count:05d}.jpg"
        Image.fromarray(image).save(
            self.image_dir / image_name,
            format="JPEG",
            quality=self.config.jpeg_quality,
        )
        row = self._build_row(record, image_name=image_name)
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=DWA_CSV_COLUMNS).writerow(row)
        sample = {
            "frame_index": self.frame_count,
            "simulation_step": sim_step,
            "pipeline_state": record.pipeline_state,
            "action": list(full_action),
            "object_state": self._object_state(record),
            "tcp_pose": list(
                state.tcp_pose or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            ),
        }
        with self.samples_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

        report = {
            "frame_index": self.frame_count,
            "simulation_step": sim_step,
            "timestamp": row["时间戳(秒)"],
            "image": f"images/front/{image_name}",
        }
        self.frame_count += 1
        self._last_sampled_sim_step = sim_step
        self._next_sample_sim_step = sim_step + self.config.capture_every_n_steps
        return report

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
            self._last_gripper_target = sum(
                float(value) for value in gripper_positions
            ) / len(gripper_positions)
        elif record.action.gripper_command == "open":
            self._last_gripper_target = 0.04
        elif record.action.gripper_command == "close":
            self._last_gripper_target = 0.0
        elif self._last_gripper_target is None:
            self._last_gripper_target = gripper_actual

        return (
            *(float(value) for value in record.action.base_velocity),
            *self._last_arm_target,
            float(self._last_gripper_target),
        )

    @staticmethod
    def _object_state(record: StepRecord) -> list[float]:
        state = record.post_step_observation
        pose = state.object_pose or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        velocity = state.object_velocity or (0.0,) * 6
        return [*(float(value) for value in pose), *(float(value) for value in velocity)]

    def _build_row(self, record: StepRecord, *, image_name: str) -> dict[str, Any]:
        state = record.post_step_observation
        root_pose = state.robot_root_pose
        root_yaw = _quat_wxyz_to_rpy(tuple(root_pose[3:7]))[2]
        body_velocity = tuple(
            float(value) for value in state.metadata.get("body_linear_velocity", ())
        )
        if len(body_velocity) < 3:
            body_velocity = (
                float(state.robot_root_velocity[0]),
                float(state.robot_root_velocity[1]),
                float(state.robot_root_velocity[2]),
            )

        tcp_pose = state.tcp_pose or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        tcp_rpy = _quat_wxyz_to_rpy(tuple(tcp_pose[3:7]))
        arm, gripper = _joint_values(record)
        row = {
            "时间戳(秒)": f"{float(self.frame_count) / float(self.config.fps):.6f}",
            "位置X": f"{float(root_pose[0]):.6f}",
            "位置Y": f"{float(root_pose[1]):.6f}",
            "位置Z": f"{float(root_pose[2]):.6f}",
            "偏航角": f"{root_yaw:.6f}",
            "线速度X": f"{body_velocity[0]:.6f}",
            "线速度Y": f"{body_velocity[1]:.6f}",
            "线速度Z": f"{body_velocity[2]:.6f}",
            "末端X": f"{float(tcp_pose[0]):.6f}",
            "末端Y": f"{float(tcp_pose[1]):.6f}",
            "末端Z": f"{float(tcp_pose[2]):.6f}",
            "末端Roll": f"{tcp_rpy[0]:.6f}",
            "末端Pitch": f"{tcp_rpy[1]:.6f}",
            "末端Yaw": f"{tcp_rpy[2]:.6f}",
            "夹爪": f"{gripper:.6f}",
            "前摄像头图像": image_name,
        }
        for index, value in enumerate(arm, start=1):
            row[f"关节{index}"] = f"{value:.6f}"
        return row


def _read_instruction(task_path: Path) -> str:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    return str(task.get("instruction") or "Complete the navigation pick and place task.")


def _read_episode_rows(episode_dir: Path) -> list[dict[str, str]]:
    csv_path = episode_dir / "data.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DWA_CSV_COLUMNS:
            raise ValueError(f"{csv_path} has unexpected DWA columns")
        return list(reader)


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
    return samples


def discover_recorded_episodes(
    episodes_root: str | Path,
    *,
    require_success: bool = True,
) -> list[Path]:
    """发现 full-physics 输出中的原始 CSV/JPEG episode。"""

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
        episodes.append(episode_dir)
    return episodes


def _compute_stats(array: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
    }


def materialize_lerobot_dataset(
    episode_dirs: Iterable[str | Path],
    output_root: str | Path,
    *,
    fps: int = 5,
    chunks_size: int = 1000,
) -> dict[str, Any]:
    """将 DWA-compatible raw episodes 转成相同的 LeRobot v2.1 目录。"""

    episodes = [Path(path).expanduser().resolve() for path in episode_dirs]
    valid_episodes: list[
        tuple[Path, str, list[dict[str, str]], list[dict[str, Any]]]
    ] = []
    for episode_dir in episodes:
        task_path = episode_dir / "task.json"
        csv_path = episode_dir / "data.csv"
        if not task_path.is_file() or not csv_path.is_file():
            continue
        rows = _read_episode_rows(episode_dir)
        if rows:
            samples = _read_episode_samples(episode_dir)
            if len(samples) != len(rows):
                raise ValueError(
                    f"{episode_dir} data.csv/samples.jsonl length mismatch: "
                    f"{len(rows)} != {len(samples)}"
                )
            valid_episodes.append(
                (episode_dir, _read_instruction(task_path), rows, samples)
            )

    output_path = Path(output_root).expanduser().resolve()
    report: dict[str, Any] = {
        "lerobot_exported": False,
        "dataset_path": str(output_path),
        "episode_count": len(valid_episodes),
        "fps": int(fps),
        "chunks_size": int(chunks_size),
        "format": "lerobot_v2.1_dwa_compatible",
    }
    if not valid_episodes:
        return {**report, "reason": "no_recorded_episode_frames"}

    try:
        import cv2
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        return {
            **report,
            "reason": "missing_conversion_dependency",
            "missing_dependency": exc.name,
        }

    if output_path.exists():
        shutil.rmtree(output_path)
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    task_index_map: dict[str, int] = {}
    for _episode_dir, instruction, _rows, _samples in valid_episodes:
        task_index_map.setdefault(instruction, len(task_index_map))
    (meta_dir / "task_index_map.json").write_text(
        json.dumps(task_index_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    schema = pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field("task_index", pa.int64()),
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("observation.base_linear_velocity", pa.list_(pa.float32())),
            pa.field("observation.object_state", pa.list_(pa.float32())),
            pa.field("observation.tcp_pose", pa.list_(pa.float32())),
            pa.field("action", pa.list_(pa.float32())),
            pa.field("next.done", pa.bool_()),
        ]
    )
    all_states: list[np.ndarray] = []
    all_base_velocities: list[np.ndarray] = []
    all_object_states: list[np.ndarray] = []
    all_tcp_poses: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episodes_meta: list[dict[str, Any]] = []
    global_frame_index = 0
    image_height = 480
    image_width = 640

    for episode_index, (episode_dir, instruction, rows, samples) in enumerate(valid_episodes):
        chunk_index = episode_index // chunks_size
        states = np.asarray(
            [[float(row[column]) for column in STATE_COLUMNS] for row in rows],
            dtype=np.float32,
        )
        base_velocities = np.asarray(
            [[float(row[column]) for column in ACTION_COLUMNS] for row in rows],
            dtype=np.float32,
        )
        actions = np.asarray(
            [sample["action"] for sample in samples],
            dtype=np.float32,
        )
        object_states = np.asarray(
            [sample["object_state"] for sample in samples],
            dtype=np.float32,
        )
        tcp_poses = np.asarray(
            [sample["tcp_pose"] for sample in samples],
            dtype=np.float32,
        )
        timestamps = [float(row["时间戳(秒)"]) for row in rows]
        frame_count = len(rows)
        table = pa.table(
            {
                "index": pa.array(
                    range(global_frame_index, global_frame_index + frame_count),
                    type=pa.int64(),
                ),
                "episode_index": pa.array([episode_index] * frame_count, type=pa.int64()),
                "frame_index": pa.array(range(frame_count), type=pa.int64()),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "task_index": pa.array(
                    [task_index_map[instruction]] * frame_count,
                    type=pa.int64(),
                ),
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
                "observation.base_linear_velocity": pa.array(
                    base_velocities.tolist(),
                    type=pa.list_(pa.float32()),
                ),
                "observation.object_state": pa.array(
                    object_states.tolist(),
                    type=pa.list_(pa.float32()),
                ),
                "observation.tcp_pose": pa.array(
                    tcp_poses.tolist(),
                    type=pa.list_(pa.float32()),
                ),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
                "next.done": pa.array(
                    [False] * (frame_count - 1) + [True],
                    type=pa.bool_(),
                ),
            },
            schema=schema,
        )
        data_dir = output_path / "data" / f"chunk-{chunk_index:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, parquet_path)

        image_paths = [episode_dir / "images" / "front" / row["前摄像头图像"] for row in rows]
        first_image = cv2.imread(str(image_paths[0]))
        if first_image is None:
            raise RuntimeError(f"failed to read episode image: {image_paths[0]}")
        image_height, image_width = first_image.shape[:2]
        video_path = (
            output_path
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / "observation.images.front"
            / f"episode_{episode_index:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
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

        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": frame_count,
            }
        )
        all_states.append(states)
        all_base_velocities.append(base_velocities)
        all_object_states.append(object_states)
        all_tcp_poses.append(tcp_poses)
        all_actions.append(actions)
        global_frame_index += frame_count

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for payload in episodes_meta:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task, task_index in sorted(task_index_map.items(), key=lambda item: item[1]):
            stream.write(
                json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n"
            )

    states_array = np.concatenate(all_states, axis=0)
    base_velocities_array = np.concatenate(all_base_velocities, axis=0)
    object_states_array = np.concatenate(all_object_states, axis=0)
    tcp_poses_array = np.concatenate(all_tcp_poses, axis=0)
    actions_array = np.concatenate(all_actions, axis=0)
    stats = {
        "observation.state": _compute_stats(states_array),
        "observation.base_linear_velocity": _compute_stats(base_velocities_array),
        "observation.object_state": _compute_stats(object_states_array),
        "observation.tcp_pose": _compute_stats(tcp_poses_array),
        "action": _compute_stats(actions_array),
        "observation.images.front": {
            "min": [[[0.0, 0.0, 0.0]]],
            "max": [[[1.0, 1.0, 1.0]]],
            "mean": [[[0.5, 0.5, 0.5]]],
            "std": [[[0.5, 0.5, 0.5]]],
        },
    }
    with (meta_dir / "stats.jsonl").open("w", encoding="utf-8") as stream:
        for key, value in stats.items():
            stream.write(json.dumps({key: value}, ensure_ascii=False) + "\n")

    total_episodes = len(valid_episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "mobile_arm",
        "total_episodes": total_episodes,
        "total_frames": global_frame_index,
        "total_tasks": len(task_index_map),
        "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "fps": fps,
        "video": True,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(STATE_NAMES)],
                "names": list(STATE_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": [len(FULL_ACTION_NAMES)],
                "names": list(FULL_ACTION_NAMES),
            },
            "observation.base_linear_velocity": {
                "dtype": "float32",
                "shape": [len(ACTION_NAMES)],
                "names": list(ACTION_NAMES),
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
            "observation.images.front": {
                "dtype": "video",
                "shape": [image_height, image_width, 3],
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        **report,
        "lerobot_exported": True,
        "frame_count": global_frame_index,
        "task_count": len(task_index_map),
        "info_path": str(meta_dir / "info.json"),
    }


__all__ = [
    "ACTION_COLUMNS",
    "DWA_CSV_COLUMNS",
    "DwaEpisodeWriter",
    "LeRobotRecordingConfig",
    "STATE_COLUMNS",
    "discover_recorded_episodes",
    "materialize_lerobot_dataset",
]
