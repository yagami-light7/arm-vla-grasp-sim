#!/usr/bin/env python3
"""Convert one LeRobot v2 episode into a Rerun .rrd recording.

该脚本刻意不依赖 Isaac/Omni/PXR，只用于普通 Python 环境中的数据检查。
本项目导出的 LeRobot 数据统一按 v2/v2.1 目录结构读取。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


def _import_dependencies() -> dict[str, Any]:
    """集中导入依赖，给缺包时的报错留出清晰入口。"""

    try:
        import rerun as rr
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import numpy as np
        import pandas as pd
        import pyarrow  # noqa: F401
        import torch
        from PIL import Image
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise SystemExit(
            f"依赖导入失败：{exc}\n"
            f"请在 lerobot_rerun 环境中安装缺失包：{missing}"
        ) from exc
    return {
        "rr": rr,
        "LeRobotDataset": LeRobotDataset,
        "np": np,
        "pd": pd,
        "torch": torch,
        "Image": Image,
    }


_DEPS = _import_dependencies()
rr = _DEPS["rr"]
LeRobotDataset = _DEPS["LeRobotDataset"]
np = _DEPS["np"]
pd = _DEPS["pd"]
torch = _DEPS["torch"]
Image = _DEPS["Image"]


IMAGE_KEY_CANDIDATES = {
    "observation.image",
    "observation.images.front",
    "observation.images.wrist",
    "observation.images.side",
    "image",
}

EE_KEYS = (
    "observation.ee_position",
    "observation.eef_position",
    "observation.ee_pose",
    "observation.eef_pose",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将指定 LeRobot episode 转换为 Rerun .rrd 可视化文件。",
    )
    parser.add_argument("--repo-id", required=True, help="LeRobot 数据集 repo_id 或名称。")
    parser.add_argument(
        "--root",
        help="本地 LeRobot 数据集根目录；传入后优先从本地 root 加载。",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="要转换的 episode 编号，默认 0。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="最多转换帧数；-1 表示转换完整 episode。",
    )
    parser.add_argument("--out", default="episode.rrd", help="输出 .rrd 文件路径。")
    parser.add_argument("--spawn", action="store_true", help="转换时直接打开 Rerun Viewer。")
    return parser


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, Image.Image):
        return np.asarray(value)
    return value


def _to_scalar(value: Any) -> int | float | str | bool | None:
    value = _to_numpy(value)
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return _to_scalar(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _to_scalar(value[0])
    if isinstance(value, (int, float, str, bool)):
        return value
    return None


def _to_vector(value: Any) -> np.ndarray | None:
    value = _to_numpy(value)
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        scalar = _to_scalar(value)
        if scalar is None or isinstance(scalar, str):
            return None
        array = np.asarray([scalar])
    if array.dtype.kind in {"U", "S", "O"}:
        try:
            array = array.astype(np.float64)
        except (TypeError, ValueError):
            return None
    return np.asarray(array, dtype=np.float64).reshape(-1)


def normalize_image(img: Any) -> np.ndarray:
    """把 Tensor/ndarray/PIL 图像统一为 uint8 HWC。"""

    array = _to_numpy(img)
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if array.ndim == 2:
        array = array[:, :, None]
    elif array.ndim == 3:
        # 兼容 CHW；优先在首维像通道且末维不像通道时转 HWC。
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"unsupported image ndim={array.ndim}")
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unsupported image shape={array.shape}")

    if array.dtype == np.uint8:
        out = array
    elif array.dtype.kind == "f":
        finite = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.nanmax(finite)) <= 1.5 and float(np.nanmin(finite)) >= -0.01:
            finite = finite * 255.0
        out = np.clip(finite, 0, 255).astype(np.uint8)
    elif array.dtype == np.bool_:
        out = array.astype(np.uint8) * 255
    else:
        out = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def _safe_name(name: Any) -> str:
    text = str(name)
    text = text.replace("/", "_").replace("\\", "_").replace(".", "_")
    text = re.sub(r"[^0-9A-Za-z_\-]+", "_", text)
    return text.strip("_") or "value"


def _log_scalar(path: str, value: Any) -> None:
    scalar = _to_scalar(value)
    if scalar is None:
        return
    if isinstance(scalar, str):
        if hasattr(rr, "TextLog"):
            rr.log(path, rr.TextLog(scalar))
        return
    if hasattr(rr, "Scalar"):
        rr.log(path, rr.Scalar(float(scalar)))
    else:
        rr.log(path, rr.Scalars([float(scalar)]))


def log_vector(prefix: str, vec: Any, names: Sequence[str] | None = None) -> None:
    """逐维记录向量；无 names 时使用 dim_i。"""

    array = _to_vector(vec)
    if array is None:
        return
    name_list = list(names or [])
    for index, value in enumerate(array):
        dim_name = name_list[index] if index < len(name_list) and name_list[index] else f"dim_{index}"
        _log_scalar(f"{prefix}/{_safe_name(dim_name)}", float(value))


def _value_summary(value: Any) -> str:
    value = _to_numpy(value)
    if isinstance(value, np.ndarray):
        return f"shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, (list, tuple)):
        return f"type={type(value).__name__} len={len(value)}"
    return f"type={type(value).__name__}"


def _feature_names(dataset: Any, key: str) -> list[str] | None:
    meta = getattr(dataset, "meta", None)
    info = getattr(meta, "info", None)
    if isinstance(info, dict):
        shortcut_key = {
            "observation.state": "observation_state_names",
            "action": "action_names",
            "observation.object_state": "object_state_names",
            "observation.tcp_pose": "tcp_pose_names",
            "observation.base_velocity": "base_velocity_names",
        }.get(key)
        if shortcut_key and isinstance(info.get(shortcut_key), list):
            return [str(name) for name in info[shortcut_key]]

    features = getattr(meta, "features", None)
    if isinstance(features, dict) and key in features:
        feature = features[key]
        names = None
        if isinstance(feature, dict):
            names = feature.get("names")
        else:
            names = getattr(feature, "names", None)
        if isinstance(names, list):
            return [str(name) for name in names]
    return None


def _camera_name_from_key(key: str) -> str:
    if key.startswith("observation.images."):
        return _safe_name(key.split("observation.images.", 1)[1])
    if key == "observation.image":
        return "observation"
    if key == "image":
        return "image"
    return _safe_name(key.split(".")[-1])


def _looks_like_image(value: Any) -> bool:
    try:
        array = normalize_image(value)
    except Exception:
        return False
    return array.ndim == 3 and array.shape[-1] in (1, 3, 4)


def detect_image_keys(dataset: Any, sample: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    meta = getattr(dataset, "meta", None)
    for key in getattr(meta, "camera_keys", []) or []:
        if key in sample and key not in keys:
            keys.append(str(key))
    for key, value in sample.items():
        if key in keys:
            continue
        if key in IMAGE_KEY_CANDIDATES or key.startswith("observation.images."):
            if _looks_like_image(value):
                keys.append(str(key))
        elif "image" in key.lower() and _looks_like_image(value):
            keys.append(str(key))
    return keys


def _sequence_from_index(index_data: Any, episode_index: int) -> list[int] | None:
    """兼容 dict/DataFrame/list 等不同 episode_data_index 表达。"""

    def scalar_at(data: Any, idx: int) -> int | None:
        if isinstance(data, dict):
            value = data.get(idx, data.get(str(idx)))
            return None if value is None else int(_to_scalar(value))
        try:
            return int(_to_scalar(data[idx]))
        except Exception:
            return None

    if index_data is None:
        return None
    if isinstance(index_data, dict):
        start_data = index_data.get("from", index_data.get("start", index_data.get("starts")))
        end_data = index_data.get("to", index_data.get("end", index_data.get("ends")))
        if start_data is not None and end_data is not None:
            start = scalar_at(start_data, episode_index)
            end = scalar_at(end_data, episode_index)
            if start is not None and end is not None and end > start:
                return list(range(start, end))
    if isinstance(index_data, pd.DataFrame):
        row = None
        if "episode_index" in index_data.columns:
            matches = index_data[index_data["episode_index"] == episode_index]
            if len(matches) > 0:
                row = matches.iloc[0]
        elif episode_index in index_data.index:
            row = index_data.loc[episode_index]
        if row is not None:
            start = row.get("from", row.get("start"))
            end = row.get("to", row.get("end"))
            if start is not None and end is not None and int(end) > int(start):
                return list(range(int(start), int(end)))
    if isinstance(index_data, (list, tuple)) and 0 <= episode_index < len(index_data):
        row = index_data[episode_index]
        if isinstance(row, dict):
            start = row.get("from", row.get("start"))
            end = row.get("to", row.get("end"))
            if start is not None and end is not None and int(end) > int(start):
                return list(range(int(start), int(end)))
    return None


def get_episode_indices(dataset: Any, episode_index: int) -> list[int]:
    """返回指定 episode 对应的数据集全局 index 列表。"""

    for holder in (dataset, getattr(dataset, "meta", None)):
        indices = _sequence_from_index(getattr(holder, "episode_data_index", None), episode_index)
        if indices:
            return indices

    indices: list[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_episode = _to_scalar(sample.get("episode_index"))
        if sample_episode is not None and int(sample_episode) == int(episode_index):
            indices.append(index)
    if not indices:
        raise RuntimeError(
            f"Cannot find episode_index = {episode_index}. Please check dataset metadata."
        )
    return indices


class LocalLeRobotV2Dataset:
    """只读 LeRobot v2/v2.1 数据集，匹配当前项目的统一导出格式。"""

    def __init__(self, repo_id: str, root: str | Path):
        self.repo_id = repo_id
        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"missing LeRobot v2 info.json: {info_path}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        data_files = sorted((self.root / "data").glob("chunk-*/episode_*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"no parquet files under {self.root / 'data'}")
        frames: list[pd.DataFrame] = []
        for data_file in data_files:
            frame = pd.read_parquet(data_file)
            frame["__parquet_path"] = str(data_file)
            frames.append(frame)
        self.data = pd.concat(frames, ignore_index=True)
        self.video_keys = [
            key
            for key, feature in (self.info.get("features") or {}).items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]
        self.meta = SimpleNamespace(
            features=self.info.get("features") or {},
            camera_keys=list(self.video_keys),
            info=self.info,
        )
        self.episode_data_index = self._build_episode_data_index()

    def _build_episode_data_index(self) -> dict[str, dict[int, int]]:
        starts: dict[int, int] = {}
        ends: dict[int, int] = {}
        for global_index, episode_value in enumerate(self.data["episode_index"].tolist()):
            episode = int(episode_value)
            starts.setdefault(episode, global_index)
            ends[episode] = global_index + 1
        return {"from": starts, "to": ends}

    def __len__(self) -> int:
        return int(len(self.data))

    def _video_path(self, video_key: str, episode_index: int) -> Path:
        chunks_size = int(self.info.get("chunks_size", 1000))
        template = str(
            self.info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        )
        return self.root / template.format(
            episode_chunk=int(episode_index) // chunks_size,
            video_key=video_key,
            episode_index=int(episode_index),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data.iloc[int(idx)].to_dict()
        episode_index = int(_to_scalar(row.get("episode_index")) or 0)
        frame_index = int(_to_scalar(row.get("frame_index")) or 0)
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("读取 v2.1 本地视频需要安装 imageio/imageio-ffmpeg") from exc
        for video_key in self.video_keys:
            path = self._video_path(video_key, episode_index)
            if not path.is_file():
                continue
            try:
                row[video_key] = iio.imread(path, index=frame_index)
            except Exception as exc:
                print(
                    f"[warning] failed to read video frame key={video_key} "
                    f"episode={episode_index} frame={frame_index}: {exc}",
                    file=sys.stderr,
                )
        return row


def _is_local_v2_dataset(root: Path | None) -> bool:
    if root is None:
        return False
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    version = str(info.get("codebase_version") or info.get("schema_version") or "")
    return (
        "v2" in version
        or "v2.1" in version
        or "lerobot_v2" in version
        or "lerobot_v2.1" in version
    )


def load_dataset(repo_id: str, root: str | Path | None) -> tuple[Any, str]:
    root_path = Path(root).expanduser().resolve() if root else None
    if root_path is not None and _is_local_v2_dataset(root_path):
        # 本项目统一导出 LeRobot v2/v2.1。当前 lerobot 包可能是 v3，
        # 直接调用 LeRobotDataset 会提示转换格式；这里按项目格式只读加载。
        return LocalLeRobotV2Dataset(repo_id, root_path), "LocalLeRobotV2Dataset"
    try:
        if root_path is None:
            return LeRobotDataset(repo_id), "LeRobotDataset"
        return LeRobotDataset(repo_id, root=root_path), "LeRobotDataset"
    except Exception as exc:
        raise RuntimeError(f"Failed to load LeRobot dataset repo_id={repo_id!r}: {exc}") from exc


def _print_sample_overview(sample: dict[str, Any], image_keys: Sequence[str]) -> None:
    print("First sample keys:")
    for key in sorted(sample):
        print(f"  {key}: {_value_summary(sample[key])}")
    print("Detected image keys:")
    for key in image_keys:
        print(f"  {key}")
    print(f"Has observation.state: {'observation.state' in sample}")
    print(f"Has action: {'action' in sample}")


def _log_optional_vector(dataset: Any, sample: dict[str, Any], key: str, prefix: str) -> None:
    if key not in sample:
        return
    log_vector(prefix, sample[key], _feature_names(dataset, key))


def _log_ee(sample: dict[str, Any]) -> None:
    for key in EE_KEYS:
        if key not in sample:
            continue
        vec = _to_vector(sample[key])
        if vec is None or len(vec) < 3:
            continue
        translation = [float(vec[0]), float(vec[1]), float(vec[2])]
        rr.log("robot/ee", rr.Transform3D(translation=translation))
        rr.log("robot/ee/point", rr.Points3D([translation]))
        # 若 vec 还包含 quaternion，需要先确认数据集使用 xyzw 还是 wxyz。
        return


def convert_episode(
    *,
    dataset: Any,
    episode_index: int,
    max_frames: int,
    out_path: Path,
    spawn: bool,
) -> int:
    indices = get_episode_indices(dataset, episode_index)
    if max_frames is not None and max_frames >= 0:
        indices = indices[: int(max_frames)]
    if not indices:
        raise RuntimeError(
            f"Cannot find episode_index = {episode_index}. Please check dataset metadata."
        )

    first_sample = dataset[indices[0]]
    image_keys = detect_image_keys(dataset, first_sample)

    print(f"Episode index: {episode_index}")
    print(f"Episode frames: {len(get_episode_indices(dataset, episode_index))}")
    print(f"Actual converted frames: {len(indices)}")
    _print_sample_overview(first_sample, image_keys)
    print(f"Output Rerun file: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("lerobot_to_rerun", spawn=spawn)
    rr.save(out_path)

    for local_frame_index, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        timestamp = _to_scalar(sample.get("timestamp"))
        if timestamp is not None and not isinstance(timestamp, str):
            rr.set_time("time", duration=float(timestamp))
        rr.set_time("frame", sequence=int(local_frame_index))

        _log_scalar("meta/episode_index", episode_index)
        _log_scalar("meta/frame_index", local_frame_index)
        _log_scalar("meta/dataset_index", int(dataset_index))

        sample_frame = _to_scalar(sample.get("frame_index"))
        if sample_frame is not None:
            _log_scalar("meta/source_frame_index", sample_frame)
        if "pipeline_state" in sample and hasattr(rr, "TextLog"):
            state = _to_scalar(sample.get("pipeline_state"))
            if state is not None:
                rr.log("meta/pipeline_state", rr.TextLog(str(state)))

        for key in image_keys:
            if key not in sample:
                continue
            try:
                image = normalize_image(sample[key])
            except Exception as exc:
                print(
                    f"[warning] skip image key={key} frame={local_frame_index}: {exc}",
                    file=sys.stderr,
                )
                continue
            camera_name = _camera_name_from_key(key)
            rr.log(f"cameras/{camera_name}/image", rr.Image(image))

        _log_optional_vector(dataset, sample, "observation.state", "observation/state")
        _log_optional_vector(dataset, sample, "action", "action")
        _log_optional_vector(dataset, sample, "observation.base_velocity", "observation/base_velocity")
        _log_optional_vector(dataset, sample, "observation.object_state", "observation/object_state")
        _log_optional_vector(dataset, sample, "observation.tcp_pose", "observation/tcp_pose")
        _log_ee(sample)

    if hasattr(rr, "flush"):
        rr.flush()
    elif hasattr(rr, "disconnect"):
        rr.disconnect()
    print(f"Saved Rerun recording to: {out_path}")
    return len(indices)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_frames < -1:
        raise SystemExit("--max-frames must be -1 or non-negative.")
    dataset, backend = load_dataset(args.repo_id, args.root)
    print(f"Loaded dataset: {args.repo_id} backend={backend}")
    if args.root:
        print(f"Dataset root: {Path(args.root).expanduser().resolve()}")
    print(f"Dataset length: {len(dataset)}")
    convert_episode(
        dataset=dataset,
        episode_index=int(args.episode_index),
        max_frames=int(args.max_frames),
        out_path=Path(args.out).expanduser().resolve(),
        spawn=bool(args.spawn),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
