#!/usr/bin/env python3

"""使用 gsplat 离线渲染 multifloor 3DGS 场景。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from source.scene.gaussian_splat_renderer import (
    CameraConfig,
    GaussianSplatLoadConfig,
    RenderConfig,
    load_gaussian_splats,
    render_gaussian_splats,
    save_rgb_image,
    write_render_report,
)


DEFAULT_SCENE_DIR = REPO_ROOT / "source" / "scene" / "multifloor"
DEFAULT_VISUAL_PLY = DEFAULT_SCENE_DIR / "ply" / "3dgs_visual.ply"
DEFAULT_OUTPUT_IMAGE = REPO_ROOT / "outputs" / "multifloor_3dgs" / "preview.png"
DEFAULT_REPORT_JSON = REPO_ROOT / "outputs" / "multifloor_3dgs" / "preview_report.json"
DEFAULT_FRAME_OUTPUT_DIR = REPO_ROOT / "outputs" / "multifloor_3dgs" / "frames"


def _project_path(raw_path: str | Path) -> Path:
    """把相对路径按仓库根目录解析。"""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _vec3(values: list[float] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _clip_percentiles(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    return (float(values[0]), float(values[1]))


def _clip_bounds(values: list[float] | None) -> tuple[float, float, float, float, float, float] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _env_hints() -> dict[str, str]:
    """记录运行 gsplat 时常见的环境变量，便于复现。"""

    keys = [
        "CUDA_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TORCH_CUDA_ARCH_LIST",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ]
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def _read_trajectory_records(path: Path) -> list[dict[str, Any]]:
    """读取 overview camera trajectory JSONL。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"相机轨迹 JSONL 第 {line_number} 行不是有效 JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"相机轨迹 JSONL 第 {line_number} 行必须是 JSON object")
            records.append(payload)
    if not records:
        raise ValueError(f"相机轨迹为空: {path}")
    return records


def _record_int(record: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(record.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _record_float(record: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(record.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _camera_from_trajectory_record(
    record: dict[str, Any],
    args: argparse.Namespace,
) -> CameraConfig:
    """将 recorder 写出的相机记录转换为渲染相机。"""

    camera_payload = record.get("camera")
    if not isinstance(camera_payload, dict) or not bool(camera_payload.get("available")):
        reason = camera_payload.get("reason") if isinstance(camera_payload, dict) else "camera_payload_missing"
        raise ValueError(f"轨迹帧缺少可用 camera: {reason}")
    video_payload = record.get("video")
    if not isinstance(video_payload, dict):
        video_payload = {}
    if args.trajectory_resolution == "json":
        width = _record_int(video_payload, "width", int(args.width))
        height = _record_int(video_payload, "height", int(args.height))
    else:
        width = int(args.width)
        height = int(args.height)
    return CameraConfig(
        eye=_vec3(camera_payload.get("eye")),
        target=_vec3(camera_payload.get("target")),
        up=_vec3(camera_payload.get("up")) or _vec3(args.camera_up) or (0.0, 0.0, 1.0),
        vertical_fov_deg=_record_float(camera_payload, "vertical_fov_deg", float(args.vertical_fov_deg)),
        width=width,
        height=height,
        near_plane=_record_float(camera_payload, "near_plane", float(args.near_plane)),
        far_plane=_record_float(camera_payload, "far_plane", float(args.far_plane)),
    )


def _trajectory_frame_name(template: str, record: dict[str, Any], sequence_index: int) -> str:
    """按轨迹 frame metadata 渲染输出文件名。"""

    frame_index = _record_int(record, "frame_index", sequence_index)
    step_index = _record_int(record, "step_index", -1)
    timestamp = _record_float(record, "timestamp", 0.0)
    return template.format(
        frame_index=frame_index,
        sequence_index=sequence_index,
        step_index=step_index,
        timestamp_ms=int(round(timestamp * 1000.0)),
    )


def _render_single_image(
    *,
    args: argparse.Namespace,
    splats: Any,
    render_config: RenderConfig,
    output_image: Path,
) -> dict[str, Any]:
    """保持原来的单张 PNG 渲染路径。"""

    camera = CameraConfig(
        eye=_vec3(args.camera_eye),
        target=_vec3(args.camera_target),
        up=_vec3(args.camera_up) or (0.0, 0.0, 1.0),
        vertical_fov_deg=float(args.vertical_fov_deg),
        width=int(args.width),
        height=int(args.height),
        near_plane=float(args.near_plane),
        far_plane=float(args.far_plane),
    )
    rgb, render_report = render_gaussian_splats(splats, camera, render_config)
    image_path = save_rgb_image(output_image, rgb)
    return {
        "mode": "single_image",
        "image_output": str(image_path),
        "render": render_report,
    }


def _render_trajectory_frames(
    *,
    args: argparse.Namespace,
    splats: Any,
    render_config: RenderConfig,
    trajectory_path: Path,
) -> dict[str, Any]:
    """按照 Isaac overview 相机轨迹批量渲染 3DGS 背景帧。"""

    records = _read_trajectory_records(trajectory_path)
    frame_output_dir = _project_path(args.frame_output_dir)
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    frame_stride = max(1, int(args.trajectory_frame_stride))
    max_frames = max(0, int(args.max_trajectory_frames))
    frames: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sequence_index = 0
    for record_index, record in enumerate(records):
        if record_index % frame_stride != 0:
            continue
        if max_frames > 0 and len(frames) >= max_frames:
            break
        try:
            camera = _camera_from_trajectory_record(record, args)
        except Exception as exc:
            skipped.append(
                {
                    "record_index": int(record_index),
                    "frame_index": record.get("frame_index"),
                    "reason": str(exc),
                }
            )
            continue
        frame_name = _trajectory_frame_name(args.frame_name_template, record, sequence_index)
        output_path = frame_output_dir / frame_name
        rgb, render_report = render_gaussian_splats(splats, camera, render_config)
        image_path = save_rgb_image(output_path, rgb)
        frames.append(
            {
                "sequence_index": int(sequence_index),
                "record_index": int(record_index),
                "frame_index": record.get("frame_index"),
                "step_index": record.get("step_index"),
                "timestamp": record.get("timestamp"),
                "pipeline_state": record.get("pipeline_state"),
                "image_output": str(image_path),
                "camera_prim_path": record.get("camera_prim_path"),
                "render_camera_prim_path": record.get("render_camera_prim_path"),
                "render": render_report,
            }
        )
        sequence_index += 1
    if not frames:
        raise ValueError("相机轨迹中没有成功渲染任何 3DGS 背景帧")
    return {
        "mode": "camera_trajectory",
        "trajectory_path": str(trajectory_path),
        "frame_output_dir": str(frame_output_dir),
        "input_record_count": len(records),
        "rendered_frame_count": len(frames),
        "skipped_frame_count": len(skipped),
        "trajectory_frame_stride": int(frame_stride),
        "max_trajectory_frames": int(max_frames),
        "trajectory_resolution": str(args.trajectory_resolution),
        "frames": frames,
        "skipped": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 gsplat 渲染 multifloor Gaussian splat PLY，输出 PNG 和 JSON 报告。",
    )
    parser.add_argument("--visual-ply", default=os.fspath(DEFAULT_VISUAL_PLY), help="Gaussian visual PLY 输入路径。")
    parser.add_argument("--output-image", default=os.fspath(DEFAULT_OUTPUT_IMAGE), help="输出 PNG 路径。")
    parser.add_argument("--report-output", default=os.fspath(DEFAULT_REPORT_JSON), help="输出 JSON 报告路径。")
    parser.add_argument("--camera-trajectory-jsonl", help="由 overview recorder 导出的相机轨迹 JSONL；传入后批量渲染背景帧。")
    parser.add_argument("--frame-output-dir", default=os.fspath(DEFAULT_FRAME_OUTPUT_DIR), help="批量背景帧输出目录。")
    parser.add_argument("--frame-name-template", default="frame_{frame_index:06d}.png", help="批量输出文件名模板，可用 frame_index/sequence_index/step_index/timestamp_ms。")
    parser.add_argument("--trajectory-frame-stride", type=int, default=1, help="批量渲染时每隔多少条轨迹记录取一帧。")
    parser.add_argument("--max-trajectory-frames", type=int, default=0, help="批量渲染最多输出多少帧；0 表示不限制。")
    parser.add_argument("--trajectory-resolution", choices=("json", "cli"), default="json", help="批量渲染分辨率来源；json 使用录制轨迹中的 video width/height。")
    parser.add_argument("--max-gaussians", type=int, default=2_000_000, help="最多载入 Gaussian 数；传 0 表示尝试全量载入。")
    parser.add_argument("--sample-stride", type=int, default=1, help="固定采样 stride，会与 max-gaussians 自动 stride 取较大值。")
    parser.add_argument("--chunk-gaussians", type=int, default=500_000, help="分块读取 PLY 的 Gaussian 数。")
    parser.add_argument(
        "--clip-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="显式按 PLY 坐标裁剪离群点。",
    )
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        default=(0.1, 99.9),
        metavar=("LOW", "HIGH"),
        help="按 PLY 坐标百分位估计裁剪范围；默认 0.1 99.9。",
    )
    parser.add_argument("--clip-estimate-stride", type=int, default=100, help="估计百分位裁剪范围时的抽样 stride。")
    parser.add_argument("--coord-mode", choices=("sim_to_pct_180deg", "ply"), default="sim_to_pct_180deg", help="渲染坐标模式。")
    parser.add_argument("--scale-multiplier", type=float, default=1.0, help="整体放大或缩小 Gaussian scale。")
    parser.add_argument("--opacity-multiplier", type=float, default=1.0, help="整体调整 opacity。")
    parser.add_argument("--opacity-threshold", type=float, default=0.0, help="过滤低于该 opacity 的 Gaussian。")
    parser.add_argument("--width", type=int, default=960, help="输出宽度。")
    parser.add_argument("--height", type=int, default=540, help="输出高度。")
    parser.add_argument("--vertical-fov-deg", type=float, default=58.0, help="垂直视场角。")
    parser.add_argument("--camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"), help="相机位置；默认根据场景 bounds 自动生成。")
    parser.add_argument("--camera-target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="相机目标点；默认根据场景 bounds 自动生成。")
    parser.add_argument("--camera-up", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("X", "Y", "Z"), help="相机 up 向量。")
    parser.add_argument("--near-plane", type=float, default=0.01, help="近裁剪面。")
    parser.add_argument("--far-plane", type=float, default=100000.0, help="远裁剪面。")
    parser.add_argument("--radius-clip", type=float, default=0.2, help="跳过屏幕半径过小的 Gaussian，用于大场景加速。")
    parser.add_argument("--eps2d", type=float, default=0.3, help="gsplat 2D covariance 稳定项。")
    parser.add_argument("--rasterize-mode", choices=("classic", "antialiased"), default="antialiased", help="gsplat rasterize 模式。")
    parser.add_argument("--background-rgb", type=float, nargs=3, default=(0.02, 0.025, 0.03), metavar=("R", "G", "B"), help="背景颜色，范围 0..1。")
    parser.add_argument("--skip-report", action="store_true", help="不写 JSON 报告。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    visual_ply = _project_path(args.visual_ply)
    output_image = _project_path(args.output_image)
    report_output = _project_path(args.report_output)
    trajectory_path = _project_path(args.camera_trajectory_jsonl) if args.camera_trajectory_jsonl else None
    clip_percentiles = None if args.clip_bounds is not None else _clip_percentiles(args.clip_percentiles)

    load_config = GaussianSplatLoadConfig(
        ply_path=visual_ply,
        max_gaussians=int(args.max_gaussians),
        sample_stride=int(args.sample_stride),
        chunk_gaussians=int(args.chunk_gaussians),
        clip_bounds=_clip_bounds(args.clip_bounds),
        clip_percentiles=clip_percentiles,
        clip_estimate_stride=int(args.clip_estimate_stride),
        coord_mode=args.coord_mode,
        scale_multiplier=float(args.scale_multiplier),
        opacity_multiplier=float(args.opacity_multiplier),
        opacity_threshold=float(args.opacity_threshold),
    )
    render_config = RenderConfig(
        radius_clip=float(args.radius_clip),
        eps2d=float(args.eps2d),
        rasterize_mode=args.rasterize_mode,
        background_rgb=_vec3(args.background_rgb) or (0.02, 0.025, 0.03),
    )

    splats = load_gaussian_splats(load_config)
    if trajectory_path is None:
        render_payload = _render_single_image(
            args=args,
            splats=splats,
            render_config=render_config,
            output_image=output_image,
        )
    else:
        render_payload = _render_trajectory_frames(
            args=args,
            splats=splats,
            render_config=render_config,
            trajectory_path=trajectory_path,
        )

    payload: dict[str, Any] = {
        "status": "passed",
        "script": "scripts/scene/render_yinluyuan_3dgs.py",
        "splat_load": splats.report,
        "output": render_payload,
        "environment": _env_hints(),
        "notes": [
            "该脚本是离线 3DGS 渲染验证入口，不会修改 Isaac Sim USD stage。",
            "默认 coord-mode 会把 PLY/PCT 坐标转换到 Isaac Sim 世界坐标：sim_x=-ply_x，sim_y=-ply_y，sim_z=ply_z。",
            "8GB 显存机器上建议先使用 100万到400万个 Gaussian 预览，最终视频可在更大显存或离线批处理环境中提高 max-gaussians。",
        ],
    }
    if not args.skip_report:
        report_path = write_render_report(report_output, payload)
        payload["report_output"] = str(report_path)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
