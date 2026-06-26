#!/usr/bin/env python3

"""从大型 Gaussian PLY 中抽样生成小型 NuRec 诊断 PLY。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from source.scene.gaussian_splat_ply import is_gaussian_splat_ply, parse_ply_header


DEFAULT_INPUT_PLY = PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_visual.ply"
DEFAULT_OUTPUT_PLY = PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_visual_debug_100k.ply"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/multifloor_nurec_debug_100k_ply.json"


def main() -> int:
    args = _parse_args()
    input_ply = _project_path(args.input_ply)
    output_ply = _project_path(args.output_ply)
    report_path = _project_path(args.report_output)

    if output_ply.exists() and not args.force:
        raise FileExistsError(f"输出已存在，覆盖请传 --force: {output_ply}")

    payload = _sample_gaussian_ply(
        input_ply=input_ply,
        output_ply=output_ply,
        max_points=args.max_points,
        sample_stride=args.sample_stride,
        chunk_points=args.chunk_points,
        clip_bounds=_clip_bounds_from_args(args),
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
    )
    payload["report_output"] = str(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _sample_gaussian_ply(
    *,
    input_ply: Path,
    output_ply: Path,
    max_points: int,
    sample_stride: int,
    chunk_points: int,
    clip_bounds: tuple[np.ndarray, np.ndarray] | None,
    min_opacity: float | None,
    max_scale: float | None,
) -> dict[str, Any]:
    header = parse_ply_header(input_ply)
    if header.format_name != "binary_little_endian":
        raise ValueError(f"当前只支持 binary_little_endian PLY: {input_ply}")
    if not header.all_vertex_properties_float32:
        raise ValueError("当前采样器只支持全部 vertex property 为 float32 的 Gaussian PLY")
    if not is_gaussian_splat_ply(header):
        raise ValueError("输入 PLY 缺少 3DGS 必需属性")
    if header.has_faces:
        raise ValueError("Gaussian visual PLY 不应包含 face")
    if max_points < 0:
        raise ValueError("--max-points 不能小于 0")
    if sample_stride <= 0:
        raise ValueError("--sample-stride 必须为正数")
    if chunk_points <= 0:
        raise ValueError("--chunk-points 必须为正数")
    if min_opacity is not None and not (0.0 <= min_opacity < 1.0):
        raise ValueError("--min-opacity 必须满足 0 <= value < 1")
    if max_scale is not None and max_scale <= 0.0:
        raise ValueError("--max-scale 必须为正数")

    stride = int(sample_stride)
    if max_points > 0:
        stride = max(stride, math.ceil(header.vertex_count / int(max_points)))
    vertices = np.memmap(
        input_ply,
        dtype="<f4",
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count, len(header.vertex_properties)),
    )

    sampled_count = _count_filtered_vertices(
        vertices=vertices,
        vertex_count=header.vertex_count,
        stride=stride,
        chunk_points=chunk_points,
        max_points=max_points,
        clip_bounds=clip_bounds,
        property_names=header.vertex_property_names,
        min_opacity=min_opacity,
        max_scale=max_scale,
    )
    if sampled_count <= 0:
        raise RuntimeError("采样和裁剪后没有可输出的 Gaussian")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with output_ply.open("wb") as stream:
        _write_ply_header(stream, header.vertex_property_names, sampled_count)
        emitted = 0
        for chunk in _iter_filtered_vertices(
            vertices=vertices,
            vertex_count=header.vertex_count,
            stride=stride,
            chunk_points=chunk_points,
            max_points=max_points,
            clip_bounds=clip_bounds,
            property_names=header.vertex_property_names,
            min_opacity=min_opacity,
            max_scale=max_scale,
        ):
            np.asarray(chunk, dtype="<f4").tofile(stream)
            emitted += int(len(chunk))

    if emitted != sampled_count:
        raise RuntimeError(f"实际写入点数不一致: expected={sampled_count} emitted={emitted}")

    return {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "source_vertex_count": int(header.vertex_count),
        "sampled_vertex_count": int(sampled_count),
        "sample_stride": int(stride),
        "clip_bounds": [bound.tolist() for bound in clip_bounds] if clip_bounds is not None else None,
        "min_opacity": min_opacity,
        "max_scale": max_scale,
        "vertex_properties": list(header.vertex_property_names),
        "output_size_bytes": int(output_ply.stat().st_size),
    }


def _iter_filtered_vertices(
    *,
    vertices: Any,
    vertex_count: int,
    stride: int,
    chunk_points: int,
    max_points: int,
    clip_bounds: tuple[np.ndarray, np.ndarray] | None,
    property_names: tuple[str, ...],
    min_opacity: float | None,
    max_scale: float | None,
) -> Any:
    indices = {name: index for index, name in enumerate(property_names)}
    emitted = 0
    step = max(1, int(stride)) * max(1, int(chunk_points))
    for source_start in range(0, vertex_count, step):
        source_stop = min(vertex_count, source_start + step)
        chunk = vertices[source_start:source_stop:stride]
        if clip_bounds is not None and len(chunk) > 0:
            lower, upper = clip_bounds
            xyz = np.asarray(chunk[:, 0:3], dtype=np.float32)
            mask = (np.isfinite(xyz).all(axis=1) & (xyz >= lower).all(axis=1) & (xyz <= upper).all(axis=1))
            chunk = chunk[mask]
        if min_opacity is not None and len(chunk) > 0:
            raw_opacity = np.asarray(chunk[:, indices["opacity"]], dtype=np.float32)
            opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
            chunk = chunk[opacity >= float(min_opacity)]
        if max_scale is not None and len(chunk) > 0:
            raw_scale = np.asarray(
                chunk[:, [indices["scale_0"], indices["scale_1"], indices["scale_2"]]],
                dtype=np.float32,
            )
            scale = np.exp(np.clip(raw_scale, -20.0, 20.0))
            chunk = chunk[scale.max(axis=1) <= float(max_scale)]
        if len(chunk) == 0:
            continue
        if max_points > 0:
            remaining = max_points - emitted
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
        yield chunk
        emitted += int(len(chunk))


def _count_filtered_vertices(
    *,
    vertices: Any,
    vertex_count: int,
    stride: int,
    chunk_points: int,
    max_points: int,
    clip_bounds: tuple[np.ndarray, np.ndarray] | None,
    property_names: tuple[str, ...],
    min_opacity: float | None,
    max_scale: float | None,
) -> int:
    return sum(
        int(len(chunk))
        for chunk in _iter_filtered_vertices(
            vertices=vertices,
            vertex_count=vertex_count,
            stride=stride,
            chunk_points=chunk_points,
            max_points=max_points,
            clip_bounds=clip_bounds,
            property_names=property_names,
            min_opacity=min_opacity,
            max_scale=max_scale,
        )
    )


def _clip_bounds_from_args(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    if args.clip_bounds is not None and args.clip_reference_ply is not None:
        raise ValueError("--clip-bounds 和 --clip-reference-ply 不能同时使用")
    if args.clip_bounds is not None:
        raw = np.asarray(args.clip_bounds, dtype=np.float32)
        lower = raw[[0, 2, 4]]
        upper = raw[[1, 3, 5]]
        if not np.all(lower < upper):
            raise ValueError("--clip-bounds 必须满足 min < max")
        return lower, upper
    if args.clip_reference_ply is None:
        return None

    reference_ply = _project_path(args.clip_reference_ply)
    header = parse_ply_header(reference_ply)
    if header.format_name != "binary_little_endian":
        raise ValueError(f"当前只支持 binary_little_endian reference PLY: {reference_ply}")
    if not header.all_vertex_properties_float32:
        raise ValueError("reference PLY 的 vertex property 必须全部是 float32")
    vertices = np.memmap(
        reference_ply,
        dtype="<f4",
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count, len(header.vertex_properties)),
    )
    xyz = vertices[:, 0:3]
    margin = np.asarray(args.clip_margin, dtype=np.float32)
    lower = np.nanmin(xyz, axis=0).astype(np.float32) - margin
    upper = np.nanmax(xyz, axis=0).astype(np.float32) + margin
    return lower, upper


def _write_ply_header(stream: Any, property_names: tuple[str, ...], vertex_count: int) -> None:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment generated_by arm_vla_pct/tools/scene/sample_gaussian_ply.py",
        f"element vertex {vertex_count}",
    ]
    lines.extend(f"property float {name}" for name in property_names)
    lines.append("end_header")
    stream.write(("\n".join(lines) + "\n").encode("ascii"))


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抽样生成小型 Gaussian PLY，用于 NuRec 渲染诊断。")
    parser.add_argument("--input-ply", default=os.fspath(DEFAULT_INPUT_PLY), help="输入 Gaussian visual PLY。")
    parser.add_argument("--output-ply", default=os.fspath(DEFAULT_OUTPUT_PLY), help="输出小型 Gaussian PLY。")
    parser.add_argument("--max-points", type=int, default=100_000, help="最多输出的 Gaussian 数量；传 0 表示输出裁剪后的全量点。")
    parser.add_argument("--sample-stride", type=int, default=1, help="固定采样 stride，会和 max-points 自动 stride 取较大值。")
    parser.add_argument("--chunk-points", type=int, default=100_000, help="分块写入点数。")
    parser.add_argument(
        "--clip-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="显式裁剪范围。",
    )
    parser.add_argument("--clip-reference-ply", default=None, help="用 reference PLY 的 bounds 生成裁剪范围。")
    parser.add_argument(
        "--clip-margin",
        type=float,
        nargs=3,
        default=(5.0, 5.0, 3.0),
        metavar=("MX", "MY", "MZ"),
        help="clip-reference-ply 的 XYZ 额外边界 margin。",
    )
    parser.add_argument("--report-output", default=os.fspath(DEFAULT_REPORT), help="输出报告 JSON。")
    parser.add_argument("--min-opacity", type=float, default=None, help="按 sigmoid(opacity) 过滤低透明度 Gaussian。")
    parser.add_argument("--max-scale", type=float, default=None, help="按 exp(max(scale_*)) 过滤过大的 Gaussian。")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
