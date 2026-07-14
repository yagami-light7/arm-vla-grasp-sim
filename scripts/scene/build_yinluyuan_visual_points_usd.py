#!/usr/bin/env python3

"""将 multifloor Gaussian PLY 转成 Isaac Sim 可加载的点云 USD 视觉层。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from source.scene.gaussian_splat_ply import describe_ply, is_gaussian_splat_ply, parse_ply_header

DEFAULT_SCENE_DIR = REPO_ROOT / "source" / "scene" / "multifloor"
DEFAULT_VISUAL_PLY = DEFAULT_SCENE_DIR / "ply" / "3dgs_visual.ply"
DEFAULT_OUTPUT_USD = DEFAULT_SCENE_DIR / "usda" / "multifloor_visual_points.usda"
DEFAULT_REPORT_JSON = DEFAULT_SCENE_DIR / "usda" / "multifloor_visual_points_report.json"
SH_C0 = 0.28209479177387814


def _project_path(raw_path: str | Path) -> Path:
    """将相对路径按仓库根目录解析。"""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _prim_name(raw_path: str) -> str:
    """点云 USD 内部只需要一个顶层 defaultPrim 名称。"""

    value = raw_path.strip().strip("/")
    if not value:
        raise ValueError("prim 名称不能为空")
    if "/" in value:
        raise ValueError("点云 visual USD 的 defaultPrim 只能是单层名称，例如 gauss")
    return value


def _effective_stride(vertex_count: int, sample_stride: int, max_points: int) -> int:
    """根据最大点数自动提高采样 stride。"""

    stride = max(1, int(sample_stride))
    if max_points > 0:
        stride = max(stride, math.ceil(vertex_count / max_points))
    return stride


def _sampled_count(vertex_count: int, stride: int, max_points: int) -> int:
    count = (vertex_count + stride - 1) // stride
    if max_points > 0:
        count = min(count, max_points)
    return count


def _property_indices(names: Iterable[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def _iter_sample_chunks(vertices: Any, *, stride: int, sampled_count: int, chunk_size: int) -> Iterable[Any]:
    """按采样 stride 分块读取 memmap，避免一次复制 1.6GB PLY。"""

    emitted = 0
    source_index = 0
    while emitted < sampled_count:
        remaining = sampled_count - emitted
        take = min(max(1, chunk_size), remaining)
        source_stop = source_index + stride * take
        chunk = vertices[source_index:source_stop:stride]
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        if len(chunk) == 0:
            break
        yield chunk
        emitted += len(chunk)
        source_index += stride * len(chunk)


def _iter_filtered_sample_chunks(
    vertices: Any,
    *,
    indices: dict[str, int],
    stride: int,
    raw_sampled_count: int,
    chunk_size: int,
    clip_bounds: tuple[float, float, float, float, float, float] | None,
    max_points: int,
) -> Iterable[Any]:
    """按需过滤离群点，并在过滤后限制最大点数。"""

    emitted = 0
    for chunk in _iter_sample_chunks(
        vertices,
        stride=stride,
        sampled_count=raw_sampled_count,
        chunk_size=chunk_size,
    ):
        if clip_bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = clip_bounds
            xyz = np.asarray(chunk[:, [indices["x"], indices["y"], indices["z"]]], dtype=np.float32)
            mask = (
                np.isfinite(xyz).all(axis=1)
                & (xyz[:, 0] >= xmin)
                & (xyz[:, 0] <= xmax)
                & (xyz[:, 1] >= ymin)
                & (xyz[:, 1] <= ymax)
                & (xyz[:, 2] >= zmin)
                & (xyz[:, 2] <= zmax)
            )
            chunk = chunk[mask]
        if len(chunk) == 0:
            continue
        if max_points > 0:
            remaining = max_points - emitted
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
        yield chunk
        emitted += len(chunk)


def _count_filtered_points(
    vertices: Any,
    *,
    indices: dict[str, int],
    stride: int,
    raw_sampled_count: int,
    chunk_size: int,
    clip_bounds: tuple[float, float, float, float, float, float] | None,
    max_points: int,
) -> int:
    """先数一次过滤后的点数，便于写入 USD metadata。"""

    return sum(
        len(chunk)
        for chunk in _iter_filtered_sample_chunks(
            vertices,
            indices=indices,
            stride=stride,
            raw_sampled_count=raw_sampled_count,
            chunk_size=chunk_size,
            clip_bounds=clip_bounds,
            max_points=max_points,
        )
    )


def _clip_bounds_from_args(args: argparse.Namespace, vertices: Any, indices: dict[str, int]) -> tuple[float, float, float, float, float, float] | None:
    """根据 CLI 参数得到可选坐标裁剪范围。"""

    if args.clip_bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = (float(value) for value in args.clip_bounds)
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            raise ValueError("--clip-bounds 必须满足 min < max")
        return xmin, xmax, ymin, ymax, zmin, zmax

    if args.clip_percentiles is None:
        return None

    low, high = (float(value) for value in args.clip_percentiles)
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("--clip-percentiles 必须满足 0 <= low < high <= 100")
    sample = np.asarray(
        vertices[:: max(1, int(args.clip_estimate_stride)), [indices["x"], indices["y"], indices["z"]]],
        dtype=np.float32,
    )
    lower = np.percentile(sample, low, axis=0)
    upper = np.percentile(sample, high, axis=0)
    return (
        float(lower[0]),
        float(upper[0]),
        float(lower[1]),
        float(upper[1]),
        float(lower[2]),
        float(upper[2]),
    )


def _gaussian_colors(chunk: Any, indices: dict[str, int], color_mode: str) -> np.ndarray:
    """把 3DGS 的 DC SH 颜色转成 USD displayColor。"""

    if color_mode == "white":
        return np.ones((len(chunk), 3), dtype=np.float32)
    if color_mode == "height":
        z_values = np.asarray(chunk[:, indices["z"]], dtype=np.float32)
        z_min = float(z_values.min()) if len(z_values) else 0.0
        z_max = float(z_values.max()) if len(z_values) else 1.0
        denom = max(z_max - z_min, 1e-6)
        t = ((z_values - z_min) / denom).reshape(-1, 1)
        return np.concatenate([0.2 + 0.6 * t, 0.45 + 0.35 * (1.0 - t), 0.9 - 0.5 * t], axis=1).astype(np.float32)

    dc = np.asarray(
        chunk[:, [indices["f_dc_0"], indices["f_dc_1"], indices["f_dc_2"]]],
        dtype=np.float32,
    )
    return np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)


def _gaussian_opacity(chunk: Any, indices: dict[str, int]) -> np.ndarray:
    raw_opacity = np.asarray(chunk[:, indices["opacity"]], dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-raw_opacity))


def _write_vec3_array(
    stream: Any,
    *,
    attr_name: str,
    value_type: str,
    chunks: Iterable[np.ndarray],
    decimals: int,
    interpolation: str | None = None,
) -> None:
    """写 USDA vec3 数组。"""

    suffix = ""
    if interpolation is not None:
        suffix = f' (\n            interpolation = "{interpolation}"\n        )'
    stream.write(f"        {value_type}[] {attr_name} = [\n")
    fmt = f"{{:.{decimals}f}}"
    for chunk in chunks:
        for x_value, y_value, z_value in chunk:
            stream.write(
                "            ("
                + fmt.format(float(x_value))
                + ", "
                + fmt.format(float(y_value))
                + ", "
                + fmt.format(float(z_value))
                + "),\n"
            )
    stream.write(f"        ]{suffix}\n")


def _write_float_array(
    stream: Any,
    *,
    attr_name: str,
    chunks: Iterable[np.ndarray],
    decimals: int,
    interpolation: str | None = None,
) -> None:
    """写 USDA float 数组。"""

    suffix = ""
    if interpolation is not None:
        suffix = f' (\n            interpolation = "{interpolation}"\n        )'
    stream.write(f"        float[] {attr_name} = [\n")
    fmt = f"{{:.{decimals}f}}"
    for chunk in chunks:
        for value in chunk:
            stream.write("            " + fmt.format(float(value)) + ",\n")
    stream.write(f"        ]{suffix}\n")


def _write_visual_points_usda(args: argparse.Namespace) -> dict[str, Any]:
    """从 Gaussian PLY 采样并写出 UsdGeom.Points ASCII layer。"""

    visual_ply = _project_path(args.visual_ply)
    output_usd = _project_path(args.output_usd)
    header = parse_ply_header(visual_ply)
    if header.format_name != "binary_little_endian":
        raise ValueError(f"当前只支持 binary_little_endian PLY: {visual_ply}")
    if not header.all_vertex_properties_float32:
        raise ValueError("当前点云 USD 生成器只支持全部 vertex property 为 float32 的 Gaussian PLY")
    if not is_gaussian_splat_ply(header):
        raise ValueError("visual PLY 缺少 3DGS 必需属性，不能按 Gaussian splat 方式生成点云视觉层")

    prop_names = header.vertex_property_names
    indices = _property_indices(prop_names)
    stride = _effective_stride(header.vertex_count, args.sample_stride, args.max_points)
    raw_sampled_count = _sampled_count(header.vertex_count, stride, 0)

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists() and not args.force:
        raise FileExistsError(f"输出已存在，重新生成请传 --force: {output_usd}")

    vertices = np.memmap(
        visual_ply,
        dtype="<f4",
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count, len(prop_names)),
    )
    clip_bounds = _clip_bounds_from_args(args, vertices, indices)
    point_count = _count_filtered_points(
        vertices,
        indices=indices,
        stride=stride,
        raw_sampled_count=raw_sampled_count,
        chunk_size=max(1, int(args.chunk_points)),
        clip_bounds=clip_bounds,
        max_points=args.max_points,
    )
    if point_count <= 0:
        raise ValueError("采样和过滤后点数为 0")
    prim_name = _prim_name(args.default_prim)
    chunk_size = max(1, int(args.chunk_points))

    def point_chunks() -> Iterable[np.ndarray]:
        for chunk in _iter_filtered_sample_chunks(
            vertices,
            indices=indices,
            stride=stride,
            raw_sampled_count=raw_sampled_count,
            chunk_size=chunk_size,
            clip_bounds=clip_bounds,
            max_points=args.max_points,
        ):
            yield np.asarray(chunk[:, [indices["x"], indices["y"], indices["z"]]], dtype=np.float32)

    def color_chunks() -> Iterable[np.ndarray]:
        for chunk in _iter_filtered_sample_chunks(
            vertices,
            indices=indices,
            stride=stride,
            raw_sampled_count=raw_sampled_count,
            chunk_size=chunk_size,
            clip_bounds=clip_bounds,
            max_points=args.max_points,
        ):
            yield _gaussian_colors(chunk, indices, args.color_mode)

    def opacity_chunks() -> Iterable[np.ndarray]:
        for chunk in _iter_filtered_sample_chunks(
            vertices,
            indices=indices,
            stride=stride,
            raw_sampled_count=raw_sampled_count,
            chunk_size=chunk_size,
            clip_bounds=clip_bounds,
            max_points=args.max_points,
        ):
            yield _gaussian_opacity(chunk, indices)

    with output_usd.open("w", encoding="utf-8") as stream:
        stream.write("#usda 1.0\n")
        stream.write("(\n")
        stream.write(f'    defaultPrim = "{prim_name}"\n')
        stream.write('    metersPerUnit = 1\n')
        stream.write('    upAxis = "Z"\n')
        stream.write(")\n\n")
        stream.write(f'def Xform "{prim_name}"\n')
        stream.write("{\n")
        stream.write('    def Points "points"\n')
        stream.write("    {\n")
        stream.write(f'        string sourceVisualPly = "{visual_ply}"\n')
        stream.write(f"        int64 sourceVertexCount = {header.vertex_count}\n")
        stream.write(f"        int64 sampledPointCount = {point_count}\n")
        stream.write(f"        int sampleStride = {stride}\n")
        stream.write(f"        float[] widths = [{float(args.point_width):.{args.decimals}f}]\n")
        _write_vec3_array(
            stream,
            attr_name="points",
            value_type="point3f",
            chunks=point_chunks(),
            decimals=args.decimals,
        )
        _write_vec3_array(
            stream,
            attr_name="primvars:displayColor",
            value_type="color3f",
            chunks=color_chunks(),
            decimals=args.decimals,
            interpolation="vertex",
        )
        if args.opacity_mode == "ply":
            _write_float_array(
                stream,
                attr_name="primvars:displayOpacity",
                chunks=opacity_chunks(),
                decimals=args.decimals,
                interpolation="vertex",
            )
        stream.write("    }\n")
        stream.write("}\n")

    return {
        "visual_ply": str(visual_ply),
        "output_usd": str(output_usd),
        "format": header.format_name,
        "source_vertex_count": header.vertex_count,
        "sampled_point_count": point_count,
        "sample_stride": stride,
        "max_points": int(args.max_points),
        "full_density": stride == 1 and point_count == header.vertex_count,
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "clip_percentiles": list(args.clip_percentiles) if args.clip_percentiles is not None else None,
        "color_mode": args.color_mode,
        "opacity_mode": args.opacity_mode,
        "point_width": float(args.point_width),
        "default_prim": prim_name,
        "points_prim": f"/{prim_name}/points",
    }


def _write_report(args: argparse.Namespace, payload: dict[str, Any]) -> Path:
    report_path = _project_path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 multifloor Gaussian PLY，并按需生成 UsdGeom.Points 视觉 USD。",
    )
    parser.add_argument("--visual-ply", default=os.fspath(DEFAULT_VISUAL_PLY), help="Gaussian visual PLY 输入路径。")
    parser.add_argument("--output-usd", default=os.fspath(DEFAULT_OUTPUT_USD), help="输出的点云 visual USD/USDA 路径。")
    parser.add_argument("--report-output", default=os.fspath(DEFAULT_REPORT_JSON), help="视觉层构建报告路径。")
    parser.add_argument("--default-prim", default="gauss", help="输出 USD 的 defaultPrim 名称。")
    parser.add_argument("--max-points", type=int, default=2_000_000, help="最多写入点数；传 0 表示全密度写入。")
    parser.add_argument("--sample-stride", type=int, default=1, help="固定采样 stride，会与 max-points 自动 stride 取较大值。")
    parser.add_argument("--chunk-points", type=int, default=100_000, help="分块读取和写入的点数。")
    parser.add_argument(
        "--clip-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="显式裁剪点云坐标范围，用于去除极端离群点。",
    )
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="从抽样点估计每个坐标轴的百分位裁剪范围，例如 0.1 99.9。",
    )
    parser.add_argument("--clip-estimate-stride", type=int, default=100, help="估计百分位裁剪范围时的抽样 stride。")
    parser.add_argument("--point-width", type=float, default=0.015, help="UsdGeom.Points 的点宽度，单位为 stage 米。")
    parser.add_argument("--color-mode", choices=("sh_dc", "height", "white"), default="sh_dc", help="点云颜色来源。")
    parser.add_argument("--opacity-mode", choices=("none", "ply"), default="none", help="是否写入 PLY opacity。")
    parser.add_argument("--decimals", type=int, default=5, help="USDA 数值小数位数。")
    parser.add_argument("--inspect-only", action="store_true", help="只检查 PLY header 和报告，不写 USD。")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出文件。")
    parser.add_argument("--skip-report", action="store_true", help="不写 JSON 报告。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_points < 0:
        raise ValueError("--max-points 不能小于 0")
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride 必须为正数")
    if args.chunk_points <= 0:
        raise ValueError("--chunk-points 必须为正数")
    if args.clip_bounds is not None and args.clip_percentiles is not None:
        raise ValueError("--clip-bounds 和 --clip-percentiles 不能同时使用")
    if args.clip_estimate_stride <= 0:
        raise ValueError("--clip-estimate-stride 必须为正数")
    if args.point_width <= 0.0:
        raise ValueError("--point-width 必须为正数")
    if args.decimals < 1:
        raise ValueError("--decimals 必须至少为 1")

    visual_ply = _project_path(args.visual_ply)
    inspection = describe_ply(visual_ply)
    payload: dict[str, Any] = {
        "status": "passed",
        "script": "scripts/scene/build_yinluyuan_visual_points_usd.py",
        "inspection": inspection,
        "notes": [
            "该脚本生成的是 UsdGeom.Points 点云视觉近似，不是原生 3DGS splat renderer。",
            "最终视频如需完整 3DGS 质量，应使用原生 3DGS 渲染扩展或外部工具生成高质量 visual USD。",
            "输出 .usd/.usda 属于本地大资产，按 .gitignore 不提交。",
        ],
    }
    if args.inspect_only:
        payload["visual_points"] = {
            "generated": False,
            "reason": "inspect_only",
        }
    else:
        payload["visual_points"] = _write_visual_points_usda(args)

    if not args.skip_report:
        report_path = _write_report(args, payload)
        payload["report_output"] = str(report_path)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
