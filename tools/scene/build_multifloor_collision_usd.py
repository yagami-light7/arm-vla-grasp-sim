#!/usr/bin/env python3

"""从 multifloor collision mesh PLY 生成 Isaac Sim 可识别的碰撞 USD。"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from source.scene.gaussian_splat_ply import parse_ply_header
from tools.scene.check_multifloor_ply import build_report


DEFAULT_PLY_DIR = PROJECT_ROOT / "source/scene/multifloor/ply"
DEFAULT_OUTPUT_USD = PROJECT_ROOT / "source/scene/multifloor/usd/multifloor_collision.usd"

_SCALAR_DTYPES = {
    "char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


def main() -> int:
    args = _parse_args()
    input_ply = _resolve_input_ply(args)
    output_usd = _project_path(args.output_usd)
    if input_ply.is_symlink():
        raise RuntimeError(f"输入 PLY 不能是软链接: {input_ply}")
    if output_usd.exists() and not args.force:
        raise FileExistsError(f"输出已存在，覆盖请传 --force: {output_usd}")
    if output_usd.exists():
        output_usd.unlink()

    points, face_counts, face_indices = _read_mesh_ply(input_ply)
    _write_collision_usd(
        output_usd=output_usd,
        points=points,
        face_counts=face_counts,
        face_indices=face_indices,
        approximation=args.approximation,
    )
    print(f"[OK] collision USD 已生成: {output_usd}")
    print(f"[OK] vertices={len(points)} faces={len(face_counts)} indices={len(face_indices)}")
    return 0


def _read_mesh_ply(path: Path) -> tuple[np.ndarray, list[int], list[int]]:
    header = parse_ply_header(path)
    if not header.has_faces:
        raise RuntimeError(f"collision PLY 必须包含 face，不能用纯点云代替: {path}")
    if header.format_name != "binary_little_endian":
        raise RuntimeError(f"当前脚本只支持 binary_little_endian mesh PLY: {path}")

    vertex_dtype = np.dtype([(prop.name, _SCALAR_DTYPES[prop.type_name]) for prop in header.vertex_properties])
    with path.open("rb") as stream:
        stream.seek(header.data_offset)
        vertices = np.fromfile(stream, dtype=vertex_dtype, count=header.vertex_count)
        points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
        face_counts, face_indices = _read_binary_faces(stream)
    return points, face_counts, face_indices


def _read_binary_faces(stream: Any) -> tuple[list[int], list[int]]:
    face_counts: list[int] = []
    face_indices: list[int] = []
    while True:
        count_raw = stream.read(1)
        if not count_raw:
            break
        count = struct.unpack("<B", count_raw)[0]
        raw_indices = stream.read(count * 4)
        if len(raw_indices) != count * 4:
            raise RuntimeError("PLY face 数据提前结束")
        indices = struct.unpack("<" + "i" * count, raw_indices)
        face_counts.append(count)
        face_indices.extend(int(index) for index in indices)
    if not face_counts:
        raise RuntimeError("PLY 中没有读取到 face 数据")
    return face_counts, face_indices


def _write_collision_usd(
    *,
    output_usd: Path,
    points: np.ndarray,
    face_counts: list[int],
    face_indices: list[int],
    approximation: str,
) -> None:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except Exception as exc:
        raise RuntimeError("无法导入 pxr。请在 /data/conda_envs/sage 或 Isaac Sim Python 中运行。") from exc

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(os.fspath(output_usd))
    stage.SetMetadata("defaultPrim", "scene_collision")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/scene_collision")
    mesh = UsdGeom.Mesh.Define(stage, "/scene_collision/CollisionMesh")
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_api.CreateApproximationAttr().Set(approximation)

    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def _resolve_input_ply(args: argparse.Namespace) -> Path:
    if args.input_ply:
        path = _project_path(args.input_ply)
        if not path.is_file():
            raise FileNotFoundError(f"输入 PLY 不存在: {path}")
        return path

    report = build_report(_project_path(args.ply_dir))
    selected = report.get("selected_collision_ply")
    if not selected:
        raise RuntimeError("无法自动选择 collision PLY，请先运行 check_multifloor_ply.py")
    return Path(str(selected))


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 multifloor collision USD。")
    parser.add_argument("--input-ply", default=None, help="collision mesh PLY；默认自动检测。")
    parser.add_argument("--ply-dir", default=os.fspath(DEFAULT_PLY_DIR), help="PLY 输入目录。")
    parser.add_argument("--output-usd", default=os.fspath(DEFAULT_OUTPUT_USD), help="输出 collision USD。")
    parser.add_argument("--approximation", default="none", help="UsdPhysics MeshCollisionAPI approximation。")
    parser.add_argument("--force", action="store_true", help="覆盖已有 USD。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
