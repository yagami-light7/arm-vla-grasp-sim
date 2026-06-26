"""Gaussian splat PLY 的离线渲染工具。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from source.scene.gaussian_splat_ply import is_gaussian_splat_ply, parse_ply_header


SH_C0 = 0.28209479177387814
SUPPORTED_COORD_MODES = ("sim_to_pct_180deg", "ply")


@dataclass(frozen=True)
class GaussianSplatLoadConfig:
    """控制从 Gaussian PLY 读取多少 splat。"""

    ply_path: Path
    max_gaussians: int = 2_000_000
    sample_stride: int = 1
    chunk_gaussians: int = 500_000
    clip_bounds: tuple[float, float, float, float, float, float] | None = None
    clip_percentiles: tuple[float, float] | None = None
    clip_estimate_stride: int = 100
    coord_mode: str = "sim_to_pct_180deg"
    scale_multiplier: float = 1.0
    opacity_multiplier: float = 1.0
    opacity_threshold: float = 0.0


@dataclass(frozen=True)
class CameraConfig:
    """描述一个 pinhole camera。"""

    eye: tuple[float, float, float] | None = None
    target: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vertical_fov_deg: float = 58.0
    width: int = 960
    height: int = 540
    near_plane: float = 0.01
    far_plane: float = 1.0e5


@dataclass(frozen=True)
class RenderConfig:
    """控制 gsplat rasterization 参数。"""

    radius_clip: float = 0.2
    eps2d: float = 0.3
    rasterize_mode: str = "antialiased"
    background_rgb: tuple[float, float, float] = (0.02, 0.025, 0.03)


@dataclass(frozen=True)
class LoadedGaussianSplats:
    """保存已载入内存的 Gaussian splat 数组。"""

    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    report: dict[str, Any]


def _property_indices(names: tuple[str, ...]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def _effective_stride(vertex_count: int, sample_stride: int, max_gaussians: int) -> int:
    """结合用户 stride 和最大点数限制，得到最终采样 stride。"""

    stride = max(1, int(sample_stride))
    if max_gaussians > 0:
        stride = max(stride, math.ceil(vertex_count / max_gaussians))
    return stride


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _normalize_quats(quats: np.ndarray) -> np.ndarray:
    """归一化 quaternion，避免 PLY 中的数值漂移影响 rasterizer。"""

    denom = np.linalg.norm(quats, axis=1, keepdims=True)
    return quats / np.maximum(denom, 1.0e-8)


def _rotate_z_180_quats_wxyz(quats: np.ndarray) -> np.ndarray:
    """把 PLY 坐标下的 quaternion 旋转到 Isaac Sim 世界坐标。"""

    w = quats[:, 0].copy()
    x = quats[:, 1].copy()
    y = quats[:, 2].copy()
    z = quats[:, 3].copy()
    rotated = np.empty_like(quats)
    rotated[:, 0] = -z
    rotated[:, 1] = -y
    rotated[:, 2] = x
    rotated[:, 3] = w
    return rotated


def _transform_xyz_and_quats(
    xyz: np.ndarray,
    quats: np.ndarray,
    coord_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """按统一坐标规则把 PLY/PCT 坐标转换到渲染坐标。"""

    if coord_mode == "ply":
        return xyz, quats
    if coord_mode == "sim_to_pct_180deg":
        converted_xyz = xyz.copy()
        converted_xyz[:, 0] = -converted_xyz[:, 0]
        converted_xyz[:, 1] = -converted_xyz[:, 1]
        return converted_xyz, _rotate_z_180_quats_wxyz(quats)
    raise ValueError(f"不支持的 coord_mode: {coord_mode}")


def _estimate_clip_bounds(
    vertices: Any,
    indices: dict[str, int],
    percentiles: tuple[float, float] | None,
    estimate_stride: int,
) -> tuple[float, float, float, float, float, float] | None:
    """从抽样点估计离群点裁剪范围。"""

    if percentiles is None:
        return None
    low, high = percentiles
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("clip_percentiles 必须满足 0 <= low < high <= 100")
    sample = np.asarray(
        vertices[:: max(1, int(estimate_stride)), [indices["x"], indices["y"], indices["z"]]],
        dtype=np.float32,
    )
    finite = np.isfinite(sample).all(axis=1)
    if not np.any(finite):
        raise ValueError("用于估计裁剪范围的点全部无效")
    sample = sample[finite]
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


def _clip_mask(
    xyz: np.ndarray,
    clip_bounds: tuple[float, float, float, float, float, float] | None,
) -> np.ndarray:
    mask = np.isfinite(xyz).all(axis=1)
    if clip_bounds is None:
        return mask
    xmin, xmax, ymin, ymax, zmin, zmax = clip_bounds
    return (
        mask
        & (xyz[:, 0] >= xmin)
        & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] <= ymax)
        & (xyz[:, 2] >= zmin)
        & (xyz[:, 2] <= zmax)
    )


def load_gaussian_splats(config: GaussianSplatLoadConfig) -> LoadedGaussianSplats:
    """从 3DGS PLY 读取 splat，并转换为 gsplat 需要的数组。"""

    ply_path = Path(config.ply_path).expanduser().resolve()
    if config.max_gaussians < 0:
        raise ValueError("max_gaussians 不能小于 0")
    if config.sample_stride <= 0:
        raise ValueError("sample_stride 必须为正数")
    if config.chunk_gaussians <= 0:
        raise ValueError("chunk_gaussians 必须为正数")
    if config.coord_mode not in SUPPORTED_COORD_MODES:
        raise ValueError(f"coord_mode 必须是 {SUPPORTED_COORD_MODES} 之一")
    if config.scale_multiplier <= 0.0:
        raise ValueError("scale_multiplier 必须为正数")
    if config.opacity_multiplier <= 0.0:
        raise ValueError("opacity_multiplier 必须为正数")

    header = parse_ply_header(ply_path)
    if header.format_name != "binary_little_endian":
        raise ValueError(f"当前只支持 binary_little_endian PLY: {ply_path}")
    if not header.all_vertex_properties_float32:
        raise ValueError("当前渲染器只支持全部 vertex property 为 float32 的 Gaussian PLY")
    if not is_gaussian_splat_ply(header):
        raise ValueError("PLY 缺少 3DGS 必需属性，不能按 Gaussian splat 渲染")

    prop_names = header.vertex_property_names
    indices = _property_indices(prop_names)
    vertices = np.memmap(
        ply_path,
        dtype="<f4",
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count, len(prop_names)),
    )
    clip_bounds = config.clip_bounds
    if clip_bounds is None:
        clip_bounds = _estimate_clip_bounds(
            vertices,
            indices,
            config.clip_percentiles,
            config.clip_estimate_stride,
        )

    stride = _effective_stride(header.vertex_count, config.sample_stride, config.max_gaussians)
    max_keep = config.max_gaussians if config.max_gaussians > 0 else header.vertex_count
    chunk_take = max(1, int(config.chunk_gaussians))

    means_chunks: list[np.ndarray] = []
    quats_chunks: list[np.ndarray] = []
    scales_chunks: list[np.ndarray] = []
    opacities_chunks: list[np.ndarray] = []
    colors_chunks: list[np.ndarray] = []
    emitted = 0

    for start in range(0, header.vertex_count, stride * chunk_take):
        if emitted >= max_keep:
            break
        stop = min(header.vertex_count, start + stride * chunk_take)
        chunk = vertices[start:stop:stride]
        if len(chunk) == 0:
            continue

        xyz = np.asarray(chunk[:, [indices["x"], indices["y"], indices["z"]]], dtype=np.float32)
        mask = _clip_mask(xyz, clip_bounds)
        if not np.any(mask):
            continue

        chunk = chunk[mask]
        xyz = xyz[mask]
        remaining = max_keep - emitted
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
            xyz = xyz[:remaining]

        quats = np.asarray(
            chunk[:, [indices["rot_0"], indices["rot_1"], indices["rot_2"], indices["rot_3"]]],
            dtype=np.float32,
        )
        quats = _normalize_quats(quats)
        xyz, quats = _transform_xyz_and_quats(xyz, quats, config.coord_mode)
        quats = _normalize_quats(quats)

        scales = np.exp(
            np.asarray(
                chunk[:, [indices["scale_0"], indices["scale_1"], indices["scale_2"]]],
                dtype=np.float32,
            )
        ).astype(np.float32)
        scales *= np.float32(config.scale_multiplier)

        opacities = _sigmoid(np.asarray(chunk[:, indices["opacity"]], dtype=np.float32))
        opacities = np.clip(opacities * np.float32(config.opacity_multiplier), 0.0, 1.0).astype(np.float32)
        keep_opacity = opacities >= np.float32(config.opacity_threshold)
        if not np.any(keep_opacity):
            continue
        xyz = xyz[keep_opacity]
        quats = quats[keep_opacity]
        scales = scales[keep_opacity]
        opacities = opacities[keep_opacity]
        chunk = chunk[keep_opacity]

        colors = np.asarray(
            chunk[:, [indices["f_dc_0"], indices["f_dc_1"], indices["f_dc_2"]]],
            dtype=np.float32,
        )
        colors = np.clip(0.5 + SH_C0 * colors, 0.0, 1.0).astype(np.float32)

        means_chunks.append(np.ascontiguousarray(xyz, dtype=np.float32))
        quats_chunks.append(np.ascontiguousarray(quats, dtype=np.float32))
        scales_chunks.append(np.ascontiguousarray(scales, dtype=np.float32))
        opacities_chunks.append(np.ascontiguousarray(opacities, dtype=np.float32))
        colors_chunks.append(np.ascontiguousarray(colors, dtype=np.float32))
        emitted += len(xyz)

    if emitted <= 0:
        raise ValueError("采样、裁剪和 opacity 过滤后没有可渲染的 Gaussian")

    means = np.concatenate(means_chunks, axis=0)
    quats = np.concatenate(quats_chunks, axis=0)
    scales = np.concatenate(scales_chunks, axis=0)
    opacities = np.concatenate(opacities_chunks, axis=0)
    colors = np.concatenate(colors_chunks, axis=0)

    bounds_min = means.min(axis=0)
    bounds_max = means.max(axis=0)
    report = {
        "ply_path": str(ply_path),
        "source_vertex_count": int(header.vertex_count),
        "loaded_gaussian_count": int(len(means)),
        "sample_stride": int(stride),
        "max_gaussians": int(config.max_gaussians),
        "full_density_requested": config.max_gaussians == 0 and stride == 1,
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "clip_percentiles": list(config.clip_percentiles) if config.clip_percentiles is not None else None,
        "coord_mode": config.coord_mode,
        "scale_multiplier": float(config.scale_multiplier),
        "opacity_multiplier": float(config.opacity_multiplier),
        "opacity_threshold": float(config.opacity_threshold),
        "render_bounds_min": [float(value) for value in bounds_min],
        "render_bounds_max": [float(value) for value in bounds_max],
    }
    return LoadedGaussianSplats(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        report=report,
    )


def _normalize_vector(values: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-8:
        raise ValueError(f"{label} 不能是零向量")
    return values / norm


def look_at_view_matrix(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """生成 gsplat 使用的 world-to-camera 矩阵，camera +Z 朝前。"""

    eye_vec = np.asarray(eye, dtype=np.float32)
    target_vec = np.asarray(target, dtype=np.float32)
    up_vec = np.asarray(up, dtype=np.float32)
    forward = _normalize_vector(target_vec - eye_vec, "camera forward")
    right = _normalize_vector(np.cross(forward, up_vec), "camera right")
    down = _normalize_vector(np.cross(forward, right), "camera down")
    rotation = np.stack([right, down, forward], axis=0).astype(np.float32)
    translation = (-rotation @ eye_vec.reshape(3, 1)).reshape(3)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = translation
    return view


def camera_intrinsics(width: int, height: int, vertical_fov_deg: float) -> np.ndarray:
    """生成 pinhole camera 内参矩阵。"""

    if width <= 0 or height <= 0:
        raise ValueError("width 和 height 必须为正数")
    if not (1.0 < vertical_fov_deg < 179.0):
        raise ValueError("vertical_fov_deg 必须在 1 到 179 度之间")
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(vertical_fov_deg))
    fx = fy
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def auto_overview_camera(means: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """根据载入的 Gaussian bounds 生成一个俯视检查相机。"""

    bounds_min = means.min(axis=0)
    bounds_max = means.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    extent = np.maximum(bounds_max - bounds_min, 1.0)
    xy_extent = float(max(extent[0], extent[1]))
    eye = center + np.asarray([0.55 * xy_extent, -0.85 * xy_extent, 0.48 * xy_extent], dtype=np.float32)
    target = center + np.asarray([0.0, 0.0, -0.10 * float(extent[2])], dtype=np.float32)
    return (
        (float(eye[0]), float(eye[1]), float(eye[2])),
        (float(target[0]), float(target[1]), float(target[2])),
    )


def render_gaussian_splats(
    splats: LoadedGaussianSplats,
    camera: CameraConfig,
    render_config: RenderConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """调用 gsplat 渲染 RGB 图像。"""

    import torch
    from gsplat.rendering import rasterization

    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境无法访问 CUDA，不能运行 gsplat 3DGS 渲染")

    eye, target = camera.eye, camera.target
    if eye is None or target is None:
        eye, target = auto_overview_camera(splats.means)

    view = look_at_view_matrix(eye, target, camera.up)
    k_matrix = camera_intrinsics(camera.width, camera.height, camera.vertical_fov_deg)
    device = torch.device("cuda")
    with torch.no_grad():
        means = torch.from_numpy(splats.means).to(device=device, dtype=torch.float32)
        quats = torch.from_numpy(splats.quats).to(device=device, dtype=torch.float32)
        scales = torch.from_numpy(splats.scales).to(device=device, dtype=torch.float32)
        opacities = torch.from_numpy(splats.opacities).to(device=device, dtype=torch.float32)
        colors = torch.from_numpy(splats.colors).to(device=device, dtype=torch.float32)
        viewmats = torch.from_numpy(view).to(device=device, dtype=torch.float32).unsqueeze(0)
        ks = torch.from_numpy(k_matrix).to(device=device, dtype=torch.float32).unsqueeze(0)
        render_colors, render_alphas, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            ks,
            camera.width,
            camera.height,
            near_plane=float(camera.near_plane),
            far_plane=float(camera.far_plane),
            radius_clip=float(render_config.radius_clip),
            eps2d=float(render_config.eps2d),
            packed=True,
            rasterize_mode=render_config.rasterize_mode,
        )
        rgb = render_colors[0].detach().float().cpu().numpy()
        alpha = render_alphas[0].detach().float().cpu().numpy()

    background = np.asarray(render_config.background_rgb, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb + (1.0 - alpha) * background, 0.0, 1.0)
    report = {
        "camera": {
            "eye": [float(value) for value in eye],
            "target": [float(value) for value in target],
            "up": [float(value) for value in camera.up],
            "vertical_fov_deg": float(camera.vertical_fov_deg),
            "width": int(camera.width),
            "height": int(camera.height),
            "near_plane": float(camera.near_plane),
            "far_plane": float(camera.far_plane),
            "view_matrix": view.tolist(),
            "intrinsics": k_matrix.tolist(),
        },
        "render": {
            "radius_clip": float(render_config.radius_clip),
            "eps2d": float(render_config.eps2d),
            "rasterize_mode": render_config.rasterize_mode,
            "background_rgb": [float(value) for value in render_config.background_rgb],
            "alpha_max": float(alpha.max()),
            "alpha_mean": float(alpha.mean()),
            "rgb_min": float(rgb.min()),
            "rgb_max": float(rgb.max()),
            "visible_tile_count": int(meta.get("tiles_per_gauss", np.asarray([])).numel())
            if hasattr(meta.get("tiles_per_gauss", None), "numel")
            else None,
        },
    }
    return rgb, report


def save_rgb_image(path: str | Path, rgb: np.ndarray) -> Path:
    """把 0..1 RGB 数组保存为 PNG。"""

    import imageio.v2 as imageio

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)
    imageio.imwrite(output_path, image)
    return output_path


def write_render_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """写出渲染报告 JSON。"""

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path
