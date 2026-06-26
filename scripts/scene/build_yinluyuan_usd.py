#!/usr/bin/env python3

"""将 multifloor PLY 场景转换为 Isaac Sim 可导入的 legacy USD stage。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from source.scene.gaussian_splat_ply import is_gaussian_splat_ply, parse_ply_header

DEFAULT_SCENE_DIR = REPO_ROOT / "source" / "scene" / "multifloor"
DEFAULT_COLLISION_PLY = DEFAULT_SCENE_DIR / "ply" / "3dgs_collision.ply"
DEFAULT_VISUAL_PLY = DEFAULT_SCENE_DIR / "ply" / "3dgs_visual.ply"
DEFAULT_COLLISION_USD = DEFAULT_SCENE_DIR / "usd" / "multifloor_collision_legacy.usd"
DEFAULT_VISUAL_USD = DEFAULT_SCENE_DIR / "usd" / "multifloor_visual_legacy.usd"
DEFAULT_VISUAL_POINTS_USD = DEFAULT_SCENE_DIR / "usda" / "multifloor_visual_points.usda"
DEFAULT_STAGE_USD = DEFAULT_SCENE_DIR / "usda" / "multifloor_legacy_go2_x5.usd"
DEFAULT_REPORT_JSON = DEFAULT_SCENE_DIR / "usda" / "multifloor_legacy_usd_build_report.json"


@dataclass(frozen=True)
class ConvertedAsset:
    """记录已转换并挂载到主 stage 的资产。"""

    label: str
    source_ply: Path
    output_usd: Path
    prim_path: str


def _project_path(raw_path: str | Path) -> Path:
    """将相对路径按仓库根目录解析，避免写死用户本机绝对路径。"""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _prim_path(raw_path: str) -> str:
    """统一 prim path 格式，避免调用端漏写开头斜杠。"""

    value = raw_path.strip()
    if not value:
        raise ValueError("prim path 不能为空")
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/") or "/"


def _usd_asset_path(stage_path: Path, asset_path: Path) -> str:
    """返回适合写入 USD reference 的相对 asset path。"""

    relative = os.path.relpath(asset_path.resolve(), start=stage_path.resolve().parent)
    return relative.replace(os.sep, "/")


def _ensure_input(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} 是空文件: {path}")


def _prepare_output(path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"输出已存在，重新生成请传 --force: {path}")
    path.unlink()


def _ensure_visual_ply_mesh_convertible(path: Path) -> None:
    """避免把 3DGS Gaussian PLY 当普通 mesh 交给 asset converter。"""

    header = parse_ply_header(path)
    if is_gaussian_splat_ply(header):
        raise RuntimeError(
            "visual PLY 是 Gaussian splat 点云，不是普通 mesh；"
            "请先运行 scripts/scene/build_yinluyuan_visual_points_usd.py 生成点云 visual USD，"
            "或传入真正的 3DGS/mesh visual USD 到 --visual-reference-usd。"
        )
    if not header.has_faces:
        raise RuntimeError(
            "visual PLY 没有 face，不能通过 mesh asset converter 生成完整视觉层；"
            "请使用 --visual-reference-usd 挂载专门生成的 visual USD。"
        )


def _progress_callback(*values: Any) -> int:
    """兼容不同 Kit 版本的一参或两参 progress callback。"""

    if not values:
        return 0
    if len(values) == 1:
        current = int(values[0])
        print(f"[converter] progress={current}")
        return current
    current = int(values[0])
    total = int(values[1])
    print(f"[converter] progress={current}/{total}")
    return current


def _start_simulation_app(*, headless: bool):
    """启动 Isaac Sim Kit，asset converter 需要在 Kit 进程内运行。"""

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp  # type: ignore

    return SimulationApp({"headless": headless})


def _converter_context(*, merge_meshes: bool, single_mesh: bool, ignore_materials: bool):
    """创建保守的 mesh 转换配置，单位和 up-axis 交给主 stage 明确声明。"""

    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = ignore_materials
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.merge_all_meshes = merge_meshes
    context.single_mesh = single_mesh
    context.smooth_normals = True
    context.create_world_as_default_root_prim = True
    context.convert_stage_up_z = True
    context.use_double_precision_to_usd_transform_op = True
    return context


async def _convert_asset_with_kit(
    *,
    source_ply: Path,
    output_usd: Path,
    merge_meshes: bool,
    single_mesh: bool,
    ignore_materials: bool,
) -> None:
    """调用 Kit asset converter，把单个 PLY 转成 USD。"""

    import omni.kit.asset_converter

    context = _converter_context(
        merge_meshes=merge_meshes,
        single_mesh=single_mesh,
        ignore_materials=ignore_materials,
    )
    converter = omni.kit.asset_converter.get_instance()
    task = converter.create_converter_task(
        os.fspath(source_ply),
        os.fspath(output_usd),
        _progress_callback,
        context,
    )
    success = await task.wait_until_finished()
    if success and output_usd.is_file() and output_usd.stat().st_size > 0:
        return

    status = task.get_status() if hasattr(task, "get_status") else "unknown"
    message = task.get_error_message() if hasattr(task, "get_error_message") else ""
    raise RuntimeError(
        "PLY 转 USD 失败: "
        f"source={source_ply}, output={output_usd}, status={status}, error={message}"
    )


def _run_async(coro: Any) -> Any:
    """在普通 Python main 线程里执行 Kit async task。"""

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _convert_assets(args: argparse.Namespace) -> list[ConvertedAsset]:
    """按参数转换 collision PLY，并按需转换 visual PLY。"""

    collision_ply = _project_path(args.collision_ply)
    collision_usd = _project_path(args.collision_output)
    _ensure_input(collision_ply, "collision PLY")
    _prepare_output(collision_usd, force=args.force)

    assets = [
        ConvertedAsset(
            label="collision",
            source_ply=collision_ply,
            output_usd=collision_usd,
            prim_path=_prim_path(args.collision_prim_path),
        )
    ]

    if args.include_visual:
        visual_ply = _project_path(args.visual_ply)
        visual_usd = _project_path(args.visual_output)
        _ensure_input(visual_ply, "visual PLY")
        _ensure_visual_ply_mesh_convertible(visual_ply)
        _prepare_output(visual_usd, force=args.force)
        assets.append(
            ConvertedAsset(
                label="visual",
                source_ply=visual_ply,
                output_usd=visual_usd,
                prim_path=_prim_path(args.visual_prim_path),
            )
        )

    for asset in assets:
        print(f"[INFO] 转换 {asset.label}: {asset.source_ply} -> {asset.output_usd}")
        _run_async(
            _convert_asset_with_kit(
                source_ply=asset.source_ply,
                output_usd=asset.output_usd,
                merge_meshes=args.merge_meshes,
                single_mesh=args.single_mesh,
                ignore_materials=args.ignore_materials,
            )
        )

    return assets


def _first_reference_prim_path(Usd: Any, asset_path: Path) -> str | None:
    """在 converter 未写 defaultPrim 时，退回引用唯一根 prim。"""

    stage = Usd.Stage.Open(os.fspath(asset_path))
    if stage is None:
        raise RuntimeError(f"无法打开转换后的 USD: {asset_path}")
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return None

    root_children = [child for child in stage.GetPseudoRoot().GetChildren() if child.IsValid()]
    if len(root_children) == 1:
        return str(root_children[0].GetPath())
    raise RuntimeError(
        f"转换后的 USD 没有 defaultPrim，且根 prim 数量不是 1，无法安全引用: {asset_path}"
    )


def _author_xform(
    *,
    UsdGeom: Any,
    prim: Any,
    rotate_z_deg: float,
    scale: float,
    translate: tuple[float, float, float],
) -> None:
    """给场景根 prim 写入统一变换，默认 180 度 Z 旋转对齐 PCT 坐标约定。"""

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if scale != 1.0:
        xform.AddScaleOp().Set((scale, scale, scale))
    if rotate_z_deg != 0.0:
        xform.AddRotateZOp().Set(float(rotate_z_deg))
    if translate != (0.0, 0.0, 0.0):
        xform.AddTranslateOp().Set(translate)


def _add_reference(*, Usd: Any, Sdf: Any, stage_path: Path, prim: Any, asset_path: Path) -> None:
    """把转换后的 USD 作为 reference 挂到主 stage。"""

    reference_asset = _usd_asset_path(stage_path, asset_path)
    reference_prim_path = _first_reference_prim_path(Usd, asset_path)
    if reference_prim_path is None:
        prim.GetReferences().AddReference(reference_asset)
    else:
        prim.GetReferences().AddReference(Sdf.Reference(reference_asset, reference_prim_path))


def _apply_collision_api(
    *,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    root_prim: Any,
    approximation: str,
) -> tuple[int, int]:
    """给 collision subtree 下的 mesh 补物理碰撞 schema。"""

    mesh_count = 0
    collision_count = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(approximation)
        collision_count += 1
    return mesh_count, collision_count


def _visual_proxy_asset(args: argparse.Namespace, assets: list[ConvertedAsset]) -> ConvertedAsset | None:
    """在没有真实视觉 USD 时，用 collision USD 生成轻量可视化代理。"""

    if args.include_visual or args.visual_reference_usd or not args.visual_proxy_from_collision:
        return None
    collision_asset = next((asset for asset in assets if asset.label == "collision"), None)
    if collision_asset is None:
        return None
    return ConvertedAsset(
        label="visual_proxy",
        source_ply=collision_asset.source_ply,
        output_usd=collision_asset.output_usd,
        prim_path=_prim_path(args.visual_prim_path),
    )


def _visual_reference_asset(args: argparse.Namespace) -> ConvertedAsset | None:
    """引用已经生成的真实或点云视觉 USD。"""

    if not args.visual_reference_usd:
        return None
    visual_reference = _project_path(args.visual_reference_usd)
    _ensure_input(visual_reference, "visual reference USD")
    return ConvertedAsset(
        label="visual_reference",
        source_ply=visual_reference,
        output_usd=visual_reference,
        prim_path=_prim_path(args.visual_prim_path),
    )


def _apply_visual_style(
    *,
    Gf: Any,
    Usd: Any,
    UsdGeom: Any,
    root_prim: Any,
    color: tuple[float, float, float],
    opacity: float,
) -> int:
    """给可视化代理写 displayColor，便于在 Isaac Sim 里辨认结构。"""

    styled_count = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
        styled_count += 1
    return styled_count


def _add_inspection_helpers(args: argparse.Namespace, *, Gf: Any, UsdGeom: Any, UsdLux: Any, stage: Any) -> dict[str, str]:
    """添加灯光和相机，方便直接打开 stage 做人工检查。"""

    helpers: dict[str, str] = {}
    if args.add_inspection_light:
        light = UsdLux.DistantLight.Define(stage, "/World/inspection_light")
        light.CreateIntensityAttr().Set(float(args.inspection_light_intensity))
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-60.0, 0.0, 35.0))
        helpers["light"] = "/World/inspection_light"

    if args.add_inspection_camera:
        camera = UsdGeom.Camera.Define(stage, "/World/inspection_camera")
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(
            Gf.Vec3d(
                float(args.inspection_camera_x),
                float(args.inspection_camera_y),
                float(args.inspection_camera_z),
            )
        )
        xform.AddRotateXYZOp().Set(
            Gf.Vec3f(
                float(args.inspection_camera_rot_x),
                float(args.inspection_camera_rot_y),
                float(args.inspection_camera_rot_z),
            )
        )
        camera.CreateFocalLengthAttr().Set(float(args.inspection_camera_focal_length))
        helpers["camera"] = "/World/inspection_camera"
    return helpers


def _write_main_stage(args: argparse.Namespace, assets: list[ConvertedAsset]) -> dict[str, Any]:
    """生成 legacy 调试用 multifloor 主 USD stage。"""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    stage_output = _project_path(args.stage_output)
    _prepare_output(stage_output, force=args.force)

    stage = Usd.Stage.CreateNew(os.fspath(stage_output))
    if stage is None:
        raise RuntimeError(f"无法创建主 USD stage: {stage_output}")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, float(args.meters_per_unit))

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    collision_mesh_count = 0
    collision_api_count = 0
    collision_visual_visibility = "inherited"
    mounted_assets: list[dict[str, str]] = []
    visual_styled_count = 0
    translate = (float(args.translate_x), float(args.translate_y), float(args.translate_z))
    stage_assets = list(assets)
    visual_reference = _visual_reference_asset(args)
    if visual_reference is not None:
        stage_assets.append(visual_reference)
    visual_proxy = _visual_proxy_asset(args, assets)
    if visual_proxy is not None:
        stage_assets.append(visual_proxy)

    for asset in stage_assets:
        root = UsdGeom.Xform.Define(stage, asset.prim_path)
        _author_xform(
            UsdGeom=UsdGeom,
            prim=root.GetPrim(),
            rotate_z_deg=float(args.rotate_z_deg),
            scale=float(args.scale),
            translate=translate,
        )
        _add_reference(
            Usd=Usd,
            Sdf=Sdf,
            stage_path=stage_output,
            prim=root.GetPrim(),
            asset_path=asset.output_usd,
        )
        if asset.label == "collision":
            mesh_count, api_count = _apply_collision_api(
                Usd=Usd,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
                root_prim=root.GetPrim(),
                approximation=args.collision_approximation,
            )
            collision_mesh_count += mesh_count
            collision_api_count += api_count
            if not args.collision_visual_visible:
                UsdGeom.Imageable(root.GetPrim()).MakeInvisible()
                collision_visual_visibility = "invisible"
        elif asset.label == "visual_proxy":
            visual_styled_count += _apply_visual_style(
                Gf=Gf,
                Usd=Usd,
                UsdGeom=UsdGeom,
                root_prim=root.GetPrim(),
                color=(
                    float(args.visual_proxy_color_r),
                    float(args.visual_proxy_color_g),
                    float(args.visual_proxy_color_b),
                ),
                opacity=float(args.visual_proxy_opacity),
            )

        mounted_assets.append(
            {
                "label": asset.label,
                "source_path": os.fspath(asset.source_ply),
                "output_usd": os.fspath(asset.output_usd),
                "prim_path": asset.prim_path,
            }
        )

    if args.add_physics_scene:
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    helpers = _add_inspection_helpers(args, Gf=Gf, UsdGeom=UsdGeom, UsdLux=UsdLux, stage=stage)
    stage.GetRootLayer().Save()

    return {
        "stage_output": os.fspath(stage_output),
        "mounted_assets": mounted_assets,
        "collision_prim_path": _prim_path(args.collision_prim_path),
        "collision_mesh_count": collision_mesh_count,
        "collision_api_count": collision_api_count,
        "collision_visual_visibility": collision_visual_visibility,
        "visual_included": bool(args.include_visual),
        "visual_reference_usd": os.fspath(_project_path(args.visual_reference_usd)) if args.visual_reference_usd else None,
        "visual_proxy_from_collision": bool(visual_proxy is not None),
        "visual_mode": (
            "reference"
            if visual_reference is not None
            else "mesh_converter"
            if any(asset.label == "visual" for asset in assets)
            else "collision_proxy"
            if visual_proxy is not None
            else "none"
        ),
        "visual_styled_count": visual_styled_count,
        "inspection_helpers": helpers,
        "rotate_z_deg": float(args.rotate_z_deg),
        "scale": float(args.scale),
        "translate": list(translate),
        "meters_per_unit": float(args.meters_per_unit),
    }


def _validate_stage(stage_path: Path, collision_prim_path: str) -> dict[str, Any]:
    """复用 runtime 前置条件，确认主 stage 可打开且 collision subtree 有 mesh。"""

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(os.fspath(stage_path))
    if stage is None:
        raise RuntimeError(f"无法打开主 USD stage: {stage_path}")
    prim = stage.GetPrimAtPath(collision_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"主 USD 缺少 collision prim: {collision_prim_path}")

    mesh_count = 0
    collision_api_count = 0
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            mesh_count += 1
        if child.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
    if mesh_count == 0:
        raise RuntimeError(f"collision prim 下没有 mesh: {collision_prim_path}")
    collision_visibility = str(UsdGeom.Imageable(prim).ComputeVisibility())
    return {
        "stage_opened": True,
        "collision_prim_path": collision_prim_path,
        "mesh_count": mesh_count,
        "collision_api_count": collision_api_count,
        "collision_visual_visibility": collision_visibility,
    }


def _validate_visual_prim(stage_path: Path, visual_prim_path: str) -> dict[str, Any]:
    """确认视觉 prim 存在，并统计 mesh / points。"""

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(os.fspath(stage_path))
    if stage is None:
        raise RuntimeError(f"无法打开主 USD stage: {stage_path}")
    prim = stage.GetPrimAtPath(visual_prim_path)
    mesh_count = 0
    points_count = 0
    gprim_count = 0
    if prim.IsValid():
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                mesh_count += 1
            if child.IsA(UsdGeom.Points):
                points_count += 1
            if child.IsA(UsdGeom.Gprim):
                gprim_count += 1
    return {
        "visual_prim_path": visual_prim_path,
        "visual_prim_valid": bool(prim.IsValid()),
        "mesh_count": mesh_count,
        "points_count": points_count,
        "gprim_count": gprim_count,
    }


def _write_report(args: argparse.Namespace, stage_report: dict[str, Any]) -> Path:
    """写出本地构建报告，方便后续记录实际生成路径。"""

    report_path = _project_path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "script": "scripts/scene/build_yinluyuan_usd.py",
        "stage": stage_report,
        "notes": [
            "生成的 USD 属于本地资产，按 .gitignore 不提交。",
            "默认只转换 collision PLY；Gaussian visual PLY 不能按 mesh converter 处理。",
            "主 stage 默认隐藏 /World/scene_collision 的渲染可见性，但保留 CollisionAPI 和物理碰撞。",
            "最终视频应通过 --visual-reference-usd 挂载真实 3DGS/mesh visual USD，或先生成点云 visual USD。",
            "未指定 visual reference 时，默认用 collision USD 在 /World/gauss 下生成轻量可视化代理，方便人工检查。",
            "默认在 USD root 上施加 180 度 Z 旋转，以匹配 PCT 坐标规则。",
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 multifloor PLY 资产转换为 Isaac Sim 可导入的 legacy USD stage。",
    )
    parser.add_argument("--collision-ply", default=os.fspath(DEFAULT_COLLISION_PLY), help="碰撞 PLY 输入路径。")
    parser.add_argument("--visual-ply", default=os.fspath(DEFAULT_VISUAL_PLY), help="视觉 PLY 输入路径。")
    parser.add_argument("--collision-output", default=os.fspath(DEFAULT_COLLISION_USD), help="转换后的碰撞 USD 输出路径。")
    parser.add_argument("--visual-output", default=os.fspath(DEFAULT_VISUAL_USD), help="转换后的视觉 USD 输出路径。")
    parser.add_argument("--visual-reference-usd", default=None, help="已生成的真实/点云视觉 USD；指定后挂载到 visual prim。")
    parser.add_argument("--stage-output", default=os.fspath(DEFAULT_STAGE_USD), help="最终主 USD stage 输出路径。")
    parser.add_argument("--report-output", default=os.fspath(DEFAULT_REPORT_JSON), help="本地构建报告输出路径。")
    parser.add_argument("--collision-prim-path", default="/World/scene_collision", help="主 stage 中的碰撞 prim 路径。")
    parser.add_argument("--visual-prim-path", default="/World/gauss", help="主 stage 中的视觉 prim 路径。")
    parser.add_argument(
        "--collision-visual-visible",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否显示碰撞网格；默认隐藏渲染，但物理碰撞仍保留。",
    )
    parser.add_argument("--include-visual", action="store_true", help="仅当 visual PLY 是普通带 face mesh 时才允许转换。")
    parser.add_argument(
        "--visual-proxy-from-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="未转换 visual PLY 时，用 collision USD 在视觉 prim 下创建可视化代理。",
    )
    parser.add_argument("--visual-proxy-color-r", type=float, default=0.36, help="可视化代理颜色 R。")
    parser.add_argument("--visual-proxy-color-g", type=float, default=0.68, help="可视化代理颜色 G。")
    parser.add_argument("--visual-proxy-color-b", type=float, default=0.92, help="可视化代理颜色 B。")
    parser.add_argument("--visual-proxy-opacity", type=float, default=1.0, help="可视化代理透明度。")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否以 headless 模式启动 Isaac Sim。",
    )
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出 USD。")
    parser.add_argument(
        "--merge-meshes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="转换时尽量合并 mesh。",
    )
    parser.add_argument(
        "--single-mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="转换时禁用实例化并输出单一资产。",
    )
    parser.add_argument(
        "--ignore-materials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="转换 PLY 时忽略材质。",
    )
    parser.add_argument("--collision-approximation", default="none", help="UsdPhysics.MeshCollisionAPI 的 approximation 值。")
    parser.add_argument("--meters-per-unit", type=float, default=1.0, help="主 stage 的 metersPerUnit。")
    parser.add_argument("--rotate-z-deg", type=float, default=180.0, help="挂载 PLY USD 时施加的 Z 轴旋转角度。")
    parser.add_argument("--scale", type=float, default=1.0, help="挂载 PLY USD 时施加的统一缩放。")
    parser.add_argument("--translate-x", type=float, default=0.0, help="挂载 PLY USD 时施加的 X 平移。")
    parser.add_argument("--translate-y", type=float, default=0.0, help="挂载 PLY USD 时施加的 Y 平移。")
    parser.add_argument("--translate-z", type=float, default=0.0, help="挂载 PLY USD 时施加的 Z 平移。")
    parser.add_argument(
        "--add-physics-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在主 stage 中写入 physicsScene。",
    )
    parser.add_argument(
        "--add-inspection-light",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在主 stage 中添加检查用灯光。",
    )
    parser.add_argument("--inspection-light-intensity", type=float, default=800.0, help="检查用灯光强度。")
    parser.add_argument(
        "--add-inspection-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在主 stage 中添加检查用相机。",
    )
    parser.add_argument("--inspection-camera-x", type=float, default=0.0, help="检查相机 X 坐标。")
    parser.add_argument("--inspection-camera-y", type=float, default=-12.0, help="检查相机 Y 坐标。")
    parser.add_argument("--inspection-camera-z", type=float, default=7.0, help="检查相机 Z 坐标。")
    parser.add_argument("--inspection-camera-rot-x", type=float, default=60.0, help="检查相机 X 轴旋转。")
    parser.add_argument("--inspection-camera-rot-y", type=float, default=0.0, help="检查相机 Y 轴旋转。")
    parser.add_argument("--inspection-camera-rot-z", type=float, default=0.0, help="检查相机 Z 轴旋转。")
    parser.add_argument("--inspection-camera-focal-length", type=float, default=24.0, help="检查相机焦距。")
    parser.add_argument("--skip-convert", action="store_true", help="跳过 PLY 转换，只用已有中间 USD 重新生成主 stage。")
    parser.add_argument("--require-real-visual", action="store_true", help="如果没有 --include-visual 或 --visual-reference-usd，则直接报错。")
    parser.add_argument("--skip-report", action="store_true", help="不写本地构建报告。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scale == 0.0:
        raise ValueError("--scale 不能为 0")
    if args.meters_per_unit <= 0.0:
        raise ValueError("--meters-per-unit 必须为正数")
    if not 0.0 <= args.visual_proxy_opacity <= 1.0:
        raise ValueError("--visual-proxy-opacity 必须在 [0, 1] 范围内")
    if args.include_visual and args.visual_reference_usd:
        raise ValueError("--include-visual 和 --visual-reference-usd 不能同时使用")
    if args.require_real_visual and not (args.include_visual or args.visual_reference_usd):
        raise ValueError("最终视频模式需要真实视觉层；请传 --visual-reference-usd 或提供可转换的 --include-visual mesh")
    if args.include_visual and not args.skip_convert:
        visual_ply = _project_path(args.visual_ply)
        _ensure_input(visual_ply, "visual PLY")
        _ensure_visual_ply_mesh_convertible(visual_ply)

    simulation_app = _start_simulation_app(headless=args.headless)
    try:
        if args.skip_convert:
            assets = [
                ConvertedAsset(
                    label="collision",
                    source_ply=_project_path(args.collision_ply),
                    output_usd=_project_path(args.collision_output),
                    prim_path=_prim_path(args.collision_prim_path),
                )
            ]
            if args.include_visual:
                assets.append(
                    ConvertedAsset(
                        label="visual",
                        source_ply=_project_path(args.visual_ply),
                        output_usd=_project_path(args.visual_output),
                        prim_path=_prim_path(args.visual_prim_path),
                    )
                )
            for asset in assets:
                _ensure_input(asset.output_usd, f"{asset.label} USD")
        else:
            assets = _convert_assets(args)

        stage_report = _write_main_stage(args, assets)
        validation = _validate_stage(
            _project_path(args.stage_output),
            _prim_path(args.collision_prim_path),
        )
        visual_validation = _validate_visual_prim(
            _project_path(args.stage_output),
            _prim_path(args.visual_prim_path),
        )
        stage_report["validation"] = validation
        stage_report["visual_validation"] = visual_validation

        report_path = None
        if not args.skip_report:
            report_path = _write_report(args, stage_report)

        print(f"[INFO] 主 USD stage: {_project_path(args.stage_output)}")
        print(
            "[INFO] 碰撞网格: "
            f"{validation['mesh_count']} 个 mesh, {validation['collision_api_count']} 个 CollisionAPI"
        )
        print(
            "[INFO] 可视化层: "
            f"{visual_validation['mesh_count']} 个 mesh, "
            f"{visual_validation['points_count']} 个 Points, "
            f"prim={visual_validation['visual_prim_valid']}"
        )
        if report_path is not None:
            print(f"[INFO] 构建报告: {report_path}")
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
