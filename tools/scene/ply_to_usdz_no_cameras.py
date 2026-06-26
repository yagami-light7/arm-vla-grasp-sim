#!/usr/bin/env python3

"""使用 threedgrut 将 3DGS PLY 导出为 NuRec USDZ，且不导出相机。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from tools.scene.check_multifloor_ply import build_report


DEFAULT_PLY_DIR = PROJECT_ROOT / "source/scene/multifloor/ply"
DEFAULT_OUTPUT_USDZ = PROJECT_ROOT / "source/scene/multifloor/usdz/multifloor.usdz"
DEFAULT_CONFIG_DIR = Path(os.environ.get("THREEDGRUT_CONFIG_DIR", "~/workspace/3dgrut/configs")).expanduser()

LOGGER = logging.getLogger("ply_to_usdz_no_cameras")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    input_ply = _resolve_input_ply(args)
    output_usdz = _project_path(args.output_usdz)
    config_dir = _project_path(args.config_dir)
    output_usdz.parent.mkdir(parents=True, exist_ok=True)

    if input_ply.is_symlink():
        raise RuntimeError(f"输入 PLY 不能是软链接: {input_ply}")
    if output_usdz.exists() and not args.force:
        raise FileExistsError(f"输出已存在，覆盖请传 --force: {output_usdz}")
    if output_usdz.exists():
        output_usdz.unlink()

    _export_usdz(
        input_ply=input_ply,
        output_usdz=output_usdz,
        config_dir=config_dir,
        config_name=args.config_name,
    )
    LOGGER.info("USDZ 已生成: %s", output_usdz)
    return 0


def _export_usdz(*, input_ply: Path, output_usdz: Path, config_dir: Path, config_name: str) -> None:
    try:
        from hydra import compose, initialize_config_dir
        from threedgrut.export import NuRecExporter
        from threedgrut.model.model import MixtureOfGaussians
    except Exception as exc:
        raise RuntimeError(
            "无法导入 threedgrut/hydra。请先激活: conda activate /data/conda_envs/sage"
        ) from exc

    if not config_dir.is_dir():
        raise FileNotFoundError(f"threedgrut config 目录不存在: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=os.fspath(config_dir)):
        conf = compose(config_name=config_name)

    model = MixtureOfGaussians(conf)
    LOGGER.info("读取 3DGS PLY: %s", input_ply)
    model.init_from_ply(os.fspath(input_ply), init_model=False)

    LOGGER.info("导出 NuRec USDZ，不导出相机: %s", output_usdz)
    exporter = NuRecExporter(export_cameras=False)
    exporter.export(model, output_usdz, dataset=None, conf=conf)


def _resolve_input_ply(args: argparse.Namespace) -> Path:
    if args.input_ply:
        path = _project_path(args.input_ply)
        if not path.is_file():
            raise FileNotFoundError(f"输入 PLY 不存在: {path}")
        return path

    report = build_report(_project_path(args.ply_dir))
    selected = report.get("selected_3dgs_visual_ply")
    if not selected:
        raise RuntimeError("无法自动选择 3DGS visual PLY，请先运行 check_multifloor_ply.py")
    return Path(str(selected))


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLY -> NuRec USDZ，不导出相机。")
    parser.add_argument("--input-ply", default=None, help="3DGS visual PLY；默认自动检测。")
    parser.add_argument("--ply-dir", default=os.fspath(DEFAULT_PLY_DIR), help="PLY 输入目录。")
    parser.add_argument("--output-usdz", default=os.fspath(DEFAULT_OUTPUT_USDZ), help="输出 USDZ 路径。")
    parser.add_argument("--config-dir", default=os.fspath(DEFAULT_CONFIG_DIR), help="threedgrut configs 目录。")
    parser.add_argument("--config-name", default="apps/colmap_3dgut.yaml", help="Hydra config 名称。")
    parser.add_argument("--force", action="store_true", help="覆盖已有 USDZ。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
