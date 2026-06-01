#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from source.navigation.navlib import OccupancyGridMap, write_ppm


FREE_COLOR = np.array([255, 255, 255], dtype=np.uint8)
RAW_OCCUPIED_COLOR = np.array([25, 25, 25], dtype=np.uint8)
INFLATED_ONLY_COLOR = np.array([242, 140, 40], dtype=np.uint8)
SEPARATOR_COLOR = np.array([210, 210, 210], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize how different inflation radii expand a 2D navigation occupancy map."
    )
    parser.add_argument("--map", required=True, help="Path to a nav-map metadata file (.json/.yaml).")
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        required=True,
        help="One or more inflation radii in meters.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the rendered previews. Defaults next to the nav map metadata.",
    )
    parser.add_argument(
        "--separator-width",
        type=int,
        default=8,
        help="Pixel width of the separator between previews in the combined comparison image.",
    )
    return parser.parse_args()


def _resolve_output_dir(map_path: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return map_path.parent / "inflation_preview"


def _radius_slug(radius_m: float) -> str:
    return f"{radius_m:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _render_overlay(raw_map: OccupancyGridMap, inflated_map: OccupancyGridMap) -> np.ndarray:
    rgb = np.empty((raw_map.height, raw_map.width, 3), dtype=np.uint8)
    rgb[...] = FREE_COLOR

    raw_mask = raw_map.occupancy
    inflated_mask = inflated_map.occupancy
    inflated_only_mask = inflated_mask & ~raw_mask

    rgb[raw_mask] = RAW_OCCUPIED_COLOR
    rgb[inflated_only_mask] = INFLATED_ONLY_COLOR
    return rgb


def _combine_previews(previews: list[np.ndarray], separator_width: int) -> np.ndarray:
    if not previews:
        raise ValueError("at least one preview is required to build a combined image")

    if len(previews) == 1:
        return previews[0]

    separator = np.empty((previews[0].shape[0], separator_width, 3), dtype=np.uint8)
    separator[...] = SEPARATOR_COLOR

    strips: list[np.ndarray] = []
    for index, preview in enumerate(previews):
        if index > 0:
            strips.append(separator)
        strips.append(preview)
    return np.concatenate(strips, axis=1)


def main():
    args = parse_args()
    map_path = Path(args.map).expanduser().resolve()
    raw_map = OccupancyGridMap.from_meta_file(map_path)
    output_dir = _resolve_output_dir(map_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_radii = sorted({max(0.0, float(radius)) for radius in args.radii})
    previews: list[np.ndarray] = []
    manifest: list[dict[str, int | float | str]] = []

    raw_preview = _render_overlay(raw_map, raw_map)
    raw_preview_path = output_dir / "raw_map.ppm"
    write_ppm(raw_preview_path, raw_preview)

    print(f"[INFO] map={map_path}")
    print(f"[INFO] size={raw_map.width}x{raw_map.height} resolution={raw_map.resolution:.3f} m/cell")
    print("[INFO] colors: raw obstacle=dark, inflated-only=orange, free=white")
    print(f"[INFO] wrote raw preview: {raw_preview_path}")

    for radius_m in unique_radii:
        inflated_map = raw_map.inflate(radius_m)
        preview = _render_overlay(raw_map, inflated_map)
        preview_path = output_dir / f"inflate_{_radius_slug(radius_m)}.ppm"
        write_ppm(preview_path, preview)
        previews.append(preview)

        occupied_cells = int(np.count_nonzero(inflated_map.occupancy))
        raw_occupied_cells = int(np.count_nonzero(raw_map.occupancy))
        inflated_only_cells = occupied_cells - raw_occupied_cells
        radius_cells = int(math.ceil(radius_m / raw_map.resolution))
        manifest.append(
            {
                "radius_m": radius_m,
                "radius_cells": radius_cells,
                "occupied_cells": occupied_cells,
                "inflated_only_cells": inflated_only_cells,
                "preview": preview_path.name,
            }
        )
        print(
            f"[INFO] radius={radius_m:.3f} m ({radius_cells} cells) "
            f"occupied={occupied_cells} inflated_only={inflated_only_cells} preview={preview_path.name}"
        )

    comparison = _combine_previews(previews, separator_width=max(0, args.separator_width))
    comparison_path = output_dir / "comparison.ppm"
    write_ppm(comparison_path, comparison)
    print(f"[INFO] wrote comparison: {comparison_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "map": str(map_path),
                "resolution": raw_map.resolution,
                "origin": list(raw_map.origin),
                "raw_preview": raw_preview_path.name,
                "comparison": comparison_path.name,
                "radii": manifest,
            },
            indent=2,
        )
    )
    print(f"[INFO] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
