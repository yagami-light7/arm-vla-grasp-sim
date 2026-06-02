from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MapDefinition:
    """Metadata describing an occupancy grid raster on disk."""

    image: str
    resolution: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196
    negate: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapDefinition":
        origin_raw = data.get("origin", (0.0, 0.0, 0.0))
        if len(origin_raw) != 3:
            raise ValueError("map origin must contain exactly 3 values: [x, y, yaw].")
        return cls(
            image=str(data["image"]),
            resolution=float(data["resolution"]),
            origin=(float(origin_raw[0]), float(origin_raw[1]), float(origin_raw[2])),
            occupied_thresh=float(data.get("occupied_thresh", 0.65)),
            free_thresh=float(data.get("free_thresh", 0.196)),
            negate=int(data.get("negate", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "resolution": self.resolution,
            "origin": list(self.origin),
            "occupied_thresh": self.occupied_thresh,
            "free_thresh": self.free_thresh,
            "negate": self.negate,
        }


@dataclass(frozen=True)
class OccupancyGridMap:
    """2D occupancy grid with ROS-style origin semantics."""

    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float, float]
    image_path: Path | None = None
    meta_path: Path | None = None

    def __post_init__(self):
        if self.occupancy.ndim != 2:
            raise ValueError("occupancy grid must be a 2D array.")
        object.__setattr__(self, "occupancy", self.occupancy.astype(bool, copy=False))
        if self.resolution <= 0.0:
            raise ValueError("map resolution must be positive.")

    @property
    def height(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape

    @classmethod
    def from_meta_file(cls, meta_path: str | Path) -> "OccupancyGridMap":
        meta_path = Path(meta_path).expanduser().resolve()
        definition = MapDefinition.from_dict(_load_meta_dict(meta_path))
        image_path = Path(definition.image)
        if not image_path.is_absolute():
            image_path = (meta_path.parent / image_path).resolve()
        occupancy = _load_raster(image_path, definition)
        return cls(
            occupancy=occupancy,
            resolution=definition.resolution,
            origin=definition.origin,
            image_path=image_path,
            meta_path=meta_path,
        )

    def to_meta_dict(self, image_path: str | None = None) -> dict[str, Any]:
        return MapDefinition(
            image=image_path or (self.image_path.name if self.image_path is not None else "occupancy.pgm"),
            resolution=self.resolution,
            origin=self.origin,
        ).to_dict()

    def save_meta_file(self, meta_path: str | Path, image_path: str | None = None):
        meta_path = Path(meta_path)
        data = self.to_meta_dict(image_path=image_path)
        suffix = meta_path.suffix.lower()
        if suffix == ".json":
            meta_path.write_text(json.dumps(data, indent=2))
            return
        if suffix in {".yaml", ".yml"}:
            yaml_lines = [
                f"image: {json.dumps(data['image'])}",
                f"resolution: {data['resolution']}",
                f"origin: {json.dumps(data['origin'])}",
                f"occupied_thresh: {data['occupied_thresh']}",
                f"free_thresh: {data['free_thresh']}",
                f"negate: {data['negate']}",
                "",
            ]
            meta_path.write_text("\n".join(yaml_lines))
            return
        raise ValueError(f"unsupported map metadata extension: {meta_path.suffix}")

    def save_pgm(self, image_path: str | Path):
        image_path = Path(image_path)
        gray = np.where(self.occupancy, 0, 255).astype(np.uint8)
        _write_pgm(image_path, gray)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return bool(self.occupancy[row, col])

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert a world-frame position to a row/col index.

        The origin is interpreted as the world-frame position of the bottom-left cell.
        The first image row corresponds to the top of the map.
        """

        delta_x = x - self.origin[0]
        delta_y = y - self.origin[1]
        cos_yaw = math.cos(self.origin[2])
        sin_yaw = math.sin(self.origin[2])
        local_x = cos_yaw * delta_x + sin_yaw * delta_y
        local_y = -sin_yaw * delta_x + cos_yaw * delta_y
        col = int(math.floor(local_x / self.resolution))
        row_from_bottom = int(math.floor(local_y / self.resolution))
        row = self.height - 1 - row_from_bottom
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        local_x = (float(col) + 0.5) * self.resolution
        row_from_bottom = self.height - 1 - row
        local_y = (float(row_from_bottom) + 0.5) * self.resolution
        cos_yaw = math.cos(self.origin[2])
        sin_yaw = math.sin(self.origin[2])
        return (
            self.origin[0] + cos_yaw * local_x - sin_yaw * local_y,
            self.origin[1] + sin_yaw * local_x + cos_yaw * local_y,
        )

    def inflate(self, radius_m: float) -> "OccupancyGridMap":
        radius_cells = int(math.ceil(radius_m / self.resolution))
        if radius_cells <= 0:
            return self

        offsets: list[tuple[int, int]] = []
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= radius_cells * radius_cells:
                    offsets.append((dr, dc))

        # Unknown space beyond the raster is not navigable. Padding with occupied
        # cells keeps a footprint-sized clearance from map boundaries.
        padded = np.pad(self.occupancy, radius_cells, mode="constant", constant_values=True)
        inflated = self.occupancy.copy()
        for dr, dc in offsets:
            row_start = radius_cells + dr
            col_start = radius_cells + dc
            inflated |= padded[row_start : row_start + self.height, col_start : col_start + self.width]

        return OccupancyGridMap(
            occupancy=inflated,
            resolution=self.resolution,
            origin=self.origin,
            image_path=self.image_path,
            meta_path=self.meta_path,
        )

    def nearest_free_cell(self, start: tuple[int, int], max_radius_cells: int) -> tuple[int, int] | None:
        row, col = start
        if self.in_bounds(row, col) and not self.is_occupied(row, col):
            return start
        for radius in range(1, max_radius_cells + 1):
            row_min = row - radius
            row_max = row + radius
            col_min = col - radius
            col_max = col + radius
            candidates: list[tuple[int, int]] = []
            for r in range(row_min, row_max + 1):
                candidates.append((r, col_min))
                candidates.append((r, col_max))
            for c in range(col_min + 1, col_max):
                candidates.append((row_min, c))
                candidates.append((row_max, c))
            for candidate in candidates:
                if self.in_bounds(*candidate) and not self.is_occupied(*candidate):
                    return candidate
        return None


def _load_meta_dict(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        return _load_simple_yaml(text)
    raise ValueError(f"unsupported map metadata extension: {path.suffix}")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the limited key/value YAML subset used for nav map metadata."""

    data: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"invalid yaml line: {line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = None
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            data[key] = lowered == "true"
            continue
        try:
            data[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            data[key] = value.strip("'\"")
    return data


def _load_raster(image_path: Path, definition: MapDefinition) -> np.ndarray:
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        raster = np.load(image_path)
        if raster.ndim != 2:
            raise ValueError("occupancy .npy files must be 2D arrays.")
        if raster.dtype == np.bool_:
            return raster.astype(bool)
        if np.issubdtype(raster.dtype, np.integer) and raster.min() >= 0 and raster.max() <= 1:
            return raster.astype(bool)
        if np.issubdtype(raster.dtype, np.floating):
            return raster >= 0.5
        return raster.astype(np.int32) > 0
    if suffix == ".pgm":
        gray, max_value = _read_pgm(image_path)
        gray = gray.astype(np.float32) / float(max_value)
        occ_prob = gray if definition.negate else 1.0 - gray
        occupied = occ_prob >= definition.occupied_thresh
        free = occ_prob <= definition.free_thresh
        unknown = ~(occupied | free)
        if unknown.any():
            occupied = occupied | unknown
        return occupied
    raise ValueError(f"unsupported occupancy raster: {image_path.suffix}")


def _read_pgm(path: Path) -> tuple[np.ndarray, int]:
    with path.open("rb") as fh:
        magic = fh.readline().strip()
        if magic not in {b"P2", b"P5"}:
            raise ValueError(f"unsupported PGM magic number: {magic!r}")

        def _next_token() -> bytes:
            while True:
                token = fh.readline()
                if not token:
                    raise ValueError("unexpected EOF while reading PGM.")
                token = token.strip()
                if token and not token.startswith(b"#"):
                    return token

        dims = _next_token().split()
        while len(dims) < 2:
            dims.extend(_next_token().split())
        width, height = int(dims[0]), int(dims[1])
        max_value = int(_next_token())
        if magic == b"P2":
            values = fh.read().split()
            raster = np.array([int(item) for item in values], dtype=np.uint16).reshape(height, width)
        else:
            raw = fh.read(width * height)
            raster = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        return raster, max_value


def _write_pgm(path: Path, gray: np.ndarray):
    if gray.ndim != 2:
        raise ValueError("PGM writer expects a single-channel image.")
    header = f"P5\n{gray.shape[1]} {gray.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(gray.astype(np.uint8).tobytes())
