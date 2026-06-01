#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk


@dataclass
class MapMeta:
    image: str
    resolution: float
    origin: tuple[float, float, float]


@dataclass
class PickedPoint:
    index: int
    row: int
    col: int
    x: float
    y: float
    raw_free: bool
    clearance_free: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "grid": {"row": self.row, "col": self.col},
            "world": {"x": self.x, "y": self.y},
            "raw_free": self.raw_free,
            "clearance_free": self.clearance_free,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click on a nav-map preview image and read back world coordinates.")
    parser.add_argument("--map", required=True, help="Path to nav-map metadata (.json/.yaml).")
    parser.add_argument(
        "--preview",
        default=None,
        help="Preview image to display. Defaults to preview.ppm next to the map metadata.",
    )
    parser.add_argument("--scale", type=int, default=6, help="Integer zoom factor for the display.")
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.25,
        help="Safety clearance in meters used to judge whether a point is comfortably free.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Root directory for episode data. Defaults to <repo>/episodes.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=1,
        help="Task ID subfolder under --dataset-dir.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of episode slots to create (one folder per planned trajectory).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=0.0,
        help="Max random perturbation in world X (m) applied independently to start and goal.",
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=0.0,
        help="Max random perturbation in world Y (m) applied independently to start and goal.",
    )
    parser.add_argument("--start-yaw", type=float, default=None, help="Start pose yaw in radians.")
    parser.add_argument("--goal-yaw", type=float, default=None, help="Goal pose yaw in radians.")
    parser.add_argument("--instruction", type=str, default="", help="Natural language instruction for the episode.")
    parser.add_argument("--object-prim-path", type=str, default="", help="Optional object prim path saved in task.csv.")
    return parser.parse_args()


def load_meta(path: Path) -> MapMeta:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = {}
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value.strip("'\"")
    else:
        raise ValueError(f"Unsupported map metadata extension: {path.suffix}")

    origin = data.get("origin", [0.0, 0.0, 0.0])
    return MapMeta(
        image=str(data["image"]),
        resolution=float(data["resolution"]),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
    )


def load_pgm(path: Path) -> list[list[bool]]:
    with path.open("rb") as fh:
        magic = fh.readline().strip()
        if magic != b"P5":
            raise ValueError("Only binary PGM (P5) occupancy maps are supported.")

        def next_token_line():
            line = fh.readline()
            while line.startswith(b"#"):
                line = fh.readline()
            return line.strip()

        dims = next_token_line()
        while len(dims.split()) < 2:
            dims += b" " + next_token_line()
        width, height = map(int, dims.split())
        _max_value = int(next_token_line())
        raw = fh.read(width * height)
        if len(raw) != width * height:
            raise ValueError("PGM file ended unexpectedly.")
        pixels = list(raw)

    grid: list[list[bool]] = []
    for row in range(height):
        row_values = []
        for col in range(width):
            value = pixels[row * width + col]
            row_values.append(value == 0)
        grid.append(row_values)
    return grid


class GridMap:
    def __init__(self, occupancy: list[list[bool]], resolution: float, origin: tuple[float, float, float]):
        self.occupancy = occupancy
        self.resolution = resolution
        self.origin = origin
        self.height = len(occupancy)
        self.width = len(occupancy[0]) if occupancy else 0

    @classmethod
    def from_meta_file(cls, meta_path: Path) -> "GridMap":
        meta = load_meta(meta_path)
        image_path = Path(meta.image)
        if not image_path.is_absolute():
            image_path = (meta_path.parent / image_path).resolve()
        occupancy = load_pgm(image_path)
        return cls(occupancy=occupancy, resolution=meta.resolution, origin=meta.origin)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return self.occupancy[row][col]

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        local_x = (col + 0.5) * self.resolution
        row_from_bottom = self.height - 1 - row
        local_y = (row_from_bottom + 0.5) * self.resolution
        cos_yaw = math.cos(self.origin[2])
        sin_yaw = math.sin(self.origin[2])
        return (
            self.origin[0] + cos_yaw * local_x - sin_yaw * local_y,
            self.origin[1] + sin_yaw * local_x + cos_yaw * local_y,
        )

    def inflate(self, radius_m: float) -> "GridMap":
        radius_cells = int((radius_m / self.resolution) + 0.999999)
        if radius_cells <= 0:
            return self
        inflated = [[self.occupancy[r][c] for c in range(self.width)] for r in range(self.height)]
        offsets = []
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= radius_cells * radius_cells:
                    offsets.append((dr, dc))
        for row in range(self.height):
            for col in range(self.width):
                if not self.occupancy[row][col]:
                    continue
                for dr, dc in offsets:
                    rr = row + dr
                    cc = col + dc
                    if 0 <= rr < self.height and 0 <= cc < self.width:
                        inflated[rr][cc] = True
        return GridMap(inflated, self.resolution, self.origin)


def default_preview_path(map_path: Path) -> Path:
    preview_path = map_path.parent / "preview.ppm"
    if not preview_path.exists():
        raise FileNotFoundError("preview.ppm not found next to the map metadata; pass --preview explicitly.")
    return preview_path


def main():
    args = parse_args()
    map_path = Path(args.map).expanduser().resolve()
    preview_path = (
        Path(args.preview).expanduser().resolve()
        if args.preview is not None
        else default_preview_path(map_path)
    )

    grid_map = GridMap.from_meta_file(map_path)
    clearance_map = grid_map.inflate(args.clearance)

    root = tk.Tk()
    root.title(f"Nav Point Picker - {preview_path.name}")
    root.resizable(False, False)

    image = tk.PhotoImage(file=str(preview_path))
    display_image = image.zoom(args.scale, args.scale) if args.scale != 1 else image

    canvas = tk.Canvas(root, width=display_image.width(), height=display_image.height(), highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=display_image)

    status_var = tk.StringVar()
    status_var.set("Click start point, then goal point | S: save slots | Right click: undo | C: clear | Q/Esc: quit")
    status = tk.Label(root, textvariable=status_var, anchor="w", justify="left")
    status.pack(fill=tk.X)

    picked_points: list[PickedPoint] = []
    marker_ids: list[tuple[int, int]] = []

    def canvas_to_grid(event_x: int, event_y: int):
        col = event_x // args.scale
        row = event_y // args.scale
        if not grid_map.in_bounds(row, col):
            return None
        return row, col

    def build_point(row: int, col: int) -> PickedPoint:
        x, y = grid_map.grid_to_world(row, col)
        return PickedPoint(
            index=len(picked_points) + 1,
            row=row,
            col=col,
            x=x,
            y=y,
            raw_free=not grid_map.is_occupied(row, col),
            clearance_free=not clearance_map.is_occupied(row, col),
        )

    def redraw_markers():
        for oval_id, text_id in marker_ids:
            canvas.delete(oval_id)
            canvas.delete(text_id)
        marker_ids.clear()
        radius = max(3, args.scale)
        for point in picked_points:
            cx = point.col * args.scale + args.scale / 2
            cy = point.row * args.scale + args.scale / 2
            color = "#2e8b57" if point.clearance_free else "#cc5500"
            oval_id = canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=color, width=2)
            text_id = canvas.create_text(cx + radius + 6, cy, text=str(point.index), fill=color, anchor=tk.W)
            marker_ids.append((oval_id, text_id))

    def save_episode_slots():
        if len(picked_points) < 2:
            status_var.set("[ERROR] Need at least 2 points (click start then goal) before saving.")
            print("[ERROR] Need at least 2 points (start + goal) to save episode slots.")
            return
        start = picked_points[0]
        goal = picked_points[1]

        _script_dir = Path(__file__).resolve().parent
        if args.dataset_dir is not None:
            dataset_root = Path(args.dataset_dir).expanduser().resolve()
        else:
            dataset_root = _script_dir.parent.parent / "episodes"

        task_dir = dataset_root / str(args.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        existing = [int(d.name) for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        next_id = max(existing, default=0) + 1

        created = []
        for i in range(args.count):
            ep_id = next_id + i
            ep_dir = task_dir / str(ep_id)
            ep_dir.mkdir(parents=True, exist_ok=True)
            sx = start.x + random.uniform(-args.dx, args.dx)
            sy = start.y + random.uniform(-args.dy, args.dy)
            gx = goal.x  + random.uniform(-args.dx, args.dx)
            gy = goal.y  + random.uniform(-args.dy, args.dy)
            csv_path = ep_dir / "task.csv"
            _syaw = f"{args.start_yaw:.6f}" if args.start_yaw is not None else ""
            _gyaw = f"{args.goal_yaw:.6f}"  if args.goal_yaw  is not None else ""
            csv_path.write_text(
                "start_x,start_y,goal_x,goal_y,start_yaw,goal_yaw,instruction,object_prim_path\n"
                f"{sx:.6f},{sy:.6f},{gx:.6f},{gy:.6f},{_syaw},{_gyaw},{args.instruction},{args.object_prim_path}\n",
                encoding="utf-8",
            )
            created.append(ep_id)

        msg = f"[INFO] Created {args.count} slot(s) in task {args.task_id}: episodes {created[0]}–{created[-1]} → {task_dir}"
        print(msg)
        status_var.set(msg)

    def on_motion(event):
        grid = canvas_to_grid(event.x, event.y)
        if grid is None:
            status_var.set("Cursor outside map")
            return
        row, col = grid
        x, y = grid_map.grid_to_world(row, col)
        raw_state = "free" if not grid_map.is_occupied(row, col) else "occupied"
        clearance_state = "free" if not clearance_map.is_occupied(row, col) else "occupied"
        status_var.set(
            f"grid=({row}, {col}) world=({x:.3f}, {y:.3f}) raw={raw_state} clearance={clearance_state}"
        )

    def on_left_click(event):
        grid = canvas_to_grid(event.x, event.y)
        if grid is None:
            return
        point = build_point(*grid)
        picked_points.append(point)
        redraw_markers()
        role = "start" if len(picked_points) == 1 else "goal" if len(picked_points) == 2 else f"extra-{len(picked_points)}"
        print(
            f"[POINT] {role}: grid=({point.row}, {point.col}) "
            f"world=({point.x:.3f}, {point.y:.3f}) raw_free={point.raw_free} clearance_free={point.clearance_free}"
        )
        if len(picked_points) == 2:
            status_var.set(f"Start + goal picked. Press S to create {args.count} episode slot(s).")

    def on_right_click(_event):
        if not picked_points:
            return
        removed = picked_points.pop()
        redraw_markers()
        print(
            f"[UNDO] removed point {removed.index}: grid=({removed.row}, {removed.col}) "
            f"world=({removed.x:.3f}, {removed.y:.3f})"
        )

    def clear_points(_event=None):
        if not picked_points:
            return
        picked_points.clear()
        redraw_markers()
        print("[INFO] Cleared all picked points.")

    def save_shortcut(_event=None):
        save_episode_slots()

    def close_window(_event=None):
        root.destroy()

    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Button-1>", on_left_click)
    canvas.bind("<Button-3>", on_right_click)
    root.bind("<KeyPress-c>", clear_points)
    root.bind("<KeyPress-C>", clear_points)
    root.bind("<KeyPress-s>", save_shortcut)
    root.bind("<KeyPress-S>", save_shortcut)
    root.bind("<KeyPress-q>", close_window)
    root.bind("<KeyPress-Q>", close_window)
    root.bind("<Escape>", close_window)
    root.protocol("WM_DELETE_WINDOW", close_window)

    print("[INFO] Nav point picker started.")
    print("[INFO] Click start point, then goal point. Press S to save episode slots.")
    print(f"[INFO] task_id={args.task_id}  count={args.count}  map={map_path}")
    print(f"[INFO] preview={preview_path}")

    root.mainloop()


if __name__ == "__main__":
    main()
