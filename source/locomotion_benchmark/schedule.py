"""Command schedules used by the flat-ground locomotion benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandSegment:
    name: str
    duration_s: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    evaluate: bool = True

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


QUICK_COMMANDS = (
    ("vx_pos_010", 0.10, 0.0, 0.0),
    ("vx_pos_025", 0.25, 0.0, 0.0),
    ("vy_pos_010", 0.0, 0.10, 0.0),
    ("vy_neg_010", 0.0, -0.10, 0.0),
    ("wz_pos_020", 0.0, 0.0, 0.20),
    ("wz_neg_040", 0.0, 0.0, -0.40),
    ("terminal_mix", 0.16, 0.08, 0.30),
)

FULL_COMMANDS = (
    ("vx_pos_005", 0.05, 0.0, 0.0),
    ("vx_pos_010", 0.10, 0.0, 0.0),
    ("vx_pos_016", 0.16, 0.0, 0.0),
    ("vx_pos_025", 0.25, 0.0, 0.0),
    ("vx_pos_040", 0.40, 0.0, 0.0),
    ("vx_neg_010", -0.10, 0.0, 0.0),
    ("vx_neg_025", -0.25, 0.0, 0.0),
    ("vy_pos_005", 0.0, 0.05, 0.0),
    ("vy_pos_010", 0.0, 0.10, 0.0),
    ("vy_pos_020", 0.0, 0.20, 0.0),
    ("vy_neg_010", 0.0, -0.10, 0.0),
    ("vy_neg_020", 0.0, -0.20, 0.0),
    ("wz_pos_010", 0.0, 0.0, 0.10),
    ("wz_pos_020", 0.0, 0.0, 0.20),
    ("wz_pos_040", 0.0, 0.0, 0.40),
    ("wz_pos_060", 0.0, 0.0, 0.60),
    ("wz_neg_020", 0.0, 0.0, -0.20),
    ("wz_neg_040", 0.0, 0.0, -0.40),
    ("terminal_mix", 0.16, 0.08, 0.30),
    ("dwa_arc", 0.25, 0.0, 0.30),
)


def build_schedule(
    profile: str,
    *,
    settle_s: float,
    hold_s: float,
    stop_s: float,
    repeats: int,
) -> list[CommandSegment]:
    if profile not in {"quick", "full"}:
        raise ValueError(f"unsupported profile: {profile}")
    if min(settle_s, hold_s, stop_s) <= 0.0 or repeats <= 0:
        raise ValueError("durations and repeats must be positive")
    commands = QUICK_COMMANDS if profile == "quick" else FULL_COMMANDS
    segments = [CommandSegment("initial_settle", settle_s, evaluate=False)]
    for repeat in range(repeats):
        for name, vx, vy, wz in commands:
            suffix = f"r{repeat + 1}"
            segments.append(CommandSegment(f"{name}_{suffix}", hold_s, vx, vy, wz))
            segments.append(CommandSegment(f"stop_after_{name}_{suffix}", stop_s, evaluate=False))
    return segments


def _command_from_mapping(item: dict[str, Any], index: int, default_duration_s: float) -> CommandSegment:
    unknown = set(item) - {"name", "duration_s", "vx", "vy", "wz", "evaluate"}
    if unknown:
        raise ValueError(f"command {index} has unsupported fields: {sorted(unknown)}")
    duration_s = float(item.get("duration_s", default_duration_s))
    if duration_s <= 0.0:
        raise ValueError(f"command {index} duration_s must be positive")
    return CommandSegment(
        name=str(item.get("name", f"custom_{index:03d}")),
        duration_s=duration_s,
        vx=float(item.get("vx", 0.0)),
        vy=float(item.get("vy", 0.0)),
        wz=float(item.get("wz", 0.0)),
        evaluate=bool(item.get("evaluate", True)),
    )


def load_command_file(path: str | Path, *, default_duration_s: float) -> list[CommandSegment]:
    """Load a JSON command list, optionally wrapped in a top-level ``commands`` object."""

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    commands = payload.get("commands") if isinstance(payload, dict) else payload
    if not isinstance(commands, list) or not commands:
        raise ValueError("commands JSON must contain a non-empty list")
    if not all(isinstance(item, dict) for item in commands):
        raise ValueError("each custom command must be a JSON object")
    return [_command_from_mapping(item, index + 1, default_duration_s) for index, item in enumerate(commands)]


def build_custom_schedule(
    commands: list[CommandSegment],
    *,
    settle_s: float,
    stop_s: float,
    repeats: int,
) -> list[CommandSegment]:
    if not commands:
        raise ValueError("at least one custom command is required")
    if settle_s <= 0.0 or stop_s < 0.0 or repeats <= 0:
        raise ValueError("settle_s/repeats must be positive and stop_s must be non-negative")
    segments = [CommandSegment("initial_settle", settle_s, evaluate=False)]
    for repeat in range(repeats):
        for command in commands:
            suffix = f"r{repeat + 1}"
            segments.append(
                CommandSegment(
                    name=f"{command.name}_{suffix}",
                    duration_s=command.duration_s,
                    vx=command.vx,
                    vy=command.vy,
                    wz=command.wz,
                    evaluate=command.evaluate,
                )
            )
            if stop_s > 0.0:
                segments.append(CommandSegment(f"stop_after_{command.name}_{suffix}", stop_s, evaluate=False))
    return segments
