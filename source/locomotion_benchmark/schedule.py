"""Command schedules used by the flat-ground locomotion benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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
