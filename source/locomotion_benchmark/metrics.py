"""Metrics, plots, and reports for velocity tracking samples."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


AXES = (("vx", "cmd_vx", "measured_vx"), ("vy", "cmd_vy", "measured_vy"), ("wz", "cmd_wz", "measured_wz"))


def _rmse(errors: Iterable[float]) -> float:
    values = list(errors)
    return math.sqrt(fmean(value * value for value in values)) if values else math.nan


def _settling_time(rows: list[dict[str, Any]], command: float, measured_key: str) -> float | None:
    tolerance = max(0.02, abs(command) * 0.20)
    for index, row in enumerate(rows):
        tail = rows[index:]
        if tail and all(abs(float(item[measured_key]) - command) <= tolerance for item in tail):
            return float(row["segment_time_s"])
    return None


def _rise_time(rows: list[dict[str, Any]], command: float, measured_key: str) -> float | None:
    if abs(command) < 1.0e-6:
        return None
    threshold = 0.90 * abs(command)
    direction = 1.0 if command > 0.0 else -1.0
    for row in rows:
        if direction * float(row[measured_key]) >= threshold:
            return float(row["segment_time_s"])
    return None


def analyze_samples(samples: list[dict[str, Any]], *, steady_fraction: float = 0.50) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["segment_name"])].append(sample)

    segment_metrics: list[dict[str, Any]] = []
    for segment_name, rows in grouped.items():
        if not rows or not bool(rows[0].get("evaluate", False)):
            continue
        steady_start = max(0, min(len(rows) - 1, int(len(rows) * (1.0 - steady_fraction))))
        steady = rows[steady_start:]
        metric: dict[str, Any] = {
            "segment_name": segment_name,
            "samples": len(rows),
            "duration_s": float(rows[-1]["segment_time_s"]),
            "fell_or_reset": any(bool(row.get("done", False)) for row in rows),
        }
        commanded_axis = max(AXES, key=lambda axis: abs(float(rows[0][axis[1]])))
        metric["primary_axis"] = commanded_axis[0]
        for axis, command_key, measured_key in AXES:
            command = float(rows[0][command_key])
            values = [float(row[measured_key]) for row in steady]
            mean_value = fmean(values)
            errors = [value - command for value in values]
            metric[f"cmd_{axis}"] = command
            metric[f"mean_{axis}"] = mean_value
            metric[f"std_{axis}"] = pstdev(values) if len(values) > 1 else 0.0
            metric[f"mae_{axis}"] = fmean(abs(error) for error in errors)
            metric[f"rmse_{axis}"] = _rmse(errors)
            metric[f"gain_{axis}"] = mean_value / command if abs(command) > 1.0e-6 else None
            metric[f"rise_time_{axis}_s"] = _rise_time(rows, command, measured_key)
            metric[f"settling_time_{axis}_s"] = _settling_time(rows, command, measured_key)
        commanded_axes = [axis for axis, _, _ in AXES if abs(float(metric[f"cmd_{axis}"])) > 1.0e-6]
        metric["commanded_axes"] = commanded_axes
        for axis in ("vx", "vy", "wz"):
            command = abs(float(metric[f"cmd_{axis}"]))
            gain = metric[f"gain_{axis}"]
            rmse = float(metric[f"rmse_{axis}"])
            metric[f"tracking_pass_{axis}"] = (
                None
                if command <= 1.0e-6
                else bool(gain is not None and 0.70 <= float(gain) <= 1.30 and rmse <= max(0.04, 0.30 * command))
            )
        metric["tracking_pass"] = bool(
            commanded_axes
            and all(metric[f"tracking_pass_{axis}"] for axis in commanded_axes)
            and not metric["fell_or_reset"]
        )
        segment_metrics.append(metric)

    evaluated = len(segment_metrics)
    passed = sum(bool(item["tracking_pass"]) for item in segment_metrics)
    return {
        "criteria": {
            "steady_window_fraction": steady_fraction,
            "gain_range": [0.70, 1.30],
            "rmse_limit": "max(0.04, 0.30 * abs(command))",
            "note": "This diagnoses open-loop policy tracking; it does not by itself prove DWA causality.",
        },
        "evaluated_segments": evaluated,
        "passed_segments": passed,
        "pass_rate": passed / evaluated if evaluated else 0.0,
        "failed_segments": [str(item["segment_name"]) for item in segment_metrics if not item["tracking_pass"]],
        "segments": segment_metrics,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, samples: list[dict[str, Any]]) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib_not_installed"
    times = [float(row["time_s"]) for row in samples]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for plot_axis, (axis, command_key, measured_key) in zip(axes, AXES):
        plot_axis.plot(times, [float(row[command_key]) for row in samples], "k--", linewidth=1.0, label="command")
        plot_axis.plot(times, [float(row[measured_key]) for row in samples], linewidth=1.0, label="measured")
        plot_axis.set_ylabel(f"{axis} (m/s)" if axis != "wz" else "wz (rad/s)")
        plot_axis.grid(True, alpha=0.3)
        plot_axis.legend(loc="upper right")
    axes[-1].set_xlabel("benchmark time (s)")
    fig.suptitle("Go2-X5 low-level policy velocity tracking on flat ground")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return None


def _write_markdown(path: Path, summary: dict[str, Any], metadata: dict[str, Any]) -> None:
    lines = [
        "# Locomotion policy velocity tracking report",
        "",
        f"- Checkpoint: `{metadata.get('checkpoint', '')}`",
        f"- Task interface: `{metadata.get('task', '')}`",
        "- Terrain: flat plane; original policy observation/action layout retained",
        f"- Control dt: `{metadata.get('control_dt_s', '')}` s",
        f"- Evaluated: `{summary['evaluated_segments']}`; passed: `{summary['passed_segments']}`; pass rate: `{summary['pass_rate']:.1%}`",
        "",
        "The pass flag requires steady-state gain 0.70–1.30 and RMSE <= max(0.04, 30% of command). "
        "The result isolates low-level command tracking from PCT/DWA, but correlation with DWA retuning should be confirmed by replaying actual DWA command traces.",
        "",
        "| segment | command vx/vy/wz | measured vx/vy/wz | gain vx/vy/wz | pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["segments"]:
        gains = [item[f"gain_{axis}"] for axis in ("vx", "vy", "wz")]
        gain_text = "/".join("-" if value is None else f"{value:.2f}" for value in gains)
        lines.append(
            f"| {item['segment_name']} | {item['cmd_vx']:.2f}/{item['cmd_vy']:.2f}/{item['cmd_wz']:.2f} | "
            f"{item['mean_vx']:.2f}/{item['mean_vy']:.2f}/{item['mean_wz']:.2f} | {gain_text} | "
            f"{'yes' if item['tracking_pass'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_artifacts(output_dir: Path, samples: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_samples(samples)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "segment_metrics.csv", summary["segments"])
    plot_error = _write_plot(output_dir / "velocity_tracking.png", samples)
    if plot_error:
        summary["plot_warning"] = plot_error
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "report.md", summary, metadata)
    return summary
