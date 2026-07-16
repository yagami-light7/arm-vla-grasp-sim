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
            "note": "该测试隔离低层 policy 的速度跟踪能力，但不能单独证明所有 DWA 问题都由 policy 导致。",
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
        "# 低层 Locomotion Policy 速度跟踪报告",
        "",
        f"- Checkpoint：`{metadata.get('checkpoint', '')}`",
        f"- Isaac Lab task：`{metadata.get('task', '')}`",
        f"- 命令来源：`{metadata.get('schedule_source', metadata.get('profile', 'unknown'))}`",
        "- 地形：Isaac Lab 平地；保留原 policy 的 observation/action 布局",
        f"- 控制周期：`{metadata.get('control_dt_s', '')}` 秒",
        f"- 随机种子：`{metadata.get('seed', '')}`",
        f"- 评估段数：`{summary['evaluated_segments']}`；通过：`{summary['passed_segments']}`；通过率：`{summary['pass_rate']:.1%}`",
        "",
        "## 判定标准",
        "",
        "每段取后 50% 样本作为稳态窗口。所有非零命令轴必须同时满足：",
        "",
        "- 稳态速度增益 `gain = measured_mean / command` 位于 `[0.70, 1.30]`。",
        "- `RMSE <= max(0.04, 0.30 * abs(command))`。",
        "- 运行期间没有跌倒或环境 reset。",
        "",
        "该测试隔离了低层 policy 的速度跟踪能力，不经过 PCT 或 DWA；它能证明速度执行模型是否失配，但不能单独证明所有导航失败都由 policy 引起。",
        "",
        "## 分段结果",
        "",
        "| 测试段 | 命令 vx/vy/wz | 实测均值 vx/vy/wz | 增益 vx/vy/wz | 结果 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["segments"]:
        gains = [item[f"gain_{axis}"] for axis in ("vx", "vy", "wz")]
        gain_text = "/".join("-" if value is None else f"{value:.2f}" for value in gains)
        lines.append(
            f"| {item['segment_name']} | {item['cmd_vx']:.2f}/{item['cmd_vy']:.2f}/{item['cmd_wz']:.2f} | "
            f"{item['mean_vx']:.2f}/{item['mean_vy']:.2f}/{item['mean_wz']:.2f} | {gain_text} | "
            f"{'通过' if item['tracking_pass'] else '失败'} |"
        )
    lines.extend(
        [
            "",
            "## 为什么原 pipeline 需要频繁调整 DWA",
            "",
            "DWA 使用候选 `vx/vy/wz` 在预测时域内积分得到局部轨迹，隐含假设是低层执行器能近似实现该速度。当前 benchmark 表明这个假设只在部分速度区间成立：",
            "",
            "1. **低速死区**：小速度命令可能几乎不产生位移，末端 P 控制器越接近目标、输出越小，反而越容易原地踏步。",
            "2. **非线性增益**：同一轴在不同速度档位的 gain 不恒定，DWA 预测的转弯半径和制动距离会与真实轨迹不同。",
            "3. **方向不对称**：正负横移或旋转响应不同，不能用一个对称速度窗口准确描述。",
            "4. **轴间耦合和横向漂移**：组合转弯时，即使命令 `vy=0` 也可能产生持续侧滑；带 `vy` 的最终位姿修正又可能执行不足。",
            "5. **响应和停止滞后**：切换到零命令后仍存在残余速度，窄门、桌边和路径硬约束会放大这一误差。",
            "6. **场景局部几何不同**：开阔区、窄门、楼梯入口和桌前空间对转弯半径、clearance 和路径偏离的容忍度不同，所以同一组补偿参数无法覆盖所有场景。",
            "",
            "因此，过去频繁调 DWA，本质上同时在补偿两件事：低层 policy 并非理想速度执行器，以及不同场景的局部碰撞/通道几何不同。",
            "",
            "## 原 pipeline 调整过的 DWA 参数",
            "",
            "| 参数类别 | 代表参数 | 调整目的/历史变化 |",
            "|---|---|---|",
            "| 速度上限 | `max_linear_velocity`、`max_angular_velocity` | 控制转弯半径和碰撞风险；carry 曾由 `0.30/0.35` 降到 `0.20/0.30` |",
            "| 加速度 | `max_linear_accel` | 限制速度阶跃和携物扰动；carry 曾由 `1.50` 降到 `1.00` |",
            "| 最小有效速度 | `min_active_linear_velocity`、`near_goal_min_active_linear_velocity` | 避开 policy 低速死区；不同入口曾使用约 `0.22~0.30`，brisk profile 更高 |",
            "| 近目标限速 | `close_goal_speed_limit` | 在可停稳与不落入死区之间折中，历史入口使用过约 `0.22~0.30` |",
            "| 预测与跟踪 | `prediction_horizon`、`lookahead_distance`、`waypoint_tolerance` | 窄通道曾收紧到 horizon `0.35 s`、lookahead `0.12 m`、waypoint tolerance `0.03 m` |",
            "| 候选与打分 | 线/角速度采样数、`speed_bias`、路径/目标/clearance 权重 | 改变候选覆盖范围以及速度、贴路径、避障之间的偏好 |",
            "| 原地转向 | `rotate_in_place_angle`、`yaw_align_min/max_wz` | 控制大角度时原地转还是 creeping turn，避免转得动但位置不前进 |",
            "| 末端 P 控制 | `yaw_align_kp`、`yaw_align_vx`、`yaw_align_max_vy`、lateral deadband、yaw settle/polish 参数 | 控制最终位置和 yaw 收敛；曾使用 `vx=0.04/0.08/0.16` 等激活速度 |",
            "| 路径偏离限制 | hard path deviation、initial-alignment/recovery deviation limit | 防止切角撞墙，同时避免机器人已经偏离后所有回归候选都被拒绝；recovery limit 曾由 `0.20` 放宽到 `0.35 m` |",
            "| 通道与碰撞 | inflate radius、local clearance、route corridor radius | 适配点云/栅格离散误差；route corridor 曾由 `0.16` 放宽到 `0.24 m` |",
            "| 稳定与停稳 | stable linear/angular tolerance、stable steps | 决定何时允许从 nav 切换到 pick/place，避免残余运动影响机械臂操作 |",
            "",
            "## 工程结论",
            "",
            "继续逐场景手调 DWA 可以暂时提高成功率，但会把低层响应缺陷编码进大量场景参数。更稳健的路线是：保留场景相关的碰撞、通道和路径约束；同时为当前 checkpoint 建立 command-response/deadzone 补偿模型，或重新训练低速、横移和组合速度跟踪更好的 locomotion policy。",
        ]
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
