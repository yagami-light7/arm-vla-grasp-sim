import json

from source.locomotion_benchmark.metrics import _write_markdown, analyze_samples
from source.locomotion_benchmark.schedule import CommandSegment, build_custom_schedule, build_schedule, load_command_file


def test_quick_schedule_has_stops_between_commands():
    schedule = build_schedule("quick", settle_s=1.0, hold_s=2.0, stop_s=0.5, repeats=1)
    assert schedule[0].name == "initial_settle"
    assert len(schedule) == 15
    assert all(not segment.evaluate for segment in schedule[2::2])


def test_metrics_detect_good_and_bad_tracking():
    samples = []
    for name, measured in (("good", 0.24), ("bad", 0.05)):
        for index in range(20):
            samples.append(
                {
                    "segment_name": name,
                    "segment_time_s": 0.02 * (index + 1),
                    "evaluate": True,
                    "cmd_vx": 0.25,
                    "cmd_vy": 0.0,
                    "cmd_wz": 0.0,
                    "measured_vx": measured,
                    "measured_vy": 0.0,
                    "measured_wz": 0.0,
                    "done": False,
                }
            )
    result = analyze_samples(samples)
    by_name = {item["segment_name"]: item for item in result["segments"]}
    assert by_name["good"]["tracking_pass"] is True
    assert by_name["bad"]["tracking_pass"] is False


def test_custom_schedule_supports_user_velocities():
    commands = [CommandSegment("user", 2.0, vx=0.12, vy=-0.03, wz=0.25)]
    schedule = build_custom_schedule(commands, settle_s=1.0, stop_s=0.5, repeats=2)
    assert [segment.name for segment in schedule] == [
        "initial_settle",
        "user_r1",
        "stop_after_user_r1",
        "user_r2",
        "stop_after_user_r2",
    ]
    assert (schedule[1].vx, schedule[1].vy, schedule[1].wz) == (0.12, -0.03, 0.25)


def test_load_custom_command_file(tmp_path):
    path = tmp_path / "commands.json"
    path.write_text(json.dumps({"commands": [{"name": "arc", "vx": 0.2, "wz": 0.3}]}), encoding="utf-8")
    commands = load_command_file(path, default_duration_s=3.5)
    assert commands == [CommandSegment("arc", 3.5, vx=0.2, vy=0.0, wz=0.3)]


def test_markdown_report_is_chinese_and_documents_dwa_tuning(tmp_path):
    samples = []
    for index in range(20):
        samples.append(
            {
                "segment_name": "low_speed_forward",
                "segment_time_s": 0.02 * (index + 1),
                "evaluate": True,
                "cmd_vx": 0.08,
                "cmd_vy": 0.0,
                "cmd_wz": 0.0,
                "measured_vx": 0.01,
                "measured_vy": 0.0,
                "measured_wz": 0.0,
                "done": False,
            }
        )
    report_path = tmp_path / "report.md"
    _write_markdown(
        report_path,
        analyze_samples(samples),
        {
            "checkpoint": "model_26000.pt",
            "task": "flat-ground",
            "schedule_source": "test",
            "control_dt_s": 0.02,
            "seed": 42,
        },
    )
    report = report_path.read_text(encoding="utf-8")
    assert "# 低层 Locomotion Policy 速度跟踪报告" in report
    assert "## 为什么原 pipeline 需要频繁调整 DWA" in report
    assert "`min_active_linear_velocity`" in report
    assert "| low_speed_forward |" in report
    assert "失败" in report
