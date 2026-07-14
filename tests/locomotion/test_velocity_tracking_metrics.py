from source.locomotion_benchmark.metrics import analyze_samples
from source.locomotion_benchmark.schedule import build_schedule


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
