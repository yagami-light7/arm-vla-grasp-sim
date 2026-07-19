from __future__ import annotations

from source.diagnostics.performance import WallTimeProfiler


def test_wall_time_profiler_reports_counts_units_and_percentiles() -> None:
    profiler = WallTimeProfiler()
    profiler.record("runtime.render", 0.1, work_units=3)
    profiler.record("runtime.render", 0.3, work_units=3)

    report = profiler.report(seed=7)
    render = report["operations"]["runtime.render"]

    assert report["schema_version"] == "wall_time_profile_v1"
    assert report["seed"] == 7
    assert report["operations_are_non_additive"] is True
    assert render["count"] == 2
    assert render["work_units"] == 6
    assert render["total_seconds"] == 0.4
    assert render["mean_seconds"] == 0.2
    assert render["p50_seconds"] == 0.2
    assert render["p95_seconds"] > render["p50_seconds"]

