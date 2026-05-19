"""Unit tests for the pure helpers in dev/benchmarks/metric_history/measure.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "dev" / "benchmarks" / "metric_history"))

from measure import (
    ScenarioStats,
    TrialResult,
    auto_interpretation,
    baseline_and_peak,
    capture_env,
    fmt_bytes,
    percentile,
    render_report,
    summarize,
    variance_check,
    write_raw_json,
    write_samples_csv,
)


@pytest.mark.parametrize(
    ("values", "p", "expected"),
    [
        ([], 50, 0.0),
        ([10.0], 50, 10.0),
        ([10.0], 99, 10.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 50, 3.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 0, 1.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 100, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.5),
    ],
)
def test_percentile(values, p, expected):
    assert percentile(values, p) == pytest.approx(expected)


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError, match="percentile p must be in"):
        percentile([1.0, 2.0], -1)
    with pytest.raises(ValueError, match="percentile p must be in"):
        percentile([1.0, 2.0], 101)


def test_baseline_and_peak_splits_on_window():
    # Baseline window 1.0s: samples in [0, 1.0] are baseline; samples > 1.0 are active.
    samples = [(0.0, 100), (0.5, 110), (1.0, 105), (1.5, 500), (2.0, 600), (2.5, 550)]
    baseline, peak = baseline_and_peak(samples, baseline_window_s=1.0)
    assert baseline == 105  # median of [100, 110, 105]
    assert peak == 600  # max of [500, 600, 550]


def test_baseline_and_peak_falls_back_when_no_split():
    samples = [(0.0, 100), (0.1, 200), (0.2, 150)]
    baseline, peak = baseline_and_peak(samples, baseline_window_s=10.0)  # nothing in active
    assert baseline == 100
    assert peak == 200


def test_baseline_and_peak_empty():
    assert baseline_and_peak([], 1.0) == (0, 0)


def test_fmt_bytes():
    assert fmt_bytes(0) == "0 B"
    assert fmt_bytes(1023) == "1023.0 B"
    assert fmt_bytes(1024) == "1.0 KiB"
    assert fmt_bytes(1536) == "1.5 KiB"
    assert fmt_bytes(1024 * 1024 * 5) == "5.0 MiB"
    assert fmt_bytes(-1024).startswith("-")


def _make_trial(scenario: str, params: dict, baseline: int, peak: int) -> TrialResult:
    return TrialResult(
        scenario=scenario,
        params=params,
        baseline_rss_bytes=baseline,
        peak_rss_bytes=peak,
        delta_bytes=max(0, peak - baseline),
        duration_seconds=0.42,
        samples=[(0.0, baseline), (1.0, peak)],
    )


def test_summarize_groups_and_stats():
    trials = [
        _make_trial("A_page_size", {"max_results": 100, "trial_idx": 0}, 100, 200),
        _make_trial("A_page_size", {"max_results": 100, "trial_idx": 1}, 100, 220),
        _make_trial("A_page_size", {"max_results": 100, "trial_idx": 2}, 100, 210),
    ]
    s = summarize(trials, "max_results")
    assert s.scenario == "A_page_size"
    assert s.param_label == "max_results"
    assert s.param_value == 100
    assert s.n_trials == 3
    assert s.delta_min == 100
    assert s.delta_max == 120
    assert s.delta_median == 110


def test_summarize_requires_trials():
    with pytest.raises(ValueError, match="summarize requires at least one trial"):
        summarize([], "max_results")


def test_write_raw_json_roundtrips(tmp_path: Path):
    trials = [
        _make_trial("A_page_size", {"max_results": 100, "trial_idx": 0}, 100, 200),
    ]
    out = tmp_path / "raw.json"
    write_raw_json(out, env={"mlflow_version": "x"}, params={"trials": 1}, trials=trials)
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["env"]["mlflow_version"] == "x"
    assert payload["params"]["trials"] == 1
    assert len(payload["trials"]) == 1
    t = payload["trials"][0]
    assert t["scenario"] == "A_page_size"
    assert t["baseline_rss_bytes"] == 100
    assert t["peak_rss_bytes"] == 200
    assert t["delta_bytes"] == 100
    assert t["samples"] == [[0.0, 100], [1.0, 200]]


def test_write_samples_csv_groups_by_params(tmp_path: Path):
    trials = [
        _make_trial("A_page_size", {"max_results": 100, "trial_idx": 0}, 100, 200),
        _make_trial("A_page_size", {"max_results": 100, "trial_idx": 1}, 100, 210),
        _make_trial("A_page_size", {"max_results": 1000, "trial_idx": 0}, 100, 500),
    ]
    out = tmp_path / "samples.csv"
    write_samples_csv(out, trials)
    with out.open() as f:
        rows = list(csv.reader(f))
    # header + 2 samples per trial * 3 trials = 7
    assert len(rows) == 7
    assert rows[0] == ["scenario", "trial_index", "params_json", "t_seconds", "rss_bytes"]
    # trial_index resets per unique params dict, so two for max_results=100 and one for 1000.
    indices_per_params = {}
    for r in rows[1:]:
        scenario, idx, pj, _, _ = r
        indices_per_params.setdefault(pj, set()).add(idx)
    assert len(indices_per_params) == 2
    counts = sorted(len(v) for v in indices_per_params.values())
    assert counts == [1, 2]


def test_capture_env_has_keys():
    env = capture_env()
    # Don't pin values — just structural integrity.
    for k in ("timestamp_utc", "python_version", "platform", "mlflow_version", "sqlite_version"):
        assert k in env
    assert env["python_version"].count(".") >= 1


def test_auto_interpretation_mentions_extremes():
    stats = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 1000, 1100, 900, 1200, 0.1),
        ScenarioStats("A_page_size", "max_results", 25000, 5, 100000, 110000, 95000, 120000, 0.5),
    ]
    text = auto_interpretation(stats)
    assert "max_results" in text
    assert "100" in text
    assert "25000" in text
    # ratio is at least ~100x
    assert "x" in text.lower()


def test_auto_interpretation_handles_zero_low_end():
    stats = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 0, 0, 0, 0, 0.1),
        ScenarioStats("A_page_size", "max_results", 25000, 5, 10000, 11000, 9500, 12000, 0.5),
    ]
    text = auto_interpretation(stats)
    assert "Peak delta" in text


def test_render_report_produces_tables_and_headers():
    env = {"mlflow_version": "x", "repo_git_sha": "abc123", "total_ram_bytes": 16 * 1024**3}
    params = {"trials_per_point": 5}
    safety = {"ceiling_bytes": 4 * 1024**3, "loopback_only": True}
    stats_a = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 1000, 1100, 900, 1200, 0.1),
        ScenarioStats("A_page_size", "max_results", 1000, 5, 5000, 5500, 4800, 6000, 0.2),
    ]
    sections = [("Scenario A", "Page size scan.", stats_a)]
    out = render_report(env, params, "uv run python run.py", safety, sections)
    assert "# `get_metric_history` server memory benchmark" in out
    assert "## Environment" in out
    assert "abc123" in out
    assert "16.0 GiB" in out  # ram formatted
    assert "## Reproduce" in out
    assert "uv run python run.py" in out
    assert "## Scenario A" in out
    assert "| max_results | n_trials | delta_median |" in out
    assert "**Reading:**" in out


def test_render_report_handles_empty_section():
    out = render_report(
        env={"x": 1},
        params={},
        reproduce_cmd="cmd",
        safety={},
        sections=[("Scenario A", "desc", [])],
    )
    assert "_No data" in out


def test_variance_check_flags_breach():
    # First point: spread 10% -> within threshold. Second: 50% -> breached.
    stats = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 1000, 1100, 900, 1200, 0.1),
        ScenarioStats("A_page_size", "max_results", 25000, 5, 1000, 1500, 800, 1600, 0.5),
    ]
    findings = variance_check(stats, threshold_pct=15.0)
    assert len(findings) == 2
    assert findings[0].spread_pct == pytest.approx(10.0)
    assert findings[0].breached is False
    assert findings[1].spread_pct == pytest.approx(50.0)
    assert findings[1].breached is True


def test_variance_check_handles_zero_median():
    import math

    stats = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 0, 1_000_000, 0, 1_000_000, 0.1),
        ScenarioStats("A_page_size", "max_results", 200, 5, 0, 0, 0, 0, 0.1),
    ]
    findings = variance_check(stats, threshold_pct=15.0)
    assert math.isinf(findings[0].spread_pct)
    assert findings[0].breached is True
    # Flat-zero point: no variance to report and no breach.
    assert findings[1].spread_pct == 0.0
    assert findings[1].breached is False


def test_variance_check_threshold_is_exclusive_at_boundary():
    # Spread is exactly the threshold -> not breached (we use strict >).
    stats = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 1000, 1150, 900, 1200, 0.1),
    ]
    findings = variance_check(stats, threshold_pct=15.0)
    assert findings[0].spread_pct == pytest.approx(15.0)
    assert findings[0].breached is False


def test_render_report_includes_variance_section():
    stats_a = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 1000, 1100, 900, 1200, 0.1),
        ScenarioStats("A_page_size", "max_results", 25000, 5, 1000, 5000, 500, 6000, 0.5),
    ]
    findings = variance_check(stats_a, threshold_pct=15.0)
    out = render_report(
        env={"x": 1},
        params={},
        reproduce_cmd="cmd",
        safety={},
        sections=[("Scenario A", "desc", stats_a)],
        variance_findings=findings,
        variance_threshold_pct=15.0,
    )
    assert "## Variance check (threshold 15.0%)" in out
    assert "within_threshold?" in out
    # Breach marker present somewhere in the table.
    assert "! A_page_size" in out
    # Footer with "X of Y points within ..."
    assert "1 of 2 points within" in out


def test_write_raw_json_includes_variance_findings(tmp_path: Path):
    stats = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 1000, 1500, 900, 1600, 0.1),
    ]
    findings = variance_check(stats, threshold_pct=15.0)
    trials = [
        _make_trial("A_page_size", {"max_results": 100, "trial_idx": 0}, 100, 200),
    ]
    out = tmp_path / "raw.json"
    write_raw_json(out, env={}, params={}, trials=trials, variance_findings=findings)
    payload = json.loads(out.read_text())
    assert "variance_findings" in payload
    assert len(payload["variance_findings"]) == 1
    f = payload["variance_findings"][0]
    assert f["scenario"] == "A_page_size"
    assert f["param_value"] == 100
    assert f["spread_pct"] == pytest.approx(50.0)
    assert f["breached"] is True


def test_write_raw_json_serializes_inf_as_null(tmp_path: Path):
    # spread_pct is inf when median is zero and p95 > 0 — must become null in JSON.
    stats = [
        ScenarioStats("A_page_size", "max_results", 100, 5, 0, 1_000_000, 0, 1_000_000, 0.1),
    ]
    findings = variance_check(stats, threshold_pct=15.0)
    out = tmp_path / "raw.json"
    write_raw_json(out, env={}, params={}, trials=[], variance_findings=findings)
    text = out.read_text()
    # `json.loads` won't accept Infinity in strict mode; assert null is present
    # in the serialized form.
    assert '"spread_pct": null' in text
    payload = json.loads(text)
    assert payload["variance_findings"][0]["spread_pct"] is None
