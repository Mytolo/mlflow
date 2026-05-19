"""Pure helpers for the metric-history benchmark.

Kept in a separate module so the statistics, env capture, RSS sampler, and
artifact writers can be unit-tested without spinning up an MLflow server.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TrialResult:
    """One trial of one scenario."""

    scenario: str
    params: dict[str, Any]
    baseline_rss_bytes: int
    peak_rss_bytes: int
    delta_bytes: int
    duration_seconds: float
    samples: list[tuple[float, int]] = field(default_factory=list)


@dataclass
class ScenarioStats:
    scenario: str
    param_label: str
    param_value: Any
    n_trials: int
    delta_median: int
    delta_p95: int
    delta_min: int
    delta_max: int
    duration_median: float


@dataclass
class VarianceFinding:
    scenario: str
    param_label: str
    param_value: Any
    n_trials: int
    delta_median: int
    delta_p95: int
    spread_pct: float
    breached: bool


def variance_check(stats: list[ScenarioStats], threshold_pct: float) -> list[VarianceFinding]:
    """Per-point trust gate: is delta_p95 within threshold_pct of delta_median?

    spread_pct = (p95 - median) / median * 100. When median is 0 (or near-zero)
    the percent is undefined; we set spread_pct=inf so the renderer can show
    "N/A" instead of a misleading number, and we mark the point as breached
    only if p95 is also non-zero (a flat-zero point is just a noise-floor read,
    not a variance problem).
    """
    findings: list[VarianceFinding] = []
    for s in stats:
        if s.delta_median <= 0:
            spread = math.inf if s.delta_p95 > 0 else 0.0
            breached = s.delta_p95 > 0
        else:
            spread = (s.delta_p95 - s.delta_median) / s.delta_median * 100.0
            breached = spread > threshold_pct
        findings.append(
            VarianceFinding(
                scenario=s.scenario,
                param_label=s.param_label,
                param_value=s.param_value,
                n_trials=s.n_trials,
                delta_median=s.delta_median,
                delta_p95=s.delta_p95,
                spread_pct=spread,
                breached=breached,
            )
        )
    return findings


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile. p in [0, 100]. Empty list -> 0.0."""
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def summarize(trials: list[TrialResult], param_label: str) -> ScenarioStats:
    """Reduce per-trial deltas to summary stats for one (scenario, param) point."""
    if not trials:
        raise ValueError("summarize requires at least one trial")
    deltas = [t.delta_bytes for t in trials]
    durations = [t.duration_seconds for t in trials]
    scenario = trials[0].scenario
    param_value = trials[0].params.get(param_label)
    return ScenarioStats(
        scenario=scenario,
        param_label=param_label,
        param_value=param_value,
        n_trials=len(trials),
        delta_median=int(percentile([float(d) for d in deltas], 50)),
        delta_p95=int(percentile([float(d) for d in deltas], 95)),
        delta_min=min(deltas),
        delta_max=max(deltas),
        duration_median=percentile(durations, 50),
    )


class RSSSampler:
    """Background thread that records (timestamp, rss_bytes) every interval.

    Uses psutil. Constructed lazily so unit tests can exercise the math
    without requiring psutil at import time.
    """

    def __init__(self, pid: int, interval_s: float = 0.05, ceiling_bytes: int | None = None):
        import psutil  # local import; benchmark-only dep

        self._proc = psutil.Process(pid)
        self._interval = interval_s
        self._ceiling = ceiling_bytes
        self._samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ceiling_hit = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import psutil

        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            self._samples.append((time.monotonic() - t0, rss))
            if self._ceiling is not None and rss > self._ceiling:
                self._ceiling_hit = True
                return
            self._stop.wait(self._interval)

    def stop(self) -> list[tuple[float, int]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return list(self._samples)

    @property
    def ceiling_hit(self) -> bool:
        return self._ceiling_hit


def baseline_and_peak(
    samples: list[tuple[float, int]], baseline_window_s: float
) -> tuple[int, int]:
    """Split samples into baseline (first baseline_window_s) and active.

    Baseline = median RSS during the warmup window.
    Peak = max RSS after the warmup window.
    Returns (baseline_bytes, peak_bytes). Falls back to overall min/max if there
    aren't enough samples for the split.
    """
    if not samples:
        return 0, 0
    baseline = [rss for t, rss in samples if t <= baseline_window_s]
    active = [rss for t, rss in samples if t > baseline_window_s]
    if not baseline or not active:
        rsses = [rss for _, rss in samples]
        return min(rsses), max(rsses)
    baseline_rss = int(percentile([float(b) for b in baseline], 50))
    peak_rss = max(active)
    return baseline_rss, peak_rss


def capture_env(server_pid: int | None = None) -> dict[str, Any]:
    """Capture host + tool versions for the report header."""
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import psutil

        env["total_ram_bytes"] = psutil.virtual_memory().total
        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
    except Exception:
        pass

    env["mlflow_version"] = _safe_import_version("mlflow")
    env["sqlite_version"] = _safe_sqlite_version()
    env["repo_git_sha"] = _git_sha(Path(__file__).resolve().parent)
    return env


def _safe_import_version(mod: str) -> str:
    try:
        import importlib

        m = importlib.import_module(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "unknown"


def _safe_sqlite_version() -> str:
    try:
        import sqlite3

        return sqlite3.sqlite_version
    except Exception:
        return "unknown"


def _git_sha(start: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def fmt_bytes(n: int) -> str:
    """Human-readable bytes — used in the markdown table."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    if n == 0:
        return "0 B"
    sign = "-" if n < 0 else ""
    x = abs(float(n))
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{sign}{x:.1f} {u}"
        x /= 1024.0
    return f"{sign}{x:.1f} TiB"


def write_raw_json(
    path: Path,
    env: dict[str, Any],
    params: dict[str, Any],
    trials: list[TrialResult],
    variance_findings: list[VarianceFinding] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "env": env,
        "params": params,
        "trials": [
            {
                "scenario": t.scenario,
                "params": t.params,
                "baseline_rss_bytes": t.baseline_rss_bytes,
                "peak_rss_bytes": t.peak_rss_bytes,
                "delta_bytes": t.delta_bytes,
                "duration_seconds": t.duration_seconds,
                "samples": t.samples,
            }
            for t in trials
        ],
    }
    if variance_findings is not None:
        payload["variance_findings"] = [_finding_to_jsonable(f) for f in variance_findings]
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _finding_to_jsonable(f: VarianceFinding) -> dict[str, Any]:
    d = asdict(f)
    # Strict JSON has no infinity literal; represent as null and let the
    # consumer treat that as "spread undefined (near-zero median)".
    if math.isinf(d["spread_pct"]):
        d["spread_pct"] = None
    return d


def _json_default(o: Any) -> Any:
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(f"Not JSON-serializable: {type(o).__name__}")


_PER_TRIAL_PARAM_KEYS = {"trial_idx", "points_received"}


def _config_only_params(params: dict[str, Any]) -> dict[str, Any]:
    """Project out keys that identify a specific trial rather than a configuration."""
    return {k: v for k, v in params.items() if k not in _PER_TRIAL_PARAM_KEYS}


def write_samples_csv(path: Path, trials: list[TrialResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "trial_index", "params_json", "t_seconds", "rss_bytes"])
        # Group trials by (scenario, config_params) and number them within that
        # group so the consumer can identify repeat trials of the same config.
        seen: dict[tuple[str, str], int] = {}
        for t in trials:
            cfg = _config_only_params(t.params)
            pj = json.dumps(cfg, sort_keys=True)
            key = (t.scenario, pj)
            idx = seen.get(key, 0)
            seen[key] = idx + 1
            for ts, rss in t.samples:
                w.writerow([t.scenario, idx, pj, f"{ts:.4f}", rss])


def auto_interpretation(stats: list[ScenarioStats]) -> str:
    """One short sentence describing what the numbers show, derived from data."""
    if not stats:
        return "No data."
    s = sorted(stats, key=lambda x: x.delta_median)
    lo, hi = s[0], s[-1]
    if lo.delta_median == 0:
        return (
            f"Peak delta ranges {fmt_bytes(lo.delta_median)} to {fmt_bytes(hi.delta_median)} "
            f"across {len(stats)} points."
        )
    ratio = hi.delta_median / max(lo.delta_median, 1)
    return (
        f"Peak delta scales from {fmt_bytes(lo.delta_median)} "
        f"({lo.param_label}={lo.param_value}) "
        f"to {fmt_bytes(hi.delta_median)} ({hi.param_label}={hi.param_value}) — "
        f"{ratio:.1f}x across the swept range."
    )


def _fmt_spread(spread_pct: float) -> str:
    if math.isinf(spread_pct):
        return "N/A — near-zero median"
    return f"{spread_pct:.1f}%"


def render_report(
    env: dict[str, Any],
    params: dict[str, Any],
    reproduce_cmd: str,
    safety: dict[str, Any],
    sections: list[tuple[str, str, list[ScenarioStats]]],
    variance_findings: list[VarianceFinding] | None = None,
    variance_threshold_pct: float | None = None,
) -> str:
    """Build report.md content. Sections is a list of (heading, description, stats)."""
    lines: list[str] = []
    lines.append("# `get_metric_history` server memory benchmark")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for k in sorted(env):
        v = env[k]
        if k.endswith("_bytes") and isinstance(v, int):
            v = fmt_bytes(v)
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```")
    lines.append(reproduce_cmd)
    lines.append("```")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.extend(f"- **{k}**: {params[k]}" for k in sorted(params))
    lines.append("")
    lines.append("## Safety caps in force")
    lines.append("")
    for k in sorted(safety):
        v = safety[k]
        if k.endswith("_bytes") and isinstance(v, int):
            v = fmt_bytes(v)
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    if variance_findings is not None and variance_threshold_pct is not None:
        lines.append(f"## Variance check (threshold {variance_threshold_pct:.1f}%)")
        lines.append("")
        lines.append(
            "Per-point trust gate. Spread = `(delta_p95 - delta_median) / "
            "delta_median * 100`. Rows marked `!` exceed the threshold; see "
            'README §"Reading the variance check section".'
        )
        lines.append("")
        if not variance_findings:
            lines.append("_No data._")
            lines.append("")
        else:
            lines.append(
                "| scenario | param | n_trials | delta_median | delta_p95 | "
                "spread | within_threshold? |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            lines.extend(
                f"| {'!' if f.breached else ''} {f.scenario} "
                f"| {f.param_label}={f.param_value} | {f.n_trials} | "
                f"{fmt_bytes(f.delta_median)} | {fmt_bytes(f.delta_p95)} | "
                f"{_fmt_spread(f.spread_pct)} | "
                f"{'No' if f.breached else 'Yes'} |"
                for f in variance_findings
            )
            lines.append("")
            within = sum(1 for f in variance_findings if not f.breached)
            total = len(variance_findings)
            lines.append(
                f"**{within} of {total} points within ±{variance_threshold_pct:.1f}% "
                "threshold.** Breaches indicate either too few trials or genuine "
                'multi-modal allocation behavior — see README §"Verifying" '
                "before drawing conclusions from the per-scenario tables below."
            )
            lines.append("")
    for heading, description, stats in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(description)
        lines.append("")
        if not stats:
            lines.append("_No data (scenario was skipped or aborted)._")
            lines.append("")
            continue
        param_label = stats[0].param_label
        lines.append(
            f"| {param_label} | n_trials | delta_median | delta_p95 | delta_min | "
            "delta_max | duration_median_s |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        lines.extend(
            f"| {s.param_value} | {s.n_trials} | {fmt_bytes(s.delta_median)} | "
            f"{fmt_bytes(s.delta_p95)} | {fmt_bytes(s.delta_min)} | "
            f"{fmt_bytes(s.delta_max)} | {s.duration_median:.2f} |"
            for s in stats
        )
        lines.append("")
        lines.append(f"**Reading:** {auto_interpretation(stats)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def stats_to_dicts(stats: Iterable[ScenarioStats]) -> list[dict[str, Any]]:
    return [asdict(s) for s in stats]
