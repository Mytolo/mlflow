# /// script
# requires-python = ">=3.10"
# dependencies = ["psutil>=5.9,<7", "requests>=2.31,<3"]
# ///
"""Evidence-grade benchmark for `GET /api/2.0/mlflow/metrics/get-history` server memory.

Spawns a local mlflow server against a SQLite backend, logs synthetic metric
data, then drives three parameter scans (page size, history length, concurrent
per-key requests) while sampling the server process-tree RSS. Writes a
self-contained artifact (report.md, raw.json, samples.csv) under --output-dir.

This is a measurement tool, not a stress weapon:
- It refuses to target hosts other than 127.0.0.1 / localhost.
- It aborts if server RSS exceeds --ceiling-bytes (default 4 GiB).
- It aborts on any 5xx response.

Run from the mlflow repo root with:

    uv run python dev/benchmarks/metric_history/run.py --quick

See dev/benchmarks/metric_history/README.md for methodology.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-not-found]
import requests  # type: ignore[import-not-found]

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from measure import (
    ScenarioStats,
    TrialResult,
    baseline_and_peak,
    capture_env,
    render_report,
    summarize,
    variance_check,
    write_raw_json,
    write_samples_csv,
)

ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
LOG_BATCH_SIZE = 1000
INGEST_BATCH_PAUSE_S = 0.0  # no pause; ingest as fast as the server can take it


def _uv_prefix() -> list[str]:
    """Prefix that runs subcommands inside the mlflow repo's uv environment."""
    in_repo = (
        shutil.which("uv")
        and subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, capture_output=True
        ).returncode
        == 0
    )
    return ["uv", "run", "--no-build-isolation"] if in_repo else []


def _wait_for_port(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"server at {url} did not become ready within {timeout}s")


@contextlib.contextmanager
def _start_mlflow(
    work_dir: Path, port: int, workers: int, host: str
) -> Generator[subprocess.Popen, None, None]:
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing to target host {host!r}; allowed: {sorted(ALLOWED_HOSTS)}")
    log_file = work_dir / f"mlflow-{port}.log"
    backend_uri = f"sqlite:///{work_dir / 'mlflow.db'}"
    artifact_uri = (work_dir / "artifacts").as_uri()
    (work_dir / "artifacts").mkdir(exist_ok=True)
    cmd = [
        *_uv_prefix(),
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_uri,
        "--default-artifact-root",
        artifact_uri,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--disable-security-middleware",
    ]
    env = os.environ | {"OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES"}
    with (
        log_file.open("w") as f,
        subprocess.Popen(cmd, stdout=f, stderr=f, env=env, cwd=SCRIPT_DIR) as proc,
    ):
        try:
            _wait_for_port(f"http://{host}:{port}/health")
            yield proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _all_pids(parent_pid: int) -> list[int]:
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return []
    pids = [parent.pid]
    try:
        pids.extend(c.pid for c in parent.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    return pids


def _process_tree_rss(parent_pid: int) -> int:
    total = 0
    for pid in _all_pids(parent_pid):
        try:
            total += psutil.Process(pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


class ProcessTreeSampler:
    """Background sampler that records (t, total_rss) for parent + children."""

    def __init__(self, parent_pid: int, interval_s: float, ceiling_bytes: int | None):
        import threading

        self._parent_pid = parent_pid
        self._interval = interval_s
        self._ceiling = ceiling_bytes
        self._stop = threading.Event()
        self._samples: list[tuple[float, int]] = []
        self._thread: threading.Thread | None = None
        self._ceiling_hit = False

    def start(self) -> None:
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            rss = _process_tree_rss(self._parent_pid)
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
    def samples(self) -> list[tuple[float, int]]:
        return list(self._samples)

    @property
    def ceiling_hit(self) -> bool:
        return self._ceiling_hit


def _api(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/api/2.0/mlflow/{path.lstrip('/')}"


def _post(session: requests.Session, base: str, path: str, body: dict) -> dict:
    r = session.post(_api(base, path), json=body, timeout=120)
    if r.status_code >= 500:
        raise RuntimeError(f"server 5xx on {path}: {r.status_code} {r.text[:500]}")
    r.raise_for_status()
    return r.json()


def _create_run(session: requests.Session, base: str, experiment_id: str) -> str:
    body = {"experiment_id": experiment_id, "start_time": int(time.time() * 1000)}
    return _post(session, base, "runs/create", body)["run"]["info"]["run_id"]


def _create_experiment(session: requests.Session, base: str, name: str) -> str:
    body = {"name": name}
    resp = _post(session, base, "experiments/create", body)
    return resp["experiment_id"]


def _log_history(
    session: requests.Session,
    base: str,
    run_id: str,
    keys: list[str],
    points_per_key: int,
) -> None:
    """Log `points_per_key` metric values for each key on this run."""
    # log-batch caps at 1000 metrics per request server-side.
    for key in keys:
        idx = 0
        while idx < points_per_key:
            chunk = min(LOG_BATCH_SIZE, points_per_key - idx)
            metrics = [
                {
                    "key": key,
                    "value": float(i),
                    "timestamp": int(time.time() * 1000),
                    "step": i,
                }
                for i in range(idx, idx + chunk)
            ]
            _post(session, base, "runs/log-batch", {"run_id": run_id, "metrics": metrics})
            idx += chunk
            if INGEST_BATCH_PAUSE_S:
                time.sleep(INGEST_BATCH_PAUSE_S)


def _fetch_history(
    session: requests.Session, base: str, run_id: str, key: str, max_results: int
) -> int:
    """Fetch one full key's history with the given page size.

    Returns the number of points received across all pages.
    """
    total = 0
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {"run_id": run_id, "metric_key": key, "max_results": max_results}
        if page_token:
            params["page_token"] = page_token
        r = session.get(_api(base, "metrics/get-history"), params=params, timeout=600)
        if r.status_code >= 500:
            raise RuntimeError(f"server 5xx on get-history: {r.status_code} {r.text[:500]}")
        r.raise_for_status()
        body = r.json()
        total += len(body.get("metrics", []))
        page_token = body.get("next_page_token")
        if not page_token:
            break
    return total


def _run_with_sampling(
    parent_pid: int,
    interval_s: float,
    ceiling_bytes: int,
    baseline_window_s: float,
    action,
) -> tuple[ProcessTreeSampler, float, Any]:
    """Idle baseline window, then run ``action()`` while sampling.

    Returns ``(sampler, duration, result)``.
    """
    sampler = ProcessTreeSampler(parent_pid, interval_s, ceiling_bytes)
    sampler.start()
    time.sleep(baseline_window_s)
    t_start = time.monotonic()
    try:
        result = action()
    finally:
        # small grace period so the sampler captures any post-response peak
        time.sleep(0.5)
    duration = time.monotonic() - t_start
    sampler.stop()
    if sampler.ceiling_hit:
        raise RuntimeError("server RSS exceeded ceiling; aborting")
    return sampler, duration, result


# ----------------------------- Scenarios -----------------------------


def scenario_a_page_size(
    args: argparse.Namespace,
    base: str,
    run_id: str,
    key: str,
    samples_meta: dict,
) -> list[TrialResult]:
    """Single key, fixed history, vary max_results."""
    trials: list[TrialResult] = []
    parent_pid = samples_meta["parent_pid"]
    session = requests.Session()
    for max_results in args.page_sizes:
        for trial_idx in range(args.trials):

            def action() -> int:
                return _fetch_history(session, base, run_id, key, max_results)

            sampler, dur, total_fetched = _run_with_sampling(
                parent_pid,
                args.sample_interval_s,
                args.ceiling_bytes,
                args.baseline_window_s,
                action,
            )
            samples = sampler.samples
            baseline_rss, peak_rss = baseline_and_peak(samples, args.baseline_window_s)
            trials.append(
                TrialResult(
                    scenario="A_page_size",
                    params={
                        "max_results": max_results,
                        "history_points": args.points_for_scan_a,
                        "trial_idx": trial_idx,
                        "points_received": total_fetched,
                    },
                    baseline_rss_bytes=baseline_rss,
                    peak_rss_bytes=peak_rss,
                    delta_bytes=max(0, peak_rss - baseline_rss),
                    duration_seconds=dur,
                    samples=samples,
                )
            )
    return trials


def scenario_b_history_length(
    args: argparse.Namespace,
    work_dir: Path,
    port: int,
    samples_meta_holder: dict,
) -> list[TrialResult]:
    """Single key, fixed max_results=25000, vary total history. Re-ingest per point on the curve."""
    trials: list[TrialResult] = []
    base = f"http://{args.host}:{port}"
    for points in args.history_lengths:
        # fresh experiment per point so each trial sees only the data it asked for
        with _start_mlflow(work_dir, port, args.workers, args.host) as proc:
            samples_meta_holder["parent_pid"] = proc.pid
            session = requests.Session()
            exp_id = _create_experiment(session, base, f"bench-B-{points}-{int(time.time())}")
            run_id = _create_run(session, base, exp_id)
            _log_history(session, base, run_id, ["k0"], points)

            for trial_idx in range(args.trials):

                def action() -> int:
                    return _fetch_history(session, base, run_id, "k0", 25000)

                sampler, dur, total_fetched = _run_with_sampling(
                    proc.pid,
                    args.sample_interval_s,
                    args.ceiling_bytes,
                    args.baseline_window_s,
                    action,
                )
                samples = sampler.samples
                baseline_rss, peak_rss = baseline_and_peak(samples, args.baseline_window_s)
                trials.append(
                    TrialResult(
                        scenario="B_history_length",
                        params={
                            "history_points": points,
                            "max_results": 25000,
                            "trial_idx": trial_idx,
                            "points_received": total_fetched,
                        },
                        baseline_rss_bytes=baseline_rss,
                        peak_rss_bytes=peak_rss,
                        delta_bytes=max(0, peak_rss - baseline_rss),
                        duration_seconds=dur,
                        samples=samples,
                    )
                )
    return trials


def scenario_c_concurrent_keys(
    args: argparse.Namespace,
    work_dir: Path,
    port: int,
) -> list[TrialResult]:
    """K keys at points-per-key length: fire K concurrent get-history requests, varying K."""
    trials: list[TrialResult] = []
    base = f"http://{args.host}:{port}"
    # Pre-stage data once. The keys we'll use are k0..k(maxK-1).
    max_k = max(args.concurrencies)
    keys = [f"k{i}" for i in range(max_k)]
    with _start_mlflow(work_dir, port, args.workers, args.host) as proc:
        session = requests.Session()
        exp_id = _create_experiment(session, base, f"bench-C-{int(time.time())}")
        run_id = _create_run(session, base, exp_id)
        _log_history(session, base, run_id, keys, args.points_for_scan_c)

        for k in args.concurrencies:
            for trial_idx in range(args.trials):

                def action() -> int:
                    total = 0
                    with ThreadPoolExecutor(max_workers=k) as pool:
                        futs = [
                            pool.submit(_fetch_history, session, base, run_id, keys[i], 25000)
                            for i in range(k)
                        ]
                        for fut in as_completed(futs):
                            total += fut.result()
                    return total

                sampler, dur, total_fetched = _run_with_sampling(
                    proc.pid,
                    args.sample_interval_s,
                    args.ceiling_bytes,
                    args.baseline_window_s,
                    action,
                )
                samples = sampler.samples
                baseline_rss, peak_rss = baseline_and_peak(samples, args.baseline_window_s)
                trials.append(
                    TrialResult(
                        scenario="C_concurrent_keys",
                        params={
                            "concurrency": k,
                            "points_per_key": args.points_for_scan_c,
                            "max_results": 25000,
                            "trial_idx": trial_idx,
                            "points_received": total_fetched,
                        },
                        baseline_rss_bytes=baseline_rss,
                        peak_rss_bytes=peak_rss,
                        delta_bytes=max(0, peak_rss - baseline_rss),
                        duration_seconds=dur,
                        samples=samples,
                    )
                )
    return trials


# ----------------------------- Driver -----------------------------


def _stats_for(trials: list[TrialResult], param_label: str) -> list[ScenarioStats]:
    by_param: dict[Any, list[TrialResult]] = {}
    for t in trials:
        by_param.setdefault(t.params.get(param_label), []).append(t)
    return [
        summarize(group, param_label) for _, group in sorted(by_param.items(), key=lambda kv: kv[0])
    ]


def _maybe_plot(out_dir: Path, all_stats: dict[str, list[ScenarioStats]]) -> None:
    try:
        import importlib.util

        if importlib.util.find_spec("matplotlib") is None:
            return
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return

    for name, stats in all_stats.items():
        if not stats:
            continue
        xs = [s.param_value for s in stats]
        ys = [s.delta_median for s in stats]
        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel(stats[0].param_label)
        ax.set_ylabel("delta_rss_bytes (median)")
        ax.set_title(name)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="loopback only (127.0.0.1 / localhost)")
    parser.add_argument("--port", type=int, default=5731)
    parser.add_argument("--workers", type=int, default=4, help="gunicorn workers")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[100, 1000, 5000, 25000])
    parser.add_argument(
        "--history-lengths", type=int, nargs="+", default=[1000, 25000, 100000, 250000]
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--points-for-scan-a", type=int, default=100000)
    parser.add_argument("--points-for-scan-c", type=int, default=50000)
    parser.add_argument("--baseline-window-s", type=float, default=10.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--ceiling-bytes",
        type=int,
        default=4 * 1024**3,
        help="abort if RSS exceeds this",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench-out"))
    parser.add_argument(
        "--variance-threshold-pct",
        type=float,
        default=15.0,
        help=(
            "Per-point trust gate: a (scenario, param) is flagged if "
            "(delta_p95 - delta_median) / delta_median * 100 exceeds this. "
            'See README §"Reading the variance check section".'
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="which scenarios to run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny params for smoke-testing the harness (not for evidence)",
    )
    args = parser.parse_args()

    if args.host not in ALLOWED_HOSTS:
        print(
            f"refusing to target host {args.host!r}; allowed: {sorted(ALLOWED_HOSTS)}",
            file=sys.stderr,
        )
        return 2

    if args.quick:
        args.trials = 2
        args.page_sizes = [100, 1000]
        args.history_lengths = [1000, 5000]
        args.concurrencies = [1, 2]
        args.points_for_scan_a = 5000
        args.points_for_scan_c = 5000
        args.baseline_window_s = 2.0

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = args.output_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    reproduce_cmd = "uv run python " + " ".join(shlex.quote(a) for a in sys.argv)

    all_trials: list[TrialResult] = []
    all_stats: dict[str, list[ScenarioStats]] = {
        "A_page_size": [],
        "B_history_length": [],
        "C_concurrent_keys": [],
    }

    with tempfile.TemporaryDirectory(prefix="mh-bench-") as tmp:
        work_dir = Path(tmp)

        if "A" in args.scenarios:
            print("[scenario A] page size scan", flush=True)
            with _start_mlflow(work_dir, args.port, args.workers, args.host) as proc:
                base = f"http://{args.host}:{args.port}"
                session = requests.Session()
                exp_id = _create_experiment(session, base, f"bench-A-{int(time.time())}")
                run_id = _create_run(session, base, exp_id)
                _log_history(session, base, run_id, ["k0"], args.points_for_scan_a)
                trials_a = scenario_a_page_size(args, base, run_id, "k0", {"parent_pid": proc.pid})
            all_trials.extend(trials_a)
            all_stats["A_page_size"] = _stats_for(trials_a, "max_results")

        if "B" in args.scenarios:
            print("[scenario B] history length scan", flush=True)
            trials_b = scenario_b_history_length(args, work_dir, args.port, {})
            all_trials.extend(trials_b)
            all_stats["B_history_length"] = _stats_for(trials_b, "history_points")

        if "C" in args.scenarios:
            print("[scenario C] concurrent per-key scan", flush=True)
            trials_c = scenario_c_concurrent_keys(args, work_dir, args.port)
            all_trials.extend(trials_c)
            all_stats["C_concurrent_keys"] = _stats_for(trials_c, "concurrency")

    env = capture_env()
    params = {
        "trials_per_point": args.trials,
        "workers": args.workers,
        "host": args.host,
        "port": args.port,
        "page_sizes": args.page_sizes,
        "history_lengths": args.history_lengths,
        "concurrencies": args.concurrencies,
        "points_for_scan_a": args.points_for_scan_a,
        "points_for_scan_c": args.points_for_scan_c,
        "baseline_window_s": args.baseline_window_s,
        "sample_interval_s": args.sample_interval_s,
        "scenarios": args.scenarios,
        "quick": args.quick,
    }
    safety = {
        "ceiling_bytes": args.ceiling_bytes,
        "loopback_only": True,
        "abort_on_5xx": True,
        "variance_threshold_pct": args.variance_threshold_pct,
    }
    findings = []
    for stats_list in all_stats.values():
        findings.extend(variance_check(stats_list, args.variance_threshold_pct))
    sections = [
        (
            "Scenario A — page size scan",
            "Single key, fixed history length, vary `max_results` per request.",
            all_stats["A_page_size"],
        ),
        (
            "Scenario B — history length scan",
            "Single key, `max_results=25000`, vary total points logged. Pagination "
            "is expected to keep peak roughly constant.",
            all_stats["B_history_length"],
        ),
        (
            "Scenario C — concurrent per-key scan",
            "K keys at fixed length, fire K concurrent `get-history` requests, vary K.",
            all_stats["C_concurrent_keys"],
        ),
    ]
    report_md = render_report(
        env,
        params,
        reproduce_cmd,
        safety,
        sections,
        variance_findings=findings,
        variance_threshold_pct=args.variance_threshold_pct,
    )
    (out_dir / "report.md").write_text(report_md)
    write_raw_json(out_dir / "raw.json", env, params, all_trials, variance_findings=findings)
    write_samples_csv(out_dir / "samples.csv", all_trials)
    _maybe_plot(out_dir, all_stats)

    print(f"\nReport: {out_dir / 'report.md'}")
    print(f"Raw:    {out_dir / 'raw.json'}")
    print(f"Samples:{out_dir / 'samples.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
