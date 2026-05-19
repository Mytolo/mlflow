# `get_metric_history` server memory benchmark

A reproducible measurement harness for how the MLflow tracking server's resident
set size scales with `GET /api/2.0/mlflow/metrics/get-history` page size, total
history length, and concurrent per-key requests. Produces a self-contained
artifact (markdown report, raw JSON, per-sample CSV) intended to be attached to
an upstream issue.

This is a **measurement tool**, not a stress test:

- It refuses to target hosts other than `127.0.0.1` / `localhost`.
- It aborts if the server's resident memory exceeds `--ceiling-bytes` (default 4 GiB).
- It aborts on any 5xx response.

## Running it

From the mlflow repo root, against an unmodified checkout:

```bash
# Smoke test — tiny params, ~30 s, *not* for evidence.
uv run python dev/benchmarks/metric_history/run.py --quick

# Default evidence-grade run — three scenarios, 5 trials each, ~5–15 min
# depending on machine.
uv run python dev/benchmarks/metric_history/run.py
```

CLI flags of note:

| Flag                  | Default                    | Purpose                                                         |
| --------------------- | -------------------------- | --------------------------------------------------------------- |
| `--scenarios`         | `A B C`                    | Run a subset for fast iteration.                                |
| `--trials`            | `5`                        | Trials per (scenario, param) point.                             |
| `--page-sizes`        | `100 1000 5000 25000`      | Scenario A sweep.                                               |
| `--history-lengths`   | `1000 25000 100000 250000` | Scenario B sweep.                                               |
| `--concurrencies`     | `1 4 16 32`                | Scenario C sweep.                                               |
| `--points-for-scan-a` | `100000`                   | History length used for scenarios A.                            |
| `--points-for-scan-c` | `50000`                    | Per-key history length used for scenario C.                     |
| `--workers`           | `4`                        | gunicorn workers (so scenario C can exercise real parallelism). |
| `--baseline-window-s` | `10.0`                     | Idle window before each measured request.                       |
| `--sample-interval-s` | `0.05`                     | RSS sampling cadence.                                           |
| `--ceiling-bytes`     | `4*1024**3`                | Hard abort threshold.                                           |
| `--output-dir`        | `bench-out/`               | Artifact root; a timestamped subdir is created under it.        |

## Methodology

The numbers must be defensible enough for upstream review, so the harness goes
out of its way to remove obvious sources of noise:

1. **One process tree, summed RSS.** Server is started with N gunicorn workers
   (default 4). Per-sample memory reading is `sum of memory_info().rss` across
   parent + all children, via `psutil.Process(pid).children(recursive=True)`.
2. **Baseline subtraction.** Each measured request is preceded by a 10 s idle
   window. The reported `delta_bytes` is `peak_rss_after_window − median_rss_during_window`.
   This is what isolates the per-request allocation from server warm-up,
   import overhead, and post-ingest steady state.
3. **Per-scenario server restart.** Scenarios B and C each spawn a fresh server
   process for each parameter point in their sweep, so allocator fragmentation
   and identity-map retention from one configuration cannot contaminate the
   next.
4. **N=5 trials by default**, reported as median, p95, min, max. A single peak
   reading is dismissible; a tight distribution across trials is not.
5. **Sampler starts after ingestion finishes**, so ingestion-side allocations
   are not in the baseline.
6. **Direct HTTP**, not the Python client. The client always sends
   `max_results=25000` and auto-pages, which would muddy scenario A; the harness
   calls `requests.get(.../metrics/get-history, params={"max_results": ...})`
   so the page size is exactly what we asked for.
7. **Host whitelist** at both CLI parse time and inside the server spawn helper.
   Loopback only.

## What each scenario tells you

- **A — page size scan.** A fixed-length history is fetched with varying
  `max_results`. Peak-RSS delta should grow roughly linearly with page size; if
  it doesn't, the model of the bottleneck is wrong and the rest of the analysis
  needs re-examining.
- **B — history length scan.** Fixed `max_results=25000`, varying total points.
  Because the server paginates server-side, the per-request peak is expected to
  stay roughly constant — bounded by page size, not history length. If it isn't,
  pagination is leaking.
- **C — concurrent per-key scan.** The scenario that matches the original
  reported pain (UI/client fetching many keys at once). K concurrent requests
  against K different keys, summed RSS across the worker pool. This is what
  shows whether real-world concurrent reads exhibit pathological memory growth.

## Artifact layout

The harness writes a timestamped directory under `--output-dir`:

```
bench-out/20260518T180000Z/
├── report.md       # human-readable, env header + tables + auto-interp
├── raw.json        # every per-trial sample, machine-readable
├── samples.csv     # flattened 50ms RSS samples (re-plottable)
└── *.png           # optional, only if matplotlib is available
```

### `raw.json` schema

```jsonc
{
  "schema_version": 1,
  "env": { "mlflow_version": "...", "repo_git_sha": "...", ... },
  "params": { "trials_per_point": 5, "page_sizes": [...], ... },
  "trials": [
    {
      "scenario": "A_page_size" | "B_history_length" | "C_concurrent_keys",
      "params": { /* scenario-specific knobs incl. trial_idx */ },
      "baseline_rss_bytes": 0,
      "peak_rss_bytes": 0,
      "delta_bytes": 0,
      "duration_seconds": 0.0,
      "samples": [[t_seconds, rss_bytes], ...]
    }
  ],
  "variance_findings": [
    {
      "scenario": "A_page_size",
      "param_label": "max_results",
      "param_value": 25000,
      "n_trials": 5,
      "delta_median": 63500000,
      "delta_p95": 79000000,
      "spread_pct": 24.4,                 // or null when delta_median is 0
      "breached": true                    // true iff spread_pct > threshold
    }
  ]
}
```

### `samples.csv` columns

`scenario, trial_index, params_json, t_seconds, rss_bytes` — one row per RSS
sample. Sorted by trial; `params_json` is a stable-key JSON encoding of the
trial's `params` so consumers can group/filter without parsing the schema.

## Interpreting the report

Three scenarios, three questions.

### Scenario A — page size scan

**Question:** Does per-request memory grow with `max_results`?

**What you should see:** at large page sizes (≥5000) the curve is roughly linear
in `max_results` — one big page allocates one big chunk. At small page sizes
(≤1000) the per-request peak can land in the noise floor of background GC and
the other gunicorn workers, _especially_ because retrieving a 100k-point
history at `max_results=100` takes a thousand sequential HTTP requests over
30+ seconds, giving the noise floor lots of opportunity to dominate the peak.
That's a **harness limitation**, not a finding — read the report from the
high-page-size end of the curve.

If the high end of the curve is flat or saturates as `max_results` grows,
something is wrong with the model in this README and the rest of the report's
conclusions need to be revisited.

### Scenario B — history length scan

**Question:** With `max_results=25000`, does total-history-length matter?

**What you should see:** per-request peak bounded by page size, not by total
history. Going from 25k to 250k points should not 10× the peak. If it stays
flat: pagination is paging server-side and there's no leak across pages. If it
grows linearly with total history: server-side pagination is leaking and the
upstream writeup should call that out specifically.

### Scenario C — concurrent per-key scan

**Question:** When K different metric keys are fetched in parallel (the UI does
exactly this) how does total RSS scale with K?

**What you should see:** roughly K-linear growth in total RSS — each concurrent
request multiplies the per-request peak by the worker pool's ability to handle
it in parallel.

**The number a maintainer cares about most is `delta_p95` at high K**, not the
median. The p95 is the worst plausible peak under burst, which is what causes
OOMs in production. A 200 MiB median with a 500 MiB p95 means roughly 1 in 20
bursts will allocate half a gibibyte of additional RSS — that's the risk model
to communicate upstream.

## Tuning `--trials`

`--trials N` runs each (scenario, parameter) point N independent times so the
report's `delta_median`, `delta_p95`, `delta_min`, `delta_max` reflect a
distribution, not a single number. Default is 5.

The variance check section in `report.md` tells you whether the trial count
was enough. Recipe:

1. Run with defaults (`--trials 5`).
2. Open `report.md` and read the `## Variance check` table. If every row is
   `within_threshold? Yes`, the numbers are stable enough — you're done.
3. If any row you care about reads `No`, re-run just that scenario with more
   trials:

   ```
   uv run python dev/benchmarks/metric_history/run.py --scenarios C --trials 20
   ```

4. If the variance check still flags after `--trials 20` or higher, **the
   variance is real, not measurement noise.** This usually indicates
   multi-modal allocation behavior (e.g. allocator fragmentation, GC timing
   relative to the request) and is itself a finding worth reporting upstream.
   Document the spread in the writeup — don't try to wash it out by cranking
   trials until p95 happens to land near median.

**Other knobs that reduce noise** before more trials:

- Close other apps (browsers, IDEs) on the test machine.
- Plug the laptop in to avoid thermal throttling mid-run.
- On Linux: `taskset -c 0-3 uv run python …` to pin the harness to specific
  cores.
- Increase `--baseline-window-s` from the default 10 s — a longer idle window
  gives a tighter median for the baseline subtraction.

## Reading the variance check section

The harness writes a `## Variance check (threshold N%)` section near the top
of every `report.md`. Each row reports one (scenario, parameter) point:

```
| scenario | param | n_trials | delta_median | delta_p95 | spread | within_threshold? |
| ! C_concurrent_keys | concurrency=32 | 5 | 204.3 MiB | 504.1 MiB | 146.7% | No |
```

- **`spread`** is `(delta_p95 - delta_median) / delta_median × 100`. It tells
  you how much the worst plausible trial exceeded the typical trial.
- **`within_threshold?`** is `Yes` iff `spread ≤ --variance-threshold-pct`
  (default 15%).
- A `!` at the start of the row marks a breach for quick scanning.
- When `delta_median` is near zero, `spread` is shown as `N/A — near-zero median` and recorded as `null` in `raw.json` (percent-of-zero is
  undefined). The row is treated as breached if `delta_p95 > 0`.

**The variance check is a trust gate on the report itself, not a finding about
mlflow.** If the report's own numbers don't reproduce across trials on the
same code and same machine, the rest of the report's conclusions about
mlflow's behavior are weaker. Use it to decide whether to publish the report
as-is or re-run with more trials first.

Tune the threshold with `--variance-threshold-pct` if you have a defensible
reason — e.g. an upstream conversation that already accepts ±25% as the bar
for this kind of measurement. The threshold appears in the `## Safety caps in force` block of every report so reviewers know what gate the numbers
nominally passed.

## Verifying the harness is producing trustworthy numbers

Before treating any single run as evidence:

1. Look at `## Variance check` in `report.md`. Every row that matters should
   be `within_threshold? Yes`. If not: follow §"Tuning `--trials`".
2. Run the full harness twice. Compare `delta_median` for the same
   (scenario, param) point across the two `raw.json` files; they should match
   within the same threshold. If they don't, machine state between runs is
   leaking in (other processes, thermal state, network).
3. Confirm scenario A's curve is roughly linear in `max_results` _at the high
   end_. If it's flat or saturates there, the model is wrong and the report's
   conclusions need to be revisited.
4. Inspect `samples.csv` for any trial — the RSS trace should show the
   baseline window flat, then a clear ramp up and back down during the request
   window. If it doesn't, something else on the host is allocating during
   measurement.

## Limitations to disclose in any upstream writeup

- macOS RSS reporting is shared-memory-aware in non-intuitive ways. The
  harness also captures `memory_full_info().uss` where available; the report
  uses RSS for compatibility, but you can recompute USS from `raw.json` if you
  want a tighter bound.
- A single laptop is not a controlled environment. Each `report.md` includes
  the host's CPU/RAM in the env header so a reviewer can reproduce on
  comparable hardware.
- Numbers measure the _server_ process tree, not the client. Client-side
  growth (which exists too — see `MlflowClient.get_metric_history` accumulating
  all pages into one list) is out of scope for this benchmark.

## Tests

Pure-function helpers in `measure.py` (percentile, baseline/peak split, JSON/CSV
writers, env capture) are covered by unit tests in
`tests/dev_benchmarks/metric_history/test_measure.py`. Run them with:

```bash
uv run pytest tests/dev_benchmarks/metric_history/
```
