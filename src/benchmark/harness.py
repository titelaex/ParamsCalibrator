"""
Benchmark harness: runs every calibration method (5 classical baselines +
MLP) through the identical `calibrate_one_node` / `calibrate_two_node`
interface (src/baselines/base.py, src/ml/model.py) on the same synthetic
test runs, and records everything the proposal's five evaluation axes
(Section 7) need in one pass:

  1. accuracy       -- error vs ground truth        -> run_accuracy_speed
  2. speed/latency  -- wall-clock runtime, n_evals   -> run_accuracy_speed
  3. robustness     -- spread across random restarts -> run_robustness
  4. scalability    -- same accuracy/speed, run once on each testbed and
                        compared in src/benchmark/metrics.py
  5. streaming      -- accuracy vs truncated window  -> run_window_sweep

Every function returns one row per (method, run, [restart], [window])
evaluation as plain dicts (assembled into a DataFrame), never
pre-aggregated -- so metrics.py can re-slice or re-aggregate without paying
for the (expensive, real simulator/optimizer) calibrations again.

A calibrator only ever sees I(t), T_ambient and noisy T(t) -- never the
ground-truth constant, which is looked up separately purely to score the
estimate after the fact.
"""

import numpy as np
import pandas as pd

from src.simulator.params import (
    C_HOUSING_J_PER_K,
    C_LUMPED_J_PER_K,
    C_WINDING_J_PER_K,
    R_WINDING_OHM,
)

TARGET_NAMES = {"one_node": ["hA"], "two_node": ["hA", "k_wh"]}


def _true_params(data, testbed, i):
    if testbed == "one_node":
        return {"hA": float(data["hA"][i])}
    return {"hA": float(data["hA"][i]), "k_wh": float(data["k_wh"][i])}


def _call_one_node(fn, data, i, rng, n_samples):
    t = data["t"]
    sl = slice(0, n_samples) if n_samples else slice(None)
    return fn(
        t[sl], data["I"][i][sl], data["T_measured"][i][sl], float(data["T_ambient"][i]),
        R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K, rng=rng,
    )


def _call_two_node(fn, data, i, rng, n_samples):
    t = data["t"]
    sl = slice(0, n_samples) if n_samples else slice(None)
    return fn(
        t[sl], data["I"][i][sl], data["T_w_measured"][i][sl], data["T_h_measured"][i][sl],
        float(data["T_ambient"][i]), R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K,
        C_h=C_HOUSING_J_PER_K, rng=rng,
    )


def _call(fn, testbed, data, i, rng, n_samples=None):
    if testbed == "one_node":
        return _call_one_node(fn, data, i, rng, n_samples)
    return _call_two_node(fn, data, i, rng, n_samples)


def _row(method, testbed, run_idx, true, result, extra=None):
    row = {
        "method": method, "testbed": testbed, "run_idx": run_idx,
        "runtime_s": result.runtime_s, "n_evals": result.n_evals, "converged": result.converged,
    }
    for name in TARGET_NAMES[testbed]:
        true_val = true[name]
        est_val = result.params.get(name, np.nan)
        row[f"true_{name}"] = true_val
        row[f"est_{name}"] = est_val
        row[f"abs_err_{name}"] = abs(est_val - true_val)
        row[f"rel_err_pct_{name}"] = abs(est_val - true_val) / abs(true_val) * 100.0
    if extra:
        row.update(extra)
    return row


def run_accuracy_speed(testbed, data, methods, n_runs=None, seed=0) -> pd.DataFrame:
    """One calibration per (method, run) -- feeds metrics 1 (accuracy) and 2 (speed)."""
    n_total = data["hA"].shape[0]
    n_runs = n_total if n_runs is None else min(n_runs, n_total)

    rows = []
    for method, fns in methods.items():
        fn = fns.get(testbed)
        if fn is None:
            continue
        for i in range(n_runs):
            rng = np.random.default_rng(seed * 100_000 + i)
            true = _true_params(data, testbed, i)
            result = _call(fn, testbed, data, i, rng)
            rows.append(_row(method, testbed, i, true, result))
    return pd.DataFrame(rows)


def run_robustness(testbed, data, methods, run_indices, n_restarts=8, seed=0) -> pd.DataFrame:
    """Multiple restarts (independent rng stream -> independent random init)
    per (method, run) -- feeds metric 3 (convergence robustness): success
    rate and estimate spread, computed later in metrics.py."""
    rows = []
    for method, fns in methods.items():
        fn = fns.get(testbed)
        if fn is None:
            continue
        for i in run_indices:
            true = _true_params(data, testbed, i)
            for r in range(n_restarts):
                rng = np.random.default_rng(seed * 1_000_000 + i * 1000 + r)
                result = _call(fn, testbed, data, i, rng)
                rows.append(_row(method, testbed, i, true, result, extra={"restart": r}))
    return pd.DataFrame(rows)


def run_window_sweep(testbed, data, methods, run_indices, window_fracs, seed=0) -> pd.DataFrame:
    """Same run, truncated to a fraction of its full observation window --
    feeds metric 5 (streaming/real-time suitability). This is a direct stand-in
    for what a live deployment actually has available: not a full, settled
    3000s trace, but however much has arrived so far."""
    n_full = data["t"].shape[0]
    rows = []
    for method, fns in methods.items():
        fn = fns.get(testbed)
        if fn is None:
            continue
        for i in run_indices:
            true = _true_params(data, testbed, i)
            for frac in window_fracs:
                n_samples = max(10, int(frac * n_full))
                rng = np.random.default_rng(seed * 10_000_000 + i * 1000 + int(frac * 1000))
                result = _call(fn, testbed, data, i, rng, n_samples=n_samples)
                rows.append(_row(
                    method, testbed, i, true, result,
                    extra={"window_frac": frac, "n_samples": n_samples},
                ))
    return pd.DataFrame(rows)
