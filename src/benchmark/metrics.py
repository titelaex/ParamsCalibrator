"""
Aggregation of benchmark harness rows (src/benchmark/harness.py) into the
summary tables used for the presentation -- one function per evaluation
axis from the proposal (Section 7). Every function takes the raw per-row
DataFrame a harness function returned and reduces it per method, so the
underlying (expensive) calibrations only ever need to be run once.
"""

import numpy as np
import pandas as pd

# A calibration counts as "successful" if it lands within 20% of the true
# constant. This is deliberately looser than the ~2-5% MAPE a converged run
# typically achieves (see README): it is meant to separate "converged to
# roughly the right answer" from "diverged / stuck in a bad local optimum",
# which is what metric 3 (convergence robustness) is actually about.
SUCCESS_REL_ERR_PCT = 20.0


def _target_cols(df):
    return [c for c in df.columns if c.startswith("rel_err_pct_")]


def accuracy_table(df, testbed=None) -> pd.DataFrame:
    """Metric 1: per-method error vs ground truth (median / mean / P90 MAPE, per target)."""
    if testbed:
        df = df[df["testbed"] == testbed]
    g = df.groupby("method")
    out = {"n": g.size()}
    for col in _target_cols(df):
        name = col.removeprefix("rel_err_pct_")
        out[f"{name}_mape_median"] = g[col].median()
        out[f"{name}_mape_mean"] = g[col].mean()
        out[f"{name}_mape_p90"] = g[col].quantile(0.9)
    return pd.DataFrame(out)


def speed_table(df, testbed=None) -> pd.DataFrame:
    """Metric 2: per-method wall-clock runtime and simulator-evaluation cost."""
    if testbed:
        df = df[df["testbed"] == testbed]
    g = df.groupby("method")
    out = pd.DataFrame({
        "runtime_ms_median": g["runtime_s"].median() * 1000.0,
        "runtime_ms_p90": g["runtime_s"].quantile(0.9) * 1000.0,
        "n_evals_median": g["n_evals"].median(),
    })
    return out.sort_values("runtime_ms_median")


def robustness_table(df_robust, testbed=None) -> pd.DataFrame:
    """Metric 3: success rate and estimate spread across random restarts of the
    same underlying run. Spread is computed only among successful restarts,
    so a method that mostly fails but is precise on its lucky restarts does
    not get to look "robust" -- that is exactly the pattern (proposal,
    Section 7) heuristic/gradient methods are expected to sometimes show."""
    if testbed:
        df_robust = df_robust[df_robust["testbed"] == testbed]
    out = {"n_evaluations": df_robust.groupby("method").size()}
    for col in _target_cols(df_robust):
        name = col.removeprefix("rel_err_pct_")
        success = df_robust[col] < SUCCESS_REL_ERR_PCT
        tagged = df_robust.assign(_success=success)
        out[f"{name}_success_rate"] = tagged.groupby("method")["_success"].mean()
        out[f"{name}_std_when_successful"] = (
            tagged[tagged["_success"]].groupby("method")[col].std()
        )
    return pd.DataFrame(out)


def scalability_table(acc_one, speed_one, acc_two, speed_two) -> pd.DataFrame:
    """Metric 4: degradation from the 1-parameter to the 2-3 parameter
    testbed, for the constant present in both (hA). Ratio > 1 means it got
    worse going from one_node to two_node."""
    methods = [m for m in acc_one.index if m in acc_two.index]
    return pd.DataFrame({
        "hA_mape_ratio_2n_over_1n": (
            acc_two.loc[methods, "hA_mape_median"] / acc_one.loc[methods, "hA_mape_median"].clip(lower=1e-9)
        ),
        "runtime_ratio_2n_over_1n": (
            speed_two.loc[methods, "runtime_ms_median"] / speed_one.loc[methods, "runtime_ms_median"].clip(lower=1e-9)
        ),
    }, index=methods)


def streaming_table(df_sweep, target="hA", testbed=None) -> pd.DataFrame:
    """Metric 5: median MAPE on `target` as a function of observation-window
    fraction -- one row per method, one column per window fraction. This is
    the direct, quantitative version of "how badly does each method degrade
    on partial/streaming data"."""
    if testbed:
        df_sweep = df_sweep[df_sweep["testbed"] == testbed]
    col = f"rel_err_pct_{target}"
    return df_sweep.pivot_table(index="method", columns="window_frac", values=col, aggfunc="median")


def update_cost_table(df_full, testbed=None) -> pd.DataFrame:
    """Per-new-sample amortized update cost: every batch method (GA, PSO, LM,
    BayesOpt, MLP-via-refit-window) must pay its whole calibration runtime
    again to fold in one new sample; EKF instead spreads its total runtime
    over every sample it actually processed. This turns "designed for
    streaming" into a number instead of an assertion."""
    if testbed:
        df_full = df_full[df_full["testbed"] == testbed]
    rows = []
    for method, g in df_full.groupby("method"):
        full_cost_ms = float(g["runtime_s"].median() * 1000.0)
        n_evals_median = float(g["n_evals"].median())
        per_sample_ms = full_cost_ms / n_evals_median if method == "ekf" and n_evals_median > 0 else full_cost_ms
        rows.append({
            "method": method,
            "full_refresh_cost_ms": full_cost_ms,
            "per_new_sample_cost_ms": per_sample_ms,
        })
    return pd.DataFrame(rows).set_index("method")
