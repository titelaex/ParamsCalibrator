"""
Checks for the benchmark harness (src/benchmark/): registry, harness row
generation, and metrics aggregation.

Every classical baseline is called with drastically reduced search budgets
(tiny GA population/generations, tiny PSO swarm/iterations, tiny BayesOpt
n_calls) purely so this suite runs in seconds rather than minutes -- these
tests check that the harness plumbing is correct (right columns, right
shapes, right dispatch), not that any method reaches production accuracy
with three-particle swarms.

Same standalone-runnable convention as the rest of the suite: `pytest
tests/ -v`, or `python3 tests/test_benchmark.py`.
"""

import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines import bayesopt, ekf, ga, lm, pso
from src.benchmark import metrics
from src.benchmark.harness import run_accuracy_speed, run_robustness, run_window_sweep
from src.benchmark.registry import METHOD_METADATA, build_methods
from src.ml.model import MLPCalibrator
from src.simulator.data_generator import generate_one_node_dataset, generate_two_node_dataset

# Tiny search budgets so GA / PSO / BayesOpt finish in milliseconds, not
# seconds -- only the harness plumbing is under test here.
_FAST_KWARGS = {
    ga: {"pop_size": 6, "n_generations": 3},
    pso: {"n_particles": 6, "n_iters": 3},
    bayesopt: {"n_calls": 8, "n_initial_points": 4},
}


def _fast_methods(mlp_one_node=None, mlp_two_node=None):
    """Same registry as production, but with the stochastic search-based
    baselines wrapped to use the tiny budgets above."""
    methods = build_methods(mlp_one_node=mlp_one_node, mlp_two_node=mlp_two_node)
    for mod, kwargs in _FAST_KWARGS.items():
        name = {ga: "ga", pso: "pso", bayesopt: "bayesopt"}[mod]
        methods[name] = {
            "one_node": functools.partial(mod.calibrate_one_node, **kwargs),
            "two_node": functools.partial(mod.calibrate_two_node, **kwargs),
        }
    return methods


def _one_node_data(n=6, seed=501):
    return generate_one_node_dataset(n_runs=n, seed=seed)


def _two_node_data(n=6, seed=502):
    return generate_two_node_dataset(n_runs=n, seed=seed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_has_all_six_methods_with_metadata():
    methods = build_methods(mlp_one_node=MLPCalibrator(testbed="one_node"))
    assert set(methods) == {"ga", "pso", "lm", "ekf", "bayesopt", "mlp"}
    assert set(METHOD_METADATA) == {"ga", "pso", "lm", "ekf", "bayesopt", "mlp"}
    for name in methods:
        assert "family" in METHOD_METADATA[name]
        assert "stochastic" in METHOD_METADATA[name]
        assert "streaming_capable" in METHOD_METADATA[name]
    assert METHOD_METADATA["ekf"]["streaming_capable"] is True
    assert METHOD_METADATA["mlp"]["stochastic"] is False


def test_registry_without_mlp_omits_it():
    methods = build_methods()
    assert "mlp" not in methods
    assert set(methods) == {"ga", "pso", "lm", "ekf", "bayesopt"}


# ---------------------------------------------------------------------------
# run_accuracy_speed
# ---------------------------------------------------------------------------

def test_accuracy_speed_one_node_schema_and_dispatch():
    data = _one_node_data()
    methods = _fast_methods()
    df = run_accuracy_speed("one_node", data, methods, n_runs=4)

    expected_rows = len(methods) * 4
    assert len(df) == expected_rows
    for col in ("method", "testbed", "run_idx", "runtime_s", "n_evals", "converged",
               "true_hA", "est_hA", "abs_err_hA", "rel_err_pct_hA"):
        assert col in df.columns
    assert set(df["method"]) == set(methods)
    assert (df["testbed"] == "one_node").all()
    assert np.all(np.isfinite(df["rel_err_pct_hA"]))
    assert (df["runtime_s"] >= 0).all()
    assert "k_wh" not in " ".join(df.columns)  # one_node has no second constant


def test_accuracy_speed_two_node_has_both_constants():
    data = _two_node_data()
    methods = _fast_methods()
    df = run_accuracy_speed("two_node", data, methods, n_runs=3)

    for col in ("true_hA", "est_hA", "rel_err_pct_hA", "true_k_wh", "est_k_wh", "rel_err_pct_k_wh"):
        assert col in df.columns
    assert np.all(np.isfinite(df["rel_err_pct_k_wh"]))


def test_mlp_has_zero_simulator_evaluations():
    """The headline speed claim, checked structurally: n_evals must be
    exactly 0 for every MLP row, since a forward pass runs no simulator
    rollouts at all -- unlike every classical baseline."""
    data = _one_node_data()
    mlp = MLPCalibrator.load(os.path.join(os.path.dirname(__file__), "..", "models", "mlp_one_node.joblib")) \
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models", "mlp_one_node.joblib")) \
        else MLPCalibrator(testbed="one_node").fit(*_synthetic_xy())
    methods = {"mlp": {"one_node": mlp.calibrate_one_node, "two_node": None}}
    df = run_accuracy_speed("one_node", data, methods, n_runs=4)
    assert (df["n_evals"] == 0).all()


def _synthetic_xy():
    from src.ml.features import build_one_node_xy
    data = generate_one_node_dataset(n_runs=200, seed=777)
    return build_one_node_xy(data)


def test_n_runs_is_capped_at_dataset_size():
    data = _one_node_data(n=3)
    methods = {"lm": {"one_node": lm.calibrate_one_node, "two_node": None}}
    df = run_accuracy_speed("one_node", data, methods, n_runs=1000)
    assert len(df) == 3  # capped, not padded/errored


# ---------------------------------------------------------------------------
# run_robustness
# ---------------------------------------------------------------------------

def test_robustness_produces_n_runs_times_n_restarts_rows():
    data = _one_node_data()
    methods = _fast_methods()
    n_restarts = 4
    run_indices = [0, 1]
    df = run_robustness("one_node", data, methods, run_indices=run_indices, n_restarts=n_restarts)

    assert len(df) == len(methods) * len(run_indices) * n_restarts
    assert "restart" in df.columns
    assert set(df["restart"]) == set(range(n_restarts))


def test_deterministic_method_has_zero_variance_across_restarts():
    """MLP is not stochastic (METHOD_METADATA), so every restart on the same
    run must produce the exact same estimate -- unlike GA/PSO/LM/BayesOpt,
    whose random initialization can legitimately land differently."""
    data = _one_node_data()
    mlp_path = os.path.join(os.path.dirname(__file__), "..", "models", "mlp_one_node.joblib")
    mlp = MLPCalibrator.load(mlp_path) if os.path.exists(mlp_path) else MLPCalibrator(testbed="one_node").fit(*_synthetic_xy())
    methods = {"mlp": {"one_node": mlp.calibrate_one_node, "two_node": None}}

    df = run_robustness("one_node", data, methods, run_indices=[0], n_restarts=5)
    assert df["est_hA"].nunique() == 1


def test_robustness_table_success_rate_in_unit_interval():
    data = _one_node_data()
    methods = _fast_methods()
    df = run_robustness("one_node", data, methods, run_indices=[0, 1], n_restarts=4)
    table = metrics.robustness_table(df)

    assert "hA_success_rate" in table.columns
    assert (table["hA_success_rate"] >= 0).all() and (table["hA_success_rate"] <= 1).all()
    assert set(table.index) == set(methods)


# ---------------------------------------------------------------------------
# run_window_sweep
# ---------------------------------------------------------------------------

def test_window_sweep_shrinks_n_samples_with_fraction():
    data = _one_node_data()
    methods = {"lm": {"one_node": lm.calibrate_one_node, "two_node": None}}
    fracs = [0.2, 0.5, 1.0]
    df = run_window_sweep("one_node", data, methods, run_indices=[0], window_fracs=fracs)

    assert len(df) == len(fracs)
    n_full = data["t"].shape[0]
    df_sorted = df.sort_values("window_frac")
    assert list(df_sorted["n_samples"]) == sorted(df_sorted["n_samples"])
    assert df_sorted["n_samples"].iloc[-1] == n_full  # frac=1.0 uses the full run


def test_streaming_table_pivots_by_window_frac():
    data = _one_node_data()
    methods = _fast_methods()
    fracs = [0.3, 1.0]
    df = run_window_sweep("one_node", data, methods, run_indices=[0, 1], window_fracs=fracs)
    table = metrics.streaming_table(df, target="hA")

    assert set(table.columns) == set(fracs)
    assert set(table.index) == set(methods)


# ---------------------------------------------------------------------------
# Cross-cutting metrics: scalability, update cost
# ---------------------------------------------------------------------------

def test_scalability_table_ratio_one_when_identical():
    """If accuracy/speed were identical on both testbeds, every ratio must be
    exactly 1.0 -- a basic sanity check on the ratio arithmetic itself."""
    import pandas as pd
    acc = pd.DataFrame({"hA_mape_median": [2.0, 5.0]}, index=["ga", "lm"])
    speed = pd.DataFrame({"runtime_ms_median": [10.0, 1.0]}, index=["ga", "lm"])
    table = metrics.scalability_table(acc, speed, acc, speed)
    np.testing.assert_allclose(table["hA_mape_ratio_2n_over_1n"], [1.0, 1.0])
    np.testing.assert_allclose(table["runtime_ratio_2n_over_1n"], [1.0, 1.0])


def test_update_cost_amortizes_ekf_per_sample():
    """EKF's per-new-sample cost must be its full runtime divided by the
    number of samples it processed -- strictly cheaper than its own full
    refresh cost, and (for any non-trivial run) cheaper than a batch
    method's full refresh cost, which is the whole streaming-suitability
    argument made numeric."""
    data = _one_node_data(n=3)
    methods = {
        "ekf": {"one_node": ekf.calibrate_one_node, "two_node": None},
        "lm": {"one_node": lm.calibrate_one_node, "two_node": None},
    }
    df = run_accuracy_speed("one_node", data, methods, n_runs=3)
    table = metrics.update_cost_table(df)

    assert table.loc["ekf", "per_new_sample_cost_ms"] < table.loc["ekf", "full_refresh_cost_ms"]
    assert table.loc["lm", "per_new_sample_cost_ms"] == table.loc["lm", "full_refresh_cost_ms"]


def _run_all():
    tests = [
        test_registry_has_all_six_methods_with_metadata,
        test_registry_without_mlp_omits_it,
        test_accuracy_speed_one_node_schema_and_dispatch,
        test_accuracy_speed_two_node_has_both_constants,
        test_mlp_has_zero_simulator_evaluations,
        test_n_runs_is_capped_at_dataset_size,
        test_robustness_produces_n_runs_times_n_restarts_rows,
        test_deterministic_method_has_zero_variance_across_restarts,
        test_robustness_table_success_rate_in_unit_interval,
        test_window_sweep_shrinks_n_samples_with_fraction,
        test_streaming_table_pivots_by_window_frac,
        test_scalability_table_ratio_one_when_identical,
        test_update_cost_amortizes_ekf_per_sample,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
