"""
Sanity checks for the classical calibration baselines (GA, PSO, LM, EKF,
Bayesian Optimization).

Same standalone-runnable convention as tests/test_simulator.py: pytest is
preferred (`pytest tests/ -v`) but `python3 tests/test_baselines.py` also
works without it.

These are not benchmark-accuracy tests (that is the job of src/benchmark/,
still WIP) -- they only check that each method actually recovers the known
ground-truth constant(s), within a loose tolerance, on a low-noise
synthetic run. Search budgets are reduced from each module's defaults where
needed to keep the suite fast; this trades tightness of the tolerance for
runtime, not correctness of the check.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines import bayesopt, ekf, ga, lm, pso
from src.baselines.base import CalibrationResult
from src.simulator.data_generator import time_grid
from src.simulator.load_profiles import generate_load_profile
from src.simulator.motor_thermal import simulate_one_node, simulate_two_node
from src.simulator.params import OneNodeParams, TwoNodeParams
from src.simulator.sensors import add_noise

# Shared low-noise 1-node fixture: every batch method should land close to
# hA_true, since with little noise the SSE landscape has one clean minimum.
_HA_TRUE = 15.0
_T_AMBIENT = 25.0
_NOISE_STD = 0.3


def _one_node_fixture(seed=100, profile="step"):
    rng = np.random.default_rng(seed)
    t = time_grid()
    I_t, _ = generate_load_profile(profile, t, rng)
    params = OneNodeParams(hA=_HA_TRUE, T_ambient=_T_AMBIENT)
    T_true = simulate_one_node(params, t, I_t, T0=_T_AMBIENT)
    T_measured = add_noise(T_true, _NOISE_STD, rng)
    return t, I_t, T_measured


_HA2_TRUE, _KWH_TRUE = 15.0, 35.0


def _two_node_fixture(seed=200, profile="duty_cycle"):
    rng = np.random.default_rng(seed)
    t = time_grid()
    I_t, _ = generate_load_profile(profile, t, rng)
    params = TwoNodeParams(hA=_HA2_TRUE, k_wh=_KWH_TRUE, T_ambient=_T_AMBIENT)
    T_true = simulate_two_node(params, t, I_t, T0=np.array([_T_AMBIENT, _T_AMBIENT]))
    T_w_measured = add_noise(T_true[:, 0], _NOISE_STD, rng)
    T_h_measured = add_noise(T_true[:, 1], _NOISE_STD, rng)
    return t, I_t, T_w_measured, T_h_measured


def _assert_result_shape(result, expected_keys):
    assert isinstance(result, CalibrationResult)
    assert set(result.params.keys()) == set(expected_keys)
    assert result.runtime_s >= 0.0
    assert result.n_evals > 0
    for v in result.params.values():
        assert np.isfinite(v)


# ---------------------------------------------------------------------------
# 1-node: each method should recover hA within ~10% on a low-noise run
# ---------------------------------------------------------------------------

def test_lm_recovers_hA_one_node():
    t, I_t, T_measured = _one_node_fixture()
    result = lm.calibrate_one_node(t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1))
    _assert_result_shape(result, ["hA"])
    assert abs(result.params["hA"] - _HA_TRUE) / _HA_TRUE < 0.1
    assert result.converged


def test_ga_recovers_hA_one_node():
    t, I_t, T_measured = _one_node_fixture()
    result = ga.calibrate_one_node(t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1))
    _assert_result_shape(result, ["hA"])
    assert abs(result.params["hA"] - _HA_TRUE) / _HA_TRUE < 0.1


def test_pso_recovers_hA_one_node():
    t, I_t, T_measured = _one_node_fixture()
    result = pso.calibrate_one_node(t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1))
    _assert_result_shape(result, ["hA"])
    assert abs(result.params["hA"] - _HA_TRUE) / _HA_TRUE < 0.1


def test_bayesopt_recovers_hA_one_node():
    t, I_t, T_measured = _one_node_fixture()
    # reduced budget to keep the suite fast; BayesOpt is designed to be
    # sample-efficient, so it should still land close with fewer calls
    result = bayesopt.calibrate_one_node(
        t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1),
        n_calls=20, n_initial_points=8,
    )
    _assert_result_shape(result, ["hA"])
    assert abs(result.params["hA"] - _HA_TRUE) / _HA_TRUE < 0.1


def test_ekf_recovers_hA_one_node():
    t, I_t, T_measured = _one_node_fixture()
    result = ekf.calibrate_one_node(
        t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1),
        measurement_noise_std=_NOISE_STD,
    )
    _assert_result_shape(result, ["hA"])
    assert abs(result.params["hA"] - _HA_TRUE) / _HA_TRUE < 0.1
    # the running estimate should end up much closer to truth than it started
    assert abs(result.history[-1] - _HA_TRUE) < abs(result.history[0] - _HA_TRUE)


# ---------------------------------------------------------------------------
# 2-node: both hA and k_wh should be recovered simultaneously (scalability)
# ---------------------------------------------------------------------------

def test_lm_recovers_params_two_node():
    t, I_t, T_w, T_h = _two_node_fixture()
    result = lm.calibrate_two_node(t, I_t, T_w, T_h, _T_AMBIENT, rng=np.random.default_rng(2))
    _assert_result_shape(result, ["hA", "k_wh"])
    assert abs(result.params["hA"] - _HA2_TRUE) / _HA2_TRUE < 0.15
    assert abs(result.params["k_wh"] - _KWH_TRUE) / _KWH_TRUE < 0.15


def test_ekf_recovers_params_two_node():
    t, I_t, T_w, T_h = _two_node_fixture()
    result = ekf.calibrate_two_node(
        t, I_t, T_w, T_h, _T_AMBIENT, rng=np.random.default_rng(2),
        measurement_noise_std=_NOISE_STD,
    )
    _assert_result_shape(result, ["hA", "k_wh"])
    assert abs(result.params["hA"] - _HA2_TRUE) / _HA2_TRUE < 0.15
    assert abs(result.params["k_wh"] - _KWH_TRUE) / _KWH_TRUE < 0.15


def test_ga_recovers_params_two_node():
    t, I_t, T_w, T_h = _two_node_fixture()
    # reduced budget for test speed; the 2-node objective is ~2x the cost
    result = ga.calibrate_two_node(
        t, I_t, T_w, T_h, _T_AMBIENT, rng=np.random.default_rng(2),
        pop_size=16, n_generations=15,
    )
    _assert_result_shape(result, ["hA", "k_wh"])
    assert abs(result.params["hA"] - _HA2_TRUE) / _HA2_TRUE < 0.2
    assert abs(result.params["k_wh"] - _KWH_TRUE) / _KWH_TRUE < 0.2


def test_pso_recovers_params_two_node():
    t, I_t, T_w, T_h = _two_node_fixture()
    result = pso.calibrate_two_node(
        t, I_t, T_w, T_h, _T_AMBIENT, rng=np.random.default_rng(2),
        n_particles=14, n_iters=15,
    )
    _assert_result_shape(result, ["hA", "k_wh"])
    assert abs(result.params["hA"] - _HA2_TRUE) / _HA2_TRUE < 0.2
    assert abs(result.params["k_wh"] - _KWH_TRUE) / _KWH_TRUE < 0.2


def test_bayesopt_recovers_params_two_node():
    t, I_t, T_w, T_h = _two_node_fixture()
    result = bayesopt.calibrate_two_node(
        t, I_t, T_w, T_h, _T_AMBIENT, rng=np.random.default_rng(2),
        n_calls=25, n_initial_points=10,
    )
    _assert_result_shape(result, ["hA", "k_wh"])
    assert abs(result.params["hA"] - _HA2_TRUE) / _HA2_TRUE < 0.2
    assert abs(result.params["k_wh"] - _KWH_TRUE) / _KWH_TRUE < 0.2


# ---------------------------------------------------------------------------
# Cross-cutting behavior
# ---------------------------------------------------------------------------

def test_T0_defaults_to_first_measured_sample():
    """When T0 is not given, every calibrator should fall back to the first
    noisy reading (a real calibrator has no other source for it)."""
    t, I_t, T_measured = _one_node_fixture()
    result = lm.calibrate_one_node(t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1))
    # indirectly verified via convergence above; here just check the call
    # succeeds identically whether T0 is passed explicitly as T_measured[0]
    result_explicit = lm.calibrate_one_node(
        t, I_t, T_measured, _T_AMBIENT, T0=float(T_measured[0]), rng=np.random.default_rng(1)
    )
    assert result.params["hA"] == result_explicit.params["hA"]


def test_runtime_ordering_lm_faster_than_population_methods():
    """LM (local, gradient-based) should need far fewer simulator evaluations
    than GA/PSO (population-based, gradient-free) to reach a comparable fit."""
    t, I_t, T_measured = _one_node_fixture()
    r_lm = lm.calibrate_one_node(t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1))
    r_ga = ga.calibrate_one_node(t, I_t, T_measured, _T_AMBIENT, rng=np.random.default_rng(1))
    assert r_lm.n_evals < r_ga.n_evals


def _run_all():
    tests = [
        test_lm_recovers_hA_one_node,
        test_ga_recovers_hA_one_node,
        test_pso_recovers_hA_one_node,
        test_bayesopt_recovers_hA_one_node,
        test_ekf_recovers_hA_one_node,
        test_lm_recovers_params_two_node,
        test_ekf_recovers_params_two_node,
        test_ga_recovers_params_two_node,
        test_pso_recovers_params_two_node,
        test_bayesopt_recovers_params_two_node,
        test_T0_defaults_to_first_measured_sample,
        test_runtime_ordering_lm_faster_than_population_methods,
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
