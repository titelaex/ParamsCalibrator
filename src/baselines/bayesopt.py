"""
Bayesian Optimization calibrator (scikit-optimize `gp_minimize`).

Represents modern, sample-efficient, gradient-free calibration practice
(proposal, Section 5): a Gaussian Process surrogate models the objective
(sum-of-squared-errors) over the search space from every evaluation seen so
far, and each new candidate is chosen by maximizing an acquisition function
(expected improvement) that trades off exploring uncertain regions against
exploiting the currently-best-known region. This is the right tool when
each objective evaluation is expensive -- here, one simulator rollout -- so
it is deliberately run with far fewer evaluations than GA/PSO.
"""

import time

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from src.baselines.base import (
    CalibrationResult,
    ONE_NODE_BOUNDS,
    TWO_NODE_BOUNDS,
    one_node_sse,
    resolve_T0_one_node,
    resolve_T0_two_node,
    two_node_sse,
)
from src.simulator.params import C_HOUSING_J_PER_K, C_LUMPED_J_PER_K, C_WINDING_J_PER_K, R_WINDING_OHM


def calibrate_one_node(
    t, I_t, T_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K,
    bounds=ONE_NODE_BOUNDS, rng=None,
    n_calls=30, n_initial_points=10,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_one_node(T_measured, T0)
    lo, hi = bounds
    seed = int(rng.integers(0, 2**31 - 1))

    def objective(x):
        return one_node_sse(x[0], t, I_t, T_measured, T_ambient, T0, R_winding, C)

    t0 = time.perf_counter()
    result = gp_minimize(
        objective, dimensions=[Real(lo, hi, name="hA")],
        n_calls=n_calls, n_initial_points=n_initial_points,
        random_state=seed,
    )
    runtime_s = time.perf_counter() - t0

    return CalibrationResult(
        params={"hA": float(result.x[0])},
        runtime_s=runtime_s,
        n_evals=n_calls,
        converged=True,
        history=np.array(result.func_vals),
        extra={"final_sse": float(result.fun)},
    )


def calibrate_two_node(
    t, I_t, T_w_measured, T_h_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
    bounds=TWO_NODE_BOUNDS, rng=None,
    n_calls=40, n_initial_points=12,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_two_node(T_w_measured, T_h_measured, T0)
    (hA_lo, hA_hi), (kwh_lo, kwh_hi) = bounds
    seed = int(rng.integers(0, 2**31 - 1))

    def objective(x):
        return two_node_sse(x, t, I_t, T_w_measured, T_h_measured, T_ambient, T0, R_winding, C_w, C_h)

    t0 = time.perf_counter()
    result = gp_minimize(
        objective, dimensions=[Real(hA_lo, hA_hi, name="hA"), Real(kwh_lo, kwh_hi, name="k_wh")],
        n_calls=n_calls, n_initial_points=n_initial_points,
        random_state=seed,
    )
    runtime_s = time.perf_counter() - t0

    return CalibrationResult(
        params={"hA": float(result.x[0]), "k_wh": float(result.x[1])},
        runtime_s=runtime_s,
        n_evals=n_calls,
        converged=True,
        history=np.array(result.func_vals),
        extra={"final_sse": float(result.fun)},
    )
