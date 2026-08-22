"""
Levenberg-Marquardt calibrator (scipy.optimize.least_squares, method="lm").

LM is classic nonlinear least-squares curve-fitting: given the *known* exact
functional form of the physics (the ODE), it fits the free constant(s) to
one new curve at a time, from scratch, via Gauss-Newton steps damped by a
trust-region parameter that is adjusted each iteration. It is the natural
"closest conceptual cousin" of the ML model (proposal, Section 5): both are
nonlinear regressions, but LM re-derives the fit every single time while the
MLP fits once, offline, over many curves, and then only evaluates.

scipy's "lm" method does not support bounds (true classical LM is
unconstrained), so the initial guess is drawn from the physically plausible
range and the search is otherwise free to explore outside it -- exactly how
an engineer would run curve_fit/least_squares in practice.
"""

import time

import numpy as np
from scipy.optimize import least_squares

from src.baselines.base import (
    CalibrationResult,
    ONE_NODE_BOUNDS,
    TWO_NODE_BOUNDS,
    one_node_residuals,
    resolve_T0_one_node,
    resolve_T0_two_node,
    two_node_residuals,
)
from src.simulator.params import C_HOUSING_J_PER_K, C_LUMPED_J_PER_K, C_WINDING_J_PER_K, R_WINDING_OHM


def calibrate_one_node(
    t, I_t, T_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K,
    bounds=ONE_NODE_BOUNDS, rng=None, x0=None,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_one_node(T_measured, T0)
    lo, hi = bounds
    if x0 is None:
        x0 = rng.uniform(lo, hi)

    t0 = time.perf_counter()
    result = least_squares(
        one_node_residuals, x0=[x0], method="lm",
        args=(t, I_t, T_measured, T_ambient, T0, R_winding, C),
    )
    runtime_s = time.perf_counter() - t0

    hA_hat = float(result.x[0])
    return CalibrationResult(
        params={"hA": hA_hat},
        runtime_s=runtime_s,
        n_evals=int(result.nfev),
        converged=bool(result.success),
        extra={"x0": x0, "cost": float(result.cost)},
    )


def calibrate_two_node(
    t, I_t, T_w_measured, T_h_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
    bounds=TWO_NODE_BOUNDS, rng=None, x0=None,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_two_node(T_w_measured, T_h_measured, T0)
    (hA_lo, hA_hi), (kwh_lo, kwh_hi) = bounds
    if x0 is None:
        x0 = [rng.uniform(hA_lo, hA_hi), rng.uniform(kwh_lo, kwh_hi)]

    t0 = time.perf_counter()
    result = least_squares(
        two_node_residuals, x0=x0, method="lm",
        args=(t, I_t, T_w_measured, T_h_measured, T_ambient, T0, R_winding, C_w, C_h),
    )
    runtime_s = time.perf_counter() - t0

    hA_hat, kwh_hat = float(result.x[0]), float(result.x[1])
    return CalibrationResult(
        params={"hA": hA_hat, "k_wh": kwh_hat},
        runtime_s=runtime_s,
        n_evals=int(result.nfev),
        converged=bool(result.success),
        extra={"x0": list(x0), "cost": float(result.cost)},
    )
