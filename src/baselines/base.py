"""
Common interface shared by all baseline calibration methods (GA, PSO, LM,
EKF, Bayesian Optimization).

Every batch calibrator (all except EKF) solves the same nonlinear
least-squares problem: given one run's noisy sensor trace (applied current,
ambient temperature, measured winding/housing temperature), find the
unknown physical constant(s) that make the physics simulator (RK4
integration of the same ODEs used to generate the data) best reproduce the
measured trace. EKF instead estimates the same unknowns sequentially, one
sample at a time, by treating them as (near-)constant states in an
augmented state vector.

A calibrator never sees the ground-truth constant or the noise-free
temperature — only what a real sensor would provide: I(t), T_ambient, and
noisy T(t). The initial condition T0 is likewise not given as ground truth;
it defaults to the first noisy sample, since that is what a real
calibration procedure would have to use too.
"""

from dataclasses import dataclass, field

import numpy as np

from src.simulator.motor_thermal import simulate_one_node, simulate_two_node
from src.simulator.params import (
    C_HOUSING_J_PER_K,
    C_LUMPED_J_PER_K,
    C_WINDING_J_PER_K,
    HA_RANGE_W_PER_K,
    KWH_RANGE_W_PER_K,
    R_WINDING_OHM,
    OneNodeParams,
    TwoNodeParams,
)


@dataclass
class CalibrationResult:
    """Uniform result object returned by every baseline (and, later, the ML model)."""

    params: dict            # e.g. {"hA": 14.2} or {"hA": 14.2, "k_wh": 33.1}
    runtime_s: float        # wall-clock time for the whole calibration call
    n_evals: int             # number of simulator evaluations used (cost proxy)
    converged: bool = True   # method-specific success flag
    history: np.ndarray | None = None  # optional (n_evals,) or (n_samples,) trace of estimates, for EKF/streaming plots
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Objective functions (shared by GA / PSO / LM / Bayesian Optimization)
# ---------------------------------------------------------------------------

def one_node_residuals(hA, t, I_t, T_measured, T_ambient, T0, R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K):
    """Vector of (simulated - measured) residuals for a candidate hA. Used directly by LM.

    hA may be a bare scalar or a length-1 array (least_squares always passes
    the full parameter vector, even when it has a single component).
    """
    hA = float(np.ravel(hA)[0])
    params = OneNodeParams(hA=hA, T_ambient=T_ambient, C=C, R_winding=R_winding)
    T_sim = simulate_one_node(params, t, I_t, T0=T0)
    return T_sim - T_measured


def one_node_sse(hA, t, I_t, T_measured, T_ambient, T0, R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K):
    """Scalar sum-of-squared-errors for a candidate hA. Used by GA / PSO / Bayesian Optimization."""
    r = one_node_residuals(hA, t, I_t, T_measured, T_ambient, T0, R_winding, C)
    return float(np.sum(r * r))


def two_node_residuals(
    x, t, I_t, T_w_measured, T_h_measured, T_ambient, T0,
    R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
):
    """Stacked (winding, housing) residual vector for candidate x = [hA, k_wh]. Used directly by LM."""
    hA, k_wh = x
    params = TwoNodeParams(hA=float(hA), k_wh=float(k_wh), T_ambient=T_ambient, C_w=C_w, C_h=C_h, R_winding=R_winding)
    T_sim = simulate_two_node(params, t, I_t, T0=T0)  # (N, 2)
    r_w = T_sim[:, 0] - T_w_measured
    r_h = T_sim[:, 1] - T_h_measured
    return np.concatenate([r_w, r_h])


def two_node_sse(
    x, t, I_t, T_w_measured, T_h_measured, T_ambient, T0,
    R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
):
    r = two_node_residuals(x, t, I_t, T_w_measured, T_h_measured, T_ambient, T0, R_winding, C_w, C_h)
    return float(np.sum(r * r))


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

ONE_NODE_BOUNDS = HA_RANGE_W_PER_K
TWO_NODE_BOUNDS = (HA_RANGE_W_PER_K, KWH_RANGE_W_PER_K)


def resolve_T0_one_node(T_measured, T0):
    return float(T_measured[0]) if T0 is None else T0


def resolve_T0_two_node(T_w_measured, T_h_measured, T0):
    return np.array([T_w_measured[0], T_h_measured[0]]) if T0 is None else T0
