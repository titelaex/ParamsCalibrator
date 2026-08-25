"""
Physics-motivated feature extraction for the ML calibration model.

"""

import numpy as np

from src.simulator.params import (
    C_HOUSING_J_PER_K,
    C_LUMPED_J_PER_K,
    C_WINDING_J_PER_K,
    R_WINDING_OHM,
)

_EPS = 1e-9

_HA_CLIP = (0.0, 200.0)
_KWH_CLIP = (0.0, 500.0)
_TAU_CLIP = (0.0, 20000.0)


def _edge_mean(x, frac=0.02):
    """Average of the first / last few samples, as a noise-reduced endpoint.

    """
    n = max(1, int(round(frac * x.shape[0])))
    return float(np.mean(x[:n])), float(np.mean(x[-n:]))


def _cumulative_trapezoid(y, x):
    """INT_0^t y ds sampled on the same grid as y, leading 0. Shape (N,)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def _cumulative_zoh(y, x):
    """INT_0^t y ds under a zero-order hold on y -- the convention the
    simulator integrates the applied current under."""
    return np.concatenate([[0.0], np.cumsum(y[:-1] * np.diff(x))])


def _ols_slope(x, y):
    """Slope of the least-squares line y = slope*x + intercept.

    The intercept is fitted rather than assumed, which is what lets the
    unknown initial temperature drop out of the energy-balance estimators.
    """
    x_mean = x.mean()
    Sxx = float(np.sum((x - x_mean) ** 2))
    if Sxx < _EPS:
        return 0.0
    slope = float(np.sum((x - x_mean) * (y - y.mean())) / Sxx)
    return slope if np.isfinite(slope) else 0.0


def _energy_balance_slope(cum_potential, cum_energy_minus_stored, clip):
    """Fit `cum_energy_minus_stored = coeff * cum_potential + const` for coeff."""
    return float(np.clip(_ols_slope(cum_potential, cum_energy_minus_stored), clip[0], clip[1]))


def _safe_div(num, den, clip):
    """Divide, then force the result finite and inside `clip`."""
    if abs(den) < _EPS:
        return 0.0
    val = num / den
    if not np.isfinite(val):
        return 0.0
    return float(np.clip(val, clip[0], clip[1]))


def _noise_std_estimate(T):
    """Robust sensor-noise std estimate from first differences.

    The true temperature is smooth on the sampling timescale (tau is
    hundreds of seconds, dt is 5s), so consecutive-sample differences are
    dominated by noise: std(diff) ~ sqrt(2)*sigma. A median-absolute-deviation
    version is used instead of a plain std so that the genuine jumps in a
    duty-cycle profile do not inflate the estimate.
    """
    d = np.diff(T)
    if d.shape[0] == 0:
        return 0.0
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def _slope(t, T, lo_frac, hi_frac):
    """Least-squares slope of T over a fractional sub-window of the run."""
    n = T.shape[0]
    i0, i1 = int(lo_frac * n), max(int(hi_frac * n), int(lo_frac * n) + 2)
    i1 = min(i1, n)
    if i1 - i0 < 2:
        return 0.0
    tt, TT = t[i0:i1], T[i0:i1]
    slope = np.polyfit(tt, TT, 1)[0]
    return float(slope) if np.isfinite(slope) else 0.0


def _tau_63(t, T):
    """Empirical time to reach 63.2% of the total temperature excursion.

    """
    T0, T_end = _edge_mean(T)
    span = T_end - T0
    if abs(span) < 1e-3:
        return 0.0
    target = T0 + 0.632 * span
    if span > 0:
        idx = np.argmax(T >= target)
        hit = T[idx] >= target
    else:
        idx = np.argmax(T <= target)
        hit = T[idx] <= target
    if not hit:
        return float(np.clip(t[-1], *_TAU_CLIP))
    return float(np.clip(t[idx], *_TAU_CLIP))


# ---------------------------------------------------------------------------
# 1-node
# ---------------------------------------------------------------------------

ONE_NODE_FEATURE_NAMES = [
    "hA_energy_balance",   # integral estimator -- the physics anchor
    "hA_steady_state",     # I_rms^2*R / (T_end - T_amb)
    "hA_from_tau",         # C / tau_63
    "tau_63",
    "initial_slope",
    "final_slope",
    "T_rise_final",
    "T_rise_max",
    "T_rise_mean",
    "I_rms",
    "I_mean",
    "I_std",
    "P_mean",
    "T_ambient",
    "noise_std_est",
]

ONE_NODE_TARGET_NAMES = ["hA"]


def extract_one_node_features(
    t, I, T_measured, T_ambient,
    R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K,
) -> np.ndarray:
    """Feature vector for a single 1-node run. Returns (n_features,) float array.

    """
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)
    T = np.asarray(T_measured, dtype=float)

    P = I * I * R_winding
    T_0, T_end = _edge_mean(T)

    # (1) energy balance, as a slope over the whole window:
    #     E(t) - C*T(t) = hA * INT (T - T_amb) ds - C*T(0)
    cum_E = _cumulative_zoh(P, t)
    cum_S = _cumulative_trapezoid(T - T_ambient, t)
    hA_energy = _energy_balance_slope(cum_S, cum_E - C * T, _HA_CLIP)

    # (2) steady state: P = hA * (T_ss - T_amb)
    I_rms = float(np.sqrt(np.mean(I * I)))
    P_mean = I_rms * I_rms * R_winding
    hA_ss = _safe_div(P_mean, T_end - T_ambient, _HA_CLIP)

    # (3) first-order time constant: tau = C / hA
    tau = _tau_63(t, T)
    hA_tau = _safe_div(C, tau, _HA_CLIP)

    feats = [
        hA_energy,
        hA_ss,
        hA_tau,
        tau,
        _slope(t, T, 0.0, 0.05),
        _slope(t, T, 0.90, 1.0),
        T_end - T_ambient,
        float(np.max(T)) - T_ambient,
        float(np.mean(T)) - T_ambient,
        I_rms,
        float(np.mean(I)),
        float(np.std(I)),
        P_mean,
        float(T_ambient),
        _noise_std_estimate(T),
    ]
    return np.nan_to_num(np.array(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# 2-node
# ---------------------------------------------------------------------------

TWO_NODE_FEATURE_NAMES = [
    "hA_energy_balance",    # total energy balance (both masses)
    "k_wh_energy_balance",  # winding-node energy balance
    "hA_steady_state",      # P / (T_h_end - T_amb)
    "k_wh_steady_state",    # P / (T_w_end - T_h_end)
    "dT_wh_final",
    "dT_wh_mean",
    "dT_wh_max",
    "T_w_rise_final",
    "T_w_rise_mean",
    "T_h_rise_final",
    "T_h_rise_mean",
    "tau_63_w",
    "tau_63_h",
    "initial_slope_w",
    "initial_slope_h",
    "final_slope_w",
    "final_slope_h",
    "I_rms",
    "I_mean",
    "I_std",
    "P_mean",
    "T_ambient",
    "noise_std_est_w",
    "noise_std_est_h",
]

TWO_NODE_TARGET_NAMES = ["hA", "k_wh"]


def extract_two_node_features(
    t, I, T_w_measured, T_h_measured, T_ambient,
    R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
) -> np.ndarray:
    """Feature vector for a single 2-node run. Returns (n_features,) float array."""
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)
    T_w = np.asarray(T_w_measured, dtype=float)
    T_h = np.asarray(T_h_measured, dtype=float)

    P = I * I * R_winding
    T_w_0, T_w_end = _edge_mean(T_w)
    T_h_0, T_h_end = _edge_mean(T_h)

    cum_E = _cumulative_zoh(P, t)
    cum_S_wh = _cumulative_trapezoid(T_w - T_h, t)      # drives winding -> housing conduction
    cum_S_h = _cumulative_trapezoid(T_h - T_ambient, t)  # drives housing -> ambient convection

    # winding-node energy balance -> k_wh
    k_wh_energy = _energy_balance_slope(cum_S_wh, cum_E - C_w * T_w, _KWH_CLIP)
    # total energy balance (both masses) -> hA, independent of the k_wh estimate
    hA_energy = _energy_balance_slope(cum_S_h, cum_E - C_w * T_w - C_h * T_h, _HA_CLIP)

    I_rms = float(np.sqrt(np.mean(I * I)))
    P_mean = I_rms * I_rms * R_winding
    hA_ss = _safe_div(P_mean, T_h_end - T_ambient, _HA_CLIP)
    k_wh_ss = _safe_div(P_mean, T_w_end - T_h_end, _KWH_CLIP)

    feats = [
        hA_energy,
        k_wh_energy,
        hA_ss,
        k_wh_ss,
        T_w_end - T_h_end,
        float(np.mean(T_w - T_h)),
        float(np.max(T_w - T_h)),
        T_w_end - T_ambient,
        float(np.mean(T_w)) - T_ambient,
        T_h_end - T_ambient,
        float(np.mean(T_h)) - T_ambient,
        _tau_63(t, T_w),
        _tau_63(t, T_h),
        _slope(t, T_w, 0.0, 0.05),
        _slope(t, T_h, 0.0, 0.05),
        _slope(t, T_w, 0.90, 1.0),
        _slope(t, T_h, 0.90, 1.0),
        I_rms,
        float(np.mean(I)),
        float(np.std(I)),
        P_mean,
        float(T_ambient),
        _noise_std_estimate(T_w),
        _noise_std_estimate(T_h),
    ]
    return np.nan_to_num(np.array(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def build_one_node_xy(data: dict, n_samples: int | None = None):
    """Feature matrix X (n_runs, n_features) and target y (n_runs, 1) from a dataset.

    `n_samples` truncates every run to its first N samples, which is how the
    benchmark evaluates the short-window / partial-convergence regime that
    the closed-form estimators handle worst.
    """
    t = data["t"]
    sl = slice(None) if n_samples is None else slice(0, n_samples)
    t_w = t[sl]

    X = np.array([
        extract_one_node_features(
            t_w, data["I"][i][sl], data["T_measured"][i][sl], float(data["T_ambient"][i])
        )
        for i in range(data["I"].shape[0])
    ])
    y = data["hA"].reshape(-1, 1)
    return X, y


def build_two_node_xy(data: dict, n_samples: int | None = None):
    """Feature matrix X (n_runs, n_features) and target y (n_runs, 2) from a dataset."""
    t = data["t"]
    sl = slice(None) if n_samples is None else slice(0, n_samples)
    t_w = t[sl]

    X = np.array([
        extract_two_node_features(
            t_w, data["I"][i][sl], data["T_w_measured"][i][sl],
            data["T_h_measured"][i][sl], float(data["T_ambient"][i])
        )
        for i in range(data["I"].shape[0])
    ])
    y = np.column_stack([data["hA"], data["k_wh"]])
    return X, y
