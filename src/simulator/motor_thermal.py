"""
Motor winding thermal models (1-node and 2-node lumped-parameter ODEs) and a
fixed-step RK4 integrator.

The applied current I(t) is treated as piecewise-constant over each
integration step (zero-order hold) — a standard and defensible assumption
given that in practice the "known excitation" is itself a sampled signal,
not a continuous analytic function. This also keeps integration simple and
fast (no interpolation needed) which matters when generating thousands of
synthetic runs for ML training data.

1-node:   C * dT/dt        = I(t)^2*R - hA*(T - T_amb)
2-node:   C_w * dT_w/dt    = I(t)^2*R - k_wh*(T_w - T_h)
          C_h * dT_h/dt    = k_wh*(T_w - T_h) - hA*(T_h - T_amb)
"""

from dataclasses import dataclass

import numpy as np

from src.simulator.params import OneNodeParams, TwoNodeParams


# ---------------------------------------------------------------------------
# 1-node model
# ---------------------------------------------------------------------------

def _one_node_deriv(T: float, I: float, p: OneNodeParams) -> float:
    P_loss = I * I * p.R_winding
    return (P_loss - p.hA * (T - p.T_ambient)) / p.C


def simulate_one_node(params: OneNodeParams, t: np.ndarray, I_t: np.ndarray, T0: float | None = None) -> np.ndarray:
    """Integrate the 1-node thermal model over the time grid `t`.

    Parameters
    ----------
    params : OneNodeParams (ground-truth constants for this run)
    t : (N,) array, seconds, uniform grid
    I_t : (N,) array, applied current (A) at each grid point
    T0 : initial winding temperature; defaults to T_ambient if not given.

    Returns
    -------
    T : (N,) array of winding temperature (deg C)
    """
    n = t.shape[0]
    dt = t[1] - t[0]
    T = np.empty(n, dtype=float)
    T[0] = params.T_ambient if T0 is None else T0

    for k in range(n - 1):
        Tk = T[k]
        Ik = I_t[k]
        k1 = _one_node_deriv(Tk, Ik, params)
        k2 = _one_node_deriv(Tk + 0.5 * dt * k1, Ik, params)
        k3 = _one_node_deriv(Tk + 0.5 * dt * k2, Ik, params)
        k4 = _one_node_deriv(Tk + dt * k3, Ik, params)
        T[k + 1] = Tk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return T


def steady_state_one_node(hA: float, R_winding: float, I_level: float, T_ambient: float) -> float:
    """Analytical steady-state winding temperature for a constant applied current.

    Used as an independent sanity check on the integrator (Section 6 of the
    proposal: T_ss - T_ambient = I^2 R / hA).
    """
    return T_ambient + (I_level ** 2) * R_winding / hA


# ---------------------------------------------------------------------------
# 2-node model
# ---------------------------------------------------------------------------

def _two_node_deriv(state: np.ndarray, I: float, p: TwoNodeParams) -> np.ndarray:
    T_w, T_h = state
    P_loss = I * I * p.R_winding
    dTw = (P_loss - p.k_wh * (T_w - T_h)) / p.C_w
    dTh = (p.k_wh * (T_w - T_h) - p.hA * (T_h - p.T_ambient)) / p.C_h
    return np.array([dTw, dTh])


def simulate_two_node(params: TwoNodeParams, t: np.ndarray, I_t: np.ndarray, T0: np.ndarray | None = None) -> np.ndarray:
    """Integrate the 2-node thermal model over the time grid `t`.

    Returns
    -------
    T : (N, 2) array; column 0 = winding temperature, column 1 = housing temperature (deg C)
    """
    n = t.shape[0]
    dt = t[1] - t[0]
    T = np.empty((n, 2), dtype=float)
    T[0] = np.array([params.T_ambient, params.T_ambient]) if T0 is None else T0

    for k in range(n - 1):
        Sk = T[k]
        Ik = I_t[k]
        k1 = _two_node_deriv(Sk, Ik, params)
        k2 = _two_node_deriv(Sk + 0.5 * dt * k1, Ik, params)
        k3 = _two_node_deriv(Sk + 0.5 * dt * k2, Ik, params)
        k4 = _two_node_deriv(Sk + dt * k3, Ik, params)
        T[k + 1] = Sk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return T


def steady_state_two_node(hA: float, k_wh: float, R_winding: float, I_level: float, T_ambient: float):
    """Analytical steady-state (T_w_ss, T_h_ss) for a constant applied current.

    At steady state both derivatives are zero:
        P_loss = k_wh * (T_w - T_h)              (all generated heat flows winding->housing)
        k_wh * (T_w - T_h) = hA * (T_h - T_amb)   (which then all flows housing->ambient)
    => T_h_ss = T_ambient + P_loss / hA
       T_w_ss = T_h_ss + P_loss / k_wh
    """
    P_loss = (I_level ** 2) * R_winding
    T_h_ss = T_ambient + P_loss / hA
    T_w_ss = T_h_ss + P_loss / k_wh
    return T_w_ss, T_h_ss
