"""
Extended Kalman Filter calibrator 

Unlike the batch methods (GA, PSO, LM, Bayesian Optimization), the EKF never
sees the whole run at once: it processes one sensor sample at a time and
updates a running estimate.
"""

import time

import numpy as np

from src.baselines.base import (
    CalibrationResult,
    ONE_NODE_BOUNDS,
    TWO_NODE_BOUNDS,
)
from src.simulator.params import C_HOUSING_J_PER_K, C_LUMPED_J_PER_K, C_WINDING_J_PER_K, R_WINDING_OHM


# 1-node: state x = [T, hA]

def _one_node_step(x, I, dt, R, C, T_ambient):
    T, hA = x
    T_next = T + dt * (I * I * R - hA * (T - T_ambient)) / C
    return np.array([T_next, hA])


def _one_node_jacobian(x, I, dt, R, C, T_ambient):
    T, hA = x
    return np.array([
        [1.0 - dt * hA / C, -dt * (T - T_ambient) / C],
        [0.0, 1.0],
    ])


def calibrate_one_node(
    t, I_t, T_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K,
    bounds=ONE_NODE_BOUNDS, rng=None,
    hA0=None, measurement_noise_std=1.0,
    process_noise_T=1e-3, process_noise_hA_frac=5e-4,
) -> CalibrationResult:
    """process_noise_hA_frac is relative to the bounds span, applied per-step."""
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = bounds
    T0 = float(T_measured[0]) if T0 is None else T0
    hA0 = rng.uniform(lo, hi) if hA0 is None else hA0
    dt = float(t[1] - t[0])
    n = t.shape[0]

    x = np.array([T0, hA0])
    P = np.diag([measurement_noise_std ** 2, ((hi - lo) / 2) ** 2])
    Q = np.diag([process_noise_T ** 2, (process_noise_hA_frac * (hi - lo)) ** 2])
    R = np.array([[measurement_noise_std ** 2]])
    H = np.array([[1.0, 0.0]])

    hA_history = np.empty(n)
    hA_history[0] = x[1]

    t0 = time.perf_counter()
    for k in range(n - 1):
        F = _one_node_jacobian(x, I_t[k], dt, R_winding, C, T_ambient)
        x_pred = _one_node_step(x, I_t[k], dt, R_winding, C, T_ambient)
        P_pred = F @ P @ F.T + Q

        z = np.array([T_measured[k + 1]])
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred
        hA_history[k + 1] = x[1]
    runtime_s = time.perf_counter() - t0

    return CalibrationResult(
        params={"hA": float(x[1])},
        runtime_s=runtime_s,
        n_evals=n - 1,
        converged=True,
        history=hA_history,
        extra={"hA0": hA0, "final_P_hA": float(P[1, 1])},
    )



# 2-node: state x = [T_w, T_h, hA, k_wh]


def _two_node_step(x, I, dt, R, C_w, C_h, T_ambient):
    T_w, T_h, hA, k_wh = x
    P_loss = I * I * R
    T_w_next = T_w + dt * (P_loss - k_wh * (T_w - T_h)) / C_w
    T_h_next = T_h + dt * (k_wh * (T_w - T_h) - hA * (T_h - T_ambient)) / C_h
    return np.array([T_w_next, T_h_next, hA, k_wh])


def _two_node_jacobian(x, I, dt, R, C_w, C_h, T_ambient):
    T_w, T_h, hA, k_wh = x
    return np.array([
        [1.0 - dt * k_wh / C_w, dt * k_wh / C_w, 0.0, -dt * (T_w - T_h) / C_w],
        [dt * k_wh / C_h, 1.0 - dt * (k_wh + hA) / C_h, -dt * (T_h - T_ambient) / C_h, dt * (T_w - T_h) / C_h],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def calibrate_two_node(
    t, I_t, T_w_measured, T_h_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
    bounds=TWO_NODE_BOUNDS, rng=None,
    hA0=None, kwh0=None, measurement_noise_std=1.0,
    process_noise_T=1e-3, process_noise_frac=5e-4,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    (hA_lo, hA_hi), (kwh_lo, kwh_hi) = bounds
    T0 = np.array([T_w_measured[0], T_h_measured[0]]) if T0 is None else T0
    hA0 = rng.uniform(hA_lo, hA_hi) if hA0 is None else hA0
    kwh0 = rng.uniform(kwh_lo, kwh_hi) if kwh0 is None else kwh0
    dt = float(t[1] - t[0])
    n = t.shape[0]

    x = np.array([T0[0], T0[1], hA0, kwh0])
    P = np.diag([
        measurement_noise_std ** 2,
        measurement_noise_std ** 2,
        ((hA_hi - hA_lo) / 2) ** 2,
        ((kwh_hi - kwh_lo) / 2) ** 2,
    ])
    Q = np.diag([
        process_noise_T ** 2,
        process_noise_T ** 2,
        (process_noise_frac * (hA_hi - hA_lo)) ** 2,
        (process_noise_frac * (kwh_hi - kwh_lo)) ** 2,
    ])
    R = np.diag([measurement_noise_std ** 2, measurement_noise_std ** 2])
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])

    history = np.empty((n, 2))
    history[0] = [x[2], x[3]]

    t0 = time.perf_counter()
    for k in range(n - 1):
        F = _two_node_jacobian(x, I_t[k], dt, R_winding, C_w, C_h, T_ambient)
        x_pred = _two_node_step(x, I_t[k], dt, R_winding, C_w, C_h, T_ambient)
        P_pred = F @ P @ F.T + Q

        z = np.array([T_w_measured[k + 1], T_h_measured[k + 1]])
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred
        history[k + 1] = [x[2], x[3]]
    runtime_s = time.perf_counter() - t0

    return CalibrationResult(
        params={"hA": float(x[2]), "k_wh": float(x[3])},
        runtime_s=runtime_s,
        n_evals=n - 1,
        converged=True,
        history=history,
        extra={"hA0": hA0, "kwh0": kwh0},
    )
