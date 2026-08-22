"""
Particle Swarm Optimization calibrator -- a from-scratch, real-valued PSO
(numpy only). A second, distinct population-based metaheuristic (proposal,
Section 5): a swarm of particles moves through parameter space, each pulled
towards its own best-ever position and the swarm's best-ever position, with
inertia controlling how much of the previous velocity carries over.

Included alongside GA specifically so the benchmark can show the ML model
beating the whole family of population-based heuristics, not just one
member of it.
"""

import time

import numpy as np

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


def _run_pso(objective, lo, hi, rng, n_particles, n_iters, w=0.7, c1=1.5, c2=1.5):
    """Generic PSO minimizing `objective(x) -> float` over a box [lo, hi]^d.

    w  : inertia weight (how much previous velocity persists)
    c1 : cognitive coefficient (pull towards the particle's own best)
    c2 : social coefficient (pull towards the swarm's best)

    Returns (best_x, best_f, n_evals, history) where history is the swarm's
    best-so-far fitness at each iteration.
    """
    d = lo.shape[0]
    span = hi - lo

    pos = rng.uniform(lo, hi, size=(n_particles, d))
    vel = rng.uniform(-span, span, size=(n_particles, d)) * 0.1

    fitness = np.array([objective(p) for p in pos])
    n_evals = n_particles

    pbest_pos = pos.copy()
    pbest_fit = fitness.copy()
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    history = [gbest_fit]

    for _it in range(n_iters):
        r1 = rng.random((n_particles, d))
        r2 = rng.random((n_particles, d))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, lo, hi)

        fitness = np.array([objective(p) for p in pos])
        n_evals += n_particles

        improved = fitness < pbest_fit
        pbest_pos[improved] = pos[improved]
        pbest_fit[improved] = fitness[improved]

        it_best_idx = np.argmin(pbest_fit)
        if pbest_fit[it_best_idx] < gbest_fit:
            gbest_fit = float(pbest_fit[it_best_idx])
            gbest_pos = pbest_pos[it_best_idx].copy()

        history.append(gbest_fit)

    return gbest_pos, gbest_fit, n_evals, np.array(history)


def calibrate_one_node(
    t, I_t, T_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K,
    bounds=ONE_NODE_BOUNDS, rng=None,
    n_particles=20, n_iters=25,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_one_node(T_measured, T0)
    lo, hi = np.array([bounds[0]]), np.array([bounds[1]])

    def objective(x):
        return one_node_sse(x[0], t, I_t, T_measured, T_ambient, T0, R_winding, C)

    t0 = time.perf_counter()
    best_x, best_f, n_evals, history = _run_pso(objective, lo, hi, rng, n_particles, n_iters)
    runtime_s = time.perf_counter() - t0

    return CalibrationResult(
        params={"hA": float(best_x[0])},
        runtime_s=runtime_s,
        n_evals=n_evals,
        converged=True,
        history=history,
        extra={"final_sse": best_f},
    )


def calibrate_two_node(
    t, I_t, T_w_measured, T_h_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
    bounds=TWO_NODE_BOUNDS, rng=None,
    n_particles=28, n_iters=35,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_two_node(T_w_measured, T_h_measured, T0)
    (hA_lo, hA_hi), (kwh_lo, kwh_hi) = bounds
    lo, hi = np.array([hA_lo, kwh_lo]), np.array([hA_hi, kwh_hi])

    def objective(x):
        return two_node_sse(x, t, I_t, T_w_measured, T_h_measured, T_ambient, T0, R_winding, C_w, C_h)

    t0 = time.perf_counter()
    best_x, best_f, n_evals, history = _run_pso(objective, lo, hi, rng, n_particles, n_iters)
    runtime_s = time.perf_counter() - t0

    return CalibrationResult(
        params={"hA": float(best_x[0]), "k_wh": float(best_x[1])},
        runtime_s=runtime_s,
        n_evals=n_evals,
        converged=True,
        history=history,
        extra={"final_sse": best_f},
    )
