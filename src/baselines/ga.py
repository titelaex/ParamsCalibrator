"""
Genetic Algorithm calibrator.
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


def _run_ga(objective, lo, hi, rng, pop_size, n_generations, tournament_k=3, mutation_sigma_frac=0.1, elitism=1):
    """Generic real-coded GA minimizing `objective(x) -> float` over a box [lo, hi]^d.

    lo, hi : (d,) arrays of per-dimension bounds.
    Returns (best_x, best_f, n_evals, history) where history is the best-so-far
    fitness at each generation (useful to demonstrate convergence).
    """
    d = lo.shape[0]
    pop = rng.uniform(lo, hi, size=(pop_size, d))
    fitness = np.array([objective(ind) for ind in pop])
    n_evals = pop_size
    history = [fitness.min()]

    for _gen in range(n_generations):
        order = np.argsort(fitness)
        elite = pop[order[:elitism]].copy()

        children = []
        while len(children) < pop_size - elitism:

            parents = []
            for _ in range(2):
                idx = rng.integers(0, pop_size, size=tournament_k)
                parents.append(pop[idx[np.argmin(fitness[idx])]])
            p1, p2 = parents

            alpha = rng.uniform(0.0, 1.0, size=d)
            child = alpha * p1 + (1 - alpha) * p2


            mutate_mask = rng.random(d) < 0.3
            sigma = mutation_sigma_frac * (hi - lo)
            child = np.where(mutate_mask, child + rng.normal(0.0, sigma), child)
            child = np.clip(child, lo, hi)
            children.append(child)

        new_pop = np.vstack([elite, np.array(children)])
        new_fitness = np.array([objective(ind) for ind in new_pop])
        n_evals += len(children)

        pop, fitness = new_pop, new_fitness
        history.append(fitness.min())

    best_idx = np.argmin(fitness)
    return pop[best_idx], float(fitness[best_idx]), n_evals, np.array(history)


def calibrate_one_node(
    t, I_t, T_measured, T_ambient, T0=None,
    R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K,
    bounds=ONE_NODE_BOUNDS, rng=None,
    pop_size=24, n_generations=25,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_one_node(T_measured, T0)
    lo, hi = np.array([bounds[0]]), np.array([bounds[1]])

    def objective(x):
        return one_node_sse(x[0], t, I_t, T_measured, T_ambient, T0, R_winding, C)

    t0 = time.perf_counter()
    best_x, best_f, n_evals, history = _run_ga(objective, lo, hi, rng, pop_size, n_generations)
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
    pop_size=32, n_generations=35,
) -> CalibrationResult:
    rng = np.random.default_rng() if rng is None else rng
    T0 = resolve_T0_two_node(T_w_measured, T_h_measured, T0)
    (hA_lo, hA_hi), (kwh_lo, kwh_hi) = bounds
    lo, hi = np.array([hA_lo, kwh_lo]), np.array([hA_hi, kwh_hi])

    def objective(x):
        return two_node_sse(x, t, I_t, T_w_measured, T_h_measured, T_ambient, T0, R_winding, C_w, C_h)

    t0 = time.perf_counter()
    best_x, best_f, n_evals, history = _run_ga(objective, lo, hi, rng, pop_size, n_generations)
    runtime_s = time.perf_counter() - t0

    return CalibrationResult(
        params={"hA": float(best_x[0]), "k_wh": float(best_x[1])},
        runtime_s=runtime_s,
        n_evals=n_evals,
        converged=True,
        history=history,
        extra={"final_sse": best_f},
    )
