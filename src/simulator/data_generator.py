"""
Synthetic dataset generation for both testbeds (1-node and 2-node motor
thermal model).

Each "run" = one randomly sampled ground-truth parameter set + one randomly
sampled load profile, simulated on the shared time grid, with Gaussian sensor
noise added at a randomly sampled noise level. A run is the unit that later
gets fed to both the classical calibrators and the ML model.
"""

from dataclasses import asdict

import numpy as np

from src.simulator.params import (
    AMBIENT_TEMP_RANGE_C,
    HA_RANGE_W_PER_K,
    INITIAL_TEMP_OFFSET_RANGE_C,
    KWH_RANGE_W_PER_K,
    NOISE_STD_RANGE_C,
    SIM_DT_S,
    SIM_DURATION_S,
    OneNodeParams,
    TwoNodeParams,
)
from src.simulator.load_profiles import PROFILE_TYPES, generate_load_profile
from src.simulator.motor_thermal import simulate_one_node, simulate_two_node
from src.simulator.sensors import add_noise


def _rand_in(rng, lo, hi):
    return lo + rng.random() * (hi - lo)


def time_grid() -> np.ndarray:
    return np.arange(0.0, SIM_DURATION_S + SIM_DT_S, SIM_DT_S)


# ---------------------------------------------------------------------------
# Single-run generation
# ---------------------------------------------------------------------------

def generate_one_node_run(t: np.ndarray, rng: np.random.Generator) -> dict:
    hA = _rand_in(rng, *HA_RANGE_W_PER_K)
    T_ambient = _rand_in(rng, *AMBIENT_TEMP_RANGE_C)
    noise_std = _rand_in(rng, *NOISE_STD_RANGE_C)
    T0 = T_ambient + _rand_in(rng, *INITIAL_TEMP_OFFSET_RANGE_C)
    profile_type = rng.choice(PROFILE_TYPES)

    I_t, profile_meta = generate_load_profile(profile_type, t, rng)
    params = OneNodeParams(hA=hA, T_ambient=T_ambient)
    T_true = simulate_one_node(params, t, I_t, T0=T0)
    T_measured = add_noise(T_true, noise_std, rng)

    return {
        "I": I_t,
        "T_true": T_true,
        "T_measured": T_measured,
        "hA": hA,
        "T_ambient": T_ambient,
        "noise_std": noise_std,
        "profile_type": str(profile_type),
    }


def generate_two_node_run(t: np.ndarray, rng: np.random.Generator) -> dict:
    hA = _rand_in(rng, *HA_RANGE_W_PER_K)
    k_wh = _rand_in(rng, *KWH_RANGE_W_PER_K)
    T_ambient = _rand_in(rng, *AMBIENT_TEMP_RANGE_C)
    noise_std = _rand_in(rng, *NOISE_STD_RANGE_C)
    T0 = np.array([T_ambient, T_ambient]) + _rand_in(rng, *INITIAL_TEMP_OFFSET_RANGE_C)
    profile_type = rng.choice(PROFILE_TYPES)

    I_t, profile_meta = generate_load_profile(profile_type, t, rng)
    params = TwoNodeParams(hA=hA, k_wh=k_wh, T_ambient=T_ambient)
    T_true = simulate_two_node(params, t, I_t, T0=T0)  # (N, 2)
    T_w_true, T_h_true = T_true[:, 0], T_true[:, 1]
    T_w_measured = add_noise(T_w_true, noise_std, rng)
    T_h_measured = add_noise(T_h_true, noise_std, rng)

    return {
        "I": I_t,
        "T_w_true": T_w_true,
        "T_h_true": T_h_true,
        "T_w_measured": T_w_measured,
        "T_h_measured": T_h_measured,
        "hA": hA,
        "k_wh": k_wh,
        "T_ambient": T_ambient,
        "noise_std": noise_std,
        "profile_type": str(profile_type),
    }


# ---------------------------------------------------------------------------
# Batch dataset generation
# ---------------------------------------------------------------------------

def generate_one_node_dataset(n_runs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    t = time_grid()
    runs = [generate_one_node_run(t, rng) for _ in range(n_runs)]

    return {
        "t": t,
        "I": np.stack([r["I"] for r in runs]),
        "T_true": np.stack([r["T_true"] for r in runs]),
        "T_measured": np.stack([r["T_measured"] for r in runs]),
        "hA": np.array([r["hA"] for r in runs]),
        "T_ambient": np.array([r["T_ambient"] for r in runs]),
        "noise_std": np.array([r["noise_std"] for r in runs]),
        "profile_type": np.array([r["profile_type"] for r in runs], dtype=object),
    }


def generate_two_node_dataset(n_runs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    t = time_grid()
    runs = [generate_two_node_run(t, rng) for _ in range(n_runs)]

    return {
        "t": t,
        "I": np.stack([r["I"] for r in runs]),
        "T_w_true": np.stack([r["T_w_true"] for r in runs]),
        "T_h_true": np.stack([r["T_h_true"] for r in runs]),
        "T_w_measured": np.stack([r["T_w_measured"] for r in runs]),
        "T_h_measured": np.stack([r["T_h_measured"] for r in runs]),
        "hA": np.array([r["hA"] for r in runs]),
        "k_wh": np.array([r["k_wh"] for r in runs]),
        "T_ambient": np.array([r["T_ambient"] for r in runs]),
        "noise_std": np.array([r["noise_std"] for r in runs]),
        "profile_type": np.array([r["profile_type"] for r in runs], dtype=object),
    }


def save_dataset(path: str, data: dict) -> None:
    np.savez_compressed(path, **data)


def load_dataset(path: str) -> dict:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}
