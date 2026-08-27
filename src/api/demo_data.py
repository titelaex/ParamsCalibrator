"""Loads the held-out test datasets so the demo UI can replay real runs.

Only used by the demo endpoints -- the calibration endpoints themselves stay
data-free (the client sends its own window of sensor samples).
"""

import os

import numpy as np

from src.simulator.motor_thermal import simulate_one_node, simulate_two_node
from src.simulator.params import SIM_DT_S, OneNodeParams, TwoNodeParams
from src.simulator.sensors import add_noise

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

_FILES = {
    "one_node": "motor_1node_test.npz",
    "two_node": "motor_2node_test.npz",
}

_cache: dict[str, dict] = {}


def available(testbed: str) -> bool:
    return os.path.exists(os.path.join(DATA_DIR, _FILES[testbed]))


def _load(testbed: str) -> dict:
    if testbed not in _cache:
        path = os.path.join(DATA_DIR, _FILES[testbed])
        with np.load(path, allow_pickle=True) as d:
            _cache[testbed] = {k: d[k] for k in d.files}
    return _cache[testbed]


def n_runs(testbed: str) -> int:
    return int(_load(testbed)["I"].shape[0])


def sample_run(testbed: str, index: int, window: float) -> dict:
    """Return run `index` truncated to the first `window` fraction of samples."""
    d = _load(testbed)
    n_total = d["I"].shape[0]
    if not (0 <= index < n_total):
        raise IndexError(f"run index {index} out of range (0..{n_total - 1})")

    t_full = d["t"]
    n = max(2, int(round(len(t_full) * float(window))))
    sl = slice(0, n)

    out = {
        "testbed": testbed,
        "index": index,
        "n_samples": n,
        "n_samples_total": len(t_full),
        "t": t_full[sl].tolist(),
        "I": d["I"][index, sl].tolist(),
        "T_ambient": float(d["T_ambient"][index]),
        "noise_std": float(d["noise_std"][index]),
        "true_params": {"hA": float(d["hA"][index])},
    }
    if testbed == "one_node":
        out["T_measured"] = d["T_measured"][index, sl].tolist()
        out["T_true"] = d["T_true"][index, sl].tolist()
    else:
        out["T_w_measured"] = d["T_w_measured"][index, sl].tolist()
        out["T_h_measured"] = d["T_h_measured"][index, sl].tolist()
        out["T_w_true"] = d["T_w_true"][index, sl].tolist()
        out["T_h_true"] = d["T_h_true"][index, sl].tolist()
        out["true_params"]["k_wh"] = float(d["k_wh"][index])
    return out


def reconstruct(testbed, t, I, T_ambient, T0, params) -> dict:
    """Re-run the physics forward with `params` over the same window.

    This is what makes a calibration legible on screen: instead of only showing
    the estimated number, the UI can overlay the temperature curve that number
    predicts and let the audience see it sit on (or drift off) the sensor data.
    `T0` is the first measured sample -- the same thing a real deployment knows.
    """
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)

    if testbed == "one_node":
        T = simulate_one_node(
            OneNodeParams(hA=float(params["hA"]), T_ambient=float(T_ambient)),
            t, I, T0=float(np.ravel(T0)[0]),
        )
        return {"T": T.tolist()}

    T = simulate_two_node(
        TwoNodeParams(hA=float(params["hA"]), k_wh=float(params["k_wh"]), T_ambient=float(T_ambient)),
        t, I, T0=np.asarray(T0, dtype=float).ravel()[:2],
    )
    return {"T_w": T[:, 0].tolist(), "T_h": T[:, 1].tolist()}


# --------------------------------------------------------------------------
# Synthetic ("design your own motor") runs
#
# The held-out test runs answer "how does this do on motors it has never
# seen". This answers a different question a demo audience always asks --
# "what if MY motor looks like this?" -- by letting the caller dial the
# ground-truth constants itself and checking whether the calibrators recover
# the numbers it just chose.
#
# Load profiles here are deterministic functions of the requested level,
# unlike the randomized ones in src/simulator/load_profiles.py used to build
# the training set: a demo needs the knob to do the same thing every time.
# --------------------------------------------------------------------------

PROFILES = ("step", "duty_cycle", "ramp", "constant")


def _profile(kind: str, t: np.ndarray, level: float) -> np.ndarray:
    duration = t[-1] if t[-1] > 0 else 1.0
    if kind == "constant":
        return np.full_like(t, level)
    if kind == "step":
        return np.where(t >= 0.1 * duration, level, 0.0)
    if kind == "ramp":
        return np.linspace(0.0, level, num=t.shape[0])
    if kind == "duty_cycle":
        period = duration / 5.0
        return np.where((t % period) / period < 0.5, level, 0.0)
    raise ValueError(f"unknown profile {kind!r}; expected one of {PROFILES}")


def synth_run(testbed, hA, k_wh, T_ambient, noise_std, profile, level, duration, seed) -> dict:
    """Simulate one motor with caller-chosen constants and add sensor noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, float(duration) + SIM_DT_S, SIM_DT_S)
    I = _profile(profile, t, float(level))

    out = {
        "testbed": testbed,
        "source": "synthetic",
        "index": None,
        "seed": int(seed),
        "n_samples": int(t.shape[0]),
        "n_samples_total": int(t.shape[0]),
        "t": t.tolist(),
        "I": I.tolist(),
        "T_ambient": float(T_ambient),
        "noise_std": float(noise_std),
        "profile": profile,
        "true_params": {"hA": float(hA)},
    }

    if testbed == "one_node":
        T_true = simulate_one_node(OneNodeParams(hA=float(hA), T_ambient=float(T_ambient)), t, I)
        out["T_true"] = T_true.tolist()
        out["T_measured"] = add_noise(T_true, float(noise_std), rng).tolist()
    else:
        T = simulate_two_node(
            TwoNodeParams(hA=float(hA), k_wh=float(k_wh), T_ambient=float(T_ambient)), t, I
        )
        out["T_w_true"] = T[:, 0].tolist()
        out["T_h_true"] = T[:, 1].tolist()
        out["T_w_measured"] = add_noise(T[:, 0], float(noise_std), rng).tolist()
        out["T_h_measured"] = add_noise(T[:, 1], float(noise_std), rng).tolist()
        out["true_params"]["k_wh"] = float(k_wh)

    return out
