"""
Applied-current load profiles used to excite the motor thermal model when
generating synthetic sensor data.

Three profile families are used, each representative of a common motor duty
pattern: a step load (start and hold), a duty-cycle load (periodic on/off,
e.g. an intermittent process), and a ramp load (gradually increasing load).
Randomizing the profile's parameters (not just the unknown thermal constants)
is what makes the resulting synthetic dataset diverse enough to train and
fairly benchmark a calibration model on.
"""

import numpy as np

from src.simulator.params import CURRENT_RANGE_A

PROFILE_TYPES = ("step", "duty_cycle", "ramp")


def _rand_in(rng, lo, hi):
    return lo + rng.random() * (hi - lo)


def step_profile(t: np.ndarray, rng: np.random.Generator):
    """Zero current, then a jump to a held load level."""
    duration = t[-1]
    i_lo, i_hi = CURRENT_RANGE_A
    level = _rand_in(rng, 0.3 * i_hi, i_hi)
    t_start = _rand_in(rng, 0.0, 0.1 * duration)
    I = np.where(t >= t_start, level, 0.0)
    meta = {"profile_type": "step", "level_A": level, "t_start_s": t_start}
    return I, meta


def duty_cycle_profile(t: np.ndarray, rng: np.random.Generator):
    """Periodic on/off load — an intermittent process load."""
    duration = t[-1]
    i_lo, i_hi = CURRENT_RANGE_A
    level = _rand_in(rng, 0.3 * i_hi, i_hi)
    idle_level = _rand_in(rng, 0.0, 0.15 * i_hi)
    period = _rand_in(rng, duration / 8, duration / 3)
    duty = _rand_in(rng, 0.3, 0.7)
    phase = (t % period) / period
    I = np.where(phase < duty, level, idle_level)
    meta = {
        "profile_type": "duty_cycle",
        "level_A": level,
        "idle_level_A": idle_level,
        "period_s": period,
        "duty": duty,
    }
    return I, meta


def ramp_profile(t: np.ndarray, rng: np.random.Generator):
    """Linearly increasing (or decreasing) load."""
    i_lo, i_hi = CURRENT_RANGE_A
    start_level = _rand_in(rng, 0.0, 0.5 * i_hi)
    end_level = _rand_in(rng, 0.3 * i_hi, i_hi)
    I = np.linspace(start_level, end_level, num=t.shape[0])
    meta = {"profile_type": "ramp", "start_A": start_level, "end_A": end_level}
    return I, meta


_GENERATORS = {
    "step": step_profile,
    "duty_cycle": duty_cycle_profile,
    "ramp": ramp_profile,
}


def generate_load_profile(profile_type: str, t: np.ndarray, rng: np.random.Generator):
    """Dispatch to the requested profile generator.

    Returns (I_t, meta) where I_t is an array the same shape as t (amps) and
    meta is a dict of the randomized profile parameters, useful for
    stratifying benchmark results later.
    """
    if profile_type not in _GENERATORS:
        raise ValueError(f"Unknown profile_type {profile_type!r}; expected one of {PROFILE_TYPES}")
    return _GENERATORS[profile_type](t, rng)
