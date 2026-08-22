import numpy as np


def add_noise(T_true: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Return a noisy sensor reading given the true temperature trace."""
    return T_true + rng.normal(loc=0.0, scale=noise_std, size=T_true.shape)
