"""
Sanity checks for the physics simulator.

pytest is not installable in this sandbox (restricted package registry), so
this file is written to also run standalone: `python3 tests/test_simulator.py`.
Functions are still named test_* and take no arguments, so if pytest is
available in your local environment (e.g. `pip install pytest` on your own
machine) this file also collects and runs normally under `pytest tests/`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator.data_generator import (
    generate_one_node_dataset,
    generate_two_node_dataset,
    time_grid,
)
from src.simulator.motor_thermal import (
    simulate_one_node,
    simulate_two_node,
    steady_state_one_node,
    steady_state_two_node,
)
from src.simulator.params import (
    C_LUMPED_J_PER_K,
    R_WINDING_OHM,
    OneNodeParams,
    TwoNodeParams,
)
from src.simulator.sensors import add_noise


def test_one_node_reaches_analytical_steady_state():
    """A long constant-current run should converge to T_ss = T_amb + I^2 R / hA."""
    hA = 15.0
    T_ambient = 25.0
    I_level = 8.0
    params = OneNodeParams(hA=hA, T_ambient=T_ambient)

    # simulate for ~8 time constants so it is fully settled, independent of
    # the shorter SIM_DURATION_S used for the training dataset
    tau = C_LUMPED_J_PER_K / hA
    t = np.arange(0.0, 8 * tau, 5.0)
    I_t = np.full_like(t, I_level)

    T = simulate_one_node(params, t, I_t, T0=T_ambient)
    T_ss_analytical = steady_state_one_node(hA, R_WINDING_OHM, I_level, T_ambient)

    assert abs(T[-1] - T_ss_analytical) < 0.05, (
        f"simulated steady-state {T[-1]:.3f} vs analytical {T_ss_analytical:.3f}"
    )


def test_one_node_time_constant_matches_expected():
    """Empirical 63%-rise time should match tau = C / hA for a step input."""
    hA = 15.0
    T_ambient = 25.0
    I_level = 8.0
    params = OneNodeParams(hA=hA, T_ambient=T_ambient)

    tau_expected = C_LUMPED_J_PER_K / hA
    t = np.arange(0.0, 8 * tau_expected, 1.0)
    I_t = np.full_like(t, I_level)

    T = simulate_one_node(params, t, I_t, T0=T_ambient)
    T_ss = T[-1]
    target = T_ambient + 0.632 * (T_ss - T_ambient)

    idx = np.searchsorted(T, target)
    tau_empirical = t[idx]

    rel_err = abs(tau_empirical - tau_expected) / tau_expected
    assert rel_err < 0.03, f"empirical tau {tau_empirical:.1f}s vs expected {tau_expected:.1f}s (rel err {rel_err:.3f})"


def test_two_node_reaches_analytical_steady_state():
    hA = 15.0
    k_wh = 35.0
    T_ambient = 25.0
    I_level = 8.0
    params = TwoNodeParams(hA=hA, k_wh=k_wh, T_ambient=T_ambient)

    # conservative long duration to guarantee convergence for the slower node
    t = np.arange(0.0, 40000.0, 5.0)
    I_t = np.full_like(t, I_level)

    T = simulate_two_node(params, t, I_t, T0=np.array([T_ambient, T_ambient]))
    T_w_ss_analytical, T_h_ss_analytical = steady_state_two_node(
        hA, k_wh, R_WINDING_OHM, I_level, T_ambient
    )

    assert abs(T[-1, 0] - T_w_ss_analytical) < 0.1, (
        f"winding: simulated {T[-1, 0]:.3f} vs analytical {T_w_ss_analytical:.3f}"
    )
    assert abs(T[-1, 1] - T_h_ss_analytical) < 0.1, (
        f"housing: simulated {T[-1, 1]:.3f} vs analytical {T_h_ss_analytical:.3f}"
    )
    # winding must always run hotter than housing (heat flows winding -> housing -> ambient)
    assert T[-1, 0] > T[-1, 1] > T_ambient


def test_sensor_noise_has_expected_mean_and_std():
    rng = np.random.default_rng(42)
    T_true = np.full(20000, 50.0)
    noise_std = 1.5
    T_measured = add_noise(T_true, noise_std, rng)

    residual = T_measured - T_true
    assert abs(residual.mean()) < 0.05, f"noise mean {residual.mean():.4f} should be ~0"
    assert abs(residual.std() - noise_std) < 0.05, f"noise std {residual.std():.4f} should be ~{noise_std}"


def test_generated_dataset_shapes_and_ranges():
    d1 = generate_one_node_dataset(n_runs=20, seed=7)
    t = time_grid()
    n_steps = t.shape[0]

    assert d1["I"].shape == (20, n_steps)
    assert d1["T_measured"].shape == (20, n_steps)
    assert d1["hA"].shape == (20,)
    assert np.all(d1["hA"] >= 8.0) and np.all(d1["hA"] <= 25.0)
    assert np.all(np.isfinite(d1["T_measured"]))

    d2 = generate_two_node_dataset(n_runs=20, seed=8)
    assert d2["T_w_measured"].shape == (20, n_steps)
    assert d2["T_h_measured"].shape == (20, n_steps)
    assert np.all(d2["k_wh"] >= 15.0) and np.all(d2["k_wh"] <= 60.0)
    # winding should be at or above housing temperature throughout (allow noise slack)
    assert np.mean(d2["T_w_true"] >= d2["T_h_true"] - 1e-6) > 0.999


def _run_all():
    tests = [
        test_one_node_reaches_analytical_steady_state,
        test_one_node_time_constant_matches_expected,
        test_two_node_reaches_analytical_steady_state,
        test_sensor_noise_has_expected_mean_and_std,
        test_generated_dataset_shapes_and_ranges,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
