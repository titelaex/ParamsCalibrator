"""
Checks for the ML calibration model: physics-motivated features
(src/ml/features.py) and the MLP over them (src/ml/model.py).

Same standalone-runnable convention as the other test modules: `pytest
tests/ -v`, or `python3 tests/test_ml.py`.

These tests generate their own small datasets rather than reading data/ or
models/, so the suite passes on a clean checkout with no prior
`generate_datasets.py` / `train_ml.py` run.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.base import CalibrationResult
from src.ml.features import (
    ONE_NODE_FEATURE_NAMES,
    TWO_NODE_FEATURE_NAMES,
    build_one_node_xy,
    build_two_node_xy,
    extract_one_node_features,
    extract_two_node_features,
)
from src.ml.model import MLPCalibrator, regression_metrics
from src.simulator.data_generator import (
    generate_one_node_dataset,
    generate_two_node_dataset,
    time_grid,
)
from src.simulator.load_profiles import generate_load_profile
from src.simulator.motor_thermal import simulate_one_node, simulate_two_node
from src.simulator.params import OneNodeParams, TwoNodeParams


def _feat(names, values, key):
    return values[names.index(key)]


# ---------------------------------------------------------------------------
# The physics anchor: the energy-balance estimator must recover the true
# constants on noise-free data, for every load profile. If this breaks,
# nothing downstream is trustworthy.
# ---------------------------------------------------------------------------

def test_energy_balance_recovers_hA_one_node_all_profiles():
    t = time_grid()
    for profile in ("step", "duty_cycle", "ramp"):
        for hA_true in (9.0, 15.0, 24.0):
            rng = np.random.default_rng(5)
            I_t, _ = generate_load_profile(profile, t, rng)
            params = OneNodeParams(hA=hA_true, T_ambient=25.0)
            T = simulate_one_node(params, t, I_t, T0=25.0)

            f = extract_one_node_features(t, I_t, T, 25.0)
            hA_est = _feat(ONE_NODE_FEATURE_NAMES, f, "hA_energy_balance")
            rel_err = abs(hA_est - hA_true) / hA_true
            # the regression form of the energy balance is algebraically exact,
            # so on noise-free data only integration round-off should remain
            assert rel_err < 1e-4, f"{profile} hA={hA_true}: estimate {hA_est:.5f} (rel err {rel_err:.2e})"


def test_energy_balance_recovers_both_constants_two_node_all_profiles():
    t = time_grid()
    for profile in ("step", "duty_cycle", "ramp"):
        for hA_true, kwh_true in ((9.0, 20.0), (15.0, 35.0), (24.0, 55.0)):
            rng = np.random.default_rng(5)
            I_t, _ = generate_load_profile(profile, t, rng)
            params = TwoNodeParams(hA=hA_true, k_wh=kwh_true, T_ambient=25.0)
            T = simulate_two_node(params, t, I_t, T0=np.array([25.0, 25.0]))

            f = extract_two_node_features(t, I_t, T[:, 0], T[:, 1], 25.0)
            hA_est = _feat(TWO_NODE_FEATURE_NAMES, f, "hA_energy_balance")
            kwh_est = _feat(TWO_NODE_FEATURE_NAMES, f, "k_wh_energy_balance")
            assert abs(hA_est - hA_true) / hA_true < 1e-4, f"{profile} hA {hA_est:.5f} vs {hA_true}"
            assert abs(kwh_est - kwh_true) / kwh_true < 1e-4, f"{profile} k_wh {kwh_est:.5f} vs {kwh_true}"


def test_energy_balance_beats_steady_state_on_ramp():
    """The whole reason the integral estimator is the anchor: on a ramp the
    run never settles, so the steady-state formula is badly biased while the
    energy balance stays exact."""
    t = time_grid()
    rng = np.random.default_rng(5)
    I_t, _ = generate_load_profile("ramp", t, rng)
    hA_true = 24.0
    params = OneNodeParams(hA=hA_true, T_ambient=25.0)
    T = simulate_one_node(params, t, I_t, T0=25.0)

    f = extract_one_node_features(t, I_t, T, 25.0)
    err_energy = abs(_feat(ONE_NODE_FEATURE_NAMES, f, "hA_energy_balance") - hA_true)
    err_ss = abs(_feat(ONE_NODE_FEATURE_NAMES, f, "hA_steady_state") - hA_true)
    assert err_energy < err_ss / 3, f"energy-balance err {err_energy:.3f} vs steady-state err {err_ss:.3f}"


# ---------------------------------------------------------------------------
# Feature-vector hygiene
# ---------------------------------------------------------------------------

def test_feature_vector_lengths_match_names():
    t = time_grid()
    rng = np.random.default_rng(1)
    I_t, _ = generate_load_profile("step", t, rng)

    T = simulate_one_node(OneNodeParams(hA=15.0, T_ambient=25.0), t, I_t, T0=25.0)
    assert extract_one_node_features(t, I_t, T, 25.0).shape == (len(ONE_NODE_FEATURE_NAMES),)

    T2 = simulate_two_node(TwoNodeParams(hA=15.0, k_wh=35.0, T_ambient=25.0), t, I_t, T0=np.array([25.0, 25.0]))
    assert extract_two_node_features(t, I_t, T2[:, 0], T2[:, 1], 25.0).shape == (len(TWO_NODE_FEATURE_NAMES),)


def test_features_finite_on_degenerate_zero_current_run():
    """A run with no excitation at all has no information to extract, but it
    must still produce a finite feature vector rather than inf/nan -- a single
    non-finite row would poison the whole training batch."""
    t = time_grid()
    I_t = np.zeros_like(t)
    T = np.full_like(t, 25.0)

    f1 = extract_one_node_features(t, I_t, T, 25.0)
    assert np.all(np.isfinite(f1))

    f2 = extract_two_node_features(t, I_t, T, T, 25.0)
    assert np.all(np.isfinite(f2))


def test_features_survive_realistic_noise():
    """At the worst noise level the dataset generates (sigma = 2 K), the anchor
    must stay usable. Averaged over draws rather than checked on one, since a
    single unlucky trace says nothing about the estimator."""
    t = time_grid()
    hA_true = 15.0
    errors = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        I_t, _ = generate_load_profile("step", t, rng)
        T = simulate_one_node(OneNodeParams(hA=hA_true, T_ambient=25.0), t, I_t, T0=25.0)
        T_noisy = T + rng.normal(0.0, 2.0, size=T.shape)

        f = extract_one_node_features(t, I_t, T_noisy, 25.0)
        assert np.all(np.isfinite(f))
        hA_est = _feat(ONE_NODE_FEATURE_NAMES, f, "hA_energy_balance")
        errors.append(abs(hA_est - hA_true) / hA_true)

    mean_err = float(np.mean(errors))
    assert mean_err < 0.05, f"mean rel err {mean_err:.4f} over {len(errors)} noisy draws"


def test_noise_estimate_tracks_true_noise_level():
    t = time_grid()
    rng = np.random.default_rng(4)
    I_t, _ = generate_load_profile("step", t, rng)
    T = simulate_one_node(OneNodeParams(hA=15.0, T_ambient=25.0), t, I_t, T0=25.0)

    estimates = []
    for sigma in (0.1, 0.5, 2.0):
        T_noisy = T + rng.normal(0.0, sigma, size=T.shape)
        f = extract_one_node_features(t, I_t, T_noisy, 25.0)
        estimates.append(_feat(ONE_NODE_FEATURE_NAMES, f, "noise_std_est"))

    assert estimates[0] < estimates[1] < estimates[2], f"noise estimates not monotone: {estimates}"


# ---------------------------------------------------------------------------
# Model behavior
# ---------------------------------------------------------------------------

def _small_one_node_xy(n_train=400, n_test=150):
    train = generate_one_node_dataset(n_runs=n_train, seed=11)
    test = generate_one_node_dataset(n_runs=n_test, seed=12)
    return build_one_node_xy(train), build_one_node_xy(test)


def test_mlp_matches_closed_form_on_full_window():
    """On a full, settled 3000s window the closed-form energy balance is
    already near-optimal (~1.8% MAPE), so the honest bar for the MLP here is
    parity, not victory -- it must not *degrade* the physics it is anchored
    to. The regime where it genuinely wins is covered by the next test."""
    (X_tr, y_tr), (X_te, y_te) = _small_one_node_xy()
    model = MLPCalibrator(testbed="one_node", random_state=0).fit(X_tr, y_tr)

    mlp = regression_metrics(y_te, model.predict(X_te), ["hA"])["hA"]
    anchor = X_te[:, [ONE_NODE_FEATURE_NAMES.index("hA_energy_balance")]]
    closed_form = regression_metrics(y_te, anchor, ["hA"])["hA"]

    assert mlp["rmse"] < 1.3 * closed_form["rmse"], (
        f"MLP RMSE {mlp['rmse']:.3f} vs closed form {closed_form['rmse']:.3f}"
    )
    assert mlp["mape_pct"] < 5.0
    assert mlp["r2"] > 0.95


def test_mlp_beats_closed_form_on_short_window():
    """The actual value proposition: on a truncated window the run has not
    settled, the closed-form estimator degrades sharply, and the learned
    correction recovers a large part of that loss. This is the
    partial-convergence/streaming regime the proposal is aimed at.

    Training size matters here and the threshold is sharp: measured on this
    setup the correction is worth nothing at 800 training runs (21.8% vs the
    closed form's 21.7%) and worth ~35% at 1500+ (14.2%). 2000 is used below
    to sit clear of that knee -- worth knowing, since it says the ML approach
    only pays off once enough simulated runs have been generated.
    """
    n = 120  # 600s of a 3000s run -- well short of the slowest thermal tau
    train = generate_one_node_dataset(n_runs=2000, seed=11)
    test = generate_one_node_dataset(n_runs=250, seed=12)
    X_tr, y_tr = build_one_node_xy(train, n_samples=n)
    X_te, y_te = build_one_node_xy(test, n_samples=n)

    model = MLPCalibrator(testbed="one_node", target_mode="log_residual", random_state=0).fit(X_tr, y_tr)
    mlp = regression_metrics(y_te, model.predict(X_te), ["hA"])["hA"]
    anchor = X_te[:, [ONE_NODE_FEATURE_NAMES.index("hA_energy_balance")]]
    closed_form = regression_metrics(y_te, anchor, ["hA"])["hA"]

    assert mlp["mape_pct"] < 0.85 * closed_form["mape_pct"], (
        f"short window: MLP MAPE {mlp['mape_pct']:.2f}% vs closed form {closed_form['mape_pct']:.2f}%"
    )


def test_all_target_modes_train_and_predict():
    (X_tr, y_tr), (X_te, y_te) = _small_one_node_xy(n_train=300, n_test=100)
    for mode in ("raw", "residual", "log_residual"):
        model = MLPCalibrator(testbed="one_node", target_mode=mode, random_state=0).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        assert pred.shape == (X_te.shape[0], 1)
        assert np.all(np.isfinite(pred))


def test_mlp_two_node_predicts_both_constants():
    train = generate_two_node_dataset(n_runs=400, seed=21)
    test = generate_two_node_dataset(n_runs=120, seed=22)
    X_tr, y_tr = build_two_node_xy(train)
    X_te, y_te = build_two_node_xy(test)

    model = MLPCalibrator(testbed="two_node", random_state=0).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    assert pred.shape == (X_te.shape[0], 2)

    metrics = regression_metrics(y_te, pred, ["hA", "k_wh"])
    assert metrics["hA"]["mape_pct"] < 12.0
    assert metrics["k_wh"]["mape_pct"] < 15.0


def test_save_load_roundtrip_preserves_predictions():
    (X_tr, y_tr), (X_te, _) = _small_one_node_xy(n_train=300, n_test=50)
    model = MLPCalibrator(testbed="one_node", random_state=0).fit(X_tr, y_tr)
    before = model.predict(X_te)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.joblib")
        model.save(path)
        reloaded = MLPCalibrator.load(path)

    assert reloaded.target_mode == model.target_mode
    np.testing.assert_allclose(reloaded.predict(X_te), before)


def test_predict_before_fit_raises():
    model = MLPCalibrator(testbed="one_node")
    try:
        model.predict(np.zeros((1, len(ONE_NODE_FEATURE_NAMES))))
    except RuntimeError:
        return
    raise AssertionError("predict() before fit() should raise RuntimeError")


# ---------------------------------------------------------------------------
# Baseline-compatible interface (what lets the benchmark treat ML as method #6)
# ---------------------------------------------------------------------------

def test_calibrate_one_node_matches_baseline_interface():
    (X_tr, y_tr), _ = _small_one_node_xy(n_train=400, n_test=1)
    model = MLPCalibrator(testbed="one_node", random_state=0).fit(X_tr, y_tr)

    t = time_grid()
    rng = np.random.default_rng(31)
    I_t, _ = generate_load_profile("step", t, rng)
    hA_true = 15.0
    T = simulate_one_node(OneNodeParams(hA=hA_true, T_ambient=25.0), t, I_t, T0=25.0)
    T_meas = T + rng.normal(0.0, 0.5, size=T.shape)

    result = model.calibrate_one_node(t, I_t, T_meas, 25.0)
    assert isinstance(result, CalibrationResult)
    assert set(result.params) == {"hA"}
    assert result.n_evals == 0  # zero simulator rollouts -- the headline speed claim
    assert result.converged
    assert result.runtime_s >= 0.0
    assert abs(result.params["hA"] - hA_true) / hA_true < 0.15


def test_calibrate_two_node_matches_baseline_interface():
    train = generate_two_node_dataset(n_runs=400, seed=41)
    X_tr, y_tr = build_two_node_xy(train)
    model = MLPCalibrator(testbed="two_node", random_state=0).fit(X_tr, y_tr)

    t = time_grid()
    rng = np.random.default_rng(42)
    I_t, _ = generate_load_profile("duty_cycle", t, rng)
    hA_true, kwh_true = 15.0, 35.0
    T = simulate_two_node(TwoNodeParams(hA=hA_true, k_wh=kwh_true, T_ambient=25.0), t, I_t, T0=np.array([25.0, 25.0]))
    T_w = T[:, 0] + rng.normal(0.0, 0.5, size=t.shape)
    T_h = T[:, 1] + rng.normal(0.0, 0.5, size=t.shape)

    result = model.calibrate_two_node(t, I_t, T_w, T_h, 25.0)
    assert isinstance(result, CalibrationResult)
    assert set(result.params) == {"hA", "k_wh"}
    assert result.n_evals == 0
    assert abs(result.params["hA"] - hA_true) / hA_true < 0.2
    assert abs(result.params["k_wh"] - kwh_true) / kwh_true < 0.2


def test_wrong_testbed_calibration_raises():
    (X_tr, y_tr), _ = _small_one_node_xy(n_train=300, n_test=1)
    model = MLPCalibrator(testbed="one_node", random_state=0).fit(X_tr, y_tr)
    t = time_grid()
    try:
        model.calibrate_two_node(t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), 25.0)
    except RuntimeError:
        return
    raise AssertionError("calibrate_two_node() on a one_node model should raise RuntimeError")


def _run_all():
    tests = [
        test_energy_balance_recovers_hA_one_node_all_profiles,
        test_energy_balance_recovers_both_constants_two_node_all_profiles,
        test_energy_balance_beats_steady_state_on_ramp,
        test_feature_vector_lengths_match_names,
        test_features_finite_on_degenerate_zero_current_run,
        test_features_survive_realistic_noise,
        test_noise_estimate_tracks_true_noise_level,
        test_mlp_matches_closed_form_on_full_window,
        test_mlp_beats_closed_form_on_short_window,
        test_all_target_modes_train_and_predict,
        test_mlp_two_node_predicts_both_constants,
        test_save_load_roundtrip_preserves_predictions,
        test_predict_before_fit_raises,
        test_calibrate_one_node_matches_baseline_interface,
        test_calibrate_two_node_matches_baseline_interface,
        test_wrong_testbed_calibration_raises,
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
