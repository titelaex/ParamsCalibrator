"""
The ML calibration model: a small MLP over physics-motivated features
(src/ml/features.py).

Architecture follows the proposal (Section 6): features -> Dense(32, ReLU)
-> Dense(16, ReLU) -> one output unit per calibrated constant, MSE loss,
Adam. Deliberately small -- the underlying physics is a smooth first/second
order thermal response, not a hard pattern-recognition problem, so a heavy
architecture would buy nothing and would be much harder to train well in the
time available.

Both inputs and targets are standardized. Targets especially: hA (8-25 W/K)
and k_wh (15-60 W/K) live on different numeric scales, so an unnormalized
MSE would silently weight k_wh errors ~6x more than hA errors just because
its numbers are bigger. Standardizing y makes the loss weight both constants
equally, which is what the benchmark's per-constant error metrics assume.

Target parameterization ("anchored residual", the default) is the one design
choice here that is not in the original proposal, and it came out of measured
results rather than taste. Asking the network for the *absolute* constant
made it slightly worse than the closed-form energy-balance estimator it is
handed as a feature (test MAPE 4.5% vs 4.0% on the 1-node testbed) -- the
network was spending its capacity re-deriving a formula it already had. Asking
it instead for the *correction* to that estimator

    hA_predicted = hA_energy_balance + net(features)          ("residual")
    hA_predicted = hA_energy_balance * exp(net(features))     ("log_residual")

makes the physics the default answer and learning the correction on top,
which beat the closed form on both error metrics on both testbeds. This is
exactly the framing the proposal argues for -- "a learned generalization of a
known physical formula" -- just made literal in the architecture, and it also
means a degenerate/untrained network degrades to the physics rather than to
nonsense. Set `target_mode="raw"` to reproduce the unanchored variant.

The class exposes `calibrate_one_node` / `calibrate_two_node` with the same
signature and the same `CalibrationResult` return type as every classical
baseline (src/baselines/), so the benchmark harness can treat the ML model
as just another method. The headline difference shows up in the result
object itself: `n_evals=0`, because a forward pass needs zero simulator
rollouts, against the hundreds that GA/PSO need per calibration.
"""

import time

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.baselines.base import CalibrationResult
from src.ml.features import (
    ONE_NODE_FEATURE_NAMES,
    ONE_NODE_TARGET_NAMES,
    TWO_NODE_FEATURE_NAMES,
    TWO_NODE_TARGET_NAMES,
    extract_one_node_features,
    extract_two_node_features,
)
from src.simulator.params import (
    C_HOUSING_J_PER_K,
    C_LUMPED_J_PER_K,
    C_WINDING_J_PER_K,
    R_WINDING_OHM,
)


# Which feature is the closed-form physics estimator for each calibrated
# constant. These are the anchors the network learns a correction to.
ANCHOR_FEATURES = {
    "one_node": {"hA": "hA_energy_balance"},
    "two_node": {"hA": "hA_energy_balance", "k_wh": "k_wh_energy_balance"},
}

# Floor applied to an anchor before it is used as a multiplicative base, so a
# degenerate run whose closed-form estimator collapsed to 0 (see _safe_div in
# features.py) cannot produce log(0).
_ANCHOR_FLOOR = 1.0


class MLPCalibrator:
    """MLP regressor from sensor-window features to calibrated physical constants."""

    def __init__(
        self,
        testbed: str,
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        learning_rate_init=1e-3,
        alpha=1e-4,
        early_stopping=True,
        n_iter_no_change=40,
        random_state=0,
        target_mode: str = "residual",
    ):
        if testbed not in ("one_node", "two_node"):
            raise ValueError(f"testbed must be 'one_node' or 'two_node', got {testbed!r}")
        if target_mode not in ("raw", "residual", "log_residual"):
            raise ValueError(f"unknown target_mode {target_mode!r}")
        self.testbed = testbed
        self.target_mode = target_mode
        self.feature_names = (
            ONE_NODE_FEATURE_NAMES if testbed == "one_node" else TWO_NODE_FEATURE_NAMES
        )
        self.target_names = (
            ONE_NODE_TARGET_NAMES if testbed == "one_node" else TWO_NODE_TARGET_NAMES
        )
        self._anchor_cols = [
            self.feature_names.index(ANCHOR_FEATURES[testbed][name])
            for name in self.target_names
        ]

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.net = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
        )
        self._fitted = False

    # -- target parameterization -------------------------------------------

    def _anchors(self, X) -> np.ndarray:
        """(n_runs, n_targets) closed-form physics estimate for each constant."""
        return X[:, self._anchor_cols]

    def _to_net_target(self, y, anchors):
        """Physical constants -> what the network is actually asked to predict."""
        if self.target_mode == "raw":
            return y
        if self.target_mode == "residual":
            return y - anchors
        return np.log(np.maximum(y, _ANCHOR_FLOOR)) - np.log(np.maximum(anchors, _ANCHOR_FLOOR))

    def _from_net_target(self, z, anchors):
        """Network output -> physical constants."""
        if self.target_mode == "raw":
            return z
        if self.target_mode == "residual":
            return z + anchors
        return np.exp(z) * np.maximum(anchors, _ANCHOR_FLOOR)

    # -- training ----------------------------------------------------------

    def fit(self, X, y):
        """X: (n_runs, n_features). y: (n_runs, n_targets)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        z = self._to_net_target(y, self._anchors(X))
        Xs = self.x_scaler.fit_transform(X)
        zs = self.y_scaler.fit_transform(z)
        # sklearn wants a 1-D target for single-output regression
        self.net.fit(Xs, zs.ravel() if zs.shape[1] == 1 else zs)
        self._fitted = True
        return self

    # -- inference ---------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Returns (n_runs, n_targets) in physical units."""
        if not self._fitted:
            raise RuntimeError("MLPCalibrator.predict() called before fit()/load()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        zs = self.net.predict(self.x_scaler.transform(X))
        if zs.ndim == 1:
            zs = zs.reshape(-1, 1)
        z = self.y_scaler.inverse_transform(zs)
        return self._from_net_target(z, self._anchors(X))

    def predict_dict(self, X) -> list[dict]:
        """Same as predict(), but as a list of {constant_name: value} dicts."""
        preds = self.predict(X)
        return [dict(zip(self.target_names, row)) for row in preds]

    # -- baseline-compatible interface -------------------------------------

    def calibrate_one_node(
        self, t, I_t, T_measured, T_ambient, T0=None,
        R_winding=R_WINDING_OHM, C=C_LUMPED_J_PER_K,
        bounds=None, rng=None, **_ignored,
    ) -> CalibrationResult:
        """Same signature/return type as src.baselines.*.calibrate_one_node.

        `T0`, `bounds` and `rng` are accepted and ignored: the network needs
        no initial guess, no search bounds and no randomness at inference
        time -- which is itself one of the results worth reporting, since
        every classical baseline's convergence robustness depends on exactly
        those things.
        """
        if self.testbed != "one_node":
            raise RuntimeError(f"this calibrator was trained for {self.testbed}")
        t0 = time.perf_counter()
        feats = extract_one_node_features(t, I_t, T_measured, T_ambient, R_winding, C)
        pred = self.predict(feats)[0]
        runtime_s = time.perf_counter() - t0

        return CalibrationResult(
            params={"hA": float(pred[0])},
            runtime_s=runtime_s,
            n_evals=0,  # zero simulator rollouts -- a single forward pass
            converged=True,
            extra={"features": dict(zip(self.feature_names, feats))},
        )

    def calibrate_two_node(
        self, t, I_t, T_w_measured, T_h_measured, T_ambient, T0=None,
        R_winding=R_WINDING_OHM, C_w=C_WINDING_J_PER_K, C_h=C_HOUSING_J_PER_K,
        bounds=None, rng=None, **_ignored,
    ) -> CalibrationResult:
        """Same signature/return type as src.baselines.*.calibrate_two_node."""
        if self.testbed != "two_node":
            raise RuntimeError(f"this calibrator was trained for {self.testbed}")
        t0 = time.perf_counter()
        feats = extract_two_node_features(
            t, I_t, T_w_measured, T_h_measured, T_ambient, R_winding, C_w, C_h
        )
        pred = self.predict(feats)[0]
        runtime_s = time.perf_counter() - t0

        return CalibrationResult(
            params={"hA": float(pred[0]), "k_wh": float(pred[1])},
            runtime_s=runtime_s,
            n_evals=0,
            converged=True,
            extra={"features": dict(zip(self.feature_names, feats))},
        )

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "testbed": self.testbed,
                "target_mode": self.target_mode,
                "x_scaler": self.x_scaler,
                "y_scaler": self.y_scaler,
                "net": self.net,
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "anchor_cols": self._anchor_cols,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MLPCalibrator":
        blob = joblib.load(path)
        obj = cls(testbed=blob["testbed"], target_mode=blob["target_mode"])
        obj.x_scaler = blob["x_scaler"]
        obj.y_scaler = blob["y_scaler"]
        obj.net = blob["net"]
        obj.feature_names = blob["feature_names"]
        obj.target_names = blob["target_names"]
        obj._anchor_cols = blob["anchor_cols"]
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Evaluation helpers (shared by the training script and the benchmark)
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred, target_names) -> dict:
    """Per-constant MAE / RMSE / MAPE / R^2."""
    y_true = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)

    out = {}
    for j, name in enumerate(target_names):
        err = y_pred[:, j] - y_true[:, j]
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2))
        out[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape_pct": float(np.mean(np.abs(err / y_true[:, j])) * 100.0),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }
    return out
