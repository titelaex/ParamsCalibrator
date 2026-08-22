#!/usr/bin/env python3
"""Train the MLP calibration model on both testbeds and report held-out accuracy.

Usage:
    python scripts/generate_datasets.py    # once, to create data/
    python scripts/train_ml.py

Besides the usual train/val/test metrics, this reports two things that the
proposal's headline framing depends on.

First, the accuracy of the raw `hA_energy_balance` / `k_wh_energy_balance`
features on their own -- i.e. the closed-form physics estimator with no
learning at all. That is the honest bar the ML model has to clear.

Second, a sweep over observation-window length. That sweep is what actually
locates the ML model's value: given a full settled 3000s window the
closed-form estimator is already near-optimal and the MLP only matches it,
but as the window shortens the run no longer approaches steady state, the
closed form degrades sharply, and the learned correction recovers a large
part of the loss. Since a short window is exactly what a streaming
calibration deployment has to work with, that is the regime the comparison
should be made in -- and a nuanced "wins where it matters" result is more
credible than a blanket claim (proposal, Section 12).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.features import (
    ONE_NODE_FEATURE_NAMES,
    ONE_NODE_TARGET_NAMES,
    TWO_NODE_FEATURE_NAMES,
    TWO_NODE_TARGET_NAMES,
    build_one_node_xy,
    build_two_node_xy,
)
from src.ml.model import MLPCalibrator, regression_metrics
from src.simulator.data_generator import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _fmt(metrics, target_names):
    parts = []
    for name in target_names:
        m = metrics[name]
        parts.append(f"{name}: MAPE {m['mape_pct']:5.2f}%  MAE {m['mae']:6.3f}  RMSE {m['rmse']:6.3f}  R2 {m['r2']:6.4f}")
    return "  |  ".join(parts)


def _closed_form_baseline(X, y, feature_names, target_names, feature_for_target):
    """Accuracy of the raw analytical estimator features, with no learning."""
    cols = [feature_names.index(feature_for_target[name]) for name in target_names]
    y_pred = X[:, cols]
    return regression_metrics(y, y_pred, target_names)


WINDOW_FRACTIONS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def window_sweep(testbed, data, build_xy, feature_names, target_names, feature_for_target):
    """Closed-form vs MLP accuracy as the observation window is truncated."""
    n_full = data["train"]["t"].shape[0]
    dt = float(data["train"]["t"][1] - data["train"]["t"][0])
    anchor_cols = [feature_names.index(feature_for_target[n]) for n in target_names]

    header = "    window            " + "".join(f"{n + ' closed':>16s}{n + ' MLP':>13s}" for n in target_names)
    print(f"\n  window-length sweep (MAPE, test set):")
    print(header)

    rows = []
    for frac in WINDOW_FRACTIONS:
        n = max(20, int(frac * n_full))
        X_tr, y_tr = build_xy(data["train"], n_samples=n)
        X_te, y_te = build_xy(data["test"], n_samples=n)

        cf = regression_metrics(y_te, X_te[:, anchor_cols], target_names)
        model = MLPCalibrator(testbed=testbed, target_mode="log_residual", random_state=0).fit(X_tr, y_tr)
        ml = regression_metrics(y_te, model.predict(X_te), target_names)

        line = f"    {frac * 100:4.0f}% ({n * dt:6.0f}s)"
        for name in target_names:
            line += f"{cf[name]['mape_pct']:14.2f}%{ml[name]['mape_pct']:12.2f}%"
        print(line)

        rows.append({
            "window_frac": frac,
            "n_samples": n,
            "window_s": n * dt,
            "closed_form": {k: v["mape_pct"] for k, v in cf.items()},
            "mlp": {k: v["mape_pct"] for k, v in ml.items()},
        })
    return rows


def train_testbed(testbed, splits, build_xy, feature_names, target_names, feature_for_target,
                  do_window_sweep=True):
    print(f"\n{'=' * 78}\n{testbed} testbed\n{'=' * 78}")

    data = {}
    for split in ("train", "val", "test"):
        path = os.path.join(DATA_DIR, splits[split])
        if not os.path.exists(path):
            print(f"  MISSING {path} -- run scripts/generate_datasets.py first.")
            return None
        data[split] = load_dataset(path)

    t0 = time.time()
    xy = {split: build_xy(d) for split, d in data.items()}
    print(f"  feature extraction: {time.time() - t0:.1f}s "
          f"({xy['train'][0].shape[0]} train / {xy['val'][0].shape[0]} val / {xy['test'][0].shape[0]} test runs, "
          f"{xy['train'][0].shape[1]} features)")

    # The bar to beat: the closed-form physics estimator alone.
    cf = _closed_form_baseline(*xy["test"], feature_names, target_names, feature_for_target)
    print(f"\n  closed-form estimator (no learning), test:")
    print(f"    {_fmt(cf, target_names)}")

    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]

    # Model selection over the target parameterization, scored on validation
    # only -- the test set stays untouched until the winner is chosen.
    candidates = {}
    print()
    for mode in ("raw", "residual", "log_residual"):
        t0 = time.time()
        m = MLPCalibrator(testbed=testbed, target_mode=mode, random_state=0).fit(X_train, y_train)
        elapsed = time.time() - t0
        val_metrics = regression_metrics(y_val, m.predict(X_val), target_names)
        score = float(np.mean([val_metrics[n]["mape_pct"] for n in target_names]))
        candidates[mode] = (m, score, elapsed)
        print(f"  target_mode={mode:12s} val mean MAPE {score:5.2f}%  ({elapsed:.1f}s, {m.net.n_iter_} epochs)")

    best_mode = min(candidates, key=lambda k: candidates[k][1])
    model, _, train_time = candidates[best_mode]
    print(f"  -> selected target_mode={best_mode!r}")

    results = {}
    print()
    for split in ("train", "val", "test"):
        X, y = xy[split]
        metrics = regression_metrics(y, model.predict(X), target_names)
        results[split] = metrics
        print(f"    {split:5s}  {_fmt(metrics, target_names)}")

    # Single-run inference latency -- the headline speed claim.
    X_test, _ = xy["test"]
    t0 = time.perf_counter()
    for i in range(min(200, X_test.shape[0])):
        model.predict(X_test[i])
    latency_ms = (time.perf_counter() - t0) / min(200, X_test.shape[0]) * 1000
    print(f"\n  single-run forward-pass latency: {latency_ms:.3f} ms (features already extracted)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"mlp_{testbed}.joblib")
    model.save(model_path)
    print(f"  saved -> {model_path}")

    sweep = None
    if do_window_sweep:
        sweep = window_sweep(testbed, data, build_xy, feature_names, target_names, feature_for_target)

    return {
        "testbed": testbed,
        "closed_form_test": cf,
        "window_sweep": sweep,
        "target_mode": best_mode,
        "val_mape_by_mode": {k: v[1] for k, v in candidates.items()},
        "mlp": results,
        "train_time_s": train_time,
        "latency_ms": latency_ms,
        "n_features": int(xy["train"][0].shape[1]),
        "n_train_runs": int(xy["train"][0].shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-window-sweep", action="store_true",
                        help="skip the observation-window sweep (roughly halves runtime)")
    args = parser.parse_args()
    sweep = not args.no_window_sweep

    summary = {}

    r1 = train_testbed(
        "one_node",
        {"train": "motor_1node_train.npz", "val": "motor_1node_val.npz", "test": "motor_1node_test.npz"},
        build_one_node_xy,
        ONE_NODE_FEATURE_NAMES,
        ONE_NODE_TARGET_NAMES,
        {"hA": "hA_energy_balance"},
        do_window_sweep=sweep,
    )
    if r1:
        summary["one_node"] = r1

    r2 = train_testbed(
        "two_node",
        {"train": "motor_2node_train.npz", "val": "motor_2node_val.npz", "test": "motor_2node_test.npz"},
        build_two_node_xy,
        TWO_NODE_FEATURE_NAMES,
        TWO_NODE_TARGET_NAMES,
        {"hA": "hA_energy_balance", "k_wh": "k_wh_energy_balance"},
        do_window_sweep=sweep,
    )
    if r2:
        summary["two_node"] = r2

    if summary:
        os.makedirs(MODEL_DIR, exist_ok=True)
        out = os.path.join(MODEL_DIR, "training_summary.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
