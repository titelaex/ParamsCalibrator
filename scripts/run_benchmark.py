#!/usr/bin/env python3
"""Run the full benchmark: all 6 calibration methods x 5 evaluation axes,
on both testbeds.

Usage:
    python scripts/generate_datasets.py   # once
    python scripts/train_ml.py            # once
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --quick   # small run counts, for a fast smoke test

Produces (all under reports/):
    benchmark_results/raw_*.csv        every individual calibration, unaggregated
    benchmark_results/summary.json     every summary table, machine-readable
    figures/benchmark_one_node.png     accuracy / speed / robustness / streaming panel
    figures/benchmark_two_node.png     same, for the 2-node testbed

Runtime is dominated by Bayesian Optimization (each calibration fits a
Gaussian Process from scratch) and by the population heuristics (GA/PSO run
hundreds of full simulator rollouts each) -- a full run takes on the order of
20-30 minutes; --quick finishes in under a minute and is meant to sanity-check
the pipeline, not to produce presentation numbers.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark import metrics
from src.benchmark.harness import run_accuracy_speed, run_robustness, run_window_sweep
from src.benchmark.registry import build_methods
from src.ml.model import MLPCalibrator
from src.simulator.data_generator import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "benchmark_results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

WINDOW_FRACS = (0.2, 0.4, 0.7, 1.0)

# dataviz skill reference palette (light mode) -- kept consistent with
# scripts/validate_simulator_plot.py
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
METHOD_COLORS = {
    "ga": "#c3752a", "pso": "#caa63d", "lm": "#8a63c9",
    "bayesopt": "#d1487a", "ekf": "#1baf7a", "mlp": "#2a78d6",
}
METHOD_ORDER = ["ga", "pso", "lm", "bayesopt", "ekf", "mlp"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny run counts, for pipeline smoke-testing only")
    p.add_argument("--n-main", type=int, default=None, help="test runs for accuracy/speed (default 25, quick 4)")
    p.add_argument("--n-robust", type=int, default=None, help="test runs for robustness (default 5, quick 2)")
    p.add_argument("--n-restarts", type=int, default=None, help="restarts per robustness run (default 5, quick 2)")
    p.add_argument("--n-sweep", type=int, default=None, help="test runs for window sweep (default 5, quick 2)")
    return p.parse_args()


def _ordered(index):
    return [m for m in METHOD_ORDER if m in index]


_TESTBED_FILE_TAG = {"one_node": "1node", "two_node": "2node"}


def load_testbed(testbed):
    data = load_dataset(os.path.join(DATA_DIR, f"motor_{_TESTBED_FILE_TAG[testbed]}_test.npz"))
    model_path = os.path.join(MODEL_DIR, f"mlp_{testbed}.joblib")
    mlp = MLPCalibrator.load(model_path) if os.path.exists(model_path) else None
    if mlp is None:
        print(f"  WARNING: {model_path} not found -- MLP omitted from this testbed's benchmark. "
              f"Run scripts/train_ml.py first.")
    return data, mlp


def run_testbed(testbed, n_main, n_robust, n_restarts, n_sweep):
    print(f"\n{'=' * 78}\n{testbed} testbed\n{'=' * 78}")
    data, mlp = load_testbed(testbed)
    methods = build_methods(**{f"mlp_{testbed}": mlp} if mlp else {})
    targets = ["hA"] if testbed == "one_node" else ["hA", "k_wh"]

    print(f"  [1/3] accuracy + speed  ({n_main} runs x {len(methods)} methods)")
    df_main = run_accuracy_speed(testbed, data, methods, n_runs=n_main)

    print(f"  [2/3] robustness        ({n_robust} runs x {n_restarts} restarts x {len(methods)} methods)")
    df_robust = run_robustness(testbed, data, methods, run_indices=list(range(n_robust)), n_restarts=n_restarts)

    print(f"  [3/3] window sweep      ({n_sweep} runs x {len(WINDOW_FRACS)} windows x {len(methods)} methods)")
    df_sweep = run_window_sweep(testbed, data, methods, run_indices=list(range(n_sweep)), window_fracs=list(WINDOW_FRACS))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_main.to_csv(os.path.join(RESULTS_DIR, f"raw_{testbed}_accuracy_speed.csv"), index=False)
    df_robust.to_csv(os.path.join(RESULTS_DIR, f"raw_{testbed}_robustness.csv"), index=False)
    df_sweep.to_csv(os.path.join(RESULTS_DIR, f"raw_{testbed}_window_sweep.csv"), index=False)

    acc = metrics.accuracy_table(df_main)
    speed = metrics.speed_table(df_main)
    robust = metrics.robustness_table(df_robust)
    update_cost = metrics.update_cost_table(df_main)

    print(f"\n  accuracy (median MAPE %):\n{acc.round(3).to_string()}")
    print(f"\n  speed:\n{speed.round(3).to_string()}")
    print(f"\n  robustness (20%-tolerance success rate):\n{robust.round(3).to_string()}")
    print(f"\n  per-new-sample update cost (ms):\n{update_cost.round(4).to_string()}")

    return {
        "data": data, "methods": methods, "targets": targets,
        "df_main": df_main, "df_robust": df_robust, "df_sweep": df_sweep,
        "acc": acc, "speed": speed, "robust": robust, "update_cost": update_cost,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_axes(ax, ylabel=None, log=False):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    if log:
        ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def _bar(ax, methods, values, ylabel, log=False, fmt="{:.2f}"):
    colors = [METHOD_COLORS[m] for m in methods]
    bars = ax.bar(methods, values, color=colors, width=0.6)
    _style_axes(ax, ylabel, log=log)
    ax.tick_params(axis="x", labelsize=8, colors=INK_PRIMARY, rotation=20)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)


def plot_testbed_summary(testbed, result, out_path):
    methods = _ordered(result["acc"].index)
    acc, speed, robust = result["acc"], result["speed"], result["robust"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor=SURFACE)
    fig.suptitle(f"ParamsCalibrator benchmark -- {testbed} testbed", fontsize=12,
                color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold")

    _bar(axes[0], methods, acc.loc[methods, "hA_mape_median"], "hA MAPE % (median)")
    axes[0].set_title("1. Accuracy", fontsize=9, color=INK_PRIMARY, loc="left")

    _bar(axes[1], methods, speed.loc[methods, "runtime_ms_median"], "runtime, ms (log)", log=True, fmt="{:.1f}")
    axes[1].set_title("2. Speed", fontsize=9, color=INK_PRIMARY, loc="left")

    _bar(axes[2], methods, robust.loc[methods, "hA_success_rate"] * 100, "success rate % (<20% err)", fmt="{:.0f}")
    axes[2].set_ylim(0, 115)
    axes[2].set_title("3. Convergence robustness", fontsize=9, color=INK_PRIMARY, loc="left")

    sweep_table = metrics.streaming_table(result["df_sweep"], target="hA")
    for m in _ordered(sweep_table.index):
        axes[3].plot(sweep_table.columns, sweep_table.loc[m], marker="o", markersize=3,
                    linewidth=1.6, color=METHOD_COLORS[m], label=m)
    _style_axes(axes[3], "hA MAPE % (median)", log=True)
    axes[3].set_xlabel("observation window fraction", color=INK_SECONDARY, fontsize=8)
    axes[3].set_title("5. Streaming / partial-window accuracy", fontsize=9, color=INK_PRIMARY, loc="left")
    axes[3].legend(frameon=False, fontsize=6.5, labelcolor=INK_SECONDARY, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    n_main = args.n_main or (4 if args.quick else 25)
    n_robust = args.n_robust or (2 if args.quick else 5)
    n_restarts = args.n_restarts or (2 if args.quick else 5)
    n_sweep = args.n_sweep or (2 if args.quick else 5)

    results = {}
    for testbed in ("one_node", "two_node"):
        results[testbed] = run_testbed(testbed, n_main, n_robust, n_restarts, n_sweep)
        plot_testbed_summary(testbed, results[testbed],
                            os.path.join(FIGURES_DIR, f"benchmark_{testbed}.png"))

    print(f"\n{'=' * 78}\nscalability: one_node -> two_node\n{'=' * 78}")
    scal = metrics.scalability_table(
        results["one_node"]["acc"], results["one_node"]["speed"],
        results["two_node"]["acc"], results["two_node"]["speed"],
    )
    print(scal.round(3).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        "n_main": n_main, "n_robust": n_robust, "n_restarts": n_restarts, "n_sweep": n_sweep,
        "scalability": json.loads(scal.round(4).to_json(orient="index")),
    }
    for testbed, r in results.items():
        summary[testbed] = {
            "accuracy": json.loads(r["acc"].round(4).to_json(orient="index")),
            "speed": json.loads(r["speed"].round(4).to_json(orient="index")),
            "robustness": json.loads(r["robust"].round(4).to_json(orient="index")),
            "update_cost_ms": json.loads(r["update_cost"].round(4).to_json(orient="index")),
            "streaming_mape_hA": json.loads(
                metrics.streaming_table(r["df_sweep"], target="hA").round(4).to_json(orient="index")
            ),
        }
    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary -> {out_path}")


if __name__ == "__main__":
    main()
