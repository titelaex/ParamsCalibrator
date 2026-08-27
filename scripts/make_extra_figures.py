#!/usr/bin/env python3
"""Three additional benchmark figures, built from the raw per-run CSVs and
summary.json already produced by scripts/run_benchmark.py. Reuses the exact
style/palette constants from that script so the whole deck reads as one
visual system.

Usage (after scripts/run_benchmark.py has produced reports/benchmark_results/):
    python scripts/make_extra_figures.py

Produces, under reports/figures/:
    speed_distribution.png       box plot of all 25 per-run runtimes, not just the median
    speed_accuracy_tradeoff.png  one point per method: median speed vs. median accuracy
    scalability.png              runtime & accuracy ratio, 2-node testbed / 1-node testbed
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "benchmark_results")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- exact style constants from scripts/run_benchmark.py -------------------
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
METHOD_LABEL = {"ga": "GA", "pso": "PSO", "lm": "LM", "bayesopt": "BayesOpt", "ekf": "EKF", "mlp": "MLP"}


def style_axes(ax, ylabel=None, log=False):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    if log:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def suptitle(fig, text):
    fig.suptitle(text, fontsize=12, color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold")


def main():
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        summary = json.load(f)

    df1 = pd.read_csv(os.path.join(RESULTS_DIR, "raw_one_node_accuracy_speed.csv"))
    df2 = pd.read_csv(os.path.join(RESULTS_DIR, "raw_two_node_accuracy_speed.csv"))
    df1["runtime_ms"] = df1["runtime_s"] * 1000
    df2["runtime_ms"] = df2["runtime_s"] * 1000

    # =========================================================================
    # 1. Speed distribution (box plot, log scale) -- 1-node | 2-node
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    suptitle(fig, "ParamsCalibrator - speed distribution across all 25 runs (not just the median)")

    for ax, df, title in zip(axes, [df1, df2], ["1-node testbed", "2-node testbed"]):
        data = [df.loc[df["method"] == m, "runtime_ms"].values for m in METHOD_ORDER]
        bp = ax.boxplot(
            data, positions=range(len(METHOD_ORDER)), widths=0.55, patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_MUTED, markeredgecolor="none", alpha=0.6),
            medianprops=dict(color=INK_PRIMARY, linewidth=1.6),
            whiskerprops=dict(color=BASELINE, linewidth=1.2),
            capprops=dict(color=BASELINE, linewidth=1.2),
            boxprops=dict(linewidth=0.8, edgecolor=INK_PRIMARY),
        )
        for patch, m in zip(bp["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.85)
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=8, color=INK_PRIMARY, rotation=20)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
        style_axes(ax, ylabel="runtime (ms, log scale)" if ax is axes[0] else None, log=True)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "speed_distribution.png"), dpi=160, facecolor=SURFACE)
    print("saved reports/figures/speed_distribution.png")

    # =========================================================================
    # 2. Speed vs. accuracy trade-off (scatter, direct-labeled) -- 1-node | 2-node
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    suptitle(fig, "ParamsCalibrator - speed vs. accuracy: one point summarises each method")

    # GA and PSO land on nearly identical (speed, accuracy) values in both
    # testbeds -- their default labels collide, so nudge them apart deliberately.
    label_offset = {"ga": (8, 11), "pso": (8, -14)}

    for ax, testbed, title in zip(axes, ["one_node", "two_node"], ["1-node testbed", "2-node testbed"]):
        acc = summary[testbed]["accuracy"]
        speed = summary[testbed]["speed"]
        for m in METHOD_ORDER:
            x = speed[m]["runtime_ms_median"]
            y = acc[m]["hA_mape_median"]
            ax.scatter([x], [y], s=90, color=METHOD_COLORS[m], zorder=3, edgecolor=SURFACE, linewidth=1)
            ax.annotate(
                METHOD_LABEL[m], (x, y), textcoords="offset points", xytext=label_offset.get(m, (7, 5)),
                fontsize=8.5, color=INK_PRIMARY, fontweight="bold" if m == "mlp" else "normal",
            )
        ax.set_xscale("log")
        ax.set_xlabel("median runtime (ms, log scale)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
        style_axes(ax, ylabel="median hA error (MAPE %)" if ax is axes[0] else None)
        ax.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "speed_accuracy_tradeoff.png"), dpi=160, facecolor=SURFACE)
    print("saved reports/figures/speed_accuracy_tradeoff.png")

    # =========================================================================
    # 3. Scalability (bar) -- runtime ratio (log) | accuracy ratio (linear)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    suptitle(fig, "ParamsCalibrator - scalability: 1 parameter -> 2 parameters calibrated simultaneously")

    scal = summary["scalability"]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

    ax = axes[0]
    vals = [scal[m]["runtime_ratio_2n_over_1n"] for m in METHOD_ORDER]
    bars = ax.bar(range(len(METHOD_ORDER)), vals, color=colors, width=0.6)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(len(METHOD_ORDER) - 0.4, 1.0, " no penalty (1x)", color=INK_MUTED, fontsize=7.5, va="bottom")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=8, color=INK_PRIMARY, rotation=20)
    ax.set_title("Runtime ratio (2-node / 1-node)", fontsize=10, color=INK_PRIMARY, loc="left")
    style_axes(ax, ylabel="runtime ratio (log scale)", log=True)

    ax = axes[1]
    vals = [scal[m]["hA_mape_ratio_2n_over_1n"] for m in METHOD_ORDER]
    bars = ax.bar(range(len(METHOD_ORDER)), vals, color=colors, width=0.6)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=8, color=INK_PRIMARY, rotation=20)
    ax.set_title("hA error ratio (2-node / 1-node)", fontsize=10, color=INK_PRIMARY, loc="left")
    style_axes(ax, ylabel="error ratio (MAPE)")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "scalability.png"), dpi=160, facecolor=SURFACE)
    print("saved reports/figures/scalability.png")


if __name__ == "__main__":
    main()
