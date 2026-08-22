#!/usr/bin/env python3
"""Produce a validation figure: one example 1-node run and one example 2-node
run, showing the applied current profile and the resulting true vs. noisy
sensor temperature, with the analytical steady-state overlaid as a check.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator.load_profiles import generate_load_profile
from src.simulator.motor_thermal import (
    simulate_one_node,
    simulate_two_node,
    steady_state_one_node,
)
from src.simulator.params import OneNodeParams, TwoNodeParams
from src.simulator.sensors import add_noise

# palette (dataviz skill reference palette, light mode)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"    # categorical slot 1 -> winding / 1-node temperature
AQUA = "#1baf7a"    # categorical slot 3 -> housing


def style_axes(ax, ylabel):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def main():
    rng = np.random.default_rng(123)

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 6.5), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    fig.suptitle(
        "ParamsCalibrator — simulator validation: example synthetic runs",
        fontsize=12, color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold",
    )

    # ---------------- 1-node example ----------------
    hA = 15.0
    T_ambient = 25.0
    noise_std = 0.8
    t = np.arange(0.0, 3000.0, 5.0)
    I_t, meta = generate_load_profile("step", t, rng)
    params = OneNodeParams(hA=hA, T_ambient=T_ambient)
    T_true = simulate_one_node(params, t, I_t, T0=T_ambient)
    T_measured = add_noise(T_true, noise_std, rng)
    T_ss = steady_state_one_node(hA, params.R_winding, meta["level_A"], T_ambient)

    ax_i1, ax_t1 = axes[0, 0], axes[1, 0]
    ax_i1.plot(t / 60, I_t, color=INK_MUTED, linewidth=1.5)
    style_axes(ax_i1, "Current (A)")
    ax_i1.set_title("1-node testbed (single lumped mass)", fontsize=10, color=INK_PRIMARY, loc="left")

    ax_t1.plot(t / 60, T_true, color=BLUE, linewidth=2, label="true (simulated)")
    ax_t1.scatter(t / 60, T_measured, color=BLUE, alpha=0.18, s=8, label="measured (noisy sensor)")
    ax_t1.axhline(T_ss, color=INK_MUTED, linewidth=1, linestyle="--")
    ax_t1.text(
        t[-1] / 60, T_ss, f"  analytical T_ss={T_ss:.1f}°C", color=INK_MUTED,
        fontsize=8, va="center",
    )
    style_axes(ax_t1, "Winding temp (°C)")
    ax_t1.set_xlabel("Time (min)", color=INK_SECONDARY, fontsize=9)
    ax_t1.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_SECONDARY)

    # ---------------- 2-node example ----------------
    hA2 = 15.0
    k_wh = 35.0
    noise_std2 = 0.8
    I_t2, meta2 = generate_load_profile("duty_cycle", t, rng)
    params2 = TwoNodeParams(hA=hA2, k_wh=k_wh, T_ambient=T_ambient)
    T_true2 = simulate_two_node(params2, t, I_t2, T0=np.array([T_ambient, T_ambient]))
    T_w_true, T_h_true = T_true2[:, 0], T_true2[:, 1]
    T_w_meas = add_noise(T_w_true, noise_std2, rng)
    T_h_meas = add_noise(T_h_true, noise_std2, rng)

    ax_i2, ax_t2 = axes[0, 1], axes[1, 1]
    ax_i2.plot(t / 60, I_t2, color=INK_MUTED, linewidth=1.5)
    style_axes(ax_i2, "Current (A)")
    ax_i2.set_title("2-node testbed (winding + housing)", fontsize=10, color=INK_PRIMARY, loc="left")

    ax_t2.plot(t / 60, T_w_true, color=BLUE, linewidth=2, label="winding (true)")
    ax_t2.scatter(t / 60, T_w_meas, color=BLUE, alpha=0.18, s=8)
    ax_t2.plot(t / 60, T_h_true, color=AQUA, linewidth=2, label="housing (true)")
    ax_t2.scatter(t / 60, T_h_meas, color=AQUA, alpha=0.18, s=8)
    style_axes(ax_t2, "Temp (°C)")
    ax_t2.set_xlabel("Time (min)", color=INK_SECONDARY, fontsize=9)
    ax_t2.legend(
        frameon=False, fontsize=8, loc="lower right", labelcolor=INK_SECONDARY,
        title="dots = noisy sensor reading", title_fontsize=7,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "figures", "simulator_validation.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
