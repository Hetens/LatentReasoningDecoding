"""
Rebuttal figure: full causal map of the recursion.

Two stacked panels sharing the x-axis (18 recursion indices):
  Top:    probe F1 (MLP + linear) with bootstrap CI bands. Flat.
  Bottom: patching DeltaCE (cross-puzzle swap + within-puzzle shuffle)
          with bootstrap CI bands, zero line, outer-cycle boundaries.

A companion figure shows DeltaAcc (per-cell accuracy change), answering
whether patching effects survive on task metrics rather than only CE.

Usage:
    python -m experiments.probing.plot_patching_sweep \
        --sweep-json results/probing/patching_sweep/patching_sweep.json \
        --probe-dir results/probing/probe_results \
        --output-dir results/probing/patching_sweep
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, validated colorblind-safe for this figure.
C_MLP = "#009E73"      # probe MLP
C_LIN = "#CC79A7"      # probe linear
C_CROSS = "#D55E00"    # cross-puzzle swap
C_SHUF = "#0072B2"     # within-puzzle shuffle
C_GRID = "#d8d8d4"


def _step_labels(n_T: int, n_I: int):
    return [f"({T},{i})" for T in range(1, n_T + 1) for i in range(1, n_I + 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-json", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sweep = json.load(open(args.sweep_json))
    n_T, n_I = sweep["H_cycles"], sweep["L_cycles"]
    steps = sweep["steps"]
    x = np.arange(1, len(steps) + 1)

    def probe_series(name):
        path = os.path.join(args.probe_dir, f"probe_results_{name}_act16_z_L.json")
        rows = json.load(open(path))
        rows = sorted(rows, key=lambda r: (r["T"], r["i"]))
        return (
            np.array([r["f1"] for r in rows]),
            np.array([r["f1_ci_lo"] for r in rows]),
            np.array([r["f1_ci_hi"] for r in rows]),
        )

    f1_mlp, f1_mlp_lo, f1_mlp_hi = probe_series("mlp")
    f1_lin, f1_lin_lo, f1_lin_hi = probe_series("linear")

    def patch_series(intervention, metric):
        mean = np.array([s[intervention][f"delta_{metric}_mean"] for s in steps])
        lo = np.array([s[intervention][f"delta_{metric}_ci"][0] for s in steps])
        hi = np.array([s[intervention][f"delta_{metric}_ci"][1] for s in steps])
        return mean, lo, hi

    ce_cross, ce_cross_lo, ce_cross_hi = patch_series("cross_puzzle", "ce")
    ce_shuf, ce_shuf_lo, ce_shuf_hi = patch_series("within_shuffle", "ce")
    acc_cross, acc_cross_lo, acc_cross_hi = patch_series("cross_puzzle", "acc")
    acc_shuf, acc_shuf_lo, acc_shuf_hi = patch_series("within_shuffle", "acc")

    labels = _step_labels(n_T, n_I)

    def decorate(ax, ylab):
        for b in range(1, n_T):
            ax.axvline(b * n_I + 0.5, color="#999999", lw=0.8, ls=":")
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, axis="y", color=C_GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.5, len(labels) + 0.5)

    def cycle_bands(ax):
        ymin, ymax = ax.get_ylim()
        for T in range(n_T):
            ax.text(
                T * n_I + n_I / 2 + 0.5, ymax, f"T={T+1}",
                ha="center", va="top", fontsize=10, color="#555555",
            )

    # ------------------------- Main figure: F1 + DeltaCE -------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.12},
    )

    ax1.fill_between(x, f1_mlp_lo, f1_mlp_hi, color=C_MLP, alpha=0.18, lw=0)
    ax1.plot(x, f1_mlp, color=C_MLP, lw=2, marker="o", ms=4, label="MLP probe")
    ax1.fill_between(x, f1_lin_lo, f1_lin_hi, color=C_LIN, alpha=0.18, lw=0)
    ax1.plot(x, f1_lin, color=C_LIN, lw=2, ls="--", marker="s", ms=4, label="Linear probe")
    decorate(ax1, "Probe F1")
    ax1.set_ylim(0.70, 0.88)
    cycle_bands(ax1)
    ax1.legend(loc="lower right", frameon=False, fontsize=10)

    ax2.axhline(0.0, color="#555555", lw=1)
    ax2.fill_between(x, ce_cross_lo, ce_cross_hi, color=C_CROSS, alpha=0.18, lw=0)
    ax2.plot(x, ce_cross, color=C_CROSS, lw=2, marker="o", ms=4, label="Cross-puzzle swap")
    ax2.fill_between(x, ce_shuf_lo, ce_shuf_hi, color=C_SHUF, alpha=0.18, lw=0)
    ax2.plot(x, ce_shuf, color=C_SHUF, lw=2, marker="s", ms=4, label="Within-puzzle shuffle")
    decorate(ax2, r"$\Delta$ cross-entropy")
    cycle_bands(ax2)
    ax2.legend(loc="upper left", frameon=False, fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=60, fontsize=8)
    ax2.set_xlabel("Recursion index (T, i)", fontsize=11)

    fig.align_ylabels([ax1, ax2])
    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"patching_sweep_vs_probe.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close(fig)

    # --------------------- Companion figure: DeltaAcc ----------------------
    fig2, ax3 = plt.subplots(figsize=(9, 3.6))
    ax3.axhline(0.0, color="#555555", lw=1)
    ax3.fill_between(x, acc_cross_lo, acc_cross_hi, color=C_CROSS, alpha=0.18, lw=0)
    ax3.plot(x, acc_cross, color=C_CROSS, lw=2, marker="o", ms=4, label="Cross-puzzle swap")
    ax3.fill_between(x, acc_shuf_lo, acc_shuf_hi, color=C_SHUF, alpha=0.18, lw=0)
    ax3.plot(x, acc_shuf, color=C_SHUF, lw=2, marker="s", ms=4, label="Within-puzzle shuffle")
    decorate(ax3, r"$\Delta$ per-cell accuracy")
    cycle_bands(ax3)
    ax3.legend(loc="lower left", frameon=False, fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=60, fontsize=8)
    ax3.set_xlabel("Recursion index (T, i)", fontsize=11)
    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"patching_sweep_delta_acc.{ext}")
        fig2.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
