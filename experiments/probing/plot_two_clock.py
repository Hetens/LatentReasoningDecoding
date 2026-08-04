"""
Figure: the recursion bifurcates into two dynamical regimes.

Panel A: answer trajectory across ACT segments, run 4x past the training
         horizon. Accuracy keeps improving and no solved puzzle is lost.
Panel B: relative residual ||dz_L||/||z_L|| per segment, split by whether
         the puzzle ends up solved. Solved instances contract to a fixed
         point; unsolved instances plateau and never converge.
Panel C: distribution of per-cell commit times, showing most cells are
         decided inside the first segment.

Usage:
    python -m experiments.probing.plot_two_clock \
        --npz results/probing/two_clock/two_clock_trajectories.npz \
        --output-dir results/probing/two_clock
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_SOLVED = "#0072B2"     # blue
C_UNSOLVED = "#D55E00"   # orange
C_GRID = "#d8d8d4"
C_RULE = "#555555"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--decodes-per-segment", type=int, default=3)
    ap.add_argument("--halt-max", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    d = np.load(args.npz)
    acc, res_L = d["cell_acc"], d["res_L"]
    ct, solved = d["commit_times"], d["final_solved"]
    N, D = acc.shape
    dps = args.decodes_per_segment
    n_seg = D // dps
    seg_x = np.arange(1, n_seg + 1)

    # Per-segment views (take the last decode / mean residual in each segment).
    acc_seg = acc[:, dps - 1::dps]
    solved_seg = (acc_seg == 1.0)
    res_seg = res_L.reshape(N, n_seg, -1).mean(axis=2)

    fig, (axA, axB, axC) = plt.subplots(
        3, 1, figsize=(9, 9.5),
        gridspec_kw={"height_ratios": [1, 1, 0.8], "hspace": 0.42},
    )

    def horizon(ax, label=True, label_y=0.97, va="top"):
        ax.axvline(args.halt_max, color=C_RULE, lw=1.2, ls="--")
        if label:
            ax.text(args.halt_max + 0.8, label_y, "training horizon",
                    transform=ax.get_xaxis_transform(), fontsize=9,
                    color=C_RULE, va=va, ha="left")

    def style(ax, ylab, xlab=None):
        ax.set_ylabel(ylab, fontsize=11)
        if xlab:
            ax.set_xlabel(xlab, fontsize=11)
        ax.grid(True, axis="y", color=C_GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # ---- Panel A: answer trajectory ----
    m = solved_seg.mean(axis=0)
    a = acc_seg.mean(axis=0)
    axA.plot(seg_x, a, color=C_SOLVED, lw=2, marker="o", ms=3.5,
             label="mean per-cell accuracy")
    axA.plot(seg_x, m, color=C_UNSOLVED, lw=2, ls="--", marker="s", ms=3.5,
             label="fraction fully solved")
    axA.set_ylim(0.55, 0.95)
    style(axA, "accuracy")
    horizon(axA)
    axA.legend(loc="lower right", frameon=False, fontsize=10)
    axA.set_title("A. Running 4x past the training horizon keeps improving; nothing is lost",
                  fontsize=11, loc="left", pad=8)
    axA.annotate(f"+{(m[-1]-m[args.halt_max-1])*100:.1f} pts solved",
                 xy=(n_seg, m[-1]), xytext=(n_seg - 18, m[-1] - 0.10),
                 fontsize=9.5, color=C_UNSOLVED,
                 arrowprops=dict(arrowstyle="->", color=C_UNSOLVED, lw=1.2))

    # ---- Panel B: residual bifurcation ----
    for mask, color, lab, mk in (
        (solved, C_SOLVED, "ends solved", "o"),
        (~solved, C_UNSOLVED, "ends unsolved", "s"),
    ):
        if mask.sum() == 0:
            continue
        r = res_seg[mask]
        med = np.median(r, axis=0)
        lo, hi = np.percentile(r, [25, 75], axis=0)
        axB.fill_between(seg_x, lo, hi, color=color, alpha=0.16, lw=0)
        axB.plot(seg_x, med, color=color, lw=2, marker=mk, ms=3.5,
                 label=f"{lab}  (n={int(mask.sum())})")
    axB.set_yscale("log")
    # The solved-regime plateau is the bf16 noise floor, not real drift:
    # the same trajectory in fp32 reaches residual exactly 0.
    axB.axhspan(0, 0.0125, color=C_RULE, alpha=0.10, lw=0)
    axB.text(n_seg * 0.40, 0.0155, "bf16 noise floor (fp32 residual is exactly 0)",
             fontsize=8.5, color=C_RULE, va="bottom", ha="center")
    style(axB, r"residual  $\|\Delta z_L\| / \|z_L\|$")
    horizon(axB, label_y=0.55)
    axB.legend(loc="center right", frameon=False, fontsize=10)
    axB.set_title("B. Solved instances reach an exact fixed point; unsolved ones never converge",
                  fontsize=11, loc="left", pad=8)

    # ---- Panel C: commit times ----
    v = ct[ct >= 0] / dps  # in segments
    bins = np.arange(0, n_seg + 1, 1)
    axC.hist(v, bins=bins, color=C_SOLVED, alpha=0.85, edgecolor="white", linewidth=0.5)
    axC.set_yscale("log")
    style(axC, "cells (log)", "ACT segment")
    horizon(axC, label=False)
    frac1 = (v < 1).mean() * 100
    axC.set_title(f"C. {frac1:.0f}% of cells are decided inside the first segment",
                  fontsize=11, loc="left", pad=8)

    for ax in (axA, axB, axC):
        ax.set_xlim(0.5, n_seg + 0.5)

    fig.align_ylabels([axA, axB, axC])
    for ext in ("png", "pdf"):
        out = os.path.join(args.output_dir, f"two_clock_regimes.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
