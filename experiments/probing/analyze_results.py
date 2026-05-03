"""
Post-hoc analysis of probing results:

  1. |S_c| distribution per difficulty group — explains why hard puzzles
     have higher probe F1 than easy puzzles.
  2. Activation-patching summary — loads patching JSON results and
     produces interpretive plots and a text report.

Usage (from repo root):
    python -m experiments.probing.analyze_results \
        --labels-dir      results/probing/labels \
        --patching-dir    results/probing/patching \
        --output-dir      results/probing/plots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===================================================================
# Part 1: |S_c| analysis per difficulty group
# ===================================================================

def analyze_candidate_sizes(labels_dir: str, output_dir: str) -> None:
    """Compute and plot |S_c| (candidate-set size) statistics split by
    backtracking difficulty.
    """
    y = np.load(os.path.join(labels_dir, "candidate_labels.npy"))   # (N, 81, 9)
    bt = np.load(os.path.join(labels_dir, "backtrack_flags.npy"))   # (N,)

    sc = y.sum(axis=-1)  # (N, 81)  — |S_c| per cell

    easy_mask = ~bt
    hard_mask = bt

    sc_easy = sc[easy_mask].ravel()
    sc_hard = sc[hard_mask].ravel()

    print("=" * 60)
    print("|S_c| analysis per difficulty group")
    print("=" * 60)
    print(f"  Easy puzzles: {int(easy_mask.sum())}")
    print(f"  Hard puzzles: {int(hard_mask.sum())}")
    print()
    print(f"  Easy  — mean |S_c|: {sc_easy.mean():.3f}  "
          f"median: {np.median(sc_easy):.1f}  "
          f"std: {sc_easy.std():.3f}")
    print(f"  Hard  — mean |S_c|: {sc_hard.mean():.3f}  "
          f"median: {np.median(sc_hard):.1f}  "
          f"std: {sc_hard.std():.3f}")
    print()

    # Per-size distribution
    sizes = np.arange(0, 10)
    easy_hist = np.array([(sc_easy == s).sum() for s in sizes]) / len(sc_easy)
    hard_hist = np.array([(sc_hard == s).sum() for s in sizes]) / len(sc_hard)

    print("  |S_c|    Easy (%)     Hard (%)")
    print("  " + "-" * 36)
    for s in sizes:
        print(f"    {s}      {easy_hist[s]*100:6.2f}      {hard_hist[s]*100:6.2f}")

    # Fraction of "given" cells (|S_c| == 1, i.e. already filled)
    easy_given = (sc_easy == 1).mean()
    hard_given = (sc_hard == 1).mean()
    print(f"\n  Fraction of given cells (|S_c|=1):  "
          f"Easy={easy_given*100:.1f}%  Hard={hard_given*100:.1f}%")

    # Fraction of blank cells (|S_c| > 1)
    easy_blank_mean = sc_easy[sc_easy > 1].mean() if (sc_easy > 1).any() else 0
    hard_blank_mean = sc_hard[sc_hard > 1].mean() if (sc_hard > 1).any() else 0
    print(f"  Mean |S_c| (blank cells only):      "
          f"Easy={easy_blank_mean:.3f}  Hard={hard_blank_mean:.3f}")

    # --- Plot: side-by-side histograms ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    width = 0.35
    x = sizes
    ax1.bar(x - width / 2, easy_hist * 100, width, label="Easy ($b=0$)",
            color="#4c72b0", alpha=0.85)
    ax1.bar(x + width / 2, hard_hist * 100, width, label="Hard ($b=1$)",
            color="#c44e52", alpha=0.85)
    ax1.set_xlabel("Candidate-set size $|S_c|$")
    ax1.set_ylabel("Fraction of cells (%)")
    ax1.set_title("$|S_c|$ distribution by difficulty")
    ax1.set_xticks(sizes)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- Plot: box plot ---
    data_box = [sc_easy[sc_easy > 1], sc_hard[sc_hard > 1]]
    bp = ax2.boxplot(data_box, tick_labels=["Easy", "Hard"], patch_artist=True,
                     widths=0.5)
    bp["boxes"][0].set_facecolor("#4c72b0")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#c44e52")
    bp["boxes"][1].set_alpha(0.6)
    ax2.set_ylabel("$|S_c|$ (blank cells only)")
    ax2.set_title("$|S_c|$ by difficulty (blank cells)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "candidate_size_analysis.png"), dpi=200)
    plt.close(fig)
    print(f"\n  Saved candidate_size_analysis.png")

    # Save statistics as JSON
    stats = {
        "n_easy": int(easy_mask.sum()),
        "n_hard": int(hard_mask.sum()),
        "easy_mean_sc": float(sc_easy.mean()),
        "hard_mean_sc": float(sc_hard.mean()),
        "easy_median_sc": float(np.median(sc_easy)),
        "hard_median_sc": float(np.median(sc_hard)),
        "easy_blank_mean_sc": float(easy_blank_mean),
        "hard_blank_mean_sc": float(hard_blank_mean),
        "easy_given_frac": float(easy_given),
        "hard_given_frac": float(hard_given),
    }
    stats_path = os.path.join(output_dir, "candidate_size_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved candidate_size_stats.json")


# ===================================================================
# Part 2: Activation-patching interpretation
# ===================================================================

def interpret_patching(patching_dir: str, output_dir: str) -> None:
    """Load patching JSON files, print a summary, and produce a plot."""
    files = sorted(glob.glob(os.path.join(patching_dir, "patching_T*_i*.json")))
    if not files:
        print("\n  No patching results found — skipping patching analysis.")
        return

    results: List[dict] = []
    for fp in files:
        with open(fp) as f:
            results.append(json.load(f))

    print("\n" + "=" * 60)
    print("Activation patching summary")
    print("=" * 60)
    print(f"{'(T,i)':<10} {'Cross Δ CE':>12} {'95% CI':>20} "
          f"{'Shuffle Δ CE':>14} {'95% CI':>20}")
    print("-" * 80)

    labels_list = []
    cross_means, cross_lo, cross_hi = [], [], []
    shuf_means, shuf_lo, shuf_hi = [], [], []

    for r in results:
        t, i = r["target_T"], r["target_i"]
        cp = r["cross_puzzle"]
        ws = r["within_shuffle"]
        labels_list.append(f"({t},{i})")
        cross_means.append(cp["mean"])
        cross_lo.append(cp["ci_lo"])
        cross_hi.append(cp["ci_hi"])
        shuf_means.append(ws["mean"])
        shuf_lo.append(ws["ci_lo"])
        shuf_hi.append(ws["ci_hi"])

        print(f"  ({t},{i})     {cp['mean']:>+10.4f}   "
              f"[{cp['ci_lo']:+.4f}, {cp['ci_hi']:+.4f}]   "
              f"{ws['mean']:>+10.4f}   "
              f"[{ws['ci_lo']:+.4f}, {ws['ci_hi']:+.4f}]")

    print()
    print("Interpretation:")
    for r in results:
        t, i = r["target_T"], r["target_i"]
        cp, ws = r["cross_puzzle"], r["within_shuffle"]
        if cp["ci_lo"] > 0:
            print(f"  ({t},{i}) Cross-puzzle:  Δ CE significantly positive — "
                  "swapping z from another puzzle harms performance.")
            print(f"           → The model causally relies on puzzle-specific "
                  "information encoded in z at this step.")
        else:
            print(f"  ({t},{i}) Cross-puzzle:  CI overlaps zero — "
                  "no clear causal dependence on puzzle-specific z.")

        if ws["ci_lo"] > 0:
            print(f"  ({t},{i}) Shuffle:       Δ CE significantly positive — "
                  "shuffling cell positions harms performance.")
            print(f"           → Positional structure within z matters; "
                  "the model uses which-cell-is-where.")
        else:
            print(f"  ({t},{i}) Shuffle:       CI overlaps zero — "
                  "positional structure may not be critical.")
        print()

    # --- Plot ---
    if len(results) == 1:
        _plot_patching_single(results[0], output_dir)
    else:
        _plot_patching_multi(labels_list, cross_means, cross_lo, cross_hi,
                             shuf_means, shuf_lo, shuf_hi, output_dir)


def _plot_patching_single(result: dict, output_dir: str) -> None:
    t, i = result["target_T"], result["target_i"]
    cp, ws = result["cross_puzzle"], result["within_shuffle"]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = [0, 1]
    means = [cp["mean"], ws["mean"]]
    errs_lo = [cp["mean"] - cp["ci_lo"], ws["mean"] - ws["ci_lo"]]
    errs_hi = [cp["ci_hi"] - cp["mean"], ws["ci_hi"] - ws["mean"]]
    colors = ["#4c72b0", "#c44e52"]

    ax.bar(x, means, yerr=[errs_lo, errs_hi], capsize=6,
           color=colors, alpha=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cross-puzzle\nswap", "Within-puzzle\nshuffle"])
    ax.set_ylabel("$\\Delta$ Cross-Entropy")
    ax.set_title(f"Activation patching at $(T={t}, i={i})$")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"patching_T{t}_i{i}.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved patching_T{t}_i{i}.png")


def _plot_patching_multi(
    labels: List[str],
    cross_means: List[float], cross_lo: List[float], cross_hi: List[float],
    shuf_means: List[float], shuf_lo: List[float], shuf_hi: List[float],
    output_dir: str,
) -> None:
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 4.5))

    cross_err = [np.array(cross_means) - np.array(cross_lo),
                 np.array(cross_hi) - np.array(cross_means)]
    shuf_err = [np.array(shuf_means) - np.array(shuf_lo),
                np.array(shuf_hi) - np.array(shuf_means)]

    ax.bar(x - w / 2, cross_means, w, yerr=cross_err, capsize=4,
           label="Cross-puzzle swap", color="#4c72b0", alpha=0.85)
    ax.bar(x + w / 2, shuf_means, w, yerr=shuf_err, capsize=4,
           label="Within-puzzle shuffle", color="#c44e52", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Recursion index $(T, i)$")
    ax.set_ylabel("$\\Delta$ Cross-Entropy")
    ax.set_title("Activation patching — causal effect on output")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "patching_comparison.png"), dpi=200)
    plt.close(fig)
    print("  Saved patching_comparison.png")


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis: |S_c| per difficulty + patching interpretation.",
    )
    parser.add_argument("--labels-dir", required=True,
                        help="Directory with candidate_labels.npy and backtrack_flags.npy.")
    parser.add_argument("--patching-dir", default=None,
                        help="Directory with patching_T*_i*.json files (optional).")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    analyze_candidate_sizes(args.labels_dir, args.output_dir)

    if args.patching_dir:
        interpret_patching(args.patching_dir, args.output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
