"""
Generate all visualisations described in the report (Section 4.3):

  1. Line plots: mean val F1 vs inner step i, per outer step T, linear vs MLP.
  2. Heatmaps: T × i grid coloured by mean F1.
  3. Accuracy split by backtracking (easy vs hard).
  4. CKA heatmap across (T, i) pairs.

Usage (from repo root):
    python -m experiments.probing.plot_results \
        --probe-dir  results/probing/probe_results \
        --cka-dir    results/probing/cka \
        --output-dir results/probing/plots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_probe_results(probe_dir: str) -> Dict[str, List[dict]]:
    """Load all probe_results_*.json files, keyed by probe type."""
    out: Dict[str, List[dict]] = {}
    for path in sorted(glob.glob(os.path.join(probe_dir, "probe_results_*.json"))):
        fname = os.path.basename(path)
        # e.g. probe_results_linear_act16_z_L.json
        parts = fname.replace("probe_results_", "").replace(".json", "").split("_")
        probe_type = parts[0]  # linear / mlp
        with open(path) as f:
            out[probe_type] = json.load(f)
    return out


def _build_grid(results: List[dict], key: str) -> np.ndarray:
    """Build a (H_cycles, L_cycles) grid from a list of per-(T,i) dicts."""
    H = max(r["T"] for r in results) + 1
    L = max(r["i"] for r in results) + 1
    grid = np.full((H, L), np.nan)
    for r in results:
        grid[r["T"], r["i"]] = r[key]
    return grid


# ---------------------------------------------------------------------------
# Plot 1 — F1 vs inner step i (line plots, one line per outer step T)
# ---------------------------------------------------------------------------

def plot_f1_vs_inner_step(
    all_results: Dict[str, List[dict]],
    output_dir: str,
) -> None:
    """Line plot: F1 vs inner step i for each T, separate curves per probe."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors_T = plt.cm.viridis(np.linspace(0.2, 0.85, 4))
    linestyles = {"linear": "-", "mlp": "--"}
    markers = {"linear": "o", "mlp": "s"}

    for probe_type, results in all_results.items():
        H = max(r["T"] for r in results) + 1
        L = max(r["i"] for r in results) + 1
        for T in range(H):
            rows = sorted([r for r in results if r["T"] == T], key=lambda r: r["i"])
            i_vals = [r["i"] + 1 for r in rows]
            f1_vals = [r["f1"] for r in rows]
            ci_lo = [r.get("f1_ci_lo", r["f1"]) for r in rows]
            ci_hi = [r.get("f1_ci_hi", r["f1"]) for r in rows]

            label = f"{probe_type} T={T+1}"
            ax.plot(i_vals, f1_vals, linestyle=linestyles[probe_type],
                    marker=markers[probe_type], color=colors_T[T % len(colors_T)],
                    label=label, markersize=5, linewidth=1.5)
            ax.fill_between(i_vals, ci_lo, ci_hi, alpha=0.12, color=colors_T[T % len(colors_T)])

    ax.set_xlabel("Inner step $i$")
    ax.set_ylabel("Micro-averaged $F_1$")
    ax.set_title("Candidate-set probe $F_1$ across recursion steps")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_vs_inner_step.pdf"), dpi=200)
    fig.savefig(os.path.join(output_dir, "f1_vs_inner_step.png"), dpi=200)
    plt.close(fig)
    print("  Saved f1_vs_inner_step.{pdf,png}")


# ---------------------------------------------------------------------------
# Plot 2 — F1 heatmap (T × i)
# ---------------------------------------------------------------------------

def plot_f1_heatmap(
    all_results: Dict[str, List[dict]],
    output_dir: str,
) -> None:
    """Heatmap: T × i grid coloured by F1, one panel per probe type."""
    n_probes = len(all_results)
    fig, axes = plt.subplots(1, n_probes, figsize=(5 * n_probes, 4), squeeze=False)

    for idx, (probe_type, results) in enumerate(all_results.items()):
        ax = axes[0, idx]
        grid = _build_grid(results, "f1")
        H, L = grid.shape

        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower",
                        vmin=0, vmax=max(1.0, np.nanmax(grid)))
        ax.set_xticks(range(L))
        ax.set_xticklabels([str(j + 1) for j in range(L)])
        ax.set_yticks(range(H))
        ax.set_yticklabels([str(t + 1) for t in range(H)])
        ax.set_xlabel("Inner step $i$")
        ax.set_ylabel("Outer cycle $T$")
        ax.set_title(f"{probe_type.capitalize()} probe $F_1$")

        for t in range(H):
            for j in range(L):
                if not np.isnan(grid[t, j]):
                    ax.text(j, t, f"{grid[t, j]:.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_heatmap.pdf"), dpi=200)
    fig.savefig(os.path.join(output_dir, "f1_heatmap.png"), dpi=200)
    plt.close(fig)
    print("  Saved f1_heatmap.{pdf,png}")


# ---------------------------------------------------------------------------
# Plot 3 — F1 split by backtracking difficulty
# ---------------------------------------------------------------------------

def plot_f1_by_backtracking(
    all_results: Dict[str, List[dict]],
    output_dir: str,
) -> None:
    """Line plots split by easy (b=0) and hard (b=1) puzzles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors_T = plt.cm.viridis(np.linspace(0.2, 0.85, 4))

    for ax, difficulty, key in zip(axes, ["Easy ($b=0$)", "Hard ($b=1$)"], ["f1_easy", "f1_hard"]):
        for probe_type, results in all_results.items():
            H = max(r["T"] for r in results) + 1
            ls = "-" if probe_type == "linear" else "--"
            mk = "o" if probe_type == "linear" else "s"
            for T in range(H):
                rows = sorted([r for r in results if r["T"] == T], key=lambda r: r["i"])
                i_vals = [r["i"] + 1 for r in rows]
                f1_vals = [r.get(key, np.nan) for r in rows]
                if all(np.isnan(v) for v in f1_vals):
                    continue
                ax.plot(i_vals, f1_vals, linestyle=ls, marker=mk,
                        color=colors_T[T % len(colors_T)],
                        label=f"{probe_type} T={T+1}", markersize=5, linewidth=1.5)
        ax.set_xlabel("Inner step $i$")
        ax.set_ylabel("Micro-averaged $F_1$")
        ax.set_title(f"Probe $F_1$ — {difficulty}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_by_backtracking.pdf"), dpi=200)
    fig.savefig(os.path.join(output_dir, "f1_by_backtracking.png"), dpi=200)
    plt.close(fig)
    print("  Saved f1_by_backtracking.{pdf,png}")


# ---------------------------------------------------------------------------
# Plot 4 — Exact-match heatmap
# ---------------------------------------------------------------------------

def plot_exact_match_heatmap(
    all_results: Dict[str, List[dict]],
    output_dir: str,
) -> None:
    n_probes = len(all_results)
    fig, axes = plt.subplots(1, n_probes, figsize=(5 * n_probes, 4), squeeze=False)

    for idx, (probe_type, results) in enumerate(all_results.items()):
        ax = axes[0, idx]
        grid = _build_grid(results, "exact_match")
        H, L = grid.shape

        im = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower", vmin=0, vmax=1)
        ax.set_xticks(range(L))
        ax.set_xticklabels([str(j + 1) for j in range(L)])
        ax.set_yticks(range(H))
        ax.set_yticklabels([str(t + 1) for t in range(H)])
        ax.set_xlabel("Inner step $i$")
        ax.set_ylabel("Outer cycle $T$")
        ax.set_title(f"{probe_type.capitalize()} exact-match rate")

        for t in range(H):
            for j in range(L):
                if not np.isnan(grid[t, j]):
                    ax.text(j, t, f"{grid[t, j]:.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "exact_match_heatmap.pdf"), dpi=200)
    fig.savefig(os.path.join(output_dir, "exact_match_heatmap.png"), dpi=200)
    plt.close(fig)
    print("  Saved exact_match_heatmap.{pdf,png}")


# ---------------------------------------------------------------------------
# Plot 5 — CKA heatmap
# ---------------------------------------------------------------------------

def plot_cka_heatmap(cka_dir: str, output_dir: str) -> None:
    cka_file = os.path.join(cka_dir, "self_cka_z_L.npy")
    if not os.path.exists(cka_file):
        print("  Skipping CKA heatmap (self_cka_z_L.npy not found).")
        return

    grid = np.load(cka_file)
    K = grid.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="inferno", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    # Try to label as (T,i)
    labels = []
    # Guess dimensions: K = H * L
    for H in range(1, K + 1):
        if K % H == 0:
            L = K // H
            if 1 < H <= 6 and 1 < L <= 10:
                labels = [f"({t+1},{i+1})" for t in range(H) for i in range(L)]
                break
    if not labels:
        labels = [str(k) for k in range(K)]

    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Recursion index $(T, i)$")
    ax.set_ylabel("Recursion index $(T, i)$")
    ax.set_title("Self-CKA across recursion steps")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cka_heatmap.pdf"), dpi=200)
    fig.savefig(os.path.join(output_dir, "cka_heatmap.png"), dpi=200)
    plt.close(fig)
    print("  Saved cka_heatmap.{pdf,png}")


# ---------------------------------------------------------------------------
# Plot 6 — Null comparison bar chart
# ---------------------------------------------------------------------------

def plot_null_comparison(
    all_results: Dict[str, List[dict]],
    output_dir: str,
) -> None:
    """Grouped bar chart: probe F1 vs permutation null F1 per (T,i)."""
    for probe_type, results in all_results.items():
        if not any("null_f1" in r for r in results):
            continue
        rows = sorted(results, key=lambda r: (r["T"], r["i"]))
        labels = [f"({r['T']+1},{r['i']+1})" for r in rows]
        f1_real = [r["f1"] for r in rows]
        f1_null = [r.get("null_f1", 0) for r in rows]

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7), 4))
        ax.bar(x - w / 2, f1_real, w, label="Probe", color="#4c72b0")
        ax.bar(x + w / 2, f1_null, w, label="Null (permuted)", color="#c44e52", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Micro-averaged $F_1$")
        ax.set_title(f"{probe_type.capitalize()} probe vs permutation null")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"null_comparison_{probe_type}.pdf"), dpi=200)
        fig.savefig(os.path.join(output_dir, f"null_comparison_{probe_type}.png"), dpi=200)
        plt.close(fig)
        print(f"  Saved null_comparison_{probe_type}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate probing experiment plots.")
    parser.add_argument("--probe-dir", required=True, help="Directory with probe_results_*.json")
    parser.add_argument("--cka-dir", default=None, help="Directory with self_cka_z_L.npy (optional).")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = _load_probe_results(args.probe_dir)
    if not all_results:
        print("No probe_results_*.json files found.")
        return

    print(f"Loaded probe results for: {list(all_results.keys())}")

    plot_f1_vs_inner_step(all_results, args.output_dir)
    plot_f1_heatmap(all_results, args.output_dir)
    plot_exact_match_heatmap(all_results, args.output_dir)
    plot_f1_by_backtracking(all_results, args.output_dir)
    plot_null_comparison(all_results, args.output_dir)

    if args.cka_dir:
        plot_cka_heatmap(args.cka_dir, args.output_dir)

    print("All plots generated.")


if __name__ == "__main__":
    main()
