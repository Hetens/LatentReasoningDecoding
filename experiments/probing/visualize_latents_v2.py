"""
Improved latent-state visualizations (v2).

Plot families:
  A — Recursion-step views (3-colour T-grouped, faceted, density contours)
  B — Property-coloured views (|S_c|, cell position, correctness)
  C — Delta-z PCA (inner-step deltas, outer-cycle deltas)
  D — Per-cell decoding maps for individual puzzles

Usage (from repo root):
    python -m experiments.probing.visualize_latents_v2 \
        --activations-dir results/probing/activations \
        --labels-dir      results/probing/labels \
        --output-dir      results/probing/plots \
        --latent z_L --act-step last \
        --max-puzzles 200 --cells-per-puzzle 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===================================================================
# Data helpers
# ===================================================================

def _resolve_act_step_tag(act_step: str, act_dir: str) -> str:
    if act_step == "last":
        files = glob.glob(os.path.join(act_dir, "z_L_act*.pt"))
        steps = sorted(int(f.split("act")[-1].split(".")[0]) for f in files)
        return f"act{steps[-1]}"
    if act_step == "first":
        return "act1"
    return f"act{act_step}"


def _subsample(
    z: np.ndarray, max_puz: int, cells: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_sub, puz_idx, cell_idx) after subsampling."""
    rng = np.random.default_rng(seed)
    N = z.shape[0]
    puz_idx = rng.choice(N, min(N, max_puz), replace=False)
    puz_idx.sort()
    total_cells = z.shape[-2]
    cell_idx = rng.choice(total_cells, min(total_cells, cells), replace=False)
    cell_idx.sort()
    if z.ndim == 5:
        z_sub = z[np.ix_(puz_idx, range(z.shape[1]), range(z.shape[2]),
                         cell_idx, range(z.shape[4]))]
    else:
        z_sub = z[np.ix_(puz_idx, range(z.shape[1]),
                         cell_idx, range(z.shape[3]))]
    return z_sub, puz_idx, cell_idx


def _flatten(z: np.ndarray):
    """Flatten z_L (N,H,L,C,D) → X (n_pts, D), T_arr, i_arr, puz_arr, cell_arr."""
    N, H, L, C, D = z.shape
    X_parts, T_a, i_a, puz_a, cell_a = [], [], [], [], []
    for t in range(H):
        for j in range(L):
            chunk = z[:, t, j, :, :].reshape(-1, D)
            X_parts.append(chunk)
            n = chunk.shape[0]
            T_a.append(np.full(n, t, dtype=np.int32))
            i_a.append(np.full(n, j, dtype=np.int32))
            puz_a.append(np.repeat(np.arange(N), C))
            cell_a.append(np.tile(np.arange(C), N))
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    return (X, np.concatenate(T_a), np.concatenate(i_a),
            np.concatenate(puz_a), np.concatenate(cell_a))


T_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a"}
T_NAMES = {0: "$T{=}1$", 1: "$T{=}2$", 2: "$T{=}3$"}


# ===================================================================
# A — Recursion-step views
# ===================================================================

def plot_t_grouped_scatter(Z2: np.ndarray, T_arr: np.ndarray,
                           ev: np.ndarray, out: str) -> None:
    """PCA 2D scatter with 3 bold colours for T=1/2/3."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for t in sorted(T_COLORS):
        m = T_arr == t
        ax.scatter(Z2[m, 0], Z2[m, 1], c=T_COLORS[t], s=4, alpha=0.25,
                   label=T_NAMES[t], rasterized=True)
    ax.legend(fontsize=10, markerscale=3)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title("PCA — latent states grouped by outer cycle $T$")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca_by_T.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_by_T.png")


def plot_t_grouped_density(Z2: np.ndarray, T_arr: np.ndarray,
                           ev: np.ndarray, out: str) -> None:
    """PCA 2D density contour plot, one set of contours per T."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for t in sorted(T_COLORS):
        m = T_arr == t
        pts = Z2[m]
        if len(pts) < 10:
            continue
        try:
            kde = gaussian_kde(pts.T, bw_method=0.15)
        except np.linalg.LinAlgError:
            continue
        xmin, xmax = pts[:, 0].min() - 1, pts[:, 0].max() + 1
        ymin, ymax = pts[:, 1].min() - 1, pts[:, 1].max() + 1
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contour(xx, yy, density, levels=5, colors=T_COLORS[t],
                   linewidths=1.2, alpha=0.8)
        ax.contourf(xx, yy, density, levels=5, colors=[T_COLORS[t]] * 6,
                    alpha=0.08)
    handles = [Line2D([0], [0], color=T_COLORS[t], lw=2, label=T_NAMES[t])
               for t in sorted(T_COLORS)]
    ax.legend(handles=handles, fontsize=10)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title("PCA density contours by outer cycle $T$")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca_density_by_T.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_density_by_T.png")


def plot_faceted_by_T(Z2: np.ndarray, T_arr: np.ndarray, i_arr: np.ndarray,
                      ev: np.ndarray, H: int, L: int, out: str) -> None:
    """One PCA panel per outer cycle T, inner steps in different shades."""
    fig, axes = plt.subplots(1, H, figsize=(5 * H, 4.5), sharey=True)
    if H == 1:
        axes = [axes]
    cmap = plt.cm.viridis

    for t, ax in enumerate(axes):
        for j in range(L):
            m = (T_arr == t) & (i_arr == j)
            c = cmap(j / max(L - 1, 1))
            ax.scatter(Z2[m, 0], Z2[m, 1], c=[c], s=4, alpha=0.3,
                       label=f"$i={j+1}$", rasterized=True)
        ax.set_title(f"Outer cycle $T={t+1}$", fontsize=11)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        if t == 0:
            ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.legend(fontsize=7, markerscale=2.5, loc="best")
        ax.grid(alpha=0.2)

    fig.suptitle("PCA faceted by outer cycle — inner steps coloured",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca_faceted_by_T.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved pca_faceted_by_T.png")


# ===================================================================
# B — Property-coloured views
# ===================================================================

def _property_scatter(Z2: np.ndarray, vals: np.ndarray, ev: np.ndarray,
                      cmap_name: str, label: str, title: str, fname: str,
                      out: str, vmin=None, vmax=None, discrete: bool = False,
                      legend_labels=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if discrete and legend_labels is not None:
        unique = sorted(set(vals))
        cmap = plt.cm.get_cmap(cmap_name, len(unique))
        for k, u in enumerate(unique):
            m = vals == u
            ax.scatter(Z2[m, 0], Z2[m, 1], c=[cmap(k)], s=3, alpha=0.3,
                       label=legend_labels.get(u, str(u)), rasterized=True)
        ax.legend(fontsize=7, markerscale=3, loc="best", title=label,
                  title_fontsize=8)
    else:
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=vals, cmap=cmap_name, s=3,
                        alpha=0.3, vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_property_views(
    Z2: np.ndarray, ev: np.ndarray,
    T_arr: np.ndarray, puz_arr: np.ndarray, cell_arr: np.ndarray,
    y_labels: np.ndarray, model_labels: np.ndarray,
    real_cell_idx: np.ndarray, out: str,
) -> None:
    """Generate PCA plots coloured by |S_c|, cell position, correctness."""

    n_pts = Z2.shape[0]
    n_ti_groups = len(set(zip(T_arr.tolist(), [0]*len(T_arr))))

    # |S_c| — candidate-set size per cell
    sc_per_cell = y_labels.sum(axis=-1)  # (N_sub, C_sub)
    sc_flat = np.array([sc_per_cell[puz_arr[k], cell_arr[k]] for k in range(n_pts)])
    _property_scatter(Z2, sc_flat, ev, "viridis",
                      "$|S_c|$", "PCA coloured by candidate-set size $|S_c|$",
                      "pca_by_Sc.png", out, vmin=1, vmax=8)

    # Row, column, box
    rows = real_cell_idx // 9
    cols = real_cell_idx % 9
    boxes = (rows // 3) * 3 + (cols // 3)
    row_flat = np.array([rows[cell_arr[k]] for k in range(n_pts)])
    col_flat = np.array([cols[cell_arr[k]] for k in range(n_pts)])
    box_flat = np.array([boxes[cell_arr[k]] for k in range(n_pts)])

    _property_scatter(Z2, row_flat, ev, "tab10",
                      "Row", "PCA coloured by cell row",
                      "pca_by_row.png", out, vmin=0, vmax=8)
    _property_scatter(Z2, col_flat, ev, "tab10",
                      "Column", "PCA coloured by cell column",
                      "pca_by_col.png", out, vmin=0, vmax=8)
    _property_scatter(Z2, box_flat, ev, "Set1",
                      "Box", "PCA coloured by 3×3 box",
                      "pca_by_box.png", out, vmin=0, vmax=8)

    # Correctness: compare model prediction (argmax logit) to label
    if model_labels is not None:
        correct = (model_labels == y_labels.argmax(axis=-1))
        correct_per_cell = correct.astype(np.float32)
        correct_flat = np.array([correct_per_cell[puz_arr[k], cell_arr[k]]
                                 for k in range(n_pts)])
        _property_scatter(
            Z2, correct_flat, ev, "RdYlGn",
            "Correct", "PCA coloured by cell correctness",
            "pca_by_correct.png", out, discrete=True,
            legend_labels={0.0: "Wrong", 1.0: "Correct"})


# ===================================================================
# C — Delta-z PCA
# ===================================================================

def plot_delta_z(z_sub: np.ndarray, out: str, seed: int = 42) -> None:
    """PCA on delta-z: inner-step deltas and outer-cycle deltas."""
    N, H, L, C, D = z_sub.shape

    # Inner-step deltas: z[T, i+1] - z[T, i]
    inner_deltas = z_sub[:, :, 1:, :, :] - z_sub[:, :, :-1, :, :]  # (N,H,L-1,C,D)
    id_parts, id_T, id_i = [], [], []
    for t in range(H):
        for j in range(L - 1):
            chunk = inner_deltas[:, t, j, :, :].reshape(-1, D)
            id_parts.append(chunk)
            id_T.append(np.full(chunk.shape[0], t))
            id_i.append(np.full(chunk.shape[0], j))
    X_inner = np.concatenate(id_parts, 0).astype(np.float32)
    T_inner = np.concatenate(id_T)
    i_inner = np.concatenate(id_i)

    pca_inner = PCA(n_components=2).fit(X_inner)
    Z_inner = pca_inner.transform(X_inner)
    ev_inner = pca_inner.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: inner deltas coloured by T
    ax = axes[0]
    for t in sorted(T_COLORS):
        m = T_inner == t
        ax.scatter(Z_inner[m, 0], Z_inner[m, 1], c=T_COLORS[t], s=3,
                   alpha=0.25, label=T_NAMES[t], rasterized=True)
    ax.legend(fontsize=9, markerscale=3)
    ax.set_xlabel(f"PC1 ({ev_inner[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev_inner[1]*100:.1f}%)")
    ax.set_title("$\\Delta z_L$ per inner step ($z_{T,i+1} - z_{T,i}$)")
    ax.grid(alpha=0.2)

    # Outer-cycle deltas: z[T+1, 0] - z[T, L-1]
    if H > 1:
        outer_deltas = z_sub[:, 1:, 0, :, :] - z_sub[:, :-1, -1, :, :]  # (N,H-1,C,D)
        od_parts, od_T = [], []
        for t in range(H - 1):
            chunk = outer_deltas[:, t, :, :].reshape(-1, D)
            od_parts.append(chunk)
            od_T.append(np.full(chunk.shape[0], t))
        X_outer = np.concatenate(od_parts, 0).astype(np.float32)
        T_outer = np.concatenate(od_T)

        pca_outer = PCA(n_components=2).fit(X_outer)
        Z_outer = pca_outer.transform(X_outer)
        ev_outer = pca_outer.explained_variance_ratio_

        ax = axes[1]
        outer_colors = {0: "#e41a1c", 1: "#377eb8"}
        outer_names = {0: "$T{=}1{\\to}2$", 1: "$T{=}2{\\to}3$"}
        for t in sorted(outer_colors):
            if t >= H - 1:
                break
            m = T_outer == t
            ax.scatter(Z_outer[m, 0], Z_outer[m, 1], c=outer_colors[t], s=3,
                       alpha=0.25, label=outer_names[t], rasterized=True)
        ax.legend(fontsize=9, markerscale=3)
        ax.set_xlabel(f"PC1 ({ev_outer[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev_outer[1]*100:.1f}%)")
        ax.set_title("$\\Delta z_L$ per outer cycle ($z_{T+1,1} - z_{T,L}$)")
        ax.grid(alpha=0.2)
    else:
        axes[1].set_visible(False)

    fig.suptitle("PCA of latent-state changes ($\\Delta z$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca_delta_z.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_delta_z.png")


# ===================================================================
# D — Per-cell decoding maps
# ===================================================================

def plot_per_cell_maps(
    z_full: np.ndarray, y_labels: np.ndarray, inputs: np.ndarray,
    out: str, n_puzzles: int = 10, target_T: int = 2, seed: int = 42,
) -> None:
    """For a few puzzles, plot each cell in PCA space at outer cycle
    *target_T*, annotated with position and |S_c|."""
    N, H, L, C, D = z_full.shape
    rng = np.random.default_rng(seed)
    puz_idx = rng.choice(N, min(n_puzzles, N), replace=False)

    last_i = L - 1
    z_slice = z_full[puz_idx, target_T, last_i, :, :]  # (n_puz, 81, D)
    n_puz = z_slice.shape[0]

    X_all = z_slice.reshape(-1, D).astype(np.float32)
    pca = PCA(n_components=2).fit(X_all)
    Z_all = pca.transform(X_all).reshape(n_puz, C, 2)
    ev = pca.explained_variance_ratio_

    sc = y_labels[puz_idx].sum(axis=-1)  # (n_puz, 81)

    ncols = min(5, n_puz)
    nrows = (n_puz + ncols - 1) // ncols

    fig = plt.figure(figsize=(4.5 * ncols + 0.8, 4.2 * nrows))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.04],
                          wspace=0.35, hspace=0.4)

    scatter = None
    for idx in range(n_puz):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        sc_vals = sc[idx]
        scatter = ax.scatter(Z_all[idx, :, 0], Z_all[idx, :, 1],
                             c=sc_vals, cmap="viridis", s=30, alpha=0.8,
                             edgecolors="k", linewidths=0.3,
                             vmin=1, vmax=8)

        for c_idx in range(C):
            inp_val = inputs[puz_idx[idx], c_idx]
            if inp_val >= 2:
                digit = str(int(inp_val) - 1)
                ax.annotate(digit, (Z_all[idx, c_idx, 0], Z_all[idx, c_idx, 1]),
                            fontsize=5, ha="center", va="center",
                            fontweight="bold", color="white",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="black", alpha=0.5, lw=0))

        ax.set_title(f"Puzzle {puz_idx[idx]}", fontsize=9)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=7)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.15)

    if scatter is not None:
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(scatter, cax=cax, label="$|S_c|$")

    fig.suptitle(f"Per-cell PCA at $(T={target_T+1}, i={last_i+1})$ — "
                 f"digits annotated on given cells", fontsize=12)
    fig.savefig(os.path.join(out, "per_cell_maps.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved per_cell_maps.png")


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved latent visualizations (v2).",
    )
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent", choices=["z_L", "z_H"], default="z_L")
    parser.add_argument("--act-step", default="last")
    parser.add_argument("--max-puzzles", type=int, default=200)
    parser.add_argument("--cells-per-puzzle", type=int, default=5)
    parser.add_argument("--n-decode-puzzles", type=int, default=10,
                        help="Number of puzzles for per-cell maps.")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tag = _resolve_act_step_tag(args.act_step, args.activations_dir)
    path = os.path.join(args.activations_dir, f"{args.latent}_{tag}.pt")
    print(f"Loading {path} …")
    z = torch.load(path, map_location="cpu", weights_only=True).numpy()
    print(f"  Raw shape: {z.shape}")

    N, H, L, C_full, D = z.shape

    # Load labels
    y_labels = np.load(os.path.join(args.labels_dir, "candidate_labels.npy"))
    inputs = np.load(os.path.join(args.activations_dir, "inputs.npy"))

    # Subsample
    z_sub, puz_idx, cell_idx = _subsample(
        z, args.max_puzzles, args.cells_per_puzzle, args.seed)
    print(f"  Subsampled: {z_sub.shape}")

    y_sub = y_labels[puz_idx][:, cell_idx, :]
    inputs_sub = inputs[puz_idx][:, cell_idx] if inputs.ndim == 2 else None

    X, T_arr, i_arr, puz_arr, cell_arr = _flatten(z_sub)
    print(f"  Flattened: {X.shape}")

    pca = PCA(n_components=min(10, D)).fit(X)
    Z2 = pca.transform(X)[:, :2]
    ev = pca.explained_variance_ratio_

    # ---- A: Recursion-step views ----
    print("\n--- A: Recursion-step views ---")
    plot_t_grouped_scatter(Z2, T_arr, ev, args.output_dir)
    plot_t_grouped_density(Z2, T_arr, ev, args.output_dir)
    plot_faceted_by_T(Z2, T_arr, i_arr, ev, H, L, args.output_dir)

    # ---- B: Property-coloured views ----
    print("\n--- B: Property-coloured views ---")
    plot_property_views(
        Z2, ev, T_arr, puz_arr, cell_arr,
        y_sub, model_labels=None, real_cell_idx=cell_idx,
        out=args.output_dir)

    # ---- C: Delta-z PCA ----
    print("\n--- C: Delta-z PCA ---")
    plot_delta_z(z_sub, args.output_dir, seed=args.seed)

    # ---- D: Per-cell decoding maps ----
    print("\n--- D: Per-cell decoding maps ---")
    z_decode = z[:min(N, 500)]  # keep all 81 cells for decoding
    y_decode = y_labels[:min(N, 500)]
    inputs_decode = inputs[:min(N, 500)]
    plot_per_cell_maps(z_decode, y_decode, inputs_decode,
                       args.output_dir,
                       n_puzzles=args.n_decode_puzzles,
                       target_T=H - 1, seed=args.seed)

    print("\nAll v2 visualizations generated.")


if __name__ == "__main__":
    main()
