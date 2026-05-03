"""
Visualize latent-state geometry across recursion steps using
dimensionality reduction (PCA, UMAP) and corner (pair) plots.

Produces:
  - PCA 2D scatter coloured by (T, i)
  - PCA 3D scatter coloured by (T, i)
  - PCA corner plot (pair-plot of top-k components)
  - UMAP 2D scatter coloured by (T, i)
  - Explained-variance bar chart for PCA

Usage (from repo root):
    python -m experiments.probing.visualize_latents \
        --activations-dir results/probing/activations \
        --labels-dir      results/probing/labels \
        --output-dir      results/probing/plots \
        --latent z_L --act-step last \
        --max-puzzles 500 --cells-per-puzzle 10
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from sklearn.decomposition import PCA

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _resolve_act_step_tag(act_step_arg: str, activations_dir: str) -> str:
    if act_step_arg == "last":
        files = glob.glob(os.path.join(activations_dir, "z_L_act*.pt"))
        steps = sorted(int(f.split("act")[-1].split(".")[0]) for f in files)
        return f"act{steps[-1]}"
    if act_step_arg == "first":
        return "act1"
    return f"act{act_step_arg}"


def _subsample_cells(
    z: np.ndarray,
    max_puzzles: int,
    cells_per_puzzle: int,
    seed: int,
) -> np.ndarray:
    """Subsample puzzles and cells to keep scatter plots readable.

    Args:
        z: (N, H, L, 81, D) or (N, H, 81, D) array.
        max_puzzles: Cap on puzzles.
        cells_per_puzzle: Random cells to keep per puzzle.
        seed: RNG seed.

    Returns:
        z_sub with same rank but reduced N and cell dims.
    """
    rng = np.random.default_rng(seed)
    N = z.shape[0]
    n_puz = min(N, max_puzzles)
    puz_idx = rng.choice(N, n_puz, replace=False)

    is_z_L = z.ndim == 5
    total_cells = z.shape[-2]  # 81
    n_cells = min(total_cells, cells_per_puzzle)
    cell_idx = rng.choice(total_cells, n_cells, replace=False)

    if is_z_L:
        return z[np.ix_(puz_idx, range(z.shape[1]), range(z.shape[2]),
                        cell_idx, range(z.shape[4]))]
    else:
        return z[np.ix_(puz_idx, range(z.shape[1]),
                        cell_idx, range(z.shape[3]))]


def _flatten_for_dr(z: np.ndarray):
    """Flatten to (n_points, D) with (T, i) labels per point.

    Returns:
        X: (n_points, D) float32 array.
        ti_labels: (n_points,) array of (T, i) tuple strings.
        T_arr: (n_points,) int outer-cycle index.
        i_arr: (n_points,) int inner-step index.
    """
    is_z_L = z.ndim == 5
    if is_z_L:
        N, H, L, C, D = z.shape
    else:
        N, H, C, D = z.shape
        L = 1

    points, labels, T_arr, i_arr = [], [], [], []
    for t in range(H):
        for j in range(L):
            if is_z_L:
                chunk = z[:, t, j, :, :].reshape(-1, D)
            else:
                chunk = z[:, t, :, :].reshape(-1, D)
            points.append(chunk)
            n = chunk.shape[0]
            labels.extend([f"({t+1},{j+1})"] * n)
            T_arr.extend([t] * n)
            i_arr.extend([j] * n)

    X = np.concatenate(points, axis=0).astype(np.float32)
    return X, np.array(labels), np.array(T_arr), np.array(i_arr)


# ---------------------------------------------------------------------------
# Colour map for (T, i)
# ---------------------------------------------------------------------------

def _build_colors(H: int, L: int):
    """Return a colour for each (T, i) index and a legend."""
    cmap = plt.cm.tab20 if H * L <= 20 else plt.cm.gist_ncar
    K = H * L
    colors = {(t, i): cmap(k / max(K - 1, 1))
              for k, (t, i) in enumerate((t, i) for t in range(H) for i in range(L))}
    return colors


# ---------------------------------------------------------------------------
# PCA plots
# ---------------------------------------------------------------------------

def _fit_pca(X: np.ndarray, n_components: int = 10) -> PCA:
    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X)
    return pca


def plot_pca_2d(
    X: np.ndarray,
    T_arr: np.ndarray,
    i_arr: np.ndarray,
    pca: PCA,
    H: int,
    L: int,
    output_dir: str,
) -> None:
    Z = pca.transform(X)[:, :2]
    colors = _build_colors(H, L)

    fig, ax = plt.subplots(figsize=(8, 6))
    for t in range(H):
        for j in range(L):
            mask = (T_arr == t) & (i_arr == j)
            c = colors[(t, j)]
            ax.scatter(Z[mask, 0], Z[mask, 1], c=[c], s=3, alpha=0.35, rasterized=True)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[(t, j)],
                       markersize=5, label=f"({t+1},{j+1})")
               for t in range(H) for j in range(L)]
    ax.legend(handles=handles, fontsize=5, ncol=L, loc="upper right",
              title="$(T, i)$", title_fontsize=6)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title("PCA of TRM latent states across recursion steps")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_2d.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_2d.png")


def plot_pca_3d(
    X: np.ndarray,
    T_arr: np.ndarray,
    i_arr: np.ndarray,
    pca: PCA,
    H: int,
    L: int,
    output_dir: str,
) -> None:
    Z = pca.transform(X)[:, :3]
    colors = _build_colors(H, L)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for t in range(H):
        for j in range(L):
            mask = (T_arr == t) & (i_arr == j)
            c = colors[(t, j)]
            ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                       c=[c], s=2, alpha=0.3, rasterized=True)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=8)
    ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)", fontsize=8)
    ax.set_title("PCA 3D — TRM latent states")

    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[(t, j)],
                       markersize=4, label=f"({t+1},{j+1})")
               for t in range(H) for j in range(L)]
    ax.legend(handles=handles, fontsize=4, ncol=L, loc="upper left",
              title="$(T, i)$", title_fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_3d.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_3d.png")


# ---------------------------------------------------------------------------
# Corner plot (pair-plot of top PCA components)
# ---------------------------------------------------------------------------

def plot_pca_corner(
    X: np.ndarray,
    T_arr: np.ndarray,
    i_arr: np.ndarray,
    pca: PCA,
    H: int,
    L: int,
    output_dir: str,
    n_components: int = 5,
) -> None:
    """Pair-plot (corner plot) of the top PCA components."""
    nc = min(n_components, pca.n_components_)
    Z = pca.transform(X)[:, :nc]
    ev = pca.explained_variance_ratio_
    colors = _build_colors(H, L)

    fig, axes = plt.subplots(nc, nc, figsize=(2.5 * nc, 2.5 * nc))

    for row in range(nc):
        for col in range(nc):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue

            if row == col:
                for t in range(H):
                    for j in range(L):
                        mask = (T_arr == t) & (i_arr == j)
                        ax.hist(Z[mask, row], bins=40, color=colors[(t, j)],
                                alpha=0.4, density=True)
                ax.set_ylabel("Density" if col == 0 else "")
            else:
                for t in range(H):
                    for j in range(L):
                        mask = (T_arr == t) & (i_arr == j)
                        ax.scatter(Z[mask, col], Z[mask, row], c=[colors[(t, j)]],
                                   s=1, alpha=0.2, rasterized=True)

            if row == nc - 1:
                ax.set_xlabel(f"PC{col+1} ({ev[col]*100:.1f}%)", fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(f"PC{row+1} ({ev[row]*100:.1f}%)", fontsize=7)
            ax.tick_params(labelsize=5)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[(t, j)],
                       markersize=4, label=f"({t+1},{j+1})")
               for t in range(H) for j in range(L)]
    fig.legend(handles=handles, fontsize=5, ncol=L,
               loc="upper right", title="$(T, i)$", title_fontsize=6)
    fig.suptitle(f"Corner plot — top {nc} PCA components of TRM latent states",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_corner.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pca_corner.png ({nc} components)")


# ---------------------------------------------------------------------------
# Explained-variance bar chart
# ---------------------------------------------------------------------------

def plot_explained_variance(pca: PCA, output_dir: str) -> None:
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    nc = len(ev)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(1, nc + 1), ev * 100, color="#4c72b0", label="Individual")
    ax.plot(range(1, nc + 1), cum * 100, "o-", color="#c44e52", markersize=4,
            label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_explained_variance.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_explained_variance.png")


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def plot_umap_2d(
    X: np.ndarray,
    T_arr: np.ndarray,
    i_arr: np.ndarray,
    H: int,
    L: int,
    output_dir: str,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    seed: int = 42,
) -> None:
    try:
        import umap
    except ImportError:
        print("  Skipping UMAP (umap-learn not installed).")
        return

    print("  Fitting UMAP (this may take a few minutes) …")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed, metric="cosine")
    Z = reducer.fit_transform(X)
    colors = _build_colors(H, L)

    fig, ax = plt.subplots(figsize=(8, 6))
    for t in range(H):
        for j in range(L):
            mask = (T_arr == t) & (i_arr == j)
            c = colors[(t, j)]
            ax.scatter(Z[mask, 0], Z[mask, 1], c=[c], s=3, alpha=0.35,
                       rasterized=True)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[(t, j)],
                       markersize=5, label=f"({t+1},{j+1})")
               for t in range(H) for j in range(L)]
    ax.legend(handles=handles, fontsize=5, ncol=L, loc="upper right",
              title="$(T, i)$", title_fontsize=6)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of TRM latent states across recursion steps")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "umap_2d.png"), dpi=200)
    plt.close(fig)
    print("  Saved umap_2d.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize TRM latent geometry via PCA / UMAP / corner plots.",
    )
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--labels-dir", default=None,
                        help="Optional: path to labels dir (for backtracking colouring).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent", choices=["z_L", "z_H"], default="z_L")
    parser.add_argument("--act-step", default="last")
    parser.add_argument("--max-puzzles", type=int, default=500,
                        help="Subsample puzzles to keep plots readable.")
    parser.add_argument("--cells-per-puzzle", type=int, default=10,
                        help="Random cells per puzzle to keep.")
    parser.add_argument("--pca-components", type=int, default=10,
                        help="Number of PCA components to fit.")
    parser.add_argument("--corner-components", type=int, default=5,
                        help="Number of PCA components for the corner plot.")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP (saves time if umap-learn is unavailable).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tag = _resolve_act_step_tag(args.act_step, args.activations_dir)
    path = os.path.join(args.activations_dir, f"{args.latent}_{tag}.pt")
    print(f"Loading {path} …")
    z = torch.load(path, map_location="cpu", weights_only=True).numpy()
    print(f"  Raw shape: {z.shape}")

    is_z_L = args.latent == "z_L"
    H = z.shape[1]
    L = z.shape[2] if is_z_L else 1

    z_sub = _subsample_cells(z, args.max_puzzles, args.cells_per_puzzle, args.seed)
    print(f"  After subsampling: {z_sub.shape}")

    X, ti_labels, T_arr, i_arr = _flatten_for_dr(z_sub)
    print(f"  Flattened for DR: {X.shape}  ({H*L} groups)")

    pca = _fit_pca(X, n_components=args.pca_components)

    plot_explained_variance(pca, args.output_dir)
    plot_pca_2d(X, T_arr, i_arr, pca, H, L, args.output_dir)
    plot_pca_3d(X, T_arr, i_arr, pca, H, L, args.output_dir)
    plot_pca_corner(X, T_arr, i_arr, pca, H, L, args.output_dir,
                    n_components=args.corner_components)

    if not args.skip_umap:
        plot_umap_2d(X, T_arr, i_arr, H, L, args.output_dir, seed=args.seed)

    print("All visualizations generated.")


if __name__ == "__main__":
    main()
