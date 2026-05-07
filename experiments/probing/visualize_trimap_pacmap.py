"""
TriMAP / PaCMAP visualization of TRM latent states.

PCA to 2D captures ~20% variance and loses global structure. UMAP preserves
local structure but can distort global distances. TriMAP and PaCMAP are
designed to preserve both local AND global structure, making them better
tools for understanding the full latent geometry.

Produces:
  - PaCMAP by outer cycle T (3-colour)
  - PaCMAP by |S_c|
  - PaCMAP by inner step i (faceted per T)
  - TriMAP by outer cycle T (if trimap is installed)
  - TriMAP by |S_c|

Usage (from repo root):
    python -m experiments.probing.visualize_trimap_pacmap \
        --activations-dir results/probing/activations \
        --labels-dir      results/probing/labels \
        --output-dir      results/probing/plots \
        --max-puzzles 200 --cells-per-puzzle 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pacmap
    HAS_PACMAP = True
except ImportError:
    HAS_PACMAP = False

try:
    import trimap
    HAS_TRIMAP = True
except ImportError:
    HAS_TRIMAP = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_data(args):
    """Load activations and labels, subsample, return flat arrays with metadata."""
    act_file = os.path.join(args.activations_dir, "z_L_act16.pt")
    z_L = torch.load(act_file, map_location="cpu", weights_only=True)
    z_L = z_L.float().numpy()
    N, H, L, S, D = z_L.shape

    labels = np.load(os.path.join(args.labels_dir, "candidate_labels.npy"))

    n_puz = min(args.max_puzzles, N)
    n_cells = min(args.cells_per_puzzle, S)

    rng = np.random.RandomState(42)
    puz_idx = rng.choice(N, n_puz, replace=False)
    cell_idx = rng.choice(S, n_cells, replace=False)

    sc = labels[puz_idx][:, cell_idx].sum(axis=-1)

    vecs_list = []
    meta_T = []
    meta_i = []
    meta_sc = []

    for T in range(H):
        for i in range(L):
            v = z_L[puz_idx][:, T, i][:, cell_idx]
            vecs_list.append(v.reshape(-1, D))
            n_pts = n_puz * n_cells
            meta_T.append(np.full(n_pts, T + 1))
            meta_i.append(np.full(n_pts, i + 1))
            meta_sc.append(sc.ravel())

    X = np.concatenate(vecs_list, axis=0)
    meta = {
        "T": np.concatenate(meta_T),
        "i": np.concatenate(meta_i),
        "sc": np.concatenate(meta_sc),
    }
    return X, meta


def plot_by_T(embedding, meta, method_name, output_dir):
    """3-colour scatter grouped by outer cycle T.

    For TriMAP the T=3 points sit on top of T=1 and T=2 and obscure them, so we
    draw T=1 and T=2 as hollow rings (open markers) and T=3 as filled discs;
    this keeps the late-cycle cluster solid while letting the earlier cycles
    show through. The legend is placed in an unused corner of the plot
    (lower-right; PaCMAP/TriMAP layouts leave the lower-right relatively empty).
    """
    colors = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71"}
    fig, ax = plt.subplots(figsize=(8, 7))

    for T in [1, 2, 3]:
        mask = meta["T"] == T
        if T == 3:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=colors[T], s=10, alpha=0.45,
                       label=f"T={T}", rasterized=True,
                       edgecolors="none")
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       facecolors="none", edgecolors=colors[T],
                       s=18, alpha=0.55, linewidths=0.6,
                       label=f"T={T}", rasterized=True)

    legend = ax.legend(markerscale=2.0, loc="lower right",
                       framealpha=0.95, fontsize=11, title="Outer cycle")
    legend.get_frame().set_edgecolor("#888888")

    ax.set_title(f"{method_name}: latent states by outer cycle T")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{method_name.lower()}_by_T.png"), dpi=150)
    plt.close(fig)


def plot_by_sc(embedding, meta, method_name, output_dir):
    """Coloured by candidate-set size |S_c|."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=meta["sc"], cmap="viridis", s=4, alpha=0.3, rasterized=True)
    fig.colorbar(sc, ax=ax, label="$|S_c|$")
    ax.set_title(f"{method_name}: latent states by candidate-set size $|S_c|$")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{method_name.lower()}_by_Sc.png"), dpi=150)
    plt.close(fig)


def plot_faceted_by_T(embedding, meta, method_name, output_dir):
    """Faceted panels: one per outer cycle, inner steps coloured.

    The colorbar is placed in a dedicated axis to the right of all three panels
    (instead of being attached to the third panel by default), so it does not
    visually belong to the rightmost subplot.
    """
    fig = plt.figure(figsize=(19, 5.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05],
                          left=0.05, right=0.94, top=0.88, bottom=0.10,
                          wspace=0.12)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    cmap = plt.cm.YlGn
    sc = None
    for idx, T in enumerate([1, 2, 3]):
        ax = axes[idx]
        mask_T = meta["T"] == T
        inner_vals = meta["i"][mask_T]
        sc = ax.scatter(embedding[mask_T, 0], embedding[mask_T, 1],
                        c=inner_vals, cmap=cmap, vmin=1, vmax=6,
                        s=4, alpha=0.3, rasterized=True)
        ax.set_title(f"Outer cycle T={T}")
        ax.set_xlabel(f"{method_name} 1")
        if idx == 0:
            ax.set_ylabel(f"{method_name} 2")
        else:
            ax.set_yticklabels([])
    fig.colorbar(sc, cax=cax, label="Inner step $i$")
    fig.suptitle(f"{method_name}: faceted by outer cycle, coloured by inner step", fontsize=13)
    fig.savefig(os.path.join(output_dir, f"{method_name.lower()}_faceted_by_T.png"), dpi=150)
    plt.close(fig)


def run_method(X, meta, method_name, reducer, output_dir):
    """Run a dimensionality reduction method and produce all plots."""
    print(f"\nComputing {method_name} embedding for {X.shape[0]} points in {X.shape[1]}D...")
    embedding = reducer.fit_transform(X)
    print(f"  Done. Embedding shape: {embedding.shape}")

    plot_by_T(embedding, meta, method_name, output_dir)
    plot_by_sc(embedding, meta, method_name, output_dir)
    plot_faceted_by_T(embedding, meta, method_name, output_dir)

    np.save(os.path.join(output_dir, f"{method_name.lower()}_embedding.npy"), embedding)
    print(f"  Saved {method_name} plots and embedding.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-puzzles", type=int, default=200)
    parser.add_argument("--cells-per-puzzle", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X, meta = load_data(args)
    print(f"Loaded {X.shape[0]} points in {X.shape[1]}D")

    if HAS_PACMAP:
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5,
                                FP_ratio=2.0, random_state=42)
        run_method(X, meta, "PaCMAP", reducer, args.output_dir)
    else:
        print("pacmap not installed. pip install pacmap")

    if HAS_TRIMAP:
        reducer = trimap.TRIMAP(n_dims=2, n_inliers=10, n_outliers=5,
                                n_random=3, verbose=True)
        run_method(X, meta, "TriMAP", reducer, args.output_dir)
    else:
        print("trimap not installed. pip install trimap")

    if not HAS_PACMAP and not HAS_TRIMAP:
        print("ERROR: Neither pacmap nor trimap is installed. Install at least one:")
        print("  pip install pacmap trimap")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
