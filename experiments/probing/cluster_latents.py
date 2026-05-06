"""
Multidimensional clustering of TRM latent states.

Since PCA to 2D captures only ~20% of variance, we cluster in the full
512-D space (and optionally in top-k PCA space) to find natural groupings,
then analyze what those clusters correspond to (recursion step, |S_c|,
cell position, correctness).

Methods:
  - K-Means (k = 3..20, elbow + silhouette)
  - HDBSCAN (density-based, no k required)
  - Analysis: cluster purity w.r.t. outer cycle T, |S_c|, row/col/box

Usage (from repo root):
    python -m experiments.probing.cluster_latents \
        --activations-dir results/probing/activations \
        --labels-dir      results/probing/labels \
        --output-dir      results/probing/plots \
        --max-puzzles 200 --cells-per-puzzle 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_data(args):
    """Load activations and labels, subsample, return flat arrays."""
    act_file = os.path.join(args.activations_dir, "z_L_act16.pt")
    z_L = torch.load(act_file, map_location="cpu", weights_only=True)
    z_L = z_L.float().numpy()  # (N, H, L, 81, 512)
    N, H, L, S, D = z_L.shape

    labels = np.load(os.path.join(args.labels_dir, "candidate_labels.npy"))  # (N, 81, 9)

    n_puz = min(args.max_puzzles, N)
    n_cells = min(args.cells_per_puzzle, S)

    rng = np.random.RandomState(42)
    puz_idx = rng.choice(N, n_puz, replace=False)
    cell_idx = rng.choice(S, n_cells, replace=False)

    sc = labels[puz_idx][:, cell_idx].sum(axis=-1)  # (n_puz, n_cells)
    rows = cell_idx // 9
    cols = cell_idx % 9
    boxes = (rows // 3) * 3 + (cols // 3)

    vecs_list = []
    meta_T = []
    meta_i = []
    meta_sc = []
    meta_row = []
    meta_col = []
    meta_box = []

    for T in range(H):
        for i in range(L):
            v = z_L[puz_idx][:, T, i][:, cell_idx]  # (n_puz, n_cells, D)
            vecs_list.append(v.reshape(-1, D))
            n_pts = n_puz * n_cells
            meta_T.append(np.full(n_pts, T + 1))
            meta_i.append(np.full(n_pts, i + 1))
            meta_sc.append(sc.ravel())
            meta_row.append(np.tile(rows, n_puz))
            meta_col.append(np.tile(cols, n_puz))
            meta_box.append(np.tile(boxes, n_puz))

    X = np.concatenate(vecs_list, axis=0)
    meta = {
        "T": np.concatenate(meta_T),
        "i": np.concatenate(meta_i),
        "sc": np.concatenate(meta_sc),
        "row": np.concatenate(meta_row),
        "col": np.concatenate(meta_col),
        "box": np.concatenate(meta_box),
    }
    return X, meta


def cluster_purity(cluster_labels, property_labels):
    """Fraction of each cluster's points that share the majority label."""
    purities = []
    for c in np.unique(cluster_labels):
        if c == -1:
            continue
        mask = cluster_labels == c
        vals, counts = np.unique(property_labels[mask], return_counts=True)
        purities.append(counts.max() / counts.sum())
    return np.mean(purities) if purities else 0.0


def run_kmeans_sweep(X_scaled, meta, output_dir, k_range=range(3, 21)):
    """K-Means with elbow and silhouette analysis."""
    inertias = []
    silhouettes = []
    ks = list(k_range)

    print(f"Running K-Means sweep k={ks[0]}..{ks[-1]} on {X_scaled.shape[0]} points in {X_scaled.shape[1]}D")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(labels)))
        silhouettes.append(sil)
        print(f"  k={k:2d}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inertias, "o-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("K-Means elbow plot")
    ax2.plot(ks, silhouettes, "o-")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette vs. k")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cluster_kmeans_sweep.png"), dpi=150)
    plt.close(fig)

    best_k = ks[np.argmax(silhouettes)]
    print(f"\nBest k by silhouette: {best_k}")
    return best_k


def analyze_clusters(cluster_labels, meta, method_name, output_dir):
    """Compute and plot purity w.r.t. various properties."""
    n_clusters = len(set(cluster_labels) - {-1})
    noise_frac = (cluster_labels == -1).mean() if -1 in cluster_labels else 0.0

    results = {
        "method": method_name,
        "n_clusters": n_clusters,
        "noise_fraction": float(noise_frac),
    }

    props = {"outer_cycle_T": meta["T"], "sc": meta["sc"],
             "row": meta["row"], "box": meta["box"]}
    for name, vals in props.items():
        pur = cluster_purity(cluster_labels, vals)
        results[f"purity_{name}"] = float(pur)
        print(f"  {method_name} purity w.r.t. {name}: {pur:.3f}")

    # cluster composition heatmap for T
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # T composition
    unique_c = sorted(set(cluster_labels) - {-1})
    T_vals = sorted(np.unique(meta["T"]))
    comp_T = np.zeros((len(unique_c), len(T_vals)))
    for ci, c in enumerate(unique_c):
        mask = cluster_labels == c
        for ti, t in enumerate(T_vals):
            comp_T[ci, ti] = (meta["T"][mask] == t).mean()

    ax = axes[0]
    im = ax.imshow(comp_T, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Outer cycle T")
    ax.set_ylabel("Cluster")
    ax.set_xticks(range(len(T_vals)))
    ax.set_xticklabels([f"T={t}" for t in T_vals])
    ax.set_yticks(range(len(unique_c)))
    ax.set_yticklabels(unique_c)
    ax.set_title(f"{method_name}: cluster composition by T")
    fig.colorbar(im, ax=ax, label="Fraction")

    # |S_c| composition
    sc_bins = sorted(np.unique(meta["sc"]))
    if len(sc_bins) > 9:
        sc_bins = list(range(1, 10))
    comp_sc = np.zeros((len(unique_c), len(sc_bins)))
    for ci, c in enumerate(unique_c):
        mask = cluster_labels == c
        for si, s in enumerate(sc_bins):
            comp_sc[ci, si] = (meta["sc"][mask] == s).mean()

    ax = axes[1]
    im = ax.imshow(comp_sc, aspect="auto", cmap="YlGnBu")
    ax.set_xlabel("$|S_c|$")
    ax.set_ylabel("Cluster")
    ax.set_xticks(range(len(sc_bins)))
    ax.set_xticklabels(sc_bins)
    ax.set_yticks(range(len(unique_c)))
    ax.set_yticklabels(unique_c)
    ax.set_title(f"{method_name}: cluster composition by $|S_c|$")
    fig.colorbar(im, ax=ax, label="Fraction")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cluster_{method_name}_composition.png"), dpi=150)
    plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-puzzles", type=int, default=200)
    parser.add_argument("--cells-per-puzzle", type=int, default=5)
    parser.add_argument("--pca-dims", type=int, default=50,
                        help="Number of PCA dims to reduce to before clustering (0 = full 512D)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    X, meta = load_data(args)
    print(f"  X shape: {X.shape}, total points: {X.shape[0]}")

    # Optional PCA pre-reduction
    if args.pca_dims > 0 and args.pca_dims < X.shape[1]:
        print(f"Reducing to {args.pca_dims}D via PCA...")
        pca = PCA(n_components=args.pca_dims, random_state=42)
        X_reduced = pca.fit_transform(X)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  {args.pca_dims} PCs explain {var_explained:.1%} of variance")
    else:
        X_reduced = X

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    all_results = []

    # K-Means
    best_k = run_kmeans_sweep(X_scaled, meta, args.output_dir)
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    res = analyze_clusters(km_labels, meta, f"kmeans_k{best_k}", args.output_dir)
    all_results.append(res)

    # Also run k=3 (one per outer cycle) and k=9 (3T x 3 Sc groups)
    for k in [3, 9]:
        if k == best_k:
            continue
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_k_labels = km_k.fit_predict(X_scaled)
        res = analyze_clusters(km_k_labels, meta, f"kmeans_k{k}", args.output_dir)
        all_results.append(res)

    # HDBSCAN
    if HAS_HDBSCAN:
        print("\nRunning HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10,
                                     metric="euclidean", core_dist_n_jobs=-1)
        hdb_labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(hdb_labels) - {-1})
        noise = (hdb_labels == -1).mean()
        print(f"  HDBSCAN found {n_clusters} clusters, {noise:.1%} noise")
        if n_clusters > 1:
            res = analyze_clusters(hdb_labels, meta, "hdbscan", args.output_dir)
            all_results.append(res)
    else:
        print("\nhdbscan not installed, skipping. pip install hdbscan")

    # PCA scatter coloured by best K-Means clusters
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap="tab20",
                           s=3, alpha=0.4, rasterized=True)
    ax1.set_title(f"K-Means (k={best_k}) clusters in PCA space")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    fig.colorbar(scatter1, ax=ax1, label="Cluster")

    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=meta["T"], cmap="Set1",
                           s=3, alpha=0.4, rasterized=True)
    ax2.set_title("Outer cycle T in PCA space")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    fig.colorbar(scatter2, ax=ax2, label="T")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "cluster_pca_overview.png"), dpi=150)
    plt.close(fig)

    # Save results
    out_path = os.path.join(args.output_dir, "cluster_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
