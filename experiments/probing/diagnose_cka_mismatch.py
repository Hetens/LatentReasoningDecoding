"""
Locate the origin of the stored CKA grid.

Established: running the repo's own compute_self_cka_grid recipe (all 81
cells, all 5,000 puzzles, float64) on z_L_act16.pt gives within-cycle
0.965 and cycle 1 to 3 0.912, and those block means are stable from 3,200
to 405,000 rows, so sample size is ruled out. The stored grid
self_cka_z_L.npy instead has 0.811 / 0.637 with a global minimum of 0.475,
which is also what the published Fig. 2a shows.

Candidate explanations tested here:
  H1  float16 numerics. App. A records that "the original float16 storage
      of z_L produced overflow in the Frobenius norms" before a float64
      cast was added, so the stored artifact may predate that fix.
  H2  the grid came from ACT step 1 rather than step 16.
  H3  the grid came from z_H rather than z_L.

Because the block means are sample-size stable, 200 puzzles (16,200 rows)
suffices and is 25x cheaper than 1,000. Matmuls run on GPU; each source
file is loaded once and shared across the numeric variants.

Usage (from repo root):
    python -u -m experiments.probing.diagnose_cka_mismatch \
        --act-dir results/probing/activations \
        --stored-grid results/probing/cka/self_cka_z_L.npy \
        --n-puzzles 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRM_BASE = os.path.join(_PROJECT_ROOT, "trm_base")
for p in (_PROJECT_ROOT, _TRM_BASE):
    if p not in sys.path:
        sys.path.insert(0, p)


def log(*a):
    print(*a, flush=True)


def block_means(grid: np.ndarray) -> dict:
    n_T = 3
    per = grid.shape[0] // n_T
    if per == 0:
        return {"within": float("nan")}

    def blk(a, b):
        return float(np.nanmean(grid[a * per:(a + 1) * per, b * per:(b + 1) * per]))

    return {
        "within": round(float(np.nanmean([blk(t, t) for t in range(n_T)])), 3),
        "c1_c2": round(blk(0, 1), 3),
        "c1_c3": round(blk(0, n_T - 1), 3),
        "c2_c3": round(blk(1, n_T - 1), 3),
        "min": round(float(np.nanmin(grid)), 3),
        "n_nan": int(np.isnan(grid).sum()),
    }


@torch.no_grad()
def cka_grid(mats: list, device: torch.device, mode: str) -> np.ndarray:
    """Pairwise linear CKA.

    mode:
      'fp64'      centered fp32 matmul, fp64 norms (what the current code does)
      'fp16norm'  fp32 matmul, Gram cast to fp16 before the Frobenius norm,
                  reproducing "float16 storage overflowed in the norms"
      'fp16all'   inputs and matmul in fp16 (accumulation still fp32 on
                  tensor cores, so this is a weaker form of H1)
    """
    K = len(mats)
    Xs, self_n = [], []
    for m in mats:
        X = m.to(device, torch.float32)
        X = X - X.mean(dim=0, keepdim=True)
        if mode == "fp16all":
            X = X.half()
        Xs.append(X)
        g = (X.float().T @ X.float())
        if mode == "fp16norm":
            g = g.half()
        self_n.append(torch.linalg.matrix_norm(g.float(), ord="fro").double())

    grid = np.eye(K)
    for a in range(K):
        for b in range(a + 1, K):
            c = (Xs[b].float().T @ Xs[a].float())
            if mode == "fp16norm":
                c = c.half()
            num = (torch.linalg.matrix_norm(c.float(), ord="fro").double() ** 2)
            den = self_n[a] * self_n[b]
            v = float((num / den).item()) if torch.isfinite(den) and den > 0 else float("nan")
            grid[a, b] = grid[b, a] = v
    del Xs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return grid


def load_mats(path: str, n_puzzles: int):
    z = torch.load(path, map_location="cpu")
    if isinstance(z, dict):
        z = z.get("z_L", z.get("z_H", next(iter(z.values()))))
    shp = tuple(z.shape)
    if z.dim() == 5:
        N, H, L, C, D = shp
        K = H * L
    else:
        N, H, C, D = shp
        K = H
    flat = z.reshape(N, K, C, D)[:min(n_puzzles, N)]
    mats = [flat[:, k].reshape(-1, D).clone() for k in range(K)]
    del z, flat
    return shp, mats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", required=True)
    ap.add_argument("--stored-grid", required=True)
    ap.add_argument("--n-puzzles", type=int, default=200)
    ap.add_argument("--output", default="results/probing/cka/diagnose_cka.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    stored = np.load(args.stored_grid)
    log("=== stored grid (target to reproduce) ===")
    log("   ", block_means(stored))
    out = {"stored": block_means(stored), "candidates": {}}

    # Group by file so each 7.5 GB load happens once.
    plan = [
        ("z_L_act16.pt", ["fp64", "fp16norm", "fp16all"]),
        ("z_L_act1.pt",  ["fp64", "fp16norm"]),
        ("z_H_act16.pt", ["fp64"]),
        ("z_H_act1.pt",  ["fp64"]),
    ]

    for fname, modes in plan:
        path = os.path.join(args.act_dir, fname)
        if not os.path.exists(path):
            log(f"\n--- {fname}: SKIP (missing) ---")
            continue
        log(f"\n--- loading {fname} ---")
        try:
            shp, mats = load_mats(path, args.n_puzzles)
        except Exception as e:  # noqa: BLE001
            log(f"    FAILED to load: {type(e).__name__}: {e}")
            continue
        log(f"    shape {shp}  ->  {len(mats)} states x {mats[0].shape[0]} rows")

        for mode in modes:
            name = f"{fname} [{mode}]"
            try:
                g = cka_grid(mats, device, mode)
                bm = block_means(g)
                bm["n_states"] = len(mats)
                if g.shape == stored.shape:
                    bm["mad_vs_stored"] = round(
                        float(np.nanmean(np.abs(g - stored))), 4)
                out["candidates"][name] = bm
                log(f"    {name}")
                log(f"      {bm}")
            except Exception as e:  # noqa: BLE001
                log(f"    {name} FAILED: {type(e).__name__}: {e}")
        del mats

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved -> {args.output}")

    scored = {k: v["mad_vs_stored"] for k, v in out["candidates"].items()
              if "mad_vs_stored" in v and not np.isnan(v["mad_vs_stored"])}
    log("\n=== CONCLUSION ===")
    if scored:
        for k in sorted(scored, key=scored.get):
            log(f"  {scored[k]:.4f}  mean abs diff vs stored   {k}")
        best = min(scored, key=scored.get)
        log(f"\n  Closest: {best}")
        log("  A near-zero difference identifies what the published figure shows.")
        log("  If nothing is close, the stored grid came from a source no longer")
        log("  present in the repo, and the corrected value stands at ~0.91.")
    else:
        log("  No comparable candidate produced a grid of the stored shape.")


if __name__ == "__main__":
    main()
