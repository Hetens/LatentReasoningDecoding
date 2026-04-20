"""
Compute ground-truth candidate sets for Sudoku puzzles via constraint
propagation, and label each puzzle with a backtracking flag.

Two candidate-set modes:
  - ``initial``: one-pass row / column / box elimination.
  - ``cp``: iterative naked-single + hidden-single propagation until
    fixpoint (matches the report's "classical constraint-propagation solver").

Usage (from repo root):
    python -m experiments.probing.candidate_sets \
        --activations-dir results/probing/activations \
        --output-dir      results/probing/labels \
        --mode cp
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Set

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SUDOKU_PKG = os.path.join(_PROJECT_ROOT)
if _SUDOKU_PKG not in sys.path:
    sys.path.insert(0, _SUDOKU_PKG)

from sudoku.util import sudoku_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Puzzle string conversion
# ---------------------------------------------------------------------------

def inputs_to_puzzle_string(inputs: np.ndarray) -> str:
    """Convert a stored input row (values 1-10) to an 81-char puzzle string.

    Encoding (from ``build_sdku_data.py``):
      0 = PAD, 1 = blank ("0"), 2–10 = digits 1–9.
    """
    chars: list[str] = []
    for v in inputs:
        if v <= 1:
            chars.append(".")
        else:
            chars.append(str(int(v) - 1))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Candidate-set computation
# ---------------------------------------------------------------------------

def _peers(r: int, c: int) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i in range(9):
        if i != c:
            out.add((r, i))
        if i != r:
            out.add((i, c))
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            if (rr, cc) != (r, c):
                out.add((rr, cc))
    return list(out)


_PEERS_CACHE: dict[tuple[int, int], list[tuple[int, int]]] = {
    (r, c): _peers(r, c) for r in range(9) for c in range(9)
}


def compute_initial_candidates(puzzle: str) -> list[set[int]]:
    """One-pass row / column / box elimination.  No iterative propagation."""
    grid = [0] * 81
    for i, ch in enumerate(puzzle):
        if ch in "123456789":
            grid[i] = int(ch)

    cand: list[set[int]] = [set(range(1, 10)) if grid[i] == 0 else {grid[i]} for i in range(81)]

    for i in range(81):
        if grid[i] == 0:
            continue
        v = grid[i]
        for rr, cc in _PEERS_CACHE[(i // 9, i % 9)]:
            cand[rr * 9 + cc].discard(v)

    return cand


def compute_cp_candidates(puzzle: str) -> list[set[int]]:
    """Iterative constraint propagation (naked + hidden singles) until
    fixpoint.  Returns the candidate set for each cell at convergence.
    """
    grid = [[0] * 9 for _ in range(9)]
    for i, ch in enumerate(puzzle):
        if ch in "123456789":
            grid[i // 9][i % 9] = int(ch)

    cand: list[list[set[int]]] = [
        [set(range(1, 10)) if grid[r][c] == 0 else set() for c in range(9)]
        for r in range(9)
    ]
    # Initial elimination for givens.
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                v = grid[r][c]
                cand[r][c] = {v}
                for rr, cc in _PEERS_CACHE[(r, c)]:
                    cand[rr][cc].discard(v)

    def _set_cell(r: int, c: int, v: int) -> None:
        grid[r][c] = v
        cand[r][c] = {v}
        for rr, cc in _PEERS_CACHE[(r, c)]:
            cand[rr][cc].discard(v)

    changed = True
    while changed:
        changed = False

        # Naked singles
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0 and len(cand[r][c]) == 1:
                    _set_cell(r, c, next(iter(cand[r][c])))
                    changed = True

        # Hidden singles (box → row → col, QQWing order)
        for units in (
            [[(br + dr, bc + dc) for dr in range(3) for dc in range(3)]
             for br in range(0, 9, 3) for bc in range(0, 9, 3)],
            [[(r, c) for c in range(9)] for r in range(9)],
            [[(r, c) for r in range(9)] for c in range(9)],
        ):
            for unit in units:
                for d in range(1, 10):
                    cells = [(r, c) for r, c in unit if grid[r][c] == 0 and d in cand[r][c]]
                    if len(cells) == 1:
                        r, c = cells[0]
                        _set_cell(r, c, d)
                        changed = True

    # Flatten to 81 sets.
    return [cand[i // 9][i % 9] for i in range(81)]


def candidate_sets_to_binary(candidates: list[set[int]]) -> np.ndarray:
    """Encode 81 candidate sets as a ``(81, 9)`` binary matrix.

    ``y[c, k-1] = 1`` iff digit ``k`` is in the candidate set for cell ``c``.
    """
    y = np.zeros((81, 9), dtype=np.float32)
    for c, s in enumerate(candidates):
        for k in s:
            y[c, k - 1] = 1.0
    return y


# ---------------------------------------------------------------------------
# Backtracking flag
# ---------------------------------------------------------------------------

def puzzle_needs_backtracking(puzzle: str) -> bool:
    """Return True if a classical solver needs backtracking for *puzzle*."""
    m = sudoku_metrics(puzzle)
    return m.num_guesses > 0 or m.num_backtracks > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute candidate-set labels and backtracking flags.",
    )
    parser.add_argument(
        "--activations-dir", required=True,
        help="Directory containing inputs.npy (written by extract_activations).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode", choices=["initial", "cp"], default="cp",
        help="Candidate-set mode: 'initial' (one-pass) or 'cp' (iterative propagation).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    inputs = np.load(os.path.join(args.activations_dir, "inputs.npy"))
    N = len(inputs)
    print(f"Computing candidate sets for {N} puzzles (mode={args.mode}) …")

    compute_fn = compute_cp_candidates if args.mode == "cp" else compute_initial_candidates

    all_y = np.empty((N, 81, 9), dtype=np.float32)
    backtrack = np.empty(N, dtype=np.bool_)

    for idx in tqdm(range(N)):
        puzzle = inputs_to_puzzle_string(inputs[idx])
        cands = compute_fn(puzzle)
        all_y[idx] = candidate_sets_to_binary(cands)
        backtrack[idx] = puzzle_needs_backtracking(puzzle)

    np.save(os.path.join(args.output_dir, "candidate_labels.npy"), all_y)
    np.save(os.path.join(args.output_dir, "backtrack_flags.npy"), backtrack)
    print(f"Saved candidate_labels.npy  shape={all_y.shape}")
    print(f"Saved backtrack_flags.npy   shape={backtrack.shape}")
    bt_count = backtrack.sum()
    print(f"  {bt_count}/{N} puzzles require backtracking ({100*bt_count/N:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
