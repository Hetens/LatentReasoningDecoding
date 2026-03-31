"""
Inspect samples from Sudoku (.npy) and Zebra (.pkl) datasets.

Dataset format per:
https://github.com/kulinshah98/llm-reasoning-logic-puzzles

Run from project root: python -m sudoku.inspect_samples
"""

from pathlib import Path
import numpy as np

# Strategy names for Sudoku (from README)
SUDOKU_STRATEGY = {
    0: "given",
    2: "Lone single",
    3: "Hidden single",
    4: "Naked pair",
    5: "Naked Triplet",
    6: "Locked Candidate",
    7: "XY Wing",
    8: "Unique Rectangle",
}


def load_sudoku_samples(data_dir: Path, max_samples: int = 2):
    """Load and parse Sudoku .npy data. Each example has 325 entries."""
    for npy_file in sorted(data_dir.glob("*.npy")):
        arr = np.load(npy_file, allow_pickle=True)
        # Shape is typically (num_puzzles, 325)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = min(max_samples, len(arr))
        yield npy_file.name, arr, n


def parse_sudoku_example(raw):
    """Parse one Sudoku example (325 entries) into givens and filled cells."""
    num_givens = int(raw[0])
    # 324 = 81*4: for each cell (row, col, value, strategy)
    rest = raw[1:325].reshape(-1, 4)  # (81, 4)
    givens = rest[:num_givens]        # (row, col, value, strategy)
    empties = rest[num_givens:]        # (row, col, value, strategy) for cells to fill
    return num_givens, givens, empties


def sudoku_grid_from_example(givens, empties):
    """Build 9x9 grid (0 = empty) from givens only; then optional full from givens+empties."""
    grid = np.zeros((9, 9), dtype=int)
    for r, c, val, _ in givens:
        grid[int(r), int(c)] = int(val)
    return grid


def sudoku_grid_full(givens, empties):
    """Build full 9x9 solved grid."""
    grid = sudoku_grid_from_example(givens, empties)
    for r, c, val, _ in empties:
        grid[int(r), int(c)] = int(val)
    return grid


def sudoku_example_to_puzzle_string(raw):
    """Convert one Sudoku example (325 entries) to 81-char puzzle string for util.sudoku_metrics."""
    num_givens, givens, _ = parse_sudoku_example(raw)
    grid = sudoku_grid_from_example(givens, [])
    chars = []
    for r in range(9):
        for c in range(9):
            v = grid[r, c]
            chars.append(str(v) if v else ".")
    return "".join(chars)


def print_sudoku_grid(grid, title="Grid"):
    """Pretty-print 9x9 Sudoku grid."""
    print(f"  {title}")
    sep = "  +-------+-------+-------+"
    print(sep)
    for r in range(9):
        parts = []
        for c in range(9):
            v = grid[r, c]
            parts.append(str(v) if v else ".")
        line = "  | " + " ".join(parts[0:3]) + " | " + " ".join(parts[3:6]) + " | " + " ".join(parts[6:9]) + " |"
        print(line)
        if (r + 1) % 3 == 0:
            print(sep)
    print()


def show_sudoku_samples(data_dir: Path, num_samples: int = 2):
    """Load Sudoku .npy, print a few samples."""
    print("=" * 60)
    print("SUDOKU DATASET (.npy)")
    print("=" * 60)
    for name, arr, n in load_sudoku_samples(data_dir, max_samples=num_samples):
        print(f"\nFile: {name}  (showing {n} of {len(arr)} examples)")
        print(f"Shape: {arr.shape}  (each row = 325 entries: 1 num_givens + 81*4 cell info)")
        for idx in range(n):
            raw = arr[idx]
            num_givens, givens, empties = parse_sudoku_example(raw)
            print(f"\n--- Example {idx + 1} ---")
            print(f"  Num givens: {num_givens}")
            grid_puzzle = sudoku_grid_from_example(givens, empties)
            print_sudoku_grid(grid_puzzle, "Puzzle (givens only)")
            grid_solved = sudoku_grid_full(givens, empties)
            print_sudoku_grid(grid_solved, "Solution")
            print("  Givens (row, col, value, strategy):")
            for r, c, val, strat in givens[:15]:  # first 15
                print(f"    ({r}, {c}) = {val}  strategy={strat} ({SUDOKU_STRATEGY.get(int(strat), strat)})")
            if len(givens) > 15:
                print(f"    ... and {len(givens) - 15} more givens")
            print("  First 5 filled (empty) cells (row, col, value, strategy):")
            for r, c, val, strat in empties[:5]:
                print(f"    ({r}, {c}) = {val}  strategy={strat} ({SUDOKU_STRATEGY.get(int(strat), strat)})")
            print()


def load_zebra_samples(data_dir: Path, max_samples: int = 2):
    """Load Zebra .pkl and yield (filename, list of samples, count to show)."""
    import pickle
    for pkl_file in sorted(data_dir.glob("*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            data = [data]
        n = min(max_samples, len(data))
        yield pkl_file.name, data, n


def _zebra_clues_as_strings(clues_list):
    """Group tokenized clue list into full clue strings (split on CLUE_END token)."""
    if not clues_list:
        return []
    try:
        # Each clue ends with token "CLUE_END"
        parts, current = [], []
        for t in clues_list:
            current.append(str(t))
            if t == "CLUE_END":
                parts.append(" ".join(current))
                current = []
        if current:
            parts.append(" ".join(current))
        return parts[:15]
    except Exception:
        return [str(clues_list[:15]) + "..."]


def show_zebra_samples(data_dir: Path, num_samples: int = 2):
    """Load Zebra .pkl, print a few samples."""
    print("=" * 60)
    print("ZEBRA PUZZLE DATASET (.pkl)")
    print("=" * 60)
    for name, data, n in load_zebra_samples(data_dir, max_samples=num_samples):
        print(f"\nFile: {name}  (showing {n} of {len(data)} examples)")
        for idx in range(n):
            sample = data[idx]
            print(f"\n--- Example {idx + 1} ---")
            if len(sample) != 3:
                print(f"  Unexpected structure: len(sample) = {len(sample)}")
                print(f"  Sample: {sample[:2] if len(sample) >= 2 else sample}")
                continue
            clues, solution_box, fill_order = sample[0], sample[1], sample[2]
            # Clues are a flat list of tokens; each clue ends with CLUE_END
            num_tokens = len(clues) if hasattr(clues, "__len__") else 0
            full_clues = _zebra_clues_as_strings(clues)
            print(f"  Clues: {num_tokens} tokens -> {len(full_clues)} full clues (format: [Clue type] LHS [c/n] ... RHS ... CLUE_END)")
            print("  First 5 clues:")
            for i, c in enumerate(full_clues[:5]):
                print(f"    [{i}] {c[:90]}{'...' if len(c) > 90 else ''}")
            print("  Solution box (first row = house numbers; following rows = attributes):")
            for i, row in enumerate(solution_box):
                print(f"    Row {i}: {row}")
            print("  Fill order (first 10 entries) - order solver fills cells:")
            for i, entry in enumerate(np.array(fill_order)[:10] if hasattr(fill_order, "__len__") else []):
                print(f"    {i}: {entry}")
            total = len(fill_order) if hasattr(fill_order, "__len__") else 0
            if total > 10:
                print(f"    ... total {total} entries")
            print()


def main():
    data_dir = Path(__file__).resolve().parent.parent / "datasets"
    if not data_dir.is_dir():
        print(f"Dataset directory not found: {data_dir}")
        print("Create 'datasets' and add sudoku-test-data.npy and zebra-test-data.pkl")
        return

    num_sudoku = 1
    num_zebra = 1
    show_sudoku_samples(data_dir, num_samples=num_sudoku)
    show_zebra_samples(data_dir, num_samples=num_zebra)
    print("Done.")


if __name__ == "__main__":
    main()
