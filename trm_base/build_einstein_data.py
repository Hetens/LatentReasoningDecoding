"""
Generate Einstein / zebra logic-grid puzzles in TRM's on-disk format.

Why generate rather than use ZebraLogic: that benchmark states clues in
natural language and ships 1,000 eval puzzles, whereas TRM consumes
fixed-length token grids and needs a large training set. Generating gives
unlimited data, controllable difficulty, and exact ground-truth candidate
sets from the same constraint-propagation pattern used for Sudoku, which
keeps the probing methodology identical across tasks.

Puzzle: N houses x M attributes. The solution assigns, for each attribute,
a bijection between its N values and the N houses. Clues are true
statements about that solution; the clue set is reduced to a subset that
still admits exactly one solution.

Encoding (sequence length M*N + 5*K):
    [ solution region : M*N cells, row-major (attribute a, house h) ]
    [ clue region     : K clues x 5 tokens (TYPE, A1, V1, A2, V2)   ]

Vocabulary:
    0                       PAD / ignore label
    1                       BLANK (unfilled solution cell)
    2 .. N+1                index tokens 1..N (values and house numbers)
    N+2 .. N+M+1            attribute ids 0..M-1
    N+M+2                   ATTR_HOUSE (pseudo-attribute: "house position")
    N+M+3 .. N+M+6          clue types POSITION, SAME, LEFT, NEXTTO

Inputs blank the solution region and keep the clues; labels hold the
solution and set the clue region to 0, which the loader maps to the
ignore id so no loss is taken there.

Usage (from repo root):
    python trm_base/build_einstein_data.py \
        --n-houses 5 --n-attrs 5 \
        --n-train 40000 --n-test 2000 \
        --output-dir data/einstein-5x5
"""

from typing import List, Optional, Tuple
import itertools
import json
import os

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from metadata import PuzzleDatasetMetadata

cli = ArgParser()

# Clue type ids (local, offset into the vocab later).
POSITION, SAME, LEFT, NEXTTO = 0, 1, 2, 3
N_CLUE_TYPES = 4
CLUE_TOKENS = 5  # (TYPE, A1, V1, A2, V2)


class DataProcessConfig(BaseModel):
    n_houses: int = 5
    n_attrs: int = 5
    n_train: int = 40000
    n_test: int = 2000
    output_dir: str = "data/einstein-5x5"
    seed: int = 42
    max_clues: int = 20
    n_workers: int = 8


class Vocab:
    """Token layout for a given (N, M)."""

    def __init__(self, n: int, m: int):
        self.n, self.m = n, m
        self.PAD = 0
        self.BLANK = 1
        self.idx0 = 2                      # index tokens 1..N -> 2..N+1
        self.attr0 = 2 + n                 # attribute ids -> N+2 .. N+M+1
        self.ATTR_HOUSE = self.attr0 + m   # pseudo-attribute
        self.clue0 = self.ATTR_HOUSE + 1   # clue types
        self.size = self.clue0 + N_CLUE_TYPES

    def index(self, i: int) -> int:
        """1-based index (value or house number) -> token."""
        return self.idx0 + (i - 1)

    def attr(self, a: int) -> int:
        return self.attr0 + a

    def clue(self, t: int) -> int:
        return self.clue0 + t


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def _clue_holds(clue: Tuple, house_of) -> bool:
    """house_of[a][v] = house index (0-based) of value v for attribute a."""
    t, a1, v1, a2, v2 = clue
    if t == POSITION:
        return house_of[a1][v1] == v2
    h1, h2 = house_of[a1][v1], house_of[a2][v2]
    if t == SAME:
        return h1 == h2
    if t == LEFT:
        return h1 + 1 == h2
    if t == NEXTTO:
        return abs(h1 - h2) == 1
    raise ValueError(t)


def _propagate(dom: List[int], clues: List[Tuple], n: int, m: int, full: int) -> bool:
    """Bitmask constraint propagation to a fixpoint, in place.

    dom is a flat list indexed a*n+v holding a bitmask over houses.
    Returns False on contradiction (some domain empty).
    """
    changed = True
    while changed:
        changed = False

        for t, a1, v1, a2, v2 in clues:
            i1 = a1 * n + v1
            if t == POSITION:
                new = dom[i1] & (1 << v2)
                if new != dom[i1]:
                    dom[i1] = new
                    changed = True
                if not new:
                    return False
                continue

            i2 = a2 * n + v2
            d1, d2 = dom[i1], dom[i2]
            if t == SAME:
                n1, n2 = d1 & d2, d2 & d1
            elif t == LEFT:
                n1 = d1 & (d2 >> 1)
                n2 = d2 & ((d1 << 1) & full)
            else:  # NEXTTO
                n1 = d1 & (((d2 << 1) | (d2 >> 1)) & full)
                n2 = d2 & (((d1 << 1) | (d1 >> 1)) & full)
            if n1 != d1:
                dom[i1] = n1
                changed = True
            if n2 != d2:
                dom[i2] = n2
                changed = True
            if not n1 or not n2:
                return False

        # All-different within each attribute, both directions.
        for a in range(m):
            base = a * n
            for v in range(n):
                d = dom[base + v]
                if not d:
                    return False
                if d & (d - 1) == 0:  # singleton: strip from siblings
                    for w in range(n):
                        if w != v and dom[base + w] & d:
                            dom[base + w] &= ~d
                            if not dom[base + w]:
                                return False
                            changed = True
            for h in range(n):          # house claimed by exactly one value
                bit = 1 << h
                owners = [v for v in range(n) if dom[base + v] & bit]
                if not owners:
                    return False
                if len(owners) == 1 and dom[base + owners[0]] != bit:
                    dom[base + owners[0]] = bit
                    changed = True
    return True


def count_solutions(clues: List[Tuple], n: int, m: int, cap: int = 2) -> int:
    """Count solutions up to `cap` using propagation plus branching on the
    smallest remaining domain."""
    full = (1 << n) - 1
    dom = [full] * (n * m)
    total = 0

    def rec(dom: List[int]) -> bool:
        """Returns True once `cap` solutions have been found."""
        nonlocal total
        if not _propagate(dom, clues, n, m, full):
            return False

        best, best_size = -1, 99
        for i, d in enumerate(dom):
            sz = bin(d).count("1")
            if sz > 1 and sz < best_size:
                best, best_size = i, sz
        if best < 0:
            total += 1
            return total >= cap

        d = dom[best]
        for h in range(n):
            bit = 1 << h
            if d & bit:
                child = list(dom)
                child[best] = bit
                if rec(child):
                    return True
        return False

    rec(dom)
    return total


def all_true_clues(house_of, n: int, m: int) -> List[Tuple]:
    """Every clue of the supported types that is true of this solution."""
    out = []
    for a in range(m):
        for v in range(n):
            out.append((POSITION, a, v, 0, house_of[a][v]))
    for a1 in range(m):
        for a2 in range(m):
            for v1 in range(n):
                for v2 in range(n):
                    if a1 == a2:
                        continue
                    h1, h2 = house_of[a1][v1], house_of[a2][v2]
                    if h1 == h2:
                        out.append((SAME, a1, v1, a2, v2))
                    if h1 + 1 == h2:
                        out.append((LEFT, a1, v1, a2, v2))
                    if abs(h1 - h2) == 1 and a1 < a2:
                        out.append((NEXTTO, a1, v1, a2, v2))
    return out


def generate_puzzle(rng: np.random.Generator, n: int, m: int, max_clues: int):
    """Sample a solution, then a minimal-ish clue set with a unique solution."""
    house_of = [tuple(rng.permutation(n).tolist()) for _ in range(m)]

    candidates = all_true_clues(house_of, n, m)
    rng.shuffle(candidates)

    # Grow until unique. The growth path is random so it overshoots; the
    # shrink pass below is what has to respect max_clues, not this loop.
    chosen: List[Tuple] = []
    for c in candidates:
        chosen.append(c)
        if count_solutions(chosen, n, m) == 1:
            break
    else:
        return None

    # Shrink: drop any clue that is not needed for uniqueness.
    i = 0
    while i < len(chosen):
        trial = chosen[:i] + chosen[i + 1:]
        if trial and count_solutions(trial, n, m) == 1:
            chosen = trial
        else:
            i += 1

    if len(chosen) > max_clues:
        return None
    return house_of, chosen


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode(house_of, clues, vocab: Vocab, n: int, m: int, n_clue_slots: int):
    """Return (inputs, labels) int arrays of length m*n + 5*n_clue_slots."""
    seq_len = m * n + CLUE_TOKENS * n_clue_slots
    inp = np.zeros(seq_len, dtype=np.uint8)
    lab = np.zeros(seq_len, dtype=np.uint8)

    # Solution region: value at (attribute a, house h).
    value_at = [[0] * n for _ in range(m)]
    for a in range(m):
        for v in range(n):
            value_at[a][house_of[a][v]] = v
    for a in range(m):
        for h in range(n):
            pos = a * n + h
            inp[pos] = vocab.BLANK
            lab[pos] = vocab.index(value_at[a][h] + 1)

    # Clue region: identical in input and (ignored) in labels.
    base = m * n
    for k, (t, a1, v1, a2, v2) in enumerate(clues[:n_clue_slots]):
        off = base + k * CLUE_TOKENS
        inp[off] = vocab.clue(t)
        inp[off + 1] = vocab.attr(a1)
        inp[off + 2] = vocab.index(v1 + 1)
        if t == POSITION:
            inp[off + 3] = vocab.ATTR_HOUSE
            inp[off + 4] = vocab.index(v2 + 1)   # v2 holds the house index
        else:
            inp[off + 3] = vocab.attr(a2)
            inp[off + 4] = vocab.index(v2 + 1)
        # labels stay 0 in the clue region -> mapped to ignore by the loader

    return inp, lab


def _worker(job) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Generate `count` puzzles from `seed`. Module-level for pickling."""
    seed, count, n, m, max_clues, n_clue_slots = job
    rng = np.random.default_rng(seed)
    vocab = Vocab(n, m)
    ins, labs, hist = [], [], []
    while len(ins) < count:
        got = generate_puzzle(rng, n, m, max_clues)
        if got is None:
            continue
        house_of, clues = got
        if len(clues) > n_clue_slots:
            continue
        i_arr, l_arr = encode(house_of, clues, vocab, n, m, n_clue_slots)
        ins.append(i_arr)
        labs.append(l_arr)
        hist.append(len(clues))
    return np.vstack(ins), np.vstack(labs), hist


def build_split(set_name: str, n_examples: int, config: DataProcessConfig,
                vocab: Vocab, n_clue_slots: int, base_seed: int):
    import multiprocessing as mp

    n, m = config.n_houses, config.n_attrs
    n_workers = max(1, config.n_workers)
    per = [n_examples // n_workers] * n_workers
    for i in range(n_examples - sum(per)):
        per[i] += 1
    jobs = [
        (base_seed + 1000 * i, per[i], n, m, config.max_clues, n_clue_slots)
        for i in range(n_workers) if per[i] > 0
    ]

    print(f"[{set_name}] generating {n_examples} puzzles on {len(jobs)} workers ...")
    if len(jobs) == 1:
        parts = [_worker(jobs[0])]
    else:
        with mp.Pool(len(jobs)) as pool:
            parts = list(tqdm(pool.imap(_worker, jobs), total=len(jobs), desc=set_name))

    inputs = np.vstack([p[0] for p in parts])
    labels = np.vstack([p[1] for p in parts])
    n_clues_hist = [c for p in parts for c in p[2]]
    attempts = len(n_clues_hist)
    n_ex = len(inputs)

    results = {
        "inputs": inputs,
        "labels": labels,
        "group_indices": np.arange(n_ex + 1, dtype=np.int32),
        "puzzle_indices": np.arange(n_ex + 1, dtype=np.int32),
        "puzzle_identifiers": np.zeros(n_ex, dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=int(inputs.shape[1]),
        vocab_size=int(vocab.size),
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=n_ex,
        mean_puzzle_examples=1,
        total_puzzles=n_ex,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    print(f"[{set_name}] {n_ex} puzzles, seq_len={metadata.seq_len}, "
          f"vocab={metadata.vocab_size}, clues mean={np.mean(n_clues_hist):.1f} "
          f"min={min(n_clues_hist)} max={max(n_clues_hist)}, "
          f"accepted={attempts}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    n, m = config.n_houses, config.n_attrs
    vocab = Vocab(n, m)
    n_clue_slots = config.max_clues

    print(f"Einstein puzzles: {n} houses x {m} attributes")
    print(f"  seq_len = {m*n} solution + {CLUE_TOKENS*n_clue_slots} clue = "
          f"{m*n + CLUE_TOKENS*n_clue_slots}")
    print(f"  vocab_size = {vocab.size}")

    build_split("train", config.n_train, config, vocab, n_clue_slots, config.seed)
    build_split("test", config.n_test, config, vocab, n_clue_slots, config.seed + 777_000)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()
