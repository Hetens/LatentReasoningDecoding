"""
Build the Maze-Hard (30x30) dataset in TRM's on-disk format.

Adapted from the HRM reference builder (dataset/build_maze_dataset.py in
sapientinc/HRM) to this repo's layout: PuzzleDatasetMetadata and
dihedral_transform live in trm_base/metadata.py, and the metadata schema
here additionally requires total_puzzles.

Source: HuggingFace sapientinc/maze-30x30-hard-1k (train.csv / test.csv,
columns: source, q, a, rating). Grids are 30x30 = 900 tokens over the
charset "# SGo" (wall, path, start, goal, optimal-route cell).

Usage (from repo root):
    python trm_base/build_maze_data.py \
        --source-repo sapientinc/maze-30x30-hard-1k \
        --output-dir data/maze-30x30-hard-1k \
        --aug true
"""

from typing import Optional
import math
import os
import csv
import json

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from metadata import PuzzleDatasetMetadata, dihedral_transform


CHARSET = "# SGo"

cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/maze-30x30-hard-1k"
    output_dir: str = "data/maze-30x30-hard-1k"

    subsample_size: Optional[int] = None
    aug: bool = False


def convert_subset(set_name: str, config: DataProcessConfig):
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []

    csv_path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # header
        for source, q, a, rating in reader:
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)

            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(tqdm(inputs), labels):
        # 8-way dihedral augmentation (train only, when enabled).
        for aug_idx in range(8 if (set_name == "train" and config.aug) else 1):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        results["group_indices"].append(puzzle_id)

    assert len(all_chars - set(CHARSET)) == 0, f"unexpected chars: {all_chars - set(CHARSET)}"

    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    def _seq_to_numpy(seq):
        return np.vstack([char2id[s.reshape(-1)] for s in seq])

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),
        vocab_size=len(CHARSET) + 1,  # PAD + charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"[{set_name}] {results['inputs'].shape[0]} examples, seq_len={metadata.seq_len}, "
          f"vocab={metadata.vocab_size} -> {save_dir}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
