"""
Ablation study runner — tests whether spatial reasoning in TRMs arises from
RoPE embeddings, CastedLinear layers, or both.

Experiments (all share the same seed, data, and hyper-parameters):
  1. baseline_rope_casted      — default TRM (control)
  2. ablation_learned_casted   — swap RoPE → learned positional embeddings
  3. ablation_rope_standard    — swap CastedLinear → nn.Linear
  4. ablation_learned_standard — swap both

Usage:
    # Run all four experiments sequentially:
    python experiments/ablation/run_ablation.py

    # Run a single experiment by name:
    python experiments/ablation/run_ablation.py --experiment baseline_rope_casted

    # Dry-run (print commands without executing):
    python experiments/ablation/run_ablation.py --dry-run
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
PRETRAIN_SCRIPT = os.path.join(REPO_ROOT, "trm_base", "pretrain.py")

EXPERIMENTS = [
    "baseline_rope_casted",
    "ablation_learned_casted",
    "ablation_rope_standard",
    "ablation_learned_standard",
]


def build_command(experiment_name: str) -> list[str]:
    config_path = os.path.join(CONFIG_DIR, f"{experiment_name}.yml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    return [sys.executable, PRETRAIN_SCRIPT, "--config", config_path]


def main():
    parser = argparse.ArgumentParser(description="Run spatial-reasoning ablation experiments on the TRM.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=EXPERIMENTS,
        default=None,
        help="Run a single experiment by name. If omitted, runs all four sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    args = parser.parse_args()

    experiments_to_run = [args.experiment] if args.experiment else EXPERIMENTS

    print("=" * 70)
    print("  ABLATION STUDY: Spatial Reasoning in TRMs")
    print("  Testing: RoPE vs Learned Positional Embeddings")
    print("           CastedLinear vs Standard nn.Linear")
    print("=" * 70)
    print()

    for i, name in enumerate(experiments_to_run, 1):
        cmd = build_command(name)
        print(f"[{i}/{len(experiments_to_run)}] {name}")
        print(f"  Config : {cmd[-1]}")
        print(f"  Command: {' '.join(cmd)}")
        print()

        if args.dry_run:
            continue

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        trm_base_dir = os.path.join(REPO_ROOT, "trm_base")
        env["PYTHONPATH"] = trm_base_dir + (os.pathsep + existing if existing else "")

        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            print(f"\n  ** Experiment '{name}' exited with code {result.returncode} **")
            print("  Continuing to next experiment...\n")
        else:
            print(f"\n  Experiment '{name}' completed successfully.\n")

    print("=" * 70)
    print("  All ablation experiments finished.")
    print("  Check W&B project 'ablation-spatial-reasoning' for results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
