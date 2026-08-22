#!/usr/bin/env python3
"""Generate the synthetic train/val/test datasets for both testbeds.

Usage:
    python scripts/generate_datasets.py
    python scripts/generate_datasets.py --n-train 6000 --n-val 800 --n-test 800

Seeds are fixed per split so the three splits never overlap and the test set
stays a genuine held-out set across regenerations.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator.data_generator import (
    generate_one_node_dataset,
    generate_two_node_dataset,
    save_dataset,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DEFAULT_N_TRAIN, DEFAULT_N_VAL, DEFAULT_N_TEST = 6000, 800, 800


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    return p.parse_args()


def main():
    args = parse_args()
    # (split_name, n_runs, seed)
    ONE_NODE_SPLITS = [
        ("train", args.n_train, 1001), ("val", args.n_val, 1002), ("test", args.n_test, 1003),
    ]
    TWO_NODE_SPLITS = [
        ("train", args.n_train, 2001), ("val", args.n_val, 2002), ("test", args.n_test, 2003),
    ]

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating 1-node datasets...")
    for split, n_runs, seed in ONE_NODE_SPLITS:
        t0 = time.time()
        data = generate_one_node_dataset(n_runs=n_runs, seed=seed)
        path = os.path.join(DATA_DIR, f"motor_1node_{split}.npz")
        save_dataset(path, data)
        print(f"  {split}: {n_runs} runs -> {path} ({time.time() - t0:.1f}s)")

    print("Generating 2-node datasets...")
    for split, n_runs, seed in TWO_NODE_SPLITS:
        t0 = time.time()
        data = generate_two_node_dataset(n_runs=n_runs, seed=seed)
        path = os.path.join(DATA_DIR, f"motor_2node_{split}.npz")
        save_dataset(path, data)
        print(f"  {split}: {n_runs} runs -> {path} ({time.time() - t0:.1f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
