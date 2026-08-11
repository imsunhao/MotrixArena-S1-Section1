#!/usr/bin/env python3
"""Combine and filter Section 1 handoff-state captures."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-all", required=True, type=Path)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-test", required=True, type=Path)
    parser.add_argument("--upright-min", type=float, default=0.85)
    parser.add_argument("--angular-xy-max", type=float, default=1.2)
    parser.add_argument("--root-z-min", type=float, default=0.28)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def save_dataset(path, arrays, indices, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dof_pos=arrays["dof_pos"][indices],
        dof_vel=arrays["dof_vel"][indices],
        current_actions=arrays["current_actions"][indices],
        upright_min=np.asarray(args.upright_min, dtype=np.float32),
        angular_xy_max=np.asarray(args.angular_xy_max, dtype=np.float32),
        root_z_min=np.asarray(args.root_z_min, dtype=np.float32),
    )


def main():
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in (0, 1)")

    batches = [np.load(path, allow_pickle=False) for path in args.inputs]
    arrays = {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in ("dof_pos", "dof_vel", "current_actions")
    }
    quat = arrays["dof_pos"][:, 6:10]
    upright_cos = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
    angular_xy = np.linalg.norm(arrays["dof_vel"][:, 3:5], axis=1)
    root_z = arrays["dof_pos"][:, 5]
    healthy = np.logical_and.reduce(
        (
            upright_cos >= args.upright_min,
            angular_xy <= args.angular_xy_max,
            root_z >= args.root_z_min,
        )
    )
    healthy_indices = np.flatnonzero(healthy)
    if len(healthy_indices) < 2:
        raise ValueError("fewer than two healthy handoff states remain")

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(healthy_indices)
    test_count = max(1, int(round(len(shuffled) * args.test_fraction)))
    test_indices = shuffled[:test_count]
    train_indices = shuffled[test_count:]
    save_dataset(args.output_all, arrays, healthy_indices, args)
    save_dataset(args.output_train, arrays, train_indices, args)
    save_dataset(args.output_test, arrays, test_indices, args)
    print(
        f"captured={len(healthy)} healthy={len(healthy_indices)} "
        f"train={len(train_indices)} test={len(test_indices)}"
    )


if __name__ == "__main__":
    main()
