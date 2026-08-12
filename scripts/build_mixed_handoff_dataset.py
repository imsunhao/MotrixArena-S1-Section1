#!/usr/bin/env python3
"""Mix nominal local resets with scarce dynamic handoff states.

Dynamic states are split before oversampling so an identical state cannot leak
from the training split into the held-out split.
"""

import argparse
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("dof_pos", "dof_vel", "current_actions")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--dynamic", required=True, nargs="+", type=Path)
    parser.add_argument("--output-all", required=True, type=Path)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-test", required=True, type=Path)
    parser.add_argument("--dynamic-repeat", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--upright-min", type=float, default=0.7)
    parser.add_argument("--angular-xy-max", type=float, default=3.0)
    parser.add_argument("--root-z-min", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_many(paths):
    batches = [np.load(path, allow_pickle=False) for path in paths]
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in ARRAY_KEYS
    }


def filter_indices(arrays, args):
    quat = arrays["dof_pos"][:, 6:10]
    upright_cos = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
    angular_xy = np.linalg.norm(arrays["dof_vel"][:, 3:5], axis=1)
    root_z = arrays["dof_pos"][:, 5]
    return np.flatnonzero(
        np.logical_and.reduce(
            (
                upright_cos >= args.upright_min,
                angular_xy <= args.angular_xy_max,
                root_z >= args.root_z_min,
            )
        )
    )


def split_indices(indices, test_fraction, rng):
    shuffled = rng.permutation(indices)
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    if test_count >= len(shuffled):
        test_count = len(shuffled) - 1
    return shuffled[test_count:], shuffled[:test_count]


def select(arrays, indices):
    return {key: arrays[key][indices] for key in ARRAY_KEYS}


def concatenate(*groups):
    return {
        key: np.concatenate([group[key] for group in groups], axis=0)
        for key in ARRAY_KEYS
    }


def repeat(arrays, count):
    return {key: np.repeat(arrays[key], count, axis=0) for key in ARRAY_KEYS}


def save(path, arrays, args, dynamic_unique):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        dynamic_repeat=np.asarray(args.dynamic_repeat, dtype=np.int32),
        dynamic_unique=np.asarray(dynamic_unique, dtype=np.int32),
        upright_min=np.asarray(args.upright_min, dtype=np.float32),
        angular_xy_max=np.asarray(args.angular_xy_max, dtype=np.float32),
        root_z_min=np.asarray(args.root_z_min, dtype=np.float32),
    )


def main():
    args = parse_args()
    if args.dynamic_repeat <= 0:
        raise ValueError("--dynamic-repeat must be positive")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in (0, 1)")

    base = load_many([args.base])
    dynamic = load_many(args.dynamic)
    base_indices = filter_indices(base, args)
    dynamic_indices = filter_indices(dynamic, args)
    if len(base_indices) < 2:
        raise ValueError("fewer than two usable base states remain")
    if len(dynamic_indices) < 2:
        raise ValueError("fewer than two usable dynamic states remain")

    rng = np.random.default_rng(args.seed)
    base_train_idx, base_test_idx = split_indices(
        base_indices, args.test_fraction, rng
    )
    dynamic_train_idx, dynamic_test_idx = split_indices(
        dynamic_indices, args.test_fraction, rng
    )

    base_train = select(base, base_train_idx)
    base_test = select(base, base_test_idx)
    dynamic_train = select(dynamic, dynamic_train_idx)
    dynamic_test = select(dynamic, dynamic_test_idx)
    train = concatenate(base_train, repeat(dynamic_train, args.dynamic_repeat))
    test = concatenate(base_test, dynamic_test)
    all_unique = concatenate(
        select(base, base_indices), select(dynamic, dynamic_indices)
    )

    save(args.output_all, all_unique, args, len(dynamic_indices))
    save(args.output_train, train, args, len(dynamic_train_idx))
    save(args.output_test, test, args, len(dynamic_test_idx))
    print(
        f"base={len(base_indices)} dynamic={len(dynamic_indices)} "
        f"train={len(train['dof_pos'])} test={len(test['dof_pos'])} "
        f"dynamic_train={len(dynamic_train_idx)} "
        f"dynamic_test={len(dynamic_test_idx)}"
    )


if __name__ == "__main__":
    main()
