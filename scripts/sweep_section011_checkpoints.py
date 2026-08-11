#!/usr/bin/env python3
"""Evaluate and rank a sequence of Section 1 checkpoints.

The sweep is resumable: completed checkpoint/seed pairs are read from the JSONL
output and skipped on subsequent invocations.  This makes it suitable for a
long-running process that starts as soon as a training PID exits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--min-step", type=int, default=2500)
    parser.add_argument("--max-step", type=int, required=True)
    parser.add_argument("--step", type=int, default=2500)
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--max-control-steps", type=int, default=5000)
    parser.add_argument(
        "--ranking-mode",
        choices=("route", "skill"),
        default="route",
        help="Rank full-route checkpoints or local curriculum skills",
    )
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--evaluate-script",
        type=Path,
        default=Path(__file__).with_name("evaluate_section011.py"),
    )
    return parser.parse_args()


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int, poll_seconds: float) -> None:
    print(f"waiting for training PID {pid}", flush=True)
    while pid_is_running(pid):
        time.sleep(poll_seconds)
    # Give the CUDA context a moment to release its memory before evaluation.
    time.sleep(min(poll_seconds, 15.0))
    print(f"training PID {pid} exited; starting sweep", flush=True)


def load_completed(path: Path) -> set[tuple[int, int]]:
    completed: set[tuple[int, int]] = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") == "ok":
            completed.add((int(item["checkpoint_step"]), int(item["seed"])))
    return completed


def parse_metrics(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[index + end :].strip():
            return value
    raise ValueError("evaluation stdout did not end with a JSON object")


def ranking_key(item: dict[str, Any], ranking_mode: str) -> tuple[float, ...]:
    metrics = item["metrics"]
    if ranking_mode == "skill":
        return (
            float(metrics.get("skill_success_rate", 0.0)),
            float(metrics.get("all_time_max_y", float("-inf"))),
            float(metrics.get("mean_episode_max_y", float("-inf"))),
            -float(metrics.get("fall_rate", 1.0)),
        )
    return (
        float(metrics.get("stable_success_rate", 0.0)),
        float(metrics.get("ever_on_platform_rate", 0.0)),
        float(metrics.get("all_time_max_waypoints", 0.0)),
        float(metrics.get("all_time_max_y", float("-inf"))),
        float(metrics.get("mean_episode_max_y", float("-inf"))),
        -float(metrics.get("fall_rate", 1.0)),
    )


def write_summary(
    jsonl_path: Path, summary_path: Path, ranking_mode: str
) -> None:
    successful = []
    for line in jsonl_path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") == "ok":
            successful.append(item)
    successful.sort(
        key=lambda item: ranking_key(item, ranking_mode), reverse=True
    )
    ranking_order = (
        [
            "skill_success_rate",
            "all_time_max_y",
            "mean_episode_max_y",
            "negative_fall_rate",
        ]
        if ranking_mode == "skill"
        else [
            "stable_success_rate",
            "ever_on_platform_rate",
            "all_time_max_waypoints",
            "all_time_max_y",
            "mean_episode_max_y",
            "negative_fall_rate",
        ]
    )
    summary = {
        "ranking_mode": ranking_mode,
        "ranking_order": ranking_order,
        "evaluations": len(successful),
        "ranked_results": successful,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    args = parse_args()
    if args.min_step <= 0 or args.step <= 0 or args.max_step < args.min_step:
        raise SystemExit("invalid checkpoint step range")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    if args.wait_pid:
        wait_for_pid(args.wait_pid, args.wait_poll_seconds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    completed = load_completed(jsonl_path)

    checkpoint_dir = args.run_dir / "checkpoints"
    for checkpoint_step in range(args.min_step, args.max_step + 1, args.step):
        checkpoint = checkpoint_dir / f"agent_{checkpoint_step}.pickle"
        if not checkpoint.exists():
            print(f"missing checkpoint: {checkpoint}", flush=True)
            continue
        for seed in seeds:
            if (checkpoint_step, seed) in completed:
                print(f"skip completed step={checkpoint_step} seed={seed}", flush=True)
                continue

            stem = f"agent_{checkpoint_step}_seed{seed}"
            stdout_path = args.output_dir / f"{stem}.stdout.log"
            stderr_path = args.output_dir / f"{stem}.stderr.log"
            command = [
                args.python,
                str(args.evaluate_script),
                f"--env={args.env}",
                f"--policy={checkpoint}",
                f"--num-envs={args.num_envs}",
                f"--episodes={args.episodes}",
                f"--max-control-steps={args.max_control_steps}",
                f"--seed={seed}",
            ]
            print(f"evaluate step={checkpoint_step} seed={seed}", flush=True)
            result = subprocess.run(command, text=True, capture_output=True)
            stdout_path.write_text(result.stdout)
            stderr_path.write_text(result.stderr)
            item: dict[str, Any] = {
                "checkpoint_step": checkpoint_step,
                "seed": seed,
                "policy": str(checkpoint),
                "returncode": result.returncode,
            }
            if result.returncode == 0:
                try:
                    item["metrics"] = parse_metrics(result.stdout)
                    item["status"] = "ok"
                except ValueError as error:
                    item["status"] = "parse_error"
                    item["error"] = str(error)
            else:
                item["status"] = "process_error"
            with jsonl_path.open("a") as output:
                output.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            write_summary(jsonl_path, summary_path, args.ranking_mode)

    if jsonl_path.exists():
        write_summary(jsonl_path, summary_path, args.ranking_mode)
    print(f"sweep complete: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
