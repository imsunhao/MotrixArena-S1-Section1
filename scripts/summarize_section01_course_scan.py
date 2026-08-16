"""Summarize concatenated Section01 evaluation JSON from noisy log files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def _records(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            return
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + length
        if isinstance(value, dict) and "policy" in value:
            yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--require-successes",
        type=int,
        default=0,
        help="Only show policies with at least this many successful evaluations",
    )
    args = parser.parse_args()

    grouped = defaultdict(list)
    for log in args.logs:
        for record in _records(log):
            grouped[record["policy"]].append(record)

    summaries = []
    for policy, records in grouped.items():
        total_episodes = sum(record["completed_episodes"] for record in records)
        success_episodes = round(
            sum(
                record["success_rate"] * record["completed_episodes"]
                for record in records
            )
        )
        fall_episodes = round(
            sum(
                record["fall_rate"] * record["completed_episodes"]
                for record in records
            )
        )
        timeout_episodes = round(
            sum(
                record["timeout_rate"] * record["completed_episodes"]
                for record in records
            )
        )
        successes = sum(record["success_rate"] > 0 for record in records)
        if successes < args.require_successes:
            continue
        summaries.append(
            {
                "policy": policy,
                "evaluations": len(records),
                "seeds": [record["seed"] for record in records],
                "completed_episodes": total_episodes,
                "success_episodes": success_episodes,
                "fall_episodes": fall_episodes,
                "timeout_episodes": timeout_episodes,
                "mean_success_rate": success_episodes / total_episodes,
                "mean_fall_rate": fall_episodes / total_episodes,
                "mean_timeout_rate": timeout_episodes / total_episodes,
                "successes": successes,
                "ever_on_platform": sum(
                    record["ever_on_platform_rate"] > 0 for record in records
                ),
                "stable_successes": sum(
                    record["stable_success_rate"] > 0 for record in records
                ),
                "falls": sum(record["fall_rate"] > 0 for record in records),
                "mean_episode_max_y": sum(
                    record["mean_episode_max_y"] * record["completed_episodes"]
                    for record in records
                )
                / total_episodes,
                "all_time_max_y": max(record["all_time_max_y"] for record in records),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["stable_successes"],
            item["ever_on_platform"],
            item["mean_success_rate"],
            item["successes"],
            item["mean_episode_max_y"],
            -item["mean_fall_rate"],
            -item["mean_timeout_rate"],
        ),
        reverse=True,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
