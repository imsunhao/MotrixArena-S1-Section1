"""Summarize Section01 evaluation outcomes by formal start position."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from summarize_section01_course_scan import _records


X_EDGES = np.asarray((-0.5, -0.25, 0.0, 0.25, 0.5), dtype=np.float32)
Y_EDGES = np.asarray((-2.9, -2.65, -2.4, -2.15, -2.0), dtype=np.float32)


def _bin_label(value: float, edges: np.ndarray) -> str:
    index = int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))
    return f"[{edges[index]:.2f},{edges[index + 1]:.2f}]"


def _summarize(values: list[dict]) -> dict:
    platform = [bool(value["ever_on_platform"]) for value in values]
    waypoint_keys = tuple(values[0].get("crossing_steps", ())) if values else ()
    return {
        "episodes": len(values),
        "ever_on_platform": int(sum(platform)),
        "platform_rate": float(np.mean(platform)) if values else 0.0,
        "mean_max_y": (
            float(np.mean([value["max_y"] for value in values])) if values else None
        ),
        "waypoint_crossing_counts": {
            waypoint: int(
                sum(int(value.get("crossing_steps", {}).get(waypoint, -1)) >= 0 for value in values)
            )
            for waypoint in waypoint_keys
        },
        "termination_reasons": dict(
            Counter(str(value["termination_reason"]) for value in values)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    episodes = []
    for log in args.logs:
        for record in _records(log):
            for detail in record.get("episode_details", ()):
                if "start_y" not in detail:
                    raise ValueError(f"{log} lacks start_y; rerun with the current evaluator")
                episodes.append(detail)

    cells: dict[tuple[str, str], list[dict]] = {}
    for episode in episodes:
        key = (
            _bin_label(float(episode["start_x"]), X_EDGES),
            _bin_label(float(episode["start_y"]), Y_EDGES),
        )
        cells.setdefault(key, []).append(episode)

    result = []
    for (x_bin, y_bin), values in sorted(cells.items()):
        result.append({"x_bin": x_bin, "y_bin": y_bin, **_summarize(values)})

    x_groups: dict[str, list[dict]] = {}
    y_groups: dict[str, list[dict]] = {}
    for episode in episodes:
        x_groups.setdefault(_bin_label(float(episode["start_x"]), X_EDGES), []).append(episode)
        y_groups.setdefault(_bin_label(float(episode["start_y"]), Y_EDGES), []).append(episode)

    print(
        json.dumps(
            {
                "episodes": len(episodes),
                "overall_platform_rate": (
                    float(np.mean([value["ever_on_platform"] for value in episodes]))
                    if episodes
                    else 0.0
                ),
                "overall": _summarize(episodes),
                "x_bins": [
                    {"x_bin": label, **_summarize(values)}
                    for label, values in sorted(x_groups.items())
                ],
                "y_bins": [
                    {"y_bin": label, **_summarize(values)}
                    for label, values in sorted(y_groups.items())
                ],
                "cells": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
