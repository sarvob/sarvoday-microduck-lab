#!/usr/bin/env python3
"""Render every Challenge 006 evaluation as unique simulator footage."""

from __future__ import annotations

import json
from pathlib import Path

from render_roll_evidence import render_seed, writer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "search-evaluation-montage.mp4"
SETTLE_STEPS = 50
# The montage shows the roll plus the first 1.2 seconds of recovery. The full
# 2.0-second recovery remains represented by the machine-readable results and
# the longer selected/boundary evidence clips.
VISIBLE_RECOVERY_STEPS = 60


def main() -> None:
    result = json.loads((ROOT / "artifacts/006-controlled-roll/result.json").read_text())
    spec = json.loads((ROOT / "challenges/006-controlled-roll/spec.json").read_text())
    joint_noise = spec["training"]["initial_joint_noise_rad"]
    process = writer(OUTPUT)
    try:
        for candidate in result["candidates"]:
            roll_steps = candidate["roll_steps"]
            duration = roll_steps / 50.0
            passing = candidate["passing_seeds"]
            label = f"Search replay · {duration:0.2f} s handoff · {passing}/3 pass"
            for evaluation in candidate["evaluations"]:
                render_seed(
                    process,
                    evaluation["seed"],
                    roll_steps,
                    joint_noise,
                    label,
                    SETTLE_STEPS,
                    0,
                    VISIBLE_RECOVERY_STEPS,
                    True,
                )
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("search montage render failed")


if __name__ == "__main__":
    main()
