#!/usr/bin/env python3
"""Render a successful challenge policy as a real-time MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duck as D
import lesson as L


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("challenge_id")
    args = parser.parse_args()

    artifact_dir = ROOT / "artifacts" / args.challenge_id
    policy = json.loads((artifact_dir / "policy.json").read_text(encoding="utf-8"))
    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    if not result.get("passed"):
        raise SystemExit("Refusing to render a challenge that did not pass")

    weights = np.asarray(policy["weights"], dtype=float)
    spec = result["lesson"]
    simulator = D.Microduck(width=640, height=360, render=True)
    rollout = L.rollout(
        simulator,
        weights,
        spec,
        trace=True,
        on_frame=(2, lambda sim: sim.frame()),
    )
    if rollout["fell"] or not rollout["all_reached"]:
        raise SystemExit("Render-time verification failed")

    destination = artifact_dir / "demonstration.mp4"
    imageio.mimsave(destination, rollout["frames"], fps=25, codec="libx264",
                    quality=7, macro_block_size=1)
    print(f"Rendered: {destination.relative_to(ROOT)}")
    print(rollout["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
