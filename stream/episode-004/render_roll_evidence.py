#!/usr/bin/env python3
"""Render native-resolution evidence for Challenge 006's learned handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from duck import Microduck  # noqa: E402


SOURCE_FPS = 50
DELIVERY_FPS = 60
WIDTH, HEIGHT = 2560, 1440
PRE_ROLL_STEPS = 35
RECOVERY_STEPS = 100
POST_ROLL_STEPS = 35


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf" if not bold else "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def writer(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(SOURCE_FPS), "-i", "-", "-an", "-vf", f"fps={DELIVERY_FPS}",
        "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def annotate(frame: np.ndarray, seed: int, phase: str, elapsed: float,
             upright: float, displacement: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((70, 58, 2490, 236), radius=28, fill=(12, 21, 29, 218))
    draw.text((118, 88), "CONTROLLED ROLL · MOTION EVIDENCE", font=font(48, True),
              fill=(242, 246, 248, 255))
    draw.text((118, 158), f"Seed {seed}   |   {phase}   |   t = {elapsed:0.2f} s",
              font=font(31), fill=(161, 211, 221, 255))
    draw.rounded_rectangle((1640, 92, 2440, 206), radius=22, fill=(24, 39, 50, 235))
    draw.text((1690, 116), f"upright {upright:+0.2f}    drift {displacement:0.3f} m",
              font=font(30, True), fill=(255, 201, 108, 255))
    return np.asarray(image)


def render_seed(output: subprocess.Popen, seed: int, roll_steps: int,
                joint_noise: float) -> None:
    sim = Microduck(render=False)
    sim.model.vis.global_.offwidth = WIDTH
    sim.model.vis.global_.offheight = HEIGHT
    sim.renderer = sim.mj.Renderer(sim.model, height=HEIGHT, width=WIDTH)
    sim.reset()
    rng = np.random.default_rng(seed)
    sim.data.qpos[sim.qadr] += rng.normal(0.0, joint_noise, len(sim.qadr))
    sim.mj.mj_forward(sim.model, sim.data)
    start_xy = None
    try:
        total = PRE_ROLL_STEPS + roll_steps + RECOVERY_STEPS + POST_ROLL_STEPS
        for step in range(total):
            if step < PRE_ROLL_STEPS:
                phase, mode = "SETTLE", "stand"
            elif step < PRE_ROLL_STEPS + roll_steps:
                phase, mode = "ROULADE POLICY", "roll"
            else:
                phase, mode = "STAND RECOVERY", "stand"
            sim.control_step(mode, [0.0, 0.0, 0.0])
            if start_xy is None and step == PRE_ROLL_STEPS - 1:
                start_xy = sim.data.xpos[sim.trunk, :2].copy()
            origin = start_xy if start_xy is not None else sim.data.xpos[sim.trunk, :2]
            displacement = float(np.linalg.norm(sim.data.xpos[sim.trunk, :2] - origin))
            upright = -float(sim.proj_gravity()[2])
            frame = annotate(sim.frame(), seed, phase, step / SOURCE_FPS,
                             upright, displacement)
            assert output.stdin is not None
            output.stdin.write(frame.tobytes())
    finally:
        if sim.renderer is not None:
            sim.renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--output", type=Path, default=HERE / "roll-evidence-3-seeds.mp4")
    args = parser.parse_args()
    policy = json.loads((ROOT / "artifacts/006-controlled-roll/policy.json").read_text())
    spec = json.loads((ROOT / "challenges/006-controlled-roll/spec.json").read_text())
    seeds = args.seeds or policy["evaluation_seeds"]
    output = writer(args.output)
    try:
        for seed in seeds:
            render_seed(output, seed, policy["roll_steps"],
                        spec["training"]["initial_joint_noise_rad"])
    finally:
        if output.stdin is not None:
            output.stdin.close()
        if output.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing roll evidence")


if __name__ == "__main__":
    main()
