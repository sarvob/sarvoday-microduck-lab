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
SETTLE_STEPS = 50
RECOVERY_STEPS = 100
POST_ROLL_STEPS = 0


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


def annotate(frame: np.ndarray, pov: np.ndarray, run_label: str, seed: int,
             phase: str, elapsed: float,
             upright: float, displacement: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")

    # A restrained two-camera composition: overview fills the canvas while the
    # physical head camera remains readable as a clean picture-in-picture view.
    inset_box = (1600, 58, 2494, 561)
    draw.rounded_rectangle((1582, 40, 2512, 579), radius=30, fill=(7, 13, 18, 105))
    inset = Image.fromarray(pov).resize((880, 495), Image.Resampling.LANCZOS)
    mask = Image.new("L", inset.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 879, 494), radius=24, fill=255)
    image.paste(inset, inset_box[:2], mask)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(inset_box, radius=25, outline=(235, 241, 244, 230), width=5)
    draw.rounded_rectangle((1630, 84, 1848, 138), radius=15, fill=(8, 15, 21, 198))
    draw.text((1660, 94), "ROBOT POV", font=font(25, True), fill=(240, 246, 247, 255))

    draw.rounded_rectangle((68, 1190, 1188, 1365), radius=28, fill=(10, 18, 24, 210))
    draw.text((112, 1220), run_label.upper(), font=font(43, True),
              fill=(244, 248, 249, 255))
    draw.text((112, 1287), f"SEED {seed:03d}  ·  {phase}  ·  {elapsed:0.2f} s",
              font=font(29), fill=(166, 215, 223, 255))

    draw.rounded_rectangle((1680, 1240, 2492, 1365), radius=26, fill=(10, 18, 24, 210))
    draw.text((1722, 1279), f"UPRIGHT  {upright:+0.2f}     DRIFT  {displacement:0.3f} m",
              font=font(29, True), fill=(255, 202, 109, 255))
    draw.rounded_rectangle((68, 54, 274, 108), radius=15, fill=(10, 18, 24, 190))
    draw.text((98, 65), "OVERVIEW", font=font(24, True), fill=(240, 246, 247, 255))
    return np.asarray(image)


def render_seed(output: subprocess.Popen, seed: int, roll_steps: int,
                joint_noise: float, run_label: str, settle_steps: int,
                post_roll_steps: int) -> None:
    sim = Microduck(render=False)
    sim.model.geom("ball_geom").rgba[3] = 0.0
    sim.model.vis.global_.offwidth = WIDTH
    sim.model.vis.global_.offheight = HEIGHT
    # The authored head camera sits inside the cosmetic shell. A slightly
    # farther near plane clips that shell while leaving the overview unchanged.
    sim.model.vis.map.znear = 0.12
    sim.renderer = sim.mj.Renderer(sim.model, height=HEIGHT, width=WIDTH)
    sim.cam.distance = 1.55
    sim.cam.elevation = -48.0
    pov_width, pov_height = 880, 495
    pov_renderer = sim.mj.Renderer(sim.model, height=pov_height, width=pov_width)
    pov_camera = sim.mj.MjvCamera()
    pov_camera.type = sim.mj.mjtCamera.mjCAMERA_FIXED
    pov_camera.fixedcamid = sim.mj.mj_name2id(
        sim.model, sim.mj.mjtObj.mjOBJ_CAMERA, "head_camera")
    # The source render-camera faces inward. Point its -Z optical axis through
    # the beak and move it just beyond the cosmetic shell.
    sim.model.cam_quat[pov_camera.fixedcamid] = [0.707107, 0.0, 0.0, -0.707107]
    sim.model.cam_pos[pov_camera.fixedcamid, 2] = -0.095
    sim.reset()
    rng = np.random.default_rng(seed)
    sim.data.qpos[sim.qadr] += rng.normal(0.0, joint_noise, len(sim.qadr))
    sim.mj.mj_forward(sim.model, sim.data)
    start_xy = None
    try:
        total = settle_steps + roll_steps + RECOVERY_STEPS + post_roll_steps
        for step in range(total):
            if step < settle_steps:
                phase, mode = "SETTLE", "stand"
            elif step < settle_steps + roll_steps:
                phase, mode = "ROULADE POLICY", "roll"
            else:
                phase, mode = "STAND RECOVERY", "stand"
            sim.control_step(mode, [0.0, 0.0, 0.0])
            if start_xy is None and step == settle_steps - 1:
                start_xy = sim.data.xpos[sim.trunk, :2].copy()
            origin = start_xy if start_xy is not None else sim.data.xpos[sim.trunk, :2]
            displacement = float(np.linalg.norm(sim.data.xpos[sim.trunk, :2] - origin))
            upright = -float(sim.proj_gravity()[2])
            pov_renderer.update_scene(sim.data, camera=pov_camera, scene_option=sim.opt)
            frame = annotate(sim.frame(), pov_renderer.render(), run_label, seed,
                             phase, step / SOURCE_FPS, upright, displacement)
            assert output.stdin is not None
            output.stdin.write(frame.tobytes())
    finally:
        if sim.renderer is not None:
            sim.renderer.close()
        pov_renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--roll-steps", type=int)
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--post-roll-steps", type=int, default=POST_ROLL_STEPS)
    parser.add_argument("--run-label", default="Controlled roll")
    parser.add_argument("--output", type=Path, default=HERE / "roll-evidence-3-seeds.mp4")
    args = parser.parse_args()
    policy = json.loads((ROOT / "artifacts/006-controlled-roll/policy.json").read_text())
    spec = json.loads((ROOT / "challenges/006-controlled-roll/spec.json").read_text())
    seeds = args.seeds or policy["evaluation_seeds"]
    roll_steps = args.roll_steps if args.roll_steps is not None else policy["roll_steps"]
    output = writer(args.output)
    try:
        for seed in seeds:
            render_seed(output, seed, roll_steps,
                        spec["training"]["initial_joint_noise_rad"], args.run_label,
                        args.settle_steps, args.post_roll_steps)
    finally:
        if output.stdin is not None:
            output.stdin.close()
        if output.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing roll evidence")


if __name__ == "__main__":
    main()
