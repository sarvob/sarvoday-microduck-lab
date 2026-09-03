#!/usr/bin/env python3
"""Render native-resolution MuJoCo footage for Episode 003."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_assisted_launch import contact_state, reset  # noqa: E402
from probe_jump_feasibility import runtime as probe_runtime  # noqa: E402
from train_landing_residual import (  # noqa: E402
    control_step,
    launch_for_seed,
    make_runtime,
    residual_vector,
)


FPS = 50


def camera_for(width: int, height: int, unassisted: bool = False, view: str = "hero") -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    portrait = height > width
    if unassisted:
        camera.lookat[:] = [0.24, 0.0, 0.18]
        camera.distance = 1.45 if portrait else 1.18
    else:
        camera.lookat[:] = [0.34, 0.0, 0.31]
        camera.distance = 1.82 if portrait else 1.48
    views = {
        "hero": (145, -16, 1.0),
        "side": (90, -12, 0.95),
        "front": (178, -14, 1.05),
        "overhead": (138, -42, 1.12),
        "close": (132, -10, 0.84),
    }
    azimuth, elevation, distance_scale = views[view]
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance *= distance_scale
    return camera


def writer(path: Path, width: int, height: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-profile:v", "high", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render_landing(seed: int, width: int, height: int, path: Path, view: str = "hero",
                   trained: bool = True) -> None:
    result = json.loads((ROOT / "artifacts/005-duck-quadruped-jump/landing-result.json").read_text())
    weights = np.asarray(result["weights"], dtype=float)
    flight = residual_vector(weights[:4]) if trained else np.zeros(14, dtype=np.float32)
    landed = residual_vector(weights[4:]) if trained else np.zeros(14, dtype=np.float32)
    sim = make_runtime()
    reset(sim)
    sim.data.qvel[0:3] = launch_for_seed(seed)
    sim.model.vis.global_.offwidth = width
    sim.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    camera = camera_for(width, height, view=view)
    output = writer(path, width, height)
    landed_at = None
    try:
        for step in range(150):
            left, right, _ = contact_state(sim)
            if (left or right) and landed_at is None:
                landed_at = step / FPS
            renderer.update_scene(sim.data, camera=camera)
            output.stdin.write(renderer.render().tobytes())
            control_step(sim, landed if landed_at is not None else flight)
    finally:
        renderer.close()
        if output.stdin:
            output.stdin.close()
        if output.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {path}")


def render_unassisted(width: int, height: int, path: Path) -> None:
    sim = probe_runtime()
    for _ in range(100):
        sim.control_step("stand", [0, 0, 0])
    sim.model.vis.global_.offwidth = width
    sim.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    camera = camera_for(width, height, unassisted=True)
    output = writer(path, width, height)
    try:
        for _ in range(200):
            renderer.update_scene(sim.data, camera=camera)
            output.stdin.write(renderer.render().tobytes())
            sim.control_step("roll", [0, 0, 0])
    finally:
        renderer.close()
        if output.stdin:
            output.stdin.close()
        if output.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {path}")


def render_orbit(width: int, height: int, path: Path) -> None:
    sim = make_runtime()
    reset(sim)
    sim.model.vis.global_.offwidth = width
    sim.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    camera = camera_for(width, height)
    output = writer(path, width, height)
    try:
        for frame in range(300):
            camera.azimuth = 112 + 66 * frame / 299
            renderer.update_scene(sim.data, camera=camera)
            output.stdin.write(renderer.render().tobytes())
            control_step(sim, np.zeros(14, dtype=np.float32))
    finally:
        renderer.close()
        if output.stdin:
            output.stdin.close()
        if output.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {path}")


def render_educational_extras() -> None:
    width, height = 2560, 1440
    render_orbit(width, height, HERE / "raw-scene-orbit-landscape.mp4")
    render_landing(17, width, height, HERE / "raw-untrained-baseline-landscape.mp4",
                   view="side", trained=False)
    render_landing(17, width, height, HERE / "raw-seed-17-side-landscape.mp4", view="side")
    render_landing(71, width, height, HERE / "raw-seed-71-overhead-landscape.mp4", view="overhead")
    render_landing(173, width, height, HERE / "raw-seed-173-front-landscape.mp4", view="front")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orientation", choices=("landscape", "portrait", "all"), default="all")
    parser.add_argument("--educational-extras", action="store_true")
    args = parser.parse_args()
    if args.educational_extras:
        render_educational_extras()
        return
    layouts = []
    if args.orientation in ("landscape", "all"):
        layouts.append(("landscape", 2560, 1440))
    if args.orientation in ("portrait", "all"):
        layouts.append(("portrait", 1440, 2560))
    for label, width, height in layouts:
        render_unassisted(width, height, HERE / f"raw-unassisted-{label}.mp4")
        for seed in (17, 71, 173):
            render_landing(seed, width, height, HERE / f"raw-seed-{seed}-{label}.mp4")


if __name__ == "__main__":
    main()
