#!/usr/bin/env python3
"""Render a compact verification contact sheet for the trained Go1 landing."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_assisted_launch import contact_state, reset  # noqa: E402
from train_landing_residual import (  # noqa: E402
    control_step,
    launch_for_seed,
    make_runtime,
    residual_vector,
)


OUT = ROOT / "artifacts" / "005-duck-quadruped-jump"
FRAME_TIMES = (0.0, 0.24, 0.44, 0.64, 0.92, 1.24, 1.64, 2.04, 2.44)


def font(size: int):
    for path in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render() -> Path:
    result = json.loads((OUT / "landing-result.json").read_text(encoding="utf-8"))
    weights = np.asarray(result["weights"], dtype=float)
    sim = make_runtime()
    reset(sim)
    sim.data.qvel[0:3] = launch_for_seed(17)
    flight = residual_vector(weights[:4])
    landed = residual_vector(weights[4:])

    renderer = mujoco.Renderer(sim.model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.34, 0.0, 0.31]
    camera.distance, camera.azimuth, camera.elevation = 1.48, 145, -16
    frames = []
    landed_at = None

    for step in range(126):
        time_s = step * 0.02
        left, right, _ = contact_state(sim)
        if (left or right) and landed_at is None:
            landed_at = time_s
        if len(frames) < len(FRAME_TIMES) and time_s + 1e-9 >= FRAME_TIMES[len(frames)]:
            renderer.update_scene(sim.data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rounded_rectangle((18, 16, 375, 76), radius=12, fill=(8, 14, 24, 210))
            state = "FLIGHT" if landed_at is None else f"LANDING · feet {int(left) + int(right)}/2"
            draw.text((34, 25), f"t = {time_s:0.2f}s   {state}", font=font(25), fill="white")
            frames.append(image)
        control_step(sim, landed if landed_at is not None else flight)

    renderer.close()
    sheet = Image.new("RGB", (2880, 1740), (12, 18, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((54, 20), "MICRODUCK → UNITREE GO1 · TRAINED LANDING EVIDENCE", font=font(38), fill=(244, 248, 252))
    draw.text((54, 70), "Seed 17 · single disclosed launch velocity · zero midair force · 1.68 s verified hold", font=font(26), fill=(98, 211, 238))
    for index, image in enumerate(frames):
        x, y = (index % 3) * 960, 120 + (index // 3) * 540
        sheet.paste(image, (x, y))
    destination = OUT / "landing-contact-sheet.jpg"
    sheet.save(destination, quality=92, optimize=True, progressive=True)
    return destination


if __name__ == "__main__":
    print(render().relative_to(ROOT))
