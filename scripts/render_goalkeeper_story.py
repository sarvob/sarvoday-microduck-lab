#!/usr/bin/env python3
"""Render audience-first Microduck goalkeeper evidence for episode 006 v2.

The control and scoring match train_goalkeeper.py. This renderer changes only
camera language and presentation: action owns the main canvas, while measured
values stay in a narrow right-side lab rail.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "011-vision-guided-goalkeeper" / "story-v3"

spec = importlib.util.spec_from_file_location("goalkeeper", ROOT / "scripts" / "train_goalkeeper.py")
g = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(g)


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def frame_layout(scene: np.ndarray, pov: np.ndarray, *, mode: str, shot_number: int,
                 t: float, speed: float, measured: np.ndarray | None,
                 target: float, gain: float, blocked: bool, crossed: bool) -> np.ndarray:
    canvas = Image.new("RGB", (2560, 1440), "#060A0F")
    action = Image.fromarray(scene).resize((2140, 1204), Image.Resampling.LANCZOS)
    canvas.paste(action, (0, 118))
    draw = ImageDraw.Draw(canvas)
    accent = "#43E6A0" if mode == "predictor" else "#FFB24C"
    draw.rectangle((0, 0, 2560, 118), fill="#060A0F")
    draw.text((54, 28), f"SHOT {shot_number}", font=font(42, True), fill="#F7FAFC")
    label = "PREDICT THE CROSSING" if mode == "predictor" else "CHASE THE BALL"
    draw.text((330, 34), label, font=font(30, True), fill=accent)
    draw.rounded_rectangle((1770, 25, 2090, 92), 26, fill="#10202A")
    draw.text((1810, 41), f"{min(t / 10.5, 1) * 100:03.0f}%", font=font(28, True), fill="#DDE7EC")

    rail_x = 2140
    draw.rectangle((rail_x, 0, 2560, 1440), fill="#0A141D")
    draw.text((2180, 36), "MICRODUCK LAB", font=font(23, True), fill="#8AA1AF")
    pov_img = Image.fromarray(pov).resize((340, 191), Image.Resampling.LANCZOS)
    canvas.paste(pov_img, (2180, 105))
    draw.text((2190, 310), "WHAT THE DUCK SEES", font=font(18, True), fill="#A9BAC4")

    rows = [
        ("SHOT SPEED", f"{speed:.2f} m/s"),
        ("CAMERA X", "—" if measured is None else f"{measured[0]:+.2f} m"),
        ("CAMERA Y", "—" if measured is None else f"{measured[1]:+.2f} m"),
        ("AIM POINT", f"{target:+.2f} m"),
        ("PREDICTION", f"{gain:.2f}×"),
    ]
    y = 390
    for key, value in rows:
        draw.text((2180, y), key, font=font(18, True), fill="#6F8794")
        draw.text((2180, y + 32), value, font=font(34, True), fill="#F4F7F8")
        draw.line((2180, y + 88, 2520, y + 88), fill="#1C3441", width=2)
        y += 116

    if 0.9 < t < 2.5:
        draw.rounded_rectangle((110, 1080, 1180, 1248), 24, fill="#071018", outline="#5A7482", width=3)
        draw.text((152, 1112), "IT MUST TURN TO MOVE SIDEWAYS", font=font(35, True), fill="#FFFFFF")
        draw.text((152, 1165), "The pivot is a real locomotion constraint", font=font(25), fill="#C4D0D6")

    if blocked:
        draw.rounded_rectangle((2190, 1130, 2510, 1250), 28, fill="#123B2E", outline="#43E6A0", width=4)
        draw.text((2250, 1160), "SAVE", font=font(44, True), fill="#43E6A0")
    elif crossed:
        draw.rounded_rectangle((2190, 1130, 2510, 1250), 28, fill="#442716", outline="#FFB24C", width=4)
        draw.text((2250, 1160), "MISS", font=font(44, True), fill="#FFB24C")

    draw.text((2180, 1358), "Values shown here do not alter physics", font=font(16), fill="#67808D")
    return np.asarray(canvas)


def render(seed: int, gain: float, mode: str, shot_number: int, name: str) -> None:
    case = g.shot(seed)
    weights = np.asarray(json.loads(g.WAYPOINT_WEIGHTS.read_text())["weights"]).reshape(3, 5)
    sim = g.setup_sim(1280, 720, True)
    perception = sim.mj.Renderer(sim.model, 360, 640)
    sim.reset()
    for _ in range(25):
        sim.control_step("stand", (0, 0, 0))
    sim.place_ball(case["start_x"], case["start_y"])
    sim.data.qvel[sim.ball_d:sim.ball_d + 6] = [
        case["speed"], case["vy"], 0.0, -case["vy"] / g.D.BALL_RADIUS,
        case["speed"] / g.D.BALL_RADIUS, 0.0,
    ]
    observations: list[tuple[float, np.ndarray]] = []
    tracking_observations: list[tuple[float, np.ndarray]] = []
    measured = None
    target = 0.02
    blocked = crossed = False
    path = OUT / name
    with imageio.get_writer(path, fps=25, codec="libx264", quality=7,
                            macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as writer:
        for step in range(round(10.5 / g.D.CTRL_DT)):
            t = step * g.D.CTRL_DT
            if step % g.DETECT_EVERY == 0 and not blocked and not crossed:
                perception.update_scene(sim.data, camera="head_camera", scene_option=sim.opt)
                pframe = perception.render()
                measured = g.visual_ball_position(sim, pframe)
                if measured is not None:
                    tracking_observations.append((t, measured.copy()))
                    if t <= 1.4:
                        observations.append((t, measured.copy()))
            g.track_ball_with_head(sim, tracking_observations, t)
            if t >= g.DECIDE_AT:
                estimate = g.forecast(observations, gain)
                if estimate is not None:
                    target = estimate
            cmd = (0.0, 0.0, 0.0) if t < g.DECIDE_AT else g.waypoint_command(
                sim, target, min(t / g.SHOT_SECONDS, 1.0), weights)
            sim.control_step("walk" if t >= g.DECIDE_AT else "stand", cmd)
            blocked = blocked or g.ball_contact(sim)
            bx, _ = sim.ball_xy()
            crossed = crossed or (bx >= g.GOAL_X and not blocked)
            if step % 2 == 0:
                # Start behind the shot, then move to a three-quarter view as
                # Microduck pivots. This keeps its facing direction legible.
                sim.cam.type = sim.mj.mjtCamera.mjCAMERA_FREE
                sim.cam.lookat[:] = [-0.34, 0.17, 0.15]
                sim.cam.distance = 1.74 if t < 2.0 else 1.50
                sim.cam.azimuth = 0 if t < 2.0 else 315
                sim.cam.elevation = -20
                sim.renderer.update_scene(sim.data, camera=sim.cam, scene_option=sim.opt)
                scene = sim.renderer.render().copy()
                sim.renderer.update_scene(sim.data, camera="head_camera", scene_option=sim.opt)
                pov = sim.renderer.render().copy()
                writer.append_data(frame_layout(
                    scene, pov, mode=mode, shot_number=shot_number, t=t,
                    speed=case["speed"], measured=measured, target=target,
                    gain=gain, blocked=blocked, crossed=crossed))
            if bx > 0.52 or (crossed and t > 9.6):
                break
    perception.close()
    sim.renderer.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render(229, 0.0, "baseline", 1, "baseline-hard-miss.mp4")
    render(101, 0.0, "baseline", 7, "baseline-center-save.mp4")
    render(197, 0.0, "baseline", 14, "baseline-wide-save.mp4")
    render(229, 1.0, "predictor", 1, "predictor-hard-save.mp4")
    render(101, 1.0, "predictor", 7, "predictor-center-save.mp4")
    render(197, 1.0, "predictor", 14, "predictor-wide-save.mp4")


if __name__ == "__main__":
    main()
