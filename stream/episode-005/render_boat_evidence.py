#!/usr/bin/env python3
"""Render Challenge 012's verified harbor passes and chop exits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duck import CTRL_DT, Microduck  # noqa: E402
from evaluate_boat_balance import (DECK_HALF, add_boat_deck, contact_flags,
                                   motion, set_deck_state)  # noqa: E402
from train_boat_recenter import recenter_command  # noqa: E402
from train_boat_residual import control_step, residual_vector  # noqa: E402

W, H, SOURCE_FPS, DELIVERY_FPS = 2560, 1440, 50, 60


def add_cinematic_boat(xml: str) -> str:
    """Add visual-only hull, water, wave crests, and deck detail.

    Collision and controller dynamics still come from ``add_boat_deck``. The
    water surface is explicitly illustrative; the prescribed deck trajectory
    remains the experiment's physical disturbance input.
    """
    root = ET.fromstring(add_boat_deck(xml))
    asset = root.find("asset")
    world = root.find("worldbody")
    boat = root.find("./worldbody/body[@name='boat_deck']")
    assert asset is not None and world is not None and boat is not None
    ET.SubElement(asset, "texture", {
        "name": "watertex", "type": "2d", "builtin": "checker",
        "width": "512", "height": "512", "rgb1": "0.012 0.105 0.15",
        "rgb2": "0.018 0.135 0.18",
    })
    ET.SubElement(asset, "material", {
        "name": "watermat", "texture": "watertex", "texrepeat": "9 5",
        "reflectance": "0.22", "shininess": "0.65", "specular": "0.45",
    })
    ET.SubElement(world, "geom", {
        "name": "cinematic_water", "type": "plane", "size": "30 14 0.05",
        "pos": "5 0 -0.145", "material": "watermat", "group": "2",
        "contype": "0", "conaffinity": "0",
    })
    for i in range(34):
        x = -7.0 + i * 0.75
        ET.SubElement(world, "geom", {
            "name": f"wave_crest_{i}", "type": "capsule", "size": "0.014",
            "fromto": f"{x} -9 -0.122 {x} 9 -0.122", "rgba": "0.20 0.61 0.71 0.16",
            "group": "2", "contype": "0", "conaffinity": "0",
        })
    # Twin pontoons and bow caps make the moving body read as a boat even in
    # silhouette. Every added geom is visual-only.
    for side in (-0.29, 0.29):
        ET.SubElement(boat, "geom", {
            "type": "capsule", "size": "0.10", "fromto": f"-0.64 {side} -0.12 0.63 {side} -0.12",
            "rgba": "0.025 0.075 0.105 1", "group": "2", "mass": "0", "contype": "0", "conaffinity": "0",
        })
        ET.SubElement(boat, "geom", {
            "type": "ellipsoid", "size": "0.20 0.12 0.115", "pos": f"0.67 {side} -0.12",
            "rgba": "0.035 0.11 0.15 1", "group": "2", "mass": "0", "contype": "0", "conaffinity": "0",
        })
        ET.SubElement(boat, "geom", {
            "type": "capsule", "size": "0.014", "fromto": f"-0.70 {side} 0.12 0.70 {side} 0.12",
            "rgba": "0.74 0.82 0.84 1", "group": "2", "mass": "0", "contype": "0", "conaffinity": "0",
        })
        for x in (-0.62, 0.0, 0.62):
            ET.SubElement(boat, "geom", {
                "type": "capsule", "size": "0.012", "fromto": f"{x} {side} 0.03 {x} {side} 0.13",
                "rgba": "0.74 0.82 0.84 1", "group": "2", "mass": "0", "contype": "0", "conaffinity": "0",
            })
    for x in (-0.58, -0.30, -0.02, 0.26, 0.54):
        ET.SubElement(boat, "geom", {
            "type": "box", "size": "0.12 0.335 0.008", "pos": f"{x} 0 0.064",
            "rgba": "0.31 0.34 0.34 1", "group": "2", "mass": "0", "contype": "0", "conaffinity": "0",
        })
    # A restrained cyan boot stripe and wake lines improve separation from water.
    for side in (-0.34, 0.34):
        ET.SubElement(boat, "geom", {
            "type": "capsule", "size": "0.018", "fromto": f"-0.66 {side} -0.07 0.66 {side} -0.07",
            "rgba": "0.12 0.58 0.69 1", "group": "2", "mass": "0", "contype": "0", "conaffinity": "0",
        })
    return ET.tostring(root, encoding="unicode")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def writer(path: Path) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(SOURCE_FPS), "-i", "-", "-an", "-vf", f"fps={DELIVERY_FPS}",
        "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
        "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def annotate(frame: np.ndarray, pov: np.ndarray, profile: str, seed: int,
             t: float, speed: float, roll: float, pitch: float,
             upright: float, drift: float, state: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    inset = Image.fromarray(pov).resize((880, 495), Image.Resampling.LANCZOS)
    mask = Image.new("L", inset.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 879, 494), radius=24, fill=255)
    draw.rounded_rectangle((1582, 40, 2512, 579), radius=30, fill=(7, 13, 18, 115))
    image.paste(inset, (1600, 58), mask)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((1600, 58, 2480, 553), radius=25,
                           outline=(235, 241, 244, 230), width=5)
    draw.rounded_rectangle((1630, 84, 1848, 138), radius=15, fill=(8, 15, 21, 198))
    draw.text((1660, 94), "ROBOT POV", font=font(25, True), fill=(240, 246, 247, 255))
    draw.rounded_rectangle((68, 54, 285, 108), radius=15, fill=(10, 18, 24, 190))
    draw.text((98, 65), "BOAT VIEW", font=font(24, True), fill=(240, 246, 247, 255))
    color = (100, 214, 160, 255) if state == "IN BOUNDS" else (242, 104, 92, 255)
    draw.rounded_rectangle((68, 1170, 1260, 1365), radius=28, fill=(10, 18, 24, 215))
    draw.text((112, 1205), f"{profile.upper()} · SEED {seed:03d} · {t:05.2f} s",
              font=font(39, True), fill=(244, 248, 249, 255))
    draw.text((112, 1280), f"SPEED {speed:0.2f} m/s   ROLL {np.degrees(roll):+0.1f}°   PITCH {np.degrees(pitch):+0.1f}°",
              font=font(27), fill=(166, 215, 223, 255))
    draw.rounded_rectangle((1640, 1215, 2492, 1365), radius=26, fill=(10, 18, 24, 215))
    draw.text((1682, 1245), f"UPRIGHT {upright:+0.2f}   DRIFT {drift:0.3f} m",
              font=font(29, True), fill=(255, 202, 109, 255))
    draw.text((1682, 1304), state, font=font(27, True), fill=color)
    return np.asarray(image)


def render_seed(process: subprocess.Popen, profile: dict, seed: int,
                residual_weights: np.ndarray, gains: np.ndarray, spec: dict) -> None:
    sim = Microduck(render=False, xml_transform=add_cinematic_boat)
    sim.model.geom("ball_geom").rgba[3] = 0
    # Keep the physical safety floor active but visually reveal the hull and
    # lower illustrative water surface beneath it.
    floor_id = sim.model.geom("floor").id
    sim.model.geom_matid[floor_id] = -1
    sim.model.geom_rgba[floor_id] = [0, 0, 0, 0]
    sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight = W, H
    sim.model.vis.map.znear = 0.08
    sim.renderer = sim.mj.Renderer(sim.model, height=H, width=W)
    sim.cam.trackbodyid = sim.model.body("boat_deck").id
    sim.cam.distance, sim.cam.elevation, sim.cam_offset = 2.15, -24.0, 145.0
    pov_renderer = sim.mj.Renderer(sim.model, height=495, width=880)
    pov_camera = sim.mj.MjvCamera()
    pov_camera.type = sim.mj.mjtCamera.mjCAMERA_FIXED
    pov_camera.fixedcamid = sim.mj.mj_name2id(sim.model, sim.mj.mjtObj.mjOBJ_CAMERA, "head_camera")
    sim.model.cam_quat[pov_camera.fixedcamid] = [0.707107, 0, 0, -0.707107]
    sim.model.cam_pos[pov_camera.fixedcamid, 2] = -0.095
    sim.reset()
    rng = np.random.default_rng(seed)
    sim.data.qpos[sim.qadr] += rng.normal(0, spec["training"]["initial_joint_noise_rad"], len(sim.qadr))
    sim.mj.mj_forward(sim.model, sim.data)
    boat_q = int(sim.model.joint("boat_freejoint").qposadr[0])
    boat_d = int(sim.model.joint("boat_freejoint").dofadr[0])
    feet = {sim.model.geom("left_foot_collision").id, sim.model.geom("right_foot_collision").id}
    deck_geom, floor_geom = sim.model.geom("boat_deck_geom").id, sim.model.geom("floor").id
    for _ in range(50):
        set_deck_state(sim, boat_q, boat_d, 0, 0, 0, 0, 0, 0)
        sim.control_step("stand", [0, 0, 0])
    start_relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
    deck_x = 0.0
    previous_roll = previous_pitch = previous_speed = previous_apparent_pitch = 0.0
    try:
        for step in range(round(spec["environment"]["evaluation_duration_s"] / CTRL_DT)):
            t = step * CTRL_DT
            speed, roll, pitch = motion(profile, t, spec["environment"]["motion_ramp_s"])
            deck_x += speed * CTRL_DT
            roll_rate, pitch_rate = (roll - previous_roll) / CTRL_DT, (pitch - previous_pitch) / CTRL_DT
            acceleration = (speed - previous_speed) / CTRL_DT
            apparent_pitch = pitch + np.arctan2(acceleration, 9.81)
            apparent_pitch_rate = (apparent_pitch - previous_apparent_pitch) / CTRL_DT
            set_deck_state(sim, boat_q, boat_d, deck_x, speed, roll, pitch, roll_rate, pitch_rate)
            for index in range(34):
                gid = sim.model.geom(f"wave_crest_{index}").id
                phase = 0.72 * index - 2.1 * t
                sim.model.geom_pos[gid, 2] = -0.122 + 0.016 * np.sin(phase)
            rel_pos = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2] - start_relative
            rel_vel = sim.data.qvel[:2] - np.array([speed, 0.0])
            residual = residual_vector(residual_weights, roll, apparent_pitch, roll_rate,
                                       apparent_pitch_rate, rel_pos, rel_vel,
                                       spec["training"]["maximum_joint_residual_rad"])
            command = recenter_command(gains, rel_pos, rel_vel)
            control_step(sim, residual, mode="walk", command3=command)
            previous_roll, previous_pitch = roll, pitch
            previous_speed, previous_apparent_pitch = speed, apparent_pitch
            relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
            drift = float(np.linalg.norm(relative - start_relative))
            upright = -float(sim.proj_gravity()[2])
            inside = abs(float(relative[0])) <= DECK_HALF[0] - 0.12 and abs(float(relative[1])) <= DECK_HALF[1] - 0.12
            deck_contact, floor_contact = contact_flags(sim, deck_geom, floor_geom, feet)
            state = "IN BOUNDS" if inside and not floor_contact else "DECK EXIT"
            pov_renderer.update_scene(sim.data, camera=pov_camera, scene_option=sim.opt)
            image = annotate(sim.frame(), pov_renderer.render(), profile["name"], seed,
                             t + CTRL_DT, speed, roll, pitch, upright, drift, state)
            assert process.stdin is not None
            process.stdin.write(image.tobytes())
            if not inside or floor_contact:
                break
    finally:
        sim.renderer.close()
        pov_renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["harbor", "chop"])
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads((ROOT / "challenges/012-variable-speed-boat-balance/spec.json").read_text())
    result = json.loads((ROOT / "artifacts/012-variable-speed-boat-balance/joint-walk-refinement-result.json").read_text())
    profile = next(row for row in spec["environment"]["profiles"] if row["name"] == args.profile)
    process = writer(args.output or HERE / f"{args.profile}-3-seeds.mp4")
    try:
        for seed in args.seeds or spec["training"]["seeds"]:
            render_seed(process, profile, seed, np.asarray(result["residual_weights"]),
                        np.asarray(result["recenter_gains"]), spec)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg evidence render failed")


if __name__ == "__main__":
    main()
