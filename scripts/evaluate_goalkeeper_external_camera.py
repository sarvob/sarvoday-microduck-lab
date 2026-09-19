#!/usr/bin/env python3
"""Test a rendered-pixel outside camera under genuine head-camera occlusion.

The panel is visual only (zero collision). It changes what the head camera can
see but cannot touch the robot or ball. The outside-camera condition estimates
the ball position from a calibrated rendered camera image; it never reads the
simulator ball coordinates for control. Physics, locomotion, shot generation,
prediction gain, and goal scoring match challenge 011.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duck as D
from scripts import train_goalkeeper as G

SPEC = ROOT / "challenges" / "013-external-camera-goalkeeper" / "spec.json"
OUT = ROOT / "artifacts" / "013-external-camera-goalkeeper"
EXTERNAL_CAMERA = "outside_camera"


def arena_with_screen(xml: str) -> str:
    root = ET.fromstring(G.arena_xml(xml))
    asset = root.find("asset")
    world = root.find("worldbody")
    ET.SubElement(asset, "material", {
        "name": "screen_mat", "rgba": "0.08 0.12 0.16 1",
        "metallic": "0.15", "roughness": "0.48",
    })
    # A zero-collision screen blocks the low head-camera sight line. It is a
    # real rendered occluder, but it cannot change the ball or robot dynamics.
    ET.SubElement(world, "geom", {
        "name": "vision_screen", "type": "box", "pos": "-0.56 0.16 0.24",
        "size": "0.025 0.27 0.24", "material": "screen_mat",
        "contype": "0", "conaffinity": "0", "group": "2",
    })
    # Fixed camera above and behind the screen. xyaxes gives an upright view
    # looking down and toward +X, across the goal mouth.
    ET.SubElement(world, "camera", {
        "name": EXTERNAL_CAMERA, "mode": "fixed", "pos": "-1.65 0 2.20",
        "xyaxes": "0 1 0 -0.866 0 -0.5", "fovy": "60",
    })
    return ET.tostring(root, encoding="unicode")


def setup_sim(width: int = 640, height: int = 360) -> D.Microduck:
    sim = D.Microduck(width=width, height=height, render=True,
                      xml_transform=arena_with_screen)
    head = sim.model.camera("head_camera").id
    sim.model.cam_quat[head] = G.CAMERA_FORWARD_QUAT
    sim.mj.mj_forward(sim.model, sim.data)
    return sim


def visual_ball_position(sim: D.Microduck, frame: np.ndarray,
                         camera_name: str) -> np.ndarray | None:
    hit = G.orange_centroid(frame)
    if hit is None:
        return None
    u, v, _ = hit
    h, w = frame.shape[:2]
    camera_id = sim.model.camera(camera_name).id
    f = 0.5 * h / math.tan(math.radians(float(sim.model.cam_fovy[camera_id])) / 2)
    ray_camera = np.array([(u - (w - 1) / 2) / f,
                           -(v - (h - 1) / 2) / f, -1.0])
    rotation = sim.data.cam_xmat[camera_id].reshape(3, 3)
    ray_world = rotation @ ray_camera
    origin = sim.data.cam_xpos[camera_id]
    if abs(ray_world[2]) < 1e-6:
        return None
    distance = (D.BALL_RADIUS + 0.005 - origin[2]) / ray_world[2]
    if distance <= 0:
        return None
    return origin + distance * ray_world


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def compose_vertical(scene: np.ndarray, head: np.ndarray, outside: np.ndarray,
                     *, mode: str, t: float, detections: int, target_y: float,
                     goal: bool, saved: bool) -> np.ndarray:
    canvas = Image.new("RGB", (1440, 2560), "#071018")
    draw = ImageDraw.Draw(canvas)
    accent = "#49E6A1" if mode == "outside" else "#FFB24C"
    draw.text((72, 70), "WHAT IF THE DUCK CAN'T SEE?", font=font(55, True), fill="#F7FAFC")
    draw.text((72, 150), "HEAD CAMERA" if mode == "head" else "OUTSIDE CAMERA ADDED",
              font=font(39, True), fill=accent)

    main = Image.fromarray(scene).resize((1296, 729), Image.Resampling.LANCZOS)
    canvas.paste(main, (72, 245))
    draw.rounded_rectangle((72, 245, 1368, 974), 28, outline="#EAF2F5", width=5)

    head_img = Image.fromarray(head).resize((616, 347), Image.Resampling.LANCZOS)
    ext_img = Image.fromarray(outside).resize((616, 347), Image.Resampling.LANCZOS)
    canvas.paste(head_img, (72, 1075))
    canvas.paste(ext_img, (752, 1075))
    draw.rounded_rectangle((72, 1075, 688, 1422), 20, outline="#F1F4F6", width=4)
    draw.rounded_rectangle((752, 1075, 1368, 1422), 20, outline=accent, width=5)
    draw.text((94, 1442), "ROBOT VIEW", font=font(27, True), fill="#C9D5DA")
    draw.text((774, 1442), "OVERHEAD VIEW", font=font(27, True), fill=accent)

    draw.rounded_rectangle((72, 1550, 1368, 2185), 36, fill="#0D1B24", outline="#263E4B", width=3)
    draw.text((120, 1605), "CONTROL INPUT", font=font(27, True), fill="#7F98A5")
    control = "blocked head-camera pixels" if mode == "head" else "calibrated overhead-camera pixels"
    draw.text((120, 1650), control, font=font(42, True), fill="#F7FAFC")
    draw.text((120, 1760), f"detections  {detections}", font=font(38), fill="#CAD7DC")
    draw.text((120, 1830), f"target      {target_y:+.3f} m", font=font(38), fill="#CAD7DC")
    draw.text((120, 1900), f"time        {t:04.1f} s", font=font(38), fill="#CAD7DC")
    draw.text((120, 2020), "Same robot · same shot · same physics", font=font(35, True), fill=accent)

    if goal:
        label, color = "GOAL", "#FF7650"
    elif saved:
        label, color = "SAVE", "#49E6A1"
    else:
        label, color = "TRACKING", accent
    draw.rounded_rectangle((72, 2260, 1368, 2465), 40, fill="#10212B", outline=color, width=6)
    tw = draw.textbbox((0, 0), label, font=font(75, True))[2]
    draw.text(((1440 - tw) / 2, 2315), label, font=font(75, True), fill=color)
    return np.asarray(canvas)


def run_shot(case: dict, mode: str, capture_path: Path | None = None) -> dict:
    sim = setup_sim(960 if capture_path else 640, 540 if capture_path else 360)
    perception = sim.mj.Renderer(sim.model, 360, 640)
    sim.reset()
    for _ in range(25):
        sim.control_step("stand", (0, 0, 0))
    sim.place_ball(case["start_x"], case["start_y"])
    sim.data.qvel[sim.ball_d:sim.ball_d + 6] = [
        case["speed"], case["vy"], 0.0, -case["vy"] / D.BALL_RADIUS,
        case["speed"] / D.BALL_RADIUS, 0.0,
    ]
    camera_name = "head_camera" if mode == "head" else EXTERNAL_CAMERA
    observations: list[tuple[float, np.ndarray]] = []
    target_y = 0.02
    contacted = goal_scored = fell = left_zone = False
    contact_time = None
    writer = None
    if capture_path is not None:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            capture_path, fps=50, codec="libx264", quality=7, macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    try:
        for step in range(round(G.SHOT_SECONDS / D.CTRL_DT)):
            t = step * D.CTRL_DT
            if step % G.DETECT_EVERY == 0 and t <= 1.4:
                perception.update_scene(sim.data, camera=camera_name, scene_option=sim.opt)
                measured = visual_ball_position(sim, perception.render(), camera_name)
                if measured is not None:
                    observations.append((t, measured))
            if t >= G.DECIDE_AT:
                estimate = G.forecast(observations, 1.0)
                if estimate is not None:
                    target_y = estimate
            cmd = (0.0, 0.0, 0.0) if t < G.DECIDE_AT else G.waypoint_command(
                sim, target_y, min(t / G.SHOT_SECONDS, 1.0), np.empty((0, 0)))
            sim.control_step("walk" if t >= G.DECIDE_AT else "stand", cmd)
            contact_now = G.ball_contact(sim)
            if contact_now and not contacted:
                contact_time = t
            contacted = contacted or contact_now
            bx, by = sim.ball_xy()
            goal_scored = goal_scored or G.ball_is_goal(bx, by)
            y = float(sim.data.qpos[1])
            left_zone = left_zone or not (-0.02 <= y <= 0.39)
            fell = fell or float(sim.proj_gravity()[2]) > -0.5

            if writer is not None:
                sim.cam.type = sim.mj.mjtCamera.mjCAMERA_FREE
                sim.cam.lookat[:] = [-0.30, 0.15, 0.18]
                sim.cam.distance, sim.cam.azimuth, sim.cam.elevation = 1.65, 335, -20
                sim.renderer.update_scene(sim.data, camera=sim.cam, scene_option=sim.opt)
                scene = sim.renderer.render().copy()
                sim.renderer.update_scene(sim.data, camera="head_camera", scene_option=sim.opt)
                head = sim.renderer.render().copy()
                sim.renderer.update_scene(sim.data, camera=EXTERNAL_CAMERA, scene_option=sim.opt)
                outside = sim.renderer.render().copy()
                save_confirmed = bool(contacted and not goal_scored and contact_time is not None
                                      and t - contact_time >= 0.55)
                writer.append_data(compose_vertical(
                    scene, head, outside, mode=mode, t=t,
                    detections=len(observations), target_y=target_y,
                    goal=goal_scored, saved=save_confirmed,
                ))
            if bx > 0.55 or (goal_scored and t > 10.5):
                break
    finally:
        if writer is not None:
            writer.close()
        perception.close()
        sim.renderer.close()
    saved = contacted and not goal_scored
    return {
        **case,
        "mode": mode,
        "detections": len(observations),
        "target_y": round(float(target_y), 5),
        "contacted": bool(contacted),
        "goal_scored": bool(goal_scored),
        "saved": bool(saved),
        "fell": bool(fell),
        "left_goal_zone": bool(left_zone),
        "passed": bool(saved and not fell and not left_zone),
        "robot_final_y": round(float(sim.data.qpos[1]), 5),
        "ball_final_x": round(float(sim.ball_xy()[0]), 5),
        "ball_final_y": round(float(sim.ball_xy()[1]), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-seed", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC.read_text())
    seeds = spec["evaluation"]["seeds"]
    head = [run_shot(G.shot(seed), "head") for seed in seeds]
    outside = [run_shot(G.shot(seed), "outside") for seed in seeds]
    head_saves = sum(case["passed"] for case in head)
    outside_saves = sum(case["passed"] for case in outside)
    passed = (outside_saves >= spec["success"]["minimum_saves"]
              and all(not case["fell"] and not case["left_goal_zone"] for case in outside))
    result = {
        "challenge": spec["id"],
        "passed": passed,
        "success_gate": spec["success"],
        "head_camera_saves": head_saves,
        "outside_camera_saves": outside_saves,
        "head_camera_evaluation": head,
        "outside_camera_evaluation": outside,
        "disclosure": (
            "The screen and outside camera are visual-only and cannot alter physics. "
            "Both controllers use rendered camera pixels, the same prediction gain, "
            "the same frozen official locomotion policy, and strict whole-ball scoring."
        ),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"blocked head camera: {head_saves}/{len(seeds)} saves")
    print(f"outside camera: {outside_saves}/{len(seeds)} saves")
    print("PASS" if passed else "FAIL")
    if args.capture_seed is not None:
        if args.capture_seed not in seeds:
            raise SystemExit("capture seed must be one of the sealed evaluation seeds")
        for mode in ("head", "outside"):
            run_shot(G.shot(args.capture_seed), mode,
                     OUT / f"evidence-{mode}-seed-{args.capture_seed}.mp4")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
