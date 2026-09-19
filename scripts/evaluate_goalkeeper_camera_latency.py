#!/usr/bin/env python3
"""Measure outside-camera latency tolerance without hidden state.

Every observation is reconstructed from rendered overhead-camera pixels at its
capture time, queued, and withheld from the controller until capture time plus
the declared delay. The original capture timestamp travels with the packet.
Physics, shot generation, prediction gain, locomotion, and scoring are inherited
unchanged from challenges 011 and 013.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duck as D
from scripts import evaluate_goalkeeper_external_camera as E
from scripts import train_goalkeeper as G

SPEC = ROOT / "challenges" / "014-external-camera-latency" / "spec.json"
OUT = ROOT / "artifacts" / "014-external-camera-latency"


def deliver_packets(queue: list[tuple[float, float, np.ndarray]], now: float,
                    delivered: list[tuple[float, np.ndarray]]) -> int:
    """Deliver due observations while retaining their original timestamps."""
    count = 0
    while queue and queue[0][0] <= now + 1e-9:
        _, captured_at, measured = queue.pop(0)
        delivered.append((captured_at, measured))
        count += 1
    return count


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def compose_vertical(scene: np.ndarray, outside: np.ndarray, *, latency: float,
                     t: float, captured: int, delivered: int, queued: int,
                     target_y: float, goal: bool, saved: bool) -> np.ndarray:
    canvas = Image.new("RGB", (1440, 2560), "#071018")
    draw = ImageDraw.Draw(canvas)
    accent = "#49E6A1" if latency <= 2.0 else "#FFB24C"
    draw.text((72, 68), "CAN THE DUCK BEAT CAMERA LAG?", font=font(51, True), fill="#F7FAFC")
    draw.text((72, 145), f"{latency:.1f} SECOND DELIVERY DELAY", font=font(38, True), fill=accent)

    main = Image.fromarray(scene).resize((1296, 729), Image.Resampling.LANCZOS)
    canvas.paste(main, (72, 235))
    draw.rounded_rectangle((72, 235, 1368, 964), 28, outline="#EAF2F5", width=5)

    ext = Image.fromarray(outside).resize((1000, 562), Image.Resampling.LANCZOS)
    canvas.paste(ext, (220, 1055))
    draw.rounded_rectangle((220, 1055, 1220, 1617), 24, outline=accent, width=5)
    draw.text((242, 1640), "OVERHEAD CAMERA · RENDERED PIXELS", font=font(30, True), fill=accent)

    draw.rounded_rectangle((72, 1745, 1368, 2200), 34, fill="#0D1B24", outline="#263E4B", width=3)
    draw.text((118, 1790), f"frames captured   {captured}", font=font(35), fill="#CAD7DC")
    draw.text((118, 1855), f"frames delivered  {delivered}", font=font(35), fill="#CAD7DC")
    draw.text((118, 1920), f"waiting in queue  {queued}", font=font(35), fill="#CAD7DC")
    draw.text((118, 1985), f"target             {target_y:+.3f} m", font=font(35), fill="#CAD7DC")
    draw.text((118, 2070), "Same robot · same shot · delayed pixels", font=font(34, True), fill=accent)

    if goal:
        label, color = "TOO LATE · GOAL", "#FF7650"
    elif saved:
        label, color = "SAVE", "#49E6A1"
    else:
        label, color = f"WAITING · {t:04.1f} s", accent
    draw.rounded_rectangle((72, 2280, 1368, 2480), 40, fill="#10212B", outline=color, width=6)
    box = draw.textbbox((0, 0), label, font=font(68, True))
    draw.text(((1440 - (box[2] - box[0])) / 2, 2335), label, font=font(68, True), fill=color)
    return np.asarray(canvas)


def run_shot(case: dict, latency_seconds: float,
             capture_path: Path | None = None) -> dict:
    sim = E.setup_sim(960 if capture_path else 640, 540 if capture_path else 360)
    perception = sim.mj.Renderer(sim.model, 360, 640)
    sim.reset()
    for _ in range(25):
        sim.control_step("stand", (0, 0, 0))
    sim.place_ball(case["start_x"], case["start_y"])
    sim.data.qvel[sim.ball_d:sim.ball_d + 6] = [
        case["speed"], case["vy"], 0.0, -case["vy"] / D.BALL_RADIUS,
        case["speed"] / D.BALL_RADIUS, 0.0,
    ]
    queue: list[tuple[float, float, np.ndarray]] = []
    observations: list[tuple[float, np.ndarray]] = []
    captured = delivered = 0
    first_delivery_time = None
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
            outside_frame = None
            if step % G.DETECT_EVERY == 0 and t <= 1.4:
                perception.update_scene(sim.data, camera=E.EXTERNAL_CAMERA, scene_option=sim.opt)
                outside_frame = perception.render().copy()
                measured = E.visual_ball_position(sim, outside_frame, E.EXTERNAL_CAMERA)
                if measured is not None:
                    queue.append((t + latency_seconds, t, measured))
                    captured += 1
            newly_delivered = deliver_packets(queue, t, observations)
            if newly_delivered:
                delivered += newly_delivered
                if first_delivery_time is None:
                    first_delivery_time = t
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
                if outside_frame is None:
                    sim.renderer.update_scene(sim.data, camera=E.EXTERNAL_CAMERA, scene_option=sim.opt)
                    outside_frame = sim.renderer.render().copy()
                save_confirmed = bool(contacted and not goal_scored and contact_time is not None
                                      and t - contact_time >= 0.55)
                writer.append_data(compose_vertical(
                    scene, outside_frame, latency=latency_seconds, t=t,
                    captured=captured, delivered=delivered, queued=len(queue),
                    target_y=target_y, goal=goal_scored, saved=save_confirmed,
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
        "latency_seconds": latency_seconds,
        "captured_observations": captured,
        "delivered_observations": delivered,
        "first_delivery_time": None if first_delivery_time is None else round(first_delivery_time, 3),
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
    parser.add_argument("--capture-latency", type=float)
    parser.add_argument("--capture-seed", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC.read_text())
    seeds = spec["evaluation"]["seeds"]
    latencies = spec["evaluation"]["latency_seconds"]
    conditions = []
    for latency in latencies:
        cases = [run_shot(G.shot(seed), float(latency)) for seed in seeds]
        saves = sum(case["passed"] for case in cases)
        stable = all(not case["fell"] and not case["left_goal_zone"] for case in cases)
        meets_gate = saves >= spec["tolerance_gate"]["minimum_saves"] and stable
        conditions.append({
            "latency_seconds": latency,
            "saves": saves,
            "shots": len(cases),
            "stable": stable,
            "meets_tolerance_gate": meets_gate,
            "cases": cases,
        })
        print(f"latency {latency:.1f} s: {saves}/{len(cases)} saves"
              f" · {'GATE PASS' if meets_gate else 'GATE FAIL'}")
    passing = [item["latency_seconds"] for item in conditions if item["meets_tolerance_gate"]]
    first_failure = next((item["latency_seconds"] for item in conditions
                          if not item["meets_tolerance_gate"]), None)
    result = {
        "challenge": spec["id"],
        "tolerance_gate": spec["tolerance_gate"],
        "conditions": conditions,
        "maximum_tested_passing_latency_seconds": max(passing) if passing else None,
        "first_tested_failing_latency_seconds": first_failure,
        "disclosure": (
            "Every ball observation came from rendered overhead-camera pixels. "
            "Packets retained their capture timestamps but were hidden from the controller "
            "until the declared delivery time. No frame drops were added in this experiment."
        ),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    if args.capture_latency is not None or args.capture_seed is not None:
        if args.capture_latency not in latencies:
            raise SystemExit("capture latency must be one of the predeclared latency levels")
        if args.capture_seed not in seeds:
            raise SystemExit("capture seed must be one of the predeclared evaluation seeds")
        run_shot(G.shot(args.capture_seed), args.capture_latency,
                 OUT / f"evidence-latency-{args.capture_latency:.1f}-seed-{args.capture_seed}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
