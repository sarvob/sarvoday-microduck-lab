#!/usr/bin/env python3
"""Measure outside-camera frame-loss tolerance without hidden state.

Every candidate observation is reconstructed from rendered overhead-camera
pixels. A deterministic, predeclared per-shot mask either delivers the packet
immediately or discards it. The same mask scores are reused at every tested
drop fraction, so higher-loss conditions keep a strict subset of lower-loss
frames. Physics, shot generation, prediction gain, locomotion, and scoring are
inherited unchanged from challenges 011, 013, and 014.
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

SPEC = ROOT / "challenges" / "015-external-camera-frame-loss" / "spec.json"
OUT = ROOT / "artifacts" / "015-external-camera-frame-loss"


def frame_scores(seed: int, count: int, mask_seed: int) -> np.ndarray:
    """Return reproducible per-frame scores shared by every loss condition."""
    return np.random.default_rng(np.random.SeedSequence([mask_seed, seed])).random(count)


def keep_frame(scores: np.ndarray, frame_index: int, drop_fraction: float) -> bool:
    """Keep exactly the frames whose precomputed score clears the loss rate."""
    return bool(scores[frame_index] >= drop_fraction)


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def compose_vertical(scene: np.ndarray, outside: np.ndarray, *, drop_fraction: float,
                     t: float, rendered: int, delivered: int, dropped: int,
                     last_packet_kept: bool | None, target_y: float,
                     goal: bool, saved: bool) -> np.ndarray:
    canvas = Image.new("RGB", (1440, 2560), "#071018")
    draw = ImageDraw.Draw(canvas)
    accent = "#49E6A1" if drop_fraction <= 0.5 else "#FFB24C"
    draw.text((72, 68), "WHAT IF THE CAMERA DROPS FRAMES?", font=font(47, True), fill="#F7FAFC")
    draw.text((72, 145), f"{drop_fraction * 100:.0f}% PREDECLARED PACKET LOSS", font=font(38, True), fill=accent)

    main = Image.fromarray(scene).resize((1296, 729), Image.Resampling.LANCZOS)
    canvas.paste(main, (72, 235))
    draw.rounded_rectangle((72, 235, 1368, 964), 28, outline="#EAF2F5", width=5)

    ext = Image.fromarray(outside).resize((1000, 562), Image.Resampling.LANCZOS)
    if last_packet_kept is False:
        muted = Image.new("RGB", ext.size, "#111A20")
        ext = Image.blend(ext, muted, 0.72)
        ex = ImageDraw.Draw(ext)
        ex.line((50, 45, 950, 517), fill="#FF7650", width=18)
        ex.line((950, 45, 50, 517), fill="#FF7650", width=18)
    canvas.paste(ext, (220, 1055))
    draw.rounded_rectangle((220, 1055, 1220, 1617), 24, outline=accent, width=5)
    packet_label = "PACKET KEPT" if last_packet_kept is not False else "PACKET DROPPED"
    packet_color = "#49E6A1" if last_packet_kept is not False else "#FF7650"
    draw.text((242, 1640), f"OVERHEAD CAMERA · {packet_label}", font=font(30, True), fill=packet_color)

    draw.rounded_rectangle((72, 1745, 1368, 2200), 34, fill="#0D1B24", outline="#263E4B", width=3)
    draw.text((118, 1790), f"frames rendered   {rendered}", font=font(35), fill="#CAD7DC")
    draw.text((118, 1855), f"frames delivered  {delivered}", font=font(35), fill="#CAD7DC")
    draw.text((118, 1920), f"frames dropped    {dropped}", font=font(35), fill="#CAD7DC")
    draw.text((118, 1985), f"target             {target_y:+.3f} m", font=font(35), fill="#CAD7DC")
    draw.text((118, 2070), "Same robot · same shot · missing pixels", font=font(34, True), fill=accent)

    if goal:
        label, color = "MISSED · GOAL", "#FF7650"
    elif saved:
        label, color = "SAVE", "#49E6A1"
    else:
        label, color = f"TRACKING · {t:04.1f} s", accent
    draw.rounded_rectangle((72, 2280, 1368, 2480), 40, fill="#10212B", outline=color, width=6)
    box = draw.textbbox((0, 0), label, font=font(68, True))
    draw.text(((1440 - (box[2] - box[0])) / 2, 2335), label, font=font(68, True), fill=color)
    return np.asarray(canvas)


def run_shot(case: dict, drop_fraction: float, mask_seed: int,
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
    candidate_count = sum(
        1 for step in range(round(G.SHOT_SECONDS / D.CTRL_DT))
        if step % G.DETECT_EVERY == 0 and step * D.CTRL_DT <= 1.4
    )
    scores = frame_scores(int(case["seed"]), candidate_count, mask_seed)
    observations: list[tuple[float, np.ndarray]] = []
    rendered = dropped = 0
    last_packet_kept: bool | None = None
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
                packet_index = rendered
                rendered += 1
                last_packet_kept = keep_frame(scores, packet_index, drop_fraction)
                if last_packet_kept:
                    if measured is not None:
                        observations.append((t, measured))
                else:
                    dropped += 1
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
                # A contact is not yet a save: the ball can deflect and still
                # cross the goal line later. Confirm only at the terminal safe
                # outcome used by the evaluator's own scoring path.
                final_step = step == round(G.SHOT_SECONDS / D.CTRL_DT) - 1
                save_confirmed = bool(
                    contacted and not goal_scored and (bx > 0.55 or final_step)
                )
                writer.append_data(compose_vertical(
                    scene, outside_frame, drop_fraction=drop_fraction, t=t,
                    rendered=rendered, delivered=len(observations), dropped=dropped,
                    last_packet_kept=last_packet_kept, target_y=target_y,
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
        "drop_fraction": drop_fraction,
        "rendered_frames": rendered,
        "delivered_observations": len(observations),
        "dropped_frames": dropped,
        "realized_drop_fraction": round(dropped / rendered, 5) if rendered else 0.0,
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
    parser.add_argument("--capture-drop", type=float)
    parser.add_argument("--capture-seed", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC.read_text())
    seeds = spec["evaluation"]["seeds"]
    drops = spec["evaluation"]["drop_fractions"]
    mask_seed = spec["evaluation"]["loss_mask_seed"]
    conditions = []
    for drop in drops:
        cases = [run_shot(G.shot(seed), float(drop), mask_seed) for seed in seeds]
        saves = sum(case["passed"] for case in cases)
        stable = all(not case["fell"] and not case["left_goal_zone"] for case in cases)
        meets_gate = saves >= spec["tolerance_gate"]["minimum_saves"] and stable
        rendered = sum(case["rendered_frames"] for case in cases)
        dropped = sum(case["dropped_frames"] for case in cases)
        conditions.append({
            "drop_fraction": drop,
            "saves": saves,
            "shots": len(cases),
            "stable": stable,
            "rendered_frames": rendered,
            "dropped_frames": dropped,
            "realized_drop_fraction": round(dropped / rendered, 5) if rendered else 0.0,
            "meets_tolerance_gate": meets_gate,
            "cases": cases,
        })
        print(f"drop {drop * 100:.0f}%: {saves}/{len(cases)} saves"
              f" · {dropped}/{rendered} packets dropped"
              f" · {'GATE PASS' if meets_gate else 'GATE FAIL'}")
    passing = [item["drop_fraction"] for item in conditions if item["meets_tolerance_gate"]]
    first_failure = next((item["drop_fraction"] for item in conditions
                          if not item["meets_tolerance_gate"]), None)
    result = {
        "challenge": spec["id"],
        "tolerance_gate": spec["tolerance_gate"],
        "loss_mask_seed": mask_seed,
        "conditions": conditions,
        "maximum_tested_passing_drop_fraction": max(passing) if passing else None,
        "first_tested_failing_drop_fraction": first_failure,
        "disclosure": (
            "Every candidate observation came from rendered overhead-camera pixels. "
            "A deterministic predeclared per-shot mask discarded packets immediately; "
            "higher-loss conditions retained a strict subset of lower-loss frames. "
            "No delivery delay or hidden ball state was added."
        ),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    if args.capture_drop is not None or args.capture_seed is not None:
        if args.capture_drop not in drops:
            raise SystemExit("capture drop must be one of the predeclared loss levels")
        if args.capture_seed not in seeds:
            raise SystemExit("capture seed must be one of the predeclared evaluation seeds")
        run_shot(G.shot(args.capture_seed), args.capture_drop, mask_seed,
                 OUT / f"evidence-drop-{args.capture_drop:.2f}-seed-{args.capture_seed}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
