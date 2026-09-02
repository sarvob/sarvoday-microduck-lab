#!/usr/bin/env python3
"""Calibrate the smallest one-shot launch that reaches the Go1 landing pad."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_quadruped_scene import make_scene  # noqa: E402
from duck import DEFAULT_POSE, JOINT_NAMES, Microduck, NUM_JOINTS  # noqa: E402


OUT = ROOT / "artifacts" / "005-duck-quadruped-jump" / "launch-calibration.json"
CONTROL_DT = 0.02


def make_runtime():
    sim = Microduck(render=False)
    model, data = make_scene()
    sim.model, sim.data, sim.mj = model, data, mujoco
    sim.qadr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    sim.dadr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
    sim.gyro = model.sensor("imu_ang_vel").adr[0]
    sim.trunk = model.body("trunk_base").id
    sim.ball_q = model.joint("ball_freejoint").qposadr[0]
    sim.ball_d = model.joint("ball_freejoint").dofadr[0]
    sim.variant = "legs"
    sim._scene_qpos = data.qpos.copy()
    sim._scene_ctrl = data.ctrl.copy()
    return sim


def reset(sim) -> None:
    mujoco.mj_resetData(sim.model, sim.data)
    sim.data.qpos[:] = sim._scene_qpos
    sim.data.ctrl[:] = sim._scene_ctrl
    sim.data.qpos[:7] = [0, 0, 0.12, 1, 0, 0, 0]
    for name, value in zip(JOINT_NAMES, DEFAULT_POSE):
        sim.data.qpos[sim.model.joint(name).qposadr[0]] = value
    sim.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    sim.head_target = np.zeros(4, dtype=np.float32)
    sim.head_smooth = np.zeros(4, dtype=np.float32)
    sim.recovery = None
    sim.fall_debounce = 0
    sim.post_kick_lock = 0
    sim.ball_active = False
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(35):
        sim.control_step("stand", [0, 0, 0])
    # Re-anchor the launch so every candidate begins from the same x/y point.
    sim.data.qpos[0:2] = 0.0
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def contact_state(sim) -> tuple[bool, bool, bool]:
    left = sim.model.geom("left_foot_collision").id
    right = sim.model.geom("right_foot_collision").id
    pad = sim.model.geom("go1_landing_pad").id
    floor = sim.model.geom("floor").id
    left_pad = right_pad = ground = False
    for index in range(sim.data.ncon):
        pair = {sim.data.contact[index].geom1, sim.data.contact[index].geom2}
        left_pad |= pair == {left, pad}
        right_pad |= pair == {right, pad}
        if floor in pair and pad not in pair:
            other = next(iter(pair - {floor}))
            name = sim.model.geom(other).name or ""
            ground |= not name.startswith("go1_") and name not in {
                "ball_geom", "target_0", "target_1",
                "wall_px", "wall_nx", "wall_py", "wall_ny",
            }
    return left_pad, right_pad, ground


def rollout(sim, vx: float, vz: float) -> dict:
    reset(sim)
    sim.data.qvel[0] = vx
    sim.data.qvel[2] = vz
    target = sim.data.site_xpos[sim.model.site("go1_landing_target").id].copy()
    closest = float("inf")
    first_contact = None
    landing_speed = None
    longest_hold = current_hold = 0.0
    touched_left = touched_right = touched_both = False
    ground_after_contact = False
    maximum_z = float(sim.data.xpos[sim.trunk][2])

    for step in range(100):
        sim.control_step("stand", [0, 0, 0])
        xyz = sim.data.xpos[sim.trunk].copy()
        maximum_z = max(maximum_z, float(xyz[2]))
        closest = min(closest, float(np.linalg.norm(xyz - target)))
        left, right, ground = contact_state(sim)
        touched_left |= left
        touched_right |= right
        touched_both |= left and right
        if (left or right) and first_contact is None:
            first_contact = (step + 1) * CONTROL_DT
            landing_speed = abs(float(sim.data.qvel[2]))
        if first_contact is not None and ground:
            ground_after_contact = True
        upright = -float(sim.proj_gravity()[2]) >= 0.75
        stable = left and right and upright and not ground
        current_hold = current_hold + CONTROL_DT if stable else 0.0
        longest_hold = max(longest_hold, current_hold)

    speed = float(np.hypot(vx, vz))
    score = (
        12.0 * longest_hold
        + 1.5 * float(touched_both)
        + 0.5 * float(touched_left and touched_right)
        - 4.0 * closest
        - 0.04 * speed
        - 1.5 * float(ground_after_contact)
    )
    return {
        "vx_mps": round(vx, 3),
        "vz_mps": round(vz, 3),
        "launch_speed_mps": round(speed, 3),
        "maximum_trunk_z_m": round(maximum_z, 4),
        "closest_target_distance_m": round(closest, 4),
        "left_foot_touched_pad": touched_left,
        "right_foot_touched_pad": touched_right,
        "both_feet_simultaneous": touched_both,
        "first_pad_contact_s": None if first_contact is None else round(first_contact, 3),
        "landing_speed_mps": None if landing_speed is None else round(landing_speed, 3),
        "longest_untrained_hold_s": round(longest_hold, 3),
        "ground_contact_after_pad": ground_after_contact,
        "score": round(score, 5),
    }


def calibrate() -> dict:
    sim = make_runtime()
    candidates = [
        rollout(sim, float(vx), float(vz))
        for vx, vz in itertools.product(
            np.linspace(0.9, 1.7, 9),
            np.linspace(2.9, 3.7, 9),
        )
    ]
    candidates.sort(key=lambda row: (-row["score"], row["launch_speed_mps"]))
    best = candidates[0]
    return {
        "method": "81-candidate deterministic ballistic grid; stand policy in flight",
        "assistance": "initial root velocity only; no midair external force",
        "success_gate_met": best["both_feet_simultaneous"] and best["longest_untrained_hold_s"] >= 1.5,
        "best": best,
        "top_candidates": candidates[:10],
    }


if __name__ == "__main__":
    result = calibrate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
