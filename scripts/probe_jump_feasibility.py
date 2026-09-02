#!/usr/bin/env python3
"""Measure whether the shipped Microduck roll policy produces a true jump."""

from __future__ import annotations

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


OUT = ROOT / "artifacts" / "005-duck-quadruped-jump" / "launch-feasibility.json"


def runtime():
    sim = Microduck(render=False)
    model, data = make_scene()
    sim.model, sim.data, sim.mj = model, data, mujoco
    sim.qadr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    sim.dadr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
    sim.gyro = model.sensor("imu_ang_vel").adr[0]
    sim.trunk = model.body("trunk_base").id
    sim.ball_q = model.joint("ball_freejoint").qposadr[0]
    sim.ball_d = model.joint("ball_freejoint").dofadr[0]
    sim.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    sim.head_target = np.zeros(4, dtype=np.float32)
    sim.head_smooth = np.zeros(4, dtype=np.float32)
    sim.recovery = None
    sim.fall_debounce = 0
    sim.post_kick_lock = 0
    sim.ball_active = False
    sim.variant = "legs"

    data.qpos[:7] = [0, 0, 0.12, 1, 0, 0, 0]
    for name, value in zip(JOINT_NAMES, DEFAULT_POSE):
        data.qpos[model.joint(name).qposadr[0]] = value
    # Move the target away: this probe isolates unassisted launch capability.
    model.body_pos[model.body("go1_trunk").id, 0] = 2.0
    mujoco.mj_forward(model, data)
    return sim


def duck_has_floor_contact(sim) -> bool:
    floor = sim.model.geom("floor").id
    ignored = {sim.model.geom(name).id for name in (
        "ball_geom", "wall_px", "wall_nx", "wall_py", "wall_ny",
        "target_0", "target_1",
    )}
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        if floor not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        name = sim.model.geom(other).name or ""
        if other not in ignored and not name.startswith("go1_"):
            return True
    return False


def probe() -> dict:
    sim = runtime()
    for _ in range(100):
        sim.control_step("stand", [0, 0, 0])
    stand = sim.data.xpos[sim.trunk].copy()

    maximum = stand.copy()
    minimum = stand.copy()
    longest_air = current_air = 0.0
    for step in range(200):
        sim.control_step("roll", [0, 0, 0])
        xyz = sim.data.xpos[sim.trunk]
        maximum = np.maximum(maximum, xyz)
        minimum = np.minimum(minimum, xyz)
        airborne = not duck_has_floor_contact(sim) and xyz[2] > stand[2] + 0.01
        current_air = current_air + 0.02 if airborne else 0.0
        longest_air = max(longest_air, current_air)

    rise = float(maximum[2] - stand[2])
    return {
        "policy": "official shipped roll/roulade ONNX",
        "settled_trunk_z_m": round(float(stand[2]), 4),
        "maximum_trunk_z_m": round(float(maximum[2]), 4),
        "maximum_trunk_rise_m": round(rise, 4),
        "maximum_forward_travel_m": round(float(maximum[0] - stand[0]), 4),
        "longest_true_airborne_window_s": round(longest_air, 4),
        "target_vertical_displacement_m": 0.417,
        "unassisted_target_feasible": bool(rise >= 0.417 and longest_air >= 0.12),
        "decision": "Use a disclosed launch assist and train the landing policy"
        if rise < 0.417 or longest_air < 0.12
        else "Proceed with unassisted jump training",
    }


if __name__ == "__main__":
    result = probe()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
