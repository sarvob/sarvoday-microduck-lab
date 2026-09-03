#!/usr/bin/env python3
"""Evaluate Microduck's frozen stand policy on a moving boat deck."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from duck import CTRL_DT, Microduck  # noqa: E402


SPEC_PATH = ROOT / "challenges" / "012-variable-speed-boat-balance" / "spec.json"
OUT = ROOT / "artifacts" / "012-variable-speed-boat-balance"
DECK_HALF = np.array([0.8, 0.4])
DECK_TOP_Z = 0.105


def add_boat_deck(xml: str) -> str:
    """Add a kinematic collision deck and lift the robot's start onto it."""
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    assert world is not None
    boat = ET.SubElement(world, "body", {
        "name": "boat_deck", "pos": "0 0 0", "gravcomp": "1"
    })
    ET.SubElement(boat, "freejoint", {"name": "boat_freejoint"})
    ET.SubElement(boat, "geom", {
        "name": "boat_deck_geom", "type": "box", "size": "0.8 0.4 0.05",
        "mass": "20", "friction": "0.9 0.02 0.005", "condim": "6",
        "solref": "0.015 1", "rgba": "0.09 0.25 0.34 1"
    })
    ET.SubElement(boat, "geom", {
        "name": "boat_hull", "type": "box", "size": "0.72 0.34 0.08",
        "pos": "0 0 -0.13", "contype": "0", "conaffinity": "0",
        "rgba": "0.04 0.12 0.18 1", "group": "2"
    })
    key = root.find("./keyframe/key[@name='STAND']")
    assert key is not None
    qpos = key.get("qpos", "").split()
    qpos[2] = str(float(qpos[2]) + DECK_TOP_Z)
    qpos.extend(["0", "0", "0.055", "1", "0", "0", "0"])
    key.set("qpos", " ".join(qpos))
    return ET.tostring(root, encoding="unicode")


def quaternion(roll: float, pitch: float) -> np.ndarray:
    """Quaternion for pitch-after-roll, in MuJoCo's wxyz convention."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cp * cr, cp * sr, sp * cr, -sp * sr])


def motion(profile: dict, t: float, ramp_s: float) -> tuple[float, float, float]:
    ramp_phase = min(max(t / ramp_s, 0.0), 1.0)
    ramp = 0.5 - 0.5 * math.cos(math.pi * ramp_phase)
    speed_spec = profile["forward_speed_mps"]
    speed = speed_spec["base"] + speed_spec["amplitude"] * math.sin(
        2 * math.pi * t / speed_spec["period_s"])
    roll = math.radians(profile["roll"]["amplitude_deg"]) * math.sin(
        2 * math.pi * t / profile["roll"]["period_s"])
    pitch = math.radians(profile["pitch"]["amplitude_deg"]) * math.sin(
        2 * math.pi * t / profile["pitch"]["period_s"] + math.pi / 3)
    return ramp * speed, ramp * roll, ramp * pitch


def contact_flags(sim: Microduck, deck_geom: int, floor_geom: int,
                  feet: set[int]) -> tuple[bool, bool]:
    deck_contact = floor_contact = False
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        deck_contact |= deck_geom in pair and bool(pair & feet)
        floor_contact |= floor_geom in pair and bool(pair & feet)
    return deck_contact, floor_contact


def set_deck_state(sim: Microduck, qadr: int, dadr: int, x: float,
                   speed: float, roll: float, pitch: float,
                   roll_rate: float, pitch_rate: float) -> None:
    sim.data.qpos[qadr:qadr + 3] = [x, 0.0, 0.055]
    sim.data.qpos[qadr + 3:qadr + 7] = quaternion(roll, pitch)
    sim.data.qvel[dadr:dadr + 6] = [speed, 0.0, 0.0, roll_rate, pitch_rate, 0.0]
    sim.mj.mj_forward(sim.model, sim.data)


def rollout(sim: Microduck, profile: dict, seed: int, spec: dict) -> dict:
    sim.reset()
    rng = np.random.default_rng(seed)
    sim.data.qpos[sim.qadr] += rng.normal(
        0.0, spec["training"]["initial_joint_noise_rad"], len(sim.qadr))
    sim.mj.mj_forward(sim.model, sim.data)
    boat_q = int(sim.model.joint("boat_freejoint").qposadr[0])
    boat_d = int(sim.model.joint("boat_freejoint").dofadr[0])
    deck_geom = sim.model.geom("boat_deck_geom").id
    floor_geom = sim.model.geom("floor").id
    feet = {
        sim.model.geom("left_foot_collision").id,
        sim.model.geom("right_foot_collision").id,
    }
    deck_x = 0.0
    for _ in range(50):
        set_deck_state(sim, boat_q, boat_d, deck_x, 0.0, 0.0, 0.0, 0.0, 0.0)
        sim.control_step("stand", [0.0, 0.0, 0.0])

    start_relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
    steps = round(spec["environment"]["evaluation_duration_s"] / CTRL_DT)
    upright_samples = []
    relative_displacements = []
    deck_contact_steps = 0
    floor_contact = False
    inside_bounds = True
    first_floor_contact_s = None
    first_deck_exit_s = None
    speeds = []
    previous_roll = previous_pitch = 0.0

    for step in range(steps):
        t = step * CTRL_DT
        speed, roll, pitch = motion(profile, t, spec["environment"]["motion_ramp_s"])
        deck_x += speed * CTRL_DT
        roll_rate = (roll - previous_roll) / CTRL_DT
        pitch_rate = (pitch - previous_pitch) / CTRL_DT
        set_deck_state(sim, boat_q, boat_d, deck_x, speed, roll, pitch,
                       roll_rate, pitch_rate)
        previous_roll, previous_pitch = roll, pitch
        sim.control_step("stand", [0.0, 0.0, 0.0])
        deck_contact, on_floor = contact_flags(sim, deck_geom, floor_geom, feet)
        deck_contact_steps += int(deck_contact)
        floor_contact |= on_floor
        if on_floor and first_floor_contact_s is None:
            first_floor_contact_s = (step + 1) * CTRL_DT
        relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
        relative_displacements.append(float(np.linalg.norm(relative - start_relative)))
        inside_now = (
            abs(float(relative[0])) <= DECK_HALF[0] - 0.12
            and abs(float(relative[1])) <= DECK_HALF[1] - 0.12
        )
        inside_bounds &= inside_now
        if not inside_now and first_deck_exit_s is None:
            first_deck_exit_s = (step + 1) * CTRL_DT
        upright_samples.append(-float(sim.proj_gravity()[2]))
        speeds.append(speed)

    gate = spec["success"]
    row = {
        "profile": profile["name"],
        "seed": seed,
        "duration_s": steps * CTRL_DT,
        "speed_range_mps": [round(min(speeds), 4), round(max(speeds), 4)],
        "minimum_upright_score": round(min(upright_samples), 4),
        "final_upright_score": round(float(np.mean(upright_samples[-25:])), 4),
        "deck_contact_ratio": round(deck_contact_steps / steps, 4),
        "maximum_relative_deck_displacement_m": round(max(relative_displacements), 4),
        "floor_contact": floor_contact,
        "first_floor_contact_s": (
            None if first_floor_contact_s is None else round(first_floor_contact_s, 3)
        ),
        "first_deck_exit_s": (
            None if first_deck_exit_s is None else round(first_deck_exit_s, 3)
        ),
        "survival_time_s": round(min(
            value for value in (
                first_floor_contact_s,
                first_deck_exit_s,
                steps * CTRL_DT,
            ) if value is not None
        ), 3),
        "remained_inside_deck_bounds": bool(inside_bounds),
    }
    row["success"] = bool(
        row["minimum_upright_score"] >= gate["minimum_upright_score"]
        and row["final_upright_score"] >= gate["minimum_final_upright_score"]
        and row["deck_contact_ratio"] >= gate["minimum_deck_contact_ratio"]
        and row["maximum_relative_deck_displacement_m"]
        <= gate["maximum_relative_deck_displacement_m"]
        and not row["floor_contact"]
        and row["remained_inside_deck_bounds"]
    )
    return row


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    evaluations = [
        rollout(sim, profile, seed, spec)
        for profile in spec["environment"]["profiles"]
        for seed in spec["training"]["seeds"]
    ]
    result = {
        "challenge": spec["id"],
        "controller": spec["baseline"]["controller"],
        "policy_frozen": True,
        "evaluation_count": len(evaluations),
        "success_gate": spec["success"],
        "success": all(row["success"] for row in evaluations),
        "evaluations": evaluations,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
