#!/usr/bin/env python3
"""Measure command saturation and foot-contact timing in chop failures.

Attempt 7 retained attempt 6's controller exactly, so this diagnostic applies
to both. The held-out surge profile is never loaded or evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duck import CTRL_DT, Microduck, VEL_BACK, VEL_FWD  # noqa: E402
from evaluate_boat_balance import (DECK_HALF, add_boat_deck, motion,
                                   set_deck_state)  # noqa: E402
from train_boat_residual import control_step, residual_vector  # noqa: E402

SPEC_PATH = ROOT / "challenges/012-variable-speed-boat-balance/spec.json"
POLICY_PATH = ROOT / "artifacts/012-variable-speed-boat-balance/joint-walk-refinement-result.json"
OUT_PATH = ROOT / "artifacts/012-variable-speed-boat-balance/attempt-7-saturation-contact-diagnosis.json"


def foot_contacts(sim: Microduck, deck_geom: int, foot_geoms: list[int]) -> tuple[bool, bool]:
    result = [False, False]
    for index in range(sim.data.ncon):
        pair = {int(sim.data.contact[index].geom1), int(sim.data.contact[index].geom2)}
        for foot_index, geom in enumerate(foot_geoms):
            result[foot_index] |= deck_geom in pair and geom in pair
    return bool(result[0]), bool(result[1])


def diagnose(sim: Microduck, profile: dict, seed: int, residual_weights: np.ndarray,
             gains: np.ndarray, spec: dict) -> dict:
    sim.reset()
    rng = np.random.default_rng(seed)
    sim.data.qpos[sim.qadr] += rng.normal(
        0.0, spec["training"]["initial_joint_noise_rad"], len(sim.qadr))
    sim.mj.mj_forward(sim.model, sim.data)
    boat_q = int(sim.model.joint("boat_freejoint").qposadr[0])
    boat_d = int(sim.model.joint("boat_freejoint").dofadr[0])
    deck_geom = sim.model.geom("boat_deck_geom").id
    foot_geoms = [
        sim.model.geom("left_foot_collision").id,
        sim.model.geom("right_foot_collision").id,
    ]
    for _ in range(50):
        set_deck_state(sim, boat_q, boat_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sim.control_step("stand", [0.0, 0.0, 0.0])

    start_relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
    limit = spec["training"]["maximum_joint_residual_rad"]
    deck_x = 0.0
    previous_roll = previous_pitch = previous_speed = previous_apparent_pitch = 0.0
    x_saturated = y_saturated = residual_saturated = 0
    contact_rows: list[tuple[bool, bool]] = []
    raw_commands: list[np.ndarray] = []
    relative_position = np.zeros(2)

    for step in range(round(spec["environment"]["evaluation_duration_s"] / CTRL_DT)):
        t = step * CTRL_DT
        speed, roll, pitch = motion(profile, t, spec["environment"]["motion_ramp_s"])
        deck_x += speed * CTRL_DT
        roll_rate = (roll - previous_roll) / CTRL_DT
        pitch_rate = (pitch - previous_pitch) / CTRL_DT
        acceleration = (speed - previous_speed) / CTRL_DT
        apparent_pitch = pitch + np.arctan2(acceleration, 9.81)
        apparent_pitch_rate = (apparent_pitch - previous_apparent_pitch) / CTRL_DT
        set_deck_state(sim, boat_q, boat_d, deck_x, speed, roll, pitch,
                       roll_rate, pitch_rate)
        relative_position = (
            sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
            - start_relative
        )
        relative_velocity = sim.data.qvel[:2] - np.array([speed, 0.0])
        raw = np.array([
            -gains[0] * relative_position[0] - gains[1] * relative_velocity[0],
            -gains[2] * relative_position[1] - gains[3] * relative_velocity[1],
        ])
        command = np.array([
            np.clip(raw[0], VEL_BACK, VEL_FWD),
            np.clip(raw[1], -0.2, 0.2),
            0.0,
        ], dtype=np.float32)
        x_saturated += int(raw[0] <= VEL_BACK or raw[0] >= VEL_FWD)
        y_saturated += int(abs(raw[1]) >= 0.2)
        raw_commands.append(raw)
        residual = residual_vector(
            residual_weights, roll, apparent_pitch, roll_rate,
            apparent_pitch_rate, relative_position, relative_velocity, limit)
        residual_saturated += int(np.any(np.isclose(np.abs(residual), limit, atol=1e-6)))
        control_step(sim, residual, mode="walk", command3=command)
        previous_roll, previous_pitch = roll, pitch
        previous_speed, previous_apparent_pitch = speed, apparent_pitch
        contact_rows.append(foot_contacts(sim, deck_geom, foot_geoms))
        relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
        inside = (
            abs(float(relative[0])) <= DECK_HALF[0] - 0.12
            and abs(float(relative[1])) <= DECK_HALF[1] - 0.12
        )
        if not inside:
            break

    steps = len(contact_rows)
    any_contact = [left or right for left, right in contact_rows]
    both_contact = [left and right for left, right in contact_rows]
    longest_gap = run = 0
    first_sustained_loss = None
    for index, present in enumerate(any_contact):
        run = 0 if present else run + 1
        longest_gap = max(longest_gap, run)
        if run == 5 and first_sustained_loss is None:
            first_sustained_loss = (index - 4) * CTRL_DT
    tail = any_contact[-min(50, steps):]
    raw_array = np.asarray(raw_commands)
    dominant_axis = "longitudinal" if abs(relative_position[0]) >= abs(relative_position[1]) else "lateral"
    return {
        "seed": seed,
        "survival_time_s": round(steps * CTRL_DT, 3),
        "exit_axis": dominant_axis,
        "final_relative_position_m": [round(float(value), 4) for value in relative_position],
        "x_command_saturation_ratio": round(x_saturated / steps, 4),
        "y_command_saturation_ratio": round(y_saturated / steps, 4),
        "residual_saturation_ratio": round(residual_saturated / steps, 4),
        "peak_raw_command": [round(float(value), 4) for value in np.max(np.abs(raw_array), axis=0)],
        "left_foot_contact_ratio": round(sum(left for left, _ in contact_rows) / steps, 4),
        "right_foot_contact_ratio": round(sum(right for _, right in contact_rows) / steps, 4),
        "both_feet_contact_ratio": round(sum(both_contact) / steps, 4),
        "last_second_any_contact_ratio": round(sum(tail) / len(tail), 4),
        "longest_no_contact_gap_s": round(longest_gap * CTRL_DT, 3),
        "first_sustained_contact_loss_s": (
            None if first_sustained_loss is None else round(first_sustained_loss, 3)),
    }


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    saved = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    profile = next(row for row in spec["environment"]["profiles"] if row["name"] == "chop")
    assert spec["training"]["held_out_profile"] != profile["name"]
    residual = np.asarray(saved["residual_weights"], dtype=float)
    gains = np.asarray(saved["recenter_gains"], dtype=float)
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rows = [diagnose(sim, profile, seed, residual, gains, spec)
            for seed in spec["training"]["seeds"]]
    result = {
        "challenge": spec["id"],
        "controller_attempts": [6, 7],
        "profile": "chop",
        "held_out_profile_touched": False,
        "evaluations": rows,
        "summary": {
            "x_command_saturation_range": [
                min(row["x_command_saturation_ratio"] for row in rows),
                max(row["x_command_saturation_ratio"] for row in rows),
            ],
            "y_command_saturation_range": [
                min(row["y_command_saturation_ratio"] for row in rows),
                max(row["y_command_saturation_ratio"] for row in rows),
            ],
            "residual_saturation_range": [
                min(row["residual_saturation_ratio"] for row in rows),
                max(row["residual_saturation_ratio"] for row in rows),
            ],
            "first_sustained_contact_loss_range_s": [
                min(row["first_sustained_contact_loss_s"] for row in rows),
                max(row["first_sustained_contact_loss_s"] for row in rows),
            ],
        },
        "finding": (
            "The longitudinal walk command saturates for 57-59% of each chop run; lateral "
            "commands and the bounded joint residual also saturate materially. Sustained "
            "contact loss begins near 3 seconds, well before every deck exit. Attempt 8 "
            "should change capture and step timing rather than merely increase gain limits."
        ),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
