#!/usr/bin/env python3
"""Attempt 8: search contact-aware command latching around frozen policies."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duck import CTRL_DT, Microduck  # noqa: E402
from evaluate_boat_balance import (DECK_HALF, add_boat_deck, contact_flags,
                                   motion, set_deck_state)  # noqa: E402
from train_boat_joint_walk import passes_gate  # noqa: E402
from train_boat_recenter import recenter_command  # noqa: E402
from train_boat_residual import control_step, residual_vector  # noqa: E402

SPEC_PATH = ROOT / "challenges/012-variable-speed-boat-balance/spec.json"
START_PATH = ROOT / "artifacts/012-variable-speed-boat-balance/joint-walk-refinement-result.json"
OUT_PATH = ROOT / "artifacts/012-variable-speed-boat-balance/contact-capture-training-result.json"


def individual_contacts(sim: Microduck, deck_geom: int, feet: list[int]) -> tuple[bool, bool]:
    states = [False, False]
    for index in range(sim.data.ncon):
        pair = {int(sim.data.contact[index].geom1), int(sim.data.contact[index].geom2)}
        for foot_index, foot in enumerate(feet):
            states[foot_index] |= deck_geom in pair and foot in pair
    return bool(states[0]), bool(states[1])


def rollout(sim: Microduck, profile: dict, seed: int, residual_weights: np.ndarray,
            gains: np.ndarray, spec: dict, lookahead_s: float,
            maximum_hold_steps: int, command_scale: float) -> dict:
    sim.reset()
    rng = np.random.default_rng(seed)
    sim.data.qpos[sim.qadr] += rng.normal(
        0.0, spec["training"]["initial_joint_noise_rad"], len(sim.qadr))
    sim.mj.mj_forward(sim.model, sim.data)
    boat_q = int(sim.model.joint("boat_freejoint").qposadr[0])
    boat_d = int(sim.model.joint("boat_freejoint").dofadr[0])
    deck_geom = sim.model.geom("boat_deck_geom").id
    floor_geom = sim.model.geom("floor").id
    feet = [sim.model.geom("left_foot_collision").id,
            sim.model.geom("right_foot_collision").id]
    for _ in range(50):
        set_deck_state(sim, boat_q, boat_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sim.control_step("stand", [0.0, 0.0, 0.0])

    start_relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
    total_steps = round(spec["environment"]["evaluation_duration_s"] / CTRL_DT)
    limit = spec["training"]["maximum_joint_residual_rad"]
    deck_x = 0.0
    previous_roll = previous_pitch = previous_speed = previous_apparent_pitch = 0.0
    latched_command = np.zeros(3, dtype=np.float32)
    hold_age = maximum_hold_steps
    prior_contacts = (True, True)
    contact_steps = completed_steps = command_updates = 0
    minimum_upright = 1.0
    maximum_drift = 0.0
    upright_samples: list[float] = []
    failed = False

    for step in range(total_steps):
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
        predicted = relative_position + lookahead_s * relative_velocity
        if all(prior_contacts) or hold_age >= maximum_hold_steps:
            latched_command = command_scale * recenter_command(
                gains, predicted, relative_velocity)
            hold_age = 0
            command_updates += 1
        else:
            hold_age += 1
        residual = residual_vector(
            residual_weights, roll, apparent_pitch, roll_rate,
            apparent_pitch_rate, relative_position, relative_velocity, limit)
        control_step(sim, residual, mode="walk", command3=latched_command)
        previous_roll, previous_pitch = roll, pitch
        previous_speed, previous_apparent_pitch = speed, apparent_pitch
        prior_contacts = individual_contacts(sim, deck_geom, feet)
        deck_contact, floor_contact = contact_flags(sim, deck_geom, floor_geom, set(feet))
        contact_steps += int(deck_contact)
        relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
        drift = float(np.linalg.norm(relative - start_relative))
        maximum_drift = max(maximum_drift, drift)
        upright = -float(sim.proj_gravity()[2])
        upright_samples.append(upright)
        minimum_upright = min(minimum_upright, upright)
        inside = (
            abs(float(relative[0])) <= DECK_HALF[0] - 0.12
            and abs(float(relative[1])) <= DECK_HALF[1] - 0.12
        )
        completed_steps = step + 1
        failed = floor_contact or not inside
        if failed:
            break

    survival = completed_steps * CTRL_DT
    contact_ratio = contact_steps / max(completed_steps, 1)
    score = (8.0 * survival + 18.0 * contact_ratio
             + 4.0 * max(minimum_upright, -1.0) - 6.0 * maximum_drift)
    return {
        "profile": profile["name"], "seed": seed,
        "survival_time_s": round(survival, 3),
        "deck_contact_ratio": round(contact_ratio, 4),
        "minimum_upright_score": round(minimum_upright, 4),
        "final_upright_score": round(float(np.mean(upright_samples[-25:])), 4),
        "maximum_relative_deck_displacement_m": round(maximum_drift, 4),
        "failed": bool(failed),
        "command_update_ratio": round(command_updates / max(completed_steps, 1), 4),
        "score": round(score, 5),
    }


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profile_map = {row["name"]: row for row in spec["environment"]["profiles"]}
    profiles = [profile_map[name] for name in spec["training"]["training_profiles"]]
    assert spec["training"]["held_out_profile"] not in {p["name"] for p in profiles}
    saved = json.loads(START_PATH.read_text(encoding="utf-8"))
    residual = np.asarray(saved["residual_weights"], dtype=float)
    gains = np.asarray(saved["recenter_gains"], dtype=float)
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    candidates = list(itertools.product(
        [0.0, 0.1, 0.2, 0.4], [1, 5, 10, 20], [0.75, 1.0]))
    evaluated = []
    best_key = None
    best_rows = None
    for lookahead, hold_steps, scale in candidates:
        rows = [rollout(sim, profile, seed, residual, gains, spec,
                        lookahead, hold_steps, scale)
                for profile in profiles for seed in spec["training"]["seeds"]]
        harbor = [row for row in rows if row["profile"] == "harbor"]
        chop = [row for row in rows if row["profile"] == "chop"]
        harbor_passes = sum(passes_gate(row, spec["success"]) for row in harbor)
        chop_passes = sum(passes_gate(row, spec["success"]) for row in chop)
        key = (harbor_passes, chop_passes,
               min(row["survival_time_s"] for row in chop),
               float(np.mean([row["survival_time_s"] for row in chop])))
        evaluated.append({
            "lookahead_s": lookahead, "maximum_hold_steps": hold_steps,
            "command_scale": scale, "harbor_gates_passed": harbor_passes,
            "chop_gates_passed": chop_passes,
            "worst_chop_survival_s": key[2], "mean_chop_survival_s": round(key[3], 4),
        })
        if best_key is None or key > best_key:
            best_key, best_rows = key, rows
    assert best_key is not None and best_rows is not None
    best_summary = max(evaluated, key=lambda row: (
        row["harbor_gates_passed"], row["chop_gates_passed"],
        row["worst_chop_survival_s"], row["mean_chop_survival_s"]))
    attempt6_chop = [row for row in saved["training_evaluations"]
                     if row["profile"] == "chop"]
    attempt6_worst = min(row["survival_time_s"] for row in attempt6_chop)
    attempt6_mean = float(np.mean([row["survival_time_s"] for row in attempt6_chop]))
    result = {
        "challenge": spec["id"], "attempt": 8,
        "architecture": "frozen walk policy + contact-aware command latching and lookahead",
        "held_out_profile_touched": False,
        "candidate_count": len(candidates),
        "best_parameters": {k: best_summary[k] for k in
                            ("lookahead_s", "maximum_hold_steps", "command_scale")},
        "best_gate_summary": {k: best_summary[k] for k in
                              ("harbor_gates_passed", "chop_gates_passed",
                               "worst_chop_survival_s", "mean_chop_survival_s")},
        "comparison_to_attempt_6": {
            "attempt_6_worst_chop_survival_s": attempt6_worst,
            "attempt_6_mean_chop_survival_s": round(attempt6_mean, 4),
            "attempt_8_worst_chop_survival_delta_s": round(
                best_summary["worst_chop_survival_s"] - attempt6_worst, 4),
            "attempt_8_mean_chop_survival_delta_s": round(
                best_summary["mean_chop_survival_s"] - attempt6_mean, 4),
        },
        "candidates": evaluated,
        "training_evaluations": best_rows,
        "training_success": all(passes_gate(row, spec["success"]) for row in best_rows),
        "accepted": bool(
            best_summary["harbor_gates_passed"] == 3
            and best_summary["worst_chop_survival_s"] > attempt6_worst),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["training_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
