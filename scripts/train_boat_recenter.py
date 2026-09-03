#!/usr/bin/env python3
"""Search a disclosed deck-relative recenter controller on harbor and chop.

The official Microduck stand and walk networks remain frozen.  Attempt 2 keeps
the learned stance residual from attempt 1 and searches only four high-level
gains that turn deck-relative position and velocity into a walking command.
The held-out surge profile is intentionally never loaded or evaluated here.
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
from evaluate_boat_balance import (  # noqa: E402
    DECK_HALF,
    add_boat_deck,
    contact_flags,
    motion,
    set_deck_state,
)
from train_boat_residual import control_step, residual_vector  # noqa: E402


SPEC_PATH = ROOT / "challenges" / "012-variable-speed-boat-balance" / "spec.json"
ATTEMPT1_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "training-result.json"
OUT = ROOT / "artifacts" / "012-variable-speed-boat-balance"


def recenter_command(gains: np.ndarray, position: np.ndarray,
                     velocity: np.ndarray) -> np.ndarray:
    """PD deck-centering command for the frozen walking policy."""
    command = np.array([
        -gains[0] * position[0] - gains[1] * velocity[0],
        -gains[2] * position[1] - gains[3] * velocity[1],
        0.0,
    ], dtype=np.float32)
    command[0] = np.clip(command[0], VEL_BACK, VEL_FWD)
    command[1] = np.clip(command[1], -0.2, 0.2)
    return command


def rollout(sim: Microduck, profile: dict, seed: int, residual_weights: np.ndarray,
            recenter_gains: np.ndarray, spec: dict,
            walk_threshold_m: float | None = None) -> dict:
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
    for _ in range(50):
        set_deck_state(sim, boat_q, boat_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sim.control_step("stand", [0.0, 0.0, 0.0])

    start_relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
    total_steps = round(spec["environment"]["evaluation_duration_s"] / CTRL_DT)
    limit = spec["training"]["maximum_joint_residual_rad"]
    deck_x = 0.0
    previous_roll = previous_pitch = previous_speed = previous_apparent_pitch = 0.0
    contact_steps = completed_steps = 0
    minimum_upright = 1.0
    maximum_drift = 0.0
    failed = False
    walking = walk_threshold_m is None
    walking_steps = 0
    policy_switches = 0

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
        residual = residual_vector(
            residual_weights, roll, apparent_pitch, roll_rate,
            apparent_pitch_rate, relative_position, relative_velocity, limit)
        command = recenter_command(recenter_gains, relative_position, relative_velocity)
        if walk_threshold_m is not None:
            drift_before_control = float(np.linalg.norm(relative_position))
            next_walking = (
                drift_before_control > walk_threshold_m
                if not walking
                else drift_before_control > 0.6 * walk_threshold_m
            )
            policy_switches += int(next_walking != walking)
            walking = next_walking
        walking_steps += int(walking)
        control_step(
            sim, residual, mode="walk" if walking else "stand",
            command3=command if walking else None)

        previous_roll, previous_pitch = roll, pitch
        previous_speed, previous_apparent_pitch = speed, apparent_pitch
        deck_contact, floor_contact = contact_flags(sim, deck_geom, floor_geom, feet)
        contact_steps += int(deck_contact)
        relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
        drift = float(np.linalg.norm(relative - start_relative))
        maximum_drift = max(maximum_drift, drift)
        minimum_upright = min(minimum_upright, -float(sim.proj_gravity()[2]))
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
    score = (
        8.0 * survival + 18.0 * contact_ratio
        + 4.0 * max(minimum_upright, -1.0) - 6.0 * maximum_drift
        - 0.02 * float(np.sum(recenter_gains ** 2))
    )
    return {
        "profile": profile["name"],
        "seed": seed,
        "survival_time_s": round(survival, 3),
        "deck_contact_ratio": round(contact_ratio, 4),
        "minimum_upright_score": round(minimum_upright, 4),
        "maximum_relative_deck_displacement_m": round(maximum_drift, 4),
        "failed": bool(failed),
        "walking_ratio": round(walking_steps / max(completed_steps, 1), 4),
        "policy_switches": policy_switches,
        "score": round(score, 5),
    }


def evaluate(sim: Microduck, profiles: list[dict], seeds: list[int],
             residual_weights: np.ndarray, gains: np.ndarray,
             spec: dict) -> tuple[float, list[dict]]:
    rows = [
        rollout(sim, profile, seed, residual_weights, gains, spec)
        for profile in profiles for seed in seeds
    ]
    worst_survival = min(row["survival_time_s"] for row in rows)
    return (
        float(np.mean([row["score"] for row in rows])) + 12.0 * worst_survival,
        rows,
    )


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profile_map = {row["name"]: row for row in spec["environment"]["profiles"]}
    profiles = [profile_map[name] for name in spec["training"]["training_profiles"]]
    assert spec["training"]["held_out_profile"] not in {p["name"] for p in profiles}
    residual_weights = np.asarray(
        json.loads(ATTEMPT1_PATH.read_text(encoding="utf-8"))["weights"])
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rng = np.random.default_rng(1206)
    mean = np.array([1.0, 0.15, 1.0, 0.15])
    sigma = np.array([0.5, 0.15, 0.5, 0.15])
    best_gains = mean.copy()
    best_score = -float("inf")
    history = []

    for generation in range(8):
        population = np.clip(rng.normal(mean, sigma, size=(20, 4)), 0.0, 3.0)
        population[0] = best_gains
        scored = []
        for candidate in population:
            score, _ = evaluate(
                sim, profiles, spec["training"]["seeds"], residual_weights,
                candidate, spec)
            scored.append((score, candidate.copy()))
        scored.sort(key=lambda row: -row[0])
        if scored[0][0] > best_score:
            best_score, best_gains = scored[0]
        elite = np.asarray([gains for _, gains in scored[:5]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.03)
        _, rows = evaluate(
            sim, profiles, spec["training"]["seeds"], residual_weights,
            best_gains, spec)
        history.append({
            "generation": generation + 1,
            "best_score": round(best_score, 5),
            "worst_training_survival_s": min(row["survival_time_s"] for row in rows),
        })
        print(f"generation {generation + 1:02d}: "
              f"worst={history[-1]['worst_training_survival_s']:0.2f}s", flush=True)

    training_score, rows = evaluate(
        sim, profiles, spec["training"]["seeds"], residual_weights,
        best_gains, spec)
    result = {
        "challenge": spec["id"],
        "attempt": 2,
        "architecture": "frozen walk policy + attempt-1 stance residual + learned recenter PD gains",
        "held_out_profile_touched": False,
        "training_seed": 1206,
        "recenter_gains": [round(float(value), 7) for value in best_gains],
        "residual_weights_frozen_from_attempt_1": True,
        "training_score": round(training_score, 5),
        "history": history,
        "training_evaluations": rows,
        "training_success": all(
            not row["failed"] and row["survival_time_s"] >= 20.0 for row in rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recenter-training-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["training_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
