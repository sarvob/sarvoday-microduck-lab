#!/usr/bin/env python3
"""Train an eight-gain residual for Microduck boat-deck balance."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duck import (  # noqa: E402
    ACTION_SCALE,
    CMD_SIZE,
    CTRL_DT,
    DECIMATION,
    DEFAULT_POSE,
    HEAD_ALPHA,
    Microduck,
    NUM_JOINTS,
    OBS_SIZE,
)
from evaluate_boat_balance import (  # noqa: E402
    DECK_HALF,
    add_boat_deck,
    contact_flags,
    motion,
    set_deck_state,
)


SPEC_PATH = ROOT / "challenges" / "012-variable-speed-boat-balance" / "spec.json"
OUT = ROOT / "artifacts" / "012-variable-speed-boat-balance"


def residual_vector(weights: np.ndarray, roll: float, apparent_pitch: float,
                    roll_rate: float, apparent_pitch_rate: float,
                    relative_position: np.ndarray, relative_velocity: np.ndarray,
                    limit: float) -> np.ndarray:
    """Map physical deck state onto mirrored leg-joint corrections."""
    roll_command = (
        weights[0] * roll + weights[1] * roll_rate
        + weights[2] * relative_position[1]
        + weights[3] * relative_velocity[1]
    )
    pitch_command = (
        weights[4] * apparent_pitch + weights[5] * apparent_pitch_rate
        + weights[6] * relative_position[0]
        + weights[7] * relative_velocity[0]
    )
    residual = np.zeros(NUM_JOINTS, dtype=np.float32)
    residual[1] = residual[10] = roll_command
    residual[2], residual[11] = pitch_command, -pitch_command
    residual[3], residual[12] = -0.65 * pitch_command, 0.65 * pitch_command
    residual[4], residual[13] = -0.8 * pitch_command, 0.8 * pitch_command
    return np.clip(residual, -limit, limit)


def control_step(sim: Microduck, residual: np.ndarray) -> None:
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    offset = 0
    obs[offset:offset + 3] = sim.data.sensordata[sim.gyro:sim.gyro + 3]
    offset += 3
    obs[offset:offset + 3] = sim.proj_gravity()
    offset += 3
    obs[offset:offset + NUM_JOINTS] = sim.data.qpos[sim.qadr] - DEFAULT_POSE
    offset += NUM_JOINTS
    obs[offset:offset + NUM_JOINTS] = sim.data.qvel[sim.dadr]
    offset += NUM_JOINTS
    obs[offset:offset + NUM_JOINTS] = sim.last_action
    offset += NUM_JOINTS
    sim.head_smooth += HEAD_ALPHA * (sim.head_target - sim.head_smooth)
    command = np.zeros(CMD_SIZE, dtype=np.float32)
    command[3:7] = sim.head_smooth
    obs[offset:offset + CMD_SIZE] = command
    base = sim.sessions["stand"].run(None, {"obs": obs.reshape(1, -1)})[0][0]
    action = np.clip(base + residual, -1.5, 1.5).astype(np.float32)
    sim.last_action = action
    sim.data.ctrl[:NUM_JOINTS] = DEFAULT_POSE + action * ACTION_SCALE
    for _ in range(DECIMATION):
        mujoco.mj_step(sim.model, sim.data)


def rollout(sim: Microduck, profile: dict, seed: int, weights: np.ndarray,
            spec: dict, stop_on_failure: bool = True) -> dict:
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
    previous_roll = previous_pitch = 0.0
    previous_speed = 0.0
    previous_apparent_pitch = 0.0
    contact_steps = 0
    minimum_upright = 1.0
    maximum_drift = 0.0
    completed_steps = 0
    failed = False

    for step in range(total_steps):
        t = step * CTRL_DT
        speed, roll, pitch = motion(
            profile, t, spec["environment"]["motion_ramp_s"])
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
            weights, roll, apparent_pitch, roll_rate, apparent_pitch_rate,
            relative_position, relative_velocity, limit)
        control_step(sim, residual)
        previous_roll, previous_pitch = roll, pitch
        previous_speed = speed
        previous_apparent_pitch = apparent_pitch
        deck_contact, floor_contact = contact_flags(
            sim, deck_geom, floor_geom, feet)
        contact_steps += int(deck_contact)
        relative = sim.data.xpos[sim.trunk, :2] - sim.data.qpos[boat_q:boat_q + 2]
        drift = float(np.linalg.norm(relative - start_relative))
        maximum_drift = max(maximum_drift, drift)
        upright = -float(sim.proj_gravity()[2])
        minimum_upright = min(minimum_upright, upright)
        inside = (
            abs(float(relative[0])) <= DECK_HALF[0] - 0.12
            and abs(float(relative[1])) <= DECK_HALF[1] - 0.12
        )
        completed_steps = step + 1
        failed = floor_contact or not inside
        if failed and stop_on_failure:
            break

    survival = completed_steps * CTRL_DT
    contact_ratio = contact_steps / max(completed_steps, 1)
    score = (
        8.0 * survival
        + 18.0 * contact_ratio
        + 4.0 * max(minimum_upright, -1.0)
        - 6.0 * maximum_drift
        - 0.04 * float(np.sum(weights ** 2))
    )
    return {
        "profile": profile["name"],
        "seed": seed,
        "survival_time_s": round(survival, 3),
        "deck_contact_ratio": round(contact_ratio, 4),
        "minimum_upright_score": round(minimum_upright, 4),
        "maximum_relative_deck_displacement_m": round(maximum_drift, 4),
        "failed": failed,
        "score": round(score, 5),
    }


def evaluate(sim: Microduck, profiles: list[dict], seeds: list[int],
             weights: np.ndarray, spec: dict) -> tuple[float, list[dict]]:
    rows = [
        rollout(sim, profile, seed, weights, spec)
        for profile in profiles
        for seed in seeds
    ]
    worst_survival = min(row["survival_time_s"] for row in rows)
    score = float(np.mean([row["score"] for row in rows])) + 12.0 * worst_survival
    return score, rows


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profile_map = {row["name"]: row for row in spec["environment"]["profiles"]}
    training_profiles = [profile_map[name] for name in spec["training"]["training_profiles"]]
    seeds = spec["training"]["seeds"]
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rng = np.random.default_rng(1205)
    mean = np.zeros(8)
    sigma = np.full(8, 0.6)
    best_weights = mean.copy()
    best_score = -float("inf")
    history = []

    for generation in range(14):
        population = np.clip(rng.normal(mean, sigma, size=(32, 8)), -2.5, 2.5)
        population[0] = best_weights
        scored = []
        for candidate in population:
            score, _ = evaluate(sim, training_profiles, seeds, candidate, spec)
            scored.append((score, candidate.copy()))
        scored.sort(key=lambda row: -row[0])
        if scored[0][0] > best_score:
            best_score, best_weights = scored[0]
        elite = np.asarray([weights for _, weights in scored[:8]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.04)
        _, rows = evaluate(sim, training_profiles, seeds, best_weights, spec)
        history.append({
            "generation": generation + 1,
            "best_score": round(best_score, 5),
            "worst_training_survival_s": min(row["survival_time_s"] for row in rows),
        })
        print(
            f"generation {generation + 1:02d}: score={best_score:0.3f} "
            f"worst={history[-1]['worst_training_survival_s']:0.2f}s",
            flush=True,
        )

    training_score, training_rows = evaluate(
        sim, training_profiles, seeds, best_weights, spec)
    held_out_profile = profile_map[spec["training"]["held_out_profile"]]
    held_out_rows = [
        rollout(sim, held_out_profile, seed, best_weights, spec)
        for seed in seeds
    ]
    result = {
        "challenge": spec["id"],
        "algorithm": spec["training"]["algorithm"],
        "training_seed": 1205,
        "weights": [round(float(value), 7) for value in best_weights],
        "maximum_joint_residual_rad": spec["training"]["maximum_joint_residual_rad"],
        "training_score": round(training_score, 5),
        "history": history,
        "training_evaluations": training_rows,
        "held_out_evaluations": held_out_rows,
        "success": all(
            not row["failed"] and row["survival_time_s"] >= 20.0
            for row in training_rows + held_out_rows
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "residual-policy.json").write_text(json.dumps({
        "learned_component": "eight-gain deck-attitude residual",
        "weights": result["weights"],
        "maximum_joint_residual_rad": result["maximum_joint_residual_rad"],
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "training-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
