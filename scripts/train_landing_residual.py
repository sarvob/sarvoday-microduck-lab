#!/usr/bin/env python3
"""Train a small residual controller for the assisted Go1 landing."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_assisted_launch import contact_state, make_runtime, reset  # noqa: E402
from duck import (  # noqa: E402
    ACTION_SCALE,
    CMD_SIZE,
    DECIMATION,
    DEFAULT_POSE,
    HEAD_ALPHA,
    NUM_JOINTS,
    OBS_SIZE,
)


OUT = ROOT / "artifacts" / "005-duck-quadruped-jump"
SEEDS = (17, 71, 173)
BASE_LAUNCH = np.array([1.2, 0.0, 3.3])
CONTROL_DT = 0.02


def residual_vector(values: np.ndarray) -> np.ndarray:
    """Map four symmetric stance terms onto the 14 deployment actions."""
    hip, knee, ankle, width = values
    residual = np.zeros(NUM_JOINTS, dtype=np.float32)
    residual[2], residual[11] = hip, -hip
    residual[3], residual[12] = knee, -knee
    residual[4], residual[13] = ankle, -ankle
    residual[1], residual[10] = -width, width
    return residual


def control_step(sim, residual: np.ndarray) -> None:
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


def launch_for_seed(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return BASE_LAUNCH + np.array([
        rng.uniform(-0.035, 0.035),
        rng.uniform(-0.02, 0.02),
        rng.uniform(-0.045, 0.045),
    ])


def rollout(sim, weights: np.ndarray, seed: int) -> dict:
    reset(sim)
    sim.data.qvel[0:3] = launch_for_seed(seed)
    flight_residual = residual_vector(weights[:4])
    landed_residual = residual_vector(weights[4:])
    first_contact = None
    longest_hold = current_hold = 0.0
    touched_both = False
    ground_after_contact = False
    upright_sum = 0.0
    centered_sum = 0.0
    stable_steps = 0
    target = sim.data.site_xpos[sim.model.site("go1_landing_target").id].copy()

    for step in range(125):
        left, right, ground = contact_state(sim)
        if (left or right) and first_contact is None:
            first_contact = step * CONTROL_DT
        control_step(sim, landed_residual if first_contact is not None else flight_residual)
        left, right, ground = contact_state(sim)
        touched_both |= left and right
        if first_contact is not None and ground:
            ground_after_contact = True
        upright = max(0.0, -float(sim.proj_gravity()[2]))
        centered = float(np.exp(-np.sum((sim.data.xpos[sim.trunk][:2] - target[:2]) ** 2) / 0.05 ** 2))
        stable = left and right and upright >= 0.75 and not ground
        current_hold = current_hold + CONTROL_DT if stable else 0.0
        longest_hold = max(longest_hold, current_hold)
        if first_contact is not None:
            upright_sum += upright
            centered_sum += centered
            stable_steps += 1

    mean_upright = upright_sum / max(stable_steps, 1)
    mean_centered = centered_sum / max(stable_steps, 1)
    score = (
        14.0 * longest_hold
        + 1.5 * float(touched_both)
        + 1.2 * mean_upright
        + 0.8 * mean_centered
        - 3.0 * float(ground_after_contact)
        - 0.06 * float(np.sum(weights ** 2))
    )
    return {
        "seed": seed,
        "launch_velocity_mps": [round(float(x), 4) for x in launch_for_seed(seed)],
        "both_feet_simultaneous": touched_both,
        "longest_hold_s": round(longest_hold, 3),
        "ground_contact_after_pad": ground_after_contact,
        "mean_upright_score": round(mean_upright, 4),
        "mean_centered_score": round(mean_centered, 4),
        "score": round(score, 5),
    }


def evaluate(sim, weights: np.ndarray) -> tuple[float, list[dict]]:
    rows = [rollout(sim, weights, seed) for seed in SEEDS]
    mean = float(np.mean([row["score"] for row in rows]))
    worst_hold = min(row["longest_hold_s"] for row in rows)
    robust_score = mean + 8.0 * worst_hold
    return robust_score, rows


def train() -> dict:
    sim = make_runtime()
    rng = np.random.default_rng(505)
    mean = np.zeros(8)
    sigma = np.full(8, 0.22)
    best_score = -float("inf")
    best_weights = mean.copy()
    history = []

    for generation in range(16):
        population = np.clip(rng.normal(mean, sigma, size=(40, 8)), -0.75, 0.75)
        scored = []
        for candidate in population:
            score, _ = evaluate(sim, candidate)
            scored.append((score, candidate.copy()))
        scored.sort(key=lambda row: -row[0])
        if scored[0][0] > best_score:
            best_score, best_weights = scored[0]
        elite = np.asarray([weights for _, weights in scored[:8]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.025)
        history.append(round(best_score, 5))
        print(f"generation {generation + 1:02d}: {best_score:.4f}")

    score, evaluations = evaluate(sim, best_weights)
    success = all(
        row["both_feet_simultaneous"]
        and row["longest_hold_s"] >= 1.5
        and not row["ground_contact_after_pad"]
        for row in evaluations
    )
    return {
        "algorithm": "cross-entropy search over an 8-parameter stand-policy residual",
        "training_seed": 505,
        "evaluation_seeds": list(SEEDS),
        "weights": [round(float(x), 7) for x in best_weights],
        "score": round(score, 5),
        "history": history,
        "evaluations": evaluations,
        "success": success,
    }


if __name__ == "__main__":
    result = train()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "landing-policy.json").write_text(
        json.dumps({"weights": result["weights"]}, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "landing-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
