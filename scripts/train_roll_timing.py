#!/usr/bin/env python3
"""Tune and evaluate the handoff from Microduck's roulade to stand policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from duck import Microduck  # noqa: E402


SPEC_PATH = ROOT / "challenges" / "006-controlled-roll" / "spec.json"
OUT = ROOT / "artifacts" / "006-controlled-roll"


def quaternion_step_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest orientation change in radians, robust to quaternion sign flips."""
    similarity = np.clip(abs(float(np.dot(q0, q1))), 0.0, 1.0)
    return 2.0 * math.acos(similarity)


def rollout(sim: Microduck, roll_steps: int, recovery_steps: int, seed: int,
            initial_joint_noise_rad: float) -> dict:
    sim.reset()
    rng = np.random.default_rng(seed)
    sim.data.qpos[sim.qadr] += rng.normal(
        0.0, initial_joint_noise_rad, len(sim.qadr))
    sim.mj.mj_forward(sim.model, sim.data)

    for _ in range(50):
        sim.control_step("stand", [0.0, 0.0, 0.0])

    start_xy = sim.data.xpos[sim.trunk, :2].copy()
    previous_q = sim.data.body(sim.trunk).xquat.copy()
    cumulative_rotation = 0.0
    minimum_upright = 1.0
    final_upright_samples: list[float] = []

    for step in range(roll_steps + recovery_steps):
        mode = "roll" if step < roll_steps else "stand"
        sim.control_step(mode, [0.0, 0.0, 0.0])
        current_q = sim.data.body(sim.trunk).xquat.copy()
        cumulative_rotation += quaternion_step_angle(previous_q, current_q)
        previous_q = current_q
        upright = -float(sim.proj_gravity()[2])
        minimum_upright = min(minimum_upright, upright)
        if step >= roll_steps + recovery_steps - 25:
            final_upright_samples.append(upright)

    displacement = float(np.linalg.norm(sim.data.xpos[sim.trunk, :2] - start_xy))
    return {
        "seed": seed,
        "roll_steps": roll_steps,
        "roll_duration_s": round(roll_steps / 50.0, 3),
        "cumulative_body_rotation_deg": round(math.degrees(cumulative_rotation), 3),
        "minimum_upright_score": round(minimum_upright, 4),
        "final_upright_score": round(float(np.mean(final_upright_samples)), 4),
        "horizontal_displacement_m": round(displacement, 4),
    }


def passes(row: dict, gate: dict) -> bool:
    return (
        row["cumulative_body_rotation_deg"]
        >= gate["minimum_cumulative_body_rotation_deg"]
        and row["minimum_upright_score"]
        <= gate["maximum_inverted_upright_score"]
        and row["final_upright_score"] >= gate["minimum_final_upright_score"]
        and row["horizontal_displacement_m"]
        <= gate["maximum_horizontal_displacement_m"]
    )


def objective(rows: list[dict], gate: dict) -> tuple:
    passed = sum(passes(row, gate) for row in rows)
    worst_displacement = max(row["horizontal_displacement_m"] for row in rows)
    worst_final_upright = min(row["final_upright_score"] for row in rows)
    return passed, -worst_displacement, worst_final_upright


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    training, gate = spec["training"], spec["success"]
    duration = training["roll_duration_steps"]
    seeds = training["seeds"]
    sim = Microduck(render=False)

    candidates = []
    for roll_steps in range(duration["minimum"], duration["maximum"] + 1,
                            duration["increment"]):
        evaluations = [
            rollout(sim, roll_steps, training["recovery_duration_steps"], seed,
                    training["initial_joint_noise_rad"])
            for seed in seeds
        ]
        candidates.append({
            "roll_steps": roll_steps,
            "evaluations": evaluations,
            "passing_seeds": sum(passes(row, gate) for row in evaluations),
        })

    best = max(candidates, key=lambda row: objective(row["evaluations"], gate))
    success = best["passing_seeds"] >= gate["minimum_successful_seeds"]
    result = {
        "challenge": spec["id"],
        "algorithm": training["algorithm"],
        "candidate_count": len(candidates),
        "success_gate": gate,
        "success": success,
        "best": best,
        "candidates": candidates,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if success:
        policy = {
            "challenge": spec["id"],
            "learned_component": "roll-to-stand handoff timing",
            "roll_steps": best["roll_steps"],
            "roll_duration_s": round(best["roll_steps"] / 50.0, 3),
            "evaluation_seeds": seeds,
        }
        (OUT / "policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "success": success,
        "candidate_count": len(candidates),
        "best_roll_steps": best["roll_steps"],
        "passing_seeds": best["passing_seeds"],
        "evaluations": best["evaluations"],
    }, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
