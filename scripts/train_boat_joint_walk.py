#!/usr/bin/env python3
"""Attempt 4: jointly search walk-specific residual and recenter gains.

The official Microduck walking network remains frozen. Search is limited to an
eight-gain joint residual and a four-gain deck-centre velocity command. Harbor
and chop are the only training profiles; surge stays sealed until the declared
training gates pass.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duck import Microduck  # noqa: E402
from evaluate_boat_balance import add_boat_deck  # noqa: E402
from train_boat_recenter import rollout  # noqa: E402


SPEC_PATH = ROOT / "challenges" / "012-variable-speed-boat-balance" / "spec.json"
ATTEMPT1_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "training-result.json"
ATTEMPT2_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "recenter-training-result.json"
OUT_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "joint-walk-training-result.json"


def passes_gate(row: dict, gate: dict) -> bool:
    return bool(
        not row["failed"]
        and row["survival_time_s"] >= gate["minimum_duration_per_profile_s"]
        and row["minimum_upright_score"] >= gate["minimum_upright_score"]
        and row["final_upright_score"] >= gate["minimum_final_upright_score"]
        and row["deck_contact_ratio"] >= gate["minimum_deck_contact_ratio"]
        and row["maximum_relative_deck_displacement_m"]
        <= gate["maximum_relative_deck_displacement_m"]
    )


def evaluate(sim: Microduck, profiles: list[dict], seeds: list[int],
             parameters: np.ndarray, spec: dict) -> tuple[float, list[dict]]:
    residual, recenter = parameters[:8], parameters[8:]
    rows = [
        rollout(sim, profile, seed, residual, recenter, spec)
        for profile in profiles for seed in seeds
    ]
    worst_survival = min(row["survival_time_s"] for row in rows)
    gate_bonus = 15.0 * sum(passes_gate(row, spec["success"]) for row in rows)
    regularization = 0.03 * float(np.sum(parameters ** 2))
    score = (
        float(np.mean([row["score"] for row in rows]))
        + 14.0 * worst_survival + gate_bonus - regularization
    )
    return score, rows


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profile_map = {row["name"]: row for row in spec["environment"]["profiles"]}
    profiles = [profile_map[name] for name in spec["training"]["training_profiles"]]
    assert spec["training"]["held_out_profile"] not in {p["name"] for p in profiles}
    attempt1 = json.loads(ATTEMPT1_PATH.read_text(encoding="utf-8"))
    attempt2 = json.loads(ATTEMPT2_PATH.read_text(encoding="utf-8"))
    start = np.asarray(attempt1["weights"] + attempt2["recenter_gains"], dtype=float)
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rng = np.random.default_rng(1208)
    mean = start.copy()
    sigma = np.r_[np.full(8, 0.45), [0.35, 0.12, 0.35, 0.12]]
    low = np.r_[np.full(8, -2.5), np.zeros(4)]
    high = np.r_[np.full(8, 2.5), [3.0, 1.0, 3.0, 1.0]]
    best = start.copy()
    best_score = -float("inf")
    history = []

    for generation in range(8):
        population = np.clip(rng.normal(mean, sigma, size=(24, 12)), low, high)
        population[0] = best
        scored = []
        for candidate in population:
            score, _ = evaluate(
                sim, profiles, spec["training"]["seeds"], candidate, spec)
            scored.append((score, candidate.copy()))
        scored.sort(key=lambda row: -row[0])
        if scored[0][0] > best_score:
            best_score, best = scored[0]
        elite = np.asarray([parameters for _, parameters in scored[:6]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.03)
        _, rows = evaluate(sim, profiles, spec["training"]["seeds"], best, spec)
        history.append({
            "generation": generation + 1,
            "best_score": round(best_score, 5),
            "worst_training_survival_s": min(row["survival_time_s"] for row in rows),
            "training_gates_passed": sum(
                passes_gate(row, spec["success"]) for row in rows),
        })
        print(
            f"generation {generation + 1:02d}: "
            f"worst={history[-1]['worst_training_survival_s']:0.2f}s "
            f"gates={history[-1]['training_gates_passed']}/6",
            flush=True,
        )

    training_score, rows = evaluate(
        sim, profiles, spec["training"]["seeds"], best, spec)
    result = {
        "challenge": spec["id"],
        "attempt": 4,
        "architecture": "frozen walk policy + jointly learned residual and recenter gains",
        "held_out_profile_touched": False,
        "training_seed": 1208,
        "residual_weights": [round(float(value), 7) for value in best[:8]],
        "recenter_gains": [round(float(value), 7) for value in best[8:]],
        "history": history,
        "training_score": round(training_score, 5),
        "training_evaluations": rows,
        "training_success": all(passes_gate(row, spec["success"]) for row in rows),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["training_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
