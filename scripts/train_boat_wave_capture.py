#!/usr/bin/env python3
"""Attempt 7: learn anticipatory axis-specific deck-capture commands.

The official walking network and attempt-6 joint residual remain frozen. Search
changes only eight high-level command gains: position, velocity, and wave-state
feed-forward for the longitudinal and lateral axes. Harbor and chop are used for
training; surge remains held out.
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
from train_boat_joint_walk import passes_gate  # noqa: E402
from train_boat_recenter import rollout  # noqa: E402


SPEC_PATH = ROOT / "challenges" / "012-variable-speed-boat-balance" / "spec.json"
START_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "joint-walk-refinement-result.json"
OUT_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "wave-capture-training-result.json"


def evaluate(sim: Microduck, profiles: list[dict], seeds: list[int],
             residual: np.ndarray, gains: np.ndarray,
             spec: dict) -> tuple[float, list[dict]]:
    rows = [
        rollout(sim, profile, seed, residual, gains, spec)
        for profile in profiles for seed in seeds
    ]
    harbor = [row for row in rows if row["profile"] == "harbor"]
    chop = [row for row in rows if row["profile"] == "chop"]
    harbor_passes = sum(passes_gate(row, spec["success"]) for row in harbor)
    chop_passes = sum(passes_gate(row, spec["success"]) for row in chop)
    survival = [row["survival_time_s"] for row in chop]
    drift = [row["maximum_relative_deck_displacement_m"] for row in chop]
    feasibility = 1200.0 if harbor_passes == 3 else 250.0 * harbor_passes
    score = (
        feasibility + 240.0 * chop_passes
        + 45.0 * min(survival) + 5.0 * float(np.mean(survival))
        - 10.0 * float(np.mean(drift)) - 0.01 * float(np.sum(gains ** 2))
    )
    return score, rows


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profile_map = {row["name"]: row for row in spec["environment"]["profiles"]}
    profiles = [profile_map[name] for name in spec["training"]["training_profiles"]]
    assert spec["training"]["held_out_profile"] not in {p["name"] for p in profiles}
    start = json.loads(START_PATH.read_text(encoding="utf-8"))
    residual = np.asarray(start["residual_weights"], dtype=float)
    gains0 = np.asarray(start["recenter_gains"] + [0.0, 0.0, 0.0, 0.0], dtype=float)
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rng = np.random.default_rng(1211)
    mean = gains0.copy()
    sigma = np.array([0.12, 0.05, 0.12, 0.05, 0.25, 0.08, 0.25, 0.08])
    low = np.array([0.0, 0.0, 0.0, 0.0, -2.5, -1.0, -2.5, -1.0])
    high = np.array([3.0, 1.0, 3.0, 1.0, 2.5, 1.0, 2.5, 1.0])
    best = gains0.copy()
    best_score, _ = evaluate(
        sim, profiles, spec["training"]["seeds"], residual, best, spec)
    history = []

    for generation in range(10):
        population = np.clip(rng.normal(mean, sigma, size=(24, 8)), low, high)
        population[0] = best
        scored = []
        for candidate in population:
            score, _ = evaluate(
                sim, profiles, spec["training"]["seeds"], residual,
                candidate, spec)
            scored.append((score, candidate.copy()))
        scored.sort(key=lambda row: -row[0])
        if scored[0][0] > best_score:
            best_score, best = scored[0]
        elite = np.asarray([candidate for _, candidate in scored[:6]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.01)
        _, rows = evaluate(
            sim, profiles, spec["training"]["seeds"], residual, best, spec)
        chop = [row for row in rows if row["profile"] == "chop"]
        history.append({
            "generation": generation + 1,
            "harbor_gates_passed": sum(
                passes_gate(row, spec["success"])
                for row in rows if row["profile"] == "harbor"),
            "chop_gates_passed": sum(passes_gate(row, spec["success"]) for row in chop),
            "worst_chop_survival_s": min(row["survival_time_s"] for row in chop),
        })
        print(
            f"generation {generation + 1:02d}: "
            f"harbor={history[-1]['harbor_gates_passed']}/3 "
            f"chop={history[-1]['chop_gates_passed']}/3 "
            f"worst_chop={history[-1]['worst_chop_survival_s']:0.2f}s",
            flush=True,
        )

    best = np.round(best, 12)
    final_score, rows = evaluate(
        sim, profiles, spec["training"]["seeds"], residual, best, spec)
    result = {
        "challenge": spec["id"],
        "attempt": 7,
        "architecture": "frozen walk policy and residual + learned axis-specific wave capture",
        "held_out_profile_touched": False,
        "serialized_parameter_decimals": 12,
        "evaluated_after_serialization": True,
        "training_seed": 1211,
        "residual_weights_frozen_from_attempt_6": True,
        "capture_gain_names": [
            "x_position", "x_velocity", "y_position", "y_velocity",
            "apparent_pitch", "apparent_pitch_rate", "roll", "roll_rate"],
        "capture_gains": [float(value) for value in best],
        "history": history,
        "training_score": round(final_score, 5),
        "training_evaluations": rows,
        "training_success": all(passes_gate(row, spec["success"]) for row in rows),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["training_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
