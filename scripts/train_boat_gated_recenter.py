#!/usr/bin/env python3
"""Attempt 3: search when the frozen walk policy may recenter Microduck.

The official stand/walk networks and attempt-1 joint residual stay frozen. The
only newly learned values are four deck-relative PD gains and one drift gate.
Training uses harbor and chop; the declared surge profile remains untouched.
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
OUT_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "gated-recenter-training-result.json"


def evaluate(sim: Microduck, profiles: list[dict], seeds: list[int],
             residual_weights: np.ndarray, parameters: np.ndarray,
             spec: dict) -> tuple[float, list[dict]]:
    gains, threshold = parameters[:4], float(parameters[4])
    rows = [
        rollout(sim, profile, seed, residual_weights, gains, spec,
                walk_threshold_m=threshold)
        for profile in profiles for seed in seeds
    ]
    worst_survival = min(row["survival_time_s"] for row in rows)
    switch_penalty = 0.02 * sum(row["policy_switches"] for row in rows)
    score = (
        float(np.mean([row["score"] for row in rows]))
        + 12.0 * worst_survival - switch_penalty
    )
    return score, rows


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profile_map = {row["name"]: row for row in spec["environment"]["profiles"]}
    profiles = [profile_map[name] for name in spec["training"]["training_profiles"]]
    assert spec["training"]["held_out_profile"] not in {p["name"] for p in profiles}
    residual_weights = np.asarray(
        json.loads(ATTEMPT1_PATH.read_text(encoding="utf-8"))["weights"])
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rng = np.random.default_rng(1207)
    mean = np.array([1.4, 0.3, 1.2, 0.2, 0.16])
    sigma = np.array([0.45, 0.12, 0.45, 0.12, 0.05])
    low = np.array([0.0, 0.0, 0.0, 0.0, 0.06])
    high = np.array([3.0, 1.0, 3.0, 1.0, 0.30])
    best = mean.copy()
    best_score = -float("inf")
    history = []

    for generation in range(7):
        population = np.clip(rng.normal(mean, sigma, size=(18, 5)), low, high)
        population[0] = best
        scored = []
        for candidate in population:
            score, _ = evaluate(
                sim, profiles, spec["training"]["seeds"], residual_weights,
                candidate, spec)
            scored.append((score, candidate.copy()))
        scored.sort(key=lambda row: -row[0])
        if scored[0][0] > best_score:
            best_score, best = scored[0]
        elite = np.asarray([parameters for _, parameters in scored[:5]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), [0.03, 0.02, 0.03, 0.02, 0.01])
        _, rows = evaluate(
            sim, profiles, spec["training"]["seeds"], residual_weights, best, spec)
        history.append({
            "generation": generation + 1,
            "best_score": round(best_score, 5),
            "worst_training_survival_s": min(row["survival_time_s"] for row in rows),
            "walk_threshold_m": round(float(best[4]), 5),
        })
        print(f"generation {generation + 1:02d}: "
              f"worst={history[-1]['worst_training_survival_s']:0.2f}s "
              f"gate={best[4]:0.3f}m", flush=True)

    training_score, rows = evaluate(
        sim, profiles, spec["training"]["seeds"], residual_weights, best, spec)
    result = {
        "challenge": spec["id"],
        "attempt": 3,
        "architecture": "drift-gated frozen walk policy + frozen attempt-1 residual",
        "held_out_profile_touched": False,
        "training_seed": 1207,
        "recenter_gains": [round(float(value), 7) for value in best[:4]],
        "walk_threshold_m": round(float(best[4]), 7),
        "walk_release_threshold_m": round(float(0.6 * best[4]), 7),
        "history": history,
        "training_score": round(training_score, 5),
        "training_evaluations": rows,
        "training_success": all(
            not row["failed"] and row["survival_time_s"] >= 20.0 for row in rows),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["training_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
