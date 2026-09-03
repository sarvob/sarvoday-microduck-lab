#!/usr/bin/env python3
"""Re-evaluate serialized boat candidates exactly as public artifacts store them."""

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
OUT_DIR = ROOT / "artifacts" / "012-variable-speed-boat-balance"
CANDIDATES = (
    (4, OUT_DIR / "joint-walk-training-result.json"),
    (5, OUT_DIR / "joint-walk-continuation-result.json"),
)


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profile_map = {row["name"]: row for row in spec["environment"]["profiles"]}
    profiles = [profile_map[name] for name in spec["training"]["training_profiles"]]
    assert spec["training"]["held_out_profile"] not in {p["name"] for p in profiles}
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    candidates = []
    for attempt, path in CANDIDATES:
        saved = json.loads(path.read_text(encoding="utf-8"))
        residual = np.asarray(saved["residual_weights"], dtype=float)
        recenter = np.asarray(saved["recenter_gains"], dtype=float)
        rows = [
            rollout(sim, profile, seed, residual, recenter, spec)
            for profile in profiles for seed in spec["training"]["seeds"]
        ]
        candidates.append({
            "attempt": attempt,
            "source": str(path.relative_to(ROOT)),
            "evaluated_serialized_parameters": True,
            "held_out_profile_touched": False,
            "training_evaluations": rows,
            "training_gates_passed": sum(
                passes_gate(row, spec["success"]) for row in rows),
            "worst_chop_survival_s": min(
                row["survival_time_s"] for row in rows
                if row["profile"] == "chop"),
        })
    result = {
        "challenge": spec["id"],
        "purpose": "reproducibility check of rounded parameters in public artifacts",
        "held_out_profile_touched": False,
        "candidates": candidates,
    }
    output = OUT_DIR / "candidate-reverification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
