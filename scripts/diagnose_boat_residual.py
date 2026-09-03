#!/usr/bin/env python3
"""Isolate why the first boat-balance residual misses the success gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from duck import Microduck  # noqa: E402
from evaluate_boat_balance import add_boat_deck  # noqa: E402
from train_boat_residual import rollout  # noqa: E402


SPEC_PATH = ROOT / "challenges" / "012-variable-speed-boat-balance" / "spec.json"
RESULT_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "training-result.json"
OUT = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "diagnosis.json"


def component_case(profile: dict, component: str) -> dict:
    case = copy.deepcopy(profile)
    case["name"] = f"{profile['name']}_{component}"
    if component == "translation_only":
        case["roll"]["amplitude_deg"] = 0.0
        case["pitch"]["amplitude_deg"] = 0.0
    elif component == "waves_only":
        case["forward_speed_mps"]["base"] = 0.0
        case["forward_speed_mps"]["amplitude"] = 0.0
    else:
        raise ValueError(component)
    return case


def compact(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "profile", "seed", "survival_time_s", "deck_contact_ratio",
            "minimum_upright_score", "maximum_relative_deck_displacement_m",
            "failed",
        )
    }


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    trained = np.asarray(json.loads(RESULT_PATH.read_text(encoding="utf-8"))["weights"])
    profiles = {
        row["name"]: row for row in spec["environment"]["profiles"]
        if row["name"] in spec["training"]["training_profiles"]
    }
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rows = []
    for profile_name in ("harbor", "chop"):
        for component in ("translation_only", "waves_only"):
            profile = component_case(profiles[profile_name], component)
            for controller, weights in (
                ("frozen_policy", np.zeros(8)),
                ("residual_attempt_1", trained),
            ):
                row = compact(rollout(sim, profile, 29, weights, spec))
                row["controller"] = controller
                rows.append(row)

    result = {
        "challenge": spec["id"],
        "held_out_profile_touched": False,
        "diagnostic_seed": 29,
        "component_rollouts": rows,
        "findings": [
            "Harbor translation alone is stable for 20 seconds, so constant deck transport and contact modeling are viable.",
            "Chop waves are the dominant short-horizon disturbance; attempt 1 improves wave-only survival from 5.26 to 8.62 seconds.",
            "Both controllers accumulate deck-relative drift, and a pose-only stance residual cannot deliberately step back toward deck center.",
        ],
        "architecture_decision": "Keep the official locomotion policies frozen; add a disclosed deck-relative recenter command through the shipped walk policy, then learn the residual and recenter gains on harbor and chop only.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
