#!/usr/bin/env python3
"""Diagnose the exact training-profile failures of boat attempt 6."""

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
POLICY_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "joint-walk-refinement-result.json"
OUT_PATH = ROOT / "artifacts" / "012-variable-speed-boat-balance" / "attempt-6-chop-diagnosis.json"


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    saved = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    profile = next(
        row for row in spec["environment"]["profiles"] if row["name"] == "chop")
    assert spec["training"]["held_out_profile"] != profile["name"]
    residual = np.asarray(saved["residual_weights"], dtype=float)
    recenter = np.asarray(saved["recenter_gains"], dtype=float)
    sim = Microduck(render=False, xml_transform=add_boat_deck)
    rows = [
        rollout(sim, profile, seed, residual, recenter, spec)
        for seed in spec["training"]["seeds"]
    ]
    reason_counts = {
        reason: sum(row["failure_reason"] == reason for row in rows)
        for reason in sorted({row["failure_reason"] for row in rows})
    }
    result = {
        "challenge": spec["id"],
        "controller_attempt": 6,
        "profile": "chop",
        "held_out_profile_touched": False,
        "failure_reason_counts": reason_counts,
        "evaluations": rows,
        "finding": (
            "Attempt 6 loses the chop runs through deck-bound exits rather than "
            "direct foot-floor contact; attempt 7 should target lateral and "
            "longitudinal capture before increasing attitude-gain complexity."
            if set(reason_counts) == {"deck_exit"}
            else "Failure modes are mixed; inspect the per-seed reasons before attempt 7."
        ),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
