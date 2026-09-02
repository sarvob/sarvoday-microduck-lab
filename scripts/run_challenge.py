#!/usr/bin/env python3
"""Train and evaluate one reproducible Microduck challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duck as D
import lesson as L


def success(record: dict, gate: dict) -> bool:
    checks = []
    if "minimum_turns" in gate:
        turns = float(record["spun_rad"]) / (2.0 * np.pi)
        checks.append(turns >= float(gate["minimum_turns"]))
    if "maximum_drift_m" in gate:
        checks.append(float(record["travelled"]) <= float(gate["maximum_drift_m"]))
    if "minimum_markers" in gate:
        checks.append(len(record.get("reached", [])) >= int(gate["minimum_markers"]))
    if "maximum_time_s" in gate:
        checks.append(float(record["t_end"]) <= float(gate["maximum_time_s"]))
    if gate.get("must_stay_upright", True):
        checks.append(not record["fell"])
    return bool(checks) and all(checks)


def serializable_rollout(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in {"frames", "path"}}


def plot_training(histories: list[dict], destination: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    for attempt in histories:
        history = attempt["history"]
        generations = np.arange(1, len(history) + 1)
        ax.plot(generations, [row[0] for row in history], marker="o", ms=3,
                label=f"seed {attempt['seed']} best")
    ax.set(title=title, xlabel="generation", ylabel="score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--all-seeds", action="store_true",
                        help="continue training after the first passing seed")
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    config = json.loads(spec_path.read_text(encoding="utf-8"))
    lesson_spec, notes = L.parse_spec(json.dumps(config["lesson"]))
    training = config["training"]
    gate = config["success"]
    output = ROOT / "artifacts" / config["id"]
    output.mkdir(parents=True, exist_ok=True)

    sim = D.Microduck(render=False)
    attempts: list[dict] = []
    winner = None

    for seed in training["seeds"]:
        print(f"Training seed {seed}...")
        weights, history = L.train(
            sim,
            lesson_spec,
            generations=int(training["generations"]),
            population=int(training["population"]),
            elite=int(training["elite"]),
            seed=int(seed),
            on_generation=lambda generation, row: print(
                f"  generation {generation + 1:02d}: best={row[0]:+.3f} mean={row[1]:+.3f}"
            ),
        )
        evaluation = L.rollout(sim, weights, lesson_spec, trace=True)
        passed = success(evaluation, gate)
        attempt = {
            "seed": seed,
            "passed": passed,
            "history": history,
            "evaluation": serializable_rollout(evaluation),
        }
        attempts.append(attempt)
        print("  PASS" if passed else "  FAIL", "—", evaluation["headline"])
        if passed and winner is None:
            winner = {
                "seed": seed,
                "weights": [round(float(value), 6) for value in weights],
                "evaluation": serializable_rollout(evaluation),
            }
            if not args.all_seeds:
                break

    result = {
        "challenge": config["id"],
        "passed": winner is not None,
        "source_spec": str(spec_path.relative_to(ROOT)),
        "lesson": lesson_spec,
        "success_gate": gate,
        "notes": notes,
        "attempts": attempts,
        "winner": winner,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if winner:
        (output / "policy.json").write_text(
            json.dumps({"challenge": config["id"], **winner}, indent=2) + "\n",
            encoding="utf-8",
        )
    plot_training(attempts, output / "learning-curve.png", config["title"])
    print(f"Result: {'PASS' if winner else 'FAIL'}")
    print(f"Artifacts: {output.relative_to(ROOT)}")
    return 0 if winner else 1


if __name__ == "__main__":
    raise SystemExit(main())
