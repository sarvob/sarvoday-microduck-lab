#!/usr/bin/env python3
"""Fetch and validate the pinned Unitree Go1 MuJoCo landing platform."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import mujoco


ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_REVISION = "e4049d0a3bfd58d2a3081614e6777d4007e3f86a"
CACHE = ROOT / ".microduck_cache" / "mujoco_menagerie" / MENAGERIE_REVISION
GO1 = CACHE / "unitree_go1"
GO1_FILES = (
    "LICENSE",
    "README.md",
    "go1.xml",
    "scene.xml",
    "assets/trunk.stl",
    "assets/hip.stl",
    "assets/thigh.stl",
    "assets/thigh_mirror.stl",
    "assets/calf.stl",
)


def ensure_go1() -> Path:
    scene = GO1 / "scene.xml"
    if scene.exists() and (GO1 / "LICENSE").exists():
        return scene

    for relative in GO1_FILES:
        destination = GO1 / relative
        if destination.exists() and destination.stat().st_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://raw.githubusercontent.com/google-deepmind/"
            f"mujoco_menagerie/{MENAGERIE_REVISION}/unitree_go1/{relative}"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            destination.write_bytes(response.read())
    return scene


def validate(scene: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene))
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id < 0:
        raise RuntimeError("Unitree Go1 model is missing its trunk body")
    if model.nu != 12:
        raise RuntimeError(f"Expected 12 Go1 actuators, found {model.nu}")
    return {
        "model": "Unitree Go1",
        "source_revision": MENAGERIE_REVISION,
        "license": "BSD-3-Clause",
        "actuators": model.nu,
        "bodies": model.nbody,
        "joints": model.njnt,
        "scene": str(scene.relative_to(ROOT)),
    }


if __name__ == "__main__":
    print(json.dumps(validate(ensure_go1()), indent=2))
