#!/usr/bin/env python3
"""Assemble the real Microduck and pinned Unitree Go1 in one MuJoCo scene."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from duck import build_xml, ensure_assets
from validate_go1_platform import ensure_go1, validate


OUT = ROOT / "artifacts" / "005-duck-quadruped-jump"
PREFIX = "go1_"


def _prefix_go1(root: ET.Element, go1_dir: Path) -> None:
    """Namespace Go1 resources so they cannot collide with Microduck names."""
    references = {"class", "childclass", "joint", "material", "mesh", "target", "site"}
    for node in root.iter():
        if node.tag == "mesh" and node.get("file") and not node.get("name"):
            node.set("name", Path(node.get("file")).stem)
        if node.get("name"):
            node.set("name", PREFIX + node.get("name"))
        for attribute in references:
            if node.get(attribute):
                node.set(attribute, PREFIX + node.get(attribute))
        if node.tag == "mesh" and node.get("file"):
            node.set("file", str((go1_dir / "assets" / node.get("file")).resolve()))


def combined_xml() -> str:
    duck_mjcf, duck_meshes, _ = ensure_assets("legs")
    duck_root = ET.fromstring(build_xml(duck_mjcf, duck_meshes))
    go1_scene = ensure_go1()
    go1_root = ET.parse(go1_scene.parent / "go1.xml").getroot()
    _prefix_go1(go1_root, go1_scene.parent)

    # The baseline quadruped is a fixed, actuated standing target. Motion is a
    # later challenge; fixing the trunk isolates Microduck launch and landing.
    trunk = go1_root.find("worldbody/body")
    if trunk is None:
        raise RuntimeError("Pinned Go1 model has no root body")
    for freejoint in list(trunk.findall("freejoint")):
        trunk.remove(freejoint)
    trunk.set("pos", "0.62 0 0.445")
    ET.SubElement(trunk, "geom", {
        "name": "go1_landing_pad",
        "type": "box",
        "pos": "-0.02 0 0.078",
        "size": "0.18 0.10 0.012",
        "rgba": "0.12 0.62 0.78 0.42",
        "friction": "1.1 0.03 0.01",
        "condim": "6",
    })
    ET.SubElement(trunk, "site", {
        "name": "go1_landing_target",
        "type": "box",
        "pos": "-0.02 0 0.092",
        "size": "0.12 0.075 0.003",
        "rgba": "0.18 0.85 0.95 0.75",
    })

    for defaults in go1_root.findall("default"):
        duck_root.append(copy.deepcopy(defaults))
    duck_asset = duck_root.find("asset")
    go1_asset = go1_root.find("asset")
    for resource in list(go1_asset) if go1_asset is not None else []:
        duck_asset.append(copy.deepcopy(resource))
    duck_root.find("worldbody").append(copy.deepcopy(trunk))
    duck_actuator = duck_root.find("actuator")
    go1_actuator = go1_root.find("actuator")
    for actuator in list(go1_actuator) if go1_actuator is not None else []:
        duck_actuator.append(copy.deepcopy(actuator))

    # The original STAND key has a fixed qpos width and cannot include the
    # added Go1 joints. This scene initializes named joints explicitly below.
    keyframe = duck_root.find("keyframe")
    if keyframe is not None:
        duck_root.remove(keyframe)
    visual = duck_root.find("visual")
    if visual is None:
        visual = ET.SubElement(duck_root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "720")
    return ET.tostring(duck_root, encoding="unicode")


def make_scene():
    model = mujoco.MjModel.from_xml_string(combined_xml())
    data = mujoco.MjData(model)
    standing = {
        "hip": 0.0,
        "thigh": 0.9,
        "calf": -1.8,
    }
    for leg in ("FR", "FL", "RR", "RL"):
        for segment, value in standing.items():
            joint_name = f"go1_{leg}_{segment}_joint"
            data.qpos[model.joint(joint_name).qposadr] = value
            actuator_name = f"go1_{leg}_{segment}"
            data.ctrl[model.actuator(actuator_name).id] = value
    mujoco.mj_forward(model, data)
    return model, data


def report(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    duck = data.xpos[model.body("trunk_base").id]
    quadruped = data.xpos[model.body("go1_trunk").id]
    target = data.site_xpos[model.site("go1_landing_target").id]
    return {
        **validate(ensure_go1()),
        "combined_bodies": model.nbody,
        "combined_joints": model.njnt,
        "combined_actuators": model.nu,
        "duck_start_xyz": [round(float(v), 4) for v in duck],
        "quadruped_trunk_xyz": [round(float(v), 4) for v in quadruped],
        "landing_target_xyz": [round(float(v), 4) for v in target],
        "horizontal_launch_distance_m": round(float(target[0] - duck[0]), 4),
        "vertical_launch_distance_m": round(float(target[2] - duck[2]), 4),
    }


def render(model: mujoco.MjModel, data: mujoco.MjData, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.32, 0.0, 0.28]
    camera.distance, camera.azimuth, camera.elevation = 1.65, 145, -18
    renderer.update_scene(data, camera=camera)
    imageio.imwrite(destination, renderer.render())
    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    model, data = make_scene()
    result = report(model, data)
    if args.render:
        preview = OUT / "scene-validation.png"
        render(model, data, preview)
        result["preview"] = str(preview.relative_to(ROOT))
    print(json.dumps(result, indent=2))
