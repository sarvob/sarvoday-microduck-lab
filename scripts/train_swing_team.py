#!/usr/bin/env python3
"""Train and render the three-duck swing-team challenge in MuJoCo."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "challenges" / "004-duck-swing-team" / "spec.json"
OUT = ROOT / "artifacts" / "004-duck-swing-team"
DT = 0.01
CONTROL_DT = 0.02


def scene_xml() -> str:
    return f"""
<mujoco model="sarvoday swing team">
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="RK4"/>
  <visual><global azimuth="135" elevation="-16" offwidth="1280" offheight="720"/><quality shadowsize="2048" offsamples="4"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.13 0.20 0.34" rgb2="0.70 0.83 0.94" width="512" height="512"/>
    <texture name="ground" type="2d" builtin="checker" rgb1="0.18 0.24 0.31" rgb2="0.23 0.31 0.39" width="512" height="512"/>
    <material name="floor" texture="ground" texrepeat="12 12" reflectance="0.08"/>
    <material name="wood" rgba="0.36 0.20 0.10 1"/>
    <material name="yellow" rgba="0.96 0.68 0.09 1" metallic="0.15" roughness="0.35"/>
    <material name="dark" rgba="0.06 0.08 0.11 1" metallic="0.35" roughness="0.3"/>
    <material name="white" rgba="0.94 0.96 0.98 1"/>
    <material name="orange" rgba="1.0 0.35 0.06 1"/>
    <material name="blue" rgba="0.08 0.55 0.82 1"/>
  </asset>
  <worldbody>
    <light directional="true" pos="-2 -3 5" dir="0.4 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <light directional="true" pos="3 1 3" dir="-0.5 -0.2 -1" diffuse="0.35 0.42 0.55"/>
    <geom type="plane" size="4 4 0.1" material="floor"/>

    <geom name="frame_left_front" type="capsule" fromto="-0.18 -0.35 1.25 -0.92 -0.35 0.02" size="0.035" material="wood"/>
    <geom name="frame_left_back" type="capsule" fromto="-0.18 0.35 1.25 -0.92 0.35 0.02" size="0.035" material="wood"/>
    <geom name="frame_right_front" type="capsule" fromto="0.18 -0.35 1.25 0.92 -0.35 0.02" size="0.035" material="wood"/>
    <geom name="frame_right_back" type="capsule" fromto="0.18 0.35 1.25 0.92 0.35 0.02" size="0.035" material="wood"/>
    <geom name="frame_bar" type="capsule" fromto="0 -0.43 1.25 0 0.43 1.25" size="0.045" material="wood"/>

    <body name="swing" pos="0 0 1.22">
      <joint name="swing_hinge" type="hinge" axis="0 1 0" range="-82 82" damping="0.045" frictionloss="0.012"/>
      <geom type="capsule" fromto="0 -0.22 0 0 -0.22 -0.72" size="0.012" rgba="0.86 0.89 0.94 1"/>
      <geom type="capsule" fromto="0 0.22 0 0 0.22 -0.72" size="0.012" rgba="0.86 0.89 0.94 1"/>
      <geom name="seat" type="box" pos="0 0 -0.74" size="0.19 0.25 0.025" material="dark" mass="0.65"/>
      <body name="rider" pos="0 0 -0.57">
        <geom type="ellipsoid" size="0.15 0.12 0.18" material="dark" mass="1.8"/>
        <geom type="sphere" pos="0 0 0.20" size="0.12" material="white" mass="0.35"/>
        <geom type="box" pos="0.12 0 0.20" size="0.085 0.075 0.035" material="orange" mass="0.08"/>
        <geom type="sphere" pos="0.08 -0.09 0.24" size="0.018" material="dark"/>
        <geom type="sphere" pos="0.08 0.09 0.24" size="0.018" material="dark"/>
        <geom type="box" pos="-0.02 -0.10 -0.18" size="0.035 0.04 0.14" material="yellow"/>
        <geom type="box" pos="-0.02 0.10 -0.18" size="0.035 0.04 0.14" material="yellow"/>
      </body>
    </body>

    <body name="pusher" mocap="true" pos="-0.80 0 0.30">
      <geom type="ellipsoid" size="0.14 0.11 0.19" material="dark"/>
      <geom type="sphere" pos="0 0 0.20" size="0.115" material="white"/>
      <geom type="box" pos="0.12 0 0.20" size="0.08 0.07 0.032" material="orange"/>
      <geom type="sphere" pos="0.07 -0.085 0.24" size="0.017" material="dark"/>
      <geom type="sphere" pos="0.07 0.085 0.24" size="0.017" material="dark"/>
      <geom name="push_pad" type="capsule" fromto="0.05 0 0.08 0.29 0 0.12" size="0.035" material="yellow"/>
      <geom type="box" pos="0 -0.08 -0.20" size="0.035 0.04 0.15" material="yellow"/>
      <geom type="box" pos="0 0.08 -0.20" size="0.035 0.04 0.15" material="yellow"/>
    </body>

    <body name="clapper" mocap="true" pos="0.78 -0.72 0.30">
      <geom type="ellipsoid" size="0.14 0.11 0.19" material="dark"/>
      <geom type="sphere" pos="0 0 0.20" size="0.115" material="white"/>
      <geom type="box" pos="-0.12 0 0.20" size="0.08 0.07 0.032" material="orange"/>
      <geom type="sphere" pos="-0.07 -0.085 0.24" size="0.017" material="dark"/>
      <geom type="sphere" pos="-0.07 0.085 0.24" size="0.017" material="dark"/>
      <geom name="clap_left" type="capsule" fromto="0 -0.11 0.03 -0.02 -0.30 0.13" size="0.035" material="blue"/>
      <geom name="clap_right" type="capsule" fromto="0 0.11 0.03 -0.02 0.30 0.13" size="0.035" material="blue"/>
      <geom type="box" pos="0 -0.08 -0.20" size="0.035 0.04 0.15" material="yellow"/>
      <geom type="box" pos="0 0.08 -0.20" size="0.035 0.04 0.15" material="yellow"/>
    </body>
  </worldbody>
  <actuator><motor name="push_motor" joint="swing_hinge" gear="1.0" ctrlrange="-1 1" ctrllimited="true"/></actuator>
</mujoco>
"""


def make_model():
    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    return model, data


def controller(weights: np.ndarray, angle: float, velocity: float, phase: float) -> float:
    features = np.array([1.0, math.sin(phase), math.cos(phase), angle, velocity])
    return float(np.tanh(weights @ features))


def rollout(weights: np.ndarray, seconds: float = 14.0, trace: bool = False):
    model, data = make_model()
    data.qpos[0] = math.radians(3.0)
    mujoco.mj_forward(model, data)
    control_every = max(1, round(CONTROL_DT / DT))
    angles, controls = [], []
    u = 0.0
    for step in range(round(seconds / DT)):
        q, qd = float(data.qpos[0]), float(data.qvel[0])
        phase = math.atan2(qd, q + 1e-8)
        if step % control_every == 0:
            u = controller(weights, q, qd, phase)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if trace:
            angles.append(float(data.qpos[0]))
            controls.append(u)
    a = np.asarray(angles if trace else [float(data.qpos[0])])
    if trace:
        peaks = [abs(a[i]) for i in range(1, len(a) - 1)
                 if abs(a[i]) > abs(a[i - 1]) and abs(a[i]) >= abs(a[i + 1])]
        sustained = [p for p in peaks if p >= math.radians(30)]
        peak = float(max(peaks, default=0.0))
        tail_rms = float(np.sqrt(np.mean(a[len(a)//2:] ** 2)))
    else:
        peak, sustained, tail_rms = abs(float(data.qpos[0])), [], abs(float(data.qpos[0]))
    score = 5.0 * tail_rms + peak - 0.015 * float(np.mean(np.square(controls or [u])))
    return score, {"peak_angle_deg": math.degrees(peak), "sustained_peaks": len(sustained),
                   "tail_rms_deg": math.degrees(tail_rms), "angles": angles, "controls": controls}


def train(spec: dict):
    cfg = spec["training"]
    best_global = None
    histories = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        mu, sigma = np.zeros(5), np.full(5, 1.2)
        best_w, best_score = mu.copy(), -1e9
        history = []
        for generation in range(cfg["generations"]):
            pop = rng.normal(mu, sigma, size=(cfg["population"], 5))
            scored = [(rollout(w, seconds=10.0, trace=True)[0], w) for w in pop]
            scored.sort(key=lambda item: -item[0])
            if scored[0][0] > best_score:
                best_score, best_w = scored[0][0], scored[0][1].copy()
            elite = np.asarray([w for _, w in scored[:cfg["elite"]]])
            mu, sigma = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.12)
            history.append(round(best_score, 5))
            print(f"seed {seed} generation {generation + 1:02d}: {best_score:.3f}")
        _, metrics = rollout(best_w, seconds=16.0, trace=True)
        candidate = {"seed": seed, "score": best_score, "weights": best_w,
                     "metrics": metrics, "history": history}
        histories.append(candidate)
        if best_global is None or best_score > best_global["score"]:
            best_global = candidate
    return best_global, histories


@lru_cache(maxsize=None)
def load_font(size: int):
    for path in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def overlay(frame: np.ndarray, title: str, subtitle: str, progress: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((42, 40, 775, 150), radius=18, fill=(7, 12, 20, 205))
    draw.text((68, 57), title, font=load_font(34), fill=(245, 248, 252, 255))
    draw.text((70, 105), subtitle, font=load_font(21), fill=(126, 211, 255, 255))
    draw.rounded_rectangle((42, 665, 1238, 685), radius=8, fill=(255, 255, 255, 60))
    draw.rounded_rectangle((42, 665, 42 + int(1196 * progress), 685), radius=8,
                           fill=(247, 185, 55, 235))
    return np.asarray(image)


def render(best: dict, destination: Path, seconds: float = 60.0):
    model, data = make_model()
    data.qpos[0] = math.radians(3.0)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, -0.02, 0.62]
    camera.distance, camera.azimuth, camera.elevation = 3.05, 136, -13
    pusher_id = model.body("pusher").mocapid[0]
    clapper_id = model.body("clapper").mocapid[0]
    left_id, right_id = model.geom("clap_left").id, model.geom("clap_right").id
    fps, frames, next_frame_t = 30, [], 0.0
    weights = np.asarray(best["weights"], dtype=float)
    last_sign, claps = 0, 0
    u = 0.0
    total_steps = round(seconds / DT)
    for step in range(total_steps):
        t = step * DT
        q, qd = float(data.qpos[0]), float(data.qvel[0])
        phase = math.atan2(qd, q + 1e-8)
        if step % 2 == 0:
            u = controller(weights, q, qd, phase)
        data.ctrl[0] = u

        # The pusher leans in when learned torque is strongest.
        data.mocap_pos[pusher_id] = [-0.80 + 0.11 * abs(u), 0.0, 0.30 + 0.015 * math.sin(t * 5)]
        data.mocap_pos[clapper_id] = [0.78, -0.72, 0.30 + 0.018 * math.sin(t * 4)]
        sign = 1 if qd > 0 else -1
        if last_sign and sign != last_sign and abs(q) > math.radians(22):
            claps += 1
        last_sign = sign
        # Close the blue wings sharply at each swing apex, then reopen them.
        # These are explicitly disclosed as synchronized choreography.
        clap = math.exp(-((abs(qd) / 0.42) ** 2)) if abs(q) > math.radians(18) else 0.0
        wing_y = 0.205 - 0.105 * clap
        model.geom_pos[left_id] = [-0.01, -wing_y, 0.08]
        model.geom_pos[right_id] = [-0.01, wing_y, 0.08]
        model.geom_rgba[left_id, :3] = [0.08 + 0.28 * clap, 0.55 + 0.25 * clap, 0.82]
        model.geom_rgba[right_id, :3] = model.geom_rgba[left_id, :3]
        mujoco.mj_step(model, data)

        if t + 1e-9 >= next_frame_t and len(frames) < round(seconds * fps):
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            if t < 5:
                title, sub = "THREE-DUCK SWING TEAM", "MuJoCo challenge 004 · learned push timing"
            elif t < 46:
                title = f"SWING PEAK  {abs(math.degrees(q)):04.1f}°"
                sub = f"Duck 2 pushes · Duck 3 claps · peak events {claps}"
            elif t < 55:
                title, sub = "SUCCESS GATE PASSED", f"peak {best['metrics']['peak_angle_deg']:.1f}° · sustained peaks {best['metrics']['sustained_peaks']}"
            else:
                title, sub = "SARVODAY ROBOTICS", "Open simulations. Measurable results. Public code."
            frames.append(overlay(frame, title, sub, min(t / seconds, 1.0)))
            next_frame_t += 1.0 / fps
    renderer.close()
    imageio.mimsave(destination, frames, fps=fps, codec="libx264", quality=8,
                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC_PATH.read_text())
    result_path = OUT / "result.json"
    if args.render_only:
        result = json.loads(result_path.read_text())
        best = result["winner"]
    else:
        best, histories = train(spec)
        gate = spec["success"]
        m = best["metrics"]
        passed = (m["peak_angle_deg"] >= gate["minimum_peak_angle_deg"]
                  and m["peak_angle_deg"] <= gate["maximum_peak_angle_deg"]
                  and m["sustained_peaks"] >= gate["minimum_sustained_peaks"])
        clean_best = {k: v for k, v in best.items() if k not in {"metrics", "weights"}}
        clean_best["weights"] = [round(float(v), 7) for v in best["weights"]]
        clean_best["metrics"] = {k: v for k, v in best["metrics"].items()
                                 if k not in {"angles", "controls"}}
        result = {"challenge": spec["id"], "passed": passed, "success_gate": gate,
                  "disclosure": spec["disclosure"], "winner": clean_best}
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        (OUT / "policy.json").write_text(json.dumps(clean_best, indent=2) + "\n")
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        for row in histories:
            ax.plot(range(1, len(row["history"]) + 1), row["history"], marker="o",
                    label=f"seed {row['seed']}")
        ax.set(title="Swing timing training", xlabel="generation", ylabel="best score")
        ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(OUT / "learning-curve.png"); plt.close(fig)
        if not passed:
            print(json.dumps(result, indent=2))
            return 1
        best = clean_best
    render(best, OUT / "demonstration-60s.mp4")
    print(json.dumps({"passed": True, "video": str(OUT / "demonstration-60s.mp4"),
                      "metrics": best["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
