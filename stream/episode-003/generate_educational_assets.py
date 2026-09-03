#!/usr/bin/env python3
"""Create native-resolution animated explanatory assets for Episode 003."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, H, FPS = 2560, 1440, 30
BG = "#0A1118"
PANEL = "#101D27"
WHITE = "#F4F7F9"
MUTED = "#AEBBC5"
CYAN = "#50C9DC"
AMBER = "#F2BD5A"
GREEN = "#63D6A0"


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def writer(path: Path):
    return subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-profile:v", "high", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def base(eyebrow: str, title: str, subtitle: str = ""):
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    d.text((150, 105), eyebrow.upper(), font=font(34, True), fill=CYAN)
    d.text((150, 165), title, font=font(86, True), fill=WHITE)
    if subtitle:
        d.text((150, 280), subtitle, font=font(38), fill=MUTED)
    return image, d


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def physics() -> None:
    out = writer(HERE / "animated-physics-comparison.mp4")
    seconds = 16
    try:
        for frame in range(seconds * FPS):
            image, d = base("Feasibility before training", "Can the stock policy reach the target?",
                            "Measured trunk rise in metres")
            ground, top = 1190, 430
            max_h = ground - top
            d.line((370, ground, 2190, ground), fill="#405260", width=5)
            stock_p = ease((frame / FPS - 2.0) / 3.0)
            target_p = ease((frame / FPS - 6.0) / 3.0)
            stock_h = int(max_h * (0.0737 / 0.417) * stock_p)
            target_h = int(max_h * target_p)
            d.rounded_rectangle((620, ground-stock_h, 1000, ground), radius=24, fill=AMBER)
            d.rounded_rectangle((1530, ground-target_h, 1910, ground), radius=24, fill=CYAN)
            d.text((645, 1225), "STOCK", font=font(40, True), fill=WHITE)
            d.text((1520, 1225), "TARGET", font=font(40, True), fill=WHITE)
            d.text((650, max(ground-stock_h-85, 350)), f"{0.0737*stock_p:0.4f} m", font=font(50, True), fill=AMBER)
            d.text((1550, max(ground-target_h-85, 350)), f"{0.417*target_p:0.3f} m", font=font(50, True), fill=CYAN)
            if frame / FPS > 10:
                d.rounded_rectangle((820, 350, 1740, 470), radius=24, fill=PANEL)
                d.text((875, 380), "UNASSISTED CLAIM REJECTED", font=font(46, True), fill=AMBER)
            out.stdin.write(image.tobytes())
    finally:
        out.stdin.close()
        if out.wait() != 0:
            raise RuntimeError("physics animation failed")


def training() -> None:
    result = json.loads((ROOT / "artifacts/005-duck-quadruped-jump/landing-result.json").read_text())
    values = result["history"]
    out = writer(HERE / "animated-training-curve.mp4")
    seconds = 18
    try:
        for frame in range(seconds * FPS):
            image, d = base("Cross-entropy search", "Sixteen generations. Eight residual parameters.",
                            "Best robust score so far")
            left, top, right, bottom = 260, 430, 2300, 1180
            d.rounded_rectangle((left, top, right, bottom), radius=30, fill=PANEL)
            d.line((left+100, bottom-90, right-70, bottom-90), fill="#49606F", width=4)
            d.line((left+100, top+70, left+100, bottom-90), fill="#49606F", width=4)
            visible = max(1, min(len(values), int((frame / (seconds*FPS))* (len(values)+3))))
            points = []
            for i, value in enumerate(values[:visible]):
                x = left + 100 + i * (right-left-210) / (len(values)-1)
                y = bottom - 90 - (value-28) / 13 * (bottom-top-180)
                points.append((x, y))
            if len(points) > 1:
                d.line(points, fill=GREEN, width=12, joint="curve")
            for x, y in points:
                d.ellipse((x-10, y-10, x+10, y+10), fill=WHITE)
            if points:
                d.text((right-520, top+90), f"BEST  {values[visible-1]:0.2f}", font=font(50, True), fill=GREEN)
                d.text((left+100, bottom-45), f"GENERATION {visible:02d} / 16", font=font(34, True), fill=MUTED)
            out.stdin.write(image.tobytes())
    finally:
        out.stdin.close()
        if out.wait() != 0:
            raise RuntimeError("training animation failed")


def controller() -> None:
    out = writer(HERE / "animated-controller-diagram.mp4")
    seconds = 18
    labels = [("FROZEN", "Neural stand policy", 280, CYAN),
              ("LEARNED", "8-parameter residual", 1030, AMBER),
              ("TESTED", "Two-foot landing gate", 1780, GREEN)]
    try:
        for frame in range(seconds * FPS):
            image, d = base("Inspectable learning", "Keep the proven policy. Learn only the correction.")
            t = frame / FPS
            for i, (tag, name, x, color) in enumerate(labels):
                p = ease((t - (2+i*3)) / 1.5)
                y = int(610 + (1-p)*120)
                d.rounded_rectangle((x, y, x+500, y+300), radius=32, fill=PANEL, outline=color, width=6)
                d.text((x+42, y+48), tag, font=font(30, True), fill=color)
                d.text((x+42, y+118), name, font=font(44, True), fill=WHITE)
                if i < 2 and p > .7:
                    d.line((x+520, y+150, x+690, y+150), fill=MUTED, width=10)
                    d.polygon([(x+690,y+150),(x+650,y+125),(x+650,y+175)], fill=MUTED)
            if t > 12:
                d.text((680, 1090), "HIP  ·  KNEE  ·  ANKLE  ·  STANCE WIDTH", font=font(52, True), fill=WHITE)
            out.stdin.write(image.tobytes())
    finally:
        out.stdin.close()
        if out.wait() != 0:
            raise RuntimeError("controller animation failed")


def results() -> None:
    holds = [1.68, 1.64, 1.64]
    out = writer(HERE / "animated-results.mp4")
    seconds = 18
    try:
        for frame in range(seconds * FPS):
            image, d = base("Held-out evaluation", "Three randomized launches. One physical success gate.",
                            "Minimum required stable hold: 1.50 seconds")
            t = frame / FPS
            for i, hold in enumerate(holds):
                x = 310 + i*760
                p = ease((t-(2+i*2.4))/2.5)
                d.text((x, 520), f"SEED {17 if i==0 else 71 if i==1 else 173}", font=font(36, True), fill=MUTED)
                d.rounded_rectangle((x, 620, x+570, 790), radius=28, fill=PANEL)
                d.rounded_rectangle((x, 620, x+int(570*(hold/1.8)*p), 790), radius=28, fill=GREEN)
                d.text((x, 845), f"{hold*p:0.2f} s", font=font(62, True), fill=WHITE)
                if p > .98:
                    d.text((x, 980), "PASS", font=font(46, True), fill=GREEN)
            if t > 12:
                d.text((700, 1190), "BOTH FEET  ·  UPRIGHT  ·  NO FLOOR CONTACT", font=font(44, True), fill=WHITE)
            out.stdin.write(image.tobytes())
    finally:
        out.stdin.close()
        if out.wait() != 0:
            raise RuntimeError("results animation failed")


if __name__ == "__main__":
    physics()
    controller()
    training()
    results()
