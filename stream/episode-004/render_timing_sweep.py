#!/usr/bin/env python3
"""Render Episode 004's 26-candidate handoff timing sweep."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WIDTH, HEIGHT, FPS, SECONDS = 2560, 1440, 60, 14
BG = "#081118"
PANEL = "#101E28"
WHITE = "#F4F7F9"
MUTED = "#A7B6C0"
GRID = "#2D414E"
CYAN = "#52C9DC"
GREEN = "#64D6A0"
AMBER = "#F2BD5A"
RED = "#F07870"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def writer(path: Path) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-profile:v", "high", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def main() -> None:
    result = json.loads((ROOT / "artifacts/006-controlled-roll/result.json").read_text())
    candidates = result["candidates"]
    output = HERE / "timing-sweep.mp4"
    process = writer(output)
    try:
        for frame in range(SECONDS * FPS):
            t = frame / FPS
            image = Image.new("RGB", (WIDTH, HEIGHT), BG)
            draw = ImageDraw.Draw(image)
            draw.text((150, 92), "CONTROLLER HANDOFF SEARCH", font=font(32, True), fill=CYAN)
            draw.text((150, 150), "Twenty milliseconds changes the outcome", font=font(82, True), fill=WHITE)
            draw.text((150, 260), "26 timings · 3 perturbed starts per timing · all three must pass", font=font(38), fill=MUTED)

            left, top, right, bottom = 170, 440, 2390, 1085
            draw.rounded_rectangle((left, top, right, bottom), radius=34, fill=PANEL)
            plot_left, plot_top, plot_right, plot_bottom = left + 115, top + 90, right - 85, bottom - 105
            for pass_count in range(4):
                y = plot_bottom - pass_count * (plot_bottom - plot_top) / 3
                draw.line((plot_left, y, plot_right, y), fill=GRID, width=3)
                draw.text((plot_left - 68, y - 18), str(pass_count), font=font(29, True), fill=MUTED)
            draw.text((left + 25, top + 24), "PASSING SEEDS", font=font(26, True), fill=MUTED)

            reveal = int(ease((t - 1.1) / 6.5) * len(candidates))
            bar_space = (plot_right - plot_left) / len(candidates)
            bar_width = max(18, int(bar_space * 0.62))
            for index, candidate in enumerate(candidates[:reveal]):
                x = plot_left + index * bar_space + (bar_space - bar_width) / 2
                pass_count = candidate["passing_seeds"]
                bar_top = plot_bottom - pass_count * (plot_bottom - plot_top) / 3
                color = GREEN if pass_count == 3 else RED
                if candidate["roll_steps"] == 41:
                    color = AMBER
                draw.rounded_rectangle((x, bar_top, x + bar_width, plot_bottom), radius=8, fill=color)

            for step in (35, 41, 42, 50, 60):
                index = step - 35
                x = plot_left + (index + 0.5) * bar_space
                draw.text((x - 35, plot_bottom + 28), f"{step/50:0.2f}", font=font(25), fill=MUTED)
            draw.text((1040, bottom - 55), "HANDOFF TIME (SECONDS)", font=font(27, True), fill=MUTED)

            chosen_p = ease((t - 7.2) / 1.2)
            if chosen_p > 0:
                chosen_x = plot_left + (41 - 35 + 0.5) * bar_space
                draw.rounded_rectangle((chosen_x - 82, plot_top - 42, chosen_x + 82, plot_bottom + 18),
                                       radius=20, outline=AMBER, width=max(2, int(7 * chosen_p)))
                draw.rounded_rectangle((190, 1145, 1210, 1325), radius=28, fill="#142732", outline=AMBER, width=4)
                draw.text((235, 1180), "SELECTED  ·  0.82 s", font=font(39, True), fill=AMBER)
                draw.text((235, 1245), "3 / 3 seeds  ·  0.1185 m worst drift", font=font(31), fill=WHITE)

            boundary_p = ease((t - 9.0) / 1.2)
            if boundary_p > 0:
                boundary_x = plot_left + (42 - 35 + 0.5) * bar_space
                draw.rounded_rectangle((boundary_x - 77, plot_top - 34, boundary_x + 77, plot_bottom + 18),
                                       radius=20, outline=RED, width=max(2, int(7 * boundary_p)))
                draw.rounded_rectangle((1350, 1145, 2370, 1325), radius=28, fill="#142732", outline=RED, width=4)
                draw.text((1395, 1180), "ONE STEP LATER  ·  0.84 s", font=font(39, True), fill=RED)
                draw.text((1395, 1245), "2 / 3 seeds  ·  recovery gate fails", font=font(31), fill=WHITE)

            assert process.stdin is not None
            process.stdin.write(image.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("timing sweep render failed")


if __name__ == "__main__":
    main()
