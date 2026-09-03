#!/usr/bin/env python3
"""Render the four concise explanatory graphics used in Episode 004."""

from __future__ import annotations

from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
W, H, FPS = 2560, 1440, 60
BG, PANEL = "#081118", "#10202A"
WHITE, MUTED = "#F4F7F9", "#A8B7C1"
CYAN, GREEN, AMBER = "#52C9DC", "#64D6A0", "#F2BD5A"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def base(eyebrow: str, title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((150, 98), eyebrow.upper(), font=font(32, True), fill=CYAN)
    draw.text((150, 158), title, font=font(80, True), fill=WHITE)
    if subtitle:
        draw.text((150, 270), subtitle, font=font(36), fill=MUTED)
    return image, draw


def render(name: str, seconds: float, frame_fn) -> None:
    process = subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-profile:v", "high", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(HERE / name),
    ], stdin=subprocess.PIPE)
    try:
        for index in range(round(seconds * FPS)):
            assert process.stdin is not None
            process.stdin.write(frame_fn(index / FPS).tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"failed to render {name}")


def controller(t: float) -> Image.Image:
    image, draw = base("What was learned", "The policies stay frozen. The switch moves.",
                       "Only the high-level roll-to-stand handoff is searched")
    items = [
        ("FROZEN", "ROULADE POLICY", CYAN, 220, 1.0),
        ("LEARNED", "0.82 s HANDOFF", AMBER, 980, 3.0),
        ("FROZEN", "STAND POLICY", GREEN, 1740, 5.0),
    ]
    for label, name, color, x, start in items:
        p = ease((t - start) / 1.2)
        y = int(610 + (1 - p) * 100)
        draw.rounded_rectangle((x, y, x + 600, y + 320), radius=34,
                               fill=PANEL, outline=color, width=6)
        draw.text((x + 46, y + 48), label, font=font(30, True), fill=color)
        draw.text((x + 46, y + 132), name, font=font(43, True), fill=WHITE)
        if x < 1740 and p > 0.7:
            draw.line((x + 625, y + 160, x + 720, y + 160), fill=MUTED, width=8)
            draw.polygon([(x + 720, y + 160), (x + 684, y + 136), (x + 684, y + 184)], fill=MUTED)
    if t > 8:
        draw.text((710, 1125), "ONE CONTROL STEP = 20 MILLISECONDS", font=font(48, True), fill=AMBER)
    return image


def gates(t: float) -> Image.Image:
    image, draw = base("Physical success gate", "A dramatic roll is not enough.",
                       "Every condition must pass on all three perturbed starts")
    gates_data = [
        ("ROTATION", "≥ 300°", "complete a real roll"),
        ("INVERSION", "≤ −0.80", "trunk truly turns over"),
        ("RECOVERY", "≥ 0.90", "last 0.5 s mean upright"),
        ("DRIFT", "≤ 0.20 m", "finish near the start"),
    ]
    for index, (label, value, note) in enumerate(gates_data):
        x = 145 + index * 610
        p = ease((t - 1.0 - index * 1.2) / 1.0)
        y = int(560 + (1 - p) * 80)
        draw.rounded_rectangle((x, y, x + 540, y + 440), radius=32, fill=PANEL,
                               outline=GREEN if p > 0.98 else CYAN, width=5)
        draw.text((x + 38, y + 42), label, font=font(29, True), fill=CYAN)
        draw.text((x + 38, y + 135), value, font=font(62, True), fill=WHITE)
        draw.text((x + 38, y + 255), note, font=font(28), fill=MUTED)
        if p > 0.98:
            draw.text((x + 38, y + 345), "REQUIRED", font=font(28, True), fill=GREEN)
    return image


def results(t: float) -> Image.Image:
    image, draw = base("Held-out evaluation", "The selected timing passes 3 / 3.",
                       "Official policies · learned 0.82 s handoff · perturbed joint starts")
    rows = [
        ("023", "344.1°", "−0.877", "1.000", "0.1185 m"),
        ("079", "335.3°", "−0.849", "1.000", "0.1098 m"),
        ("191", "330.1°", "−0.815", "1.000", "0.1042 m"),
    ]
    cols = [("SEED", 210), ("ROTATION", 600), ("MIN UPRIGHT", 1040),
            ("FINAL UPRIGHT", 1540), ("DRIFT", 2070)]
    draw.rounded_rectangle((150, 470, 2410, 1145), radius=34, fill=PANEL)
    for label, x in cols:
        draw.text((x, 525), label, font=font(27, True), fill=MUTED)
    for index, row in enumerate(rows):
        p = ease((t - 1.3 - index * 1.4) / 1.0)
        y = 650 + index * 145
        draw.line((195, y + 105, 2365, y + 105), fill="#29404D", width=3)
        for value, (_, x) in zip(row, cols):
            draw.text((x, y), value, font=font(39, True), fill=WHITE)
        if p > 0.98:
            draw.rounded_rectangle((2240, y - 3, 2345, y + 64), radius=18, fill=GREEN)
            draw.text((2260, y + 13), "PASS", font=font(24, True), fill=BG)
    if t > 7.5:
        draw.text((770, 1230), "WORST DRIFT: 11.85 cm  ·  LIMIT: 20 cm", font=font(43, True), fill=GREEN)
    return image


def outro(t: float) -> Image.Image:
    image, draw = base("The reusable lesson", "Transitions are control problems, too.")
    lines = [
        "1  Define physical gates",
        "2  Search at the real control rate",
        "3  Test perturbed starts",
        "4  Inspect the first unstable boundary",
    ]
    for index, line in enumerate(lines):
        p = ease((t - 0.5 - index * 0.7) / 0.7)
        draw.text((310, 500 + index * 125), line, font=font(46, True),
                  fill=WHITE if p > 0 else BG)
    if t > 4.2:
        draw.rounded_rectangle((300, 1070, 2260, 1245), radius=30, fill=PANEL, outline=CYAN, width=4)
        draw.text((390, 1120), "github.com/sarvob/sarvoday-microduck-lab", font=font(43, True), fill=CYAN)
    return image


if __name__ == "__main__":
    render("controller-diagram.mp4", 14.0, controller)
    render("success-gates.mp4", 18.0, gates)
    render("result-table.mp4", 18.0, results)
    render("lesson-outro.mp4", 9.136, outro)
