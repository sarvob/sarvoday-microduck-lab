#!/usr/bin/env python3
"""Build the Episode 005 YouTube thumbnail from genuine simulator footage."""

from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "chop-3-seeds.mp4"
FRAME = HERE / "thumbnail-frame.png"
OUTPUT = HERE / "thumbnail.jpg"


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


subprocess.run([
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "4.5",
    "-i", str(SOURCE), "-frames:v", "1", str(FRAME),
], check=True)

image = Image.open(FRAME).convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
image = ImageEnhance.Contrast(image).enhance(1.10)
image = ImageEnhance.Color(image).enhance(0.92)
overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay, "RGBA")
draw.rectangle((0, 0, 670, 720), fill=(4, 13, 18, 210))
for x in range(670):
    draw.line((x, 0, x, 720), fill=(4, 13, 18, max(0, 210 - int(x * 0.25))))
draw.rounded_rectangle((58, 58, 300, 108), radius=14, fill=(86, 202, 220, 235))
draw.text((82, 70), "MICRODUCK LAB", font=font(21, True), fill=(4, 18, 24, 255))
draw.text((58, 160), "SEA", font=font(108, True), fill=(246, 249, 250, 255))
draw.text((58, 258), "LEGS", font=font(108, True), fill=(246, 249, 250, 255))
draw.text((62, 394), "A BALANCE CONTROLLER", font=font(29, True), fill=(166, 211, 219, 255))
draw.text((62, 434), "MEETS ROUGH WATER", font=font(29, True), fill=(166, 211, 219, 255))
draw.rounded_rectangle((58, 535, 550, 632), radius=24, fill=(11, 26, 34, 225),
                       outline=(243, 190, 96, 255), width=4)
draw.text((88, 563), "HARBOR 3/3  ·  CHOP 0/3", font=font(28, True), fill=(243, 190, 96, 255))
image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
image.save(OUTPUT, quality=90, optimize=True, progressive=True)
print(OUTPUT)
