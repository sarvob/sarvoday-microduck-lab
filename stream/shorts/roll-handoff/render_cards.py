#!/usr/bin/env python3
"""Render restrained vertical graphics for the roll-handoff Short."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
W, H = 1080, 1920


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def overlay(name: str, kicker: str, headline: str, sub: str, color: str):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((54, 72, 1026, 320), 34, fill=(5, 13, 20, 224),
                        outline=color, width=4)
    d.text((100, 110), kicker.upper(), font=font(28, True), fill=color)
    d.text((100, 158), headline, font=font(66, True), fill="#FFFFFF")
    d.text((102, 246), sub, font=font(31), fill="#CBD7DD")
    im.save(HERE / name)


def card(name: str, kicker: str, headline: tuple[str, ...], sub: str, color: str):
    im = Image.new("RGB", (W, H), "#071018")
    d = ImageDraw.Draw(im)
    d.ellipse((650, -160, 1300, 490), fill="#102F3E")
    d.ellipse((-300, 1420, 420, 2140), fill="#182B25")
    d.text((78, 190), kicker.upper(), font=font(30, True), fill=color)
    y = 410
    for line in headline:
        d.text((78, y), line, font=font(92, True), fill="#F7FAFC")
        y += 112
    d.rounded_rectangle((78, y + 70, 1002, y + 240), 30, fill="#0D202B")
    d.text((120, y + 112), sub, font=font(36), fill="#C4D0D6")
    im.save(HERE / name)


overlay("hook.png", "Boundary failure", "20 ms TOO LATE", "0.84 s handoff", "#FF7650")
overlay("win.png", "One control step earlier", "ROLL. CATCH. STAND.", "0.82 s handoff", "#43E6A0")
card("decision.png", "Same robot. Same network.", ("WHAT IF WE", "STOP EARLIER?"),
     "0.84 s  →  0.82 s", "#FFB24C")
card("evidence.png", "Three perturbed starts", ("3 TRIES", "3 RECOVERIES"),
     "The winning handoff held up", "#43E6A0")
card("cta.png", "Sarvoday Robotics", ("A TRICK", "BECOMES A SKILL"),
     "Subscribe for the next unreasonable build", "#43E6A0")
