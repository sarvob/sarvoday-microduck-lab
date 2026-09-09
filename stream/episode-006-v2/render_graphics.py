#!/usr/bin/env python3
"""Render minimal audience-facing graphics for the goalkeeper story cut."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "episode-006" / "thumbnail-source.jpg"


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def base():
    bg = Image.open(SOURCE).convert("RGB").crop((50, 200, 1880, 1270)).resize((2560, 1440))
    bg = ImageEnhance.Brightness(bg).enhance(0.34).filter(ImageFilter.GaussianBlur(2.2))
    return bg


def panel(title, kicker, lines, output, accent="#42E49B"):
    im = base()
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((120, 130, 2440, 1310), 46, fill="#08141E", outline="#1B3441", width=4)
    d.text((190, 195), kicker.upper(), font=font(30, True), fill=accent)
    d.text((190, 260), title, font=font(74, True), fill="#F7FAFC")
    y = 450
    for number, headline, detail in lines:
        d.rounded_rectangle((190, y, 290, y + 100), 30, fill=accent)
        d.text((222, y + 18), str(number), font=font(50, True), fill="#071018")
        d.text((340, y - 4), headline, font=font(43, True), fill="#F7FAFC")
        d.text((340, y + 54), detail, font=font(27), fill="#AFC0C9")
        y += 210
    im.save(HERE / output)


panel(
    "Three things working against the duck",
    "The goalkeeper brief",
    [
        (1, "No sidestep", "It must pivot before moving across goal"),
        (2, "One tiny camera", "Turning can push the ball out of view"),
        (3, "Almost right still loses", "A late foot leaves the far post open"),
    ],
    "constraints.png",
    "#FFB24C",
)

panel(
    "Six tiny controllers enter",
    "The two-hour-sized search",
    [
        (0, "React to the current position", "7 of 10 training saves"),
        (1, "Predict the goal-line crossing", "5 bounded gains reached 10 of 10"),
        (2, "Choose the simplest winner", "Direct forecast, no oversteer"),
    ],
    "training.png",
)

panel(
    "The tiny duck gets the gloves",
    "Final score",
    [
        (15, "Prediction saves", "15 of 15 unseen shots"),
        (9, "Reactive saves", "Six late decisions get through"),
        (0, "Falls or zone exits", "The physical safety gates stayed clean"),
    ],
    "result.png",
    "#43E6A0",
)

# Thumbnail keeps the authentic robot image and adds one clear emotional bet.
thumb = Image.open(SOURCE).convert("RGB").crop((50, 200, 1880, 1270)).resize((1280, 720))
thumb = ImageEnhance.Contrast(thumb).enhance(1.15)
shade = Image.new("RGBA", thumb.size, (3, 10, 16, 55))
thumb = Image.alpha_composite(thumb.convert("RGBA"), shade).convert("RGB")
d = ImageDraw.Draw(thumb)
d.rounded_rectangle((42, 38, 760, 244), 28, fill="#071018")
d.text((78, 70), "CAN THIS DUCK", font=font(55, True), fill="#FFFFFF")
d.text((78, 138), "SAVE THE GOAL?", font=font(55, True), fill="#43E6A0")
d.rounded_rectangle((917, 565, 1238, 672), 24, fill="#0C3528", outline="#43E6A0", width=5)
d.text((980, 590), "15 / 15", font=font(48, True), fill="#FFFFFF")
thumb.save(HERE / "thumbnail-v2.jpg", quality=92, optimize=True)


def overlay(name, kicker, title, subtitle, accent="#43E6A0"):
    im = Image.new("RGBA", (2560, 1440), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((92, 82, 1690, 510), 40, fill=(5, 13, 20, 225),
                        outline=(40, 67, 80, 255), width=4)
    d.text((156, 140), kicker.upper(), font=font(29, True), fill=accent)
    d.text((156, 200), title, font=font(68, True), fill="#FFFFFF")
    d.text((160, 306), subtitle, font=font(31), fill="#C2D0D7")
    im.save(HERE / name)


overlay("cold-title.png", "A two-hour-sized product bet",
        "CAN A TINY DUCK KEEP GOAL?",
        "No arms. No sidestep. One head camera.", "#FFB24C")
overlay("baseline-title.png", "The obvious first version",
        "JUST CHASE THE BALL",
        "Looks good, until the far-post shot arrives.", "#FFB24C")
overlay("predictor-title.png", "One small change",
        "AIM WHERE THE BALL WILL BE",
        "Same walk. Earlier decision.", "#43E6A0")
overlay("final-test.png", "Fifteen unseen shots",
        "GOOD KEEPER OR LUCKY SAVE?",
        "Pass line 12. Falls allowed 0.", "#43E6A0")
