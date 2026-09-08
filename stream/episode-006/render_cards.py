#!/usr/bin/env python3
"""Create native 1440p editorial cards and thumbnail for Episode 006."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 2560, 1440
BG, PANEL = "#071018", "#0E202B"
WHITE, MUTED, CYAN, GREEN, ORANGE = "#F5F8FA", "#AABAC4", "#51D0E5", "#61D7A2", "#FF7650"


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def base(eyebrow, title, subtitle=""):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((150, 100), eyebrow.upper(), font=font(32, True), fill=CYAN)
    d.text((150, 170), title, font=font(78, True), fill=WHITE)
    if subtitle:
        d.text((155, 285), subtitle, font=font(34), fill=MUTED)
    return im, d


def save(name, im):
    im.save(HERE / name, quality=95)


im, d = base("The product bet", "Move where the ball will be.", "Same robot. Same camera. Better decision target.")
for x, label, value, color in [(180, "REACTIVE", "current position", ORANGE), (1400, "PREDICTIVE", "goal-line crossing", GREEN)]:
    d.rounded_rectangle((x, 545, x + 980, 1080), radius=42, fill=PANEL, outline=color, width=6)
    d.text((x + 60, 610), label, font=font(30, True), fill=color)
    d.text((x + 60, 735), value, font=font(52, True), fill=WHITE)
    d.text((x + 60, 860), "camera → estimate → target", font=font(33), fill=MUTED)
save("card-product-bet.png", im)

im, d = base("Physical head camera", "Pixels become an interception target.", "No hidden ball coordinates are used for the decision.")
steps = [("1", "ORANGE PIXELS"), ("2", "CALIBRATED RAY"), ("3", "SHORT TRAJECTORY"), ("4", "CROSSING TARGET")]
for i, (n, label) in enumerate(steps):
    x = 135 + i * 610
    d.rounded_rectangle((x, 560, x + 500, 960), radius=34, fill=PANEL, outline=CYAN, width=4)
    d.ellipse((x + 45, 620, x + 135, 710), fill=CYAN)
    d.text((x + 76, 638), n, font=font(32, True), fill=BG)
    d.text((x + 45, 790), label, font=font(28, True), fill=WHITE)
    if i < 3:
        d.line((x + 515, 760, x + 585, 760), fill=MUTED, width=8)
save("card-vision-pipeline.png", im)

im, d = base("Training sweep", "Six bounded gains. One simple choice.", "Ten seeded training shots · official walking policy frozen")
data = [("0.00", "9 / 10", ORANGE), ("0.50", "10 / 10", GREEN), ("0.75", "10 / 10", GREEN),
        ("1.00", "10 / 10", CYAN), ("1.15", "10 / 10", GREEN), ("1.30", "10 / 10", GREEN)]
for i, (gain, score, color) in enumerate(data):
    x = 135 + i * 400
    d.rounded_rectangle((x, 560, x + 340, 1010), radius=30, fill=PANEL, outline=color, width=5)
    d.text((x + 48, 630), "GAIN", font=font(25, True), fill=MUTED)
    d.text((x + 48, 715), gain, font=font(55, True), fill=WHITE)
    d.text((x + 48, 855), score, font=font(38, True), fill=color)
    if gain == "1.00": d.text((x + 48, 945), "SELECTED", font=font(22, True), fill=CYAN)
save("card-training.png", im)

im, d = base("Unseen evaluation", "One miss became one extra save.", "Fifteen new speeds and approach angles")
for x, label, score, color in [(260, "POSITION CHASING", "14 / 15", ORANGE), (1390, "CROSSING PREDICTION", "15 / 15", GREEN)]:
    d.rounded_rectangle((x, 520, x + 910, 1070), radius=44, fill=PANEL, outline=color, width=7)
    d.text((x + 65, 610), label, font=font(30, True), fill=color)
    d.text((x + 65, 735), score, font=font(100, True), fill=WHITE)
    d.text((x + 65, 930), "0 falls · 0 zone exits", font=font(34), fill=MUTED)
save("card-results.png", im)

im, d = base("What this does not prove", "A passed gate is a starting point.")
lines = ["Orange-ball detector assumes controlled color and light", "Trajectory model is straight-line only",
         "Tracking currently happens before the robot turns", "Goal is compact; both controllers were already strong"]
for i, line in enumerate(lines):
    y = 490 + i * 170
    d.ellipse((190, y + 6, 230, y + 46), fill=ORANGE)
    d.text((280, y), line, font=font(42, True), fill=WHITE)
save("card-limitations.png", im)

im, d = base("The next product bet", "Make confidence part of the controller.", "Different light · partial occlusion · curved shots · both sides")
d.rounded_rectangle((220, 565, 2340, 1040), radius=48, fill=PANEL, outline=CYAN, width=6)
d.text((340, 660), "Should Microduck move—or hold the center?", font=font(58, True), fill=WHITE)
d.text((340, 820), "Predict the crossing. Estimate uncertainty. Choose the safer action.", font=font(38), fill=MUTED)
save("card-next.png", im)

im, d = base("Sarvoday Robotics", "Keep the robot real. Keep the bet measurable.", "github.com/sarvob/sarvoday-microduck-lab")
d.rounded_rectangle((310, 590, 2250, 1030), radius=52, fill=PANEL, outline=GREEN, width=6)
d.text((450, 690), "15 / 15", font=font(120, True), fill=GREEN)
d.text((1120, 720), "unseen predictive saves", font=font(54, True), fill=WHITE)
save("card-outro.png", im)

im, d = base("Vision-guided Microduck", "PREDICT THE SAVE?")
d.rounded_rectangle((150, 470, 2410, 1170), radius=48, fill=PANEL, outline=CYAN, width=8)
d.ellipse((320, 670, 610, 960), fill=ORANGE)
d.line((600, 815, 1740, 710), fill=CYAN, width=18)
d.ellipse((1690, 660, 1790, 760), outline=GREEN, width=18)
d.text((1880, 690), "15/15", font=font(90, True), fill=GREEN)
d.text((1880, 825), "UNSEEN", font=font(34, True), fill=MUTED)
save("thumbnail.png", im)
