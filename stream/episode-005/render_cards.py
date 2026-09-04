#!/usr/bin/env python3
"""Create Episode 005's restrained 1440p chapter cards."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 2560, 1440
BG, PANEL = "#071218", "#10232D"
WHITE, MUTED, CYAN, GREEN, AMBER, RED = "#F5F8F9", "#A7B7C0", "#58CBDD", "#68D7A2", "#F3BE60", "#EF746A"


def font(size, bold=False):
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{'Arial Bold' if bold else 'Arial'}.ttf", size)


def card(name, eyebrow, title, subtitle, blocks):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.ellipse((1880, -380, 2800, 540), fill="#0B2833")
    d.ellipse((-360, 1050, 460, 1870), fill="#0A2029")
    d.text((150, 105), eyebrow.upper(), font=font(31, True), fill=CYAN)
    d.text((150, 170), title, font=font(78, True), fill=WHITE)
    d.text((150, 285), subtitle, font=font(35), fill=MUTED)
    count = len(blocks)
    gap, left, right = 34, 150, 2410
    bw = (right - left - gap * (count - 1)) // count
    for i, (head, value, note, color) in enumerate(blocks):
        x = left + i * (bw + gap)
        d.rounded_rectangle((x, 505, x + bw, 1115), radius=34, fill=PANEL, outline=color, width=5)
        d.text((x + 42, 555), head.upper(), font=font(28, True), fill=color)
        d.text((x + 42, 670), value, font=font(54, True), fill=WHITE)
        lines = note.split("\n")
        for j, line in enumerate(lines):
            d.text((x + 42, 805 + j * 48), line, font=font(29), fill=MUTED)
    d.text((150, 1278), "SARVODAY ROBOTICS  ·  MICRODUCK LAB", font=font(25, True), fill="#5F7D8A")
    im.save(HERE / name)


card("01-intro.png", "Challenge 012", "Microduck learns its sea legs.",
     "A measured balance-control experiment — including the failure boundary",
     [("Simulator", "MuJoCo", "50 Hz control\n1.6 × 0.8 m deck", CYAN),
      ("Evidence", "3 seeds", "Same controller\nperturbed starts", AMBER),
      ("Status", "In progress", "Harbor passes\nrough water fails", GREEN)])
card("02-problem.png", "The experiment", "Motion changes the support surface.",
     "The robot must stay upright, keep foot contact, and remain centered",
     [("Harbor", "3° roll", "2° pitch\n0.10–0.26 m/s", GREEN),
      ("Chop", "7° roll", "5° pitch\n0.17–0.53 m/s", AMBER),
      ("Surge", "Held out", "Not evaluated until\ntraining gates pass", CYAN)])
card("03-attempts.png", "Seven attempts", "The useful result was architectural.",
     "Frozen robot policies; only a disclosed high-level residual and recenter controller changed",
     [("Baseline", "7.3 s", "Standing alone\nleaves the deck", RED),
      ("Attempt 2", "20 s", "Walking recentering\nsolves harbor", GREEN),
      ("Attempt 6", "8.04 s", "Best chop seed\nstill exits", AMBER),
      ("Attempt 7", "Rejected", "Wave feed-forward\nadded no gain", RED)])
card("04-gates.png", "What counts as success", "Survival alone is not enough.",
     "Every gate must pass for 20 seconds on every evaluation seed",
     [("Upright", "≥ 0.50", "Final upright\nmust be ≥ 0.90", GREEN),
      ("Contact", "≥ 95%", "Feet remain on\nthe boat deck", CYAN),
      ("Drift", "≤ 0.30 m", "No deck exit\nno floor contact", AMBER)])
card("05-diagnosis.png", "The failure boundary", "Chop defeats foot placement, not the floor.",
     "All three failures are deck exits after contact quality and orientation degrade",
     [("Seed 029", "8.04 s", "68.4% contact\n0.372 m drift", RED),
      ("Seed 083", "7.90 s", "71.4% contact\n0.412 m drift", RED),
      ("Seed 197", "6.18 s", "75.7% contact\n0.694 m drift", RED)])
card("06-next.png", "Next engineering move", "Measure before adding another controller.",
     "The next run will expose whether commands saturate before useful foot contact",
     [("Observe", "Commands", "Clipping and\naxis saturation", CYAN),
      ("Observe", "Contacts", "Timing, duty cycle,\nslip onset", AMBER),
      ("Then", "Redesign", "Foot placement or\nrecovery state", GREEN)])
card("07-outro.png", "Open robotics, honestly measured", "A partial success is still useful evidence.",
     "Code, specifications, and machine-readable results are linked below",
     [("Verified", "Harbor 3/3", "Twenty seconds\nper seed", GREEN),
      ("Unsolved", "Chop 0/3", "Surge remains\nuntouched", RED),
      ("Repository", "Open source", "Reproduce it.\nImprove it.", CYAN)])
