#!/usr/bin/env python3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRAND = ROOT.parent / "sarvoday-robotics-channel"
W, H = 1920, 1080
BG = "#0B1016"
WHITE = "#F6F8FA"
MINT = "#8ED7C6"
MUTED = "#B8C4CE"
PANEL = (7, 16, 24, 224)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def centered(draw: ImageDraw.ImageDraw, text: str, y: int, fnt, fill) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text(((W - (box[2] - box[0])) / 2, y), text, font=fnt, fill=fill)


def brand_mark(image: Image.Image, size: int = 96) -> None:
    mark = Image.open(BRAND / "sarvoday-robotics-watermark.png").convert("RGBA")
    mark.thumbnail((size, size), Image.Resampling.LANCZOS)
    image.alpha_composite(mark, (W - mark.width - 42, H - mark.height - 36))


intro = Image.new("RGBA", (W, H), BG)
d = ImageDraw.Draw(intro)
centered(d, "AI TRAINS A ROBOT DUCK", 390, font(78, True), WHITE)
centered(d, "3 measurable MuJoCo challenges  •  public code", 510, font(38), MINT)
brand_mark(intro)
intro.save(HERE / "intro.png")

overlays = [
    ("CHALLENGE 1  •  SPIN IN PLACE", "PASS  •  1.2 turns  •  0.04 m drift  •  upright", "Goal: complete one full turn without falling or drifting more than 0.35 m"),
    ("CHALLENGE 2  •  TWO-MARKER SPRINT", "PASS  •  2 of 2 markers  •  5.86 s  •  upright", "Goal: reverse steering and reach both offset markers in sequence"),
    ("CHALLENGE 3  •  PUSH THE BALL", "PASS  •  ball moved 1.00 m  •  upright", "Goal: approach a free ball and push it at least 0.30 m"),
]
for index, (title, result, goal) in enumerate(overlays, start=1):
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((52, 52, 940 if index == 2 else 840, 206), radius=18, fill=PANEL)
    d.text((78, 77), title, font=font(42, True), fill=WHITE)
    d.text((78, 142), result, font=font(31), fill=MINT)
    d.rectangle((0, 1010, W, H), fill=(7, 16, 24, 210))
    centered(d, goal, 1027, font(30), WHITE)
    brand_mark(image)
    image.save(HERE / f"overlay-{index}.png")

outro = Image.new("RGBA", (W, H), BG)
d = ImageDraw.Draw(outro)
centered(d, "THE LESSON", 265, font(40, True), MINT)
centered(d, "Keep locomotion frozen.", 365, font(66, True), WHITE)
centered(d, "Learn the supervisory controller.", 455, font(66, True), WHITE)
centered(d, "Clear gates  •  reproducible results  •  public repository", 590, font(36), MUTED)
centered(d, "SARVODAY ROBOTICS", 760, font(44, True), WHITE)
centered(d, "Build  •  Learn  •  Share", 825, font(32), MINT)
brand_mark(outro)
outro.save(HERE / "outro.png")

preview = Image.open(ROOT / "artifacts/003-ball-push/preview.png").convert("RGB")
preview = preview.crop((120, 70, 520, 295))
preview = preview.resize((1280, 720), Image.Resampling.LANCZOS).convert("RGBA")
preview.alpha_composite(Image.new("RGBA", preview.size, (0, 0, 0, 58)))
d = ImageDraw.Draw(preview)
d.rounded_rectangle((46, 62, 826, 338), radius=22, fill=PANEL)
d.text((82, 88), "AI TRAINS A", font=font(78, True), fill=WHITE)
d.text((82, 165), "ROBOT DUCK", font=font(88, True), fill=MINT)
d.text((82, 265), "3 / 3 PASSED", font=font(48, True), fill=WHITE)
mark = Image.open(BRAND / "sarvoday-robotics-watermark.png").convert("RGBA")
mark.thumbnail((112, 112), Image.Resampling.LANCZOS)
preview.alpha_composite(mark, (1280 - mark.width - 38, 720 - mark.height - 34))
preview.convert("RGB").save(HERE / "thumbnail.png", quality=95)
