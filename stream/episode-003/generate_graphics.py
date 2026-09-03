#!/usr/bin/env python3
"""Generate restrained native-resolution cards and thumbnails for Episode 003."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRAND = ROOT.parent / "sarvoday-robotics-channel" / "sarvoday-robotics-watermark.png"
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


def watermark(image: Image.Image, size: int) -> None:
    mark = Image.open(BRAND).convert("RGBA")
    mark.thumbnail((size, size), Image.Resampling.LANCZOS)
    image.alpha_composite(mark, (image.width - mark.width - size // 2, image.height - mark.height - size // 2))


def wrap(draw, text: str, fnt, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def card(path: Path, size: tuple[int, int], eyebrow: str, title: str, body: str,
         metric: str | None = None, accent: str = CYAN) -> None:
    w, h = size
    image = Image.new("RGBA", size, BG)
    draw = ImageDraw.Draw(image)
    margin = int(w * 0.075)
    top = int(h * 0.15)
    draw.rounded_rectangle((margin, top, w - margin, h - top), radius=max(28, w // 60), fill=PANEL)
    title_size = 94 if w > h else 82
    body_size = 42 if w > h else 48
    draw.text((margin + 72, top + 70), eyebrow.upper(), font=font(30 if w > h else 34, True), fill=accent)
    y = top + 145
    for line in wrap(draw, title, font(title_size, True), w - 2 * margin - 144):
        draw.text((margin + 72, y), line, font=font(title_size, True), fill=WHITE)
        y += int(title_size * 1.12)
    y += 32
    for line in wrap(draw, body, font(body_size), w - 2 * margin - 144):
        draw.text((margin + 72, y), line, font=font(body_size), fill=MUTED)
        y += int(body_size * 1.42)
    if metric:
        metric_y = h - top - (150 if w > h else 220)
        draw.text((margin + 72, metric_y), metric, font=font(58 if w > h else 60, True), fill=accent)
    watermark(image, 112 if w > h else 132)
    image.convert("RGB").save(path, quality=96)


def thumbnail() -> None:
    source = Image.open(HERE / "landing-frame.jpg").convert("RGB")
    source = source.resize((1280, 720), Image.Resampling.LANCZOS).convert("RGBA")
    shade = Image.new("RGBA", source.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    sd.rectangle((0, 0, 780, 720), fill=(5, 11, 17, 190))
    source.alpha_composite(shade)
    draw = ImageDraw.Draw(source)
    draw.text((58, 80), "ROBOT DUCK", font=font(72, True), fill=WHITE)
    draw.text((58, 160), "LANDS ON", font=font(72, True), fill=WHITE)
    draw.text((58, 240), "ROBOT DOG", font=font(72, True), fill=CYAN)
    draw.rounded_rectangle((58, 380, 520, 480), radius=20, fill=(16, 29, 39, 235))
    draw.text((88, 405), "3 / 3 TESTS", font=font(44, True), fill=GREEN)
    watermark(source, 100)
    source.convert("RGB").save(HERE / "thumbnail.jpg", quality=96)


def main() -> None:
    landscape = (2560, 1440)
    portrait = (1440, 2560)
    card(HERE / "landscape-intro.jpg", landscape, "Sarvoday Robotics · Episode 003",
         "Can a robot duck land on a robot dog?",
         "A measured MuJoCo experiment with public code.")
    card(HERE / "landscape-constraint.jpg", landscape, "First: reject the impossible claim",
         "The stock policy could not jump high enough.",
         "We measured the behavior before choosing the experiment.", "0.0737 m rise · 0.02 s airborne", AMBER)
    card(HERE / "landscape-method.jpg", landscape, "Transparent assistance",
         "One initial launch. Zero force in flight.",
         "The learned residual controls the randomized two-foot landing and stable hold.",
         "1.2 m/s forward · 3.3 m/s vertical")
    card(HERE / "landscape-results.jpg", landscape, "Reproducible evaluation",
         "Both feet down. No ground contact.",
         "The controller passed every held-out evaluation seed.", "1.68 s · 1.64 s · 1.64 s", GREEN)
    card(HERE / "landscape-outro.jpg", landscape, "The lesson",
         "Measure first. Disclose constraints. Test repeatedly.",
         "Code, controller weights, and evidence are public.")
    card(HERE / "portrait-physics-intro.jpg", portrait, "Robot learning reality check",
         "We refused to fake this jump.",
         "Before training, we measured what the stock robot could actually do.", accent=AMBER)
    card(HERE / "portrait-physics-metric.jpg", portrait, "Measured result",
         "The target was 0.417 m above the start.",
         "The stock roll policy rose only 0.0737 m with 0.02 s of true airtime.",
         "Target missed by 82%", AMBER)
    card(HERE / "portrait-physics-method.jpg", portrait, "Honest experiment design",
         "One disclosed launch velocity.",
         "No midair force. The controller learns only landing and balance.")
    card(HERE / "portrait-result-intro.jpg", portrait, "Robot duck × robot dog",
         "Can it land twice—then stay upright?",
         "We evaluated the trained controller on three randomized launches.")
    card(HERE / "portrait-result-metric.jpg", portrait, "Three-seed result",
         "3 / 3 tests passed.",
         "Both feet landed on the Go1 platform with no later floor contact.",
         "1.68 s · 1.64 s · 1.64 s", GREEN)
    card(HERE / "portrait-result-outro.jpg", portrait, "Open robotics",
         "The code and measured evidence are public.",
         "Follow Sarvoday Robotics for the next challenge.")
    thumbnail()


if __name__ == "__main__":
    main()
