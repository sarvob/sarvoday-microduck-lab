#!/usr/bin/env python3
"""Build the verified Challenge 015 native-vertical Short."""

from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W, H, FPS = 1440, 2560, 60
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else REG, size)


def centered(draw: ImageDraw.ImageDraw, text: str, y: int, size: int,
             color: str, bold: bool = False) -> None:
    face = font(size, bold)
    box = draw.textbbox((0, 0), text, font=face)
    draw.text(((W - (box[2] - box[0])) / 2, y), text, font=face, fill=color)


def base_card(kicker: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), "#071018")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((72, 90, 1368, 300), 38, fill="#0D1B24", outline="#2C4551", width=4)
    centered(draw, kicker, 150, 39, "#9FB3BD", True)
    draw.text((1000, 2460), "SARVODAY ROBOTICS", font=font(25, True), fill="#708B98")
    return image, draw


def make_cards() -> None:
    hook = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hook, "RGBA")
    draw.rounded_rectangle((62, 86, 1378, 350), 38, fill=(5, 14, 23, 226), outline="#49E6A1", width=5)
    centered(draw, "HOW MANY MISSING FRAMES", 135, 55, "#FFFFFF", True)
    centered(draw, "BEFORE THE ROBOT MISSES?", 230, 52, "#49E6A1", True)
    hook.save(HERE / "hook-overlay.png")

    image, draw = base_card("ONE VARIABLE · CAMERA PACKET LOSS")
    centered(draw, "EVERYTHING ELSE", 555, 108, "#FFFFFF", True)
    centered(draw, "STAYS FROZEN", 700, 128, "#49E6A1", True)
    items = ["AUTHENTIC MICRODUCK", "15 MATCHED SHOTS", "ZERO ADDED DELAY", "STRICT WHOLE-BALL SCORING"]
    for index, item in enumerate(items):
        y = 1065 + index * 205
        draw.ellipse((155, y + 7, 205, y + 57), fill="#49E6A1")
        draw.text((250, y), item, font=font(44, True), fill="#D9E3E7")
    centered(draw, "RENDERED PIXELS · NO HIDDEN BALL POSITION", 2085, 37, "#FFB24C", True)
    image.save(HERE / "frozen-card.png")

    image, draw = base_card("15 FRESH SHOTS AT EACH LOSS SETTING")
    rows = [("0 / 210", 15, True), ("70 / 210", 15, True), ("115 / 210", 15, True),
            ("168 / 210", 10, False), ("192 / 210", 8, False)]
    draw.text((175, 465), "PACKETS LOST", font=font(35, True), fill="#8FA6B1")
    draw.text((850, 465), "SAVES", font=font(35, True), fill="#8FA6B1")
    for index, (lost, saves, passed) in enumerate(rows):
        y = 610 + index * 285
        color = "#49E6A1" if passed else "#FF7650"
        draw.rounded_rectangle((135, y, 1305, y + 205), 32, fill="#0D1B24", outline=color, width=5)
        draw.text((175, y + 60), lost, font=font(62, True), fill="#FFFFFF")
        draw.text((850, y + 43), f"{saves} / 15", font=font(78, True), fill=color)
    centered(draw, "PREDECLARED PASS BAR: 12 / 15", 2175, 42, "#CAD7DC", True)
    image.save(HERE / "scoreboard.png")

    image, draw = base_card("THE CAMERA DIDN'T LAG. THE PICTURES VANISHED.")
    centered(draw, "168 / 210", 535, 145, "#FF7650", True)
    centered(draw, "PACKETS LOST", 730, 82, "#FFFFFF", True)
    for index, x in enumerate(range(175, 1260, 145)):
        kept = index in (2, 6)
        color = "#49E6A1" if kept else "#FF7650"
        draw.rounded_rectangle((x, 1110, x + 92, 1260), 16,
                               fill="#173644" if kept else None, outline=color, width=5)
        if not kept:
            draw.line((x + 17, 1130, x + 75, 1240), fill=color, width=7)
            draw.line((x + 75, 1130, x + 17, 1240), fill=color, width=7)
    centered(draw, "10 / 15 SAVES", 1575, 132, "#FF7650", True)
    centered(draw, "NO FALLS · NO GOAL-ZONE EXITS", 1775, 43, "#CAD7DC", True)
    centered(draw, "NOT ENOUGH USEFUL PICTURES", 2050, 49, "#FFFFFF", True)
    image.save(HERE / "result-card.png")

    image, draw = base_card("THE PRODUCT DECISION")
    centered(draw, "DETECT THE LOSS", 540, 120, "#FFFFFF", True)
    centered(draw, "FALL BACK SAFELY", 705, 128, "#49E6A1", True)
    draw.rounded_rectangle((190, 1140, 1250, 1450), 42, fill="#0D1B24", outline="#49E6A1", width=5)
    centered(draw, "NEXT: BURST LOSSES", 1210, 62, "#FFFFFF", True)
    centered(draw, "RANDOM GAPS ARE THE POLITE VERSION", 1670, 41, "#FFB24C", True)
    centered(draw, "FOLLOW THE DUCK'S NEXT LESSON", 2110, 43, "#8FA6B1", True)
    image.save(HERE / "next-card.png")


def main() -> None:
    make_cards()
    broll = ROOT / "stream/episode-006-v2/stock-broll/pexels-6084018.mp4"
    evidence = ROOT / "artifacts/015-external-camera-frame-loss"
    sparse_save = evidence / "evidence-drop-0.50-seed-487.mp4"
    sparse_miss = evidence / "evidence-drop-0.75-seed-487.mp4"
    narration = HERE / "narration.wav"
    music = HERE / "music.wav"
    final = HERE / "external-camera-frame-loss-final.mp4"

    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=86:sample_rate=48000:duration=50",
        "-f", "lavfi", "-i", "sine=frequency=129:sample_rate=48000:duration=50",
        "-f", "lavfi", "-i", "sine=frequency=774:sample_rate=48000:duration=0.15",
        "-filter_complex",
        "[0:a]volume=0.009,tremolo=f=1.7:d=0.32[a0];"
        "[1:a]volume=0.0045,tremolo=f=2.5:d=0.20[a1];"
        "[2:a]volume=0.026,adelay=27400|27400,apad=pad_dur=50[a2];"
        "[a0][a1][a2]amix=inputs=3:duration=longest,afade=t=in:st=0:d=0.35,"
        "afade=t=out:st=49:d=0.8[m]",
        "-map", "[m]", "-t", "50", "-c:a", "pcm_s16le", str(music),
    ], check=True)

    command = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", "5.2", "-t", "3.4", "-i", str(broll),
        "-loop", "1", "-t", "4.6", "-i", str(HERE / "frozen-card.png"),
        "-i", str(sparse_save),
        "-loop", "1", "-t", "4.8", "-i", str(HERE / "scoreboard.png"),
        "-i", str(sparse_miss),
        "-loop", "1", "-t", "7.6", "-i", str(HERE / "result-card.png"),
        "-loop", "1", "-t", "9.2", "-i", str(HERE / "next-card.png"),
        "-i", str(narration), "-i", str(music),
        "-loop", "1", "-t", "3.4", "-i", str(HERE / "hook-overlay.png"),
    ]
    filters = [
        "[0:v]trim=duration=3.4,setpts=PTS-STARTPTS,"
        "scale=1440:2560:force_original_aspect_ratio=increase,crop=1440:2560,fps=60,"
        "eq=saturation=0.82:contrast=1.06[0b]",
        "[9:v]trim=duration=3.4,setpts=PTS-STARTPTS,fps=60[hook]",
        "[0b][hook]overlay=0:0:shortest=1[0v]",
        "[1:v]trim=duration=4.6,setpts=PTS-STARTPTS,fps=60[1v]",
        "[2:v]trim=start=0.65,setpts=PTS-STARTPTS,"
        "tpad=stop_mode=clone:stop_duration=0.7,trim=duration=13.0,fps=60[2v]",
        "[3:v]trim=duration=4.8,setpts=PTS-STARTPTS,fps=60[3v]",
        "[4:v]trim=start=0.65:duration=7.4,setpts=PTS-STARTPTS,fps=60[4v]",
        "[5:v]trim=duration=7.6,setpts=PTS-STARTPTS,fps=60[5v]",
        "[6:v]trim=duration=9.2,setpts=PTS-STARTPTS,fps=60[6v]",
        "[0v][1v][2v][3v][4v][5v][6v]concat=n=7:v=1:a=0,"
        "fade=t=out:st=49:d=0.8[v]",
        "[7:a]atrim=duration=50[n]",
        "[8:a]atrim=duration=50[m]",
        "[n]volume=1.0[nv];[m]volume=0.48[mv];"
        "[nv][mv]amix=inputs=2:duration=longest:dropout_transition=0,"
        "loudnorm=I=-16:LRA=8:TP=-1.5,afade=t=out:st=49:d=0.8[a]",
    ]
    command += [
        "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
        "-t", "50", "-c:v", "libx264", "-profile:v", "high", "-level", "5.1",
        "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
        str(final),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
