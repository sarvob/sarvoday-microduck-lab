#!/usr/bin/env python3
"""Build the verified Challenge 014 vertical Short."""

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
    draw.rounded_rectangle((62, 86, 1378, 350), 38, fill=(5, 14, 23, 226), outline="#FFB24C", width=5)
    centered(draw, "CAN A ROBOT GOALKEEPER", 135, 57, "#FFFFFF", True)
    centered(draw, "SURVIVE FOUR SECONDS OF LAG?", 230, 49, "#FFB24C", True)
    hook.save(HERE / "hook-overlay.png")

    image, draw = base_card("ONE VARIABLE · CAMERA DELIVERY TIME")
    centered(draw, "EVERYTHING ELSE", 565, 108, "#FFFFFF", True)
    centered(draw, "STAYS FROZEN", 710, 128, "#49E6A1", True)
    items = ["AUTHENTIC MICRODUCK", "15 MATCHED SHOTS", "PHYSICS + PREDICTOR", "STRICT WHOLE-BALL SCORING"]
    for index, item in enumerate(items):
        y = 1080 + index * 205
        draw.ellipse((155, y + 7, 205, y + 57), fill="#49E6A1")
        draw.text((250, y), item, font=font(44, True), fill="#D9E3E7")
    centered(draw, "RENDERED PIXELS · NO HIDDEN BALL POSITION", 2085, 37, "#FFB24C", True)
    image.save(HERE / "frozen-card.png")

    image, draw = base_card("15 FRESH SHOTS AT EACH DELAY")
    rows = [("0 s", 15, True), ("1 s", 15, True), ("2 s", 15, True),
            ("3 s", 12, True), ("4 s", 7, False)]
    draw.text((245, 475), "DELAY", font=font(37, True), fill="#8FA6B1")
    draw.text((780, 475), "SAVES", font=font(37, True), fill="#8FA6B1")
    for index, (delay, saves, passed) in enumerate(rows):
        y = 625 + index * 285
        color = "#49E6A1" if passed else "#FF7650"
        draw.rounded_rectangle((160, y, 1280, y + 205), 32, fill="#0D1B24", outline=color, width=5)
        draw.text((245, y + 54), delay, font=font(72, True), fill="#FFFFFF")
        draw.text((780, y + 43), f"{saves} / 15", font=font(82, True), fill=color)
    centered(draw, "PREDECLARED PASS BAR: 12 / 15", 2180, 42, "#CAD7DC", True)
    image.save(HERE / "scoreboard.png")

    image, draw = base_card("THE MESSAGE ARRIVED. THE MOMENT DIDN'T WAIT.")
    centered(draw, "4 SECONDS", 540, 165, "#FFB24C", True)
    centered(draw, "OF CAMERA LAG", 760, 92, "#FFFFFF", True)
    draw.line((230, 1170, 1210, 1170), fill="#46616E", width=12)
    draw.ellipse((220, 1115, 330, 1225), fill="#49E6A1")
    draw.ellipse((1110, 1115, 1220, 1225), fill="#FF7650")
    draw.text((175, 1270), "FRAME CAPTURED", font=font(32, True), fill="#49E6A1")
    draw.text((975, 1270), "DELIVERED", font=font(32, True), fill="#FF7650")
    centered(draw, "7 / 15 SAVES", 1610, 132, "#FF7650", True)
    centered(draw, "NO FALLS · NO GOAL-ZONE EXITS", 1810, 43, "#CAD7DC", True)
    centered(draw, "THE CONTROL LOOP RAN OUT OF TIME", 2080, 44, "#FFFFFF", True)
    image.save(HERE / "result-card.png")

    image, draw = base_card("THE NEXT TEST")
    centered(draw, "WHAT IF THE WI-FI", 590, 117, "#FFFFFF", True)
    centered(draw, "DROPS FRAMES?", 755, 145, "#49E6A1", True)
    for index, x in enumerate(range(220, 1230, 170)):
        if index in (2, 5):
            draw.rounded_rectangle((x, 1260, x + 115, 1435), 18, outline="#FF7650", width=6)
            draw.line((x + 20, 1280, x + 95, 1415), fill="#FF7650", width=8)
            draw.line((x + 95, 1280, x + 20, 1415), fill="#FF7650", width=8)
        else:
            draw.rounded_rectangle((x, 1260, x + 115, 1435), 18, fill="#173644", outline="#49E6A1", width=4)
    centered(draw, "SAME DUCK · SAME SHOTS · MISSING PIXELS", 1710, 43, "#CAD7DC", True)
    centered(draw, "FOLLOW THE DUCK'S NEXT LESSON", 2110, 43, "#8FA6B1", True)
    image.save(HERE / "next-card.png")


def main() -> None:
    make_cards()
    broll = ROOT / "stream/episode-006-v2/stock-broll/pexels-6084018.mp4"
    evidence = ROOT / "artifacts/014-external-camera-latency"
    clean = evidence / "evidence-latency-0.0-seed-397.mp4"
    lagged = evidence / "evidence-latency-4.0-seed-397.mp4"
    narration = HERE / "narration.wav"
    music = HERE / "music.wav"
    final = HERE / "external-camera-latency-final.mp4"

    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=92:sample_rate=48000:duration=52",
        "-f", "lavfi", "-i", "sine=frequency=138:sample_rate=48000:duration=52",
        "-f", "lavfi", "-i", "sine=frequency=690:sample_rate=48000:duration=0.18",
        "-filter_complex",
        "[0:a]volume=0.009,tremolo=f=1.8:d=0.35[a0];"
        "[1:a]volume=0.005,tremolo=f=2.7:d=0.22[a1];"
        "[2:a]volume=0.025,adelay=33800|33800,apad=pad_dur=52[a2];"
        "[a0][a1][a2]amix=inputs=3:duration=longest,afade=t=in:st=0:d=0.4,"
        "afade=t=out:st=51:d=0.8[m]",
        "-map", "[m]", "-t", "52", "-c:a", "pcm_s16le", str(music),
    ], check=True)

    command = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", "5.2", "-t", "3.2", "-i", str(broll),
        "-loop", "1", "-t", "5.0", "-i", str(HERE / "frozen-card.png"),
        "-i", str(clean),
        "-loop", "1", "-t", "10.5", "-i", str(HERE / "scoreboard.png"),
        "-i", str(lagged),
        "-loop", "1", "-t", "9.0", "-i", str(HERE / "result-card.png"),
        "-loop", "1", "-t", "7.6", "-i", str(HERE / "next-card.png"),
        "-i", str(narration), "-i", str(music),
        "-loop", "1", "-t", "3.2", "-i", str(HERE / "hook-overlay.png"),
    ]
    filters = [
        "[0:v]trim=duration=3.2,setpts=PTS-STARTPTS,"
        "scale=1440:2560:force_original_aspect_ratio=increase,crop=1440:2560,fps=60[0b]",
        "[9:v]trim=duration=3.2,setpts=PTS-STARTPTS,fps=60[hook]",
        "[0b][hook]overlay=0:0:shortest=1[0v]",
        "[1:v]trim=duration=5.0,setpts=PTS-STARTPTS,fps=60[1v]",
        "[2:v]trim=start=0.7:duration=10.5,setpts=PTS-STARTPTS,fps=60[2v]",
        "[3:v]trim=duration=10.5,setpts=PTS-STARTPTS,fps=60[3v]",
        "[4:v]trim=duration=6.2,setpts=PTS-STARTPTS,fps=60[4v]",
        "[5:v]trim=duration=9.0,setpts=PTS-STARTPTS,fps=60[5v]",
        "[6:v]trim=duration=7.6,setpts=PTS-STARTPTS,fps=60[6v]",
        "[0v][1v][2v][3v][4v][5v][6v]concat=n=7:v=1:a=0,"
        "fade=t=out:st=51:d=0.8[v]",
        "[7:a]atrim=duration=52[n]",
        "[8:a]atrim=duration=52[m]",
        "[n]volume=1.0[nv];[m]volume=0.48[mv];"
        "[nv][mv]amix=inputs=2:duration=longest:dropout_transition=0,"
        "loudnorm=I=-16:LRA=8:TP=-1.5,afade=t=out:st=51:d=0.8[a]",
    ]
    command += [
        "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
        "-t", "52", "-c:v", "libx264", "-profile:v", "high", "-level", "5.1",
        "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
        str(final),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
