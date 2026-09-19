#!/usr/bin/env python3
"""Assemble the verified challenge 013 vertical Short."""

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
    f = font(size, bold)
    box = draw.textbbox((0, 0), text, font=f)
    draw.text(((W - (box[2] - box[0])) / 2, y), text, font=f, fill=color)


def base_card(kicker: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), "#071018")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((72, 90, 1368, 300), 38, fill="#0D1B24", outline="#2C4551", width=4)
    centered(draw, kicker, 150, 39, "#9FB3BD", True)
    draw.text((1000, 2460), "SARVODAY ROBOTICS", font=font(25, True), fill="#708B98")
    return image, draw


def make_cards() -> None:
    hook = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hook_draw = ImageDraw.Draw(hook, "RGBA")
    hook_draw.rounded_rectangle((62, 86, 1378, 350), 38, fill=(5, 14, 23, 226), outline="#49E6A1", width=5)
    centered(hook_draw, "CAN A SECOND CAMERA", 135, 61, "#FFFFFF", True)
    centered(hook_draw, "SAVE THIS ROBOT DUCK?", 230, 61, "#49E6A1", True)
    hook.save(HERE / "hook-overlay.png")

    image, draw = base_card("BLOCKED HEAD CAMERA")
    centered(draw, "2", 560, 330, "#FFB24C", True)
    centered(draw, "SAVES", 960, 86, "#FFFFFF", True)
    centered(draw, "OUT OF 15 UNSEEN SHOTS", 1090, 48, "#B9C8CF", True)
    for index in range(15):
        x = 205 + (index % 5) * 255
        y = 1415 + (index // 5) * 185
        color = "#49E6A1" if index in (4, 8) else "#FF7650"
        draw.ellipse((x, y, x + 86, y + 86), fill=color)
    centered(draw, "ZERO BALL DETECTIONS", 2080, 50, "#FFB24C", True)
    image.save(HERE / "head-result.png")

    image, draw = base_card("ONE PRODUCT DECISION")
    centered(draw, "ADD A VIEW", 535, 128, "#FFFFFF", True)
    centered(draw, "ABOVE THE SCREEN", 700, 102, "#49E6A1", True)
    draw.rounded_rectangle((180, 1025, 1260, 1785), 46, fill="#0D1B24", outline="#49E6A1", width=6)
    draw.ellipse((640, 1110, 800, 1270), outline="#49E6A1", width=10)
    draw.line((720, 1270, 720, 1450), fill="#49E6A1", width=12)
    draw.polygon([(600, 1450), (840, 1450), (920, 1640), (520, 1640)], outline="#49E6A1")
    centered(draw, "RENDERED PIXELS · CALIBRATED TO THE FLOOR", 1900, 39, "#CAD7DC", True)
    centered(draw, "NO HIDDEN BALL COORDINATES", 1990, 45, "#FFB24C", True)
    image.save(HERE / "decision.png")

    image, draw = base_card("OUTSIDE CAMERA")
    centered(draw, "15 / 15", 560, 245, "#49E6A1", True)
    centered(draw, "SAVES", 865, 88, "#FFFFFF", True)
    centered(draw, "NO FALLS · NO GOAL-ZONE EXITS", 1045, 46, "#B9C8CF", True)
    for index in range(15):
        x = 205 + (index % 5) * 255
        y = 1360 + (index // 5) * 185
        draw.ellipse((x, y, x + 86, y + 86), fill="#49E6A1")
    centered(draw, "SAME ROBOT · SAME PHYSICS", 2060, 49, "#49E6A1", True)
    image.save(HERE / "final-result.png")

    image, draw = base_card("THE NEXT TEST")
    centered(draw, "HOW LATE", 600, 150, "#FFFFFF", True)
    centered(draw, "CAN THE VIEW ARRIVE?", 800, 94, "#49E6A1", True)
    draw.line((200, 1260, 1240, 1260), fill="#46616E", width=10)
    for x in (280, 520, 760, 1000, 1240):
        draw.line((x, 1215, x, 1305), fill="#FFB24C", width=9)
    centered(draw, "DELAY · DROPPED FRAMES · OCCLUSION", 1510, 47, "#CAD7DC", True)
    centered(draw, "FOLLOW THE DUCK'S NEXT LESSON", 2110, 43, "#8FA6B1", True)
    image.save(HERE / "next.png")


def main() -> None:
    make_cards()
    broll = ROOT / "stream/episode-006-v2/stock-broll/pexels-6084018.mp4"
    evidence = ROOT / "artifacts/013-external-camera-goalkeeper"
    head = evidence / "evidence-head-seed-307.mp4"
    outside = evidence / "evidence-outside-seed-307.mp4"
    narration = HERE / "narration.wav"
    music = HERE / "music.wav"
    final = HERE / "external-camera-goalkeeper-final.mp4"

    # Original procedural bed: quiet pulse, no third-party audio assets.
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=98:sample_rate=48000:duration=49",
        "-f", "lavfi", "-i", "sine=frequency=147:sample_rate=48000:duration=49",
        "-filter_complex",
        "[0:a]volume=0.010,tremolo=f=2.2:d=0.4[a0];"
        "[1:a]volume=0.006,tremolo=f=3.3:d=0.25[a1];"
        "[a0][a1]amix=inputs=2,afade=t=in:st=0:d=0.5,afade=t=out:st=47.8:d=1.0[m]",
        "-map", "[m]", "-c:a", "pcm_s16le", str(music),
    ], check=True)

    command = ["ffmpeg", "-y", "-v", "error",
               "-ss", "5.5", "-t", "2.8", "-i", str(broll),
               "-i", str(head),
               "-loop", "1", "-t", "5.9", "-i", str(HERE / "head-result.png"),
               "-loop", "1", "-t", "5.5", "-i", str(HERE / "decision.png"),
               "-i", str(outside),
               "-loop", "1", "-t", "7.5", "-i", str(HERE / "final-result.png"),
               "-loop", "1", "-t", "7.3", "-i", str(HERE / "next.png"),
               "-i", str(narration), "-i", str(music),
               "-loop", "1", "-t", "2.8", "-i", str(HERE / "hook-overlay.png")]
    filters = [
        "[0:v]trim=duration=2.8,setpts=PTS-STARTPTS,scale=1440:2560,fps=60[0b]",
        "[9:v]trim=duration=2.8,setpts=PTS-STARTPTS,fps=60[hook]",
        "[0b][hook]overlay=0:0:shortest=1[0v]",
        "[1:v]trim=duration=5.8,setpts=PTS-STARTPTS,fps=60[1v]",
        "[2:v]trim=duration=5.9,setpts=PTS-STARTPTS,fps=60[2v]",
        "[3:v]trim=duration=5.5,setpts=PTS-STARTPTS,fps=60[3v]",
        "[4:v]trim=duration=13,setpts=PTS-STARTPTS,fps=60[4v]",
        "[5:v]trim=duration=7.5,setpts=PTS-STARTPTS,fps=60[5v]",
        "[6:v]trim=duration=7.3,setpts=PTS-STARTPTS,fps=60[6v]",
        "[0v][1v][2v][3v][4v][5v][6v]concat=n=7:v=1:a=0,"
        "fade=t=out:st=47.8:d=1.0[v]",
        "[7:a]apad=pad_dur=3,atrim=duration=48.8,volume=1.0[n]",
        "[8:a]atrim=duration=48.8[m]",
        "[n][m]amix=inputs=2:duration=longest:dropout_transition=0,"
        "loudnorm=I=-16:LRA=8:TP=-1.5,afade=t=out:st=47.8:d=1.0[a]",
    ]
    command += [
        "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
        "-t", "48.8", "-c:v", "libx264", "-profile:v", "high", "-level", "5.1",
        "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
        str(final),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
