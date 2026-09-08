#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

HERE = Path(__file__).resolve().parent
source = Image.open(HERE / "thumbnail-source.jpg").convert("RGB")
# Use only the authentic overview panel, then reframe it as the hero image.
source = source.crop((55, 200, 1880, 1268)).resize((2560, 1440), Image.Resampling.LANCZOS)
source = ImageEnhance.Contrast(source).enhance(1.14)
overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
d = ImageDraw.Draw(overlay)
d.rounded_rectangle((75, 65, 2485, 350), radius=40, fill=(5, 13, 20, 220),
                    outline=(76, 210, 232, 255), width=6)
bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
d.text((145, 112), "CAN MICRODUCK PREDICT THE SAVE?",
       font=ImageFont.truetype(bold, 85), fill=(247, 250, 252, 255))
d.rounded_rectangle((1940, 1120, 2450, 1360), radius=36, fill=(10, 31, 38, 235),
                    outline=(96, 216, 164, 255), width=7)
d.text((2022, 1155), "15 / 15", font=ImageFont.truetype(bold, 78),
       fill=(96, 216, 164, 255))
d.text((2053, 1270), "UNSEEN", font=ImageFont.truetype(bold, 31),
       fill=(196, 208, 216, 255))
Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB").save(
    HERE / "thumbnail.jpg", quality=94, optimize=True)
