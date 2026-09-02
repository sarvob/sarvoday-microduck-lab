#!/usr/bin/env python3
"""Generate local narration with a stock Kokoro voice."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "kokoro-v1.0.int8.onnx"
VOICES = ROOT / "models" / "voices-v1.0.bin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Narration text")
    parser.add_argument("--voice", default="af_heart", help="Stock Kokoro voice")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "narration.wav")
    args = parser.parse_args()

    if not MODEL.exists() or not VOICES.exists():
        raise SystemExit("Kokoro model files are missing from stream/tts/models")

    engine = Kokoro(str(MODEL), str(VOICES))
    samples, sample_rate = engine.create(
        args.text,
        voice=args.voice,
        speed=args.speed,
        lang="en-us",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, samples, sample_rate)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
