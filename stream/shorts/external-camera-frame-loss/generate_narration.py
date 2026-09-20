#!/usr/bin/env python3
"""Generate timed Kokoro Heart narration one beat at a time."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from stream.tts.generate_narration import FULL_MODEL, INT8_MODEL, VOICES
from kokoro_onnx import Kokoro


def main() -> None:
    beats = json.loads((HERE / "narration-beats.json").read_text())
    model = FULL_MODEL if FULL_MODEL.exists() else INT8_MODEL
    if not model.exists() or not VOICES.exists():
        raise SystemExit("Kokoro model files are missing")
    engine = Kokoro(str(model), str(VOICES))
    voice = engine.get_voice_style("af_heart")
    sample_rate = 24000
    total = np.zeros(round(50.0 * sample_rate), dtype=np.float32)
    manifest = []
    for index, beat in enumerate(beats, 1):
        samples, actual_rate = engine.create(
            beat["text"], voice=voice, speed=beat["speed"], lang="en-us",
            sentence_pause=0.30, clause_pause=0.12,
        )
        if actual_rate != sample_rate:
            raise RuntimeError(f"unexpected sample rate {actual_rate}")
        peak = float(np.max(np.abs(samples)))
        if peak:
            samples = samples * (10 ** (-1.5 / 20) / peak)
        start = round(float(beat["start"]) * sample_rate)
        end = start + len(samples)
        if end > len(total):
            raise RuntimeError(f"beat {index} overruns the 50-second master")
        if np.any(total[start:end]):
            raise RuntimeError(f"beat {index} overlaps the previous beat")
        total[start:end] = samples
        beat_path = HERE / f"narration-{index:02d}.wav"
        sf.write(beat_path, samples, sample_rate)
        manifest.append({**beat, "duration": round(len(samples) / sample_rate, 3),
                         "file": beat_path.name})
    sf.write(HERE / "narration.wav", total, sample_rate)
    (HERE / "narration-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(HERE / "narration.wav")


if __name__ == "__main__":
    main()
