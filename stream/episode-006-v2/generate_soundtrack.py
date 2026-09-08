#!/usr/bin/env python3
"""Generate an original sports-electronic bed and episode sound effects."""

from pathlib import Path
import wave

import numpy as np

HERE = Path(__file__).resolve().parent
SR = 48_000
DURATION = 318.0
rng = np.random.default_rng(6006)
n = int(SR * DURATION)
mix = np.zeros((n, 2), dtype=np.float32)


def add(start, signal, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= n:
        return
    signal = np.asarray(signal, dtype=np.float32)[: n - i]
    left = np.sqrt((1 - pan) * 0.5)
    right = np.sqrt((1 + pan) * 0.5)
    mix[i:i + len(signal), 0] += signal * gain * left
    mix[i:i + len(signal), 1] += signal * gain * right


def tone(freq, seconds, decay=0.0, kind="sine"):
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    phase = 2 * np.pi * freq * t
    x = np.sin(phase)
    if kind == "triangle":
        x = 2 / np.pi * np.arcsin(x)
    env = np.ones_like(t) if not decay else np.exp(-t * decay)
    return x * env


def kick():
    seconds = 0.24
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    phase = 2 * np.pi * (92 * t - 34 * t * t)
    return np.sin(phase) * np.exp(-t * 20)


def noise_hit(seconds=0.12, decay=30):
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return rng.normal(0, 1, len(t)).astype(np.float32) * np.exp(-t * decay)


def intensity(t):
    if t < 18:
        return 0.45
    if t < 78:
        return 0.23
    if t < 138:
        return 0.36
    if t < 195:
        return 0.42
    if t < 235:
        return 0.32
    if t < 300:
        return 0.50 + 0.22 * (t - 235) / 65
    return 0.78


# Four-chord minor progression. Sparse under narration, energetic in tests.
roots = [65.41, 51.91, 58.27, 43.65]
beat = 0.5
for b, start in enumerate(np.arange(0, DURATION, beat)):
    level = intensity(float(start))
    root = roots[(b // 8) % len(roots)]
    add(start, kick(), gain=0.16 * level)
    if b % 2 == 1:
        add(start, noise_hit(0.10, 34), pan=(-0.2 if b % 4 else 0.2), gain=0.035 * level)
    if b % 4 == 2:
        add(start, noise_hit(0.19, 19), gain=0.075 * level)
    add(start, tone(root, 0.42, decay=5.0, kind="triangle"), gain=0.055 * level)
    arp = root * [2.0, 2.5, 3.0, 4.0][b % 4]
    add(start + 0.25, tone(arp, 0.18, decay=9), pan=(-0.35 if b % 2 else 0.35), gain=0.025 * level)

# Soft stadium air.
air = rng.normal(0, 1, n).astype(np.float32)
cs = np.cumsum(np.pad(air, (220, 0)), dtype=np.float64)
air = ((cs[220:] - cs[:-220]) / 220).astype(np.float32)
mix[:, 0] += air * 0.009
mix[:, 1] += np.roll(air, 110) * 0.009

# Referee whistle at the opening and the final save.
for start in (0.8, 301.4):
    t = np.arange(int(0.72 * SR), dtype=np.float32) / SR
    sig = (np.sin(2 * np.pi * (1850 + 160 * np.sin(2 * np.pi * 7 * t)) * t)
           + 0.35 * np.sin(2 * np.pi * 3700 * t)) * np.sin(np.pi * np.minimum(t / 0.08, 1))
    add(start, sig * np.exp(-t * 1.8), pan=-0.08, gain=0.07)

# Miss buzzer and save bells.
add(124.0, tone(155, 0.75, decay=3.2, kind="triangle"), gain=0.11)
for start in (177.0, 302.0):
    bell = tone(880, 1.4, decay=3.8) + 0.55 * tone(1320, 1.4, decay=4.2)
    add(start, bell, pan=0.18, gain=0.055)

# Original applause texture over the finish.
for start in np.arange(299.5, 317.5, 0.075):
    if rng.random() < 0.68:
        add(float(start + rng.uniform(-0.02, 0.02)), noise_hit(0.08, 42),
            pan=float(rng.uniform(-0.9, 0.9)), gain=float(rng.uniform(0.015, 0.045)))

peak = float(np.max(np.abs(mix)))
mix *= 0.78 / max(peak, 1e-6)
pcm = np.int16(np.clip(mix, -1, 1) * 32767)
with wave.open(str(HERE / "soundtrack.wav"), "wb") as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    wf.writeframes(pcm.tobytes())
print(HERE / "soundtrack.wav")
