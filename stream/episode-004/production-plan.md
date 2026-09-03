# Episode 004 — Controlled roll and recovery

Target: 3:45–4:30 educational landscape video, 2560×1440 at 60 fps.

## Story

| Chapter | Purpose | Visual evidence |
| --- | --- | --- |
| Question and gate | Explain why controller handoff timing matters | Clean overview, four success thresholds |
| Frozen versus learned | Disclose the official frozen policies and learned timing | Two-block controller diagram |
| Search | Show 26 candidates from 0.70–1.20 s | Animated timing sweep and pass-count plot |
| Boundary failure | Compare 0.82 s with the failing 0.84 s case | Synchronized dual-view replay |
| Evaluation | Report all three perturbed seeds | Native dual-view motion evidence and result table |
| Lesson | Generalize transition-timing method | Compact summary and public repository link |

## Visual rules

- Follow `stream/DUAL-VIEW-STYLE.md`.
- Overview is the dominant elevated camera; robot head-camera POV is the inset.
- Use real motion for at least two thirds of the runtime.
- Slow motion is allowed only around the handoff and must be labeled.
- Do not imply that the frozen roulade or stand networks were trained here.
- State that the learned component is the high-level 0.82 s handoff.

## Production result

- Final runtime: 4:17 (257.333 seconds).
- Delivery: 2560×1440 at 60 fps, H.264 High, yuv420p, AAC 48 kHz.
- Motion coverage: 184.2 seconds of unique simulator footage (71.6%).
- Audio: −16.0 dB mean, −1.5 dB peak.
- Full decode, nine-point visual inspection, engineering tests, and privacy
  checks passed. The episode is ready for immediate public upload.
