# Episode 003 — Microduck lands on a Unitree Go1

## Daily three-video package

All footage must be rendered from MuJoCo at its final delivery resolution. Do
not reuse or upscale the 720p technical preview.

| Cut | Format | Duration | Native render | Distinct lesson |
| --- | --- | ---: | --- | --- |
| Main episode | 16:9 | 2:00–2:30 | 2560×1440, 60 fps | Full challenge, feasibility decision, training, and measured result |
| Short A | 9:16 | 0:35–0:45 | 1440×2560, 60 fps | Why the unassisted jump claim was rejected |
| Short B | 9:16 | 0:35–0:45 | 1440×2560, 60 fps | Three-seed landing result and what was actually learned |

## Visual structure

1. Open on the final two-foot landing, without revealing the outcome text.
2. Establish the 0.60 m horizontal and 0.417 m vertical target.
3. Show the rejected stock-policy probe: 0.0737 m rise and 0.02 s airtime.
4. Disclose the single initial launch velocity on screen: 1.2 m/s forward,
   3.3 m/s vertical, and no force after launch.
5. Show the 0.96 s untrained baseline, then the residual-controller search.
6. Finish with the three evaluation seeds and hold times: 1.68, 1.64, 1.64 s.
7. End on the public repository and the next challenge prompt.

## Quality gates

- H.264 High Profile, yuv420p, CRF 16, slow preset.
- AAC 48 kHz, 192 kbps when narration is present.
- Simulator, overlays, and text are native-resolution; no 720p scaling.
- Full playback inspection for motion, text, audio, accuracy, and privacy.
- Do not render until at least 20 GB of disk space is available.

## Evidence

- `artifacts/005-duck-quadruped-jump/landing-result.json`
- `artifacts/005-duck-quadruped-jump/landing-contact-sheet.jpg`
- `artifacts/005-duck-quadruped-jump/launch-feasibility.json`
- `artifacts/005-duck-quadruped-jump/launch-calibration.json`
