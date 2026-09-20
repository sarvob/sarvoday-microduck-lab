# Challenge 015 final QA

- Master: `external-camera-frame-loss-final.mp4`
- Duration: 50.000 seconds with matched 50.000-second video and audio tracks
- Video: 1440×2560, 60 fps, 3,000 frames, H.264 High, yuv420p
- Audio: AAC-LC, 48 kHz, mono
- Loudness: -16.70 LUFS integrated, -1.49 dBTP, 3.10 LU LRA
- Decode: complete with no decode errors
- Black-frame scan: only the intentional 0.200-second closing fade
- Visual review: licensed hook crop, authentic Microduck framing, overhead-camera
  pixels, packet-kept/dropped treatment, counters, terminal SAVE and MISSED · GOAL
  labels, scoreboard, result card, and next-build card reviewed from the full export
- Evidence: matched seed 487 shows a genuine save under the 50% predeclared mask
  (7 of 14 observations delivered) and a genuine goal under the nested 75% mask
  (3 of 14 observations delivered). The full 75-shot result is stored in
  `artifacts/015-external-camera-frame-loss/result.json`.
- Factual review: 15/15 saves after 0, 70, and 115 of 210 packets were dropped;
  10/15 after 168 were dropped; 8/15 after 192 were dropped; no falls or
  goal-zone exits in any condition
- Tests: all 153 repository tests passed after the final evidence-label fix
- Rights: documented in `credits.md`
- Privacy: no owner email address or full name appears in the video, metadata,
  narration, or Challenge 015 production files

## Upload state

YouTube Studio was authenticated on SarvodayRobotics, but Chrome rejected the
browser-assisted local-file transfer with `Not allowed`. No upload or YouTube
URL was created. Open `chrome://extensions`, choose Details for the ChatGPT
browser extension, enable `Allow access to file URLs`, return to the preserved
YouTube Studio upload dialog, and select this master.
