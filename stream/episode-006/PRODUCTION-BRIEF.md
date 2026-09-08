# Episode 006 — Can Microduck Learn to Be a Goalkeeper?

## Product question

Can the authentic Pollen Robotics Microduck use only its head-camera view to
predict where a rolling ball will cross the goal line, then move sideways in
time to block it?

This is a product story, not a victory montage. The episode should make the
problem, trade-offs, failed attempts, evidence, and next product bet easy to
follow. Do not write or imply a successful outcome until the measured test has
passed.

## Non-negotiable production contract

- Publish as a normal 16:9 YouTube episode; never as a Short.
- Final duration must be at least 5:00. Target 5:30–6:00 without padding.
- Deliver at 2560×1440, 60 fps, H.264 High, with clean 48 kHz audio.
- The hero is the authentic Microduck from the Pollen Robotics CAD/MJCF assets.
  Never substitute a generic duck or redesign its body, face, legs, or feet.
- All claimed robot and ball motion must come from the real MuJoCo experiment.
- Blender may provide the stadium, goal, floor materials, lighting, atmosphere,
  camera movement, ball trails, and presentation graphics. Those elements must
  not modify the experiment or obscure what is simulated versus illustrative.
- Show both an overview and Microduck's physical head-camera POV. Keep labels
  visible enough that a viewer always knows which view is evidence.
- Use the curious-builder voice: playful and product-minded, with short clear
  explanations. Avoid lecture language and fact-dumping.
- Do not expose a personal name, email address, credentials, notifications, or
  unrelated work anywhere in the image, audio, metadata, or repository.

## Engineering contract

- Randomize unseen ball speed and approach angle.
- Estimate the ball's goal-line crossing from the head camera.
- Command a bounded lateral interception without leaving the goal zone.
- Compare against a baseline that reacts to current ball position without
  predicting the future crossing point.
- Publish the declared test set and result artifact with the code.

Success gate: block at least 12 of 15 unseen shots, remain inside the goal zone,
and avoid falling in every evaluated shot. A miss remains in the episode when it
teaches something useful.

## Story and edit map

| Time | Story beat | Evidence and visual treatment |
| --- | --- | --- |
| 0:00–0:20 | Cold open: one painful miss, then the best verified save | Fast overview/POV cuts in the lit Blender-style arena; no result claim yet |
| 0:20–0:55 | The product bet | Why prediction could beat chasing where the ball is now |
| 0:55–1:35 | What Microduck can see | Authentic head-camera view with restrained tracking and crossing-point overlays |
| 1:35–2:15 | Baseline | Real MuJoCo attempts, miss patterns, and a simple score panel |
| 2:15–3:15 | Build the predictor and controller | One clean diagram, then motion showing how the intercept target changes |
| 3:15–4:10 | Iteration | The two or three failures that changed the design; keep the narration conversational |
| 4:10–5:10 | Unseen 15-shot evaluation | Honest shot counter, speed/angle labels, overview plus robot POV, no cherry-picking |
| 5:10–5:45 | Decision and next bet | State the measured result, what remains fragile, and the next experiment |

## Visual language

Aim for an elegant compact indoor robotics arena: warm key light, cool rim light,
matte floor, subtle goal markings, restrained depth of field, and clean motion
graphics. Use a thin trajectory line and a small predicted-crossing marker; do
not turn the screen into a dashboard. Let the authentic Microduck remain the
visual focus.

The thumbnail should show the real Microduck braced in front of the goal with a
single incoming ball and one clear prediction line. Working title:
“Can Microduck Learn to Be a Goalkeeper?”
