# Sarvoday dual-view shooting style

Reference: Alex Bodner's Microduck object-tracking video
<https://x.com/AlexBodner_/status/2094806087228662270>

## Composition

- Master: 2560×1440, 60 fps, 16:9.
- Overview: full canvas, elevated 35–55° tracking camera with generous negative space.
- Robot POV: 880×495 picture-in-picture in the upper-right safe area.
- Inset treatment: 5 px warm-white border, 24 px radius, soft translucent shadow.
- Camera labels: one quiet pill per view; never a full-width title bar over motion.
- Metrics: no more than two live measurements, grouped in a single lower-right pill.
- Episode context: challenge, seed, phase, and time in one lower-left card.

## Art direction

- Prefer a simple low-poly training world with one dominant ground color and a
  small number of high-contrast task objects.
- Keep the robot and goal visually distinct; remove decorative objects that do
  not explain the task.
- Use restrained navy translucent panels, cool-cyan secondary text, and amber
  only for live measurements or pass/fail emphasis.
- Let motion carry the shot. Reserve full-screen graphics for short chapter
  transitions, diagrams, and final measured results.

## Editing rhythm

1. Establish the arena in the overview.
2. Introduce the robot POV when perception or targeting becomes relevant.
3. Cut to a closer overview at the decisive movement.
4. Replay the key instant once from the alternate view when it teaches something.
5. End on a held result frame with the success gate and measured outcome.

Episode 004's evidence renderer is the first implementation of this style.
