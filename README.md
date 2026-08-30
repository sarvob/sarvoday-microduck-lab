---
title: Microduck School · gr.Workflow
emoji: 🎓
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - inference-api
---

# 🎓 Microduck School

Set a lesson in plain English. The duck **doesn't know how to do it**, tries
about a hundred times, and gets better — while you watch the score climb.

```
[lesson text] ─▶ (fn) design_lesson ─┬─▶ 📝 The lesson, as a goal
                     LLM             │
                                     ▼
[generations] ─────────▶ (fn) train_policy ─┬─▶ 📊 Learning curve
[seed] ─────────────────    CEM, ~96 tries  │
                                            ▼
                              (fn) demonstrate ─┬─▶ 🎬 The duck, after school
                                                ├─▶ 📊 Before / after
                                                └─▶ 📝 Report card
```

This is the companion to
[Microduck Lab](https://huggingface.co/spaces/ysharma/gr-workflow-microduck-lab),
which *performs* skills the robot already has. This one *acquires* a new one.

## What is actually being learned

Be precise about this, because it is the interesting part.

The nine shipped policies are **frozen**. Nothing about walking, balancing or
getting up is retrained — that took Pollen Robotics a GPU cluster and PPO, and
it is not happening in a CPU Space.

What is learned is the layer **above** them: a controller that watches where the
duck is and decides what to *ask* the walking policy for — a forward speed and
a turn rate, twenty times a second. Eight numbers:

```
[vx, wz] = W · [1, distance_to_goal, cos(heading_error), sin(heading_error)]
```

That is the same split real robots use: a locomotion controller you don't
retrain, and a task policy on top that you do. Eight parameters is small enough
to optimise by **search** rather than gradients — no autograd, no GPU, no replay
buffer — which is the whole reason this fits in a Space at all.

The optimiser is the cross-entropy method: sample a batch of controllers, score
each by actually running it, keep the best few, refit the distribution to them,
repeat. Roughly 96 attempts over 8 generations, about 30–60 seconds of real
physics.

Before training, the controller is all zeros: the duck commands nothing and
stands there. Everything it does afterwards was found by search.

## Lessons that work

- *"walk to a marker behind you, without falling over"* — it has to learn to
  turn around first
- *"get to the marker on your left as fast as you can"*
- *"run a two-marker course"* — reach one, then the other

Distance is a hard budget: the duck sustains about **0.11 m/s**, so a goal more
than ~1 m away cannot be reached within an episode and the lesson planner pulls
it back in. Two-marker courses keep each leg under 0.5 m.

## What it cannot do

It cannot teach a headstand, or skating on one leg. Those need new *joint-level*
policies — a reward function in
[`microduck_rl`](https://github.com/pollen-robotics/microduck_rl), PPO, millions
of steps on a GPU, then an ONNX export. This Space searches over eight numbers,
not over a neural network.

A headstand may not be trainable on this robot at all: 14 motors, no arms, and
the head is a sensor pod on a light neck whose roll joint only travels ±0.44 rad.
It isn't built to carry load. One-legged skating is a better bet — the roller
policy already balances dynamically.

## Running it

```bash
pip install -r apps/10_microduck_school/requirements.txt
python apps/10_microduck_school/app.py
```

First launch downloads ~10 MB of robot meshes and policies into
`.microduck_cache/`. The lesson planner uses `Qwen/Qwen3-4B-Instruct-2507` when
a token is available; **without one it falls back to a keyword planner**, so
nothing is behind a sign-in wall.

Training is the slow part — raise **Generations** for a better chance on a hard
lesson, lower it for a quick answer.

## Files

| file | what it is |
|---|---|
| `duck.py` | asset fetch, MJCF assembly, the 50 Hz policy/physics loop, rendering |
| `lesson.py` | the lesson spec, the learned controller, the rollout, and CEM |
| `nodes.py` | the three bound fn nodes |
| `build_workflow.py` | generates `workflow.json` — edit this, not the JSON |
| `test_pipelines.py` | runs every subject through the real `WorkflowExecutor` |

## Notes for anyone extending it

- **Keep the best individual, never the elite mean.** Averaging two good but
  different controllers can land between them and score far worse — seen live:
  best 1.27, elite mean 0.41, and the demo silently looked broken.
- **Floor the sigma and restart on stagnation.** Without a floor the elites
  agree by generation 3, the search stops, and a hard lesson gets stuck on a
  mediocre controller. With a floor plus a restart when the best score stalls
  below "solved", the turn-around lesson went from 2/5 to 5/5 across seeds.
- **Video dimensions must be even.** libx264 with `yuv420p` rejects odd heights,
  and `macro_block_size=1` means imageio won't quietly pad for you — 640×360 is
  fine, 560×315 dies with a broken pipe.
- Structured data travels as `text`, never `json` — the canvas serializes those
  with `String(obj)` and the receiver gets `"[object Object]"`.
- `workflow.json` is autosaved whenever the app runs, browser or not. Re-run
  `build_workflow.py` after any local launch.
