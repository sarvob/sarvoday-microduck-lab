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

> Sarvoday Robotics challenge lab, derived from the Hugging Face Microduck
> School and Pollen Robotics Microduck simulator. Upstream provenance and its
> original documentation are retained below.

## Sarvoday challenge loop

Each challenge has a machine-readable goal and pass/fail gate under
`challenges/`. A result is publishable only after the simulator passes the gate,
the tests pass, and `scripts/check_privacy.sh` reports clean.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_challenge.py challenges/001-spin-in-place/spec.json
.venv/bin/python scripts/render_challenge.py 001-spin-in-place
./scripts/check_privacy.sh
```

Challenge 001 teaches a supervisory controller to spin in place for at least
one full turn while staying upright and drifting no more than 0.35 m. The
shipped joint-level locomotion network remains frozen; the learned artifact is
the higher-level controller that commands it.

Challenge 002 teaches that controller to reach two offset markers in sequence,
forcing a steering reversal while staying upright and inside a 12.5 s gate.

Challenge 003 adds object interaction: approach a free ball, push it at least
0.30 m, and remain upright.

Challenge 004 is a coordinated three-duck MuJoCo scene inspired by a public
swing demonstration. Cross-entropy search learns the five-parameter timing
controller that pushes the swing; a third duck's applause is deterministic
choreography synchronized to the swing peaks. The shipped result reaches a
44.8-degree peak and sustains eight strong peaks in a 16-second evaluation.

Challenge 006 tunes the high-level handoff from the official frozen roulade
policy to the standing policy. It must invert, return upright, and finish within
0.20 m of its starting point across three perturbed initial poses.

Set a lesson in plain English. The duck **doesn't know how to do it**, tries
a couple of hundred times, and gets better — while you watch the score climb.

```
[lesson text] ─▶ (fn) design_lesson ─┬─▶ 📝 The lesson, as a goal
                     LLM             │
                                     ▼
[generations] ─────────▶ (fn) train_policy ─┬─▶ 📊 Learning curve
[seed] ─────────────────   CEM, ~200 tries  │
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

What is learned is the layer **above** them: a controller that watches the world
and decides what to *ask* the walking policy for — forward speed, sideways speed
and turn rate, ten times a second. Fifteen numbers:

```
[vx, vy, wz] = W · [1, error1, error2, error3, how_far_through_the_attempt]
```

The three error signals depend on the task: distance and heading to a marker,
to the ball, or how far off the circle it is.

That is the same split real robots use: a locomotion controller you don't
retrain, and a task policy on top that you do. Fifteen parameters is small enough
to optimise by **search** rather than gradients — no autograd, no GPU, no replay
buffer — which is the whole reason this fits in a Space at all.

The optimiser is the cross-entropy method: sample a batch of controllers, score
each by actually running it, keep the best few, refit the distribution to them,
repeat. Roughly 160-200 attempts over 8 generations, about 30-60 seconds of real
physics.

Before training, the controller is all zeros: the duck commands nothing and
stands there. Everything it does afterwards was found by search.

## What you can teach it

Five task families. The lesson planner picks one from what you type, and they
score genuinely different things — this is the part that decides whether your
words change anything:

| say something like | family | what gets measured |
|---|---|---|
| "walk to a marker behind you, without falling" | `goto` | markers reached, and how fast |
| "see how far it can get" | `explore` | distance from the start |
| "teach it to dance" / "spin on the spot" | `spin` | turns completed, minus drift |
| "walk a lap around where it started" | `circle` | how much of a lap, and how close to the circle |
| "shove the ball as far as it can" | `ball` | how far the ball ends up |

Adding a sixth means one `features_*` and one `score_*` in `lesson.py` — the
workflow, the app and the optimiser do not change.

Physics sets hard limits, and the planner clamps to them rather than accepting
an impossible lesson: the duck sustains about **0.11 m/s**, so a marker beyond
~0.8 m cannot be reached in one attempt, and a circle bigger than 0.25 m radius
cannot be walked in one (a 0.5 m circle is 3.1 m of walking — 29 seconds).

## What it cannot do

It cannot teach a headstand, or skating on one leg. Those need new *joint-level*
policies — a reward function in
[`microduck_rl`](https://github.com/pollen-robotics/microduck_rl), PPO, millions
of steps on a GPU, then an ONNX export. This Space searches over fifteen numbers,
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
