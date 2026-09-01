"""The bound fn nodes for Microduck School.

  design_lesson(lesson)          -> lesson spec, as TEXT
  train_policy(spec, gens, seed) -> (learned controller TEXT, learning curve IMAGE)
  demonstrate(spec, policy)      -> (video, before/after tracks IMAGE, report TEXT)

Structured data travels as `text` and media as {"path", "url"} dicts, because
the workflow canvas string-coerces `json` fn outputs and needs a data URI for
media.
"""

import base64
import json
import os
import re
import sys
import tempfile
from typing import Optional

import numpy as np
from gradio import OAuthToken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duck as D  # noqa: E402
import lesson as L  # noqa: E402

LLM = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM = """You set lessons for Microduck, a 25 cm bipedal robot.

Pick ONE task family and fill in what it needs.

  "goto"     walk to one or two markers, in order.
             needs "targets": [[x, y]] -- one or two pairs.
  "explore"  get as far from the starting spot as possible.
  "spin"     turn on the spot without wandering off.
  "circle"   walk a circle around the starting spot.
             needs "radius" in metres, 0.12 to 0.25 (bigger is not
             walkable in one attempt).
  "ball"     shove a ball as far as possible. A ball is put in front of it.

The duck starts at (0, 0) facing +x. In metres: +x is ahead, -x is behind,
+y is to its left, -y is to its right. It walks at about 0.11 m/s, so a single
marker belongs 0.35-0.8 m out, and a two-marker course within 0.5 m each.

Reply with ONLY a JSON object, and include only the fields the task needs:

{"task": "spin", "label": "spin on the spot"}
{"task": "goto", "targets": [[-0.55, 0.12]], "label": "reach the marker behind you"}

Choose the family that matches what was asked. Do NOT force everything into
"goto": dancing or twirling is "spin", roaming is "explore", a lap is
"circle", anything about the ball is "ball".

Output the JSON object and nothing else."""

FEWSHOT = [
    {"role": "user", "content": "teach it to dance"},
    {"role": "assistant", "content":
        '{"task": "spin", "label": "spin on the spot like a dance"}'},
    {"role": "user", "content": "teach it to walk to a spot behind it without falling over"},
    {"role": "assistant", "content":
        '{"task": "goto", "targets": [[-0.55, 0.12]], '
        '"label": "turn around and reach the marker behind you"}'},
    {"role": "user", "content": "see how far it can get"},
    {"role": "assistant", "content":
        '{"task": "explore", "label": "get as far from the start as possible"}'},
]

# Keyword -> task family, for the no-token fallback. Order matters: the first
# family whose word appears wins, so the specific ones come before "goto".
_TASK_WORDS = [
    ("ball", ("ball", "kick", "shove", "push", "football", "soccer", "dribble")),
    ("circle", ("circle", "lap", "laps", "orbit", "loop", "round", "around")),
    ("spin", ("spin", "twirl", "pirouette", "dance", "rotate", "whirl")),
    ("explore", ("far", "furthest", "farthest", "explore", "roam", "wander",
                 "away", "distance", "journey")),
]
_DIRECTIONS = [
    (("behind", "back", "backwards", "reverse"), (-0.55, 0.12)),
    (("left",), (0.35, 0.5)),
    (("right",), (0.35, -0.5)),
    (("ahead", "forward", "front", "straight"), (0.7, 0.0)),
]

WORD = r"\b"          # word boundary, so "far" does not match "farm"


def _token(oauth_token):
    tok = getattr(oauth_token, "token", None) or os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _has(text, word):
    return re.search(WORD + re.escape(word) + WORD, text) is not None


def _fallback_spec(text):
    t = (text or "").lower()
    for task, words in _TASK_WORDS:
        if any(_has(t, w) for w in words):
            out = {"task": task, "label": (text or task).strip()[:120]}
            if task == "circle":
                out["radius"] = 0.2
            return out
    hits = []
    for words, tgt in _DIRECTIONS:
        pos = min((m.start() for w in words
                   for m in [re.search(WORD + re.escape(w) + WORD, t)] if m),
                  default=-1)
        if pos >= 0:
            hits.append((pos, tgt))
    hits.sort(key=lambda h: h[0])
    return {"task": "goto",
            "targets": [list(g) for _, g in hits[:2]] or [[0.45, 0.35]],
            "label": (text or "reach the marker").strip()[:120]}


def design_lesson(lesson_text: str,
                  oauth_token: Optional[OAuthToken] = None) -> str:
    """Turn a plain-English lesson into a spec the trainer can optimise."""
    lesson_text = (lesson_text or "").strip() or "walk to the marker ahead of you"
    token = _token(oauth_token)
    spec = None
    if token:
        try:
            from huggingface_hub import InferenceClient
            reply = InferenceClient(token=token).chat_completion(
                model=LLM,
                messages=([{"role": "system", "content": SYSTEM}] + FEWSHOT
                          + [{"role": "user", "content": lesson_text}]),
                max_tokens=300, temperature=0.2,
            ).choices[0].message.content
            parsed, _ = L.parse_spec(reply)
            # Gate on "task", NOT on "targets": only the goto family has
            # targets, so checking those rejected every spin/circle/ball plan
            # and silently fell back to walking to a marker.
            if parsed.get("task"):
                spec = parsed
        except Exception:
            spec = None
    if spec is None:
        spec, _ = L.parse_spec(json.dumps(_fallback_spec(lesson_text)))
    spec["label"] = spec.get("label") or lesson_text
    return json.dumps(spec, indent=2)


def _filedata(path, mime):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"path": path, "url": "data:" + mime + ";base64," + b64}


def _fig(fig):
    png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig.savefig(png, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return _filedata(png, "image/png")


def _curve_chart(history, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = np.arange(1, len(history) + 1)
    best = [h[0] for h in history]
    mean = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=110)
    fig.patch.set_facecolor("white")
    ax.plot(gens, best, lw=2.2, color="#c9711f", marker="o", ms=4,
            label="best attempt so far")
    ax.plot(gens, mean, lw=1.7, color="#4a7fb5", ls="--", marker="s", ms=3.5,
            label="average of the whole batch")
    # Only the goto family has a score that means "arrived". The others are
    # open-ended (further, more turns, ball shoved harder), so no target line.
    if spec["task"] == "goto":
        n = float(len(spec["targets"]))
        ax.axhline(n, color="#2e7d32", ls=":", lw=1.4)
        ax.annotate("reaches the marker(s)", xy=(gens[-1], n), xytext=(0, 5),
                    textcoords="offset points", ha="right", fontsize=9,
                    color="#2e7d32")
    ax.set_xlabel("generation")
    ax.set_ylabel("score")
    ax.set_title("Learning to " + spec["label"][:58], fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return _fig(fig)


def _tracks_chart(spec, before, after):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = spec["task"]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=110)
    fig.patch.set_facecolor("white")

    # What the lesson was aiming at, drawn per family.
    if task == "goto":
        for i, t in enumerate(spec["targets"]):
            ax.add_patch(plt.Circle((t[0], t[1]), L.REACH, zorder=1,
                                    facecolor="#d94a2b" if i == 0 else "#2f6fbf",
                                    alpha=0.20, edgecolor="none"))
            ax.annotate(str(i + 1), xy=(t[0], t[1]), ha="center", va="center",
                        fontsize=11, weight="bold", color="#555", zorder=3)
    elif task == "circle":
        ax.add_patch(plt.Circle((0, 0), spec["radius"], fill=False, zorder=1,
                                edgecolor="#2f6fbf", ls="--", lw=1.6, alpha=0.7))
        ax.annotate("the circle to walk", xy=(0, spec["radius"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="#2f6fbf")
    elif task == "ball":
        ax.scatter([0.32], [0.0], s=260, color="#d94a2b", alpha=0.35, zorder=1)
        ax.annotate("ball", xy=(0.32, 0.0), ha="center", va="center",
                    fontsize=9, color="#7a2d18", zorder=3)

    for pts, color, label in ((before["path"], "#9aa5b1", "before training"),
                              (after["path"], "#c9711f", "after training")):
        if pts:
            arr = np.array(pts)
            ax.plot(arr[:, 0], arr[:, 1], lw=2.2, color=color, label=label,
                    zorder=2)
    ax.scatter([0], [0], s=80, marker="s", color="#2e7d32", zorder=4,
               label="start")

    # An untrained controller commands nothing, so its "path" is a dot under
    # the start marker and the legend entry looks like a bug. Say so instead.
    b = np.array(before["path"]) if before["path"] else np.zeros((1, 2))
    if float(np.abs(b - b[0]).max()) < 0.05:
        ax.annotate("before training:\nnever left the spot", xy=(0, 0),
                    xytext=(-16, -52), textcoords="offset points", fontsize=9,
                    ha="center", color="#6b7480",
                    arrowprops=dict(arrowstyle="->", color="#9aa5b1", lw=1))

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Where it walked", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return _fig(fig)


def train_policy(spec_text: str, generations: float = 8, seed: float = 0):
    """Search for a controller that solves the lesson. Returns (policy, curve)."""
    spec, notes = L.parse_spec(spec_text)
    gens, pop, ep_cost = L.plan_search(spec, int(generations or 8))
    sim = D.Microduck(render=False)
    w, history = L.train(sim, spec, generations=gens, population=pop,
                         seed=int(seed or 0))
    payload = {"weights": [round(float(x), 4) for x in w],
               "spec": spec, "history": history, "notes": notes,
               "rollouts": history[-1][2] if history else 0,
               "population": pop}
    return json.dumps(payload), _curve_chart(history, spec)


def demonstrate(spec_text: str, policy_text: str):
    """Run the untrained and the trained controller. Returns (video, tracks, report)."""
    try:
        payload = json.loads(policy_text)
        w = np.array(payload["weights"], dtype=float)
        spec = payload.get("spec") or L.parse_spec(spec_text)[0]
        history = payload.get("history") or []
        notes = payload.get("notes") or []
        rollouts = payload.get("rollouts", 0)
    except Exception as e:
        raise ValueError("could not read the trained controller: " + str(e))

    sim = D.Microduck(width=640, height=360)   # even dims: libx264 rejects odd
    before = L.rollout(sim, np.zeros(L.N_PARAMS), spec, trace=True)
    # 50 Hz control, every 2nd step photographed -> 25 fps, real time.
    after = L.rollout(sim, w, spec, trace=True, on_frame=(2, lambda s: s.frame()))

    import imageio
    mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    imageio.mimsave(mp4, after["frames"], fps=25, codec="libx264",
                    quality=7, macro_block_size=1)

    detail = {
        "goto": "markers: " + ", ".join("(%.2f, %.2f)" % (t[0], t[1])
                                        for t in spec["targets"]),
        "circle": "radius: %.2f m" % spec["radius"],
        "explore": "measured: how far it ends up from the start",
        "spin": "measured: turns completed, minus any drift",
        "ball": "measured: how far the ball ends up from where it started",
    }.get(spec["task"], "")

    lines = [
        "Lesson: " + spec["label"],
        "",
        "task family: " + spec["task"] + "  --  " + L.TASK_HELP[spec["task"]],
        detail,
        "episode: %.1f s     training: %d attempts over %d generations"
        % (spec["episode_s"], rollouts, len(history)),
        "",
        "                 result                                    score",
        "---------------------------------------------------------------",
        "before training  %-40s %+6.2f" % (before["headline"], before["score"]),
        "after training   %-40s %+6.2f" % (after["headline"], after["score"]),
        "",
    ]
    if after["all_reached"]:
        lines.append("The duck learned the lesson.")
    else:
        lines.append("Partly there. Try more generations, another seed, or an "
                     "easier version of the lesson.")
    if after["fell"]:
        lines.append("It fell during the final run.")
    lines += ["",
              "learned controller -- [forward, sideways, turn] from",
              "[1, error1, error2, error3, how far through the attempt]:"]
    W = w.reshape(L.N_OUTPUTS, L.N_FEATURES)
    for name, row in zip(("forward ", "sideways", "turn    "), W):
        lines.append("  " + name + "  " + "  ".join("%+.2f" % x for x in row))
    lines += ["",
              "The nine shipped policies were not modified. Only these "
              + str(L.N_PARAMS) + " numbers were learned."]
    if notes:
        lines += [""] + ["note: " + x for x in notes]

    return (_filedata(mp4, "video/mp4"),
            _tracks_chart(spec, before, after),
            "\n".join(lines))


BIND = {"design_lesson": design_lesson, "train_policy": train_policy,
        "demonstrate": demonstrate}
