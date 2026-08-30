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

SYSTEM = """You set lessons for Microduck, a 25 cm bipedal robot learning to
walk to places.

The duck starts at (0, 0) facing +x. In metres:
  +x is straight ahead, -x is behind it
  +y is to its left,    -y is to its right

It walks at about 0.11 m/s, so distance is a hard budget.
  ONE marker: put it 0.35 to 1.0 m from the origin.
  TWO markers (a course): keep each within 0.5 m of the origin.

Reply with ONLY a JSON object:

{"targets": [[x, y]],
 "weights": {"progress": 1.0, "speed": 0.5, "fall": 1.5},
 "label": "a short description of the lesson"}

"targets" is one or two [x, y] pairs, in the order they must be reached.
Weights (each 0-4) say what to care about: "speed" rewards arriving sooner,
"fall" punishes falling over, "progress" rewards closing the distance.
Raise the one the user emphasises. If they say nothing about it, keep the
defaults.

Output the JSON object and nothing else."""

FEWSHOT = [
    {"role": "user", "content": "teach it to walk to a spot behind it without falling over"},
    {"role": "assistant", "content":
        '{"targets": [[-0.55, 0.12]], "weights": {"progress": 1.0, "speed": 0.4, '
        '"fall": 2.5}, "label": "turn around and reach the marker behind you"}'},
]

# Direction words -> a target, for the no-token fallback.
_DIRECTIONS = [
    (("behind", "back", "backwards", "turn around", "reverse"), (-0.55, 0.12)),
    (("left",), (0.35, 0.5)),
    (("right",), (0.35, -0.5)),
    (("ahead", "forward", "front", "straight"), (0.7, 0.0)),
]


def _token(oauth_token):
    tok = getattr(oauth_token, "token", None) or os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _fallback_spec(text):
    t = (text or "").lower()
    hits = []
    for words, tgt in _DIRECTIONS:
        pos = min((m.start() for w in words
                   for m in [re.search(r"\b" + re.escape(w) + r"\b", t)] if m),
                  default=-1)
        if pos >= 0:
            hits.append((pos, tgt))
    hits.sort(key=lambda h: h[0])
    targets = [list(t) for _, t in hits[:2]] or [[0.45, 0.35]]
    weights = dict(L.DEFAULT_WEIGHTS)
    if re.search(r"\b(fast|quick|quickly|hurry|speed|sprint)\b", t):
        weights["speed"] = 1.5
    if re.search(r"\b(without falling|don't fall|stay up|steady|careful)\b", t):
        weights["fall"] = 3.0
    return {"targets": targets, "weights": weights,
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
            if parsed.get("targets"):
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
    solved = float(len(spec["targets"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=110)
    fig.patch.set_facecolor("white")
    ax.plot(gens, best, lw=2.2, color="#c9711f", marker="o", ms=4,
            label="best attempt so far")
    ax.plot(gens, mean, lw=1.7, color="#4a7fb5", ls="--", marker="s", ms=3.5,
            label="average of the whole batch")
    ax.axhline(solved, color="#2e7d32", ls=":", lw=1.4)
    ax.annotate("reaches the goal", xy=(gens[-1], solved), xytext=(0, 5),
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

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=110)
    fig.patch.set_facecolor("white")
    for pts, color, label in ((before["path"], "#9aa5b1", "before training"),
                              (after["path"], "#c9711f", "after training")):
        if pts:
            a = np.array(pts)
            ax.plot(a[:, 0], a[:, 1], lw=2.2, color=color, label=label, zorder=2)
    # Draw the goals at their true size: a run counts as arrived once the duck
    # is inside this radius, so a circle in data units makes "it got there"
    # readable instead of a dot floating near the end of a line.
    for i, t in enumerate(spec["targets"]):
        ax.add_patch(plt.Circle((t[0], t[1]), L.REACH, zorder=1,
                                facecolor="#d94a2b" if i == 0 else "#2f6fbf",
                                alpha=0.20, edgecolor="none"))
        ax.annotate(str(i + 1), xy=(t[0], t[1]), ha="center", va="center",
                    fontsize=11, weight="bold", color="#555", zorder=3)
    ax.scatter([0], [0], s=80, marker="s", color="#2e7d32", zorder=4,
               label="start")
    # An untrained controller commands nothing, so its "path" is a dot under
    # the start marker and the legend entry looks like a bug. Say so instead.
    b = np.array(before["path"]) if before["path"] else np.zeros((1, 2))
    if float(np.abs(b - b[0]).max()) < 0.05:
        ax.annotate("before training:\nnever left the spot", xy=(0, 0),
                    xytext=(-16, -52), textcoords="offset points", fontsize=9,
                    ha="center",
                    color="#6b7480",
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
    gens, pop, ep_cost = L.plan_search(spec, int(generations or 6))
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

    sim = D.Microduck(width=640, height=360)   # both even: libx264 yuv420p rejects odd
    before = L.rollout(sim, np.zeros(L.N_PARAMS), spec, trace=True)
    # 50 Hz control, every 2nd step photographed -> 25 fps, real time.
    after = L.rollout(sim, w, spec, trace=True, on_frame=(2, lambda s: s.frame()))

    import imageio
    mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    imageio.mimsave(mp4, after["frames"], fps=25, codec="libx264",
                    quality=7, macro_block_size=1)

    n = len(spec["targets"])
    lines = [
        "Lesson: " + spec["label"],
        "",
        "goal" + ("s" if n > 1 else "") + ": "
        + ", ".join("(%.2f, %.2f)" % (t[0], t[1]) for t in spec["targets"])
        + "   episode: %.1f s" % spec["episode_s"],
        "reward weights: " + ", ".join("%s %.1f" % kv
                                       for kv in spec["weights"].items()),
        "",
        "training: %d attempts over %d generations" % (rollouts, len(history)),
        "",
        "                    reached      time      distance left",
        "----------------------------------------------------------",
        "before training     %d/%d          %5.2f s     %.3f m"
        % (len(before["reached"]), n, before["t_end"], before["distance_left"]),
        "after training      %d/%d          %5.2f s     %.3f m"
        % (len(after["reached"]), n, after["t_end"], after["distance_left"]),
        "",
        "score %+.2f  ->  %+.2f" % (before["score"], after["score"]),
    ]
    if after["all_reached"]:
        lines.append("The duck learned the lesson.")
    else:
        lines.append("Not solved this time. Try more generations, a different "
                     "seed, or a closer marker.")
    if after["fell"]:
        lines.append("It fell during the final run.")
    lines += ["",
              "learned controller (vx and turn rate from "
              "[1, distance, cos(heading error), sin(heading error)]):",
              "  vx = " + "  ".join("%+.3f" % x for x in w[:4]),
              "  wz = " + "  ".join("%+.3f" % x for x in w[4:]),
              "",
              "The nine shipped policies were not modified. Only these eight "
              "numbers were learned."]
    if notes:
        lines += [""] + ["note: " + x for x in notes]

    return (_filedata(mp4, "video/mp4"),
            _tracks_chart(spec, before, after),
            "\n".join(lines))


BIND = {"design_lesson": design_lesson, "train_policy": train_policy,
        "demonstrate": demonstrate}
