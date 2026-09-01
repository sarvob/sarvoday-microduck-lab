"""Teaching Microduck a new skill, without touching its trained gaits.

The nine shipped policies stay frozen. What gets LEARNED here is the layer
above them: a small controller that watches the world and decides what to ask
the walking policy for -- forward speed, sideways speed and turn rate, ten
times a second. That is the same split real robots use: a locomotion
controller you do not retrain, and a task policy on top that you do.

    [vx, vy, wz] = W @ [1, e1, e2, e3, phase]

`phase` is how far through the attempt it is, which lets a behaviour change
over time. `e1..e3` are the error signals for THIS task -- distance and
heading to a marker, distance and heading to the ball, how far off a circle
it is. 15 numbers, small enough to optimise by search rather than gradients:
no autograd, no GPU, no replay buffer.

There are five task families, and they score genuinely different things.
Adding a sixth means adding a `features_*` and a `score_*` here -- not
touching the workflow, the app or the optimiser.
"""

import json

import numpy as np

import duck as D

N_FEATURES = 5
N_OUTPUTS = 3
N_PARAMS = N_FEATURES * N_OUTPUTS

CMD_EVERY = 5              # re-decide the command at 10 Hz
REACH = 0.15               # metres; close enough to count as arrived
MAX_EPISODE_S = 14.0
MAX_TARGETS = 2
REACH_LIMIT_ONE = 0.8
REACH_LIMIT_COURSE = 0.5
SIGMA_FLOOR = 0.15

TASKS = ("goto", "explore", "spin", "circle", "ball")

TASK_HELP = {
    "goto": "reach one or two markers, in order",
    "explore": "get as far from the start as possible",
    "spin": "turn on the spot without wandering off",
    "circle": "walk a circle of a given radius around the start",
    "ball": "shove the ball as far as possible",
}


# -- the lesson spec -----------------------------------------------------
def parse_spec(text):
    """Read a lesson spec (JSON text). Returns (spec, notes). Tolerant: this
    usually arrives from a language model."""
    notes = []
    raw = (text or "").strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]
    try:
        spec = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        notes.append("could not read the lesson (" + str(e) + ")")
        spec = {}
    if not isinstance(spec, dict):
        spec = {}

    task = str(spec.get("task") or "goto").strip().lower()
    if task not in TASKS:
        notes.append("unknown task " + repr(task) + "; using goto")
        task = "goto"

    targets = spec.get("targets") or spec.get("target") or []
    if targets and not isinstance(targets[0], (list, tuple)):
        targets = [targets]
    wanted = targets[:MAX_TARGETS]
    limit = REACH_LIMIT_ONE if len(wanted) < 2 else REACH_LIMIT_COURSE
    clean = []
    for t in wanted:
        try:
            x, y = float(t[0]), float(t[1])
        except (TypeError, ValueError, IndexError):
            continue
        r = float(np.hypot(x, y))
        if r > limit:
            x, y = x * limit / r, y * limit / r
            notes.append("marker pulled in to %.2f m -- the duck sustains "
                         "about 0.11 m/s" % limit)
        elif r < 0.25:
            a = np.arctan2(y, x) if (x or y) else 0.0
            x, y = 0.35 * np.cos(a), 0.35 * np.sin(a)
            notes.append("marker was almost underfoot; pushed out to 0.35 m")
        clean.append([round(x, 3), round(y, 3)])
    if task == "goto" and not clean:
        clean = [[0.45, 0.35]]
        notes.append("no usable marker; using one 0.57 m ahead-left")

    try:
        radius = float(np.clip(float(spec.get("radius", 0.2)), 0.12, 0.25))
    except (TypeError, ValueError):
        radius = 0.2

    try:
        ep = float(spec.get("episode_s", 0))
    except (TypeError, ValueError):
        ep = 0.0
    if task == "goto":
        need = 2.0 + 11.0 * sum(float(np.hypot(*t)) for t in clean) + 1.5 * len(clean)
    elif task == "circle":
        # One lap is 2*pi*r of walking at ~0.11 m/s. A 0.5 m circle would
        # need 29 s, well past the cap, so the search just learned a tiny
        # circle instead -- the radius has to stay lap-able.
        need = 3.0 + 2 * np.pi * radius / 0.105
    else:
        need = 9.0
    ep = float(np.clip(max(ep, min(need, MAX_EPISODE_S)), 4.0, MAX_EPISODE_S))

    return {"task": task, "targets": clean, "radius": round(radius, 2),
            "episode_s": round(ep, 1),
            "label": str(spec.get("label") or TASK_HELP[task])[:120]}, notes


def markers_for(spec):
    """Discs to draw on the floor, so the video shows what it was aiming at."""
    if spec["task"] == "goto":
        return spec["targets"]
    if spec["task"] == "circle":
        r = spec["radius"]
        return [[r, 0.0], [-r, 0.0]]
    return []


# -- per-task error signals ----------------------------------------------
def _bearing(sim, tx, ty):
    x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
    dx, dy = tx - x, ty - y
    d = float(np.hypot(dx, dy))
    he = float(np.arctan2(dy, dx) - sim.yaw())
    return d, (he + np.pi) % (2 * np.pi) - np.pi


def features(sim, spec, st, phase):
    """[1, e1, e2, e3, phase] -- the three error signals depend on the task."""
    task = spec["task"]
    if task == "goto":
        d, he = _bearing(sim, *spec["targets"][st["idx"]])
        e = (min(d, 1.5), np.cos(he), np.sin(he))
    elif task == "ball":
        d, he = _bearing(sim, *sim.ball_xy())
        e = (min(d, 1.5), np.cos(he), np.sin(he))
    elif task == "circle":
        x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
        r = float(np.hypot(x, y))
        # tangent direction: 90 deg from the outward radius
        tangent = np.arctan2(y, x) + np.pi / 2
        te = (tangent - sim.yaw() + np.pi) % (2 * np.pi) - np.pi
        e = (r - spec["radius"], np.cos(te), np.sin(te))
    elif task == "explore":
        x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
        r = float(np.hypot(x, y))
        away = np.arctan2(y, x) if r > 1e-6 else sim.yaw()
        he = (away - sim.yaw() + np.pi) % (2 * np.pi) - np.pi
        e = (min(r, 2.0), np.cos(he), np.sin(he))
    else:                                   # spin: no place to go
        e = (0.0, 0.0, 0.0)
    return np.array([1.0, e[0], e[1], e[2], phase])


def command(W, f):
    out = W @ f
    return (float(np.clip(out[0], D.VEL_BACK, D.VEL_FWD)),
            float(np.clip(out[1], -0.15, 0.15)),
            float(np.clip(out[2], -D.VEL_ANG, D.VEL_ANG)))


# -- rollout -------------------------------------------------------------
def rollout(sim, w, spec, trace=False, on_frame=None):
    """Run one attempt and score it. Returns a dict of what happened."""
    W = np.asarray(w, dtype=float).reshape(N_OUTPUTS, N_FEATURES)
    task = spec["task"]
    steps = int(spec["episode_s"] / D.CTRL_DT)

    sim.reset()
    sim.set_targets(markers_for(spec))
    if task == "ball":
        sim.place_ball(0.32, 0.0)
        ball0 = np.array(sim.ball_xy())
    else:
        sim.park_ball()
        ball0 = None

    st = {"idx": 0}
    reached, path, frames = [], [], []
    spun, radial_err, ticks, orbit = 0.0, 0.0, 0, 0.0
    last_yaw = sim.yaw()
    last_polar = float(np.arctan2(sim.data.qpos[1], sim.data.qpos[0]))
    legs_d0 = [_bearing(sim, *spec["targets"][0])[0]] if task == "goto" else [1.0]
    cmd = (0.0, 0.0, 0.0)
    fell = False
    halfway = steps // 2

    for i in range(steps):
        phase = i / max(steps - 1, 1)
        if i % CMD_EVERY == 0:
            cmd = command(W, features(sim, spec, st, phase))
        sim.control_step("walk", cmd)
        if trace:
            path.append((float(sim.data.qpos[0]), float(sim.data.qpos[1])))
        if on_frame is not None and i % on_frame[0] == 0:
            frames.append(on_frame[1](sim))

        yaw = sim.yaw()
        spun += abs((yaw - last_yaw + np.pi) % (2 * np.pi) - np.pi)
        last_yaw = yaw
        if task == "circle":
            r = float(np.hypot(sim.data.qpos[0], sim.data.qpos[1]))
            radial_err += abs(r - spec["radius"])
            ticks += 1
            # Angle swept AROUND the start, not how much the body turned:
            # scoring body yaw let the duck pirouette on the spot and collect
            # the whole reward without ever walking a circle.
            polar = float(np.arctan2(sim.data.qpos[1], sim.data.qpos[0]))
            if r > 0.1:
                orbit += abs((polar - last_polar + np.pi) % (2 * np.pi) - np.pi)
            last_polar = polar

        if task == "goto":
            d, _ = _bearing(sim, *spec["targets"][st["idx"]])
            if d < REACH:
                reached.append(round(i * D.CTRL_DT, 2))
                st["idx"] += 1
                if st["idx"] >= len(spec["targets"]):
                    break
                legs_d0.append(_bearing(sim, *spec["targets"][st["idx"]])[0])
            if i == halfway and not reached and d > 0.95 * legs_d0[0]:
                break                      # hopeless candidate, stop early

        if i > 30 and float(sim.proj_gravity()[2]) > -0.5:
            fell = True
            break

    t_end = (i + 1) * D.CTRL_DT
    x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
    travelled = float(np.hypot(x, y))
    rec = {"reached": reached, "fell": fell, "t_end": round(t_end, 2),
           "travelled": round(travelled, 3), "spun_rad": round(spun, 2),
           "path": path, "frames": frames, "distance_left": 0.0,
           "all_reached": False}

    # -- scoring, per task ------------------------------------------------
    if task == "goto":
        d, _ = _bearing(sim, *spec["targets"][min(st["idx"], len(spec["targets"]) - 1)])
        frac = float(np.clip((legs_d0[-1] - d) / max(legs_d0[-1], 1e-6), -1.0, 1.0))
        score = len(reached) + frac - 1.5 * fell
        if len(reached) == len(spec["targets"]):
            score += 0.5 * (1.0 - t_end / spec["episode_s"])
        rec["distance_left"] = round(d, 3)
        rec["all_reached"] = len(reached) == len(spec["targets"])
        rec["headline"] = "%d/%d markers reached" % (len(reached), len(spec["targets"]))
    elif task == "explore":
        score = travelled - 1.5 * fell
        rec["all_reached"] = travelled > 0.6
        rec["headline"] = "%.2f m from the start" % travelled
    elif task == "spin":
        # Turning is the point; wandering away from the spot is not.
        score = spun - 2.0 * travelled - 1.5 * fell
        rec["all_reached"] = spun > 2 * np.pi and travelled < 0.5
        rec["headline"] = "%.1f full turns, drifted %.2f m" % (spun / (2 * np.pi), travelled)
    elif task == "circle":
        mean_err = radial_err / max(ticks, 1)
        score = 2.0 * orbit - 6.0 * mean_err - 1.5 * fell
        rec["all_reached"] = orbit > 0.6 * np.pi and mean_err < 0.22
        rec["headline"] = ("%.2f of a lap around the start, %.2f m off the circle"
                           % (orbit / (2 * np.pi), mean_err))
    else:                                   # ball
        moved = float(np.hypot(*(np.array(sim.ball_xy()) - ball0)))
        score = 3.0 * moved - 1.5 * fell
        rec["all_reached"] = moved > 0.25
        rec["headline"] = "ball shoved %.2f m" % moved
        rec["ball_moved"] = round(moved, 3)

    rec["score"] = float(score)
    return rec


# -- the optimiser -------------------------------------------------------
STEP_COST_S = 0.002
PHYSICS_BUDGET_S = 220.0


def plan_search(spec, generations):
    """Population sized from a fixed physics budget, so a long episode costs
    the same wall clock as a short one."""
    ep_cost = spec["episode_s"] / D.CTRL_DT * STEP_COST_S
    gens = int(np.clip(int(generations), 3, 14))
    pop = int(round(PHYSICS_BUDGET_S / max(gens * ep_cost, 1e-6)))
    return gens, int(np.clip(pop, 10, 26)), ep_cost


def train(sim, spec, generations=8, population=12, elite=4, seed=0,
          sigma0=1.2, on_generation=None):
    """Cross-entropy method over the controller weights.

    Returns (best_w, history); history is (best_so_far, batch_mean, rollouts).
    """
    rng = np.random.default_rng(int(seed))
    mu = np.zeros(N_PARAMS)
    sigma = np.full(N_PARAMS, float(sigma0))
    best_w, best_score = np.zeros(N_PARAMS), -1e9
    history, used, stale = [], 0, 0

    for g in range(int(generations)):
        pop = rng.normal(mu, sigma, size=(int(population), N_PARAMS))
        scored = []
        for p in pop:
            scored.append((rollout(sim, p, spec)["score"], p))
            used += 1
        scored.sort(key=lambda s: -s[0])
        # Keep the best INDIVIDUAL, never the elite mean: averaging two good
        # but different controllers can land between them and score far worse.
        improved = scored[0][0] > best_score + 1e-6
        if improved:
            best_score, best_w = scored[0][0], scored[0][1].copy()
        el = np.array([p for _, p in scored[:int(elite)]])
        mu = el.mean(axis=0)
        # Floor the spread, or the elites agree by generation 3 and the search
        # stops; restart outright when it has converged on something that
        # still is not solving the lesson.
        sigma = np.maximum(el.std(axis=0), SIGMA_FLOOR)
        stale = 0 if improved else stale + 1
        if stale >= 3 and not rollout(sim, best_w, spec)["all_reached"]:
            mu = np.zeros(N_PARAMS)
            sigma = np.full(N_PARAMS, float(sigma0))
            stale = 0
        history.append((round(best_score, 4),
                        round(float(np.mean([s for s, _ in scored])), 4), used))
        if on_generation:
            on_generation(g, history[-1])

    return best_w, history
