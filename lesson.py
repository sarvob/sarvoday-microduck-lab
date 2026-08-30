"""Teaching Microduck a new skill, without touching its trained gaits.

The nine shipped policies stay frozen. What gets LEARNED here is the layer
above them: a small controller that watches where the duck is and decides what
to ask the walking policy for -- a forward speed and a turn rate, twenty times
a second. That is the same split real robots use: a locomotion controller you
do not retrain, and a task policy on top that you do.

The learned controller is a 2x4 matrix, 8 numbers:

    [vx, wz] = W @ [1, distance_to_goal, cos(heading_error), sin(heading_error)]

8 numbers is small enough to optimise by search rather than gradients, which
is what makes this fit in a CPU Space: no autograd, no GPU, no replay buffer.
The optimiser is CEM -- sample a population, keep the best few, refit the
distribution to them, repeat.
"""

import json

import numpy as np

import duck as D

N_PARAMS = 8
CMD_EVERY = 5              # re-decide the command at 10 Hz
REACH = 0.15               # metres; close enough to count as arrived
MAX_EPISODE_S = 14.0
MAX_TARGETS = 2
# The duck sustains only ~0.11 m/s, so reachability is a hard budget, not a
# preference: one marker can sit up to 1.0 m out, but a two-marker course has
# to keep each leg short or the episode ends mid-course.
REACH_LIMIT_ONE = 0.8
REACH_LIMIT_COURSE = 0.5

SIGMA_FLOOR = 0.15         # keep exploring; see train()
DEFAULT_WEIGHTS = {"progress": 1.0, "speed": 0.5, "fall": 1.5}


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
        notes.append("could not read the lesson (" + str(e) + "); "
                     "falling back to a marker straight ahead")
        spec = {}
    if not isinstance(spec, dict):
        spec = {}

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
        if r > limit:                  # pull an over-ambitious goal back in
            x, y = x * limit / r, y * limit / r
            notes.append("goal pulled in to %.2f m -- the duck sustains about "
                         "0.11 m/s, so anything further cannot be reached in "
                         "one episode" % limit)
        if float(np.hypot(x, y)) < 0.25:
            notes.append("goal was almost underfoot; pushed out to 0.35 m")
            a = np.arctan2(y, x) if (x or y) else 0.0
            x, y = 0.35 * np.cos(a), 0.35 * np.sin(a)
        clean.append([round(x, 3), round(y, 3)])
    if not clean:
        clean = [[0.45, 0.35]]
        notes.append("no usable goal in the lesson; using one 0.57 m ahead-left")

    w = dict(DEFAULT_WEIGHTS)
    for k, v in (spec.get("weights") or {}).items():
        if k in w:
            try:
                w[k] = float(np.clip(float(v), 0.0, 4.0))
            except (TypeError, ValueError):
                pass

    try:
        ep = float(spec.get("episode_s", 7.0))
    except (TypeError, ValueError):
        ep = 7.0
    # Needs to be long enough to walk there and back off a bad start.
    # 11 s per metre of walking, plus a couple of seconds per turn.
    need = 2.0 + 11.0 * sum(float(np.hypot(*t)) for t in clean) + 1.5 * len(clean)
    ep = float(np.clip(max(ep, min(need, MAX_EPISODE_S)), 4.0, MAX_EPISODE_S))

    return {"targets": clean, "weights": w, "episode_s": round(ep, 1),
            "label": str(spec.get("label") or "reach the marker")[:120]}, notes


# -- the learned controller ----------------------------------------------
def _leg(sim, target):
    x, y = float(sim.data.qpos[0]), float(sim.data.qpos[1])
    dx, dy = target[0] - x, target[1] - y
    d = float(np.hypot(dx, dy))
    he = float(np.arctan2(dy, dx) - sim.yaw())
    he = (he + np.pi) % (2 * np.pi) - np.pi
    return np.array([1.0, min(d, 1.5), np.cos(he), np.sin(he)]), d


def command(W, sim, target):
    out = W @ _leg(sim, target)[0]
    return (float(np.clip(out[0], D.VEL_BACK, D.VEL_FWD)), 0.0,
            float(np.clip(out[1], -D.VEL_ANG, D.VEL_ANG)))


def rollout(sim, w, spec, trace=False, on_frame=None):
    """Run one attempt. Returns a dict of what happened."""
    W = np.asarray(w, dtype=float).reshape(2, 4)
    targets = spec["targets"]
    ep, wt = spec["episode_s"], spec["weights"]
    steps = int(ep / D.CTRL_DT)

    sim.reset()
    sim.set_targets(targets)
    halfway = steps // 2
    idx, reached, path, frames = 0, [], [], []
    _, leg_d0 = _leg(sim, targets[0])
    legs_d0 = [leg_d0]
    cmd = (0.0, 0.0, 0.0)
    fell = False

    for i in range(steps):
        if i % CMD_EVERY == 0:
            cmd = command(W, sim, targets[idx])
        sim.control_step("walk", cmd)
        if trace:
            path.append((float(sim.data.qpos[0]), float(sim.data.qpos[1])))
        if on_frame is not None and i % on_frame[0] == 0:
            frames.append(on_frame[1](sim))
        _, d = _leg(sim, targets[idx])
        if d < REACH:
            reached.append(round(i * D.CTRL_DT, 2))
            idx += 1
            if idx >= len(targets):
                break
            legs_d0.append(_leg(sim, targets[idx])[1])
        # A fall ends the attempt: the get-up policy would rescue it, but the
        # lesson is about walking there, not about recovering.
        if i > 30 and float(sim.proj_gravity()[2]) > -0.5:
            fell = True
            break
        # Give up on a candidate that has made no headway by halfway. Most of
        # an early population just stands still or wanders off, and running
        # those to the end is most of the training cost for no information.
        if i == halfway and not reached and d > 0.95 * legs_d0[0]:
            break

    t_end = (i + 1) * D.CTRL_DT
    _, d_final = _leg(sim, targets[min(idx, len(targets) - 1)])
    frac = float(np.clip((legs_d0[-1] - d_final) / max(legs_d0[-1], 1e-6), -1.0, 1.0))
    score = len(reached) + wt["progress"] * frac - wt["fall"] * fell
    if len(reached) == len(targets):
        score += wt["speed"] * (1.0 - t_end / ep)     # reward getting there sooner

    return {"score": float(score), "reached": reached, "fell": fell,
            "distance_left": round(float(d_final), 3), "t_end": round(t_end, 2),
            "path": path, "frames": frames,
            "all_reached": len(reached) == len(targets)}


# -- the optimiser -------------------------------------------------------
# Roughly how long one control step costs, measured on this machine. Only used
# to size the search; being off by 2x just makes training longer or shorter.
STEP_COST_S = 0.002
PHYSICS_BUDGET_S = 220.0  # nominal; most candidates abort early, so the real
                          # cost lands near a third of this


def plan_search(spec, generations):
    """Pick a population that keeps total training time roughly constant.

    Episode length is the cost driver and it scales with how far the goal is:
    a 14 s episode is 700 control steps, so a fixed 12-wide population makes a
    distant lesson take three times as long as a near one. Trading population
    for episode length keeps every lesson in the same ballpark.
    """
    ep_cost = spec["episode_s"] / D.CTRL_DT * STEP_COST_S
    gens = int(np.clip(int(generations), 3, 14))
    pop = int(round(PHYSICS_BUDGET_S / max(gens * ep_cost, 1e-6)))
    return gens, int(np.clip(pop, 10, 26)), ep_cost


def train(sim, spec, generations=8, population=12, elite=4, seed=0,
          sigma0=1.2, on_generation=None):
    """Cross-entropy method over the 8 controller weights.

    Returns (best_w, history). `history` is per generation:
    (best_so_far, population_mean, rollouts_used).
    """
    rng = np.random.default_rng(int(seed))
    mu = np.zeros(N_PARAMS)
    sigma = np.full(N_PARAMS, float(sigma0))
    best_w, best_score = np.zeros(N_PARAMS), -1e9
    history, used = [], 0
    solved = float(len(spec["targets"]))   # score >= this means it got there
    stale = 0

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
        # Floor the spread. Without it the elites agree, sigma collapses to
        # ~0.03 by generation 3 and the search stops -- which is exactly how a
        # hard lesson (a marker BEHIND the duck) got stuck on a mediocre
        # controller and never found the turn-around.
        mu = el.mean(axis=0)
        sigma = np.maximum(el.std(axis=0), SIGMA_FLOOR)

        # Restart when the search has converged on something that does not
        # actually solve the lesson. A floor keeps it wandering; only a restart
        # gets it out of the wrong basin, which is what a hard turn-around
        # needs on an unlucky draw.
        stale = 0 if improved else stale + 1
        if stale >= 3 and best_score < solved:
            mu = np.zeros(N_PARAMS)
            sigma = np.full(N_PARAMS, float(sigma0))
            stale = 0
        history.append((round(best_score, 4),
                        round(float(np.mean([s for s, _ in scored])), 4), used))
        if on_generation:
            on_generation(g, history[-1])

    return best_w, history
