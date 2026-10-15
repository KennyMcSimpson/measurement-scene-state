"""EXP-3D-CAJU-HEADROOM-V1 (CAJU-V1 Gate 0): the frozen episode, readout, utility, oracle and gate logic.

Free of torch and DepthSplat, so the unit tests and every CPU stage import it directly.
- Episodes (record x role) follow EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3: context = the role's frames; stream =
  stream_frame_ids (four evenly spaced free frames, shared by both roles); queries = primary_query, which come after
  every arrived frame. History views = the other free frames: never measured, all between the first and the last
  stream frame.
- Actions start from the same CACHE state: OFF (0 TTT steps), SHORT (10), FULL (100). Every action is read out by the
  frozen Core B readout CORE_B = sqrt(R7 x F7), F7 = ray depth where alpha >= 0.5, else R7.
- Quality utility of an action = AbsRel(OFF) - AbsRel(action), averaged over the episode's queries. The statistical
  unit is the physical place (one ARKitScenes venue, one scan per venue here); a place's value is the mean over its
  episodes (roles).
"""

import random
from pathlib import Path

import numpy as np

ACTIONS = ("OFF", "SHORT", "FULL")
STEPS = {"OFF": 0, "SHORT": 10, "FULL": 100}
ALPHA_MIN = 0.5
SELECTION_SEED = 20261014
N_SEL, N_EVAL = 16, 34
DRAWS, BOOT_SEED = 10000, 20260928
DELTA_MIN = 0.0010  # practical minimum quality headroom (place-mean AbsRel), fixed before any utility is seen
EPISODE_BAND = 0.0005  # per-episode band: beneficial / harmful update, and the cheapest-sufficient tolerance
TOP1_MAX = 0.5
COMPUTE_SAVING_MIN = 0.5
COMPUTE_LOSS_MEAN_MAX = 0.0003
COMPUTE_LOSS_CI_MAX = 0.0005
FIXED_RULE_TOLERANCE = 0.0005
RELIABLE_ERROR = 0.05  # observed views: a pixel is reliable when OFF's normalized depth error is at most 5 %
LAMBDAS = (0.0,) + tuple(float(x) for x in np.logspace(-8, -3, 51))  # AbsRel per TTT step, Pareto sweep


# ---------------------------------------------------------------- episodes and places


def free_frames(record):
    """Frames in no context or query role (the pool stream_frame_ids draws its four frames from)."""
    roles = record["roles"]
    used = set(roles["context_a"]) | set(roles["context_b"]) | set(roles["primary_query"])
    used |= set(roles.get("query", ())) | set(roles.get("secondary_query", ()))
    return sorted(f["frame_id"] for f in record["frames"] if f["frame_id"] not in used)


def build_episodes(record, stream):
    """The two episodes of a record. `stream` is stream_frame_ids(record, "A") (the caller checks that role B gets
    the same stream); history views are the free frames the stream leaves out."""
    if stream is None:
        raise ValueError(f"{record['scene_id']}: no four-frame stream")
    stream = list(stream)
    history = [f for f in free_frames(record) if f not in stream]
    queries = list(record["roles"]["primary_query"])
    episodes = []
    for role, key in (("A", "context_a"), ("B", "context_b")):
        context = list(record["roles"][key])
        inputs = set(context) | set(stream)
        if inputs & set(queries) or inputs & set(history) or set(queries) & set(history):
            raise ValueError(f"{record['scene_id']}/{role}: frame roles overlap")
        if min(queries) <= max(inputs):
            raise ValueError(f"{record['scene_id']}/{role}: a query does not come after every arrived frame")
        if history and not (min(stream) < min(history) and max(history) < max(stream)):
            raise ValueError(f"{record['scene_id']}/{role}: a history view lies outside the stream span")
        episodes.append({"scene_id": record["scene_id"], "place": record["physical_scene_id"], "role": role,
                         "context": context, "stream": stream, "queries": queries, "history": history})
    return episodes


def select_places(train_places):
    """Seeded split of the training places: 16 for the fixed-action selection, 34 for the headroom evaluation; the
    rest stays untouched (reserved for a possible later gate)."""
    places = sorted(set(train_places))
    if len(places) < N_SEL + N_EVAL:
        raise ValueError("not enough training places")
    random.Random(SELECTION_SEED).shuffle(places)
    return {"CAJU_SEL": sorted(places[:N_SEL]), "CAJU_EVAL": sorted(places[N_SEL:N_SEL + N_EVAL]),
            "RESERVED_UNTOUCHED": sorted(places[N_SEL + N_EVAL:])}


class AccessAudit:
    """Every data file opened while an episode runs; a file outside the episode's arrived frames is a violation."""

    def __init__(self):
        self.allowed, self.opened = set(), []

    def begin(self, allowed):
        self.allowed, self.opened = {str(Path(p).resolve()) for p in allowed}, []

    def note(self, path):
        self.opened.append(str(Path(str(path)).resolve()))

    def violations(self):
        return sorted({p for p in self.opened if p not in self.allowed})


# ---------------------------------------------------------------- readout and losses


def core_b(depth_z, alpha, r7, cosine):
    """V3's readout: ray = depth_z / alpha / cosine (alpha floored at 1e-12); F7 = ray where alpha >= 0.5, else R7;
    CORE_B = sqrt(R7 x F7) with both floored at 1e-6. Returns (CORE_B, F7)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    ray = np.asarray(depth_z, dtype=np.float64) / np.maximum(alpha, 1e-12) / cosine
    f7 = np.where(alpha >= ALPHA_MIN, ray, r7)
    return np.sqrt(np.maximum(r7, 1e-6) * np.maximum(f7, 1e-6)), f7


def absrel(prediction, gt, mask=None):
    """AbsRel on valid ground truth (finite, > 0), optionally within a mask; None when nothing is scored."""
    valid = np.isfinite(gt) & (gt > 0)
    if mask is not None:
        valid &= mask
    return float(np.mean(np.abs(prediction[valid] - gt[valid]) / gt[valid])) if valid.any() else None


# ---------------------------------------------------------------- utilities and policies


def episodes_from_rows(rows, kind="query", metric="loss"):
    """{episode: {"place": p, "frames": {frame: {action: loss}}}}; frames lacking a loss for some action are left
    out."""
    out = {}
    for row in rows:
        if row["kind"] != kind or any(row[metric][a] is None for a in ACTIONS):
            continue
        entry = out.setdefault(row["episode"], {"place": row["place"], "frames": {}})
        entry["frames"][row["frame"]] = {a: float(row[metric][a]) for a in ACTIONS}
    return out


def utilities(frames):
    """Quality utility per action = loss(OFF) - loss(action), each loss the mean over the frames."""
    mean = {a: float(np.mean([f[a] for f in frames.values()])) for a in ACTIONS}
    return {a: mean["OFF"] - mean[a] for a in ACTIONS}


def best_action(u):
    """The highest utility; ties (to 1e-12) go to the cheaper action."""
    return max(ACTIONS, key=lambda a: (round(u[a], 12), -STEPS[a]))


def cheapest_sufficient(u, reference, band=EPISODE_BAND):
    """The cheapest action whose utility is within `band` of the reference action's utility."""
    return min((a for a in ACTIONS if u[a] >= u[reference] - band), key=lambda a: STEPS[a])


def _cross(frames, chooser):
    """Choose on each frame alone, evaluate on the remaining frames; the values and costs averaged over the choices.
    None when the episode has fewer than two scored frames."""
    ids = sorted(frames)
    if len(ids) < 2:
        return None
    values, costs, picks = [], [], []
    for f in ids:
        pick = chooser(utilities({f: frames[f]}))
        held = utilities({g: frames[g] for g in ids if g != f})
        values.append(held[pick])
        costs.append(STEPS[pick])
        picks.append(pick)
    return float(np.mean(values)), float(np.mean(costs)), picks


def episode_policies(frames, fixed):
    """Value (quality utility) and cost (TTT steps) of every policy on one episode. Cross-query policies fall back to
    the fixed action when the episode has a single scored query (no held-out frame to evaluate a choice on)."""
    u = utilities(frames)
    value = {f"FIXED_{a}": u[a] for a in ACTIONS}
    cost = {f"FIXED_{a}": float(STEPS[a]) for a in ACTIONS}
    picks = {}
    value["FIX"], cost["FIX"] = u[fixed], float(STEPS[fixed])
    oracle = best_action(u)
    value["ORACLE"], cost["ORACLE"], picks["ORACLE"] = u[oracle], float(STEPS[oracle]), [oracle]
    sufficient = cheapest_sufficient(u, fixed)
    value["SUFFICIENT"], cost["SUFFICIENT"], picks["SUFFICIENT"] = u[sufficient], float(STEPS[sufficient]), [sufficient]
    for name, chooser in (("XORACLE", best_action), ("XSUFFICIENT", lambda x: cheapest_sufficient(x, fixed))):
        crossed = _cross(frames, chooser)
        if crossed is None:
            value[name], cost[name], picks[name] = u[fixed], float(STEPS[fixed]), [fixed]
        else:
            value[name], cost[name], picks[name] = crossed
    return value, cost, picks


def place_policies(episode_frames, fixed):
    """Place-level policies over a place's episodes (roles): VORACLE = one action for the whole place chosen on its
    mean utility (in-sample); XROLE = choose on one role, evaluate on the other(s) (the fixed action when the place
    has a single episode)."""
    us = [utilities(frames) for frames in episode_frames]
    mean = {a: float(np.mean([u[a] for u in us])) for a in ACTIONS}
    vor = best_action(mean)
    value = {"VORACLE": mean[vor]}
    cost = {"VORACLE": float(STEPS[vor])}
    if len(us) < 2:
        value["XROLE"], cost["XROLE"] = mean[fixed], float(STEPS[fixed])
    else:
        vals, costs = [], []
        for i, u in enumerate(us):
            pick = best_action(u)
            vals.append(float(np.mean([us[j][pick] for j in range(len(us)) if j != i])))
            costs.append(STEPS[pick])
        value["XROLE"], cost["XROLE"] = float(np.mean(vals)), float(np.mean(costs))
    return value, cost


def place_means(per_episode, episode_place):
    """{place: mean over the place's episodes} of a per-episode value."""
    grouped = {}
    for episode, v in per_episode.items():
        grouped.setdefault(episode_place[episode], []).append(v)
    return {p: float(np.mean(v)) for p, v in grouped.items()}


def policy_tables(episodes, fixed):
    """Place tables {policy: {place: value}} and {policy: {place: cost}} for every episode- and place-level policy,
    plus the per-episode picks."""
    episode_place = {e: v["place"] for e, v in episodes.items()}
    values, costs, picks = {}, {}, {}
    for e, entry in episodes.items():
        v, c, p = episode_policies(entry["frames"], fixed)
        for k in v:
            values.setdefault(k, {})[e] = v[k]
            costs.setdefault(k, {})[e] = c[k]
        picks[e] = p
    value_table = {k: place_means(v, episode_place) for k, v in values.items()}
    cost_table = {k: place_means(v, episode_place) for k, v in costs.items()}
    by_place = {}
    for e, entry in episodes.items():
        by_place.setdefault(entry["place"], []).append(entry["frames"])
    for place, frames_list in by_place.items():
        v, c = place_policies(frames_list, fixed)
        for k in v:
            value_table.setdefault(k, {})[place] = v[k]
            cost_table.setdefault(k, {})[place] = c[k]
    return value_table, cost_table, picks


def select_fixed(episodes):
    """The best fixed action on the selection places: highest mean place utility, ties to the cheaper action."""
    episode_place = {e: v["place"] for e, v in episodes.items()}
    u = {a: float(np.mean(list(place_means({e: utilities(v["frames"])[a] for e, v in episodes.items()},
                                           episode_place).values()))) for a in ACTIONS}
    return best_action(u), u


# ---------------------------------------------------------------- Pareto


def pareto_points(episodes, mode, lambdas=LAMBDAS):
    """(mean place cost in TTT steps, mean place utility) of per-episode choices maximizing u(a) - lambda x steps(a).
    mode 'in_sample' chooses and evaluates on the same queries; 'cross_query' chooses on one query and evaluates on
    the other(s) (episodes with a single scored query use the in-sample choice)."""
    episode_place = {e: v["place"] for e, v in episodes.items()}
    points = []
    for lam in lambdas:
        def chooser(u, lam=lam):
            return max(ACTIONS, key=lambda a: (round(u[a] - lam * STEPS[a], 12), -STEPS[a]))

        vals, costs = {}, {}
        for e, entry in episodes.items():
            crossed = _cross(entry["frames"], chooser) if mode == "cross_query" else None
            if crossed is None:
                u = utilities(entry["frames"])
                pick = chooser(u)
                vals[e], costs[e] = u[pick], float(STEPS[pick])
            else:
                vals[e], costs[e] = crossed[0], crossed[1]
        v = float(np.mean(list(place_means(vals, episode_place).values())))
        c = float(np.mean(list(place_means(costs, episode_place).values())))
        points.append({"lambda": lam, "steps": c, "utility": v})
    return points


def frontier(points):
    """The non-dominated points (lower cost, higher utility), sorted by cost."""
    best, out = -np.inf, []
    for p in sorted(points, key=lambda p: (p["steps"], -p["utility"])):
        if p["utility"] > best + 1e-15:
            out.append(p)
            best = p["utility"]
    return out


def fixed_hull_value(fixed_points, steps):
    """The best utility a random mixture of fixed actions reaches at a mean cost of at most `steps` (upper concave
    envelope of the fixed points)."""
    pts = sorted(fixed_points, key=lambda p: p["steps"])
    best = max(p["utility"] for p in pts if p["steps"] <= steps + 1e-12)
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            if a["steps"] <= steps <= b["steps"] and b["steps"] > a["steps"]:
                w = (steps - a["steps"]) / (b["steps"] - a["steps"])
                best = max(best, (1 - w) * a["utility"] + w * b["utility"])
    return best


# ---------------------------------------------------------------- gate


def quality_checks(stats):
    """Primary: cross-query oracle minus best fixed action, per place (describe() of the project statistics)."""
    return {
        "mean_at_least_delta_min": stats["mean"] >= DELTA_MIN,
        "ci95_low_positive": stats["ci95"][0] > 0,
        "every_leave_one_place_out_mean_positive": bool(stats["loso"]) and min(stats["loso"].values()) > 0,
        "positive_top1_share_at_most_0_5": stats["positive_top1_contribution"] is not None
        and stats["positive_top1_contribution"] <= TOP1_MAX,
    }


def compute_checks(fixed, saving, loss_stats, cheaper_fixed_losses):
    """Secondary branch: the cross-query cheapest-sufficient policy keeps the fixed action's quality at half its
    compute or less, and no cheaper fixed action already does."""
    return {
        "fixed_action_has_cost": STEPS[fixed] > 0,
        "saving_at_least_half": saving is not None and saving >= COMPUTE_SAVING_MIN,
        "mean_quality_loss_at_most_0_0003": loss_stats is not None and loss_stats["mean"] <= COMPUTE_LOSS_MEAN_MAX,
        "quality_loss_ci_high_at_most_0_0005": loss_stats is not None and loss_stats["ci95"][1] <= COMPUTE_LOSS_CI_MAX,
        "no_cheaper_fixed_action_within_0_0005": all(v > FIXED_RULE_TOLERANCE for v in cheaper_fixed_losses.values()),
    }


def gate0(quality, compute):
    if all(quality.values()):
        return "QUALITY_HEADROOM_SUPPORTED"
    if all(compute.values()):
        return "COMPUTE_HEADROOM_SUPPORTED"
    if all(v for k, v in compute.items() if k != "no_cheaper_fixed_action_within_0_0005"):
        return "NOT_ESTABLISHED_FIXED_RULE_SUFFICES"
    return "NOT_ESTABLISHED"
