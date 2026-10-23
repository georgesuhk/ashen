"""Suggesting JOREK's time step from the runs' recorded histories.

The limit that matters is the solver's: a step fails when GMRES does not
converge in ``gmres_max_iter`` iterations (see ``ashen.tstep_history``). Its
iteration count N rises with the time step dt, and steeply near the limit:
for preconditioned GMRES, N ~ ln(tol) / ln(rho) with a convergence factor rho
growing with dt times the coupling the preconditioner leaves out, so N runs
away as rho -> 1. Locally that is N ~ dt^q. The exponent is measured where
the same state was stepped with two time steps (a crashed job and its
restart at a smaller step, matched by start time): in the qa2.1_g1.8,
eta = 1e-5 run, dt = 1 against dt = 0.5 gave 200 against 44 iterations
(q ~ 2.2) where both refactorised every step.

A step that took N iterations at dt would take N_target at about
dt * (N_target / N)^(1/q). The suggestion is the smallest of:

- **the iteration limit**: over every step this run is about to repeat
  (all jobs' steps from the restart time on, failed ones included, plus the
  steps just before it -- the state it starts from), so a later job's crash
  counts; N_target = tstep_safety * gmres_max_iter. q is taken from the
  run's own same-time pairs, else from comparable runs', else assumed in
  Q_ASSUMED, and on the safe side: the largest q when the step would grow,
  the smallest when it must shrink. Never more than MAX_GROWTH times the
  largest step a job has run: beyond that the history says nothing.
- **similar runs**: for a new run, or a restart with no history ahead of
  it, each of the NEIGHBOURS most similar runs' worst step (its hardest
  phase), the smallest of them. Only runs whose iteration counts compare
  (same solver settings and grid, ``tstep_history.SOLVER_KEYS``).
- **a NaN crash** at or after the restart time: half that step. NaNs give no
  measure of how far over the limit the step was; halving is a heuristic.
- **accuracy**: tstep_accuracy / gamma, gamma the linear growth rate. A
  step the solver manages is not therefore accurate; with tstep_accuracy =
  0.1 Gears' (BDF2) error in a growth rate is under 1%.

Everything is in JOREK time units (t0 = sqrt(mu0 rho0)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ashen.namelist import NamelistError, effective_fields, normalise_key
from ashen.tstep_history import (
    SOLVER_KEYS, JobHistory, RunHistory, find_histories, harvest, load_history,
)

__all__ = ["Settings", "Pair", "Suggestion", "find_pairs", "exponent_range", "suggest",
           "format_suggestion"]

#: q (N ~ dt^q) assumed with no same-time pairs to measure it from.
Q_ASSUMED = (1.0, 3.0)
#: Measured q is kept within this range: a pair of small counts can give anything.
Q_BOUNDS = (0.5, 6.0)
#: Pairs need both counts at least this: one or two iterations measure nothing.
MIN_PAIR_ITERS = 5
#: Steps with fewer iterations than this are not extrapolated from on their
#: own: the count is at its floor. (MAX_GROWTH still bounds them.)
MIN_ITERS = 3
#: Never suggest more than this times the largest step any job has run.
MAX_GROWTH = 2.0
#: Steps before the restart time counted as "the state it starts from".
RECENT_STEPS = 50
#: Similar runs used for a run with no history of its own ahead.
NEIGHBOURS = 3


@dataclass(frozen=True)
class Settings:
    safety: float = 0.5        #: N_target = safety * gmres_max_iter (shotfile tstep_safety)
    accuracy: float = 0.1      #: dt <= accuracy / gamma (shotfile tstep_accuracy)


@dataclass(frozen=True)
class Pair:
    """One state stepped with two time steps."""

    t: float
    dt_a: float
    n_a: int
    dt_b: float
    n_b: int
    fresh: bool                #: both steps refactorised the preconditioner

    @property
    def q(self) -> float:
        return math.log(self.n_a / self.n_b) / math.log(self.dt_a / self.dt_b)


@dataclass
class Limit:
    """One bound on the time step and why."""

    kind: str                  #: "iterations", "neighbours", "nan", "accuracy", "growth"
    dt: float
    why: str


@dataclass
class Suggestion:
    run_dir: Path
    t_now: float | None        #: restart time, None for a new run
    dt: float | None           #: the suggestion; None if nothing to go on
    limits: list[Limit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    q: tuple[float, float] = Q_ASSUMED
    q_source: str = ""
    n_target: float | None = None
    max_iter: int | None = None

    @property
    def binding(self) -> Limit | None:
        return min(self.limits, key=lambda lim: lim.dt) if self.limits else None


# =============================================================================
# Same-time pairs and the exponent q
# =============================================================================


def _same_time(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a))


def find_pairs(jobs: list[JobHistory]) -> list[Pair]:
    """Steps of two jobs (same solver settings, different time steps) that
    started from the same time. A step that failed is left out: it only
    says N >= the cap."""
    pairs = []
    for i, a in enumerate(jobs):
        for b in jobs[i + 1:]:
            if a.solver_key() != b.solver_key():
                continue
            ta, tb = a.t_start, b.t_start
            fa, fb = a.failed_step(), b.failed_step()
            for ia in range(len(ta)):
                if ia == fa:
                    continue
                ib = int(np.searchsorted(tb, ta[ia] - 1e-6 * max(1.0, abs(ta[ia]))))
                if ib >= len(tb) or ib == fb or not _same_time(ta[ia], tb[ib]):
                    continue
                if _same_time(a.dt[ia], b.dt[ib]):
                    continue
                na, nb = int(a.iters[ia]), int(b.iters[ib])
                if na < MIN_PAIR_ITERS or nb < MIN_PAIR_ITERS:
                    continue
                pairs.append(Pair(float(ta[ia]), float(a.dt[ia]), na, float(b.dt[ib]), nb,
                                  bool(a.refactor[ia] and b.refactor[ib])))
    return pairs


def exponent_range(pairs: list[Pair]) -> tuple[tuple[float, float], str] | None:
    """(q_low, q_high) and where it came from, None without pairs. Pairs where
    both steps refactorised are used if there are any: a reused, staler
    preconditioner adds iterations to one side only."""
    if not pairs:
        return None
    fresh = [p for p in pairs if p.fresh]
    use = fresh or pairs
    qs = np.clip([p.q for p in use], *Q_BOUNDS)
    what = "both refactorised" if fresh else "preconditioner reused on one side"
    return (float(qs.min()), float(qs.max())), f"{len(use)} same-time pair(s), {what}"


# =============================================================================
# Limits
# =============================================================================


def _step_limit(dt: float, n: int, n_target: float, q: tuple[float, float],
                failed: bool) -> float:
    """The time step at which a step that took n iterations at dt would take
    n_target, on the safe side of q. A failed step's n is a lower bound
    (it is the cap), which the smallest q already treats safely."""
    q_low, q_high = q
    if failed or n >= n_target:
        return dt * (n_target / max(n, 1)) ** (1.0 / q_low)
    return dt * (n_target / max(n, MIN_ITERS)) ** (1.0 / q_high)


def _worst(jobs: list[JobHistory], select, n_target: float, q) -> tuple[float, str] | None:
    """The smallest step limit over the selected steps, and which step set it."""
    best = None
    for job in jobs:
        mask = select(job)
        failed = job.failed_step()
        for i in np.flatnonzero(mask):
            lim = _step_limit(float(job.dt[i]), int(job.iters[i]), n_target, q, i == failed)
            if best is None or lim < best[0]:
                what = (f"{job.source}: step {job.index[i]} (t = {job.t_start[i]:.6g}), "
                        f"dt = {job.dt[i]:g}, {job.iters[i]} iterations"
                        + (" -- did not converge" if i == failed else ""))
                best = (lim, what)
    return best


def _comparable(jobs: list[JobHistory], planned: dict) -> tuple[list[JobHistory], list[str]]:
    """The jobs whose iteration counts compare with the planned run's, and
    notes on those left out. A setting the planned namelist does not give is
    not compared."""
    keep, notes = [], []
    for job in jobs:
        differ = [f"{k} {job.params.get(k)!r} vs {planned[k]!r}" for k in SOLVER_KEYS
                  if planned.get(k) is not None and job.params.get(k) is not None
                  and not _equal(job.params.get(k), planned[k])]
        if differ:
            notes.append(f"{job.source} left out, solver settings differ: {', '.join(differ)}")
        else:
            keep.append(job)
    return keep, notes


def _equal(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9)
    return str(a).strip().lower() == str(b).strip().lower()


def planned_settings(run_dir: Path, *, restart: bool) -> dict:
    """The solver settings the next job will run with, from in_main_r (a
    restart) or in_main, for what they give."""
    path = run_dir / ("in_main_r" if restart else "in_main")
    try:
        fields = effective_fields(path)
    except (OSError, NamelistError):
        return {}
    names = {"gmres_tol": "gmres_tol", "gmres_4": "gmres_4", "gmres_m": "gmres_m",
             "iter_precon": "iter_precon", "time_evol_scheme": "time_evol_scheme",
             "n_radial": "n_radial", "n_pol": "n_pol", "gmres_max_iter": "gmres_max_iter"}
    planned = {key: fields[normalise_key(name)] for key, name in names.items()
               if normalise_key(name) in fields}
    if fields.get("use_strumpack") is True:
        planned["solver"] = "strumpack"
    elif fields.get("use_mumps") is True:
        planned["solver"] = "mumps"
    elif fields.get("use_pastix") is True:
        planned["solver"] = "pastix"
    return planned


def restart_time(run_dir: Path) -> float | None:
    """The time JOREK will restart from: jorek_restart.h5's t_now. None if
    there is no restart file (a new run) or it cannot be read."""
    path = run_dir / "jorek_restart.h5"
    if not path.is_file():
        return None
    try:
        import h5py  # lazy, as elsewhere: the package imports without it

        with h5py.File(path, "r") as f:
            return float(np.asarray(f["t_now"]).ravel()[0])
    except (OSError, KeyError, ImportError, ValueError):
        return None


# =============================================================================
# Similar runs
# =============================================================================


def _features(history: RunHistory) -> dict:
    shot = history.shot
    params = history.jobs[0].params if history.jobs else {}
    out = {}
    if shot.get("eta") or params.get("eta"):
        out["log_eta"] = math.log10(shot.get("eta") or params.get("eta"))
    n0 = shot.get("rho_const") or (params.get("central_density") or 0) * 1e20
    if n0:
        out["log_n0"] = math.log10(n0)
    qa = shot.get("current_qa") or shot.get("qa")
    if qa:
        out["qa"] = float(qa)
    li = shot.get("current_li")
    if li:
        out["li"] = float(li)
    if shot.get("re_adv_fact") is not None:
        out["re_adv_fact"] = float(shot["re_adv_fact"])
    return out


#: How far apart one unit of each feature is, in the distance between runs.
_SCALES = {"log_eta": 1.0, "log_n0": 1.0, "qa": 0.5, "li": 0.2, "re_adv_fact": 0.5}


def _distance(a: dict, b: dict) -> float:
    shared = [k for k in _SCALES if k in a and k in b]
    if not shared:
        return math.inf
    return math.sqrt(sum(((a[k] - b[k]) / _SCALES[k]) ** 2 for k in shared))


def _describe(features: dict) -> str:
    parts = []
    if "log_eta" in features:
        parts.append(f"eta {10 ** features['log_eta']:.3g}")
    if "log_n0" in features:
        parts.append(f"n0 {10 ** features['log_n0']:.3g}")
    if "qa" in features:
        parts.append(f"qa {features['qa']:g}")
    if "li" in features:
        parts.append(f"l_i {features['li']:g}")
    return ", ".join(parts)


# =============================================================================
# The suggestion
# =============================================================================


def suggest(run_dir: Path | str, campaign_root: Path | str | None = None, *,
            settings: Settings = Settings(), t_now: float | None = ...,
            span: float | None = None) -> Suggestion:
    """Suggest the time step for the next job in ``run_dir``.

    Harvests the run's logs first (writing its record). ``t_now``: the time
    it restarts from (default: jorek_restart.h5's; None for a new run).
    ``campaign_root``: where to look for other runs' records (default: none).
    ``span``: how long the job is to run, for the notes."""
    run_dir = Path(run_dir).resolve()
    if t_now is ...:
        t_now = restart_time(run_dir)
    restart = t_now is not None
    history = harvest(run_dir)
    planned = planned_settings(run_dir, restart=restart)
    jobs, notes = _comparable(history.jobs, planned)

    others = []
    if campaign_root is not None:
        for folder in find_histories(campaign_root):
            if folder.resolve() == run_dir:
                continue
            try:
                other = load_history(folder)
            except (OSError, ValueError, KeyError):
                continue
            if bool(other.shot.get("with_refluid")) != bool(history.shot.get("with_refluid")):
                continue
            comparable, _ = _comparable(other.jobs, planned)
            if comparable:
                others.append((other, comparable))

    max_iter = planned.get("gmres_max_iter") or next(
        (job.max_iter for job in reversed(jobs) if job.max_iter), None)
    out = Suggestion(run_dir, t_now, None, notes=notes,
                     max_iter=int(max_iter) if max_iter else None)
    if max_iter is None:
        out.notes.append("no gmres_max_iter known (in_main or a log): no iteration limit")
        return out
    n_target = settings.safety * float(max_iter)
    out.n_target = n_target

    # q: this run's pairs, else every comparable run's
    found = exponent_range(find_pairs(jobs))
    if found is None:
        pooled = [p for _, comparable in others for p in find_pairs(comparable)]
        found = exponent_range(pooled)
        if found is not None:
            found = (found[0], found[1].replace("pair(s)", "pair(s) in similar runs"))
    if found is None:
        out.q, out.q_source = Q_ASSUMED, "assumed: no same-time pairs to measure it from"
    else:
        out.q, out.q_source = found

    # this run, from the restart time on
    if restart and jobs:
        def ahead(job):
            return job.t_start >= t_now - 1e-6 * max(1.0, abs(t_now))

        def just_before(job):
            before = np.flatnonzero(job.t_start < t_now - 1e-6 * max(1.0, abs(t_now)))
            mask = np.zeros(len(job.t_start), bool)
            mask[before[-RECENT_STEPS:]] = True
            return mask

        def current(job):
            return ahead(job) | just_before(job)

        worst = _worst(jobs, current, n_target, out.q)
        if worst is not None:
            out.limits.append(Limit("iterations", worst[0], worst[1]))
        furthest = max(job.t_end for job in jobs)
        if furthest <= t_now + 1e-6 * max(1.0, abs(t_now)):
            out.notes.append(
                f"no job has run past t = {furthest:.6g}: nothing is known of the time ahead, "
                "and the iteration count may still rise" + _trend(jobs, t_now))
        else:
            out.notes.append(f"history ahead of the restart: to t = {furthest:.6g}")
        for job in jobs:
            if job.outcome == "nan" and job.t_end >= t_now:
                out.limits.append(Limit(
                    "nan", 0.5 * float(job.dt[-1]),
                    f"{job.source} ended with NaNs at t = {job.t_end:.6g}, dt = {job.dt[-1]:g} "
                    "(half that: a heuristic)"))
    elif jobs:
        out.notes.append("a new run: this folder's own jobs are earlier attempts from t = 0")
        worst = _worst(jobs, lambda job: np.ones(len(job.iters), bool), n_target, out.q)
        if worst is not None:
            out.limits.append(Limit("iterations", worst[0], worst[1]))

    # similar runs: the limit for a run with no history of its own, else context
    if others:
        mine = _features(history)
        ranked = sorted(others, key=lambda o: _distance(mine, _features(o[0])))
        near = [o for o in ranked if _distance(mine, _features(o[0])) < math.inf][:NEIGHBOURS]
        per_run = []
        for other, comparable in near:
            worst = _worst(comparable, lambda job: np.ones(len(job.iters), bool), n_target, out.q)
            if worst is not None:
                per_run.append((worst[0], other, worst[1]))
                out.notes.append(
                    f"similar run {other.run_dir.name} ({_describe(_features(other))}): "
                    f"its hardest step allows dt <= {worst[0]:.3g} -- {worst[1]}")
        if per_run and not jobs:
            lim, other, what = min(per_run, key=lambda x: x[0])
            out.limits.append(Limit("neighbours", lim,
                                    f"hardest step of similar run {other.run_dir.name}: {what}"))
    if not jobs and not any(lim.kind == "neighbours" for lim in out.limits):
        out.notes.append("no comparable history in this run or in similar runs")

    # the step any job has run, times MAX_GROWTH
    tried = [float(job.dt.max()) for job in jobs] or \
        [float(job.dt.max()) for _, comparable in others for job in comparable]
    if tried and out.limits:
        out.limits.append(Limit("growth", MAX_GROWTH * max(tried),
                                f"{MAX_GROWTH:g} x the largest step run ({max(tried):g})"))

    onset = history.growth.get("t_onset") if history.growth else None
    if onset is None or (t_now if restart else 0.0) < onset:
        rise = _nonlinear_rise([job for _, comparable in others for job in comparable] + jobs,
                               [o for o, _ in others] + [history])
        where = "has not reached" if onset is None else f"is before (t = {onset:.6g})"
        out.notes.append(f"this run {where} its nonlinear phase" + (
            f"; there the iteration count rose: {rise}" if rise else
            "; no record yet of how much the iteration count rises there"))

    growth = history.growth or _max_growth(o for o, _ in others)
    if growth and growth.get("gamma"):
        source = "this run" if history.growth else "similar runs"
        out.limits.append(Limit(
            "accuracy", settings.accuracy / growth["gamma"],
            f"gamma = {growth['gamma']:.3g} (n = {growth['n']}, linear phase, {source}), "
            f"tstep_accuracy = {settings.accuracy:g}"))
    elif out.limits:
        out.notes.append("no linear growth rate known: no accuracy limit")

    if any(lim.kind in ("iterations", "neighbours", "nan") for lim in out.limits):
        out.dt = _round_down(min(lim.dt for lim in out.limits))
    if span is not None and out.dt:
        out.notes.append(f"the job's {span:g} time units at dt = {out.dt:g}: "
                         f"{math.ceil(span / out.dt - 1e-9)} steps")
    return out


def _max_growth(histories) -> dict | None:
    best = None
    for h in histories:
        if h.growth and h.growth.get("gamma") and (best is None or h.growth["gamma"] > best["gamma"]):
            best = h.growth
    return best


def _nonlinear_rise(jobs: list[JobHistory], histories: list[RunHistory]) -> str:
    """The largest rise of the iteration count, at one time step, from a
    run's linear phase to its nonlinear one, as recorded: "2 -> 200 at
    dt = 1 (run, job)". Empty if no job spans an onset."""
    onsets = {}
    for h in histories:
        if h.growth and h.growth.get("t_onset") is not None:
            for job in h.jobs:
                onsets[(h.run_dir.name, job.sha256)] = h.growth["t_onset"]
    best = None
    for h in histories:
        for job in h.jobs:
            onset = onsets.get((h.run_dir.name, job.sha256))
            if onset is None or job not in jobs:
                continue
            before = job.iters[job.t_start < onset]
            after = job.iters[job.t_start >= onset]
            if len(before) < 10 or not len(after):
                continue
            n_before = max(int(np.median(before[-RECENT_STEPS:])), 1)
            n_after = int(after.max())
            if best is None or n_after / n_before > best[0]:
                best = (n_after / n_before,
                        f"{n_before} -> {n_after} iterations at dt = {job.dt[-1]:g} "
                        f"({h.run_dir.name}, {job.source})")
    return best[1] if best else ""


def _trend(jobs: list[JobHistory], t_now: float) -> str:
    """How the iteration count was moving over the last steps before t_now."""
    latest = max(jobs, key=lambda job: job.t_end)
    mask = latest.t_start < t_now + 1e-6 * max(1.0, abs(t_now))
    t, n = latest.t_start[mask][-RECENT_STEPS * 2:], latest.iters[mask][-RECENT_STEPS * 2:]
    if len(t) < 10 or t[-1] <= t[0]:
        return ""
    slope = np.polyfit(t, n, 1)[0] * 100 * float(latest.dt[mask][-1])
    return (f" (over its last {len(t)} steps: {n.min()}-{n.max()} iterations, "
            f"{slope:+.2g} per 100 steps)")


def _round_down(dt: float) -> float:
    """Two significant figures, rounded down: a suggestion never rounds up."""
    if dt <= 0:
        return dt
    scale = 10 ** (math.floor(math.log10(dt)) - 1)
    return math.floor(dt / scale + 1e-9) * scale


def format_suggestion(s: Suggestion) -> str:
    lines = []
    where = (f"restart from t = {s.t_now:.6g}" if s.t_now is not None else "a new run")
    lines.append(f"time step for {s.run_dir.name}: {where}")
    if s.max_iter:
        lines.append(f"  aim: at most {s.n_target:g} GMRES iterations per step "
                     f"(tstep_safety x gmres_max_iter {s.max_iter})")
    lines.append(f"  N ~ dt^q, q = {s.q[0]:.2g}..{s.q[1]:.2g}: {s.q_source}")
    binding = s.binding
    for lim in sorted(s.limits, key=lambda lim: lim.dt):
        mark = "*" if lim is binding else " "
        lines.append(f"  {mark} {lim.kind:<10s} dt <= {lim.dt:.3g}   {lim.why}")
    for note in s.notes:
        lines.append(f"  note: {note}")
    if s.dt is None:
        lines.append("  no suggestion: nothing in the history to go on")
    else:
        lines.append(f"  suggested: tstep = {s.dt:g}  (set by: {binding.kind})")
    return "\n".join(lines)
