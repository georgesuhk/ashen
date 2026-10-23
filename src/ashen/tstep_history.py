"""A run's time-step history: what time step each JOREK job took, and how
hard each step was for the solver.

JOREK (model600, Gears or Crank-Nicolson) is implicit, so a large time step
is not unstable in the CFL sense; a step *fails* when the preconditioned
GMRES solve does not converge in ``gmres_max_iter`` iterations
(solvers/mod_sparse.f90: ``step_success = iter_gmres < iter_max``), and the
run stops with ``>>>>> NO CONVERGENCE AFTER ... ITERATIONS. ABORTING``. The
preconditioner factorises each toroidal harmonic on its own, so GMRES has
to supply the coupling between harmonics, which grows with the time step
and with the size of the perturbation: the iteration count climbs as a run
goes nonlinear and as the step grows, and steeply near the limit. The
per-step count, ``Number of iterations: N`` in the log, is therefore the
measure of how close a step was to failing.

This module reads that record out of every log a run has kept
(``ashen.log_archive``: ``logs/*.gz``, plus the current ``log``), one
:class:`JobHistory` per job: each step's index, time step, start time,
GMRES iterations and whether the preconditioner was refactorised (``Elapsed
time updating preconditioner``; while it is reused, the count also reflects
how stale it is), with the solver and physics settings the job's own log
header gives. :func:`harvest` gathers them, with the linear growth rate
from ``macroscopic_vars.dat``, into ``tstep_history.json.gz`` in the run
folder: a compact index for ``ashen.tstep_suggest``, rebuilt from the logs
at any time. The logs stay the record.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ashen.diagnostics.live_data import read_live_data
from ashen.log_archive import (
    ARCHIVE_DIR, BANNER, archived_logs, banners_follow, open_log, plan_archive, read_index,
    scan_span,
)

__all__ = [
    "RECORD_FILE", "JobHistory", "RunHistory", "parse_job_log", "linear_growth",
    "harvest", "load_history", "find_histories", "SOLVER_KEYS",
]

RECORD_FILE = "tstep_history.json.gz"
#: Bumped whenever parsing changes, so saved records are read again from the logs.
_VERSION = 2

#: Header lines of the log (mod_log_params.f90) kept per job: log name -> key.
_HEADER_FIELDS = {
    "n_tor": "n_tor", "n_plane": "n_plane", "n_period": "n_period",
    "time_evol_scheme": "time_evol_scheme",
    "eta": "eta", "eta_num": "eta_num", "eta_T_dependent": "eta_T_dependent",
    "n_radial": "n_radial", "n_pol": "n_pol",
    "R_geo": "R_geo", "F0": "F0",
    "central_density": "central_density", "central_mass": "central_mass",
    "gmres": "gmres", "gmres_max_iter": "gmres_max_iter", "gmres tolerance": "gmres_tol",
    "iter_precon": "iter_precon", "gmres_m": "gmres_m", "gmres_4": "gmres_4",
    "max_steps_noUpdate": "max_steps_noUpdate",
    "use_mumps": "use_mumps", "use_pastix": "use_pastix", "use_strumpack": "use_strumpack",
    "use_newton": "use_newton",
    "re_adv_fact": "re_adv_fact",
    "sqrt(mu0*rho0)": "sqrt_mu0_rho0",
}

#: Settings two jobs must share for their iteration counts to be compared:
#: the convergence test, the restart length, the preconditioner and its
#: reuse rule, and the problem's size. gmres_max_iter is not among them --
#: it only moves the cap.
SOLVER_KEYS = ("gmres_tol", "gmres_4", "gmres_m", "iter_precon", "solver", "time_evol_scheme",
               "n_tor", "n_plane", "n_radial", "n_pol")

_HEADER_LINE = re.compile(r"^\s*([A-Za-z_][\w ()*/]*?)\s*=\s*(\S.*?)\s*$")
_GIT = re.compile(r"^\s*GIT revision\s*:\s*(\S.*?)\s*$")
_RESTART_TIME = re.compile(r"restart time :\s+(\d+)\s+(\S+)")
_ITERATIONS = re.compile(r"Number of iterations:\s+(\d+)")


def _value(text: str):
    text = text.strip().strip('"').strip("'")
    if text in ("T", ".true."):
        return True
    if text in ("F", ".false."):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return text


@dataclass
class JobHistory:
    """One JOREK job's steps, as its log records them."""

    source: str                       #: the log it was read from (archive name, or "log")
    sha256: str
    outcome: str | None               #: gmresfail, nan, sigterm, walltime or None
    params: dict = field(default_factory=dict)
    t0: float = 0.0                   #: time the job started from (its restart's)
    index: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    dt: np.ndarray = field(default_factory=lambda: np.zeros(0))
    iters: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   #: -1: not logged
    refactor: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def t_start(self) -> np.ndarray:
        """Each step's start time: the state its solve began from."""
        return self.t0 + np.concatenate(([0.0], np.cumsum(self.dt)[:-1]))

    @property
    def t_end(self) -> float:
        return float(self.t0 + self.dt.sum())

    @property
    def max_iter(self) -> int | None:
        value = self.params.get("gmres_max_iter")
        return int(value) if isinstance(value, (int, float)) else None

    def solver_key(self) -> tuple:
        return tuple(self.params.get(k) for k in SOLVER_KEYS)

    def failed_step(self) -> int | None:
        """Position of the step that did not converge, if the job stopped on one."""
        if self.outcome != "gmresfail" or not len(self.iters):
            return None
        return len(self.iters) - 1

    # -- compact (de)serialisation: per-step arrays as run-lengths -----------
    def to_json(self) -> dict:
        return {
            "source": self.source, "sha256": self.sha256, "outcome": self.outcome,
            "params": self.params, "t0": self.t0,
            "first_index": int(self.index[0]) if len(self.index) else None,
            "index_gaps": _gaps(self.index),
            "dt": _runs(self.dt.tolist()),
            "iters": self.iters.tolist(),
            "refactor": _runs([int(x) for x in self.refactor]),
        }

    @classmethod
    def from_json(cls, data: dict) -> JobHistory:
        iters = np.array(data["iters"], dtype=int)
        n = len(iters)
        index = _ungaps(data["first_index"], data.get("index_gaps", []), n)
        return cls(
            source=data["source"], sha256=data["sha256"], outcome=data["outcome"],
            params=data["params"], t0=float(data["t0"]),
            index=index, dt=np.array(_unruns(data["dt"]), dtype=float),
            iters=iters, refactor=np.array(_unruns(data["refactor"]), dtype=bool),
        )


def _runs(values: list) -> list:
    """Run-length encoding [[value, count], ...]."""
    out: list = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def _unruns(runs: list) -> list:
    return [v for v, count in runs for _ in range(count)]


def _gaps(index: np.ndarray) -> list:
    """Where consecutive step indices do not go up by one: [[position, index], ...]."""
    return [[int(i), int(index[i])] for i in range(1, len(index)) if index[i] != index[i - 1] + 1]


def _ungaps(first: int | None, gaps: list, n: int) -> np.ndarray:
    if first is None:
        return np.zeros(0, int)
    index = np.empty(n, dtype=int)
    jumps = dict((pos, value) for pos, value in gaps)
    current = first
    for i in range(n):
        current = jumps.get(i, current + (i > 0))
        index[i] = current
    return index


def parse_job_log(path: Path | str, *, source: str | None = None, sha256: str = "",
                  outcome: str | None = None) -> JobHistory:
    """Read one job's log (plain or gzipped) into a JobHistory.

    The steps are those that finished their solve: a step the job was killed
    during (a banner with no ``Number of iterations``) is left out, and so
    is every step of a direct-solver run (``gmres = F``), which logs no
    count. With the inexact Newton solver a step logs one count per Newton
    iteration; the largest is kept, as that is what the convergence test
    applies to.

    Where another job's output follows in the same file (two jobs launched
    in one folder at once, ``log_archive.banners_follow``), reading stops at
    the first banner that does not follow, and ``params["spliced_after"]``
    records the last step kept.
    The ending is then only what was read before that point: ``outcome``
    (from the whole file) is not used."""
    path = Path(path)
    params: dict = {}
    t0 = 0.0
    rows: list[list] = []          # [index, dt, iters, refactor]
    pending: list | None = None    # the step whose solve is under way
    in_header = True
    last_banner: tuple | None = None
    ending = None
    with open_log(path) as f:
        for line in f:
            if "time step :" in line:
                m = BANNER.match(line)
                if m:
                    in_header = False
                    banner = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    if last_banner is not None and not banners_follow(last_banner, banner):
                        params["spliced_after"] = last_banner[2]
                        break
                    last_banner = banner
                    if pending is not None and pending[2] >= 0:
                        rows.append(pending)
                    pending = [banner[2], float(m.group(4)), -1, False]
                    continue
            if in_header:
                if "restart time :" in line:
                    m = _RESTART_TIME.search(line)
                    if m:
                        t0 = float(m.group(2))
                    continue
                m = _GIT.match(line)
                if m and "git" not in params:
                    params["git"] = m.group(1)
                    continue
                m = _HEADER_LINE.match(line)
                if m and m.group(1) in _HEADER_FIELDS:
                    key = _HEADER_FIELDS[m.group(1)]
                    if key not in params:
                        params[key] = _value(m.group(2).split()[0] if key != "time_evol_scheme"
                                             else m.group(2))
                continue
            if pending is None:
                continue
            if "updating preconditioner" in line:
                pending[3] = True
            elif "Number of iterations:" in line:
                m = _ITERATIONS.search(line)
                if m:
                    pending[2] = max(pending[2], int(m.group(1)))
            elif ending is None and ">>>>>" in line:
                if "NO CONVERGENCE AFTER" in line:
                    ending = "gmresfail"
                elif "NaNs DETECTED" in line:
                    ending = "nan"
                elif "SIGTERM RECEIVED" in line:
                    ending = "sigterm"
    if pending is not None and pending[2] >= 0:
        rows.append(pending)
    if ending is None and "spliced_after" not in params:
        ending = outcome            # e.g. walltime, from SLURM's .err
    params["solver"] = ("strumpack" if params.get("use_strumpack") else
                        "mumps" if params.get("use_mumps") else
                        "pastix" if params.get("use_pastix") else None)
    table = np.array(rows, dtype=float).reshape(-1, 4)
    return JobHistory(
        source=source or path.name, sha256=sha256, outcome=ending, params=params, t0=t0,
        index=table[:, 0].astype(int), dt=table[:, 1], iters=table[:, 2].astype(int),
        refactor=table[:, 3].astype(bool),
    )


# =============================================================================
# The linear phase, from macroscopic_vars.dat
# =============================================================================

#: The linear phase of harmonic n: while its magnetic energy is between these
#: fractions of the n = 0 energy -- above the start-up noise, below saturation.
LINEAR_WINDOW = (1e-20, 1e-5)
#: The nonlinear phase starts when all n > 0 together pass this fraction.
NONLINEAR_ONSET = 1e-5


def linear_growth(path: Path | str) -> dict | None:
    """The fastest linear growth rate and when the run turned nonlinear.

    ``{"gamma": ..., "n": ..., "t_onset": ...}`` in JOREK units (1/t0, t0 =
    sqrt(mu0 rho0)): gamma is JOREK's own amplitude growth rate
    (``magnetic_growth_rates``, 1/2 d ln E / dt) for each harmonic, its median
    over that harmonic's linear phase (LINEAR_WINDOW), the largest over n > 0
    (a harmonic driven by others, n = 2 by n = 1, grows at their sum, and
    counts too). t_onset: first time the n > 0 energy passes NONLINEAR_ONSET of
    n = 0, None if it never has. None if the file lacks the data or no
    harmonic has a linear phase yet."""
    try:
        rates = _normalised(path, "magnetic_growth_rates")
        energies = _normalised(path, "magnetic_energies")
    except (OSError, ValueError):
        return None
    tg, gamma = rates
    te, energy = energies
    best = None
    for n in range(1, energy.shape[1]):
        ratio = energy[:, n] / np.where(energy[:, 0] > 0, energy[:, 0], np.nan)
        window = (ratio > LINEAR_WINDOW[0]) & (ratio < LINEAR_WINDOW[1])
        if window.sum() < 5 or n >= gamma.shape[1]:
            continue
        g = float(np.median(np.interp(te[window], tg, gamma[:, n])))
        if g > 0 and (best is None or g > best[0]):
            best = (g, n)
    if best is None:
        return None
    total = energy[:, 1:].sum(axis=1) / energy[:, 0]
    passed = np.flatnonzero(total > NONLINEAR_ONSET)
    return {"gamma": best[0], "n": best[1],
            "t_onset": float(te[passed[0]]) if len(passed) else None}


def _normalised(path, quantity) -> tuple[np.ndarray, np.ndarray]:
    """A live-data quantity in JOREK units: read_live_data converts to SI
    with the file's factors; undo them."""
    live = read_live_data(path, quantity)
    data = Path(path).read_bytes()

    def factor(key):
        m = re.search(rb"^@" + quantity.encode() + b"_" + key.encode() + rb":[ \t]*(\S+)", data, re.M)
        return float(m.group(1).replace(b"D", b"E")) if m else 1.0

    return live.x / factor("x2si"), live.values / factor("y2si")


# =============================================================================
# The run's record
# =============================================================================


@dataclass
class RunHistory:
    run_dir: Path
    jobs: list[JobHistory]
    growth: dict | None          #: linear_growth's result
    shot: dict                   #: what the shotfile says (eta, density, q, l_i, ...)
    hints: list[str]             #: comments on the tstep_n/nstep_n lines: not data
    version: int = _VERSION      #: of the parser that wrote it

    def to_json(self) -> dict:
        return {"version": _VERSION, "run": self.run_dir.name, "growth": self.growth,
                "shot": self.shot, "hints": self.hints,
                "jobs": [job.to_json() for job in self.jobs]}


#: Shotfile values kept with the record, to find runs like another.
_SHOT_FIELDS = ("eta", "rho_const", "qa", "g", "current_q0", "current_li", "current_qa",
                "re_adv_fact", "with_refluid", "exe")


def _shot_values(run_dir: Path) -> tuple[dict, list[str]]:
    """The shotfile's plain values (read without running it: a shotfile may
    compute values, and this must not fail on one that no longer loads), and
    the comments on its tstep_n / nstep_n lines."""
    path = run_dir / "shotfile.py"
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return {}, []
    names: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                names[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                if isinstance(node.value, ast.Name) and node.value.id in names:
                    names[node.targets[0].id] = names[node.value.id]
    shot = {k: names[k] for k in _SHOT_FIELDS if k in names}
    hints = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("tstep_n", "nstep_n")) and "#" in stripped:
            hints.append(stripped)
    return shot, hints


def harvest(run_dir: Path | str, *, write: bool = True) -> RunHistory:
    """Read every log the run has kept, and its live data, into a RunHistory;
    with ``write``, save it as RECORD_FILE (atomically).

    Jobs already in the saved record (same sha256) are not parsed again. The
    current ``log`` is read too: it is the newest job, and may not be
    archived yet. SLURM ``.out`` copies not yet archived are read as well, so
    a run whose history is still only there counts in full. A job in the
    saved record whose log is no longer on disk stays in it: the record only
    grows."""
    run_dir = Path(run_dir).resolve()
    try:
        saved = load_history(run_dir).jobs
    except (OSError, ValueError, KeyError):
        saved = []
    # a record from an older parser: every log still on disk is read again
    known = {job.sha256: job for job in saved} if _saved_version(run_dir) == _VERSION else {}

    jobs: list[JobHistory] = []
    seen: set[str] = set()
    index = read_index(run_dir / ARCHIVE_DIR)
    for action in plan_archive(run_dir):            # log and SLURM .out files
        if action.kind == "empty" or action.sha256 in seen:
            continue
        seen.add(action.sha256)
        source, outcome = action.source.name, action.span.outcome
        if action.kind == "duplicate":              # kept already: its archive's name and
            source = action.target.name             # ending (a walltime is known only from
            outcome = index.get(source, {}).get("outcome") or outcome   # the .out's .err)
        if action.sha256 in known:
            known[action.sha256].source = source
            jobs.append(known[action.sha256])
        else:
            jobs.append(parse_job_log(action.source, source=source,
                                      sha256=action.sha256, outcome=outcome))
    for path in archived_logs(run_dir):
        sha = index.get(path.name, {}).get("sha256") or f"file:{path.name}"
        if sha in seen:
            continue
        seen.add(sha)
        if sha in known:
            known[sha].source = path.name      # the copy it was first read from may be gone
            jobs.append(known[sha])
        else:
            outcome = index.get(path.name, {}).get("outcome") or scan_span(path).outcome
            jobs.append(parse_job_log(path, source=path.name, sha256=sha, outcome=outcome))
    jobs += [job for job in saved if job.sha256 not in seen]   # its log is gone
    jobs = [job for job in jobs if len(job.iters)]
    jobs.sort(key=lambda j: (j.t0, int(j.index[0]) if len(j.index) else 0))

    shot, hints = _shot_values(run_dir)
    history = RunHistory(run_dir, jobs, linear_growth(run_dir / "macroscopic_vars.dat"),
                         shot, hints)
    if write and (jobs or history.growth):      # nothing to record: no file
        tmp = run_dir / (RECORD_FILE + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(history.to_json(), f, separators=(",", ":"))
        os.replace(tmp, run_dir / RECORD_FILE)
    return history


def _saved_version(run_dir: Path) -> int | None:
    try:
        with gzip.open(run_dir / RECORD_FILE, "rt", encoding="utf-8") as f:
            return json.load(f).get("version")
    except (OSError, ValueError):
        return None


def load_history(run_dir: Path | str) -> RunHistory:
    """A run's saved record. Raises OSError if there is none."""
    run_dir = Path(run_dir)
    with gzip.open(run_dir / RECORD_FILE, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return RunHistory(run_dir, [JobHistory.from_json(j) for j in data["jobs"]],
                      data.get("growth"), data.get("shot", {}), data.get("hints", []),
                      version=data.get("version", 0))


def find_histories(root: Path | str, *, max_depth: int = 4) -> list[Path]:
    """Run folders under ``root`` (up to ``max_depth`` levels down) that hold a
    saved record. Does not descend into a run folder's own subfolders."""
    root = Path(root)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if RECORD_FILE in filenames:
            found.append(Path(dirpath))
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    return sorted(found)
