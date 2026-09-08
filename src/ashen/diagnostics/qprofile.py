"""q-profile gathering (jorek2_postproc's qprofile command) and
rational-surface lookup.

Answers, for a given (n, m) toroidal/poloidal mode pair, where in the
domain (as psi_n) the safety factor satisfies q=m/n -- the resonant
surface a tearing/kink mode of that helicity actually grows on.
diagnostics.four_modes uses this to pin its amplitude time series to that
surface instead of an unlocalised domain-wide max.

Gathering: run_qprofile_step runs jorek2_postproc in place in run_dir,
same pattern as jorek2.run_zero_d and poincare._write_flux_surface --
nothing to stage, output persists under run_dir/postproc/ as a
step-keyed cache, not collected into a scratch dir and discarded.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ashen.jorek2 import Jorek2Error, Jorek2Run, MissingRestartError
from ashen.paths import RunPaths
from ashen.postproc import qprofile_script, read_postproc_profile

__all__ = [
    "run_qprofile_step", "read_qprofile", "find_rational_surfaces",
    "rational_surface_matches", "track_branches", "MAX_BRANCH_JUMP",
]

POSTPROC_TOOL = "jorek2_postproc"

#: How far (in psi_n) one rational surface may move between adjacent steps and
#: still be considered the same surface by :func:`track_branches`. A resonant
#: surface drifts as q evolves, but it drifts continuously; a jump larger than
#: this is a different surface, not the same one teleporting.
MAX_BRANCH_JUMP = 0.1


def run_qprofile_step(run: Jorek2Run, step: int, paths: RunPaths) -> Path:
    """q-profile for one step. Ports the shape of jorek2.run_zero_d,
    swapping in postproc.qprofile_script.

    Control script is a unique per-call temp file, not a fixed name, same
    reason as run_zero_d's: concurrent steps against the same run_dir
    must not race on it.
    """
    exe = run.exe_dir / POSTPROC_TOOL
    if not exe.is_file():
        raise FileNotFoundError(f"{POSTPROC_TOOL} not found at {exe}")
    restart_src = run.restart_path(step)
    if not restart_src.is_file():
        raise MissingRestartError(f"restart file not found: {restart_src}")

    paths.postproc_dir.mkdir(parents=True, exist_ok=True)
    fd, script_name = tempfile.mkstemp(
        prefix="postproc_qprofile_script_", suffix=".in", dir=run.run_dir
    )
    script_path = Path(script_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(qprofile_script(run.namelist.name, paths.step_str(step)))
        with open(script_path, encoding="utf-8") as stdin_file:
            result = subprocess.run(
                [str(exe)],
                stdin=stdin_file,
                cwd=run.run_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
    finally:
        script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise Jorek2Error(
            f"jorek2_postproc exited {result.returncode} for qprofile at step "
            f"{step} in {run.run_dir}: {result.stderr.decode(errors='replace')}"
        )

    out = paths.qprofile(step)
    if not out.is_file():
        raise Jorek2Error(f"qprofile not produced at {out}")
    return out


def read_qprofile(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """(psi_n, q) from one step's qprofile_s<step>.dat.

    A single-step `for step` loop always writes exactly one block, so
    this takes whichever one block postproc.read_postproc_profile found,
    not requiring the caller to know the step number the file was
    written with.

    Columns selected POSITIONALLY, not by the `headers` name list
    read_postproc_profile returns. exec_commands.f90::qprofile sets
    tmp_expr_list%n_expr=0 immediately before assigning
    expr(1)%name='Psi_n' / expr(2)%name='q' (exec_commands.f90:2368-2370)
    without ever incrementing it back up, so write_ascii_header's
    `do i=1,expr_list%n_expr` loop runs zero times and the header line
    carries no column names -- just "# " and nothing. Real bug in
    vendored JOREK (Columbia/jorek_RE, never edited here), but column
    ORDER (Psi_n then q, from res1d(k2,:) = (/get_psi_n(...), q(k)/) a
    few lines below) is unaffected, so position is what's reliable.
    """
    _headers, blocks = read_postproc_profile(path)
    data = next(iter(blocks.values()))
    return data[:, 0], data[:, 1]


def find_rational_surfaces(
    psi_n: np.ndarray, q: np.ndarray, q_target: float
) -> list[float]:
    """Every psi_n where q(psi_n) crosses q_target, linearly interpolated
    between adjacent samples.

    Ports the crossing search in exec_commands.f90::find_q_surface
    ((q(i)-qvalue)*(q(i+1)-qvalue) < 0). Returns every crossing, not just
    the first -- reversed-shear profiles cross a given q more than once,
    each a distinct physical rational surface.
    """
    crossings: list[float] = []
    for i in range(len(q) - 1):
        q0, q1 = float(q[i]), float(q[i + 1])
        if q0 == q_target:
            crossings.append(float(psi_n[i]))
            continue
        if (q0 - q_target) * (q1 - q_target) < 0.0:
            frac = (q_target - q0) / (q1 - q0)
            crossings.append(float(psi_n[i]) + frac * (float(psi_n[i + 1]) - float(psi_n[i])))
    return crossings


def rational_surface_matches(
    psi_n_q: np.ndarray,
    q: np.ndarray,
    modes: Sequence[tuple[int, int, str]],
    traced_psi_n: Sequence[float],
) -> dict[float, str]:
    """Snap each requested (n, m, color)'s q=m/n rational surface(s) onto
    the nearest value in traced_psi_n -- the discrete grid a Poincare scan
    actually traced, since a computed crossing essentially never lands
    exactly on one. All arguments share the same (normalised psi_n)
    units; the caller owns any conversion to/from the physical units
    poincare_cache.LineKey stores.

    A reversed-shear profile can cross a given q more than once; every
    crossing is matched and coloured the same, each a distinct physical
    resonance for that mode (same precedent as four_modes.
    rational_surface_series). n==0 entries are skipped (m/0 undefined),
    not errored, so a mixed resonant/non-resonant mode list needs no
    caller-side filtering. Returns {} if traced_psi_n is empty.
    """
    traced = list(traced_psi_n)
    matches: dict[float, str] = {}
    if not traced:
        return matches
    for n, m, color in modes:
        if n == 0:
            continue
        q_target = m / n
        for crossing in find_rational_surfaces(psi_n_q, q, q_target):
            nearest = min(traced, key=lambda p: abs(p - crossing))
            matches[nearest] = color
    return matches


def track_branches(
    crossings_by_step: Mapping[int, Sequence[float]],
    *,
    max_jump: float = MAX_BRANCH_JUMP,
) -> list[dict[int, float]]:
    """Follow one mode's rational surfaces across steps.

    Takes {step: [psi_n, ...]} -- the crossings :func:`find_rational_surfaces`
    found for a single q=m/n at each step -- and returns one dict per
    *surface*, mapping the steps where that surface exists to its position.

    Tracking is needed because a mode does not have a fixed number of
    surfaces. A monotonic q crosses q=m/n once, but a reversed-shear profile
    crosses it two or three times, and those extra surfaces appear and merge
    as q evolves through the run. Pairing crossings by their rank within each
    step would silently re-label every surface the moment an inner one
    vanishes, turning a band into nonsense; matching by position does not.

    Greedy nearest-neighbour: walking steps in order, each crossing joins the
    unclaimed branch whose last known position is nearest, provided it is
    within `max_jump`. Anything further away starts a new branch. A branch a
    step does not reach simply has no entry for it -- callers should not
    assume every branch spans every step.

    `max_jump` exists only to tell competing surfaces apart, so it is not
    applied where there is no competition: if one crossing and one
    still-live branch are left over after that pass, they are paired however
    far apart they are. A lone surface cannot be confused with anything, and
    it can legitimately move a long way between two restarts saved far apart
    in time -- a limit tuned to separate a reversed-shear pair would
    otherwise split that single surface into a string of stubs.
    """
    branches: list[dict[int, float]] = []
    previous_step: int | None = None

    for step in sorted(crossings_by_step):
        crossings = sorted(float(c) for c in crossings_by_step[step])
        unmatched: list[float] = []

        for x in crossings:
            best: int | None = None
            best_distance = float("inf")
            for i, branch in enumerate(branches):
                if step in branch:  # already claimed at this step
                    continue
                head = branch[max(branch)]
                distance = abs(head - x)
                if distance < best_distance:
                    best, best_distance = i, distance
            if best is not None and best_distance <= max_jump:
                branches[best][step] = x
            else:
                unmatched.append(x)

        # The unambiguous leftover: exactly one crossing with nowhere to go
        # and exactly one branch that was alive last step and went unclaimed.
        live_unclaimed = [
            i for i, branch in enumerate(branches)
            if step not in branch and previous_step is not None
            and previous_step in branch
        ]
        if len(unmatched) == 1 and len(live_unclaimed) == 1:
            branches[live_unclaimed[0]][step] = unmatched.pop()

        for x in unmatched:
            branches.append({step: x})
        previous_step = step

    return branches
