"""q-profile gathering (``jorek2_postproc``'s ``qprofile`` command) and
rational-surface lookup.

Exists to answer, for a given ``(n, m)`` toroidal/poloidal mode pair, *where*
in the domain (as ``psi_n``) the safety factor satisfies ``q = m/n`` -- the
resonant surface a tearing/kink mode of that helicity actually grows on.
:mod:`ashen.diagnostics.four_modes` uses this to pin its amplitude time
series to that surface instead of an unlocalised domain-wide max.

**Gathering.** ``run_qprofile_step`` runs ``jorek2_postproc`` **in place**
in ``run_dir``, the same pattern as :func:`ashen.jorek2.run_zero_d` and
:func:`ashen.diagnostics.poincare._write_flux_surface`: nothing needs
staging, and the output is meant to persist under ``run_dir/postproc/`` as a
cache keyed by step, not be collected into a scratch dir and discarded.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ashen.jorek2 import Jorek2Error, Jorek2Run, MissingRestartError
from ashen.paths import RunPaths
from ashen.postproc import qprofile_script, read_postproc_profile

__all__ = ["run_qprofile_step", "read_qprofile", "find_rational_surfaces"]

POSTPROC_TOOL = "jorek2_postproc"


def run_qprofile_step(run: Jorek2Run, step: int, paths: RunPaths) -> Path:
    """q-profile for one step. Ports the shape of
    :func:`ashen.jorek2.run_zero_d`, swapping in
    :func:`ashen.postproc.qprofile_script`.

    The control script is a unique per-call temp file (not a fixed name),
    for the same reason ``run_zero_d``'s is: concurrent steps against the
    same ``run_dir`` must not race on it.
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
    """``(psi_n, q)`` from one step's ``qprofile_s<step>.dat``.

    A single-step ``for step`` loop always writes exactly one block, so this
    takes whichever one block :func:`ashen.postproc.read_postproc_profile`
    found rather than requiring the caller to know the step number the file
    was written with.

    Columns are selected **positionally**, not by the ``headers`` name list
    ``read_postproc_profile`` returns. ``exec_commands.f90::qprofile`` sets
    ``tmp_expr_list%n_expr = 0`` immediately before assigning
    ``expr(1)%name = 'Psi_n'`` / ``expr(2)%name = 'q'``
    (``exec_commands.f90:2368-2370``) without ever incrementing it back up,
    so ``write_ascii_header``'s ``do i = 1, expr_list%n_expr`` loop runs zero
    times and the header line it writes carries no column names at all --
    only ``# `` followed by nothing. A real bug in vendored JOREK
    (``Columbia/jorek_RE``, never to be edited here), but the column *order*
    (``Psi_n`` then ``q``, from ``res1d(k2,:) = (/ get_psi_n(...), q(k) /)``
    a few lines below) is unaffected, so position is what's reliable.
    """
    _headers, blocks = read_postproc_profile(path)
    data = next(iter(blocks.values()))
    return data[:, 0], data[:, 1]


def find_rational_surfaces(
    psi_n: np.ndarray, q: np.ndarray, q_target: float
) -> list[float]:
    """Every ``psi_n`` where ``q(psi_n)`` crosses ``q_target``, linearly
    interpolated between adjacent samples.

    Ports the crossing search in ``exec_commands.f90::find_q_surface``
    (``(q(i)-qvalue)*(q(i+1)-qvalue) < 0``). Returns every crossing, not just
    the first -- reversed-shear profiles cross a given q more than once, and
    each is a distinct physical rational surface.
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
