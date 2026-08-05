"""Field-line Poincare tracing via jorek2_poincare.

Ports ``castor3d/util/diagnostics/poinc_diag.py``, but inverts how the work is
parallelised and cached.

**Parallelism.** The legacy code (and Phase 4's faithful port of it) launched
one ``jorek2_poincare`` process *per field line* -- ``n_lines = 1`` in every
``stpts``, so 96 processes per step with a 12-psi_n, 8-sample scan. Each one
independently copied the restart ``.h5`` and redid the tool's O(N^2)
element-neighbour scan (``jorek2_poincare.f90:60-70``) before tracing a single
line, while the tool's own OpenMP loop over field lines (``.f90:204-212``) sat
unused. Here one invocation per step traces *all* that step's pending lines,
with ``OMP_NUM_THREADS`` set explicitly, and :func:`run_poincare_scan` fans out
across steps -- which the legacy code did strictly serially.

``jorek2_poincare`` has no MPI at all; see :class:`ashen.config.Diagnostics`.

**Demultiplexing.** All lines share one ``poinc_R-Z.dat`` / one
``poinc_rho-theta.dat``, separated by double blank lines (``.f90:457-460``).
The writes happen inside ``!$omp critical`` in **thread-completion order, not
line order** (``.f90:450-455``), so a block's position does *not* identify its
line. The same critical section prints ``=> Line{i:6d}:{ip:6d} points``
(``.f90:449``), and :func:`_demux_blocks` uses that to assign blocks, refusing
to guess if the two disagree. Assigning traces to the wrong starting positions
would be an near-invisible corruption, so every mismatch raises.

**Incremental caching.** Work is planned per line against the existing cache
(:mod:`ashen.diagnostics.poincare_cache`), so widening ``psi_n_in`` traces only
the new positions and raising ``n_turns`` traces only the shortfall.

**Resuming is valid.** ``jorek2_poincare`` advances ``n_phi = 1500`` substeps
totalling exactly one toroidal period (``.f90:162-163``) and only then records
a puncture (``.f90:441-455``), so every puncture lies on the *same* toroidal
plane. Restarting a line at its last puncture with its original ``phi_start``
continues the same trajectory. Caveat, stated rather than hidden: for
stochastic lines the resumed trajectory diverges from an uninterrupted trace of
the same total length (exponential sensitivity, plus the element is re-located
from a written-out ``R, Z``). It samples the same field and the same invariant
set -- Poincare plots, island widths and diffusion statistics are unaffected --
but it is not bit-reproducible against a single long trace. Records carry
``n_segments`` so a stitched trace is always identifiable.

**Not ported in this pass**: the plotting functions (``plot_poincare``,
``plot_field_line_diffusion``, ``plot_connection_length``, ``get_island_width``)
in ``castor3d/util/data_jorek.py``, which read the old ``.npz`` format and
cannot read this cache -- see ``ashen/KNOWN_ISSUES.md`` #4 and #5.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from ashen.castor_io import load_two_col_data
from ashen.diagnostics import poincare_cache as pc
from ashen.jorek2 import Jorek2Error, Jorek2Run, run_tool
from ashen.paths import RunPaths
from ashen.postproc import flux_surface_script

__all__ = [
    "StepReport",
    "resolve_start_points",
    "trace_lines",
    "run_poincare_step",
    "run_poincare_scan",
]

#: ``jorek2_poincare.f90:449`` -- ``write(*,'(1x,a,i6,a,i6,a)') '=> Line',i_lines,':',ip,' points'``.
#: The i6 fields can run together with the separators when they overflow, hence
#: the tolerant whitespace.
_LINE_MSG = re.compile(r"=>\s*Line\s*(\d+)\s*:\s*(\d+)\s+points")

#: The JOREK executables this module drives, by the names they are symlinked
#: into a prepared run folder under. Module-level so tests can point them at
#: stubs -- on Windows an extensionless file cannot be executed, so a stub has
#: to be called ``jorek2_poincare.cmd``.
POINCARE_TOOL = "jorek2_poincare"
POSTPROC_TOOL = "jorek2_postproc"


@dataclass(frozen=True)
class StepReport:
    """What one step's scan actually had to do -- the visible payoff of the
    incremental cache."""

    step: int
    cache: Path
    cached: int
    traced: int
    extended: int

    def __str__(self) -> str:
        return (
            f"step {self.step}: {self.cached} cached, "
            f"{self.traced} new, {self.extended} extended"
        )


# --- starting positions --------------------------------------------------------


def _sampled_rz(fs_file: Path, n_sample_freq: int) -> list[tuple[float, float]]:
    """Ports ``poinc_diag.py:51`` ``get_sampled_RZ_at_psi_N``."""
    if n_sample_freq <= 0:
        raise ValueError("n_sample_freq must be greater than 0")
    data = load_two_col_data(fs_file)
    if len(data) == 0:
        return []
    idx = np.linspace(0, len(data) - 1, num=n_sample_freq, dtype=int)
    return [(float(data[i, 0]), float(data[i, 1])) for i in idx]


def _write_flux_surface(run: Jorek2Run, step: int, psi_n: float, namelist_name: str) -> None:
    """Generate one flux surface with ``jorek2_postproc``, in place under
    ``run_dir/postproc/``.

    Ports part of ``poinc_diag.py:143-154``, but for a single ``psi_n``:
    surfaces are now only generated for the values that actually have
    uncached lines. Runs in place (like :func:`ashen.jorek2.run_zero_d`)
    because the output is meant to land in the run folder.

    **The script file must be unique per call, not just per ``psi_n``.**
    :func:`ashen.diagnostics.poincare.run_poincare_scan` runs multiple steps
    concurrently, each in its own OS process, all against the same
    ``run_dir``. A name keyed only on ``psi_n`` collides across steps: two
    processes tracing the same ``psi_n`` at different steps can overwrite or
    delete each other's control script before ``jorek2_postproc`` reads it,
    so the tool generates the wrong step's surface (or none), and the step
    that actually asked for it then fails with a genuine
    ``FileNotFoundError`` on its own flux-surface path. The script is only
    ever consumed via stdin, never looked up by name, so a real per-call
    temp file (rather than a step-qualified but still-guessable name) removes
    the race outright instead of narrowing it.
    """
    exe = run.exe_dir / POSTPROC_TOOL
    if not exe.is_file():
        raise FileNotFoundError(f"{POSTPROC_TOOL} not found at {exe}")

    fd, script_name = tempfile.mkstemp(
        prefix="postproc_fs_script_", suffix=".in", dir=run.run_dir
    )
    script_path = Path(script_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(flux_surface_script(namelist_name, step, psi_n))
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
            f"jorek2_postproc exited {result.returncode} generating the flux "
            f"surface at psi_n={psi_n} for step {step} in {run.run_dir}: "
            f"{result.stderr.decode(errors='replace')}"
        )


def resolve_start_points(
    run: Jorek2Run,
    paths: RunPaths,
    step: int,
    psi_n_list,
    *,
    ang_sample_freq: int,
    phi_start: float = 0.0,
    known: set[float] | None = None,
) -> list[pc.LineKey]:
    """The starting points a scan asks for, as cache keys.

    ``known`` lists ``psi_n`` values whose keys are already fully derivable
    from the cache, so their flux surface need not be regenerated. Anything
    else gets its surface built (and then removed again, as the legacy code
    did -- it is a large intermediate, cheap to rebuild, and the cache now
    records the sampled positions that were the only reason to keep it).
    """
    keys: list[pc.LineKey] = []
    for psi_n in psi_n_list:
        psi_n = float(psi_n)
        if known and psi_n in known:
            continue
        fs_file = paths.flux_surface(psi_n, step)
        made_here = not fs_file.is_file()
        if made_here:
            _write_flux_surface(run, step, psi_n, run.namelist.name)
        try:
            samples = _sampled_rz(fs_file, ang_sample_freq)
        finally:
            if made_here:
                fs_file.unlink(missing_ok=True)
        keys.extend(
            pc.LineKey(psi_n=psi_n, R=R, Z=Z, phi=float(phi_start)).quantised()
            for R, Z in samples
        )
    return keys


# --- tracing --------------------------------------------------------------------


def _stpts(work: list[pc.LineWork]) -> str:
    """The ``stpts`` starting-point file for a batch of lines.

    Ports ``poinc_diag.py:19`` ``write_fieldline_block``, with two changes.
    ``n_lines`` is the whole batch rather than 1, and ``n_turns`` is per line
    (column 5, ``jorek2_poincare.f90:108``) -- both already supported by the
    tool, neither previously used. Values are written at full precision rather
    than the legacy ``{:10.6f}``: the file is read list-directed
    (``read(21,*)``, ``.f90:113``) so the format is free, and quantising a
    resume point to ~1um for no reason would be careless.

    Rows must be complete and in ascending ``nr`` order; a gap would be
    silently *interpolated* by the tool (``.f90:132-137``).
    """
    if not work:
        raise ValueError("no field lines to trace")
    lines = ["# n_lines", f"   {len(work)}", "# nr  R_start  Z_start  phi_start  n_turns"]
    for i, item in enumerate(work, start=1):
        R, Z = item.start
        lines.append(
            f"{i:6d}  {R:.12g}  {Z:.12g}  {item.key.phi:.12g}  {int(item.n_turns):6d}"
        )
    return "\n".join(lines) + "\n"


def _split_blocks(path: Path) -> list[np.ndarray]:
    """Split one of jorek2_poincare's two output files into per-line blocks.

    Blocks are separated by blank lines (``.f90:457-460``); ``#`` comment
    lines head the file (``.f90:190-196``).
    """
    blocks: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(np.asarray(current, dtype=float))
                current = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        current.append((float(parts[0]), float(parts[1])))
    if current:
        blocks.append(np.asarray(current, dtype=float))
    return blocks


def _demux_blocks(
    stdout: str, blocks: list[np.ndarray], n_lines: int, what: str
) -> dict[int, np.ndarray]:
    """Map output blocks to 1-based line numbers using the tool's own log.

    Blocks are emitted in OpenMP thread-completion order, so position is
    meaningless; the ``=> Line N: M points`` messages are printed from the
    *same* ``!$omp critical`` section as the writes, so their order matches.
    Everything is cross-checked and any disagreement raises rather than
    risking a silent mis-assignment of traces to starting positions.
    """
    reports = [(int(a), int(b)) for a, b in _LINE_MSG.findall(stdout)]
    if not reports:
        raise Jorek2Error(
            f"jorek2_poincare produced no '=> Line N: M points' messages, so "
            f"its {what} output blocks cannot be matched to field lines. "
            "Was stdout captured?"
        )
    if len(reports) != len(blocks):
        raise Jorek2Error(
            f"jorek2_poincare reported {len(reports)} traced lines but wrote "
            f"{len(blocks)} {what} blocks"
        )

    out: dict[int, np.ndarray] = {}
    for (line_no, n_points), block in zip(reports, blocks):
        if not 1 <= line_no <= n_lines:
            raise Jorek2Error(
                f"jorek2_poincare reported line {line_no}, outside 1..{n_lines}"
            )
        if line_no in out:
            raise Jorek2Error(f"jorek2_poincare reported line {line_no} twice")
        if len(block) != n_points:
            raise Jorek2Error(
                f"jorek2_poincare reported {n_points} points for line "
                f"{line_no} but its {what} block has {len(block)}"
            )
        out[line_no] = block
    return out


def trace_lines(
    run: Jorek2Run,
    step: int,
    work: list[pc.LineWork],
    *,
    dest_dir: Path,
    omp_threads: int,
) -> dict[pc.LineKey, dict[str, np.ndarray]]:
    """Trace a whole batch of field lines in **one** jorek2_poincare call.

    Returns ``{key: {"R", "Z", "rho", "theta"}}``. A line the tool did not
    report at all produced no punctures (it left the mesh immediately) and
    comes back with empty arrays.
    """
    result = run_tool(
        run,
        POINCARE_TOOL,
        step=step,
        dest_dir=dest_dir,
        outputs=["poinc_rho-theta.dat", "poinc_R-Z.dat"],
        stdin_is_namelist=True,
        extra_files={"stpts": _stpts(work)},
        env={"OMP_NUM_THREADS": str(max(int(omp_threads), 1))},
        capture_stdout=True,
    )

    rz = _demux_blocks(
        result.stdout, _split_blocks(result["poinc_R-Z.dat"]), len(work), "R-Z"
    )
    rt = _demux_blocks(
        result.stdout,
        _split_blocks(result["poinc_rho-theta.dat"]),
        len(work),
        "rho-theta",
    )
    if rz.keys() != rt.keys():
        raise Jorek2Error(
            "jorek2_poincare's R-Z and rho-theta outputs cover different "
            f"field lines for step {step} in {run.run_dir}"
        )

    empty = np.empty(0, dtype=float)
    out: dict[pc.LineKey, dict[str, np.ndarray]] = {}
    for i, item in enumerate(work, start=1):
        a, b = rz.get(i), rt.get(i)
        out[item.key] = {
            "R": a[:, 0] if a is not None else empty,
            "Z": a[:, 1] if a is not None else empty,
            "rho": b[:, 0] if b is not None else empty,
            "theta": b[:, 1] if b is not None else empty,
        }
    return out


# --- one step ---------------------------------------------------------------------


def run_poincare_step(
    run: Jorek2Run,
    paths: RunPaths,
    step: int,
    psi_n_list,
    *,
    ang_sample_freq: int,
    n_turns: int,
    phi_start: float = 0.0,
    omp_threads: int = 1,
    force: bool = False,
) -> StepReport:
    """One restart step's Poincare scan, doing only the work not already cached.

    Ports ``poinc_diag.py:135`` ``run_single_t``, but the four dense ``.npz``
    files it wrote are replaced by one per-line HDF5 cache (see
    :mod:`ashen.diagnostics.poincare_cache`), and only the missing lines are
    traced. ``force=True`` discards the existing cache and retraces everything.
    """
    cache_path = paths.poincare_cache(step)
    if force and cache_path.exists():
        cache_path.unlink()

    cached = pc.read_cache(cache_path)
    # psi_n values whose full sample set is already cached need no flux
    # surface -- the cache records the sampled positions, which was the only
    # reason to keep the surface around.
    by_psi: dict[float, int] = {}
    for key in cached:
        by_psi[key.psi_n] = by_psi.get(key.psi_n, 0) + 1
    # Exactly, not at least: a psi_n cached at ang_sample_freq=16 and now
    # requested at 8 has a *different* sample set (np.linspace re-spaces every
    # index), so its surface must be rebuilt to find which 8 are wanted.
    known = {p for p, n in by_psi.items() if n == ang_sample_freq}

    keys = resolve_start_points(
        run, paths, step, psi_n_list,
        ang_sample_freq=ang_sample_freq, phi_start=phi_start, known=known,
    )
    # Fill in the keys for psi_n values skipped above.
    for key in cached:
        if key.psi_n in known:
            keys.append(key)

    new, extensions = pc.plan_work(cached, keys, n_turns)
    report = StepReport(
        step=step,
        cache=cache_path,
        cached=len(set(keys)) - len(new) - len(extensions),
        traced=len(new),
        extended=len(extensions),
    )
    if not new and not extensions:
        return report

    scratch = paths.poinc_dir / f"_scratch_s{paths.step_str(step)}"
    batch = new + extensions
    traced = trace_lines(
        run, step, batch, dest_dir=scratch, omp_threads=omp_threads
    )
    for leftover in scratch.glob("poinc_*.dat"):
        leftover.unlink()
    if scratch.is_dir() and not any(scratch.iterdir()):
        scratch.rmdir()

    with pc.open_cache(cache_path, step=step, pad_width=run.pad_width) as handle:
        for item in new:
            arrays = traced[item.key]
            pc.append_line(
                handle, item.key, arrays,
                n_turns=item.n_turns,
                terminated=arrays["R"].size < item.n_turns,
                replace=item.key in cached,
            )
        for item in extensions:
            arrays = traced[item.key]
            pc.extend_line(
                handle, item.key, arrays,
                added_turns=item.n_turns,
                terminated=arrays["R"].size < item.n_turns,
            )
    return report


# --- a whole scan ------------------------------------------------------------------


def run_poincare_scan(
    run: Jorek2Run,
    paths: RunPaths,
    steps,
    psi_n_list,
    *,
    ang_sample_freq: int,
    n_turns: int,
    phi_start: float = 0.0,
    n_workers: int = 1,
    omp_threads: int = 1,
    force: bool = False,
) -> list[StepReport]:
    """Every step in a case, with steps traced concurrently.

    The legacy driver looped over steps strictly serially
    (``poinc_diag.py:234``) while fanning out over field lines; this does the
    opposite, because each step is one process holding one restart file and
    each process threads internally over its own lines.
    """
    steps = list(steps)
    one = partial(
        run_poincare_step,
        run,
        paths,
        psi_n_list=psi_n_list,
        ang_sample_freq=ang_sample_freq,
        n_turns=n_turns,
        phi_start=phi_start,
        omp_threads=omp_threads,
        force=force,
    )
    if n_workers <= 1 or len(steps) <= 1:
        return [one(step) for step in steps]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(one, steps))
