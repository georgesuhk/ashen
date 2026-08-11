"""Toroidal Fourier decomposition via `jorek2_four`.

No legacy Python wrapper exists anywhere in this tree (checked castor3d/util
and Columbia/jorek_RE/util) -- new ground, not a port.

I/O contract (jorek_RE/diagnostics/jorek2_four.f90): reads the run's
namelist from stdin (initialise_parameters(0,"__NO_FILENAME__") falls
through to read(5,in1), model600/initialise_parameters.f90:257-267, same
convention as jorek2_poincare) and the restart under fixed name
jorek_restart via import_restart, so jorek2.run_tool's default
restart_name="jorek_restart.h5" fits unchanged. Optional four_params.nml
in the working dir supplies nstpts/nTht/nmaxsteps/deltaphi/nsmallsteps/
rad_range (jorek2_four.f90:44-59); this module's defaults reproduce the
tool's own fallbacks, so an unconfigured case == a bare jorek2_four run.

output_glob, not a fixed outputs list: output files are one per (variable,
toroidal mode n) -- {variable_name}_modes_n{n:03d} -- written directly into
the working dir (jorek2_four.f90:99-119), not under postproc/.
variable_names/n_var are compile-time and model-dependent
(mod_parameters.f90 -- RE-fluid runs carry an extra n_RE var others don't),
so filenames can't be predicted ahead of a run; run_tool's output_glob
collects whatever matches instead.

Parsing: each file is nTht/2+1 blank-line-separated blocks (one per
poloidal mode m, ascending from 0), each a (psi_n, abs, real, imag, phase)
table over nstpts radial points. Every block's own "# l: m=.., n=.." header
(jorek2_four.f90:107) is cross-checked against block position and the
filename's n -- a format change upstream raises, doesn't silently mislabel.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from ashen.diagnostics import four_cache as fc
from ashen.jorek2 import Jorek2Error, Jorek2Run, MissingRestartError, run_tool
from ashen.paths import RunPaths

__all__ = ["FourStepReport", "four_params_nml", "run_four_step", "run_four_scan"]

#: The jorek2_four executable, symlinked into a prepared run folder. Module-level
#: so tests can point it at a stub -- on Windows an extensionless file cannot
#: be executed, so a stub has to be called ``jorek2_four.cmd``.
FOUR_TOOL = "jorek2_four"

#: jorek2_four.f90:99-119 -- one file per (variable, toroidal mode n).
_FILE_RE = re.compile(r"^(?P<var>.+)_modes_n(?P<n>\d{3})$")

#: jorek2_four.f90:107 -- `write(42,'("# ",I3,":   m=",I3,", n=",I3)') l, i-1, ...`
_HEADER_RE = re.compile(r"#\s*(-?\d+)\s*:\s*m\s*=\s*(-?\d+)\s*,\s*n\s*=\s*(-?\d+)")


@dataclass(frozen=True)
class FourStepReport:
    """What one step's decomposition produced."""

    step: int
    cache: Path
    cached: bool
    n_records: int

    def __str__(self) -> str:
        state = "cached" if self.cached else "new"
        return f"step {self.step}: {self.n_records} (variable, n, m) records [{state}]"


def four_params_nml(
    *,
    nstpts: int = 30,
    ntht: int = 32,
    nmaxsteps: int = 2500,
    deltaphi: float = 0.3,
    nsmallsteps: int = 3,
    rad_range: tuple[float, float] = (0.001, 0.999),
) -> str:
    """The four_params.nml control file jorek2_four reads for its
    field-line-tracing (magnetic-coordinate) parameters.

    Ports jorek_RE/util/examples/four_params.nml's layout. jorek2_four
    validates nTht itself (multiple of 4, >=32; jorek2_four.f90:80-84) and
    exits non-zero if not -- run_tool already turns that into Jorek2Error.
    """
    lines = [
        "&four_params",
        f"  nstpts = {int(nstpts)}",
        f"  nTht = {int(ntht)}",
        f"  nmaxsteps = {int(nmaxsteps)}",
        f"  deltaphi = {float(deltaphi)}",
        f"  nsmallsteps = {int(nsmallsteps)}",
        f"  rad_range = {float(rad_range[0])}, {float(rad_range[1])}",
        "/",
        "",
    ]
    return "\n".join(lines)


def _parse_four_file(path: Path, *, expected_n: int) -> list[tuple[int, np.ndarray]]:
    """One {variable}_modes_n{n} file -> [(m, data)] in block order.
    data columns: (psi_n, abs, real, imag, phase)."""
    blocks: list[tuple[int, np.ndarray]] = []
    header: tuple[int, int, int] | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal header, rows
        if header is None:
            if rows:
                raise Jorek2Error(f"{path}: data rows with no preceding header block")
            return
        l, m, n = header
        if l != len(blocks):
            raise Jorek2Error(
                f"{path}: block {len(blocks)} carries header index {l}, out of order"
            )
        if n != expected_n:
            raise Jorek2Error(
                f"{path}: header says n={n}, filename says n={expected_n}"
            )
        if not rows:
            raise Jorek2Error(f"{path}: block for m={m} has no data rows")
        blocks.append((m, np.asarray(rows, dtype=float)))
        header, rows = None, []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            match = _HEADER_RE.match(line)
            if not match:
                raise Jorek2Error(f"{path}: unrecognised header line {raw!r}")
            header = tuple(int(g) for g in match.groups())
            continue
        rows.append([float(x) for x in line.split()])
    flush()
    return blocks


def run_four_step(
    run: Jorek2Run,
    paths: RunPaths,
    step: int,
    *,
    nstpts: int = 30,
    ntht: int = 32,
    nmaxsteps: int = 2500,
    deltaphi: float = 0.3,
    nsmallsteps: int = 3,
    rad_range: tuple[float, float] = (0.001, 0.999),
    omp_threads: int = 1,
    force: bool = False,
) -> FourStepReport:
    """One restart step's Fourier decomposition.

    Cached whole: unlike Poincare tracing, not incremental -- a step with
    an existing cache does nothing unless force=True.
    """
    cache_path = paths.four_cache(step)
    if not force and cache_path.is_file():
        return FourStepReport(
            step=step, cache=cache_path, cached=True,
            n_records=fc.count_records(cache_path),
        )

    scratch = paths.four_dir / f"_scratch_s{paths.step_str(step)}"
    result = run_tool(
        run,
        FOUR_TOOL,
        step=step,
        dest_dir=scratch,
        output_glob="*_modes_n[0-9][0-9][0-9]",
        stdin_is_namelist=True,
        extra_files={
            "four_params.nml": four_params_nml(
                nstpts=nstpts, ntht=ntht, nmaxsteps=nmaxsteps,
                deltaphi=deltaphi, nsmallsteps=nsmallsteps, rad_range=rad_range,
            )
        },
        env={"OMP_NUM_THREADS": str(max(int(omp_threads), 1))},
    )

    records: list[fc.FourRecord] = []
    for name, out_path in result.outputs.items():
        match = _FILE_RE.match(name)
        if not match:
            raise Jorek2Error(f"{FOUR_TOOL}: unexpected output {name!r} for step {step}")
        variable = match["var"]
        n = int(match["n"])
        for m, data in _parse_four_file(out_path, expected_n=n):
            if data.shape[1] != 5:
                raise Jorek2Error(
                    f"{out_path}: expected 5 columns per row, got {data.shape[1]}"
                )
            records.append(
                fc.FourRecord(
                    variable=variable, n=n, m=m,
                    psi_n=data[:, 0], real=data[:, 2], imag=data[:, 3],
                )
            )
        out_path.unlink()
    if scratch.is_dir() and not any(scratch.iterdir()):
        scratch.rmdir()

    fc.write_cache(cache_path, step=step, pad_width=run.pad_width, records=records)
    return FourStepReport(step=step, cache=cache_path, cached=False, n_records=len(records))


def run_four_scan(
    run: Jorek2Run,
    paths: RunPaths,
    steps,
    *,
    nstpts: int = 30,
    ntht: int = 32,
    nmaxsteps: int = 2500,
    deltaphi: float = 0.3,
    nsmallsteps: int = 3,
    rad_range: tuple[float, float] = (0.001, 0.999),
    n_workers: int = 1,
    omp_threads: int = 1,
    force: bool = False,
    on_progress: Callable[[int, int, FourStepReport], None] | None = None,
) -> list[FourStepReport]:
    """Every step in a case, traced concurrently -- one process per step,
    same fan-out as poincare.run_poincare_scan.

    on_progress(done, total, report), if given, fires in process-pool
    completion order (not necessarily `steps`' order). The returned list
    is always in `steps` order.
    """
    steps = list(steps)
    total = len(steps)
    one = partial(
        run_four_step,
        run,
        paths,
        nstpts=nstpts, ntht=ntht, nmaxsteps=nmaxsteps, deltaphi=deltaphi,
        nsmallsteps=nsmallsteps, rad_range=rad_range,
        omp_threads=omp_threads, force=force,
    )
    if n_workers <= 1 or len(steps) <= 1:
        reports = []
        for i, step in enumerate(steps, start=1):
            try:
                report = one(step)
            except MissingRestartError as exc:
                warnings.warn(f"skipping four step {step}: {exc}", stacklevel=2)
                continue
            reports.append(report)
            if on_progress is not None:
                on_progress(i, total, report)
        return reports

    reports: list[FourStepReport | None] = [None] * total
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(one, step): idx for idx, step in enumerate(steps)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            step = steps[idx]
            try:
                reports[idx] = future.result()
            except MissingRestartError as exc:
                warnings.warn(f"skipping four step {step}: {exc}", stacklevel=2)
                done += 1
                continue
            done += 1
            if on_progress is not None:
                on_progress(done, total, reports[idx])
    return [r for r in reports if r is not None]
