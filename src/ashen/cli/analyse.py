"""``analyse`` entry point: the data-gathering side of ``analysis.py``'s rewrite.

Ports the zeroD-gathering and Poincare/profile *data collection* stages of
``Columbia/NL_kinks/analysis.py``, driven by a declarative ``cases.toml``
(:mod:`ashen.cases`) instead of ~25 stacked variable reassignments and a
hand-edited ``diags`` list.

**Not ported in this pass**: the plotting stages (``plot_poincare``,
``plot_field_line_diffusion``, ``plot_stochastic_factor``,
``plot_postproc_profiles``'s plotting half, ``max_fieldline_pos`` --
the last of which calls a function, ``plot_max_fieldline_pos``, that does not
exist anywhere in the legacy tree and would raise ``NameError`` if selected).
Profile caches keep the ``.npz`` format the legacy plotting code reads. The
Poincare cache does **not** -- it moved to per-line HDF5 so scans can be
widened and extended in place, which the legacy dense format cannot express.
See ``ashen/KNOWN_ISSUES.md`` #4 and #5.
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from ashen.cases import Case, CasesError, load_cases
from ashen.config import SiteConfigError, load_site
from ashen.diagnostics import four as four_diag
from ashen.diagnostics import poincare as poincare_diag
from ashen.diagnostics import profiles as profiles_diag
from ashen.diagnostics import qprofile as qprofile_diag
from ashen.jorek2 import Jorek2Run, MissingRestartError, run_zero_d
from ashen.paths import RunPaths, read_float

DIAG_CHOICES = ("zerod", "poincare", "profiles", "four")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyse",
        description="Gather JOREK postprocessing data for cases defined in cases.toml.",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("cases.toml"),
        help="path to cases.toml (default: ./cases.toml)",
    )
    parser.add_argument(
        "--case", action="append", dest="selected",
        help="case name to run (repeatable; default: every case in the file)",
    )
    parser.add_argument("--list", action="store_true", help="list defined cases and exit")
    parser.add_argument(
        "--diag", action="append", dest="diags", choices=DIAG_CHOICES,
        help="which diagnostic(s) to gather (repeatable; default: zerod)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run even if a cached output already exists (replaces the "
        "legacy hardcoded force_data=True at analysis.py:76)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="restart steps to process concurrently (default: site.toml's "
        "[diagnostics] n_workers)",
    )
    parser.add_argument(
        "--omp-threads", type=int, default=None,
        help="OpenMP threads per jorek2_* process (default: site.toml's "
        "[diagnostics] omp_threads)",
    )
    parser.add_argument("--site", type=Path, default=None, help="explicit site.toml")
    parser.add_argument(
        "--show-config", action="store_true",
        help="print where site.toml was found and what each key resolved to",
    )
    return parser


def _gather_zero_d(
    jrun: Jorek2Run, paths: RunPaths, steps: list[int], *, force: bool, n_workers: int
) -> None:
    """zeroD for every step, cache-gated and fanned out across processes --
    same shape as :func:`ashen.diagnostics.poincare.run_poincare_scan`.

    A step whose restart file is missing is warned about and skipped rather
    than aborting the whole case: restart files for a run still in progress
    are routinely incomplete for the tail of a requested step range.
    """
    total = len(steps)
    tasks: list[tuple[int, int]] = []
    for i, step in enumerate(steps, start=1):
        if force or not paths.zero_d(step).is_file():
            tasks.append((i, step))
        else:
            print(f"  zerod {i}/{total}: step {step} [cached]")

    if not tasks:
        return

    if n_workers <= 1 or len(tasks) <= 1:
        for i, step in tasks:
            print(f"  zerod {i}/{total}: step {step}")
            try:
                run_zero_d(jrun, step, paths)
            except MissingRestartError as exc:
                warnings.warn(f"skipping zerod step {step}: {exc}", stacklevel=2)
        return

    one = partial(run_zero_d, jrun, paths=paths)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(one, step): (i, step) for i, step in tasks}
        for future in as_completed(futures):
            i, step = futures[future]
            try:
                future.result()
            except MissingRestartError as exc:
                warnings.warn(f"skipping zerod step {step}: {exc}", stacklevel=2)
                continue
            print(f"  zerod {i}/{total}: step {step}")


def _gather_qprofile(
    jrun: Jorek2Run, paths: RunPaths, steps: list[int], *, force: bool, n_workers: int
) -> None:
    """q-profile for every step, cache-gated and fanned out across processes
    -- same shape as :func:`_gather_zero_d`, since
    :func:`ashen.diagnostics.qprofile.run_qprofile_step` runs
    ``jorek2_postproc`` in place exactly like :func:`ashen.jorek2.run_zero_d`.
    """
    total = len(steps)
    tasks: list[tuple[int, int]] = []
    for i, step in enumerate(steps, start=1):
        if force or not paths.qprofile(step).is_file():
            tasks.append((i, step))
        else:
            print(f"  qprofile {i}/{total}: step {step} [cached]")

    if not tasks:
        return

    if n_workers <= 1 or len(tasks) <= 1:
        for i, step in tasks:
            print(f"  qprofile {i}/{total}: step {step}")
            try:
                qprofile_diag.run_qprofile_step(jrun, step, paths)
            except MissingRestartError as exc:
                warnings.warn(f"skipping qprofile step {step}: {exc}", stacklevel=2)
        return

    one = partial(qprofile_diag.run_qprofile_step, jrun, paths=paths)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(one, step): (i, step) for i, step in tasks}
        for future in as_completed(futures):
            i, step = futures[future]
            try:
                future.result()
            except MissingRestartError as exc:
                warnings.warn(f"skipping qprofile step {step}: {exc}", stacklevel=2)
                continue
            print(f"  qprofile {i}/{total}: step {step}")


def _run_case(
    case: Case,
    *,
    diags: list[str],
    force: bool,
    n_workers: int,
    omp_threads: int,
) -> None:
    run_dir = Path.cwd() / case.name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"case {case.name!r}: no such folder {run_dir}")

    paths = RunPaths.detect(run_dir)
    jrun = Jorek2Run(
        run_dir=run_dir,
        exe_dir=run_dir,
        namelist=run_dir / case.namelist,
        pad_width=paths.pad_width,
    )

    # poincare implies zerod even if not requested explicitly: `plot`'s LCTT
    # figure reads each step's true time from the zeroD cache
    # (cli/plot.py:_plot_connection_length), so a poincare-only gather that
    # skipped it would leave LCTT with nothing to read. Cache-gated per step,
    # so this costs nothing once zerod has already run.
    if "zerod" in diags or "poincare" in diags:
        _gather_zero_d(jrun, paths, case.steps, force=force, n_workers=n_workers)

    if "poincare" in diags:
        real_psi_edge = read_float(paths.real_psi_edge)
        psi_n_in = [p * real_psi_edge for p in case.psi_n_in]

        def _poincare_progress(done: int, total: int, report) -> None:
            print(f"  poincare {done}/{total}: {report}")

        # No cache-existence check here any more: run_poincare_step plans
        # against the cache per field line, so an already-satisfied step costs
        # a read and traces nothing. --force still discards and retraces.
        poincare_diag.run_poincare_scan(
            jrun, paths, case.steps, psi_n_in,
            ang_sample_freq=case.ang_sample_freq,
            n_turns=case.n_turns,
            phi_start=case.phi_start,
            n_workers=n_workers,
            omp_threads=omp_threads,
            force=force,
            on_progress=_poincare_progress,
        )

    if "profiles" in diags:

        def _profiles_progress(done: int, total: int, step: int, var: str, mode: str) -> None:
            print(f"  profiles {done}/{total}: step {step} {var} [{mode}]")

        succeeded = profiles_diag.gather_profiles(
            jrun, paths, case.steps, case.vars,
            coords_var=case.coords_var, tor_modes=case.tor_mode,
            n_points=case.n_points, n_workers=n_workers, force=force,
            surfaces=case.profile_surfaces,
            rad_range=tuple(case.profile_rad_range),
            nmaxsteps=case.profile_nmaxsteps,
            deltaphi=case.profile_deltaphi,
            on_progress=_profiles_progress,
        )
        # A mode that partly worked is expected -- `average` stops being
        # computable once the flux surfaces it averages over are gone. A mode
        # that never worked once is a configuration problem, and easy to miss
        # among per-step warnings, so it is called out separately.
        for mode, n_ok in succeeded.items():
            if n_ok == 0:
                print(
                    f"  warning: tor_mode {mode!r} produced no profiles at all "
                    f"(every step failed or was skipped)"
                )
                if mode.split()[0] == "average":
                    print(
                        "    `average` traces field lines and dies where flux "
                        "surfaces no longer close; try lowering "
                        "profile_rad_range's upper bound (see KNOWN_ISSUES.md #9)"
                    )

    if "four" in diags:
        # The q-profile locates each mode's q=m/n rational surface for
        # plot's rational_surface_series (ashen.diagnostics.four_modes) --
        # gathered alongside four so a plot-only run never needs a second
        # `analyse` pass just to add it.
        _gather_qprofile(jrun, paths, case.steps, force=force, n_workers=n_workers)

        def _four_progress(done: int, total: int, report) -> None:
            print(f"  four {done}/{total}: {report}")

        four_diag.run_four_scan(
            jrun, paths, case.steps,
            nstpts=case.nstpts, ntht=case.ntht, nmaxsteps=case.nmaxsteps,
            deltaphi=case.deltaphi, nsmallsteps=case.nsmallsteps,
            rad_range=tuple(case.rad_range),
            n_workers=n_workers, omp_threads=omp_threads, force=force,
            on_progress=_four_progress,
        )


def _resolve_parallelism(args) -> tuple[int, int]:
    """CLI flags override ``site.toml``'s ``[diagnostics]``, which in turn
    overrides the derived default.

    A missing or unreadable ``site.toml`` is not fatal here -- the analysis
    path needs no machine paths, only a core budget -- so it falls back to the
    same derivation an empty ``[diagnostics]`` table would give.
    """
    from ashen.config import Diagnostics

    try:
        diagnostics = load_site(args.site).diagnostics
    except SiteConfigError:
        diagnostics = Diagnostics()
    diagnostics = Diagnostics(
        n_workers=args.n_workers if args.n_workers is not None else diagnostics.n_workers,
        omp_threads=(
            args.omp_threads if args.omp_threads is not None else diagnostics.omp_threads
        ),
    )
    return diagnostics.resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_config:
        try:
            site = load_site(args.site)
        except SiteConfigError as exc:
            print(f"error: {exc}")
            return 1
        print(site.describe())
        return 0

    try:
        cases = load_cases(args.cases)
    except CasesError as exc:
        print(f"error: {exc}")
        return 1

    if args.list:
        for name, case in cases.items():
            note = f" -- {case.note}" if case.note else ""
            print(f"{name} ({len(case.steps)} steps){note}")
        return 0

    selected = args.selected or list(cases)
    unknown = [name for name in selected if name not in cases]
    if unknown:
        print(f"error: unknown case(s) {unknown}; --list to see defined cases")
        return 1

    diags = args.diags or ["zerod"]
    n_workers, omp_threads = _resolve_parallelism(args)

    for name in selected:
        print(f"==== {name} ====")
        try:
            _run_case(
                cases[name],
                diags=diags,
                force=args.force,
                n_workers=n_workers,
                omp_threads=omp_threads,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}")
            return 1

    return 0
