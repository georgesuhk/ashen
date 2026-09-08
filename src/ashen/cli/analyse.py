"""`analyse` entry point: the data-gathering side of analysis.py's rewrite.

Ports the zeroD-gathering and Poincare/profile data-collection stages of
Columbia/NL_kinks/analysis.py, driven by declarative cases.toml
(ashen.cases) instead of ~25 stacked variable reassignments and a
hand-edited diags list.

Not ported: the plotting stages (plot_poincare, plot_field_line_diffusion,
plot_stochastic_factor, plot_postproc_profiles's plotting half,
max_fieldline_pos -- the last calls plot_max_fieldline_pos, which doesn't
exist anywhere in the legacy tree and would raise NameError if selected).
Profile caches keep the .npz format legacy plotting reads. The Poincare
cache does NOT -- moved to per-line HDF5 so scans can be widened/extended
in place, which the legacy dense format can't express. KNOWN_ISSUES.md
#4, #5.
"""

from __future__ import annotations

import argparse
import warnings
from functools import partial
from pathlib import Path

from ashen.cases import Case
from ashen.cli._common import (
    CASE_ERRORS,
    error,
    load_cases_or_exit,
    print_case_list,
    resolve_selection,
    run_steps,
    show_config,
    stale_steps,
)
from ashen.config import SiteConfigError, load_site
from ashen.diagnostics import four as four_diag
from ashen.diagnostics import poincare as poincare_diag
from ashen.diagnostics import profiles as profiles_diag
from ashen.diagnostics import qprofile as qprofile_diag
from ashen.jorek2 import Jorek2Run, run_zero_d
from ashen.paths import RunPaths, read_float
from ashen.postproc import zero_d_is_usable

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


def _gather(
    label: str,
    run_one,
    is_current,
    steps: list[int],
    *,
    force: bool,
    n_workers: int,
) -> None:
    """Gather one per-step diagnostic: cache-gated, fanned out across
    processes, one step's failure never aborting the rest.

    `is_current(step)` decides whether a cached result counts as done --
    for zeroD that means parsing it, not just finding it (see
    postproc.zero_d_is_usable). `run_one` must be picklable for n_workers > 1.

    A step whose restart is missing, or whose jorek2_* call fails for any
    other reason, is warned about and skipped: restart files for a run in
    progress are routinely incomplete at the tail of a requested range, and
    one unreadable step shouldn't cost the other twenty their gather.
    """
    total = len(steps)
    order = {step: i for i, step in enumerate(steps, start=1)}
    todo = stale_steps(steps, is_current, force=force)
    pending = set(todo)
    for step in steps:
        if step not in pending:
            print(f"  {label} {order[step]}/{total}: step {step} [cached]")

    run_steps(
        run_one,
        todo,
        n_workers=n_workers,
        on_done=lambda step: print(f"  {label} {order[step]}/{total}: step {step}"),
        on_skip=lambda step, exc: warnings.warn(
            f"skipping {label} step {step}: {exc}", stacklevel=2
        ),
    )


def _gather_zero_d(
    jrun: Jorek2Run, paths: RunPaths, steps: list[int], *, force: bool, n_workers: int
) -> None:
    """zeroD for every step. See :func:`_gather`.

    Cache validity goes through zero_d_is_usable, not bare existence: an
    interrupted jorek2_postproc leaves a header-only file that used to be
    reported `[cached]` here forever while `plot` correctly regathered it.
    """
    _gather(
        "zerod",
        partial(run_zero_d, jrun, paths=paths),
        lambda step: zero_d_is_usable(paths.zero_d(step)),
        steps, force=force, n_workers=n_workers,
    )


def _gather_qprofile(
    jrun: Jorek2Run, paths: RunPaths, steps: list[int], *, force: bool, n_workers: int
) -> None:
    """q-profile for every step -- same shape as _gather_zero_d, since
    qprofile.run_qprofile_step runs jorek2_postproc in place exactly like
    jorek2.run_zero_d."""
    _gather(
        "qprofile",
        partial(qprofile_diag.run_qprofile_step, jrun, paths=paths),
        lambda step: paths.qprofile(step).is_file(),
        steps, force=force, n_workers=n_workers,
    )


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

    # poincare implies zerod even if not requested: plot's LCTT figure reads
    # each step's true time from the zeroD cache (cli/plot.py:
    # _plot_connection_length), so a poincare-only gather would leave LCTT
    # with nothing to read. Cache-gated per step, so free once zerod has
    # run. Each diag's own steps_for() override is respected -- zerod
    # covers the union, so separately-configured zerod/poincare step lists
    # both get what they need in one pass.
    zerod_steps: set[int] = set()
    if "zerod" in diags:
        zerod_steps.update(case.steps_for("zerod"))
    if "poincare" in diags:
        zerod_steps.update(case.steps_for("poincare"))
    if zerod_steps:
        _gather_zero_d(jrun, paths, sorted(zerod_steps), force=force, n_workers=n_workers)

    if "poincare" in diags:
        poincare_steps = case.steps_for("poincare")

        # Rational-surface highlight needs the qprofile cache to locate
        # q=m/n, same reasoning as poincare implying zerod above: gathered
        # here so a poincare-only case doesn't also need --diag four.
        if case.poincare_highlight:
            _gather_qprofile(jrun, paths, poincare_steps, force=force, n_workers=n_workers)

        real_psi_edge = read_float(paths.real_psi_edge)
        psi_n_in = [p * real_psi_edge for p in case.psi_n_in]

        def _poincare_progress(done: int, total: int, report) -> None:
            print(f"  poincare {done}/{total}: {report}")

        # No cache-existence check here: run_poincare_step plans against the
        # cache per field line, so an already-satisfied step costs a read
        # and traces nothing. --force still discards and retraces.
        poincare_diag.run_poincare_scan(
            jrun, paths, poincare_steps, psi_n_in,
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
            jrun, paths, case.steps_for("profiles"), case.vars,
            coords_var=case.coords_var, tor_modes=case.tor_mode,
            n_points=case.n_points, n_workers=n_workers, force=force,
            surfaces=case.profile_surfaces,
            rad_range=tuple(case.profile_rad_range),
            nmaxsteps=case.profile_nmaxsteps,
            deltaphi=case.profile_deltaphi,
            on_progress=_profiles_progress,
        )
        # A mode that partly worked is expected -- `average` stops being
        # computable once the flux surfaces it averages over are gone. A
        # mode that never worked is a config problem, easy to miss among
        # per-step warnings, so called out separately.
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
        four_steps = case.steps_for("four")

        # q-profile locates each mode's q=m/n rational surface for plot's
        # rational_surface_series (diagnostics.four_modes) -- gathered
        # alongside four so a plot-only run never needs a second `analyse`
        # pass just to add it.
        _gather_qprofile(jrun, paths, four_steps, force=force, n_workers=n_workers)

        def _four_progress(done: int, total: int, report) -> None:
            print(f"  four {done}/{total}: {report}")

        four_diag.run_four_scan(
            jrun, paths, four_steps,
            nstpts=case.nstpts, ntht=case.ntht, nmaxsteps=case.nmaxsteps,
            deltaphi=case.deltaphi, nsmallsteps=case.nsmallsteps,
            rad_range=tuple(case.rad_range),
            n_workers=n_workers, omp_threads=omp_threads, force=force,
            on_progress=_four_progress,
        )


def _resolve_parallelism(args) -> tuple[int, int]:
    """CLI flags override site.toml's [diagnostics], which overrides the
    derived default.

    A missing/unreadable site.toml isn't fatal here -- the analysis path
    needs no machine paths, only a core budget -- so it falls back to the
    same derivation an empty [diagnostics] table would give.
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
        return show_config(args.site)

    cases = load_cases_or_exit(args.cases)
    if cases is None:
        return 1

    if args.list:
        return print_case_list(cases)

    selected = resolve_selection(args.selected, cases)
    if selected is None:
        return 1

    diags = args.diags or ["zerod"]
    n_workers, omp_threads = _resolve_parallelism(args)

    # One bad case is reported and skipped, not fatal: an overnight gather
    # over twenty cases shouldn't lose the other nineteen because one folder
    # is missing or has no restart files yet. The exit code still reflects it.
    failed: list[str] = []
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
        except CASE_ERRORS as exc:
            error(f"{name}: {exc}")
            failed.append(name)

    if failed:
        error(f"{len(failed)} of {len(selected)} case(s) failed: {', '.join(failed)}")
        return 1

    return 0
