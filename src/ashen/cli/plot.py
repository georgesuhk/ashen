"""``plot`` entry point: figures from data ``analyse`` already gathered.

Deliberately a separate command from ``analyse``. Gathering is slow and
batch (minutes per step, meant to run once and be cached); plotting is fast
and iterative (re-run constantly while tuning a figure) -- keeping them apart
means re-plotting never risks touching the gathering path, and this CLI can
carry flags (``--dpi``, ``--log``/``--linear``, ``--smooth``) that would only
clutter ``analyse``.

Reads the same ``cases.toml`` as ``analyse`` (:mod:`ashen.cases`), and never
runs a ``jorek2_*`` tool itself -- only :mod:`ashen.diagnostics.poincare_cache`
and :mod:`ashen.postproc.read_zeroD` are read.

**In scope this pass**: Poincare puncture plots and the LC/LCTT
connection-length maps -- the two that consume the field-line cache built in
Phase 4b. Everything else the legacy ``data_jorek.py`` / ``gather_profiles.py``
plotted (macroscopic-variable traces, field-line diffusion, radial profiles,
stochastic factor, the never-implemented ``max_fieldline_pos``) is not ported
-- see ``ashen/KNOWN_ISSUES.md`` #4.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from ashen.cases import Case, CasesError, load_cases
from ashen.config import SiteConfigError, load_site
from ashen.diagnostics.connection_length import connection_length_matrix
from ashen.diagnostics.poincare_cache import read_step
from ashen.logfile import LogfileError, r_axis
from ashen.paths import RunPaths, read_float
from ashen.plotting.connection_length import plot_connection_length_map
from ashen.plotting.poincare import plot_poincare_step
from ashen.postproc import read_zeroD

DIAG_CHOICES = ("poincare", "connection_length")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot",
        description="Draw figures from data gathered by `analyse` for cases in cases.toml.",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("cases.toml"),
        help="path to cases.toml (default: ./cases.toml)",
    )
    parser.add_argument(
        "--case", action="append", dest="selected",
        help="case name to plot (repeatable; default: every case in the file)",
    )
    parser.add_argument("--list", action="store_true", help="list defined cases and exit")
    parser.add_argument(
        "--diag", action="append", dest="diags", choices=DIAG_CHOICES,
        help="which figure(s) to draw (repeatable; default: both)",
    )
    parser.add_argument(
        "--step", type=int, action="append", dest="steps",
        help="restart step to plot (repeatable; poincare only, default: every case step)",
    )
    parser.add_argument("--dpi", type=int, default=None, help="override figure DPI")
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="poincare steps rendered concurrently (default: site.toml's "
        "[diagnostics] n_workers)",
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="connection_length: linear colour scale instead of the default log",
    )
    parser.add_argument(
        "--smooth", action="store_true",
        help="connection_length: smooth over time, ignoring confined (inf) cells",
    )
    parser.add_argument("--site", type=Path, default=None, help="explicit site.toml")
    parser.add_argument(
        "--show-config", action="store_true",
        help="print where site.toml was found and what each key resolved to",
    )
    return parser


def _render_poincare_step(step: int, records, *, paths: RunPaths, kwargs: dict) -> Path:
    """Module-level so it can be pickled for :class:`ProcessPoolExecutor`."""
    out = paths.figures_dir / f"{step}_poincare.png"
    plot_poincare_step(records, out, title=f"t={step}", **kwargs)
    return out


def _plot_poincare(
    case: Case, paths: RunPaths, steps: list[int], *, dpi: int | None, n_workers: int = 1
) -> None:
    kwargs = {} if dpi is None else {"dpi": dpi}

    to_render: list[tuple[int, dict]] = []
    for step in steps:
        records = read_step(paths, step)
        if not records:
            print(f"  step {step}: no Poincare cache, skipped")
            continue
        to_render.append((step, records))

    if not to_render:
        return

    if n_workers <= 1 or len(to_render) <= 1:
        for step, records in to_render:
            out = _render_poincare_step(step, records, paths=paths, kwargs=kwargs)
            print(f"  step {step}: {out}")
        return

    one = partial(_render_poincare_step, paths=paths, kwargs=kwargs)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(one, step, records): step for step, records in to_render}
        outputs: dict[int, Path] = {}
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    for step, _ in to_render:
        print(f"  step {step}: {outputs[step]}")


def _plot_connection_length(
    case: Case, paths: RunPaths, steps: list[int], *, log: bool, smooth: bool, dpi: int | None
) -> None:
    real_psi_edge = read_float(paths.real_psi_edge)
    try:
        R0 = r_axis(paths.log)
    except LogfileError as exc:
        print(f"  error: {exc}")
        return

    psi_n_targets = [p * real_psi_edge for p in case.psi_n_in]
    records_by_step = {step: read_step(paths, step) for step in steps}
    matrix = connection_length_matrix(
        records_by_step, steps, psi_n_targets, real_psi_edge=real_psi_edge, R0=R0
    )

    true_times = []
    for step in steps:
        try:
            true_times.append(read_zeroD(paths.zero_d(step))["Time"])
        except (FileNotFoundError, KeyError):
            true_times = None
            break

    kwargs = {} if dpi is None else {"dpi": dpi}
    for plot_true_times in (True, False):
        if plot_true_times and true_times is None:
            print("  skipping LCTT: no zeroD cache for one or more steps (run analyse --diag zerod)")
            continue
        out = plot_connection_length_map(
            matrix, steps, case.psi_n_in, paths.figures_dir,
            true_times=true_times, plot_true_times=plot_true_times,
            log=log, smooth=smooth, **kwargs,
        )
        print(f"  {out}")


def _resolve_n_workers(args) -> int:
    """``--n-workers`` if given, else ``site.toml``'s ``[diagnostics]
    n_workers`` if that's explicitly set, else 1 (serial).

    Unlike ``cli/analyse.py``'s ``_resolve_parallelism``, an *unset*
    ``[diagnostics]`` (or no ``site.toml`` at all) does not derive a worker
    count from the machine's core count: rendering a handful of PNGs is cheap
    enough that auto-parallelising by default would just spawn processes for
    no measurable benefit, so plotting stays serial unless asked otherwise.
    """
    if args.n_workers is not None:
        return max(1, args.n_workers)
    try:
        diagnostics = load_site(args.site).diagnostics
    except SiteConfigError:
        return 1
    return max(1, diagnostics.n_workers) if diagnostics.n_workers else 1


def _run_case(
    case: Case,
    *,
    diags: list[str],
    steps: list[int] | None,
    log: bool,
    smooth: bool,
    dpi: int | None,
    n_workers: int = 1,
) -> None:
    run_dir = Path.cwd() / case.folder
    if not run_dir.is_dir():
        raise FileNotFoundError(f"case {case.name!r}: no such folder {run_dir}")
    paths = RunPaths.detect(run_dir)
    case_steps = steps or case.steps

    if "poincare" in diags:
        _plot_poincare(case, paths, case_steps, dpi=dpi, n_workers=n_workers)
    if "connection_length" in diags:
        _plot_connection_length(case, paths, case_steps, log=log, smooth=smooth, dpi=dpi)


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
            print(f"{name}: {case.folder} ({len(case.steps)} steps){note}")
        return 0

    selected = args.selected or list(cases)
    unknown = [name for name in selected if name not in cases]
    if unknown:
        print(f"error: unknown case(s) {unknown}; --list to see defined cases")
        return 1

    diags = args.diags or list(DIAG_CHOICES)
    n_workers = _resolve_n_workers(args)

    for name in selected:
        print(f"==== {name} ====")
        try:
            _run_case(
                cases[name],
                diags=diags,
                steps=args.steps,
                log=not args.linear,
                smooth=args.smooth,
                dpi=args.dpi,
                n_workers=n_workers,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}")
            return 1

    return 0
