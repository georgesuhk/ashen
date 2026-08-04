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
This CLI gathers and caches the same ``.npz`` files the legacy plotting code
reads, so legacy ``analysis.py`` remains usable for plotting against
Ashen-gathered data in the meantime. See ``ashen/KNOWN_ISSUES.md``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ashen.cases import Case, CasesError, load_cases
from ashen.config import SiteConfigError, load_site
from ashen.diagnostics import poincare as poincare_diag
from ashen.diagnostics import profiles as profiles_diag
from ashen.jorek2 import Jorek2Run, run_zero_d
from ashen.paths import RunPaths, read_float

DIAG_CHOICES = ("zerod", "poincare", "profiles")


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
    parser.add_argument("--n-workers", type=int, default=None, help="parallel worker count")
    parser.add_argument("--site", type=Path, default=None, help="explicit site.toml")
    parser.add_argument(
        "--show-config", action="store_true",
        help="print where site.toml was found and what each key resolved to",
    )
    return parser


def _run_case(case: Case, *, diags: list[str], force: bool, n_workers: int | None) -> None:
    run_dir = Path.cwd() / case.folder
    if not run_dir.is_dir():
        raise FileNotFoundError(f"case {case.name!r}: no such folder {run_dir}")

    paths = RunPaths.detect(run_dir)
    jrun = Jorek2Run(
        run_dir=run_dir,
        exe_dir=run_dir,
        namelist=run_dir / case.namelist,
        pad_width=paths.pad_width,
    )

    if "zerod" in diags:
        for step in case.steps:
            if force or not paths.zero_d(step).is_file():
                run_zero_d(jrun, step, paths)

    if "poincare" in diags:
        real_psi_edge = read_float(paths.real_psi_edge)
        psi_n_in = [p * real_psi_edge for p in case.psi_n_in]
        workers = n_workers or min(
            case.ang_sample_freq * max(len(psi_n_in), 1), os.cpu_count() or 1
        )
        for step in case.steps:
            if force or not paths.poincare_cache(step).is_file():
                poincare_diag.run_poincare_step(
                    jrun, paths, step, psi_n_in,
                    ang_sample_freq=case.ang_sample_freq,
                    n_turns=case.n_turns,
                    n_workers=workers,
                )

    if "profiles" in diags:
        profiles_diag.gather_profiles(
            jrun, paths, case.steps, case.vars,
            coords_var=case.coords_var, tor_mode=case.tor_mode,
            n_points=case.n_points, n_workers=n_workers or 4, force=force,
        )


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

    diags = args.diags or ["zerod"]

    for name in selected:
        print(f"==== {name} ====")
        try:
            _run_case(cases[name], diags=diags, force=args.force, n_workers=args.n_workers)
        except FileNotFoundError as exc:
            print(f"error: {exc}")
            return 1

    return 0
