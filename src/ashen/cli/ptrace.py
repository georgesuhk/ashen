"""`ptrace` entry point: run a particle program (JOREK's own, or ashen's
ptrace_gc) against an existing run's restarts.

A case traces when it sets ptrace_exe (and ptrace_start_step) in the same
cases.toml `analyse` and `plot` read -- see ashen.cases; staging and running
is ashen.ptracing. Run from the folder holding cases.toml, like `analyse`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ashen.cli._common import CASE_ERRORS, error, load_cases_or_exit, show_config
from ashen.config import SiteConfigError, load_site
from ashen.jorek2 import enable_tool_output
from ashen.ptracing import LOG_FILE, PtraceError, plan_ptrace, run_ptrace

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptrace",
        description="Run a particle program (JOREK's particles/examples, or ashen's "
        "ptrace_gc) against an existing run's restarts, for cases in cases.toml "
        "that set ptrace_exe.",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("cases.toml"),
        help="path to cases.toml (default: ./cases.toml)",
    )
    parser.add_argument(
        "--case", action="append", dest="selected",
        help="case to trace (repeatable; default: every case that sets ptrace_exe)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list the cases that trace, then exit",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rerun even if the trace already completed with these settings",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show what each trace would link and run, without doing it",
    )
    parser.add_argument(
        "--tool-output", action="store_true",
        help=f"echo the program's output live (it is always written to {LOG_FILE})",
    )
    parser.add_argument("--site", type=Path, default=None, help="explicit site.toml")
    parser.add_argument(
        "--show-config", action="store_true",
        help="print where site.toml was found and what each key resolved to",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.tool_output:
        enable_tool_output()

    if args.show_config:
        return show_config(args.site)

    cases = load_cases_or_exit(args.cases)
    if cases is None:
        return 1
    tracing = [name for name, case in cases.items() if case.ptrace_exe is not None]

    if args.list:
        if not tracing:
            print(f"no case in {args.cases} sets ptrace_exe")
        for name in tracing:
            case = cases[name]
            note = f" -- {case.note}" if case.note else ""
            end = f" to {case.ptrace_end_step}" if case.ptrace_end_step is not None else ""
            print(f"{name}: {case.ptrace_exe} from step {case.ptrace_start_step}{end}{note}")
        return 0

    selected = args.selected or tracing
    unknown = [name for name in selected if name not in cases]
    if unknown:
        error(f"unknown case(s) {unknown}; --list to see the cases that trace")
        return 1
    untraced = [name for name in selected if name not in tracing]
    if untraced:
        error(f"case(s) {untraced} set no ptrace_exe")
        return 1
    if not selected:
        error(f"no case in {args.cases} sets ptrace_exe")
        return 1

    try:
        site = load_site(args.site)
    except SiteConfigError as exc:
        error(str(exc))
        return 1
    _, omp_threads = site.diagnostics.resolve()

    failed: list[str] = []
    for name in selected:
        case = cases[name]
        print(f"==== {name} ({case.ptrace_exe}) ====")
        try:
            plan = plan_ptrace(case, Path.cwd() / name, site, omp_threads=omp_threads)
            if args.dry_run:
                for line in plan.describe():
                    print(f"  {line}")
                continue
            print(
                f"  steps {plan.steps[0]}..{plan.steps[-1]} ({len(plan.steps)} restart(s))"
            )
            result = run_ptrace(plan, force=args.force)
            lost = " -- stopped at its first lost particle" if result.lost else ""
            status = "done" if result.ran else "[cached]"
            print(f"  {status}: {plan.work_dir}{lost}")
            for note in result.notes:
                print(f"  note: {note}")
        except (PtraceError, *CASE_ERRORS) as exc:
            error(f"{name}: {exc}")
            failed.append(name)

    if failed:
        error(f"{len(failed)} of {len(selected)} trace(s) failed: {', '.join(failed)}")
        return 1
    return 0
