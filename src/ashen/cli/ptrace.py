"""`ptrace` entry point: run a particle program (JOREK's own, or ashen's
ptrace_gc) against an existing run's restarts.

A case traces when it sets ptrace_exe in the same cases.toml `analyse` and
`plot` read -- see ashen.cases; staging and running is ashen.ptracing. Run
from the folder holding cases.toml, like `analyse`.

Launched like run_jorek's main run: --run_i runs it here (interactive_prelude,
mpirun with the case's ptrace_n_mpi and ptrace_omp_threads); --run queues it
with sbatch and a jobscript from site.toml's jobscripts folder (-job, default
2h), whose own #SBATCH lines and srun set the ranks and threads. Without
either, it reports where each trace stands -- and concludes queued ones that
have ended.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ashen.cli._common import (
    CASE_ERRORS,
    CASE_HELP,
    error,
    load_cases_or_exit,
    resolve_selection,
    show_config,
)
from ashen.config import SiteConfigError, load_site
from ashen.jorek2 import enable_tool_output
from ashen.particle_programs import SETTINGS_FILE
from ashen.ptracing import (
    LOG_FILE,
    META_FILE,
    PtraceError,
    is_current,
    last_error,
    last_jobscript,
    plan_ptrace,
    poll_job,
    ptrace_dir,
    run_ptrace,
)

__all__ = ["DEFAULT_JOB", "build_parser", "main"]

#: The jobscript --run queues with unless -job names another.
DEFAULT_JOB = "2h"


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
        "--case", action="extend", nargs="+", dest="selected", metavar="CASE",
        help=f"case(s) to trace: {CASE_HELP} (default: every case that sets ptrace_exe)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list the cases that trace, then exit",
    )
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument(
        "--run_i", action="store_true",
        help="run here, with site.toml's interactive_prelude and mpirun, and the "
        "case's ptrace_n_mpi ranks and ptrace_omp_threads threads",
    )
    launch.add_argument(
        "--run", action="store_true",
        help="queue with sbatch, using the jobscript -job names",
    )
    parser.add_argument(
        "-job", "--job", default=None, metavar="NAME",
        help=f"jobscript for --run, a file in site.toml's jobscripts folder "
        f"(e.g. 2h, 23h; default {DEFAULT_JOB})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rerun even if the trace already completed with these settings",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show what each trace would link and run (or queue), without doing it",
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
            start = case.ptrace_start_step
            end = case.ptrace_end_step
            span = (f"from step {start}" if start is not None else "from the first restart") + (
                f" to {end}" if end is not None else " to the last restart"
            )
            print(f"{name}: {case.ptrace_exe} {span}{note}")
        return 0

    if not tracing:
        error(f"no case in {args.cases} sets ptrace_exe")
        return 1
    matched = resolve_selection(args.selected, cases)
    if matched is None:
        return 1
    # A case named outright must trace; a pattern or folder picks the ones
    # that do out of what it matches.
    untraced = [
        name.rstrip("/") for name in args.selected or ()
        if name.rstrip("/") in cases and name.rstrip("/") not in tracing
    ]
    if untraced:
        error(f"case(s) {untraced} set no ptrace_exe")
        return 1
    selected = [name for name in matched if name in tracing]
    if not selected:
        error(f"none of the cases {args.selected} select sets ptrace_exe")
        return 1

    if args.job is not None and not args.run:
        error("-job chooses the jobscript for --run; add --run")
        return 1
    job = (args.job or DEFAULT_JOB) if args.run else None

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
        run_dir = Path.cwd() / name
        try:
            if not (args.run or args.run_i or args.dry_run):
                _report(case, run_dir, site, omp_threads)
                continue
            plan = plan_ptrace(case, run_dir, site, omp_threads=omp_threads, job=job)
            if args.dry_run:
                for line in plan.describe():
                    print(f"  {line}")
                continue
            _print_concluded(poll_job(plan.work_dir))
            print(
                f"  steps {plan.steps[0]}..{plan.steps[-1]} ({len(plan.steps)} restart(s))"
            )
            if plan.settings and (args.force or not is_current(plan)):
                print(f"  settings ({SETTINGS_FILE} in the trace folder):")
                for line in plan.describe_settings():
                    print(f"    {line}")
            result = run_ptrace(plan, force=args.force)
            if result.ran and result.job_id is not None:
                print(f"  queued: job {result.job_id} ({job}) in {plan.work_dir}; "
                      "`ptrace` again once it has ended")
                continue
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


def _print_concluded(queued) -> None:
    """Report a queued trace poll_job has just found ended."""
    if queued is None or queued.running:
        return
    if queued.error is not None:
        print(f"  job {queued.job_id} failed: {queued.error}")
        return
    lost = " -- stopped at its first lost particle" if queued.result.lost else ""
    print(f"  job {queued.job_id} done{lost}")
    for note in queued.result.notes:
        print(f"  note: {note}")


def _report(case, run_dir: Path, site, omp_threads: int) -> None:
    """Where a case's trace stands, concluding a queued one that has ended.
    Compared against the launch it last used: the same jobscript, or here."""
    work_dir = ptrace_dir(case, run_dir)
    queued = poll_job(work_dir)
    if queued is not None and queued.running:
        print(f"  job {queued.job_id}: {queued.state.lower()}")
        return
    _print_concluded(queued)
    if queued is not None:
        return
    plan = plan_ptrace(
        case, run_dir, site, omp_threads=omp_threads, job=last_jobscript(case, run_dir),
    )
    span = f"steps {plan.steps[0]}..{plan.steps[-1]} ({len(plan.steps)} restart(s))"
    if is_current(plan):
        print(f"  [cached] {span}: {work_dir}")
    elif not (work_dir / META_FILE).is_file():
        print(f"  not traced yet ({span}); --run_i to trace here, --run to queue it")
    elif last_error(work_dir):
        print(f"  last trace failed: {last_error(work_dir)}")
    else:
        print(f"  out of date or unfinished ({span}): the settings, restarts or "
              "executable changed since, or it was interrupted; --run_i or --run to retrace")
