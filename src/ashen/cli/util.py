"""`util` entry point: housekeeping that is not gathering, plotting or
tracing -- one function at a time, chosen with --func.

- trace_organize: in each case's run folder, remove the restart links and
  copies left in trace folders, and move traces from older layouts to where
  ashen looks for them now (ashen.trace_tidy).

Run from the folder holding cases.toml, like `analyse`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ashen.cli._common import CASE_ERRORS, CASE_HELP, error, load_cases_or_exit, resolve_selection
from ashen.trace_tidy import apply_tidy, plan_tidy

__all__ = ["FUNCS", "build_parser", "main"]

#: What --func can be.
FUNCS = ("trace_organize",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="util",
        description="Housekeeping for the runs in cases.toml, one function at a time.",
    )
    parser.add_argument(
        "--func", "-func", required=True, choices=FUNCS,
        help="trace_organize: remove restart links/copies left in trace folders, and move "
        "traces from older layouts to ptrace/<exe>/E<eV>eV_n<markers>/",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("cases.toml"),
        help="path to cases.toml (default: ./cases.toml)",
    )
    parser.add_argument(
        "--case", action="extend", nargs="+", dest="selected", metavar="CASE",
        help=f"case(s) to work on: {CASE_HELP} (default: every case in the file)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show what would be removed and moved, without doing it",
    )
    return parser


def _trace_organize(names: list[str], *, dry_run: bool) -> int:
    root = Path.cwd()
    failed: list[str] = []
    freed = n_removed = n_moved = n_kept = 0
    for name in names:
        run_dir = root / name
        if not run_dir.is_dir():
            print(f"==== {name} ====\n  no such folder, skipped")
            continue
        try:
            plan = plan_tidy(run_dir)
        except CASE_ERRORS as exc:
            error(f"{name}: {exc}")
            failed.append(name)
            continue
        if not plan.actions:
            continue
        print(f"==== {name} ====")
        for action in plan.actions:
            print(f"  {action.describe(run_dir)}")
        if not dry_run:
            try:
                apply_tidy(plan)
            except OSError as exc:
                error(f"{name}: {exc}")
                failed.append(name)
                continue
        freed += plan.freed
        n_removed += sum(a.kind == "remove" for a in plan.actions)
        n_moved += sum(a.kind == "move" for a in plan.actions)
        n_kept += sum(a.kind == "keep" for a in plan.actions)

    size = f" (freeing {_human(freed)})" if freed else ""
    verb = ("would remove", "would move") if dry_run else ("removed", "moved")
    print(f"{verb[0]} {n_removed} restart link(s) or copies{size}; {verb[1]} {n_moved} "
          f"trace(s); {n_kept} left as they are")
    if failed:
        error(f"{len(failed)} case(s) failed: {', '.join(failed)}")
        return 1
    return 0


def _human(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases_or_exit(args.cases)
    if cases is None:
        return 1
    names = resolve_selection(args.selected, cases)
    if names is None:
        return 1
    if args.func == "trace_organize":
        return _trace_organize(names, dry_run=args.dry_run)
    return 1
