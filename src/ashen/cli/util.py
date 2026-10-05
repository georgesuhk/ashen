"""`util` entry point: housekeeping that is not gathering, plotting or
tracing -- one function at a time, chosen with --func.

- trace_organize: in each case's run folder, remove the restart links and
  copies left in trace folders, and move traces from older layouts to where
  ashen looks for them now (ashen.trace_tidy).
- downsample_restarts: delete the restarts whose step is not a multiple of
  --every, keeping every step cases.toml uses and the first and last
  (ashen.restart_thin). Irreversible, so it only shows what it would delete
  until given --apply.
- delete_figures: delete the run's figures (.png, .gif) wherever `plot`
  writes them, keeping every cache and trace output, to free disk space;
  `plot` draws them again (ashen.figure_clean). Like downsample_restarts,
  it only says what it would delete, and how much space that frees, until
  given --apply.
- compress_traces: repack each trace's particle diagnostics file to the
  size of its data, losslessly -- JOREK writes it ~50000/n_markers times
  too big (ashen.diag_repack). Says how much it would free until given
  --apply. Finished traces are repacked by `ptrace` anyway; this is for
  the ones from before.

Run from the folder holding cases.toml, like `analyse`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ashen.cli._common import CASE_ERRORS, CASE_HELP, error, load_cases_or_exit, resolve_selection
from ashen.diag_repack import data_size, diag_files, is_compressed, repack
from ashen.figure_clean import figure_files
from ashen.restart_thin import apply_thin, case_steps, plan_thin
from ashen.trace_tidy import job_may_be_running, apply_tidy, plan_tidy

__all__ = ["FUNCS", "build_parser", "main"]

#: What --func can be.
FUNCS = ("trace_organize", "downsample_restarts", "delete_figures", "compress_traces")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="util",
        description="Housekeeping for the runs in cases.toml, one function at a time.",
    )
    parser.add_argument(
        "--func", "-func", required=True, choices=FUNCS,
        help="trace_organize: remove restart links/copies left in trace folders, and move "
        "traces from older layouts to ptrace/<exe>/E<eV>eV_n<markers>/. "
        "downsample_restarts: keep only the restarts at multiples of --every steps. "
        "delete_figures: delete the figures plot made (it can draw them again). "
        "compress_traces: repack trace diagnostics files losslessly to their data's size",
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
        help="trace_organize: show what would be removed and moved, without doing it",
    )
    parser.add_argument(
        "--every", type=int, default=None, metavar="STEPS",
        help="downsample_restarts: keep the restarts whose step is a multiple of this "
        "(e.g. 40 to go from every 20 steps to every 40)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="downsample_restarts, delete_figures, compress_traces: actually do it. "
        "Without it, only what would be done, and the space it frees, is shown",
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
    if args.func == "downsample_restarts":
        if args.every is None or args.every < 1:
            error("downsample_restarts needs --every STEPS (>= 1): the step spacing to keep")
            return 1
        return _downsample_restarts(names, cases, every=args.every, apply=args.apply)
    if args.func == "delete_figures":
        return _delete_figures(names, apply=args.apply)
    if args.func == "compress_traces":
        return _compress_traces(names, apply=args.apply)
    return 1


#: A diagnostics file written to more recently than this [s] may belong to
#: a trace still running, which appends to it: left alone.
_RECENT = 600


def _compress_traces(names: list[str], *, apply: bool) -> int:
    root = Path.cwd()
    before = after = n_files = 0
    failed = []
    for name in names:
        run_dir = root / name
        files = diag_files(run_dir) if run_dir.is_dir() else []
        if not files:
            continue
        print(f"==== {name} ====")
        for path in files:
            where = path.relative_to(run_dir)
            size = path.stat().st_size
            try:
                # first: a running trace's file is never compressed, and a
                # repack has just written it, which would look like one
                if is_compressed(path):
                    print(f"  keep     {where}  ({_human(size)}): already compressed")
                    continue
            except Exception as exc:
                error(f"{name}: {where}: {exc} -- left as it is")
                failed.append(f"{name}/{where}")
                continue
            busy = None
            if (path.parent / "ptrace_meta.json").is_file():
                busy = job_may_be_running(path.parent)
            age = time.time() - path.stat().st_mtime
            if busy is None and age < _RECENT:
                busy = f"written to {age:.0f} s ago, so its trace may still be running"
            if busy:
                print(f"  keep     {where}  ({_human(size)}): {busy}")
                continue
            try:
                if not apply:
                    data = data_size(path)
                    print(f"  repack   {where}  {_human(size)} -> at most {_human(data)}")
                    before, after, n_files = before + size, after + data, n_files + 1
                    continue
                result = repack(path)
            except Exception as exc:  # report, keep the file, go on with the others
                error(f"{name}: {where}: {exc} -- left as it is")
                failed.append(f"{name}/{where}")
                continue
            if result.skipped:
                print(f"  keep     {where}  ({_human(size)}): {result.skipped}")
                continue
            print(f"  repacked {where}  {_human(result.before)} -> {_human(result.after)}")
            before, after, n_files = before + result.before, after + result.after, n_files + 1
    if apply:
        print(f"repacked {n_files} file(s), {_human(before)} -> {_human(after)}, freeing "
              f"{_human(before - after)}; every value checked identical")
    else:
        print(f"would repack {n_files} file(s), {_human(before)} -> at most {_human(after)} "
              f"(less after compression), freeing at least {_human(before - after)} -- nothing "
              "changed yet; add --apply to repack")
    if failed:
        error(f"{len(failed)} file(s) not repacked: {', '.join(failed)}")
        return 1
    return 0


def _delete_figures(names: list[str], *, apply: bool) -> int:
    root = Path.cwd()
    total = count = 0
    for name in names:
        run_dir = root / name
        if not run_dir.is_dir():
            print(f"==== {name} ====\n  no such folder, skipped")
            continue
        files = figure_files(run_dir)
        if not files:
            continue
        size = sum(p.stat().st_size for p in files)
        by_folder: dict[Path, int] = {}
        for path in files:
            by_folder[path.parent] = by_folder.get(path.parent, 0) + 1
        print(f"==== {name} ====")
        for folder, n in sorted(by_folder.items()):
            print(f"  {'delete' if apply else 'would delete'} {n} figure(s) in "
                  f"{folder.relative_to(run_dir)}/")
        if apply:
            for path in files:
                path.unlink()
        total += size
        count += len(files)
    if apply:
        print(f"deleted {count} figure(s), freeing {_human(total)} -- `plot` draws them again")
    else:
        print(f"would delete {count} figure(s), freeing {_human(total)} -- nothing deleted "
              "yet; add --apply to delete")
    return 0


def _downsample_restarts(names: list[str], cases, *, every: int, apply: bool) -> int:
    root = Path.cwd()
    freed = n_deleted = 0
    for name in names:
        run_dir = root / name
        print(f"==== {name} ====")
        if not run_dir.is_dir():
            print("  no such folder, skipped")
            continue
        plan = plan_thin(run_dir, every, keep=case_steps(cases[name]))
        if not plan.n_steps:
            print("  no restarts")
            continue
        n_steps_deleted = len({step for step, _, _ in plan.delete})
        verb = "deleted" if apply else "would delete"
        print(f"  {plan.n_steps} restart step(s): {verb} {n_steps_deleted}, keeping "
              f"{plan.n_kept} ({_human(plan.freed)})")
        by_reason: dict[str, list[int]] = {}
        for step, reason in sorted(plan.kept_anyway.items()):
            by_reason.setdefault(reason, []).append(step)
        for reason, steps in by_reason.items():
            shown = ", ".join(map(str, steps[:8])) + (f", ... ({len(steps)})" if len(steps) > 8 else "")
            print(f"  kept, not a multiple of {every} but {reason}: {shown}")
        if apply:
            apply_thin(plan)
        freed += plan.freed
        n_deleted += n_steps_deleted
    if apply:
        print(f"deleted {n_deleted} restart step(s), freeing {_human(freed)}")
    else:
        print(f"would delete {n_deleted} restart step(s), freeing {_human(freed)} -- "
              "nothing deleted yet; add --apply to delete (it cannot be undone)")
    return 0
