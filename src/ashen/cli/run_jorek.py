"""``run_jorek`` entry point: prepare a run folder and submit JOREK jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ashen.case_notebook import NOTEBOOK_NAME, write_case_notebook
from ashen.cli._common import error, show_config
from ashen.config import SiteConfigError, load_site
from ashen.runner import (
    prepare_restart,
    restart_density_change,
    prepare_run,
    submit_eq,
    submit_main,
    submit_restart,
    submit_starwall,
)
from ashen.scan import ScanError, apply_plan, format_value, load_scan, plan_scan, update_cases_toml
from ashen.shotfile import ShotfileError, load_shotfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_jorek",
        description="Prepare and submit a JOREK run from a shotfile.",
    )
    parser.add_argument(
        "shot_file",
        nargs="?",
        default=None,
        help="shotfile.py describing the run (required unless --show-config)",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print where site.toml was found and what each key resolved to",
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=None,
        help="explicit site.toml (default: $ASHEN_SITE, then upward search)",
    )

    scan = parser.add_argument_group("several runs from one scan file")
    scan.add_argument("--scan", type=Path, default=None, metavar="FILE",
                      help="a scan file: list the run folders it would create; "
                           "with --apply, create and prepare them (and run the stages given)")
    scan.add_argument("--apply", action="store_true",
                      help="with --scan: create the folders instead of only listing them")
    scan.add_argument("--force", action="store_true",
                      help="with --scan --apply: also rewrite a run folder whose "
                           "shotfile.py differs from the scan's")

    stages = parser.add_argument_group("run stages")
    stages.add_argument("--dry-run", action="store_true",
                        help="show planned actions without touching disk")
    stages.add_argument("--run", action="store_true", help="submit the main run")
    stages.add_argument("--run_i", action="store_true", help="main run, interactive")
    stages.add_argument("--run_r", action="store_true", help="submit a restart run")
    stages.add_argument("--keep-inputs", action="store_true",
                        help="with --run_r: do not prepare the folder again. Only tstep_n, "
                             "nstep_n and nout of in_main_r are set from the shotfile; "
                             "profiles, boundary and every other namelist field stay as the "
                             "run was started with. For runs prepared by an older ashen.")
    stages.add_argument("--run_eq", action="store_true", help="equilibrium, interactive")
    stages.add_argument("--run_sw", action="store_true", help="equilibrium then STARWALL")
    return parser


def _submit(args, paths, site, params, *, dry_run: bool) -> None:
    if args.run_eq:
        submit_eq(paths, site, params, dry_run=dry_run)
    if args.run_i:
        submit_main(paths, site, params, interactive=True, dry_run=dry_run)
    if args.run:
        submit_main(paths, site, params, interactive=False, dry_run=dry_run)
    if args.run_r:
        submit_restart(paths, site, params, dry_run=dry_run)
    if args.run_sw:
        submit_starwall(paths, site, params, dry_run=dry_run)


def _restart_keeping_inputs(args, params, site, run_dir: Path) -> int:
    """--run_r --keep-inputs: set the run length in in_main_r and submit."""
    try:
        result = prepare_restart(params, site, run_dir, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        error(str(exc))
        return 1
    if args.dry_run:
        print("Dry run -- no files were written. Planned actions:")
        for action in result.actions:
            print(f"  {action}")
    else:
        print(f"inputs kept; in_main_r: tstep_n = {params.tstep_n}, nstep_n = {params.nstep_n}, "
              f"nout = {params.nout}")
    command = submit_restart(result.paths, site, params, dry_run=args.dry_run)
    if args.dry_run:
        print(f"  would run: {command.splitlines()[-1]}")
    return 0


def _ensure_notebook(run_dir: Path) -> None:
    """Put the viewer notebook in the run folder if it has none, before the
    shotfile is even read: a shotfile that run_jorek refuses (a q0 outside
    the reachable window, say) is exactly what the viewer helps to fix."""
    try:
        if write_case_notebook(run_dir) is not None:
            print(f"viewer notebook written: {run_dir / NOTEBOOK_NAME}")
    except OSError as exc:
        error(f"could not write {NOTEBOOK_NAME}: {exc}")


def _run_scan(args) -> int:
    """--scan: list what a scan file would create, or with --apply create,
    prepare and (with a stage flag) launch each run."""
    try:
        scan = load_scan(args.scan)
        # the campaign is the one the scan file sits in, wherever this is run from
        site = load_site(args.site, start=args.scan.resolve().parent)
        plan = plan_scan(scan, site.root)
    except (ScanError, SiteConfigError) as exc:
        error(str(exc))
        return 1

    stages = [flag for flag in ("run_eq", "run_i", "run", "run_r", "run_sw") if getattr(args, flag)]
    print(f"scan {args.scan}  ->  {site.root / scan.folder}  ({len(plan)} run(s))")
    notes = {"new": "new", "same": "already there: not touched",
             "differs": "exists with a different shotfile.py: "
                        + ("will be rewritten (--force)" if args.force else "left alone (--force rewrites)")}
    for run in plan:
        what = ", ".join(f"{k} = {format_value(v)}" for k, v in run.values.items())
        print(f"  {run.name:<40s} {what:<30s} {notes[run.status]}")

    if not args.apply:
        print("Nothing written. Add --apply to create these folders"
              + (f" and run: {', '.join('--' + s for s in stages)}" if stages else "") + ".")
        return 0

    written = apply_plan(scan, site.root, plan, force=args.force)
    failed = 0
    for run in written:
        _ensure_notebook(run.run_dir)
        try:
            params = load_shotfile(run.run_dir / "shotfile.py")
            result = prepare_run(params, site, run.run_dir, run_sw=args.run_sw)
            print(f"  prepared {run.name}")
            _submit(args, result.paths, site, params, dry_run=False)
        except (ShotfileError, NotImplementedError, FileNotFoundError, OSError) as exc:
            failed += 1
            error(f"{run.name}: {exc}")
    if not written:
        print("  nothing to create")

    if scan.cases:
        try:
            added, cases_notes = update_cases_toml(scan, site.root, plan)
        except ScanError as exc:
            error(str(exc))
            return 1
        if added:
            print(f"  added to {site.root / 'cases.toml'}:")
            print("".join(f"    {line}\n" for line in added.splitlines()), end="")
        for note in cases_notes:
            print(f"  note: {note}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_config:
        return show_config(args.site)

    if args.scan is not None:
        if args.shot_file is not None or args.dry_run or args.keep_inputs:
            error("--scan takes no shotfile, no --dry-run and no --keep-inputs: "
                  "without --apply it only lists")
            return 1
        return _run_scan(args)
    if args.apply or args.force:
        error("--apply and --force go with --scan")
        return 1

    if args.shot_file is None:
        build_parser().print_help()
        return 0

    if args.keep_inputs and (
        not args.run_r or args.run or args.run_i or args.run_eq or args.run_sw
    ):
        error("--keep-inputs goes with --run_r alone")
        return 1

    run_dir = Path.cwd()
    if not args.dry_run:
        _ensure_notebook(run_dir)

    try:
        params = load_shotfile(args.shot_file)
    except ShotfileError as exc:
        error(str(exc))
        return 1

    try:
        site = load_site(args.site)
    except SiteConfigError as exc:
        error(str(exc))
        return 1

    if args.keep_inputs:
        return _restart_keeping_inputs(args, params, site, run_dir)

    if args.run_r:
        change = restart_density_change(params, run_dir)
        if change is not None:
            error(
                f"this folder's in_main_r has central_density = {change[0]:g}; preparing it "
                f"again would write {change[1]:g} (rho_const / 1e20) and a profile in other "
                "units, changing the density normalisation of the run being restarted. "
                "Nothing was written.\n"
                "  To restart it as it was started:  run_jorek shotfile.py --run_r --keep-inputs\n"
                "  To prepare it again on purpose:   run_jorek shotfile.py   (then --run_r)"
            )
            return 1

    try:
        result = prepare_run(
            params, site, run_dir,
            dry_run=args.dry_run, run_sw=args.run_sw,
        )
    except (ShotfileError, NotImplementedError, FileNotFoundError) as exc:
        error(str(exc))
        return 1

    if args.dry_run:
        print("Dry run -- no files were written. Planned actions:")
        for action in result.actions:
            print(f"  {action}")
    else:
        print(f"with_refluid: {params.with_refluid}")
        print(f"freeboundary: {params.freeboundary}")
        print(f"extend_bnd:   {params.extend_bnd}")
        print(f"eta:          {params.eta}")
        print(f"JOREK shot folder populated at: {run_dir}")

    _submit(args, result.paths, site, params, dry_run=args.dry_run)

    return 0
