"""``run_jorek`` entry point: prepare a run folder and submit JOREK jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ashen.config import SiteConfigError, load_site
from ashen.runner import (
    prepare_run,
    submit_eq,
    submit_main,
    submit_restart,
    submit_starwall,
)
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

    stages = parser.add_argument_group("run stages")
    stages.add_argument("--dry-run", action="store_true",
                        help="show planned actions without touching disk")
    stages.add_argument("--run", action="store_true", help="submit the main run")
    stages.add_argument("--run_i", action="store_true", help="main run, interactive")
    stages.add_argument("--run_r", action="store_true", help="submit a restart run")
    stages.add_argument("--run_eq", action="store_true", help="equilibrium, interactive")
    stages.add_argument("--run_sw", action="store_true", help="equilibrium then STARWALL")
    stages.add_argument("--replace", action="store_true",
                        help="overwrite an existing run folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_config:
        try:
            site = load_site(args.site)
        except SiteConfigError as exc:
            print(f"error: {exc}")
            return 1
        print(site.describe())
        missing = site.missing()
        if missing:
            print(
                f"\nnote: {len(missing)} path(s) do not exist here: "
                f"{', '.join(missing)}"
            )
        return 0

    if args.shot_file is None:
        build_parser().print_help()
        return 0

    try:
        params = load_shotfile(args.shot_file)
    except ShotfileError as exc:
        print(f"error: {exc}")
        return 1

    try:
        site = load_site(args.site)
    except SiteConfigError as exc:
        print(f"error: {exc}")
        return 1

    run_dir = Path.cwd()

    try:
        result = prepare_run(
            params, site, run_dir,
            replace=args.replace, dry_run=args.dry_run, run_sw=args.run_sw,
        )
    except (ShotfileError, NotImplementedError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}")
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

    if args.run_eq:
        submit_eq(result.paths, site, params, dry_run=args.dry_run)
    if args.run_i:
        submit_main(result.paths, site, params, interactive=True, dry_run=args.dry_run)
    if args.run:
        submit_main(result.paths, site, params, interactive=False, dry_run=args.dry_run)
    if args.run_r:
        submit_restart(result.paths, site, params, dry_run=args.dry_run)
    if args.run_sw:
        submit_starwall(result.paths, site, params, dry_run=args.dry_run)

    return 0
