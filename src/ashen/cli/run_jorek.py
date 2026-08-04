"""``run_jorek`` entry point.

Phase 1 implements only ``--show-config``. The run stages (``--run_eq``,
``--run``, ``--run_r``, ``--run_sw``, ``--dry-run``) arrive in Phase 3 once
shotfile validation and the runner exist; they are declared here already so the
CLI surface is visible and stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ashen.config import SiteConfigError, load_site

NOT_YET = (
    "Not implemented yet (Phase 3). Phase 1 provides --show-config only; use "
    "the existing Columbia/run_jorek.py until the runner lands."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_jorek",
        description="Prepare and submit a JOREK run from a shotfile.",
    )
    parser.add_argument(
        "shot_file",
        nargs="?",
        help="shotfile.py describing the run (default: ./shotfile.py)",
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

    stages = parser.add_argument_group("run stages (Phase 3)")
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

    stage_flags = (
        args.run, args.run_i, args.run_r, args.run_eq, args.run_sw, args.dry_run
    )
    if any(stage_flags) or args.shot_file:
        print(f"error: {NOT_YET}")
        return 2

    build_parser().print_help()
    return 0
