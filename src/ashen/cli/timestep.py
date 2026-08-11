"""`timestep` entry point: simulation time at one or two restart steps, in
both SI seconds and JOREK's own code units.

Run from inside a prepared run folder (like `run_jorek --show-config`),
not driven by cases.toml -- a one-off lookup, not a campaign-wide gather.
Runs jorek2_postproc directly (diagnostics.timestep), so needs the same
environment analyse/plot do (a run folder with the tool symlinked in).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ashen.diagnostics.timestep import StepTime, step_time
from ashen.jorek2 import Jorek2Error, Jorek2Run, MissingRestartError
from ashen.paths import PaddingError, RunPaths

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timestep",
        description="Simulation time at one or two restart steps, in SI "
        "seconds and JOREK's own code units. Run from inside a prepared "
        "run folder.",
    )
    parser.add_argument(
        "steps", type=int, nargs="+", metavar="STEP",
        help="one restart step (report its time), or two (also report the "
        "elapsed time between them)",
    )
    parser.add_argument(
        "--namelist", default="in_main",
        help="namelist file to read (default: in_main)",
    )
    return parser


def _format(result: StepTime) -> str:
    return (
        f"step {result.step}: t = {result.time_si:.6e} s (SI), "
        f"t = {result.time_jorek:.6e} (JOREK units)"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.steps) > 2:
        print("error: pass at most two steps")
        return 1

    run_dir = Path.cwd()
    try:
        paths = RunPaths.detect(run_dir)
    except PaddingError as exc:
        print(f"error: {exc}")
        return 1
    jrun = Jorek2Run(
        run_dir=run_dir,
        exe_dir=run_dir,
        namelist=run_dir / args.namelist,
        pad_width=paths.pad_width,
    )

    results: list[StepTime] = []
    for step in args.steps:
        try:
            results.append(step_time(jrun, paths, step))
        except (FileNotFoundError, MissingRestartError, Jorek2Error) as exc:
            print(f"error: {exc}")
            return 1

    for result in results:
        print(_format(result))

    if len(results) == 2:
        first, second = results
        d_si = second.time_si - first.time_si
        d_jorek = second.time_jorek - first.time_jorek
        d_steps = second.step - first.step
        print(
            f"\N{GREEK CAPITAL LETTER DELTA}t (step {first.step} -> {second.step}): "
            f"{d_si:.6e} s (SI), {d_jorek:.6e} (JOREK units)"
        )
        if d_steps != 0:
            print(
                f"  = {d_si / d_steps:.6e} s/step (SI), "
                f"{d_jorek / d_steps:.6e} /step (JOREK units), over {d_steps} steps"
            )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
