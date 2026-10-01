"""Scaffolding shared by the `analyse` and `plot` entry points.

Both drive the same `cases.toml`, both fan per-step `jorek2_*` work out
across processes, and both had grown their own copy of each. The copies had
already drifted in ways that were bugs rather than choices:

- `plot` treated an unparseable zeroD cache as absent (self-healing after an
  interrupted `jorek2_postproc`); `analyse` gated on mere existence, so the
  same truncated file was reported `[cached]` and never regenerated.
- `plot` tolerated any `Jorek2Error` on one step and carried on; `analyse`
  caught only `MissingRestartError`, so an unrelated tool failure escaped
  through `_run_case` and aborted the whole batch.
- `--show-config` printed the "path(s) do not exist here" note in
  `run_jorek` but not in the other two.

:func:`run_steps` is the fan-out engine both now share. It deliberately does
not own the reporting: `analyse` reports `i/total` progress over a batch it
planned up front, while `plot` reports an unexpected on-demand top-up. Those
messages differ because the situations differ, so they stay with the caller
and only the plumbing is shared.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

from ashen.cases import Case, CasesError, load_cases
from ashen.config import SiteConfigError, load_site
from ashen.diagnostics.four_cache import FourCacheError
from ashen.diagnostics.particles import ParticleFileError
from ashen.diagnostics.poincare_cache import PoincareCacheError
from ashen.jorek2 import Jorek2Error
from ashen.logfile import LogfileError
from ashen.padding import PaddingError
from ashen.shotfile import ShotfileError

__all__ = [
    "CASE_ERRORS",
    "CASE_HELP",
    "STEP_ERRORS",
    "error",
    "load_cases_or_exit",
    "matching_cases",
    "print_case_list",
    "resolve_selection",
    "run_steps",
    "show_config",
    "stale_steps",
]

#: Per-step failures a gather tolerates: the step is reported and skipped,
#: every other step still runs. `MissingRestartError` is a `FileNotFoundError`
#: subclass, so a run in progress whose tail restarts don't exist yet is
#: covered here too. A bare `FileNotFoundError` also covers "the exe isn't
#: symlinked into this run folder".
STEP_ERRORS = (FileNotFoundError, Jorek2Error)

#: Failures that abort one case but not the whole invocation. `PaddingError`
#: is the one a new user hits first -- running from a folder with no
#: `jorek*.h5` in it -- and used to escape as a raw traceback.
CASE_ERRORS = (
    FileNotFoundError,
    PaddingError,
    Jorek2Error,
    LogfileError,
    PoincareCacheError,
    FourCacheError,
    ParticleFileError,
    ShotfileError,
)


def error(message: str) -> None:
    """Report a fatal message on stderr.

    stderr, not stdout: these used to be printed to stdout, so redirecting a
    long `plot` run to a log file swallowed the errors into it.
    """
    print(f"error: {message}", file=sys.stderr)


def show_config(site_arg: Path | None) -> int:
    """Back `--show-config` for every entry point, including the missing-path
    note that only `run_jorek` used to print."""
    try:
        site = load_site(site_arg)
    except SiteConfigError as exc:
        error(str(exc))
        return 1
    print(site.describe())
    missing = site.missing()
    if missing:
        print(
            f"\nnote: {len(missing)} path(s) do not exist here: "
            f"{', '.join(missing)}"
        )
    return 0


def load_cases_or_exit(path: Path) -> dict[str, Case] | None:
    """Load cases.toml, reporting a parse failure. None means "already reported"."""
    try:
        return load_cases(path)
    except CasesError as exc:
        error(str(exc))
        return None


def print_case_list(cases: dict[str, Case]) -> int:
    """Back `--list` for both entry points."""
    for name, case in cases.items():
        note = f" -- {case.note}" if case.note else ""
        print(f"{name} ({len(case.steps)} steps){note}")
    return 0


#: What --case takes, for every entry point's help.
CASE_HELP = (
    "a case name, a pattern ('qa2.1*', quoted so the shell leaves it alone) or "
    "a folder holding cases (qa2.1_g2.3); several may follow one --case, and "
    "--case may be repeated"
)


def matching_cases(selector: str, names: Iterable[str]) -> list[str]:
    """The case names a --case value selects, in `names` order:

    - the case of exactly that name, if there is one; otherwise
    - with *, ? or [ in it, every name the pattern matches (as the shell
      would, except that * also crosses "/": 'qa2.1*' matches
      qa2.1_g2.3/eta1e-3_RE), or that lies in a folder it matches;
    - every case in that folder (qa2.1_g2.3 selects qa2.1_g2.3/eta1e-3_RE)
      -- which is also what an unquoted pattern becomes once the shell has
      expanded it to the run folders in the current directory.
    """
    names = list(names)
    selector = selector.rstrip("/")
    if selector in names:
        return [selector]
    if any(char in selector for char in "*?["):
        return [
            name for name in names
            if fnmatchcase(name, selector) or any(
                fnmatchcase(parent.as_posix(), selector)
                for parent in PurePosixPath(name).parents
            )
        ]
    return [name for name in names if name.startswith(selector + "/")]


def resolve_selection(
    selected: list[str] | None, cases: dict[str, Case]
) -> list[str] | None:
    """The cases to run, in cases.toml order -- every one when nothing was
    selected, else those the --case values select (matching_cases). None if
    a value selected nothing (already reported)."""
    if not selected:
        return list(cases)
    chosen: set[str] = set()
    unknown = []
    for selector in selected:
        matched = matching_cases(selector, cases)
        if not matched:
            unknown.append(selector)
        chosen.update(matched)
    if unknown:
        error(
            f"unknown case(s) {unknown}; --list to see defined cases "
            "(quote a pattern, --case 'qa2.1*', so the shell leaves it alone)"
        )
        return None
    return [name for name in cases if name in chosen]


def run_steps(
    run_one: Callable[[int], object],
    steps: Sequence[int],
    *,
    n_workers: int,
    on_done: Callable[[int], None],
    on_skip: Callable[[int, Exception], None],
    tolerate: tuple[type[BaseException], ...] = STEP_ERRORS,
) -> None:
    """Run `run_one(step)` for each step, serially or across processes.

    A step raising one of `tolerate` goes to `on_skip` and does not stop the
    others; anything else propagates. `run_one` must be picklable when
    `n_workers > 1` (a `functools.partial` of a module-level function is).

    The fan-out is skipped for a single step: spawning a process pool to run
    one `jorek2_*` invocation costs more than it saves.
    """
    if not steps:
        return

    if n_workers <= 1 or len(steps) <= 1:
        for step in steps:
            try:
                run_one(step)
            except tolerate as exc:
                on_skip(step, exc)
            else:
                on_done(step)
        return

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one, step): step for step in steps}
        for future in as_completed(futures):
            step = futures[future]
            try:
                future.result()
            except tolerate as exc:
                on_skip(step, exc)
            else:
                on_done(step)


def stale_steps(
    steps: Iterable[int], is_current: Callable[[int], bool], *, force: bool = False
) -> list[int]:
    """The steps needing work: those whose cache isn't current, or all of them
    under `force`."""
    return [step for step in steps if force or not is_current(step)]
