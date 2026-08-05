"""Radial profile extraction via jorek2_postproc.

Ports ``castor3d/util/diagnostics/gather_profiles.py``'s ``run_single_var``
and ``get_postproc_profiles``. Staging and execution now go through
:func:`ashen.jorek2.run_tool` instead of a second hand-rolled
``shell=True`` subprocess call.

**Not ported in this pass**: ``plot_postproc_profs`` and ``postproc_get_q``
(matplotlib plotting on top of the ``.npz`` caches this module writes) --
see ``ashen/diagnostics/poincare.py``'s module docstring for why plotting is
deferred, and ``ashen/KNOWN_ISSUES.md`` for the hardcoded-``R0`` issue that
plotting code carries.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np

from ashen.jorek2 import Jorek2Run, MissingRestartError, run_tool
from ashen.paths import RunPaths, step_str
from ashen.postproc import profile_script, read_postproc_profile

__all__ = ["extract_profile", "expand_compound_vars", "gather_profiles"]

_TOR_MODE_PREFIX = {"average": "exprs_averaged_s", "midplane": "exprs_midplane_s"}

#: Compound variables that aren't real jorek2_postproc outputs -- they're
#: derived downstream (e.g. q from r_minor/Btheta/Btor) from these components.
#: Ports the two ad hoc expansions in gather_profiles.py:96-106.
_COMPOUND_VARS = {
    "q": ("r_minor", "Btheta", "Btor"),
    "Jgrad": ("currdens", "Btheta", "Btor", "r_minor"),
}


def expand_compound_vars(variables: list[str]) -> list[str]:
    """Replace derived variables (``q``, ``Jgrad``) with their components,
    matching ``gather_profiles.py:94-106``."""
    out = list(variables)
    for compound, components in _COMPOUND_VARS.items():
        if compound in out:
            out.remove(compound)
            out.extend(c for c in components if c not in out)
    return out


def extract_profile(
    run: Jorek2Run,
    step: int,
    var: str,
    coords_var: str,
    *,
    n_points: int,
    tor_mode: str,
    dest_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """One ``(step, var)`` pair. Ports ``gather_profiles.py:17``
    ``run_single_var``.

    Stages the *actual* padded restart filename into the scratch dir (not
    ``jorek_restart.h5``) and copies the exe alongside it, matching the
    legacy behaviour exactly -- unlike Poincare tracing, this wasn't
    confirmed to be safe to simplify, so it is preserved as-is.
    """
    if tor_mode not in _TOR_MODE_PREFIX:
        raise ValueError(f"tor_mode {tor_mode!r} has no associated file prefix")
    out_name = f"{_TOR_MODE_PREFIX[tor_mode]}{step_str(step, run.pad_width)}.dat"

    script = profile_script(run.namelist.name, step, coords_var, var, n_points, tor_mode=tor_mode)
    collected = run_tool(
        run,
        "jorek2_postproc",
        step=step,
        dest_dir=dest_dir,
        outputs=[f"postproc/{out_name}"],
        stdin_text=script,
        restart_name=run.restart_path(step).name,
        copy_exe=True,
    )
    headers, blocks = read_postproc_profile(collected[f"postproc/{out_name}"])
    data = blocks[step]
    return data[:, headers.index(coords_var)], data[:, headers.index(var)]


def gather_profiles(
    run: Jorek2Run,
    paths: RunPaths,
    steps: list[int],
    variables: list[str],
    *,
    coords_var: str = "Psi_N",
    tor_mode: str = "midplane",
    n_points: int = 100,
    n_workers: int = 4,
    force: bool = False,
    on_progress: Callable[[int, int, int, str], None] | None = None,
) -> None:
    """Gathers every ``(step, var)`` profile in parallel and caches each as
    ``postproc/{coords_var}_{var}_{step}.npz``. Ports
    ``gather_profiles.py:89`` ``get_postproc_profiles``.

    ``force`` replaces the legacy ``force_data = True`` hardcoded at
    ``analysis.py:76`` (which meant every diagnostic always re-ran, cache or
    not) with a real opt-in flag.

    ``on_progress(done, total, step, var)``, if given, fires as each
    ``(step, var)`` pair finishes -- in completion order, not submission
    order, since that's the only order a process pool can report in.
    """
    variables = expand_compound_vars(variables)
    paths.postproc_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for step in steps:
        for var in variables:
            cache = paths.postproc_dir / f"{coords_var}_{var}_{step_str(step, paths.pad_width)}.npz"
            if force or not cache.is_file():
                tasks.append((step, var, cache))

    if not tasks:
        return

    extract_one = partial(
        extract_profile,
        run,
        coords_var=coords_var,
        n_points=n_points,
        tor_mode=tor_mode,
        dest_dir=paths.postproc_dir / "_scratch",
    )
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(extract_one, step, var): (step, var, cache)
            for step, var, cache in tasks
        }
        done = 0
        for future in as_completed(futures):
            step, var, cache = futures[future]
            try:
                coords_out, var_out = future.result()
            except MissingRestartError as exc:
                warnings.warn(f"skipping profile step {step} var {var!r}: {exc}", stacklevel=2)
                done += 1
                continue
            np.savez_compressed(cache, x=coords_out, y=var_out)
            done += 1
            if on_progress is not None:
                on_progress(done, total, step, var)
