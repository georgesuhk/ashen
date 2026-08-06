"""Radial profile extraction via jorek2_postproc.

Ports ``castor3d/util/diagnostics/gather_profiles.py``'s ``run_single_var``
and ``get_postproc_profiles``. Staging and execution now go through
:func:`ashen.jorek2.run_tool` instead of a second hand-rolled
``shell=True`` subprocess call.

Drawing lives in :mod:`ashen.plotting.profiles`, which consumes the ``.npz``
caches written here via :func:`read_profile_series`. ``postproc_get_q``
(the derived q-profile half of the legacy ``plot_postproc_profs``, and the
``dJ/dr``-at-the-q=2-surface scatter buried inside it) is still not ported --
see ``ashen/KNOWN_ISSUES.md`` #8.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ashen.jorek2 import Jorek2Error, Jorek2Run, MissingRestartError, run_tool
from ashen.paths import RunPaths, step_str
from ashen.postproc import profile_script, read_postproc_profile

__all__ = [
    "extract_profile", "expand_compound_vars", "gather_profiles",
    "read_profile_series", "TOR_MODES",
]

#: Postproc command -> the output filename prefix it writes under ``postproc/``.
#:
#: The three midplane variants are distinct commands with distinct output
#: names (``exec_commands.f90:1332-1345``), not one command with an option:
#: bare ``midplane`` is ``BOTH_SIDES``, so its cut crosses the magnetic axis
#: and ``Psi_N`` runs 1 -> 0 -> 1 along it -- double-valued, and useless as a
#: radial coordinate. ``midplane outer`` (LFS only) is the one to use when
#: ``coords_var = "Psi_N"``.
_TOR_MODE_PREFIX = {
    "average": "exprs_averaged_s",
    "midplane": "exprs_midplane_s",
    "midplane outer": "exprs_outer-midplane_s",
    "midplane inner": "exprs_inner-midplane_s",
}

#: The valid ``tor_mode`` values, for config validation in ashen.cases.
TOR_MODES: tuple[str, ...] = tuple(_TOR_MODE_PREFIX)

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
    surfaces: int = 100,
    rad_range: tuple[float, float] = (0.001, 0.999),
    nmaxsteps: int = 2500,
    deltaphi: float = 0.3,
    nsmallsteps: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """One ``(step, var, tor_mode)`` triple. Ports ``gather_profiles.py:17``
    ``run_single_var``.

    Stages the *actual* padded restart filename into the scratch dir (not
    ``jorek_restart.h5``) and copies the exe alongside it, matching the
    legacy behaviour exactly -- unlike Poincare tracing, this wasn't
    confirmed to be safe to simplify, so it is preserved as-is.

    The ``surfaces``/``rad_range``/``nmaxsteps``/``deltaphi``/``nsmallsteps``
    arguments only reach ``average`` (see :func:`ashen.postproc.
    profile_script`); the midplane family ignores them and uses ``n_points``.
    """
    if tor_mode not in _TOR_MODE_PREFIX:
        raise ValueError(
            f"tor_mode {tor_mode!r} has no associated file prefix; "
            f"expected one of {list(_TOR_MODE_PREFIX)}"
        )
    out_name = f"{_TOR_MODE_PREFIX[tor_mode]}{step_str(step, run.pad_width)}.dat"

    script = profile_script(
        run.namelist.name, step, coords_var, var, n_points, tor_mode=tor_mode,
        surfaces=surfaces, rad_range=rad_range, nmaxsteps=nmaxsteps,
        deltaphi=deltaphi, nsmallsteps=nsmallsteps,
    )
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
    tor_modes: list[str] | str = "midplane",
    n_points: int = 100,
    n_workers: int = 4,
    force: bool = False,
    surfaces: int = 100,
    rad_range: tuple[float, float] = (0.001, 0.999),
    nmaxsteps: int = 2500,
    deltaphi: float = 0.3,
    nsmallsteps: int = 3,
    on_progress: Callable[[int, int, int, str, str], None] | None = None,
) -> dict[str, int]:
    """Gathers every ``(step, var, tor_mode)`` profile in parallel, caching
    each via :meth:`ashen.paths.RunPaths.profile_cache`. Ports
    ``gather_profiles.py:89`` ``get_postproc_profiles``, extended to several
    modes at once.

    Returns ``{tor_mode: n_succeeded}`` so the caller can tell a mode that
    partly worked from one that never worked at all.

    **A failing step no longer aborts the whole gather.** ``average`` does
    real field-line tracing, and a trace that fails hits a hard Fortran
    ``stop`` (``mod_straight_field_line.f90:518``) rather than returning an
    error -- so ``jorek2_postproc`` exits non-zero and :func:`ashen.jorek2.
    run_tool` raises :class:`~ashen.jorek2.Jorek2Error`. That is the *normal*
    outcome for late steps of a nonlinear run, where the flux surfaces the
    average is defined on no longer exist (see ``KNOWN_ISSUES.md`` #9), so it
    is caught per task, warned about, and stepped over. Where the average
    stops succeeding is itself the physics signal.

    ``force`` replaces the legacy ``force_data = True`` hardcoded at
    ``analysis.py:76`` (which meant every diagnostic always re-ran, cache or
    not) with a real opt-in flag.

    ``on_progress(done, total, step, var, tor_mode)``, if given, fires as
    each task finishes -- in completion order, not submission order, since
    that's the only order a process pool can report in.
    """
    if isinstance(tor_modes, str):
        tor_modes = [tor_modes]
    variables = expand_compound_vars(variables)
    paths.postproc_dir.mkdir(parents=True, exist_ok=True)

    succeeded = {mode: 0 for mode in tor_modes}

    tasks = []
    for mode in tor_modes:
        for step in steps:
            for var in variables:
                cache = paths.profile_cache(coords_var, var, step, mode)
                if force or not cache.is_file():
                    tasks.append((step, var, mode, cache))
                else:
                    succeeded[mode] += 1

    if not tasks:
        return succeeded

    scratch = paths.postproc_dir / "_scratch"
    total = len(tasks)

    def _record(step: int, var: str, mode: str, cache: Path, done: int) -> None:
        try:
            coords_out, var_out = extract_profile(
                run, step, var, coords_var,
                n_points=n_points, tor_mode=mode, dest_dir=scratch,
                surfaces=surfaces, rad_range=rad_range, nmaxsteps=nmaxsteps,
                deltaphi=deltaphi, nsmallsteps=nsmallsteps,
            )
        except (MissingRestartError, Jorek2Error) as exc:
            warnings.warn(
                f"skipping profile step {step} var {var!r} mode {mode!r}: {exc}",
                stacklevel=2,
            )
            return
        np.savez_compressed(cache, x=coords_out, y=var_out)
        succeeded[mode] += 1
        if on_progress is not None:
            on_progress(done, total, step, var, mode)

    # Serial fallback, same shape (and same reason) as run_four_scan and
    # cli/analyse.py's _gather_zero_d/_gather_qprofile: it keeps a single
    # task or n_workers<=1 out of a process pool entirely, so a caller that
    # substitutes a stand-in extract_profile (e.g. via monkeypatch) is
    # actually exercised -- a pooled call is pickled by reference and
    # re-imports the real function fresh in the child process, where no
    # in-process patch can reach it.
    if n_workers <= 1 or len(tasks) <= 1:
        for done, (step, var, mode, cache) in enumerate(tasks, start=1):
            _record(step, var, mode, cache, done)
        return succeeded

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                extract_profile, run, step, var, coords_var,
                n_points=n_points, tor_mode=mode, dest_dir=scratch,
                surfaces=surfaces, rad_range=rad_range, nmaxsteps=nmaxsteps,
                deltaphi=deltaphi, nsmallsteps=nsmallsteps,
            ): (step, var, mode, cache)
            for step, var, mode, cache in tasks
        }
        done = 0
        for future in as_completed(futures):
            step, var, mode, cache = futures[future]
            done += 1
            try:
                coords_out, var_out = future.result()
            except (MissingRestartError, Jorek2Error) as exc:
                warnings.warn(
                    f"skipping profile step {step} var {var!r} mode {mode!r}: {exc}",
                    stacklevel=2,
                )
                continue
            np.savez_compressed(cache, x=coords_out, y=var_out)
            succeeded[mode] += 1
            if on_progress is not None:
                on_progress(done, total, step, var, mode)

    return succeeded


def read_profile_series(
    paths: RunPaths,
    steps: list[int],
    coords_var: str,
    var: str,
    tor_mode: str = "midplane",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{step: (x, y)}`` from the cached ``.npz`` profiles.

    A step with no cache is simply absent from the result rather than
    ``nan``-filled: unlike a mode-amplitude time series, where a gap has to
    line up with its neighbours on a shared time axis, each profile here is
    an independent curve, so "not gathered" and "gathered as empty" stay
    distinguishable by just not drawing a line.
    """
    series: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for step in steps:
        cache = paths.profile_cache(coords_var, var, step, tor_mode)
        if not cache.is_file():
            continue
        with np.load(cache) as data:
            series[step] = (np.asarray(data["x"]), np.asarray(data["y"]))
    return series
