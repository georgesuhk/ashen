"""Radial profile extraction via jorek2_postproc.

Ports gather_profiles.py's run_single_var and get_postproc_profiles;
staging/execution now goes through jorek2.run_tool, not a second hand-
rolled shell=True subprocess call.

Drawing lives in ashen.plotting.profiles, reading the .npz caches written
here via read_profile_series. Not yet ported: postproc_get_q (legacy
plot_postproc_profs's derived-q-profile half, incl. the dJ/dr-at-q=2
scatter) -- KNOWN_ISSUES.md #8.
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
    "read_profile_series", "edge_toroidal_field", "ensure_edge_toroidal_field",
    "TOR_MODES",
]

#: Postproc command -> output filename prefix under postproc/. The 3
#: midplane variants are distinct commands with distinct outputs
#: (exec_commands.f90:1332-1345), not one command with an option: bare
#: "midplane" is BOTH_SIDES, crosses the magnetic axis, Psi_N runs
#: 1->0->1 (double-valued, useless as a radial coord). Use "midplane
#: outer" (LFS only) when coords_var="Psi_N".
_TOR_MODE_PREFIX = {
    "average": "exprs_averaged_s",
    "midplane": "exprs_midplane_s",
    "midplane outer": "exprs_outer-midplane_s",
    "midplane inner": "exprs_inner-midplane_s",
}

#: The valid ``tor_mode`` values, for config validation in ashen.cases.
TOR_MODES: tuple[str, ...] = tuple(_TOR_MODE_PREFIX)

#: Compound variables that aren't real jorek2_postproc outputs -- derived
#: downstream from these components (e.g. q from r_minor/Btheta/Btor).
#: Ports gather_profiles.py:96-106's two ad hoc expansions.
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
    """One (step, var, tor_mode) triple. Ports gather_profiles.py:17
    run_single_var.

    Stages the actual padded restart filename (not jorek_restart.h5) and
    copies the exe alongside it, matching legacy exactly -- unlike Poincare
    tracing, not confirmed safe to simplify, so preserved as-is.

    surfaces/rad_range/nmaxsteps/deltaphi/nsmallsteps only reach `average`
    (postproc.profile_script); midplane family ignores them, uses n_points.
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
    missing = [name for name in (coords_var, var) if name not in headers]
    if missing:
        raise Jorek2Error(
            f"jorek2_postproc's {tor_mode!r} output for step {step} has no "
            f"{missing!r} column(s) (got {headers}) -- {var!r} may not be a "
            "valid expression for this tor_mode/model"
        )
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
    """Gathers every (step, var, tor_mode) profile in parallel, caching each
    via RunPaths.profile_cache. Ports gather_profiles.py:89
    get_postproc_profiles, extended to several modes at once.

    Returns {tor_mode: n_succeeded} so a partly-working mode is
    distinguishable from one that never worked.

    A failing step no longer aborts the whole gather. `average` does real
    field-line tracing; a failed trace hits a hard Fortran stop
    (mod_straight_field_line.f90:518), so jorek2_postproc exits non-zero and
    run_tool raises Jorek2Error -- the normal outcome for late nonlinear-run
    steps once the average's flux surfaces stop existing (KNOWN_ISSUES.md
    #9). Caught per task, warned, stepped over -- where `average` stops
    succeeding IS the physics signal.

    `force` replaces legacy analysis.py:76's hardcoded force_data=True
    (every diagnostic always re-ran) with a real opt-in flag.

    on_progress(done, total, step, var, tor_mode), if given, fires in
    completion order (the only order a process pool can report).
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
    """{step: (x, y)} from the cached .npz profiles.

    A step with no cache is absent, not nan-filled: unlike a mode-amplitude
    series needing a shared time axis, each profile here is an independent
    curve, so "not gathered" and "gathered empty" stay distinguishable by
    just not drawing a line.
    """
    series: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for step in steps:
        cache = paths.profile_cache(coords_var, var, step, tor_mode)
        if not cache.is_file():
            continue
        with np.load(cache) as data:
            series[step] = (np.asarray(data["x"]), np.asarray(data["y"]))
    return series


def edge_toroidal_field(
    paths: RunPaths, *, step: int = 0, psi_n: float = 1.0
) -> float | None:
    """Btor interpolated to psi_n (plasma edge by default) from the cached
    midplane profile at `step` (initial equilibrium by default) -- a fixed
    reference field for normalising a perturbation amplitude, independent
    of the axis-based logfile.b_axis.

    Uses "midplane outer", not bare "midplane": with coords_var="Psi_N",
    bare midplane crosses the axis, Psi_N runs 1->0->1 (double-valued,
    useless as an interpolation axis; see _TOR_MODE_PREFIX above).

    None if that profile hasn't been gathered (needs "Btor" in the case's
    profile vars, tor_mode incl. "midplane outer", for `step`) -- caller
    decides how to report, same convention as a missing logfile field.
    """
    cache = paths.profile_cache("Psi_N", "Btor", step, "midplane outer")
    if not cache.is_file():
        return None
    with np.load(cache) as data:
        x, y = np.asarray(data["x"]), np.asarray(data["y"])
    if x.size == 0:
        return None
    order = np.argsort(x)
    return float(np.interp(psi_n, x[order], y[order]))


def ensure_edge_toroidal_field(
    run: Jorek2Run,
    paths: RunPaths,
    *,
    step: int = 0,
    psi_n: float = 1.0,
    n_points: int = 100,
) -> float | None:
    """edge_toroidal_field, gathering the Btor "midplane outer" profile
    first if not cached yet.

    Runs a single jorek2_postproc call (gather_profiles, one (step, "Btor",
    "midplane outer") task) instead of requiring a separate `analyse
    --diag profiles` pass -- cheap/single-valued enough that gathering it
    on demand doesn't blur plot's slow-batch/fast-iterative split (see
    cli/plot's module docstring) the way a whole profile diagnostic would.

    None if the restart at `step` doesn't exist or the tool fails --
    gather_profiles already warns and skips per-task, nothing further to catch.
    """
    value = edge_toroidal_field(paths, step=step, psi_n=psi_n)
    if value is not None:
        return value
    gather_profiles(
        run, paths, [step], ["Btor"], tor_modes=["midplane outer"], n_points=n_points,
    )
    return edge_toroidal_field(paths, step=step, psi_n=psi_n)
