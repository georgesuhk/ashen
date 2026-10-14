"""Which restart steps a diag already has cached: what ``plot --step all``
draws.

Each diag's cache files are found by their names, built from the RunPaths
method that names them for step 0, so the naming lives in one place. A
step counts as cached when its file is there; whether it reads cleanly is
left to the plot, as for a step named with ``--step``.
"""

from __future__ import annotations

import re
from pathlib import Path

from ashen.cases import Case
from ashen.diagnostics import re_current as rc
from ashen.diagnostics.profiles import expand_compound_vars
from ashen.paths import RunPaths

__all__ = ["CACHED_STEP_DIAGS", "cached_steps"]

#: Diags whose figures are drawn over restart steps; the rest (the particle
#: diags, scan_map) have no step list for ``--step all`` to fill.
CACHED_STEP_DIAGS = (
    "poincare", "connection_length", "theta_hist", "wetted_fraction",
    "four", "profiles", "re_current", "zerod",
)

#: The diags drawn from the Poincare caches.
_FROM_POINCARE = ("poincare", "connection_length", "theta_hist", "wetted_fraction")


def _steps_named_like(example: Path, step_str: str) -> set[int]:
    """Steps of the files in ``example``'s folder named like it, ``example``
    being the file for the step written ``step_str``. Any padding width."""
    name = example.name
    at = name.rfind(step_str)
    pattern = re.compile(
        re.escape(name[:at]) + r"(\d+)" + re.escape(name[at + len(step_str):])
    )
    if not example.parent.is_dir():
        return set()
    return {
        int(match.group(1))
        for path in example.parent.iterdir()
        if (match := pattern.fullmatch(path.name))
    }


def cached_steps(diag: str, case: Case, paths: RunPaths) -> list[int]:
    """Every step ``diag`` has a cache for in this run, ascending.

    - poincare, connection_length, theta_hist, wetted_fraction: the Poincare
      caches (``poinc_s<step>.h5``, or the legacy ``poinc_t<step>_*.npz``).
    - four: ``four_s<step>.h5``.
    - profiles: a cache for any of the case's variables and tor_modes.
    - re_current: both current-density profiles (``currdens`` and
      ``recurrdens``).
    - zerod: the zeroD files.
    """
    zero = paths.step_str(0)
    if diag in _FROM_POINCARE:
        steps = _steps_named_like(paths.poincare_cache(0), zero)
        for kind in ("psi_n", "theta", "R", "Z"):
            steps |= _steps_named_like(paths.poincare_cache_legacy(0, kind), zero)
        return sorted(steps)
    if diag == "four":
        return sorted(_steps_named_like(paths.four_cache(0), zero))
    if diag == "profiles":
        steps: set[int] = set()
        for var in expand_compound_vars(case.vars):
            for mode in case.tor_mode:
                steps |= _steps_named_like(
                    paths.profile_cache(case.coords_var, var, 0, mode), zero
                )
        return sorted(steps)
    if diag == "re_current":
        per_var = [
            _steps_named_like(paths.profile_cache(rc.COORDS_VAR, var, 0, rc.TOR_MODE), zero)
            for var in rc.VARIABLES
        ]
        return sorted(set.intersection(*per_var))
    if diag == "zerod":
        return sorted(_steps_named_like(paths.zero_d(0), zero))
    raise ValueError(f"{diag!r} is not drawn over restart steps")
