"""How much of the toroidal current is carried by runaways, and how much by
the thermal plasma: per radial position, and in total.

For a run with the RE fluid (``with_refluid``), JOREK's Ohm's law acts on
the current that is *not* runaway current: its resistive term is
``eta (zj - zj_RE)``, with ``zj_RE = Vlight F0 / (|B| R) n_RE``
(mod_integrals3D.f90, mod_expression.f90). So the split used here is the
model's own:

    j_thermal = j_total - j_RE

Per radial position, from two jorek2_postproc expressions gathered on the
outer midplane against Psi_N (SI units, A/m^2, toroidally averaged):

- ``currdens``    = -zj / (R mu0)            the total toroidal current density
- ``recurrdens``  = -zj_RE / (R mu0)         the runaways' share of it

Both carry the same sign convention, so they subtract directly.

In total, from the zeroD cache: ``Ip_tot`` and ``Ipre_tot``. These two do
NOT share a sign convention: ``Ip_tot`` integrates ``-zj/R``, ``Ipre_tot``
integrates ``+|Vlight| F0/(|B| R) n_RE / R`` (mod_integrals3D.f90, where
the RE integrand takes ``abs(Vlight)``). The runaway current in Ip_tot's
convention is therefore ``-sign(vpar_re_sign) * Ipre_tot`` (total_re_current).

A midplane profile is a cut, not a flux-surface average, and gives no
current enclosed by a surface: that needs the poloidal plane, not a line.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ashen import namelist as nml
from ashen.diagnostics.profiles import gather_profiles, read_profile_series
from ashen.jorek2 import Jorek2Run
from ashen.paths import RunPaths
from ashen.postproc import read_zeroD

__all__ = [
    "COORDS_VAR", "TOR_MODE", "VARIABLES",
    "CurrentDensity", "CurrentTotals",
    "current_density_series", "current_totals", "gather_re_current",
    "total_re_current", "vpar_re_sign",
]

#: The two postproc expressions, their radial coordinate and the cut.
VARIABLES = ("currdens", "recurrdens")
COORDS_VAR = "Psi_N"
TOR_MODE = "midplane outer"


@dataclass(frozen=True)
class CurrentDensity:
    """Toroidal current density on the outer midplane at one step [A/m^2]."""

    psi_n: np.ndarray     #: JOREK-grid psi_N (1 at the domain edge)
    total: np.ndarray
    re: np.ndarray
    thermal: np.ndarray   #: total - re


@dataclass(frozen=True)
class CurrentTotals:
    """Toroidal current through the whole poloidal plane, per step [A]."""

    steps: list[int]
    time: np.ndarray      #: seconds; nan where the step has no zeroD cache
    total: np.ndarray     #: Ip_tot
    re: np.ndarray        #: in Ip_tot's sign convention (total_re_current)
    thermal: np.ndarray   #: total - re


def gather_re_current(
    run: Jorek2Run,
    paths: RunPaths,
    steps: list[int],
    *,
    n_points: int = 200,
    n_workers: int = 4,
    force: bool = False,
    on_progress: Callable[[int, int, int, str, str], None] | None = None,
) -> int:
    """Gather both expressions for every step (two jorek2_postproc calls per
    step, cached like any other profile). Returns how many (step, variable)
    profiles are now cached."""
    succeeded = gather_profiles(
        run, paths, steps, list(VARIABLES), coords_var=COORDS_VAR, tor_modes=[TOR_MODE],
        n_points=n_points, n_workers=n_workers, force=force, on_progress=on_progress,
    )
    return succeeded[TOR_MODE]


def current_density_series(paths: RunPaths, steps: list[int]) -> dict[int, CurrentDensity]:
    """{step: CurrentDensity} for the steps that have both profiles cached."""
    total = read_profile_series(paths, steps, COORDS_VAR, "currdens", TOR_MODE)
    runaway = read_profile_series(paths, steps, COORDS_VAR, "recurrdens", TOR_MODE)
    series: dict[int, CurrentDensity] = {}
    for step in steps:
        if step not in total or step not in runaway:
            continue
        psi_n, j_total = total[step]
        psi_re, j_re = runaway[step]
        if psi_n.size == 0 or psi_re.size == 0:
            continue
        order = np.argsort(psi_n)
        psi_n, j_total = psi_n[order], j_total[order]
        # Same command, same linepoints: the two grids are the same. Put the
        # RE profile on the total's grid anyway rather than assume it.
        order_re = np.argsort(psi_re)
        j_re = np.interp(psi_n, psi_re[order_re], j_re[order_re])
        series[step] = CurrentDensity(psi_n, j_total, j_re, j_total - j_re)
    return series


def vpar_re_sign(paths: RunPaths) -> float | None:
    """``vpar_re_sign`` from the run's ``in_main``; None if it is not set
    there (then the sign of the total RE current cannot be told)."""
    try:
        return float(nml.read_field(paths.in_main, "vpar_re_sign"))
    except (OSError, nml.NamelistError, TypeError, ValueError):
        return None


def total_re_current(ipre_tot: float, sign: float) -> float:
    """zeroD's ``Ipre_tot`` in ``Ip_tot``'s sign convention.

    Ip_tot integrates -zj/R. Ipre_tot integrates +|Vlight| F0/(|B| R)
    n_RE / R, while the RE part of zj is Vlight F0/(|B| R) n_RE, so the
    RE current comparable with Ip_tot is -sign(Vlight) * Ipre_tot, and
    sign(Vlight) is ``vpar_re_sign``'s.
    """
    return -float(np.sign(sign)) * float(ipre_tot)


def current_totals(paths: RunPaths, steps: list[int], sign: float | None) -> CurrentTotals:
    """Total, RE and thermal current at each step from the zeroD caches.

    A step without a usable cache is nan. With ``sign`` None (vpar_re_sign
    unknown) the RE and thermal currents are nan: their sign would be a guess.
    """
    n = len(steps)
    time, total, runaway = (np.full(n, np.nan) for _ in range(3))
    for i, step in enumerate(steps):
        try:
            values = read_zeroD(paths.zero_d(step))
        except (OSError, ValueError):
            continue
        time[i] = values.get("Time", np.nan)
        total[i] = values.get("Ip_tot", np.nan)
        if sign is not None and "Ipre_tot" in values:
            runaway[i] = total_re_current(values["Ipre_tot"], sign)
    return CurrentTotals(list(steps), time, total, runaway, total - runaway)
