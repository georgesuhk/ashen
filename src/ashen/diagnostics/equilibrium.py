"""q0, edge q and l_i that a JOREK equilibrium actually has.

For comparison with what ``ffprime_method = "q_li"`` asked for
(ashen.current_profile.current_from_q_li), so they are defined the same
way: on the plasma, as a circular cylinder of the surfaces' effective radii.
JOREK's own zeroD ``li3`` and ``q95`` are for everything inside its last
closed surface, which with an extended boundary includes the vacuum region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ashen.diagnostics.four_modes import effective_minor_radius
from ashen.diagnostics.qprofile import read_qprofile
from ashen.paths import RunPaths, read_float
from ashen.postproc import read_zeroD, zero_d_is_usable

__all__ = ["AchievedQLi", "LiSeries", "achieved_q_li", "cylinder_li", "li_series"]


@dataclass(frozen=True)
class AchievedQLi:
    q0: float            #: q at the innermost point of the q-profile
    q_edge: float        #: q at the plasma edge (psi_N = real_psi_edge)
    li: float            #: cylinder l_i of the plasma, see cylinder_li
    a: float             #: effective minor radius of the plasma edge [m]
    real_psi_edge: float
    psi_n: np.ndarray    #: JOREK-grid psi_N of the q-profile, from the axis
    q: np.ndarray
    r: np.ndarray        #: effective minor radius at each psi_n [m]
    zero_d: dict         #: the step's zeroD values (li3, q95, Ip_tot, ...)


def cylinder_li(r: np.ndarray, q: np.ndarray, a: float) -> float:
    """l_i = 2 int_0^a B_p^2 r dr / (a^2 B_p(a)^2) with B_p ~ r / q.

    The same definition the q0/l_i/q_edge model uses: for that model's own
    r and q it returns the l_i it was built with.
    """
    r = np.asarray(r, dtype=float)
    q = np.abs(np.asarray(q, dtype=float))
    inside = r < a
    r_in = np.append(r[inside], a)
    q_in = np.append(q[inside], np.interp(a, r, q))
    b_p = r_in / q_in
    integral = np.sum(0.5 * (b_p[1:] ** 2 * r_in[1:] + b_p[:-1] ** 2 * r_in[:-1]) * np.diff(r_in))
    return float(2.0 * integral / (a**2 * b_p[-1] ** 2))


def achieved_q_li(paths: RunPaths, step: int = 0, *, f0: float) -> AchievedQLi | None:
    """q0, edge q and l_i of the equilibrium at ``step``, from its q-profile
    and zeroD caches. None if either cache is missing or unreadable."""
    q_path, zero_d_path = paths.qprofile(step), paths.zero_d(step)
    if not q_path.is_file() or not zero_d_is_usable(zero_d_path):
        return None
    zero_d = read_zeroD(zero_d_path)
    if not {"psi_axis", "psi_bnd", "R_axis"} <= zero_d.keys():
        return None
    psi_n_q, q_raw = read_qprofile(q_path)
    if psi_n_q.size < 2:
        return None
    try:
        real_psi_edge = read_float(paths.real_psi_edge)
    except (OSError, ValueError):
        real_psi_edge = 1.0

    psi_n, r = effective_minor_radius(
        psi_n_q, q_raw, delta_psi=zero_d["psi_bnd"] - zero_d["psi_axis"],
        f0=f0, r_axis=zero_d["R_axis"],
    )
    q = np.abs(q_raw)
    if len(psi_n) > len(q):                       # the axis point effective_minor_radius adds
        q = np.concatenate(([q[0]], q))
    edge = min(real_psi_edge, float(psi_n[-1]))
    a = float(np.sqrt(np.interp(edge, psi_n, r**2)))
    return AchievedQLi(
        q0=float(q[0]), q_edge=float(np.interp(edge, psi_n, q)), li=cylinder_li(r, q, a),
        a=a, real_psi_edge=float(real_psi_edge), psi_n=psi_n, q=q, r=r, zero_d=zero_d,
    )


@dataclass(frozen=True)
class LiSeries:
    """Internal inductance at each step, two ways."""

    steps: list[int]
    #: JOREK's zeroD ``li3``: over everything inside its last flux surface,
    #: which with an extended boundary includes the vacuum region. nan
    #: where the step has no usable zeroD cache.
    li3: np.ndarray
    #: achieved_q_li's l_i: the plasma only, to the real plasma edge, in the
    #: cylinder definition (cylinder_li). nan where the step has no
    #: q-profile cache, or ``f0`` was not given.
    li_plasma: np.ndarray


def li_series(paths: RunPaths, steps: list[int], *, f0: float | None = None) -> LiSeries:
    """l_i against step from the zeroD and q-profile caches. Gathers nothing."""
    li3 = np.full(len(steps), np.nan)
    li_plasma = np.full(len(steps), np.nan)
    for i, step in enumerate(steps):
        try:
            li3[i] = read_zeroD(paths.zero_d(step)).get("li3", np.nan)
        except (OSError, ValueError):
            pass
        if f0 is not None:
            try:
                achieved = achieved_q_li(paths, step, f0=f0)
            except (OSError, ValueError):
                achieved = None
            if achieved is not None:
                li_plasma[i] = achieved.li
    return LiSeries(list(steps), li3, li_plasma)
