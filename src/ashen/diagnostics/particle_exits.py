"""Where traced particles leave the plasma: poloidal (theta) and toroidal
(phi) angle at each particle's exit.

The particle counterpart of ashen.diagnostics.theta_histogram, which finds
where field lines first cross out past a target psi_n. Here the input is a
particle program's diagnostics file -- JOREK's write_particle_diagnostics
(``particles/diagnostics/mod_particle_diagnostics.f90``), e.g. ptrace_gc's
``ptrace_diag.h5`` -- which holds, per particle group, one row per
diagnostics time of

- ``psi_n`` -- (psi - psi_axis)/(psi_limit - psi_axis), psi_limit the
  X-point's psi (or 0 with no X-point, which JOREK warns about in the log),
- ``R``, ``Z``, ``phi``, and ``theta`` = atan2(Z - Z_axis, R - R_axis) when
  the program asked for it,
- ``lost`` -- 1 once the particle has left the grid; its other values are
  then written as 0.

A particle *exits* at the first diagnostics time its psi_n exceeds the
chosen threshold, or -- given a boundary, e.g. the plasma boundary before
extend_bnd -- it is outside that boundary, whichever comes first; or, if it
leaves the grid before either, at its last recorded position on the grid.
It is not tracked after that. Exits are only as fine as the program's
diag_step.

Pure data: no matplotlib here (see ashen.plotting.particle_exits).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ashen.diagnostics.hdf5 import require_h5py
from ashen.diagnostics.particles import BoundaryExits, ParticleFileError, inside_polygon

__all__ = [
    "ExitResult",
    "ParticleHistory",
    "exit_angles",
    "exits_from_history",
    "read_particle_diag",
]

#: Datasets exit_angles needs from every group; theta is optional.
_REQUIRED = ("psi_n", "R", "Z", "phi", "lost")


@dataclass(frozen=True)
class ParticleHistory:
    """Every particle's diagnostics over time, all groups side by side.
    Arrays are (n_times, n_particles)."""

    path: Path
    #: Diagnostics times [s], (n_times,).
    time: np.ndarray
    psi_n: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    #: None when the program didn't write theta (see exit_angles' axis).
    theta: np.ndarray | None
    lost: np.ndarray

    @property
    def n(self) -> int:
        return int(self.psi_n.shape[1])

    def within(self, t_min: float, t_max: float | None) -> "ParticleHistory":
        """Only the diagnostics times in [t_min, t_max] (None: no upper
        bound), with a relative tolerance for the 4-byte times. Particles
        are not tracked past t_max -- an exit after it doesn't count."""
        tol = 1e-6
        keep = self.time >= t_min - abs(t_min) * tol
        if t_max is not None:
            keep &= self.time <= t_max + abs(t_max) * tol
        if keep.all():
            return self
        return replace(
            self, time=self.time[keep], psi_n=self.psi_n[keep], R=self.R[keep],
            Z=self.Z[keep], phi=self.phi[keep], lost=self.lost[keep],
            theta=None if self.theta is None else self.theta[keep],
        )


@dataclass(frozen=True)
class ExitResult:
    """Each exiting particle's angles, and what happened to the rest."""

    #: Poloidal angle at exit, in (-pi, pi].
    theta: np.ndarray
    #: Toroidal angle at exit, in [0, 2 pi).
    phi: np.ndarray
    #: Diagnostics time [s] of each exit.
    time: np.ndarray
    #: Of the exits: how many crossed the psi_n threshold on the grid...
    n_crossed: int
    #: ...how many were first outside the boundary exit_angles was given...
    n_outside_boundary: int
    #: ...and how many left the grid before a diagnostics time showed them
    #: past it (their angles are from their last position on the grid).
    n_left_grid: int
    #: Particles on the grid at the first diagnostics time -- the ones that
    #: could exit. A particle off the grid from the start is not counted.
    n_considered: int

    @property
    def n_exited(self) -> int:
        return self.n_crossed + self.n_outside_boundary + self.n_left_grid


def _as_time_by_particle(data: np.ndarray, n_times: int, where: str) -> np.ndarray:
    """Fortran writes (n_particles, n_times); h5py sees it (n_times,
    n_particles). A transposed layout is accepted too, told apart by the
    time axis' length."""
    data = np.asarray(data)
    if data.ndim == 2 and data.shape[0] == n_times:
        return data
    if data.ndim == 2 and data.shape[1] == n_times:
        return data.T
    raise ParticleFileError(
        f"{where} has shape {data.shape}, expected ({n_times}, n_particles) "
        "-- one row per diagnostics time in t"
    )


def read_particle_diag(path: Path | str) -> ParticleHistory:
    """Read a write_particle_diagnostics file (e.g. ptrace_diag.h5)."""
    h5py = require_h5py("reading particle diagnostics", ParticleFileError)
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise ParticleFileError(f"{path}: cannot open -- {exc}") from exc

    with f:
        groups = f.get("groups")
        if groups is None or not len(groups):
            raise ParticleFileError(f"{path}: no particle groups under /groups")
        time = None
        columns: dict[str, list[np.ndarray]] = {name: [] for name in (*_REQUIRED, "theta")}
        has_theta = True
        for name in sorted(groups):
            group = groups[name]
            where = f"{path}: group {name!r}"
            if "t" not in group:
                raise ParticleFileError(f"{where} has no 't' -- not a particle diagnostics file")
            t = np.asarray(group["t"], dtype=float).reshape(-1)
            if time is None:
                time = t
            elif t.shape != time.shape or not np.allclose(t, time):
                raise ParticleFileError(f"{where}: its times differ from the first group's")
            missing = [d for d in _REQUIRED if d not in group]
            if missing:
                raise ParticleFileError(
                    f"{where} has no {missing} -- the program must write psi_n, R, Z, "
                    "phi and lost (write_particle_diagnostics' `only`)"
                )
            for d in _REQUIRED:
                columns[d].append(_as_time_by_particle(group[d], t.size, f"{where}: {d}"))
            if "theta" in group and has_theta:
                columns["theta"].append(
                    _as_time_by_particle(group["theta"], t.size, f"{where}: theta")
                )
            else:
                has_theta = False

    def joined(d: str, dtype) -> np.ndarray:
        return np.concatenate(columns[d], axis=1).astype(dtype)

    return ParticleHistory(
        path=path, time=time,
        psi_n=joined("psi_n", float), R=joined("R", float), Z=joined("Z", float),
        phi=joined("phi", float), lost=joined("lost", int) > 0,
        theta=joined("theta", float) if has_theta else None,
    )


def _first(mask: np.ndarray) -> np.ndarray:
    """Per column (particle), the first row where mask is set; n_rows if none."""
    n_rows = mask.shape[0]
    return np.where(mask.any(axis=0), np.argmax(mask, axis=0), n_rows)


def _outside(history: ParticleHistory, boundary: np.ndarray) -> np.ndarray:
    """(n_times, n_particles): on the grid and outside boundary. A lost
    particle's position is written as 0, so it never counts."""
    inside = inside_polygon(history.R.ravel(), history.Z.ravel(), boundary)
    return ~history.lost & ~inside.reshape(history.R.shape)


def exits_from_history(history: ParticleHistory, boundary: np.ndarray) -> BoundaryExits:
    """When and where each particle was first outside boundary on the grid,
    at the diagnostics' resolution."""
    first = _first(_outside(history, boundary))
    n_times = history.time.size
    exited = first < n_times
    cols = np.flatnonzero(exited)
    time = np.full(history.n, np.inf)
    R = np.full(history.n, np.nan)
    Z = np.full(history.n, np.nan)
    time[cols] = history.time[first[cols]]
    R[cols] = history.R[first[cols], cols]
    Z[cols] = history.Z[first[cols], cols]
    return BoundaryExits(time=time, R=R, Z=Z, source=f"{history.path.name} (every diag_step)")


def exit_angles(
    history: ParticleHistory,
    *,
    psi_n: float,
    axis: tuple[float, float] | None = None,
    boundary: np.ndarray | None = None,
) -> ExitResult:
    """Where each particle first goes past psi_n (in the file's own psi_n,
    see the module docstring), outside boundary if given, or leaves the grid.

    theta is the file's own when it has one; otherwise it is computed as
    atan2(Z - Z_axis, R - R_axis) from axis = (R_axis, Z_axis), which is
    then required.
    """
    if history.theta is None and axis is None:
        raise ParticleFileError(
            f"{history.path}: no theta in the file, and no magnetic axis given to "
            "compute it from R and Z"
        )
    on_grid = ~history.lost
    crossed = on_grid & (history.psi_n > psi_n)
    theta_all = history.theta
    if theta_all is None:
        theta_all = np.arctan2(history.Z - axis[1], history.R - axis[0])

    n_times = history.time.size
    first_cross = _first(crossed)
    first_out = (
        _first(_outside(history, boundary)) if boundary is not None
        else np.full(history.n, n_times)
    )
    first_lost = _first(~on_grid)

    thetas, phis, times = [], [], []
    n_crossed = n_out = n_left = n_considered = 0
    for p in range(history.n):
        if n_times == 0 or not on_grid[0, p]:
            continue
        n_considered += 1
        on_grid_exit = min(first_cross[p], first_out[p])
        if on_grid_exit <= first_lost[p] and on_grid_exit < n_times:
            k = int(on_grid_exit)
            if first_out[p] <= first_cross[p]:
                n_out += 1
            else:
                n_crossed += 1
        elif first_lost[p] < n_times:
            k = int(first_lost[p]) - 1  # its last position on the grid; >= 0 as on_grid[0, p]
            n_left += 1
        else:
            continue
        thetas.append(theta_all[k, p])
        phis.append(history.phi[k, p])
        times.append(history.time[k])

    theta = np.asarray(thetas, dtype=float)
    theta = np.mod(theta, 2 * np.pi)
    theta[theta > np.pi] -= 2 * np.pi
    return ExitResult(
        theta=theta,
        phi=np.mod(np.asarray(phis, dtype=float), 2 * np.pi),
        time=np.asarray(times, dtype=float),
        n_crossed=n_crossed, n_outside_boundary=n_out, n_left_grid=n_left,
        n_considered=n_considered,
    )
