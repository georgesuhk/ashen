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
    "LossMap",
    "ParticleHistory",
    "exit_angles",
    "exits_from_history",
    "loss_map",
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
    #: Rows at the end of the file left out because not every dataset had
    #: them: the trace was cut off while writing (read_particle_diag).
    dropped_rows: int = 0

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


    def select(self, particles: np.ndarray) -> "ParticleHistory":
        """Only the particles where the boolean mask `particles` is True."""
        return replace(
            self, psi_n=self.psi_n[:, particles], R=self.R[:, particles],
            Z=self.Z[:, particles], phi=self.phi[:, particles], lost=self.lost[:, particles],
            theta=None if self.theta is None else self.theta[:, particles],
        )

    def starting_within(self, psi_n_min: float, psi_n_max: float) -> "ParticleHistory":
        """Only the particles whose psi_n at the first diagnostics time is
        in [psi_n_min, psi_n_max] -- where they started, whatever they did
        afterwards."""
        if not self.time.size:
            return self
        start = self.psi_n[0]
        return self.select((start >= psi_n_min) & (start <= psi_n_max))


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


def _time_first(arrays: dict[str, np.ndarray], n_times: int, where: str) -> dict[str, np.ndarray]:
    """Each dataset as (n_rows, n_particles). Fortran writes (n_particles,
    n_times); h5py sees it (n_times, n_particles); a transposed layout is
    accepted too, when its time axis has exactly n_times. In the usual
    layout the rows may be within one of n_times: a trace killed while it
    was writing a row leaves the datasets one row apart, as they are
    extended one after another and the time last."""
    for name, data in arrays.items():
        if data.ndim != 2:
            raise ParticleFileError(
                f"{where}: {name} has shape {data.shape}, expected (n_times, n_particles)"
            )
    off = [max(abs(data.shape[axis] - n_times) for data in arrays.values()) for axis in (0, 1)]
    # An exact fit either way round first; then the layout the programs
    # write, give or take the cut-off row.
    if off[0] == 0:
        axis = 0
    elif off[1] == 0:
        axis = 1
    elif off[0] <= 1:
        axis = 0
    else:
        shapes = {name: data.shape for name, data in arrays.items()}
        raise ParticleFileError(
            f"{where}: {shapes} do not have one row per diagnostics time in t "
            f"({n_times} times)"
        )
    arrays = {name: data if axis == 0 else data.T for name, data in arrays.items()}
    if len({data.shape[1] for data in arrays.values()}) != 1:
        shapes = {name: data.shape for name, data in arrays.items()}
        raise ParticleFileError(f"{where}: datasets differ in their number of particles: {shapes}")
    return arrays


def read_particle_diag(path: Path | str) -> ParticleHistory:
    """Read a write_particle_diagnostics file (e.g. ptrace_diag.h5).

    A trace that stopped early -- still running, out of time, killed -- is
    read as far as it got: the rows every dataset has. One that was cut off
    while writing a row has that row in some datasets only; it is left out
    and counted in dropped_rows.
    """
    h5py = require_h5py("reading particle diagnostics", ParticleFileError)
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise ParticleFileError(
            f"{path}: cannot open -- {exc} (if its trace was killed while writing "
            "this file, it cannot be recovered: retrace)"
        ) from exc

    with f:
        groups = f.get("groups")
        if groups is None or not len(groups):
            raise ParticleFileError(f"{path}: no particle groups under /groups")
        times: list[np.ndarray] = []
        columns: dict[str, list[np.ndarray]] = {name: [] for name in (*_REQUIRED, "theta")}
        has_theta = True
        for name in sorted(groups):
            group = groups[name]
            where = f"{path}: group {name!r}"
            if "t" not in group:
                raise ParticleFileError(f"{where} has no 't' -- not a particle diagnostics file")
            missing = [d for d in _REQUIRED if d not in group]
            if missing:
                raise ParticleFileError(
                    f"{where} has no {missing} -- the program must write psi_n, R, Z, "
                    "phi and lost (write_particle_diagnostics' `only`)"
                )
            t = np.asarray(group["t"], dtype=float).reshape(-1)
            names = [*_REQUIRED, *(["theta"] if "theta" in group else [])]
            arrays = _time_first({d: np.asarray(group[d]) for d in names}, t.size, where)
            times.append(t)
            for d in _REQUIRED:
                columns[d].append(arrays[d])
            if "theta" in arrays and has_theta:
                columns["theta"].append(arrays["theta"])
            else:
                has_theta = False

    used = [d for d in (*_REQUIRED, "theta") if d != "theta" or has_theta]
    lengths = [t.size for t in times] + [a.shape[0] for d in used for a in columns[d]]
    n_rows = min(lengths)
    time = times[0][:n_rows]
    for index, t in enumerate(times):
        if not np.allclose(t[:n_rows], time):
            raise ParticleFileError(
                f"{path}: group {index + 1}'s times differ from the first group's"
            )

    def joined(d: str, dtype) -> np.ndarray:
        # Straight into dtype: one new array, not a joined copy and then another.
        parts = [a[:n_rows] for a in columns[d]]
        if len(parts) == 1:
            return parts[0].astype(dtype)
        return np.concatenate(parts, axis=1, dtype=dtype)

    return ParticleHistory(
        path=path, time=time,
        psi_n=joined("psi_n", float), R=joined("R", float), Z=joined("Z", float),
        phi=joined("phi", float), lost=joined("lost", int) > 0,
        theta=joined("theta", float) if has_theta else None,
        dropped_rows=max(lengths) - n_rows,
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
    rows, kinds = _exit_rows(history, psi_n=psi_n, boundary=boundary)
    cols = np.flatnonzero(rows >= 0)
    if history.theta is not None:
        theta_exit = history.theta[rows[cols], cols]
    else:
        # Only where each particle exits: the same numbers, not every row's.
        theta_exit = np.arctan2(history.Z[rows[cols], cols] - axis[1],
                                history.R[rows[cols], cols] - axis[0])
    theta = np.mod(theta_exit.astype(float), 2 * np.pi)
    theta[theta > np.pi] -= 2 * np.pi
    return ExitResult(
        theta=theta,
        phi=np.mod(history.phi[rows[cols], cols].astype(float), 2 * np.pi),
        time=history.time[rows[cols]].astype(float),
        n_crossed=int(np.count_nonzero(kinds == _CROSSED)),
        n_outside_boundary=int(np.count_nonzero(kinds == _OUTSIDE)),
        n_left_grid=int(np.count_nonzero(kinds == _LEFT_GRID)),
        n_considered=int(np.count_nonzero(kinds != _NOT_CONSIDERED)),
    )


#: How a particle's trace ended, per _exit_rows.
_NOT_CONSIDERED, _STAYED, _CROSSED, _OUTSIDE, _LEFT_GRID = range(5)


def _exit_rows(
    history: ParticleHistory, *, psi_n: float, boundary: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per particle, the diagnostics row at which it left and how:

    - _CROSSED: first row with its psi_n past the threshold, on the grid;
    - _OUTSIDE: first row outside boundary (if given), on the grid, when
      that is no later than crossing;
    - _LEFT_GRID: lost from the grid before either -- the row is its last
      one on the grid;
    - _STAYED (row -1): none of these within the history;
    - _NOT_CONSIDERED (row -1): off the grid at the first row already.
    """
    n_times = history.time.size
    rows = np.full(history.n, -1, dtype=int)
    kinds = np.full(history.n, _NOT_CONSIDERED, dtype=int)
    if n_times == 0:
        return rows, kinds
    on_grid = ~history.lost
    first_cross = _first(on_grid & (history.psi_n > psi_n))
    first_out = (
        _first(_outside(history, boundary)) if boundary is not None
        else np.full(history.n, n_times)
    )
    first_lost = _first(~on_grid)
    for p in range(history.n):
        if not on_grid[0, p]:
            continue
        kinds[p] = _STAYED
        on_grid_exit = min(first_cross[p], first_out[p])
        if on_grid_exit <= first_lost[p] and on_grid_exit < n_times:
            rows[p] = int(on_grid_exit)
            kinds[p] = _OUTSIDE if first_out[p] <= first_cross[p] else _CROSSED
        elif first_lost[p] < n_times:
            rows[p] = int(first_lost[p]) - 1  # its last row on the grid; >= 0 as on_grid[0, p]
            kinds[p] = _LEFT_GRID
    return rows, kinds


@dataclass(frozen=True)
class LossMap:
    """What fraction of the particles that started at each psi_n have left
    by each time."""

    #: Times [s], (n_times,).
    time: np.ndarray
    #: Edges of the bins in starting psi_n, (n_psi + 1,).
    psi_edges: np.ndarray
    #: (n_psi, n_times): of the particles that started in the bin, the
    #: fraction that had left by the time. nan where none started.
    fraction: np.ndarray
    #: (n_psi,): how many particles started in each bin.
    counts: np.ndarray
    #: How many of all the counted particles had left by the last time.
    n_lost: int

    @property
    def n_considered(self) -> int:
        return int(self.counts.sum())


def loss_map(
    history: ParticleHistory,
    *,
    psi_n: float,
    boundary: np.ndarray | None = None,
    n_psi: int = 40,
    psi_range: tuple[float, float] | None = None,
    max_times: int = 400,
) -> LossMap:
    """Bin the particles by the psi_n they started at and, for each bin,
    follow the fraction that has left -- by exit_angles' rule: past psi_n,
    outside boundary, or off the grid. A particle off the grid from the
    start is not counted, nor one that started outside psi_range (default:
    0 to the largest starting psi_n). The times are the diagnostics',
    thinned evenly to at most max_times.
    """
    rows, kinds = _exit_rows(history, psi_n=psi_n, boundary=boundary)
    counted = kinds != _NOT_CONSIDERED
    start = history.psi_n[0] if history.time.size else np.array([])
    if psi_range is None:
        top = float(np.nanmax(start[counted])) if counted.any() else 1.0
        psi_range = (0.0, top if top > 0 else 1.0)
    edges = np.linspace(psi_range[0], psi_range[1], n_psi + 1)
    # np.digitize puts the top edge in bin n_psi: fold it into the last bin.
    bins = np.clip(np.digitize(start, edges) - 1, None, n_psi - 1) if start.size else start
    counted &= (start >= edges[0]) & (start <= edges[-1])

    n_times = history.time.size
    picked = np.unique(np.linspace(0, n_times - 1, min(n_times, max_times)).round().astype(int)) \
        if n_times else np.array([], dtype=int)
    counts = np.zeros(n_psi, dtype=int)
    fraction = np.full((n_psi, picked.size), np.nan)
    for b in range(n_psi):
        members = counted & (bins == b)
        counts[b] = int(np.count_nonzero(members))
        if counts[b]:
            left = rows[members]
            left = left[left >= 0]
            # left by row r: its exit row is r or earlier
            fraction[b] = np.searchsorted(np.sort(left), picked, side="right") / counts[b]
    return LossMap(
        time=history.time[picked], psi_edges=edges, fraction=fraction, counts=counts,
        n_lost=int(np.count_nonzero(counted & (rows >= 0))),
    )
