"""Reading particle positions from JOREK particle files (part_restart*.h5).

A particle file is what JOREK's ``write_simulation_hdf5``
(``particles/mod_particle_io.f90``) writes: the simulation time in SI
seconds at ``/time``, and one HDF5 group per particle group under
``/groups/<id>/``, each holding at least

- ``x``     -- position (R [m], Z [m], phi [rad]) per particle,
- ``i_elm`` -- the grid element the particle is in; <= 0 once it is lost,
- ``weight`` -- how many physical particles it stands for.

A ptrace folder (ashen.ptracing) holds one file per snapshot the program
wrote: e.g. re_gc_current_density_initialisation's
``part_restart<time>.h5`` every ``write_step`` plus ``part_restart.h5`` at
the end, or ptrace_gc's ``part_restart_s<step>_t<time>.h5``, whose
``<step>`` :func:`named_step` reads back. :func:`find_snapshots` collects
them in time order.

Pure data: no matplotlib here (see ashen.plotting.particles).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import re

import numpy as np

from ashen.diagnostics.hdf5 import require_h5py

__all__ = [
    "BoundaryExits",
    "PARTICLE_FILE_GLOB",
    "ParticleFileError",
    "ParticleSnapshot",
    "exits_from_snapshots",
    "find_snapshots",
    "freeze_exited",
    "inside_polygon",
    "named_step",
    "read_snapshot",
]

#: Every particle file a JOREK particle program writes into its folder.
PARTICLE_FILE_GLOB = "part_restart*.h5"

#: ptrace_gc's snapshot names: part_restart_s<step>_t<time>.h5.
_NAMED_STEP = re.compile(r"^part_restart_s(\d+)_t")


class ParticleFileError(RuntimeError):
    """A particle file that is missing, unreadable, or not laid out as
    write_simulation_hdf5 writes it."""


@dataclass(frozen=True)
class ParticleSnapshot:
    """Every particle in one particle file, all groups concatenated."""

    path: Path
    #: Simulation time [s].
    time: float
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    #: True for a particle that has left the grid (i_elm <= 0). Its position
    #: is where it was when it was lost.
    lost: np.ndarray
    weight: np.ndarray

    @property
    def n(self) -> int:
        return int(self.R.size)

    @property
    def n_lost(self) -> int:
        return int(np.count_nonzero(self.lost))


def _positions(x: np.ndarray, n_elm: int, where: str) -> np.ndarray:
    """``x`` as (n_particles, 3).

    Fortran writes x_arr(3, n); read back through HDF5's row-major view
    that is (n, 3). Transposed data (3, n) is accepted too, told apart by
    i_elm's length, so a file written some other way still reads.
    """
    if x.ndim == 2 and x.shape == (n_elm, 3):
        return x
    if x.ndim == 2 and x.shape == (3, n_elm):
        return x.T
    raise ParticleFileError(
        f"{where}: x has shape {x.shape}, expected ({n_elm}, 3) -- one "
        "(R, Z, phi) per particle, matching i_elm"
    )


def read_snapshot(path: Path | str) -> ParticleSnapshot:
    """Read every particle's position from one particle file."""
    h5py = require_h5py("reading particle files", ParticleFileError)
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise ParticleFileError(f"{path}: cannot open -- {exc}") from exc

    with f:
        if "time" not in f:
            raise ParticleFileError(f"{path}: no /time -- not a JOREK particle file")
        time = float(np.asarray(f["time"]).reshape(-1)[0])
        groups = f.get("groups")
        if groups is None or not len(groups):
            raise ParticleFileError(f"{path}: no particle groups under /groups")

        R, Z, phi, lost, weight = [], [], [], [], []
        for name in sorted(groups):
            group = groups[name]
            where = f"{path}: group {name!r}"
            for dataset in ("x", "i_elm"):
                if dataset not in group:
                    raise ParticleFileError(f"{where} has no {dataset!r}")
            i_elm = np.asarray(group["i_elm"]).reshape(-1)
            x = _positions(np.asarray(group["x"]), i_elm.size, where)
            R.append(x[:, 0])
            Z.append(x[:, 1])
            phi.append(x[:, 2])
            lost.append(i_elm <= 0)
            weight.append(
                np.asarray(group["weight"], dtype=float).reshape(-1)
                if "weight" in group else np.ones(i_elm.size)
            )

    return ParticleSnapshot(
        path=path, time=time,
        R=np.concatenate(R).astype(float), Z=np.concatenate(Z).astype(float),
        phi=np.concatenate(phi).astype(float), lost=np.concatenate(lost),
        weight=np.concatenate(weight),
    )


def find_snapshots(folder: Path | str) -> list[ParticleSnapshot]:
    """Every particle file in `folder`, read and sorted by simulation time.

    Two files at the same time -- e.g. a snapshot written on the final step
    alongside part_restart.h5 -- are the same state; the first by name is
    kept.
    """
    folder = Path(folder)
    snapshots: dict[float, ParticleSnapshot] = {}
    for path in sorted(folder.glob(PARTICLE_FILE_GLOB)):
        snapshot = read_snapshot(path)
        snapshots.setdefault(snapshot.time, snapshot)
    return [snapshots[t] for t in sorted(snapshots)]


def named_step(path: Path | str) -> int | None:
    """The JOREK step ptrace_gc put in a snapshot's name -- that of the
    restart closest in time -- or None for a name without one."""
    match = _NAMED_STEP.match(Path(path).name)
    return int(match.group(1)) if match else None


def inside_polygon(R: np.ndarray, Z: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Whether each point (R, Z) is inside the closed polygon, an (N, 2)
    array of (R, Z) vertices (first and last need not repeat). Even-odd ray
    casting, so points exactly on an edge may go either way.

    An edge from a_z to b_z is crossed by the points with min <= Z < max of
    the two -- exactly (a_z > Z) != (b_z > Z). Between two neighbouring
    vertex heights every point crosses the same edges, so each point looks
    up its band of Z (a search among the vertex heights) and is tested
    against that band's few edges only -- not against every edge. The
    crossing R is the same expression on the same numbers as testing every
    edge, so the answer is too. Done in chunks of points, to bound memory.
    (Only edges with finite ends count; a NaN end never crosses anyway.)"""
    R, Z = np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    shape = np.broadcast_shapes(R.shape, Z.shape)
    R, Z = np.broadcast_to(R, shape).ravel(), np.broadcast_to(Z, shape).ravel()
    poly = np.asarray(polygon, dtype=float)
    a_r, a_z = poly[:, 0], poly[:, 1]
    b_r, b_z = np.roll(a_r, -1), np.roll(a_z, -1)
    inside = np.zeros(R.size, dtype=bool)
    # Edges that something can cross: not flat, ends finite.
    edges = np.flatnonzero(np.isfinite(a_z) & np.isfinite(b_z) & (a_z != b_z))
    if not edges.size:
        return inside.reshape(shape)
    a_r, a_z, b_r, b_z = a_r[edges], a_z[edges], b_r[edges], b_z[edges]
    d_r, d_z = b_r - a_r, b_z - a_z
    # Bands [levels[k], levels[k+1]) between the vertex heights, and the
    # edges spanning each: those with lo <= levels[k] and levels[k+1] <= hi.
    levels = np.unique(np.concatenate([a_z, b_z]))
    lo = np.searchsorted(levels, np.minimum(a_z, b_z))
    hi = np.searchsorted(levels, np.maximum(a_z, b_z))
    n_bands = levels.size - 1
    per_band = np.zeros(n_bands, dtype=int)
    for k0, k1 in zip(lo, hi):
        per_band[k0:k1] += 1
    band_edges = np.full((n_bands, int(per_band.max())), -1, dtype=int)
    filled = np.zeros(n_bands, dtype=int)
    for e, (k0, k1) in enumerate(zip(lo, hi)):
        band_edges[np.arange(k0, k1), filled[k0:k1]] = e
        filled[k0:k1] += 1

    for start in range(0, R.size, _INSIDE_CHUNK):
        r, z = R[start:start + _INSIDE_CHUNK], Z[start:start + _INSIDE_CHUNK]
        band = np.searchsorted(levels, z, "right") - 1   # NaN: past the last band
        in_band = np.flatnonzero((band >= 0) & (band < n_bands))
        r, z, band = r[in_band], z[in_band], band[in_band]
        parity = np.zeros(in_band.size, dtype=bool)
        for slot in range(band_edges.shape[1]):
            e = band_edges[band, slot]
            has = np.flatnonzero(e >= 0)
            e = e[has]
            r_cross = a_r[e] + (z[has] - a_z[e]) * d_r[e] / d_z[e]
            parity[has] ^= r[has] < r_cross
        inside[start + in_band] = parity
    return inside.reshape(shape)


#: Points inside_polygon tests at a time.
_INSIDE_CHUNK = 1 << 22


#: Relative tolerance on "exited by this snapshot's time": diagnostics
#: times are stored as 4-byte reals, snapshot times as 8-byte.
_TIME_RTOL = 1e-6


@dataclass(frozen=True)
class BoundaryExits:
    """When and where each particle first left a boundary while still on
    the grid. inf time (and nan position) for one that never did."""

    time: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    #: What the times were read from, for messages: e.g. "ptrace_diag.h5
    #: (every diag_step)" or "the snapshots".
    source: str = ""

    @property
    def n(self) -> int:
        return int(self.time.size)

    def exited_by(self, time: float) -> np.ndarray:
        """Which particles had left the boundary at `time`."""
        return self.time <= time + abs(time) * _TIME_RTOL


def exits_from_snapshots(
    snapshots: Sequence[ParticleSnapshot], boundary: np.ndarray,
) -> BoundaryExits:
    """Each particle's first snapshot outside boundary while on the grid --
    only as fine as the snapshots; exits_from_history (ashen.diagnostics.
    particle_exits) is finer where the program writes a diagnostics file.
    Every snapshot must hold the same particles, in the same order."""
    n = snapshots[0].n
    time = np.full(n, np.inf)
    R = np.full(n, np.nan)
    Z = np.full(n, np.nan)
    for snapshot in snapshots:
        if snapshot.n != n:
            raise ParticleFileError(
                f"{snapshot.path}: {snapshot.n} particles, but {snapshots[0].path} has {n}"
            )
        out = (~snapshot.lost & ~inside_polygon(snapshot.R, snapshot.Z, boundary)
               & np.isinf(time))
        time[out] = snapshot.time
        R[out] = snapshot.R[out]
        Z[out] = snapshot.Z[out]
    return BoundaryExits(time=time, R=R, Z=Z, source="the snapshots")


def freeze_exited(
    snapshot: ParticleSnapshot, exits: BoundaryExits,
) -> tuple[ParticleSnapshot, np.ndarray]:
    """The snapshot with every particle that had left the boundary by its
    time put back where it first left -- no longer tracked from there on --
    and not counted as lost, whatever happened to it later. Returns that
    snapshot and the mask of exited particles."""
    if exits.n != snapshot.n:
        raise ParticleFileError(
            f"{snapshot.path}: {snapshot.n} particles, but the boundary exits are for {exits.n}"
        )
    exited = exits.exited_by(snapshot.time)
    if not exited.any():
        return snapshot, exited
    return replace(
        snapshot,
        R=np.where(exited, exits.R, snapshot.R),
        Z=np.where(exited, exits.Z, snapshot.Z),
        lost=snapshot.lost & ~exited,
    ), exited
