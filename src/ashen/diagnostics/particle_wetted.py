"""How much of the wall traced particles wet: where each escaping particle
hits a wall surface -- the plasma boundary before extend_bnd -- and how
widely those hits spread, poloidally, toroidally and in total.

**Where on the wall.** A hit is placed by its arc length l along the
boundary curve (from the outboard midplane, R max, counter-clockwise in
R-Z) and its toroidal angle phi -- not by geometric theta, whose equal
steps cover unequal lengths of wall. It comes from the particle's
diagnostics history (ashen.diagnostics.particle_exits.ParticleHistory):
the crossing between its last row inside the boundary and its first row
outside is interpolated -- where the straight R-Z segment between them
meets the boundary, phi interpolated in step -- so it is finer than
diag_step. A particle that leaves the grid before any row shows it outside
the boundary is placed at the wall point nearest its last position on the
grid, at that row's phi.

**How widely.** Each measure is a participation ratio -- the
(sum n)^2 / sum(n^2 / A) of the hit counts n over cells of area A -- the
area the hits would cover if spread evenly at their actual mean density;
the P/q_peak of a heat load, but using every cell rather than the one peak
cell, so much less noisy with few markers. As fractions of the wall:

- f_pol: over poloidal bins (all phi together), each of area 2 pi int R dl;
- f_tor: over toroidal bins (all l together), each an equal share of the wall;
- f_tot: over (l, phi) cells, of area R dl dphi;
- s = f_tot / (f_pol * f_tor): exactly 1 for a separable footprint (the
  same poloidal pattern at every phi); well below 1 for a helical one,
  which can reach every poloidal and toroidal angle while wetting little.

Error bars are bootstrap standard deviations: the hits resampled with
replacement and each measure recomputed.

Pure data: no matplotlib here (see ashen.plotting.particle_wetted).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ashen.diagnostics.particle_exits import ParticleHistory
from ashen.diagnostics.particles import inside_polygon

__all__ = [
    "ParticleHits", "Wall", "WallHits", "WettedResult", "particle_hits", "wall_hits",
    "wetted_area",
]

TWO_PI = 2 * np.pi


@dataclass(frozen=True)
class Wall:
    """A closed R-Z curve, counter-clockwise from its outboard midplane
    (R max) point, with arc length along it."""

    R: np.ndarray
    Z: np.ndarray
    #: Arc length at each vertex; cum_l[-1] is the closing edge's end, = length.
    cum_l: np.ndarray
    #: int R dl from the first vertex to each vertex, likewise.
    cum_Rl: np.ndarray

    @classmethod
    def from_points(cls, points: np.ndarray) -> "Wall":
        pts = np.asarray(points, dtype=float)
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        R, Z = pts[:, 0], pts[:, 1]
        # Counter-clockwise in (R, Z): positive signed area.
        if np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z) < 0:
            R, Z = R[::-1], Z[::-1]
        start = int(np.argmax(R))
        R, Z = np.roll(R, -start), np.roll(Z, -start)
        R_next, Z_next = np.roll(R, -1), np.roll(Z, -1)
        seg = np.hypot(R_next - R, Z_next - Z)
        cum_l = np.concatenate([[0.0], np.cumsum(seg)])
        cum_Rl = np.concatenate([[0.0], np.cumsum(seg * (R + R_next) / 2)])
        return cls(R=R, Z=Z, cum_l=cum_l, cum_Rl=cum_Rl)

    @property
    def length(self) -> float:
        return float(self.cum_l[-1])

    @property
    def area(self) -> float:
        """The surface's area, 2 pi int R dl."""
        return float(TWO_PI * self.cum_Rl[-1])

    def int_R_dl(self, l: np.ndarray) -> np.ndarray:
        """int R dl from l = 0 to each l (exact: R is linear along an edge)."""
        l = np.asarray(l, dtype=float)
        edge = np.clip(np.searchsorted(self.cum_l, l, side="right") - 1, 0, self.R.size - 1)
        u = l - self.cum_l[edge]
        seg = self.cum_l[edge + 1] - self.cum_l[edge]
        R0 = self.R[edge]
        R1 = np.roll(self.R, -1)[edge]
        slope = np.divide(R1 - R0, seg, out=np.zeros_like(seg), where=seg > 0)
        return self.cum_Rl[edge] + R0 * u + slope * u**2 / 2

    def crossing(self, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float] | None:
        """Where the segment p0 -> p1 (R, Z) first meets the wall: (fraction
        along the segment, arc length there), or None if it doesn't."""
        a = np.column_stack([self.R, self.Z])
        b = np.roll(a, -1, axis=0)
        d = p1 - p0
        e = b - a
        denom = d[0] * e[:, 1] - d[1] * e[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((a[:, 0] - p0[0]) * e[:, 1] - (a[:, 1] - p0[1]) * e[:, 0]) / denom
            u = ((a[:, 0] - p0[0]) * d[1] - (a[:, 1] - p0[1]) * d[0]) / denom
        ok = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if not ok.any():
            return None
        edge = int(np.flatnonzero(ok)[np.argmin(t[ok])])
        seg = self.cum_l[edge + 1] - self.cum_l[edge]
        return float(t[edge]), float(self.cum_l[edge] + u[edge] * seg)

    def nearest(self, p: np.ndarray) -> float:
        """Arc length of the wall point nearest p (R, Z)."""
        a = np.column_stack([self.R, self.Z])
        e = np.roll(a, -1, axis=0) - a
        seg2 = np.einsum("ij,ij->i", e, e)
        u = np.clip(np.divide(np.einsum("ij,ij->i", p - a, e), seg2,
                              out=np.zeros_like(seg2), where=seg2 > 0), 0, 1)
        dist = np.hypot(*(a + u[:, None] * e - p).T)
        edge = int(np.argmin(dist))
        return float(self.cum_l[edge] + u[edge] * (self.cum_l[edge + 1] - self.cum_l[edge]))


@dataclass(frozen=True)
class WallHits:
    """Where each escaping particle hit the wall."""

    #: Arc length along the wall [m], in [0, wall.length).
    l: np.ndarray
    #: Toroidal angle [rad], in [0, 2 pi).
    phi: np.ndarray
    #: Of the hits: interpolated crossings of the wall...
    n_crossed: int
    #: ...and particles that left the grid first, placed at the nearest wall point.
    n_left_grid: int
    #: Particles inside the wall and on the grid at the first diagnostics time.
    n_considered: int

    @property
    def n(self) -> int:
        return int(self.l.size)


@dataclass(frozen=True)
class ParticleHits:
    """Every particle's first hit on the wall, one entry per particle, so a
    selection of particles can be taken afterwards (and the whole kept in
    ashen.diagnostics.wetted_cache)."""

    #: Inside the wall and on the grid at the first diagnostics time.
    considered: np.ndarray
    #: Of those, the ones that hit the wall...
    hit: np.ndarray
    #: ...and of the hits, the ones that left the grid first.
    left_grid: np.ndarray
    #: Where each hit: arc length [m], in [0, wall.length), and toroidal
    #: angle [rad], not yet wrapped; nan for a particle that didn't hit.
    l: np.ndarray
    phi: np.ndarray

    @property
    def n(self) -> int:
        return int(self.considered.size)

    def wall_hits(self, particles: np.ndarray | None = None) -> WallHits:
        """The hits of the particles where the boolean mask is True (all
        without one), in particle order."""
        keep = np.ones(self.n, dtype=bool) if particles is None else np.asarray(particles, bool)
        hit = self.hit & keep
        return WallHits(
            l=self.l[hit], phi=np.mod(self.phi[hit], TWO_PI),
            n_crossed=int(np.count_nonzero(hit & ~self.left_grid)),
            n_left_grid=int(np.count_nonzero(hit & self.left_grid)),
            n_considered=int(np.count_nonzero(self.considered & keep)),
        )


def particle_hits(history: ParticleHistory, wall: Wall) -> ParticleHits:
    """Each particle's first hit on wall (see the module docstring)."""
    n = history.n
    considered = np.zeros(n, dtype=bool)
    hit = np.zeros(n, dtype=bool)
    left_grid = np.zeros(n, dtype=bool)
    ls = np.full(n, np.nan)
    phis = np.full(n, np.nan)
    if history.time.size == 0:
        return ParticleHits(considered, hit, left_grid, ls, phis)
    polygon = np.column_stack([wall.R, wall.Z])
    on_grid = ~history.lost
    inside = on_grid & inside_polygon(
        history.R.ravel(), history.Z.ravel(), polygon
    ).reshape(history.R.shape)
    for p in range(n):
        if not inside[0, p]:
            continue
        considered[p] = True
        gone = ~inside[:, p]
        if not gone.any():
            continue
        k = int(np.argmax(gone))
        p0 = np.array([history.R[k - 1, p], history.Z[k - 1, p]])
        phi0 = history.phi[k - 1, p]
        if on_grid[k, p]:
            p1 = np.array([history.R[k, p], history.Z[k, p]])
            crossing = wall.crossing(p0, p1)
            frac, l = crossing if crossing is not None else (1.0, wall.nearest(p1))
            phi = phi0 + frac * (history.phi[k, p] - phi0)
        else:
            l, phi = wall.nearest(p0), phi0
            left_grid[p] = True
        hit[p] = True
        ls[p] = l % wall.length
        phis[p] = phi
    return ParticleHits(considered, hit, left_grid, ls, phis)


def wall_hits(history: ParticleHistory, wall: Wall) -> WallHits:
    """Each particle's first hit on wall (see the module docstring)."""
    return particle_hits(history, wall).wall_hits()


@dataclass(frozen=True)
class WettedResult:
    """How widely the hits spread over the wall, with bootstrap errors."""

    f_pol: float
    f_tor: float
    f_tot: float
    s: float
    #: f_tot times the wall's area [m^2].
    area: float
    f_pol_err: float
    f_tor_err: float
    f_tot_err: float
    s_err: float
    #: Hits per (l, phi) cell, (n_l, n_phi), and the edges.
    counts: np.ndarray
    l_edges: np.ndarray
    phi_edges: np.ndarray
    #: Each (l, phi) cell's area [m^2].
    cell_area: np.ndarray

    def as_dict(self) -> dict:
        return {
            key: float(getattr(self, key))
            for key in ("f_pol", "f_tor", "f_tot", "s", "area",
                        "f_pol_err", "f_tor_err", "f_tot_err", "s_err")
        }


def _participation(counts: np.ndarray, areas: np.ndarray) -> float:
    """(sum n)^2 / sum(n^2 / A): the area the hits would cover at their
    mean density. nan with no hits."""
    total = counts.sum()
    if total == 0:
        return float("nan")
    return float(total**2 / np.sum(counts**2 / areas))


def _fractions(counts: np.ndarray, cell_area: np.ndarray, wall_area: float) -> np.ndarray:
    pol = _participation(counts.sum(axis=1), cell_area.sum(axis=1)) / wall_area
    tor = _participation(counts.sum(axis=0), cell_area.sum(axis=0)) / wall_area
    tot = _participation(counts.ravel(), cell_area.ravel()) / wall_area
    return np.array([pol, tor, tot, tot / (pol * tor)])


def wetted_area(
    hits: WallHits,
    wall: Wall,
    *,
    n_l: int = 36,
    n_phi: int = 36,
    n_boot: int = 200,
    seed: int = 0,
) -> WettedResult:
    """f_pol, f_tor, f_tot and s over n_l x n_phi equal (l, phi) cells, with
    bootstrap errors from n_boot resamplings of the hits."""
    l_edges = np.linspace(0.0, wall.length, n_l + 1)
    phi_edges = np.linspace(0.0, TWO_PI, n_phi + 1)
    strip = np.diff(wall.int_R_dl(l_edges))            # int R dl per l bin
    cell_area = np.outer(strip, np.diff(phi_edges))     # R dl dphi
    counts, _, _ = np.histogram2d(hits.l, hits.phi, bins=[l_edges, phi_edges])
    values = _fractions(counts, cell_area, wall.area)

    errors = np.full(4, np.nan)
    if hits.n > 1 and n_boot > 1:
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n_boot):
            pick = rng.integers(0, hits.n, hits.n)
            c, _, _ = np.histogram2d(hits.l[pick], hits.phi[pick], bins=[l_edges, phi_edges])
            samples.append(_fractions(c, cell_area, wall.area))
        errors = np.nanstd(np.asarray(samples), axis=0)

    return WettedResult(
        f_pol=values[0], f_tor=values[1], f_tot=values[2], s=values[3],
        area=values[2] * wall.area,
        f_pol_err=errors[0], f_tor_err=errors[1], f_tot_err=errors[2], s_err=errors[3],
        counts=counts, l_edges=l_edges, phi_edges=phi_edges, cell_area=cell_area,
    )
