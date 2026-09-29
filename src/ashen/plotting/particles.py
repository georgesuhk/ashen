"""Particle positions on the R-Z plane, and R-Z figures built from layers.

:func:`draw_particles` draws one snapshot onto an ax, like
:func:`ashen.plotting.poincare.draw_poincare` does for punctures. The
figure functions don't know about either: an :class:`RZPanel` is a title
and a list of *layers* -- callables that each draw onto the panel's ax, in
order. A particle figure's panels are ``[particles]``; overlaying a
Poincare plot is ``[poincare, particles]``, with nothing special-cased.

Particles are projected onto R-Z from every toroidal angle phi, whereas a
Poincare plot is one phi plane -- keep that in mind when overlaying.

:func:`particle_panels` builds those layers: an optional Poincare layer
(:class:`PoincareOverlay`, one per snapshot) at the bottom, drawn fainter so
the particles read over it; the particles; and optionally the plasma
boundary a run had before extend_bnd, with particles outside it marked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ashen.diagnostics.particles import (
    BoundaryExits,
    ParticleSnapshot,
    exits_from_snapshots,
    freeze_exited,
)
from ashen.diagnostics.poincare_cache import LineKey, LineRecord
from ashen.plotting import DEFAULT_DPI, style
from ashen.plotting.poincare import draw_poincare

__all__ = [
    "PoincareOverlay",
    "RZPanel",
    "animate_rz_panels",
    "draw_boundary",
    "draw_particles",
    "particle_caption",
    "particle_panels",
    "particle_point_size",
    "plot_rz_panels",
    "snapshot_label",
]

#: A layer draws onto one ax. Layers are applied in list order.
Layer = Callable[[object], None]

#: Particle colours: current positions, where they started, lost ones.
PARTICLE_COLOR = "black"
REFERENCE_COLOR = "0.75"
LOST_COLOR = "tab:red"
#: The pre-extension plasma boundary, and particles outside it: magenta is
#: in neither viridis (the Poincare colours) nor the particle colours.
BOUNDARY_COLOR = "magenta"
#: Poincare punctures under particles: a little faint, so the particles
#: stay the thing you see.
POINCARE_ALPHA = 0.5


@dataclass(frozen=True)
class RZPanel:
    """One R-Z panel (or animation frame): a title and the layers drawn
    onto it, first to last."""

    title: str
    layers: Sequence[Layer]


@dataclass(frozen=True)
class PoincareOverlay:
    """The Poincare punctures drawn under one snapshot: which restart step
    they were traced at, and the lines."""

    step: int
    records: Mapping[LineKey, LineRecord]
    #: Marker area, as poincare_point_size.
    s: float = 0.1
    alpha: float = POINCARE_ALPHA

    def draw(self, ax) -> None:
        draw_poincare(ax, self.records, s=self.s, alpha=self.alpha)


def particle_point_size(n: int) -> float:
    """A marker area that keeps a handful of particles visible and
    thousands from merging into a blob."""
    if n <= 100:
        return 12.0
    if n <= 5000:
        return 4.0
    return 1.0


def _duration(seconds: float) -> str:
    """A short time span, in microseconds below a millisecond, else ms."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} " + r"$\mu$s"
    return f"{seconds * 1e3:.3g} ms"


def snapshot_label(
    snapshot: ParticleSnapshot, start: float | None = None, *, n_outside: int = 0,
) -> str:
    """Panel title: time in ms -- and, given the first snapshot's time,
    the time since it, which is what changes between panels -- and how many
    particles are lost so far, and how many have left the plasma boundary
    (n_outside)."""
    label = f"t = {snapshot.time * 1e3:.5g} ms"
    if start is not None and snapshot.time != start:
        label += f" (+{_duration(snapshot.time - start)})"
    if snapshot.n_lost:
        label += f", {snapshot.n_lost}/{snapshot.n} lost"
    if n_outside:
        label += f", {n_outside} left boundary"
    return label


def draw_boundary(ax, boundary: np.ndarray) -> None:
    """A closed (R, Z) outline, over the punctures and particles."""
    closed = np.vstack([boundary, boundary[:1]])
    ax.plot(
        closed[:, 0], closed[:, 1], color=BOUNDARY_COLOR, linewidth=1.2,
        linestyle="--", zorder=5,
    )


def draw_particles(
    ax,
    snapshot: ParticleSnapshot,
    *,
    reference: ParticleSnapshot | None = None,
    outside: np.ndarray | None = None,
    s: float | None = None,
    alpha: float = 0.8,
) -> None:
    """Scatter one snapshot's particle positions onto ax, R vs Z.

    reference, if given (usually the first snapshot), is drawn behind in
    light grey, so each panel shows where the distribution started as well
    as where it is. Lost particles are drawn as red crosses where they were
    lost; those flagged in outside (left the boundary) as magenta
    crosses. Drawn above anything already on ax (zorder), so a Poincare
    layer drawn first stays underneath.
    """
    if s is None:
        s = particle_point_size(snapshot.n)
    if reference is not None and reference is not snapshot:
        ax.scatter(
            reference.R, reference.Z, s=s, color=REFERENCE_COLOR,
            alpha=0.6, linewidths=0, zorder=2,
        )
    outside = np.zeros(snapshot.n, dtype=bool) if outside is None else outside
    inside = ~snapshot.lost & ~outside
    ax.scatter(
        snapshot.R[inside], snapshot.Z[inside], s=s, color=PARTICLE_COLOR,
        alpha=alpha, linewidths=0, zorder=3,
    )
    for mask, color in ((snapshot.lost, LOST_COLOR), (outside, BOUNDARY_COLOR)):
        if mask.any():
            ax.scatter(
                snapshot.R[mask], snapshot.Z[mask], s=max(s, 4.0) * 3,
                marker="x", color=color, linewidths=0.8, zorder=4,
            )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ [m]")
    ax.set_ylabel(r"$Z$ [m]")


def _common_limits(views) -> tuple[tuple[float, float], tuple[float, float]]:
    """The union of (xlim, ylim) views, padded, so all panels (or frames)
    share one view and movement between them reads directly."""
    xs, ys = zip(*views)
    x0, x1 = min(x[0] for x in xs), max(x[1] for x in xs)
    y0, y1 = min(y[0] for y in ys), max(y[1] for y in ys)
    pad_x = (x1 - x0) * 0.03 or 0.05
    pad_y = (y1 - y0) * 0.03 or 0.05
    return (x0 - pad_x, x1 + pad_x), (y0 - pad_y, y1 + pad_y)


def _draw_panel(ax, panel: RZPanel) -> None:
    for layer in panel.layers:
        layer(ax)
    ax.set_title(panel.title, fontsize=10)


def plot_rz_panels(
    panels: Sequence[RZPanel],
    out_path: Path | str,
    *,
    n_cols: int = 4,
    figsize_per_panel: tuple[float, float] = (3.2, 3.6),
    caption: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save a grid of R-Z panels, row-major in panels order, all
    on the same R and Z limits. caption, if given, goes above the grid --
    e.g. what each layer's colours mean."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = max(1, min(n_cols, len(panels)))
    n_rows = math.ceil(len(panels) / n_cols)

    with style():
        fig, axes = plt.subplots(
            n_rows, n_cols, squeeze=False, layout="constrained",
            figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        )
        axes = axes.flatten()
        used = axes[:len(panels)]
        for ax, panel in zip(used, panels):
            _draw_panel(ax, panel)
        for ax in axes[len(panels):]:
            ax.set_visible(False)

        xlim, ylim = _common_limits([(ax.get_xlim(), ax.get_ylim()) for ax in used])
        for idx, ax in enumerate(used):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if idx % n_cols:
                ax.set_ylabel("")
            if idx < len(panels) - n_cols:
                ax.set_xlabel("")
        if caption:
            fig.suptitle(caption, fontsize=9)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def animate_rz_panels(
    frames: Sequence[RZPanel],
    out_path: Path | str,
    *,
    figsize: tuple[float, float] = (5.0, 5.6),
    fps: int = 2,
    caption: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path | None:
    """Write the panels as frames of an animated GIF, all on the same R and
    Z limits. None (nothing written) for fewer than two frames -- a
    one-frame animation isn't one. Same Pillow writer as the profile GIFs.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    if len(frames) < 2:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with style():
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        if caption:
            fig.suptitle(caption, fontsize=9)
        # Draw every frame once to find the view that holds all of them.
        views = []
        for frame in frames:
            ax.cla()
            _draw_panel(ax, frame)
            views.append((ax.get_xlim(), ax.get_ylim()))
        xlim, ylim = _common_limits(views)

        def _update(index):
            ax.cla()
            _draw_panel(ax, frames[index])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            return []

        anim = animation.FuncAnimation(fig, _update, frames=len(frames), blit=False)
        anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path


def particle_panels(
    snapshots: Sequence[ParticleSnapshot],
    *,
    boundary: np.ndarray | None = None,
    exits: BoundaryExits | None = None,
    poincare: Sequence[PoincareOverlay | None] | None = None,
) -> list[RZPanel]:
    """One panel per snapshot, each with the first snapshot behind it.

    boundary, an (N, 2) array of (R, Z), is drawn on every panel. A particle
    that has left it is no longer tracked: from then on it is drawn where it
    first left, as a cross. exits says when and where that was (e.g.
    exits_from_history, at the diagnostics' resolution); without it, it is
    judged from the snapshots themselves. poincare, one entry per snapshot
    (None for none), is drawn underneath, and its step added to the title.
    """
    if not snapshots:
        return []
    first = snapshots[0]
    poincare = list(poincare) if poincare is not None else [None] * len(snapshots)
    if len(poincare) != len(snapshots):
        raise ValueError(f"{len(poincare)} Poincare overlays for {len(snapshots)} snapshots")

    if boundary is not None and exits is None:
        exits = exits_from_snapshots(snapshots, boundary)

    panels = []
    for snapshot, overlay in zip(snapshots, poincare):
        outside = None
        if exits is not None:
            snapshot, outside = freeze_exited(snapshot, exits)
        layers: list[Layer] = []
        if overlay is not None:
            layers.append(overlay.draw)
        layers.append(
            lambda ax, snap=snapshot, out=outside: draw_particles(
                ax, snap, reference=first, outside=out,
            )
        )
        if boundary is not None:
            layers.append(lambda ax: draw_boundary(ax, boundary))
        title = snapshot_label(
            snapshot, start=first.time,
            n_outside=int(np.count_nonzero(outside)) if outside is not None else 0,
        )
        if overlay is not None:
            title += f"\nPoincare: step {overlay.step}"
        panels.append(RZPanel(title=title, layers=layers))
    return panels


def particle_caption(
    snapshots: Sequence[ParticleSnapshot], *, boundary: bool = False, poincare: bool = False,
) -> str:
    """What the colours in particle_panels mean, a line per layer."""
    first = snapshots[0]
    lines = [
        f"black: particles now; grey: at t = {first.time * 1e3:.5g} ms; "
        "red x: lost (where it left the grid); all toroidal angles projected onto R-Z"
    ]
    if boundary:
        lines.append("magenta dashes: plasma boundary before extension; magenta x: "
                     "left it, drawn where it first did and no longer tracked")
    if poincare:
        lines.append("faint dots: Poincare punctures at the case's phi_start, coloured by psi_n")
    return "\n".join(lines)
