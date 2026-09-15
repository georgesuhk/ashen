"""2D scan maps: one point per JOREK run.

x and y are each a named per-run scalar (ashen.quantities) -- eta, edge_q,
li, ... -- and a third quantity is encoded either as point colour (with a
colourbar) or as ring size. Under colour encoding, several datasets on one
axes are told apart by marker shape (MARKER_CYCLE); under size encoding,
colour is free again and datasets are told apart the way
plot_wetted_fraction_datasets already does, from DISCRETE_PALETTE.

draw_*/plot_* split, matching every other module in this package: draw_*
takes an existing ax and draws nothing but data (no colourbar, no legend);
plot_* owns the figure, every legend/colourbar artist, and the file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ashen.plotting import DEFAULT_DPI, MARKER_CYCLE, style
from ashen.plotting.colors import DISCRETE_PALETTE, PsiColorer, colorer as build_colorer

__all__ = [
    "COLOR_ENCODING",
    "ENCODINGS",
    "RingSizer",
    "SIZE_ENCODING",
    "ScanPoint",
    "colorer_for",
    "draw_scan_map",
    "plot_scan_map",
    "plot_scan_map_datasets",
    "sizer",
]

#: How the third quantity is encoded. "color" spends colour on the
#: quantity and distinguishes datasets by marker shape (MARKER_CYCLE);
#: "size" spends ring radius on it and distinguishes datasets by colour
#: (DISCRETE_PALETTE). Exactly one of the two, never both -- a figure
#: where colour AND size both vary with the same number is redundant, and
#: one where they vary with different numbers is unreadable.
COLOR_ENCODING = "color"
SIZE_ENCODING = "size"
ENCODINGS = (COLOR_ENCODING, SIZE_ENCODING)

#: Floor for a log-scale colour/size mapping -- a value at or below zero has
#: no log10, but a scan map must not crash on one; it is drawn as if it sat
#: at this floor rather than raising, since the alternative (dropping the
#: point) would silently shrink the figure caller-side of anything visible.
_LOG_FLOOR = 1e-300


def _log_safe(value: float) -> float:
    return math.log10(max(value, _LOG_FLOOR))


@dataclass(frozen=True)
class RingSizer:
    """Maps a value to a scatter marker area, over a fixed [vmin, vmax].

    The size counterpart of colors.PsiColorer, and built the same way:
    once per figure from the full set of values, so a ring of a given
    radius means the same number on every dataset drawn on the axes.
    Re-normalising per dataset would make the same physics a different
    size depending on which sibling scan happened to be overlaid.

    Area, not radius, is interpolated (matplotlib scatter's `s` is an area
    in points^2) -- a reader compares ring *diameters* by eye, so
    interpolating linearly in area is the choice that makes a 4x value
    look like a 2x ring rather than a 4x one. log_scale interpolates the
    value in log space first, for a quantity spanning decades
    (delta_b_over_b typically does) where linear mapping collapses every
    point but the largest into the same dot.
    """

    vmin: float
    vmax: float
    s_min: float = 20.0
    s_max: float = 400.0
    log_scale: bool = False

    def _fraction(self, value: float) -> float:
        vmin, vmax, value = self.vmin, self.vmax, value
        if self.log_scale:
            vmin, vmax, value = _log_safe(vmin), _log_safe(vmax), _log_safe(value)
        span = vmax - vmin
        if span <= 0:
            return 0.5
        return min(max((value - vmin) / span, 0.0), 1.0)

    def __call__(self, value: float) -> float:
        frac = self._fraction(value)
        return self.s_min + frac * (self.s_max - self.s_min)

    def legend_handles(
        self, *, n: int = 3, color: str = "0.35", label_fmt: str = "{:.3g}"
    ) -> tuple[list, list[str]]:
        """(handles, labels) for a size legend: n representative rings
        spanning [vmin, vmax], drawn as unfilled Line2D proxies.

        A colourbar is self-labelling; a size encoding is not -- without
        this the ring radii are decorative. Proxy artists rather than real
        points so the legend does not misrepresent a run that does not
        exist.
        """
        import numpy as np
        from matplotlib.lines import Line2D

        if self.log_scale and self.vmin > 0 and self.vmax > 0:
            values = np.geomspace(self.vmin, self.vmax, n)
        else:
            values = np.linspace(self.vmin, self.vmax, n)

        handles = [
            Line2D(
                [], [], marker="o", linestyle="none", markerfacecolor="none",
                markeredgecolor=color, markersize=(self(float(v)) ** 0.5),
            )
            for v in values
        ]
        labels = [label_fmt.format(v) for v in values]
        return handles, labels


def sizer(
    values: Sequence[float], *, s_min: float = 20.0, s_max: float = 400.0,
    log_scale: bool = False,
) -> RingSizer:
    """Build a RingSizer spanning `values` -- the size counterpart of
    colors.colorer, including its empty-input fallback."""
    vals = list(values)
    if not vals:
        return RingSizer(vmin=0.0, vmax=1.0, s_min=s_min, s_max=s_max, log_scale=log_scale)
    return RingSizer(vmin=min(vals), vmax=max(vals), s_min=s_min, s_max=s_max, log_scale=log_scale)


@dataclass(frozen=True)
class ScanPoint:
    """One run on a scan map.

    `c` is the third quantity -- the one encoded as colour or ring size.
    None is allowed only when the figure encodes nothing third (a plain 2D
    scatter); a figure that does encode one drops incomplete points
    upstream in cli/plot rather than drawing a point whose colour means
    "missing".

    `label` is the case's x_tick_label (or its name), drawn beside the
    point under annotate=True. Carried per point rather than as a parallel
    list so a dropped run cannot shift every subsequent label by one --
    the failure mode that makes an annotated scan map worse than an
    unannotated one.
    """

    x: float
    y: float
    c: float | None = None
    label: str = ""


def _color_values(points: Sequence[ScanPoint], *, log_c: bool) -> list[float]:
    """Each point's `c`, log10-transformed first if `log_c` -- the single
    place that transform happens, so a colorer built from this and the
    colours drawn through this can never disagree."""
    if log_c:
        return [_log_safe(p.c) for p in points]
    return [p.c for p in points]


def colorer_for(
    points: Sequence[ScanPoint], *, cmap: str = "viridis", log_c: bool = False
) -> PsiColorer:
    """A PsiColorer spanning `points`' third quantity, transformed for
    `log_c` exactly as :func:`_color_values` transforms the values drawn
    through it -- keep the two in sync; nothing else calls colorer(psi_n)
    with `psi_n` a non-psi quantity here except through this pair."""
    values = [v for v in _color_values(points, log_c=log_c) if v is not None]
    return build_colorer(values, cmap=cmap, discrete=False)


def draw_scan_map(
    ax,
    points: Sequence[ScanPoint],
    *,
    encode: str = COLOR_ENCODING,
    xlabel: str = "",
    ylabel: str = "",
    log_x: bool = False,
    log_y: bool = False,
    log_c: bool = False,
    marker: str = "o",
    color: str = "tab:blue",
    colorer: PsiColorer | None = None,
    sizer: RingSizer | None = None,
    point_size: float = 60.0,
    label: str | None = None,
    annotate: bool = False,
    annotate_offset: tuple[float, float] = (5.0, 4.0),
) -> None:
    """Scatter one dataset's runs onto ax: one point per run, x and y each
    a per-run scalar, a third scalar encoded per `encode`.

    One call draws one dataset, so a multi-dataset figure calls this once
    per dataset with a different `marker` (colour encoding) or `color`
    (size encoding) -- the same composition plot_wetted_fraction_datasets
    already uses. The shared `colorer`/`sizer` are passed in rather than
    derived from `points`: derived per call they would be normalised per
    dataset, and the same physics would draw a different colour or radius
    depending on which sibling scan it sat beside.

    Under COLOR_ENCODING a `colorer` is required and `color` is ignored;
    under SIZE_ENCODING a `sizer` is required and `colorer` is ignored.
    The one exception: when every point's `c` is None -- no third quantity
    was ever configured (ScanPoint's docstring), so there is nothing to
    build a colorer/sizer from -- `colorer`/`sizer` may be omitted and the
    points are drawn plain, at `point_size`/`color`, with no encoding at
    all. A colorer/sizer omitted while some points DO carry a `c` is still
    an error: that is a caller bug, not a missing-quantity figure.

    `log_c` must match whatever the shared colorer/sizer were themselves
    built with (colorer_for / sizer(..., log_scale=...)) -- it only
    controls how *this call's* values are transformed before being handed
    to them. Rings are unfilled (facecolors="none") so overlapping runs
    stay individually countable -- the failure mode of a filled size
    encoding is a large point swallowing its neighbours entirely.

    log_x/log_y default False, unlike draw_wetted_fraction_vs_x's log_x
    default of True: there the x-axis is always a scan parameter, here
    either axis can be any registered quantity and most of them (q95, li,
    edge_q) are O(1). The caller takes the default from the quantity
    itself (Quantity.log_scale).

    annotate labels each point with its ScanPoint.label, offset in points
    so the text never lands under its own marker. Off by default: a scan
    of twenty runs annotates into illegibility, and the case names here
    are long ("qa2.1_g2.3/eta1e-3_RE").

    Does not attach a colourbar or a legend -- those are figure-level
    artists, so they belong to the plot_* wrappers (same draw_*/plot_*
    split as every other module here).
    """
    if encode not in ENCODINGS:
        raise ValueError(f"encode must be one of {ENCODINGS}, got {encode!r}")

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    kwargs = {}
    if label:
        kwargs["label"] = label

    has_c = any(p.c is not None for p in points)

    if encode == SIZE_ENCODING:
        if sizer is None:
            if has_c:
                raise ValueError("SIZE_ENCODING needs a sizer")
            ax.scatter(
                xs, ys, s=point_size, marker=marker, color=color,
                edgecolors="0.25", linewidths=0.4, **kwargs,
            )
        else:
            sizes = [sizer(p.c) for p in points]
            ax.scatter(
                xs, ys, s=sizes, marker=marker, facecolors="none",
                edgecolors=color, linewidths=1.2, **kwargs,
            )
    else:
        if colorer is None:
            if has_c:
                raise ValueError("COLOR_ENCODING needs a colorer")
            ax.scatter(
                xs, ys, s=point_size, marker=marker, color=color,
                edgecolors="0.25", linewidths=0.4, **kwargs,
            )
        else:
            colors = [colorer(v) for v in _color_values(points, log_c=log_c)]
            ax.scatter(
                xs, ys, s=point_size, marker=marker, c=colors,
                edgecolors="0.25", linewidths=0.4, **kwargs,
            )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.4)

    if annotate:
        for p in points:
            ax.annotate(
                p.label, (p.x, p.y), textcoords="offset points",
                xytext=annotate_offset, fontsize=7,
            )


def plot_scan_map(
    points: Sequence[ScanPoint],
    out_path: Path | str,
    *,
    encode: str = COLOR_ENCODING,
    xlabel: str = "",
    ylabel: str = "",
    clabel: str = "",
    log_x: bool = False,
    log_y: bool = False,
    log_c: bool = False,
    cmap: str = "viridis",
    marker: str = "o",
    color: str = "tab:blue",
    annotate: bool = False,
    size_legend_n: int = 3,
    figsize: tuple[float, float] = (6.5, 4.5),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save one scan map for a single (flat) comparison.

    Owns the figure, the colourbar or size legend, and the file -- the
    file-owning counterpart to draw_scan_map. Builds the colorer/sizer
    here, from every point, for the reason draw_scan_map's docstring
    gives.

    log_c applies to the third quantity's own scale: under COLOR_ENCODING
    it log10-transforms the values before colouring (colorer_for/
    _color_values); under SIZE_ENCODING it is the RingSizer's log_scale.
    Separate from log_x/log_y because delta_b/B spans decades whichever
    axis it is on.

    If every point's `c` is None -- no `c_quantity` was ever configured --
    this draws a plain 2D scatter instead: no colorer/sizer is built and no
    colourbar/size legend is attached, since there is nothing to encode
    (see ScanPoint and draw_scan_map's docstrings).
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_c = any(p.c is not None for p in points)

    with style():
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        if not has_c:
            draw_scan_map(
                ax, points, encode=encode, xlabel=xlabel, ylabel=ylabel,
                log_x=log_x, log_y=log_y, marker=marker, color=color,
                point_size=60.0, annotate=annotate,
            )
        elif encode == SIZE_ENCODING:
            values = [p.c for p in points if p.c is not None]
            ring_sizer = sizer(values, log_scale=log_c)
            draw_scan_map(
                ax, points, encode=encode, xlabel=xlabel, ylabel=ylabel,
                log_x=log_x, log_y=log_y, marker=marker, color=color,
                sizer=ring_sizer, annotate=annotate,
            )
            handles, labels = ring_sizer.legend_handles(n=size_legend_n)
            if handles:
                ax.legend(handles, labels, title=clabel or None, loc="best", frameon=True)
        else:
            shared_colorer = colorer_for(points, cmap=cmap, log_c=log_c)
            draw_scan_map(
                ax, points, encode=encode, xlabel=xlabel, ylabel=ylabel,
                log_x=log_x, log_y=log_y, log_c=log_c, marker=marker,
                colorer=shared_colorer, annotate=annotate,
            )
            fig.colorbar(shared_colorer.scalar_mappable(), ax=ax, label=clabel)

        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_scan_map_datasets(
    series: Sequence[tuple[str, Sequence[ScanPoint]]],
    out_path: Path | str,
    *,
    encode: str = COLOR_ENCODING,
    xlabel: str = "",
    ylabel: str = "",
    clabel: str = "",
    log_x: bool = False,
    log_y: bool = False,
    log_c: bool = False,
    cmap: str = "viridis",
    colors: Sequence[str | None] | None = None,
    markers: Sequence[str | None] | None = None,
    annotate: bool = False,
    size_legend_n: int = 3,
    figsize: tuple[float, float] = (6.5, 4.5),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Overlay several named datasets on one scan map.

    `series` is [(label, points), ...], parallel to `colors` and
    `markers`; a None entry (or the list omitted) assigns from
    DISCRETE_PALETTE / MARKER_CYCLE by position, cycling -- the same
    convention plot_wetted_fraction_datasets already uses for colour.

    Which of the two actually distinguishes the datasets follows `encode`,
    and the unused one is ignored: under COLOR_ENCODING the palette is
    already spent on the colourbar, so a per-dataset colour would
    contradict it; under SIZE_ENCODING the markers all stay circular and
    colour tells datasets apart instead.

    The shared colorer/sizer are built from every dataset's points
    together (flattened), so the same physics draws the same colour/size
    regardless of which sibling dataset it is plotted beside.

    Under SIZE_ENCODING the axes carries TWO legends -- datasets and ring
    sizes. matplotlib replaces a legend on a second ax.legend() call, so
    the first is re-added with ax.add_artist before the second is made.

    As in plot_scan_map: if every point's `c` is None (no `c_quantity`
    configured), no colorer/sizer is built and no colourbar/size legend is
    attached -- datasets are still told apart per `encode` (marker or
    colour), there is simply nothing third to encode.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_points = [p for _, pts in series for p in pts]
    has_c = any(p.c is not None for p in all_points)

    with style():
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        if encode == SIZE_ENCODING:
            ring_sizer = None
            if has_c:
                values = [p.c for p in all_points if p.c is not None]
                ring_sizer = sizer(values, log_scale=log_c)
            for idx, (label, pts) in enumerate(series):
                color = colors[idx] if colors is not None and idx < len(colors) else None
                if color is None:
                    color = DISCRETE_PALETTE[idx % len(DISCRETE_PALETTE)]
                draw_scan_map(
                    ax, pts, encode=encode, xlabel=xlabel, ylabel=ylabel,
                    log_x=log_x, log_y=log_y, color=color, sizer=ring_sizer,
                    point_size=60.0, label=label, annotate=annotate,
                )
            dataset_legend = ax.legend(loc="upper left")
            ax.add_artist(dataset_legend)
            if ring_sizer is not None:
                handles, labels = ring_sizer.legend_handles(n=size_legend_n)
                if handles:
                    ax.legend(handles, labels, title=clabel or None, loc="best", frameon=True)
        else:
            shared_colorer = colorer_for(all_points, cmap=cmap, log_c=log_c) if has_c else None
            for idx, (label, pts) in enumerate(series):
                marker = markers[idx] if markers is not None and idx < len(markers) else None
                if marker is None:
                    marker = MARKER_CYCLE[idx % len(MARKER_CYCLE)]
                draw_scan_map(
                    ax, pts, encode=encode, xlabel=xlabel, ylabel=ylabel,
                    log_x=log_x, log_y=log_y, log_c=log_c, marker=marker,
                    colorer=shared_colorer, point_size=60.0, label=label,
                    annotate=annotate,
                )
            ax.legend(loc="upper left")
            if shared_colorer is not None:
                fig.colorbar(shared_colorer.scalar_mappable(), ax=ax, label=clabel)

        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
