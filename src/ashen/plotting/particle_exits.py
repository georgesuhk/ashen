"""Histograms of where particles leave the plasma: theta and phi side by
side (data from ashen.diagnostics.particle_exits).

Weighted to the fraction of exiting particles per bin, like
ashen.plotting.theta_histogram, so a case with more particles reads on the
same scale. :func:`animate_exit_histograms` shows them filling in over time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ashen.diagnostics.particle_exits import ExitResult
from ashen.plotting import DEFAULT_DPI, style

__all__ = ["animate_exit_histograms", "exit_caption", "plot_exit_histograms"]

_THETA_TICKS = ([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
_PHI_TICKS = ([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
              ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])


def exit_caption(result: ExitResult, *, psi_n: float, boundary: bool = False) -> str:
    """How many particles the histograms stand for, and how each exited."""
    where = f"psi_n = {psi_n:g}"
    if boundary:
        where += " or the plasma boundary before extension"
    text = f"{result.n_exited} of {result.n_considered} particles exit past {where}"
    if boundary:
        text += (
            f" ({result.n_outside_boundary} at the boundary, "
            f"{result.n_crossed} at psi_n)"
        )
    if result.n_left_grid:
        text += (
            f" ({result.n_left_grid} left the grid first -- "
            "drawn at their last position on it)"
        )
    return text


def _draw(ax, angles: np.ndarray, *, bins: int, lo: float, hi: float, ticks) -> None:
    if angles.size == 0:
        ax.text(0.5, 0.5, "No exits", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.hist(
            angles, bins=bins, range=(lo, hi), weights=np.full(angles.shape, 1.0 / angles.size),
            color="tab:blue", edgecolor="tab:blue", alpha=0.7, linewidth=0.5,
        )
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks[0])
    ax.set_xticklabels(ticks[1])
    ax.tick_params(direction="in", which="both", top=True, right=True)


def plot_exit_histograms(
    result: ExitResult,
    out_path: Path | str,
    *,
    bins: int = 72,
    caption: str | None = None,
    figsize: tuple[float, float] = (8.0, 3.2),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw and save the theta (left) and phi (right) exit histograms."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with style():
        fig, (ax_theta, ax_phi) = plt.subplots(
            1, 2, figsize=figsize, layout="constrained", sharey=True,
        )
        _draw(ax_theta, result.theta, bins=bins, lo=-np.pi, hi=np.pi, ticks=_THETA_TICKS)
        _draw(ax_phi, result.phi, bins=bins, lo=0.0, hi=2 * np.pi, ticks=_PHI_TICKS)
        ax_theta.set_xlabel(r"poloidal angle $\theta$ at exit")
        ax_phi.set_xlabel(r"toroidal angle $\phi$ at exit")
        ax_theta.set_ylabel("fraction of exits")
        if caption:
            fig.suptitle(caption, fontsize=9)
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def animate_exit_histograms(
    result: ExitResult,
    out_path: Path | str,
    *,
    t_range: tuple[float, float],
    bins: int = 72,
    n_frames: int = 40,
    fps: int = 8,
    caption: str | None = None,
    figsize: tuple[float, float] = (8.0, 3.6),
    dpi: int = DEFAULT_DPI,
) -> Path | None:
    """An animated GIF of the theta and phi exit histograms filling in:
    frame k counts the exits up to its time, n_frames times spread evenly
    over t_range (the trace's start and end). Bars are fractions of *all*
    the exits, on the final histogram's y scale -- drawn as a grey outline
    -- so they grow into it. None (nothing written) with no exits or an
    empty t_range. Same Pillow writer as the other GIFs."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    t0, t1 = t_range
    if result.n_exited == 0 or not t1 > t0:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_times = np.linspace(t0, t1, max(2, n_frames))
    weight = 1.0 / result.n_exited
    panels = (
        (result.theta, -np.pi, np.pi, _THETA_TICKS, r"poloidal angle $\theta$ at exit"),
        (result.phi, 0.0, 2 * np.pi, _PHI_TICKS, r"toroidal angle $\phi$ at exit"),
    )

    with style():
        fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained", sharey=True)
        bars = []
        y_max = 0.0
        for ax, (angles, lo, hi, ticks, label) in zip(axes, panels):
            edges = np.linspace(lo, hi, bins + 1)
            final, _ = np.histogram(angles, bins=edges, weights=np.full(angles.shape, weight))
            y_max = max(y_max, final.max())
            ax.stairs(final, edges, color="0.6", linewidth=0.8)
            bars.append((angles, edges, ax.bar(
                edges[:-1], np.zeros(bins), width=np.diff(edges), align="edge",
                color="tab:blue", edgecolor="tab:blue", alpha=0.7, linewidth=0.5,
            )))
            ax.set_xlim(lo, hi)
            ax.set_xticks(ticks[0])
            ax.set_xticklabels(ticks[1])
            ax.set_xlabel(label)
            ax.tick_params(direction="in", which="both", top=True, right=True)
        axes[0].set_ylim(0, y_max * 1.1)
        axes[0].set_ylabel("fraction of exits")
        title = fig.suptitle("", fontsize=9)

        def _update(index):
            t = frame_times[index]
            done = result.time <= t + abs(t) * 1e-6
            for angles, edges, container in bars:
                counts, _ = np.histogram(angles[done], bins=edges)
                for bar, height in zip(container, counts * weight):
                    bar.set_height(height)
            text = (f"t = {t * 1e3:.5g} ms (+{(t - t0) * 1e6:.3g} $\\mu$s): "
                    f"{int(done.sum())} of {result.n_exited} exits so far")
            title.set_text(f"{caption}\n{text}" if caption else text)
            return []

        anim = animation.FuncAnimation(fig, _update, frames=len(frame_times), blit=False)
        anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path
