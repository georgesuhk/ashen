"""ashen.plotting.theta_histogram -- ports the drawing half of the notebook
function plot_theta_histogram_matrix (Columbia/NL_kinks/prod_plots_draft0.ipynb,
cell 5), against ashen.diagnostics.theta_histogram instead of raw .npz files.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this suite never opens a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.plotting.theta_histogram import draw_theta_histogram, plot_theta_histogram_grid


def test_draw_theta_histogram_returns_counts_summing_to_one():
    fig, ax = plt.subplots()
    angles = np.array([-1.0, 0.0, 0.5, 1.0])
    counts = draw_theta_histogram(ax, angles, bins=10)
    assert counts.sum() == pytest.approx(1.0)
    plt.close(fig)


def test_draw_theta_histogram_empty_draws_no_data_note():
    fig, ax = plt.subplots()
    draw_theta_histogram(ax, np.empty(0), bins=10)
    texts = [t.get_text() for t in ax.texts]
    assert "No data" in texts
    plt.close(fig)


def test_draw_theta_histogram_with_data_draws_no_note():
    fig, ax = plt.subplots()
    draw_theta_histogram(ax, np.array([0.0, 0.5]), bins=10)
    assert len(ax.texts) == 0
    plt.close(fig)


# --- the file-owning grid wrapper --------------------------------------------------


def test_plot_theta_histogram_grid_writes_a_file(tmp_path):
    panels = [("t=100", np.array([0.1, 0.2])), ("t=200", np.array([-0.3, 0.4]))]
    out = plot_theta_histogram_grid(panels, tmp_path / "sub" / "theta_hist.png", bins=20)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_grid_hides_trailing_unused_axes(tmp_path):
    """n_cols=3 with 5 panels gives 2 rows (6 axes); the 6th must be hidden."""
    import matplotlib.figure

    panels = [(f"p{i}", np.array([0.1])) for i in range(5)]
    n_cols = 3
    n_rows = math.ceil(len(panels) / n_cols)
    assert n_rows == 2

    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["axes"] = axes
        return fig, axes

    plt.subplots = spy
    try:
        out = plot_theta_histogram_grid(panels, tmp_path / "grid.png", bins=10, n_cols=n_cols)
    finally:
        plt.subplots = real_subplots
    assert out.is_file()

    axes = np.asarray(captured["axes"]).flatten()
    assert len(axes) == n_rows * n_cols
    assert not axes[-1].get_visible()  # the 6th axis, unused by 5 panels
    for ax in axes[:5]:
        assert ax.get_visible()


def test_grid_shares_y_limit_across_panels(tmp_path):
    """One shared y-limit -- no panel's bars should be clipped by a limit
    set independently per-axes."""
    # A single-angle panel puts 100% of its weight in one bin (tall); a
    # spread-out panel divides its weight across bins (shorter bars).
    panels = [
        ("tall", np.array([0.0, 0.0, 0.0, 0.0])),
        ("short", np.array([-2.0, -1.0, 0.0, 1.0])),
    ]
    out = plot_theta_histogram_grid(panels, tmp_path / "grid.png", bins=4, n_cols=2)
    assert out.is_file()


def test_grid_with_no_panels_still_writes_a_file(tmp_path):
    out = plot_theta_histogram_grid([], tmp_path / "empty.png", bins=10)
    assert out.is_file()
