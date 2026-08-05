"""ashen.plotting.four_modes -- one figure per variable, one line per (n, m)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.plotting.four_modes import draw_mode_amplitudes, plot_mode_amplitudes


@pytest.fixture
def series():
    return {
        ("Psi", 0, 1): np.array([1.0, 2.0, 3.0]),
        ("Psi", 1, 0): np.array([0.5, 0.4, 0.3]),
        ("u", 0, 1): np.array([10.0, 20.0, 30.0]),
    }


def test_draws_one_line_per_mode_of_the_requested_variable(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi")
    assert len(ax.lines) == 2  # only Psi's two modes, not u's
    plt.close(fig)


def test_other_variables_are_not_drawn(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="u")
    assert len(ax.lines) == 1
    plt.close(fig)


def test_modes_get_distinct_colors(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi")
    colors = {line.get_color() for line in ax.lines}
    assert len(colors) == 2
    plt.close(fig)


def test_legend_labels_encode_n_and_m(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi")
    labels = {line.get_label() for line in ax.lines}
    assert labels == {"n=0, m=1", "n=1, m=0"}
    plt.close(fig)


def test_log_scale_is_the_default(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi")
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_linear_scale_opt_out(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", log=False)
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_ylabel_names_the_variable(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="u")
    assert "u" in ax.get_ylabel()
    plt.close(fig)


def test_no_modes_for_variable_draws_nothing(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="rho")
    assert len(ax.lines) == 0
    plt.close(fig)


# --- the file-owning wrapper -------------------------------------------------------------------


def test_plot_mode_amplitudes_writes_a_file(series, tmp_path):
    out = plot_mode_amplitudes([100, 200, 300], series, "Psi", tmp_path / "Psi_modes.png")
    assert out.is_file()


def test_output_directory_is_created(series, tmp_path):
    out_dir = tmp_path / "four_dir"
    out = plot_mode_amplitudes([100, 200, 300], series, "Psi", out_dir / "Psi_modes.png")
    assert out.parent == out_dir
    assert out.is_file()
