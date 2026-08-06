"""ashen.plotting.four_modes -- one figure per variable, one line per (n, m)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.diagnostics.four_modes import GrowthFit
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


def test_lines_carry_markers_for_each_data_point(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi")
    for line in ax.lines:
        assert line.get_marker() != "None"
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


# --- rational_series overlay -----------------------------------------------------


def test_rational_series_adds_a_dashed_line_per_mode(series):
    rational = {
        # n=0,m=1 has no rational surface (m/0) and must be skipped even
        # though an entry is present; only n=1,m=0 gets an overlay.
        ("Psi", 0, 1): np.array([1.5, 2.5, 3.5]),
        ("Psi", 1, 0): np.array([0.6, 0.5, 0.4]),
    }
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", rational_series=rational)
    # 2 solid domain-max lines + 1 dashed rational-surface line (n=1,m=0 only).
    assert len(ax.lines) == 3
    plt.close(fig)


def test_n_zero_mode_has_no_rational_overlay(series):
    """n=0,m=1 has no q=m/n rational surface (m/0), so even if a caller
    somehow provides a value keyed on it, it must not get a dashed line."""
    rational = {("Psi", 0, 1): np.array([0.6, 0.5, 0.4])}
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", rational_series=rational)
    assert len(ax.lines) == 2  # 2 solid lines, no dashed overlay
    plt.close(fig)


def test_missing_rational_entry_draws_no_extra_line(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", rational_series={})
    assert len(ax.lines) == 2
    plt.close(fig)


def test_rational_overlay_uses_dashed_linestyle(series):
    rational = {("Psi", 1, 0): np.array([0.6, 0.5, 0.4])}
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", rational_series=rational)
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(dashed) == 1
    plt.close(fig)


def test_plot_mode_amplitudes_accepts_rational_series(series, tmp_path):
    rational = {("Psi", 0, 1): np.array([1.5, 2.5, 3.5])}
    out = plot_mode_amplitudes(
        [100, 200, 300], series, "Psi", tmp_path / "Psi_modes.png", rational_series=rational,
    )
    assert out.is_file()


# --- growth_fits: legend annotation ------------------------------------------------


def test_growth_fit_is_appended_to_the_legend_label(series):
    growth_fits = {("Psi", 0, 1): GrowthFit(gamma=1.23e5, intercept=0.0, n_points=10)}
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", growth_fits=growth_fits)
    labels = {line.get_label() for line in ax.lines}
    assert any("1.23e+05" in label for label in labels)
    plt.close(fig)


def test_growth_fit_does_not_add_extra_lines(series):
    growth_fits = {("Psi", 0, 1): GrowthFit(gamma=1.0, intercept=0.0, n_points=10)}
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", growth_fits=growth_fits)
    assert len(ax.lines) == 2  # still just the two solid domain-max lines
    plt.close(fig)


def test_missing_growth_fit_entry_leaves_plain_label(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", growth_fits={})
    labels = {line.get_label() for line in ax.lines}
    assert labels == {"n=0, m=1", "n=1, m=0"}
    plt.close(fig)


def test_plot_mode_amplitudes_accepts_growth_fits(series, tmp_path):
    growth_fits = {("Psi", 0, 1): GrowthFit(gamma=1.0, intercept=0.0, n_points=10)}
    out = plot_mode_amplitudes(
        [100, 200, 300], series, "Psi", tmp_path / "Psi_modes.png", growth_fits=growth_fits,
    )
    assert out.is_file()


# --- ylabel / label_suffix overrides -----------------------------------------------


def test_default_ylabel_and_labels_are_unchanged(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi")
    assert ax.get_ylabel() == "max |Psi|"
    labels = {line.get_label() for line in ax.lines}
    assert labels == {"n=0, m=1", "n=1, m=0"}
    plt.close(fig)


def test_ylabel_override_replaces_the_default(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(ax, [100, 200, 300], series, variable="Psi", ylabel="|Psi| @ rational surface")
    assert ax.get_ylabel() == "|Psi| @ rational surface"
    plt.close(fig)


def test_label_suffix_is_appended_to_every_mode_label(series):
    fig, ax = plt.subplots()
    draw_mode_amplitudes(
        ax, [100, 200, 300], series, variable="Psi", label_suffix=" @ rational surface"
    )
    labels = {line.get_label() for line in ax.lines}
    assert labels == {"n=0, m=1 @ rational surface", "n=1, m=0 @ rational surface"}
    plt.close(fig)


def test_label_suffix_comes_before_the_growth_rate_suffix(series):
    growth_fits = {("Psi", 0, 1): GrowthFit(gamma=1.0, intercept=0.0, n_points=10)}
    fig, ax = plt.subplots()
    draw_mode_amplitudes(
        ax, [100, 200, 300], series, variable="Psi",
        label_suffix=" @ rational surface", growth_fits=growth_fits,
    )
    labels = {line.get_label() for line in ax.lines}
    assert "n=0, m=1 @ rational surface (\N{GREEK SMALL LETTER GAMMA}=1 /s)" in labels
    plt.close(fig)
