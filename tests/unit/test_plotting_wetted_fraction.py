"""ashen.plotting.wetted_fraction -- one scalar per case plotted against a
scan parameter (e.g. wetted fraction vs. eta), ported from the core of the
notebook's eta_plot (Columbia/NL_kinks/prod_plots_draft0.ipynb, cell 0)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: this suite never opens a display

import matplotlib.pyplot as plt

from ashen.plotting.colors import DISCRETE_PALETTE
from ashen.plotting.wetted_fraction import (
    draw_wetted_fraction_vs_x,
    plot_wetted_fraction_datasets,
    plot_wetted_fraction_vs_x,
)


def test_draw_plots_one_line_with_markers():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4, 1e-5], [0.1, 0.2, 0.3])
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert line.get_marker() == "o"
    plt.close(fig)


def test_draw_defaults_to_log_x_scale():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4, 1e-5], [0.1, 0.2, 0.3])
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_draw_log_x_can_be_disabled():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1, 2, 3], [0.1, 0.2, 0.3], log_x=False)
    assert ax.get_xscale() == "linear"
    plt.close(fig)


def test_draw_sets_labels_when_given():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(
        ax, [1e-3, 1e-4], [0.1, 0.2], xlabel=r"$\eta$", ylabel="Wetted fraction",
    )
    assert ax.get_xlabel() == r"$\eta$"
    assert ax.get_ylabel() == "Wetted fraction"
    plt.close(fig)


def test_draw_omits_labels_when_not_given():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4], [0.1, 0.2])
    assert ax.get_xlabel() == ""
    plt.close(fig)


# --- the file-owning wrapper -----------------------------------------------------


def test_plot_wetted_fraction_vs_x_writes_a_file(tmp_path):
    out = plot_wetted_fraction_vs_x(
        [1e-3, 1e-4, 1e-5], [0.1, 0.2, 0.3], tmp_path / "sub" / "wetted.png",
    )
    assert out.is_file()
    assert out.stat().st_size > 0


# --- label/color: distinguishing more than one series on the same axes -----------


def test_draw_default_color_is_unchanged():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4], [0.1, 0.2])
    assert ax.lines[0].get_color() == "tab:blue"
    plt.close(fig)


def test_draw_custom_color():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4], [0.1, 0.2], color="tab:red")
    assert ax.lines[0].get_color() == "tab:red"
    plt.close(fig)


def test_draw_without_label_has_no_legend_entry():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4], [0.1, 0.2])
    assert ax.lines[0].get_label().startswith("_")  # matplotlib's "excluded" convention
    plt.close(fig)


def test_draw_with_label_is_used_by_the_legend():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4], [0.1, 0.2], label="normal")
    assert ax.lines[0].get_label() == "normal"
    plt.close(fig)


def test_two_draw_calls_on_one_ax_produce_two_lines():
    fig, ax = plt.subplots()
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4], [0.1, 0.2], label="normal", color="tab:blue")
    draw_wetted_fraction_vs_x(ax, [1e-3, 1e-4], [0.15, 0.25], label="rho19", color="tab:red")
    assert [l.get_label() for l in ax.lines] == ["normal", "rho19"]
    assert [l.get_color() for l in ax.lines] == ["tab:blue", "tab:red"]
    plt.close(fig)


# --- plot_wetted_fraction_datasets: the multi-series file-owning wrapper ---------


def test_plot_wetted_fraction_datasets_writes_a_file(tmp_path):
    series = [
        ("normal", [1e-3, 1e-4], [0.1, 0.2]),
        ("rho19", [1e-3, 1e-4], [0.15, 0.25]),
    ]
    out = plot_wetted_fraction_datasets(series, tmp_path / "sub" / "wetted.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_wetted_fraction_datasets_draws_one_line_per_series(tmp_path, monkeypatch):
    captured_axes = []
    original_subplots = plt.subplots

    def spy_subplots(*a, **k):
        fig, ax = original_subplots(*a, **k)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    series = [
        ("normal", [1e-3, 1e-4], [0.1, 0.2]),
        ("rho19", [1e-3, 1e-4], [0.15, 0.25]),
        ("third", [1e-3, 1e-4], [0.05, 0.35]),
    ]
    plot_wetted_fraction_datasets(series, tmp_path / "wetted.png")

    ax = captured_axes[-1]
    assert [l.get_label() for l in ax.lines] == ["normal", "rho19", "third"]


def test_plot_wetted_fraction_datasets_assigns_colors_from_the_discrete_palette(
    tmp_path, monkeypatch
):
    captured_axes = []
    original_subplots = plt.subplots

    def spy_subplots(*a, **k):
        fig, ax = original_subplots(*a, **k)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    series = [("a", [1e-3], [0.1]), ("b", [1e-3], [0.2])]
    plot_wetted_fraction_datasets(series, tmp_path / "wetted.png")

    ax = captured_axes[-1]
    assert ax.lines[0].get_color() == DISCRETE_PALETTE[0]
    assert ax.lines[1].get_color() == DISCRETE_PALETTE[1]


def test_plot_wetted_fraction_datasets_respects_explicit_colors(tmp_path, monkeypatch):
    captured_axes = []
    original_subplots = plt.subplots

    def spy_subplots(*a, **k):
        fig, ax = original_subplots(*a, **k)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    series = [("a", [1e-3], [0.1]), ("b", [1e-3], [0.2])]
    plot_wetted_fraction_datasets(
        series, tmp_path / "wetted.png", colors=["tab:green", None],
    )

    ax = captured_axes[-1]
    assert ax.lines[0].get_color() == "tab:green"
    assert ax.lines[1].get_color() == DISCRETE_PALETTE[1]  # None -> palette fallback
