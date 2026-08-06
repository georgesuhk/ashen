"""ashen.plotting.wetted_fraction -- one scalar per case plotted against a
scan parameter (e.g. wetted fraction vs. eta), ported from the core of the
notebook's eta_plot (Columbia/NL_kinks/prod_plots_draft0.ipynb, cell 0)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: this suite never opens a display

import matplotlib.pyplot as plt

from ashen.plotting.wetted_fraction import draw_wetted_fraction_vs_x, plot_wetted_fraction_vs_x


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
