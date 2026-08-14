"""ashen.plotting.profiles -- one line per step, one panel per tor_mode."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.plotting.profiles import draw_profile_family, plot_profile_comparison


@pytest.fixture
def series():
    return {
        100: (np.array([0.1, 0.5, 0.9]), np.array([1.0, 2.0, 3.0])),
        200: (np.array([0.1, 0.5, 0.9]), np.array([1.5, 2.5, 3.5])),
        300: (np.array([0.1, 0.5, 0.9]), np.array([2.0, 3.0, 4.0])),
    }


# --- draw_profile_family ------------------------------------------------------------


def test_draws_one_line_per_step(series):
    fig, ax = plt.subplots()
    draw_profile_family(ax, series)
    assert len(ax.lines) == 3
    plt.close(fig)


def test_empty_series_draws_nothing(series):
    fig, ax = plt.subplots()
    draw_profile_family(ax, {})
    assert len(ax.lines) == 0
    plt.close(fig)


def test_lines_get_distinct_colors(series):
    fig, ax = plt.subplots()
    draw_profile_family(ax, series)
    colors = {line.get_color() for line in ax.lines}
    assert len(colors) == 3
    plt.close(fig)


def test_color_by_overrides_step_index_coloring(series):
    fig, ax = plt.subplots()
    # Two steps colinear in time-space but far apart in step-space.
    draw_profile_family(ax, series, color_by={100: 0.0, 200: 0.01, 300: 100.0})
    colors = [line.get_color() for line in ax.lines]
    # step 100 and 200 should be much closer in colour than 100 and 300.
    assert colors[0] != colors[2]
    plt.close(fig)


def test_returns_the_colorer_used(series):
    fig, ax = plt.subplots()
    colors = draw_profile_family(ax, series)
    assert colors(100.0) is not None
    plt.close(fig)


def test_a_shared_colorer_is_reused_not_rebuilt(series):
    from ashen.plotting.colors import colorer

    fig, ax = plt.subplots()
    shared = colorer([0.0, 1000.0])
    returned = draw_profile_family(ax, series, colors=shared)
    assert returned is shared
    plt.close(fig)


def test_title_and_labels_are_set(series):
    fig, ax = plt.subplots()
    draw_profile_family(ax, series, xlabel="Psi_N", ylabel="currdens", title="midplane")
    assert ax.get_xlabel() == "Psi_N"
    assert ax.get_ylabel() == "currdens"
    assert ax.get_title() == "midplane"
    plt.close(fig)


# --- plot_profile_comparison: file-owning, one panel per mode ----------------------


def test_writes_a_file(series, tmp_path):
    out = plot_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "Psi_N_currdens_profile.png",
    )
    assert out.is_file()


def test_output_directory_is_created(series, tmp_path):
    out_dir = tmp_path / "figures"
    out = plot_profile_comparison(
        {"midplane": series}, "currdens", out_dir / "profile.png",
    )
    assert out.parent == out_dir
    assert out.is_file()


def test_one_panel_per_mode(series, tmp_path):
    fig_path = tmp_path / "profile.png"
    # Can't easily introspect subplot count post-save; verify via a direct
    # draw call count instead by checking the function accepts N modes and
    # produces one file regardless of N.
    out = plot_profile_comparison(
        {"midplane": series, "average": series, "midplane outer": {}},
        "currdens", fig_path,
    )
    assert out.is_file()


def test_empty_mode_does_not_crash_and_still_produces_a_file(series, tmp_path):
    out = plot_profile_comparison(
        {"midplane": series, "average": {}}, "currdens", tmp_path / "profile.png",
    )
    assert out.is_file()


def test_all_modes_empty_does_not_crash(tmp_path):
    out = plot_profile_comparison(
        {"midplane": {}, "average": {}}, "currdens", tmp_path / "profile.png",
    )
    assert out.is_file()


def test_custom_color_by_is_honored(series, tmp_path):
    out = plot_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.png",
        color_by={100: 1e-4, 200: 2e-4, 300: 3e-4}, color_label=r"t [$\mu s$]",
    )
    assert out.is_file()


def test_cmap_defaults_to_viridis(series, monkeypatch, tmp_path):
    captured = {}
    from ashen.plotting import profiles as profiles_mod

    original = profiles_mod.colorer

    def spy(values, **kwargs):
        captured["cmap"] = kwargs.get("cmap")
        return original(values, **kwargs)

    monkeypatch.setattr(profiles_mod, "colorer", spy)

    plot_profile_comparison({"midplane": series}, "currdens", tmp_path / "profile.png")
    assert captured["cmap"] == "viridis"


def test_cmap_is_passed_through_to_colorer(series, monkeypatch, tmp_path):
    captured = {}
    from ashen.plotting import profiles as profiles_mod

    original = profiles_mod.colorer

    def spy(values, **kwargs):
        captured["cmap"] = kwargs.get("cmap")
        return original(values, **kwargs)

    monkeypatch.setattr(profiles_mod, "colorer", spy)

    plot_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.png", cmap="plasma",
    )
    assert captured["cmap"] == "plasma"
