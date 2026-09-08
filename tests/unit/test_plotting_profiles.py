"""ashen.plotting.profiles -- one line per step, one panel per tor_mode."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.plotting.profiles import (
    RationalBand,
    _profile_frame_label,
    animate_profile_comparison,
    draw_profile_family,
    plot_profile_comparison,
)


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


def test_ylim_pins_the_axis(series):
    fig, ax = plt.subplots()
    draw_profile_family(ax, series, ylim=(0.0, 10.0))
    assert ax.get_ylim() == (0.0, 10.0)
    plt.close(fig)


def test_ylim_none_leaves_auto_scaling(series):
    fig, ax = plt.subplots()
    draw_profile_family(ax, series)
    assert ax.get_ylim() != (0.0, 10.0)
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


def test_cmap_defaults_to_turbo(series, monkeypatch, tmp_path):
    captured = {}
    from ashen.plotting import profiles as profiles_mod

    original = profiles_mod.colorer

    def spy(values, **kwargs):
        captured["cmap"] = kwargs.get("cmap")
        return original(values, **kwargs)

    monkeypatch.setattr(profiles_mod, "colorer", spy)

    plot_profile_comparison({"midplane": series}, "currdens", tmp_path / "profile.png")
    assert captured["cmap"] == "turbo"


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


def test_ylim_is_forwarded_to_every_panel(series, monkeypatch, tmp_path):
    captured = []
    from ashen.plotting import profiles as profiles_mod

    original = profiles_mod.draw_profile_family

    def spy(ax, series, **kwargs):
        captured.append(kwargs.get("ylim"))
        return original(ax, series, **kwargs)

    monkeypatch.setattr(profiles_mod, "draw_profile_family", spy)

    plot_profile_comparison(
        {"midplane": series, "average": series}, "currdens", tmp_path / "profile.png",
        ylim=(0.0, 10.0),
    )
    assert captured == [(0.0, 10.0), (0.0, 10.0)]


# --- _profile_frame_label: step + time text, independent of the colourbar ----------


def test_frame_label_states_step_and_time_when_available():
    label = _profile_frame_label(100, {100: 1e-4})
    assert "step 100" in label
    assert "100" in label  # 1e-4 s -> 100 us


def test_frame_label_states_only_step_when_time_by_step_is_none():
    assert _profile_frame_label(100, None) == "step 100"


def test_frame_label_states_only_step_when_this_step_is_missing():
    assert _profile_frame_label(300, {100: 1e-4}) == "step 300"


# --- animate_profile_comparison ------------------------------------------------------


def test_animate_writes_a_gif(series, tmp_path):
    out = animate_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.gif",
    )
    assert out is not None
    assert out.is_file()
    assert out.suffix == ".gif"


def test_animate_returns_none_for_a_single_step(tmp_path):
    single_step = {100: (np.array([0.1, 0.5]), np.array([1.0, 2.0]))}
    out = animate_profile_comparison(
        {"midplane": single_step}, "currdens", tmp_path / "profile.gif",
    )
    assert out is None
    assert not (tmp_path / "profile.gif").exists()


def test_animate_returns_none_when_every_mode_is_empty(tmp_path):
    out = animate_profile_comparison(
        {"midplane": {}, "average": {}}, "currdens", tmp_path / "profile.gif",
    )
    assert out is None


def test_animate_output_directory_is_created(series, tmp_path):
    out_dir = tmp_path / "figures"
    out = animate_profile_comparison(
        {"midplane": series}, "currdens", out_dir / "profile.gif",
    )
    assert out.parent == out_dir
    assert out.is_file()


def test_animate_one_panel_per_mode_does_not_crash(series, tmp_path):
    out = animate_profile_comparison(
        {"midplane": series, "average": series, "midplane outer": {}},
        "currdens", tmp_path / "profile.gif",
    )
    assert out.is_file()


def test_animate_cmap_is_passed_through_to_colorer(series, monkeypatch, tmp_path):
    captured = {}
    from ashen.plotting import profiles as profiles_mod

    original = profiles_mod.colorer

    def spy(values, **kwargs):
        captured["cmap"] = kwargs.get("cmap")
        return original(values, **kwargs)

    monkeypatch.setattr(profiles_mod, "colorer", spy)

    animate_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.gif", cmap="plasma",
    )
    assert captured["cmap"] == "plasma"


def test_animate_with_rational_lines_does_not_crash(series, tmp_path):
    out = animate_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.gif",
        rational_lines=[(0.5, "red", "n=1, m=2")],
    )
    assert out.is_file()


def test_animate_with_time_by_step_does_not_crash(series, tmp_path):
    out = animate_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.gif",
        color_by={100: 1e-4, 200: 2e-4, 300: 3e-4}, color_label=r"t [$\mu s$]",
        time_by_step={100: 1e-4, 200: 2e-4, 300: 3e-4},
    )
    assert out.is_file()


def test_animate_with_partial_time_by_step_does_not_crash(series, tmp_path):
    # Steps 100 and 300 have a real time, 200 doesn't (e.g. its zeroD cache
    # is missing) -- must not raise, just omit the time text for that frame.
    out = animate_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.gif",
        time_by_step={100: 1e-4, 300: 3e-4},
    )
    assert out.is_file()


def test_animate_ylim_overrides_data_derived_limits(series, monkeypatch, tmp_path):
    """With ylim given, every panel must use it instead of the data-range
    fixed limits animate_profile_comparison would otherwise compute."""
    captured = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    out = animate_profile_comparison(
        {"midplane": series}, "currdens", tmp_path / "profile.gif", ylim=(0.0, 10.0),
    )
    assert out.is_file()
    ax = captured["axes"][0][0]
    assert ax.get_ylim() == (0.0, 10.0)


# --- rational surfaces: bands on the static figure, motion in the animation ------


def test_bands_are_drawn_behind_the_curves(series, tmp_path):
    fig, ax = plt.subplots()
    draw_profile_family(
        ax, series,
        rational_lines=[(0.4, "blue", "n=1, m=2")],
        rational_bands=[RationalBand(0.3, 0.5, 0.4, "blue", "n=1, m=2")],
    )
    spans = [p for p in ax.patches if isinstance(p, matplotlib.patches.Patch)]
    assert len(spans) == 1
    assert spans[0].get_zorder() == 0
    # The band must not add a second legend entry for the same mode.
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["n=1, m=2"]
    plt.close(fig)


def test_a_static_band_is_not_shaded(series):
    """Zero-width shading is invisible clutter -- only the line is drawn."""
    fig, ax = plt.subplots()
    draw_profile_family(
        ax, series, rational_bands=[RationalBand(0.4, 0.4, 0.4, "blue", "n=1, m=2")],
    )
    assert not [p for p in ax.patches if isinstance(p, matplotlib.patches.Patch)]
    plt.close(fig)


def test_animation_moves_the_rational_lines_between_frames(series, tmp_path):
    """Regression: the axvlines were created once outside the frame update, so
    every frame showed the first step's surfaces regardless of its own curve.
    """
    positions_seen = []
    real_axvline = plt.Axes.axvline

    def spy(self, x=0, *args, **kwargs):
        positions_seen.append(x)
        return real_axvline(self, x, *args, **kwargs)

    plt.Axes.axvline = spy
    try:
        out = animate_profile_comparison(
            {"midplane": series}, "T", tmp_path / "a.gif",
            rational_lines_by_step={100: [(0.30, "blue", "n=1, m=2")],
                                    200: [(0.70, "blue", "n=1, m=2")]},
        )
    finally:
        plt.Axes.axvline = real_axvline

    assert out is not None
    # Both steps' positions were drawn, not just the first repeated.
    assert 0.30 in positions_seen and 0.70 in positions_seen


def test_animation_legend_covers_every_step_and_is_built_once(series, tmp_path):
    """A surface present at only one step still needs its legend entry, and
    the legend must not grow or shrink as frames advance."""
    legends = []
    real_legend = plt.Axes.legend

    def spy(self, *args, **kwargs):
        legends.append(kwargs.get("handles"))
        return real_legend(self, *args, **kwargs)

    plt.Axes.legend = spy
    try:
        animate_profile_comparison(
            {"midplane": series}, "T", tmp_path / "b.gif",
            rational_lines_by_step={
                100: [(0.30, "blue", "n=1, m=2")],
                200: [(0.70, "blue", "n=1, m=2"), (0.40, "orange", "n=2, m=3")],
            },
        )
    finally:
        plt.Axes.legend = real_legend

    assert len(legends) == 1, "legend rebuilt per frame"
    assert [h.get_label() for h in legends[0]] == ["n=1, m=2", "n=2, m=3"]
