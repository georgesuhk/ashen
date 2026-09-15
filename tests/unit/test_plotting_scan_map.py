"""ashen.plotting.scan_map -- one point per JOREK run, x/y from named
per-run scalars, a third quantity encoded as point colour or ring size."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: this suite never opens a display

import matplotlib.pyplot as plt
import pytest

from ashen.plotting import MARKER_CYCLE
from ashen.plotting.colors import DISCRETE_PALETTE
from ashen.plotting.scan_map import (
    COLOR_ENCODING,
    ENCODINGS,
    SIZE_ENCODING,
    RingSizer,
    ScanPoint,
    colorer_for,
    draw_scan_map,
    plot_scan_map,
    plot_scan_map_datasets,
    sizer,
)

_POINTS = [ScanPoint(x=1e-3, y=3.0, c=0.1), ScanPoint(x=1e-4, y=2.0, c=0.5)]


def _spy_subplots(monkeypatch):
    captured = []
    original = plt.subplots

    def spy(*a, **k):
        fig, ax = original(*a, **k)
        captured.append((fig, ax))
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    return captured


# --- draw_scan_map: colour encoding ------------------------------------------


def test_draw_color_encoding_scatters_one_collection():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c)
    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_offsets()) == len(_POINTS)
    plt.close(fig)


def test_draw_color_encoding_needs_a_colorer():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING)
    plt.close(fig)


def test_draw_size_encoding_needs_a_sizer():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        draw_scan_map(ax, _POINTS, encode=SIZE_ENCODING)
    plt.close(fig)


def test_draw_rejects_unknown_encoding():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        draw_scan_map(ax, _POINTS, encode="rainbow")
    plt.close(fig)


def test_draw_defaults_to_linear_scales_on_both_axes():
    """Deliberate divergence from draw_wetted_fraction_vs_x's log-x default:
    either axis here can be any registered quantity and most (q95, li,
    edge_q) are O(1), so there's no single sensible default."""
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c)
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_draw_log_x_and_log_y():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c, log_x=True, log_y=True)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_draw_sets_labels_when_given():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(
        ax, _POINTS, encode=COLOR_ENCODING, colorer=c, xlabel=r"$\eta$", ylabel="q95",
    )
    assert ax.get_xlabel() == r"$\eta$"
    assert ax.get_ylabel() == "q95"
    plt.close(fig)


def test_draw_omits_labels_when_not_given():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c)
    assert ax.get_xlabel() == ""
    plt.close(fig)


def test_draw_without_label_has_no_legend_entry():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c)
    assert ax.collections[0].get_label().startswith("_")
    plt.close(fig)


def test_draw_with_label_is_used_by_the_legend():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c, label="normal")
    assert ax.collections[0].get_label() == "normal"
    plt.close(fig)


def test_two_draw_calls_with_different_markers_produce_two_collections():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c, marker="o")
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c, marker="s")
    assert len(ax.collections) == 2
    plt.close(fig)


def test_color_encoding_constant_size_varying_facecolor():
    fig, ax = plt.subplots()
    c = colorer_for(_POINTS)
    draw_scan_map(ax, _POINTS, encode=COLOR_ENCODING, colorer=c, point_size=60.0)
    coll = ax.collections[0]
    sizes = coll.get_sizes()
    assert len(set(sizes.tolist())) == 1  # constant
    colors = coll.get_facecolor()
    assert len(colors) == len(_POINTS)
    assert (colors[0] != colors[1]).any()
    plt.close(fig)


# --- draw_scan_map: size encoding --------------------------------------------


def test_size_encoding_transparent_facecolor_varying_size():
    fig, ax = plt.subplots()
    s = sizer([p.c for p in _POINTS])
    draw_scan_map(ax, _POINTS, encode=SIZE_ENCODING, sizer=s, color="tab:blue")
    coll = ax.collections[0]
    facecolors = coll.get_facecolor()
    assert len(facecolors) == 0 or (facecolors[:, 3] == 0).all()
    sizes = coll.get_sizes()
    assert len(set(sizes.tolist())) > 1  # varies with c
    plt.close(fig)


def test_annotate_labels_each_point():
    fig, ax = plt.subplots()
    points = [ScanPoint(x=1, y=1, c=0.1, label="a"), ScanPoint(x=2, y=2, c=0.2, label="b")]
    c = colorer_for(points)
    draw_scan_map(ax, points, encode=COLOR_ENCODING, colorer=c, annotate=True)
    assert len(ax.texts) == 2
    assert {t.get_text() for t in ax.texts} == {"a", "b"}
    plt.close(fig)


def test_no_annotate_by_default():
    fig, ax = plt.subplots()
    points = [ScanPoint(x=1, y=1, c=0.1, label="a")]
    c = colorer_for(points)
    draw_scan_map(ax, points, encode=COLOR_ENCODING, colorer=c)
    assert len(ax.texts) == 0
    plt.close(fig)


# --- RingSizer ----------------------------------------------------------


def test_ring_sizer_endpoints():
    s = RingSizer(vmin=0.0, vmax=1.0, s_min=10.0, s_max=100.0)
    assert s(0.0) == pytest.approx(10.0)
    assert s(1.0) == pytest.approx(100.0)


def test_ring_sizer_midpoint_is_midway_in_area():
    s = RingSizer(vmin=0.0, vmax=1.0, s_min=10.0, s_max=100.0)
    assert s(0.5) == pytest.approx(55.0)


def test_ring_sizer_single_value_does_not_divide_by_zero():
    s = RingSizer(vmin=1.0, vmax=1.0, s_min=10.0, s_max=100.0)
    assert s(1.0) == pytest.approx(55.0)  # midpoint fallback, no crash


def test_ring_sizer_log_scale_maps_geometric_mean_to_midpoint():
    s = RingSizer(vmin=1.0, vmax=100.0, s_min=10.0, s_max=100.0, log_scale=True)
    assert s(10.0) == pytest.approx(55.0)  # sqrt(1*100) = 10 -> midpoint


def test_sizer_empty_input_has_a_usable_default():
    s = sizer([])
    s(0.5)  # must not raise


def test_ring_sizer_legend_handles_count():
    s = RingSizer(vmin=1.0, vmax=100.0)
    handles, labels = s.legend_handles(n=3)
    assert len(handles) == 3
    assert len(labels) == 3


# --- plot_scan_map: file-owning wrapper --------------------------------------


def test_plot_scan_map_writes_a_file(tmp_path):
    out = plot_scan_map(_POINTS, tmp_path / "sub" / "scan.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_scan_map_color_encoding_has_colorbar(tmp_path, monkeypatch):
    captured = _spy_subplots(monkeypatch)
    plot_scan_map(_POINTS, tmp_path / "scan.png", encode=COLOR_ENCODING, clabel="c")
    fig, ax = captured[-1]
    assert len(fig.axes) == 2  # main axes + colorbar axes


def test_plot_scan_map_size_encoding_has_size_legend(tmp_path, monkeypatch):
    captured = _spy_subplots(monkeypatch)
    plot_scan_map(_POINTS, tmp_path / "scan.png", encode=SIZE_ENCODING, clabel="c")
    fig, ax = captured[-1]
    assert ax.get_legend() is not None


# --- plot_scan_map_datasets: overlaying several datasets ---------------------


def test_plot_scan_map_datasets_writes_a_file(tmp_path):
    series = [("normal", _POINTS), ("rho19", [ScanPoint(x=1e-3, y=1.0, c=0.3)])]
    out = plot_scan_map_datasets(series, tmp_path / "sub" / "scan.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_scan_map_datasets_color_encoding_assigns_markers_from_cycle(tmp_path, monkeypatch):
    captured = _spy_subplots(monkeypatch)
    series = [
        ("a", [ScanPoint(x=1, y=1, c=0.1)]),
        ("b", [ScanPoint(x=2, y=2, c=0.2)]),
        ("c", [ScanPoint(x=3, y=3, c=0.3)]),
    ]
    plot_scan_map_datasets(series, tmp_path / "scan.png", encode=COLOR_ENCODING)
    fig, ax = captured[-1]
    assert len(ax.collections) == 3


def test_plot_scan_map_datasets_color_encoding_explicit_markers(tmp_path):
    series = [("a", [ScanPoint(x=1, y=1, c=0.1)]), ("b", [ScanPoint(x=2, y=2, c=0.2)])]
    out = plot_scan_map_datasets(
        series, tmp_path / "scan.png", encode=COLOR_ENCODING, markers=["*", None],
    )
    assert out.is_file()


def test_plot_scan_map_datasets_size_encoding_assigns_colors_from_palette(tmp_path, monkeypatch):
    captured = _spy_subplots(monkeypatch)
    series = [
        ("a", [ScanPoint(x=1, y=1, c=0.1)]),
        ("b", [ScanPoint(x=2, y=2, c=0.2)]),
    ]
    plot_scan_map_datasets(series, tmp_path / "scan.png", encode=SIZE_ENCODING)
    fig, ax = captured[-1]
    colors = [coll.get_edgecolor()[0] for coll in ax.collections]
    from matplotlib.colors import to_rgba

    assert colors[0] == pytest.approx(to_rgba(DISCRETE_PALETTE[0]))
    assert colors[1] == pytest.approx(to_rgba(DISCRETE_PALETTE[1]))


def test_plot_scan_map_datasets_size_encoding_has_two_legends(tmp_path, monkeypatch):
    """matplotlib's ax.legend() replaces rather than adds -- the dataset
    legend must be re-added with ax.add_artist before the size legend is
    made, or it silently disappears."""
    captured = _spy_subplots(monkeypatch)
    series = [
        ("a", [ScanPoint(x=1, y=1, c=0.1)]),
        ("b", [ScanPoint(x=2, y=2, c=0.9)]),
    ]
    plot_scan_map_datasets(series, tmp_path / "scan.png", encode=SIZE_ENCODING, clabel="c")
    fig, ax = captured[-1]
    from matplotlib.legend import Legend

    legends = [child for child in ax.get_children() if isinstance(child, Legend)]
    assert len(legends) == 2


def test_shared_normalisation_across_datasets(tmp_path, monkeypatch):
    """Two datasets with disjoint c ranges must be coloured/sized from one
    combined range -- the max point of the smaller dataset must not become
    the largest ring, which would happen if each dataset normalised itself."""
    captured = _spy_subplots(monkeypatch)
    series = [
        ("small", [ScanPoint(x=1, y=1, c=0.1), ScanPoint(x=2, y=2, c=0.2)]),
        ("large", [ScanPoint(x=3, y=3, c=0.8), ScanPoint(x=4, y=4, c=1.0)]),
    ]
    plot_scan_map_datasets(series, tmp_path / "scan.png", encode=SIZE_ENCODING)
    fig, ax = captured[-1]
    small_sizes = ax.collections[0].get_sizes()
    large_sizes = ax.collections[1].get_sizes()
    # The smaller dataset's largest point (c=0.2) must be smaller than the
    # larger dataset's smallest point (c=0.8) -- true only under one shared
    # normalisation across both datasets.
    assert max(small_sizes) < min(large_sizes)


def test_encode_not_in_encodings_raises():
    assert set(ENCODINGS) == {"color", "size"}
    with pytest.raises(ValueError):
        plot_scan_map([ScanPoint(x=1, y=1, c=0.1)], "unused.png", encode="rainbow")
