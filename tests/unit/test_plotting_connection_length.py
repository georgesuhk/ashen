"""ashen.plotting.connection_length -- ports
castor3d/util/data_jorek.py:597 color_con_length_plot, drawing only (the
matrix computation lives in ashen.diagnostics.connection_length)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.plotting.connection_length import (
    draw_connection_length_map,
    plot_connection_length_map,
)


@pytest.fixture
def matrix():
    return np.array([
        [10.0, 100.0, np.inf],
        [20.0, np.inf, 5000.0],
    ])  # (n_steps=2, n_psi=3)


def test_draw_returns_a_mappable_for_an_external_colorbar(matrix):
    """Unlike the legacy version, which always drew its own colourbar
    (data_jorek.py:656) -- composing into a grid needs one shared bar."""
    fig, ax = plt.subplots()
    pcm = draw_connection_length_map(ax, matrix, np.array([100, 200]), np.array([0.2, 0.5, 0.8]))
    assert pcm is not None
    plt.close(fig)


def test_infinite_cells_are_masked_not_plotted_as_data(matrix):
    fig, ax = plt.subplots()
    pcm = draw_connection_length_map(ax, matrix, np.array([100, 200]), np.array([0.2, 0.5, 0.8]))
    array = pcm.get_array()
    assert np.ma.is_masked(array) or array.mask.any()
    plt.close(fig)


def test_ylabel_is_psi_n(matrix):
    fig, ax = plt.subplots()
    draw_connection_length_map(ax, matrix, np.array([100, 200]), np.array([0.2, 0.5, 0.8]))
    assert ax.get_ylabel() == r"$\Psi_N$"
    plt.close(fig)


def test_smooth_flag_changes_the_plotted_values(matrix):
    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    pcm1 = draw_connection_length_map(ax1, matrix, np.array([100, 200]), np.array([0.2, 0.5, 0.8]), smooth=False)
    pcm2 = draw_connection_length_map(ax2, matrix, np.array([100, 200]), np.array([0.2, 0.5, 0.8]), smooth=True)
    finite1 = pcm1.get_array().compressed()
    finite2 = pcm2.get_array().compressed()
    assert not np.array_equal(finite1, finite2)
    plt.close(fig)
    plt.close(fig2)


# --- the file-owning wrapper, filename convention -----------------------------


def test_true_time_plot_is_prefixed_lctt(matrix, tmp_path):
    """Ports the legacy LCTT_/LC_ naming exactly (data_jorek.py:663,666)."""
    out = plot_connection_length_map(
        matrix, [100, 200], np.array([0.2, 0.5, 0.8]), tmp_path,
        true_times=[1e-4, 2e-4], plot_true_times=True,
    )
    assert out.name.startswith("LCTT_")
    assert out.is_file()


def test_step_index_plot_is_prefixed_lc(matrix, tmp_path):
    out = plot_connection_length_map(
        matrix, [100, 200], np.array([0.2, 0.5, 0.8]), tmp_path,
        plot_true_times=False,
    )
    assert out.name.startswith("LC_")
    assert not out.name.startswith("LCTT_")


def test_true_times_required_when_requested(matrix, tmp_path):
    with pytest.raises(ValueError, match="true_times"):
        plot_connection_length_map(
            matrix, [100, 200], np.array([0.2, 0.5, 0.8]), tmp_path,
            plot_true_times=True,
        )


def test_filename_encodes_step_endpoints(matrix, tmp_path):
    out = plot_connection_length_map(
        matrix, [100, 500], np.array([0.2, 0.5, 0.8]), tmp_path,
        plot_true_times=False,
    )
    assert "100" in out.name and "500" in out.name


def test_output_directory_is_created(matrix, tmp_path):
    out_dir = tmp_path / "poinc_dir"
    out = plot_connection_length_map(
        matrix, [100, 200], np.array([0.2, 0.5, 0.8]), out_dir,
        plot_true_times=False,
    )
    assert out.parent == out_dir
    assert out.is_file()
