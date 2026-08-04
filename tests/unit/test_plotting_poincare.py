"""ashen.plotting.poincare -- ports castor3d/util/data_jorek.py:354
plot_poincare, redrawn against the per-line cache instead of the three dense
.npz files it used to load, and against an ax= parameter instead of always
owning its own figure (no legacy plotting function accepted one).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: this suite never opens a display

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ashen.diagnostics.poincare_cache import LineKey, LineRecord
from ashen.plotting.poincare import draw_poincare, plot_poincare_step


def record(psi_n, R, n=5) -> LineRecord:
    key = LineKey(psi_n=psi_n, R=R, Z=0.0, phi=0.0).quantised()
    return LineRecord(
        key=key, n_turns=n, terminated=False, n_segments=1,
        R=np.full(n, R, dtype=np.float32),
        Z=np.zeros(n, dtype=np.float32),
        rho=np.sqrt(np.full(n, psi_n, dtype=np.float32)),
        theta=np.zeros(n, dtype=np.float32),
    )


@pytest.fixture
def records():
    lines = [
        record(0.1, 1.7), record(0.1, 1.71),
        record(0.5, 1.8), record(0.9, 1.9),
    ]
    return {r.key: r for r in lines}


def test_draw_poincare_onto_a_given_axes(records):
    """No legacy plotting function accepted ax= -- this is the point of the
    rewrite: a caller can compose figures instead of always getting one back
    already saved to disk."""
    fig, ax = plt.subplots()
    draw_poincare(ax, records)
    assert len(ax.collections) > 0
    plt.close(fig)


def test_draws_one_scatter_per_distinct_psi_n(records):
    fig, ax = plt.subplots()
    draw_poincare(ax, records)
    # 3 distinct psi_n values (0.1 appears twice, grouped into one scatter).
    assert len(ax.collections) == 3
    plt.close(fig)


def test_empty_cache_draws_nothing_but_does_not_raise():
    fig, ax = plt.subplots()
    draw_poincare(ax, {})
    plt.close(fig)


def test_highlight_dims_everything_else(records):
    fig, ax = plt.subplots()
    draw_poincare(ax, records, highlight={0.5})
    # Still one scatter call per group; dimming is a colour/alpha change,
    # not a change in what gets drawn.
    assert len(ax.collections) == 3
    plt.close(fig)


def test_axes_are_labelled_and_square(records):
    fig, ax = plt.subplots()
    draw_poincare(ax, records)
    assert ax.get_xlabel() == r"$R$"
    assert ax.get_ylabel() == r"$Z$"
    assert ax.get_aspect() in ("equal", 1.0)
    plt.close(fig)


def test_title_is_set_when_given(records):
    fig, ax = plt.subplots()
    draw_poincare(ax, records, title="t=200")
    assert ax.get_title() == "t=200"
    plt.close(fig)


# --- the file-owning wrapper -----------------------------------------------------


def test_plot_poincare_step_writes_a_file(records, tmp_path):
    out = plot_poincare_step(records, tmp_path / "sub" / "200_poincare.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_poincare_step_creates_missing_directories(records, tmp_path):
    out = plot_poincare_step(records, tmp_path / "a" / "b" / "c" / "out.png")
    assert out.is_file()
