"""ashen.plotting.colors -- colours a field line by its psi_n, replacing the
legacy 135-line hand-written contrast_colors list indexed by scan position
(data_jorek.py:27-135), which has no meaning once lines are keyed by start
position rather than scan order.
"""

from __future__ import annotations

import pytest

from ashen.plotting.colors import DISCRETE_PALETTE, colorer


def test_colormap_is_monotonic_along_one_channel_direction():
    """Not a strict requirement of any colormap, but viridis increases in
    lightness with value, so two extremes should differ."""
    c = colorer([0.1, 0.9], cmap="viridis")
    low = c(0.1)
    high = c(0.9)
    assert low != high


def test_same_psi_n_gives_the_same_colour_repeatedly():
    c = colorer([0.1, 0.5, 0.9])
    assert c(0.5) == c(0.5)


def test_colour_depends_only_on_the_fitted_range_not_call_order():
    c = colorer([0.0, 1.0])
    a = c(0.3)
    c(0.9)
    b = c(0.3)
    assert a == b


def test_empty_value_set_still_produces_a_colorer():
    c = colorer([])
    c(0.5)  # must not raise


def test_discrete_mode_uses_the_palette():
    c = colorer([0.0, 1.0], discrete=True)
    assert c(0.0) in DISCRETE_PALETTE
    assert c(1.0) in DISCRETE_PALETTE


def test_discrete_mode_is_deterministic_by_position_not_insertion_order():
    """A scan can be extended incrementally, so colour must be a function of
    the value alone, not of when it was first seen."""
    c = colorer([0.0, 0.5, 1.0], discrete=True)
    first = c(0.5)
    c2 = colorer([1.0, 0.5, 0.0], discrete=True)
    assert first == c2(0.5)


def test_discrete_palette_has_no_duplicates():
    """The legacy list pasted the same ~19-colour block four times with
    drifting values, including one colour appearing four times consecutively
    (data_jorek.py:48-51) -- the replacement must not repeat that."""
    assert len(DISCRETE_PALETTE) == len(set(DISCRETE_PALETTE))


def test_scalar_mappable_spans_the_fitted_range():
    c = colorer([0.2, 0.8])
    sm = c.scalar_mappable()
    assert sm.norm.vmin == pytest.approx(0.2)
    assert sm.norm.vmax == pytest.approx(0.8)
