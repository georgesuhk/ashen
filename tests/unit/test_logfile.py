"""ashen.logfile -- ports castor3d/util/io.py:220 extract_from_file, but
raises instead of returning math.nan on failure."""

from __future__ import annotations

import pytest

from ashen.logfile import LogfileError, b_axis, extract_from_file, r_axis


def write(tmp_path, text):
    path = tmp_path / "log"
    path.write_text(text, encoding="utf-8")
    return path


def test_extracts_a_simple_field(tmp_path):
    path = write(tmp_path, "R_axis = 1.363245\n")
    assert extract_from_file(path, "R_axis") == pytest.approx(1.363245)


def test_field_must_be_left_of_the_delimiter(tmp_path):
    """A value that happens to contain the field name must not match."""
    path = write(tmp_path, "note = R_axis is discussed below\nR_axis = 1.4\n")
    assert extract_from_file(path, "R_axis") == pytest.approx(1.4)


def test_first_occurrence_by_default(tmp_path):
    path = write(tmp_path, "R_axis = 1.1\nR_axis = 1.2\n")
    assert extract_from_file(path, "R_axis") == pytest.approx(1.1)


def test_nth_occurrence(tmp_path):
    path = write(tmp_path, "R_axis = 1.1\nR_axis = 1.2\nR_axis = 1.3\n")
    assert extract_from_file(path, "R_axis", occurrence=2) == pytest.approx(1.2)


def test_last_occurrence(tmp_path):
    """A restarted run logs the same field once per stage; 'last' is what a
    caller wants for the final, post-restart value."""
    path = write(tmp_path, "R_axis = 1.1\nR_axis = 1.2\nR_axis = 1.3\n")
    assert extract_from_file(path, "R_axis", occurrence="last") == pytest.approx(1.3)


def test_missing_field_raises(tmp_path):
    path = write(tmp_path, "eta = 1e-3\n")
    with pytest.raises(LogfileError, match="R_axis"):
        extract_from_file(path, "R_axis")


def test_missing_file_raises(tmp_path):
    with pytest.raises(LogfileError, match="cannot be read"):
        extract_from_file(tmp_path / "nope", "R_axis")


def test_unparseable_value_raises(tmp_path):
    path = write(tmp_path, "R_axis = notanumber\n")
    with pytest.raises(LogfileError, match="R_axis"):
        extract_from_file(path, "R_axis")


def test_occurrence_past_the_end_raises(tmp_path):
    path = write(tmp_path, "R_axis = 1.1\n")
    with pytest.raises(LogfileError, match="occurrence 2"):
        extract_from_file(path, "R_axis", occurrence=2)


def test_r_axis_reads_the_named_field(tmp_path):
    path = write(tmp_path, "some other line\nR_axis = 1.363245\nmore\n")
    assert r_axis(path) == pytest.approx(1.363245)


def test_r_axis_raises_rather_than_returning_nan(tmp_path):
    """The legacy extract_from_file swallowed every failure into math.nan
    (io.py:259-262), which then poisoned q-profile calculations silently."""
    path = write(tmp_path, "no axis info here\n")
    with pytest.raises(LogfileError):
        r_axis(path)


def test_b_axis_divides_f0_by_r_axis(tmp_path):
    path = write(tmp_path, "F0 = 3.0\nR_axis = 1.5\n")
    assert b_axis(path) == pytest.approx(2.0)


def test_b_axis_raises_if_f0_missing(tmp_path):
    path = write(tmp_path, "R_axis = 1.5\n")
    with pytest.raises(LogfileError, match="F0"):
        b_axis(path)


def test_b_axis_raises_if_r_axis_missing(tmp_path):
    path = write(tmp_path, "F0 = 3.0\n")
    with pytest.raises(LogfileError, match="R_axis"):
        b_axis(path)
