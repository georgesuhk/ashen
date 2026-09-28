"""Tests for ashen.padding -- the one place step widths are decided.

The width-tolerant readers built on this (RunPaths, run_tool) have their own
tests in test_paths.py and test_jorek2.py; these pin the primitives and the
per-build width read from JOREK's source.
"""

from __future__ import annotations

import pytest

from ashen.padding import (
    PaddingError,
    find_step_file,
    resolve_step_file,
    restart_name,
    source_pad_width,
)

_IMPORT_RESTART = "communication/mod_import_restart.f90"


def _jorek_tree(tmp_path, declaration: str):
    source = tmp_path / _IMPORT_RESTART
    source.parent.mkdir(parents=True)
    source.write_text(
        "module mod_import_restart\n"
        f"{declaration}\n"
        "contains\n"
        "end module mod_import_restart\n",
        encoding="utf-8",
    )
    return tmp_path


# --- restart_name -------------------------------------------------------------


@pytest.mark.parametrize("width, expected", [(5, "jorek08002.h5"), (6, "jorek008002.h5")])
def test_restart_name_pads_to_the_width_given(width, expected):
    assert restart_name(8002, width) == expected


def test_restart_name_prefix_and_extension():
    assert restart_name(100, 6, prefix="jorek2", ext=".rst") == "jorek2000100.rst"


# --- find_step_file / resolve_step_file ------------------------------------------


def test_find_prefers_the_canonical_spelling(tmp_path):
    canonical = tmp_path / "jorek08002.h5"
    canonical.write_bytes(b"mine")
    (tmp_path / "jorek008002.h5").write_bytes(b"other")
    assert find_step_file(canonical, 8002) == canonical


def test_find_falls_back_to_the_other_width(tmp_path):
    written = tmp_path / "jorek008002.h5"
    written.write_bytes(b"x")
    assert find_step_file(tmp_path / "jorek08002.h5", 8002) == written


def test_find_returns_none_when_no_width_exists(tmp_path):
    assert find_step_file(tmp_path / "jorek08002.h5", 8002) is None


def test_find_repads_the_filename_not_the_directory(tmp_path):
    """A directory whose name happens to contain the step stays as given."""
    folder = tmp_path / "s08002"
    folder.mkdir()
    written = folder / "qprofile_s008002.dat"
    written.write_text("x", encoding="utf-8")
    assert find_step_file(folder / "qprofile_s08002.dat", 8002) == written


def test_find_ignores_a_directory_with_the_file_s_name(tmp_path):
    (tmp_path / "jorek08002.h5").mkdir()
    assert find_step_file(tmp_path / "jorek08002.h5", 8002) is None


def test_resolve_falls_back_to_canonical(tmp_path):
    canonical = tmp_path / "jorek08002.h5"
    assert resolve_step_file(canonical, 8002) == canonical


# --- source_pad_width ------------------------------------------------------------


def test_source_width_is_the_first_format_entry(tmp_path):
    """jorek_RE's actual declaration: 6-wide first."""
    tree = _jorek_tree(
        tmp_path,
        "character(len=20), parameter :: rst_file_ind_fmt(2) = (/'(a,i6.6)', '(a,i5.5)'/)",
    )
    assert source_pad_width(tree) == 6


def test_source_width_with_the_other_ordering(tmp_path):
    tree = _jorek_tree(
        tmp_path,
        "character(len=20), parameter :: rst_file_ind_fmt(2) = (/'(a,i5.5)', '(a,i6.6)'/)",
    )
    assert source_pad_width(tree) == 5


def test_source_width_tolerates_case_and_spacing(tmp_path):
    tree = _jorek_tree(
        tmp_path,
        'CHARACTER(len=20), PARAMETER :: RST_FILE_IND_FMT( 2 ) = (/ "(A, I5.5)", "(A, I6.6)" /)',
    )
    assert source_pad_width(tree) == 5


def test_source_width_missing_tree_raises(tmp_path):
    with pytest.raises(PaddingError, match="cannot read"):
        source_pad_width(tmp_path / "no_such_jorek")


def test_source_width_without_a_declaration_raises(tmp_path):
    tree = _jorek_tree(tmp_path, "integer :: unrelated = 1")
    with pytest.raises(PaddingError, match="no rst_file_ind_fmt declaration"):
        source_pad_width(tree)
