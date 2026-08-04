"""Tests for run-folder conventions and step padding.

The central property: whatever width a run uses, readers and writers must
agree. The old code mixed a sniffed width with a hardcoded default at
different call sites, so non-6-padded runs silently produced caches nobody
looked for.
"""

from __future__ import annotations

import pytest

from ashen.paths import (
    DEFAULT_PAD_WIDTH,
    PaddingError,
    RunPaths,
    detect_pad_width,
    step_str,
)


def make_run(tmp_path, width: int, steps=(0, 100, 2000)):
    for step in steps:
        (tmp_path / f"jorek{step:0{width}d}.h5").write_bytes(b"")
    return tmp_path


# --- detection ---------------------------------------------------------------


@pytest.mark.parametrize("width", [4, 5, 6, 7])
def test_detects_each_width(tmp_path, width):
    make_run(tmp_path, width)

    assert detect_pad_width(tmp_path) == width


def test_majority_wins_over_a_stray_file(tmp_path):
    make_run(tmp_path, 5, steps=(0, 100, 2000, 3000))
    (tmp_path / "jorek0000000042.h5").write_bytes(b"")

    assert detect_pad_width(tmp_path) == 5


def test_underscored_restart_names_are_recognised(tmp_path):
    (tmp_path / "jorek_restart_00100.h5").write_bytes(b"")

    assert detect_pad_width(tmp_path) == 5


def test_empty_directory_is_an_explicit_error(tmp_path):
    with pytest.raises(PaddingError, match="jorek"):
        detect_pad_width(tmp_path)


def test_unrelated_h5_files_are_ignored(tmp_path):
    (tmp_path / "something00100.h5").write_bytes(b"")

    with pytest.raises(PaddingError):
        detect_pad_width(tmp_path)


# --- the headline property ---------------------------------------------------


@pytest.mark.parametrize("width", [4, 5, 6, 7])
def test_reader_and_writer_agree_on_every_width(tmp_path, width):
    """Regression for the poinc_diag read/write mismatch.

    Previously the .npz was written with width=6 and read with the sniffed
    width, so on a width-5 run the reader looked for a file that never existed.
    """
    run = make_run(tmp_path, width)
    paths = RunPaths.detect(run)

    writer_wrote = paths.poincare_cache(100)
    reader_expects = RunPaths.detect(run).poincare_cache(100)

    assert writer_wrote == reader_expects
    assert paths.pad_width == width


@pytest.mark.parametrize("width", [4, 5, 7])
def test_non_default_width_is_actually_used(tmp_path, width):
    """Guards against silently falling back to 6."""
    run = make_run(tmp_path, width)
    paths = RunPaths.detect(run)

    assert paths.step_str(100) == f"{100:0{width}d}"
    assert paths.step_str(100) != step_str(100, DEFAULT_PAD_WIDTH)


def test_all_artefacts_share_one_width(tmp_path):
    run = make_run(tmp_path, 5)
    paths = RunPaths.detect(run)

    stamped = [
        paths.restart(100).name,
        paths.zero_d(100).name,
        paths.poincare_cache(100).name,
        paths.flux_surface(0.5, 100).name,
    ]
    assert all("00100" in name for name in stamped)
    assert not any("000100" in name for name in stamped)


# --- filenames ---------------------------------------------------------------


def test_step_str_accepts_floats_from_numpy_ranges():
    assert step_str(100.0, 6) == "000100"


def test_restart_filename(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert paths.restart(100).name == "jorek000100.h5"
    assert paths.restart(100, prefix="jorek2", ext=".rst").name == "jorek2000100.rst"


def test_live_restart_is_not_padded(tmp_path):
    """jorek2_* tools always read a fixed name, not a stepped one."""
    paths = RunPaths(tmp_path, pad_width=5)

    assert paths.live_restart.name == "jorek_restart.h5"


def test_postproc_artefact_names(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert paths.zero_d(10).name == "zeroD_quantities_s000010.dat"
    assert paths.flux_surface(0.5, 10).name == "fluxsurface_at_psi_0.500_s000010.dat"
    assert paths.poincare_cache(10).name == "poinc_t000010_psi_n.npz"


def test_artefacts_live_under_the_run_directory(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert paths.zero_d(10).parent == tmp_path / "postproc"
    assert paths.poincare_cache(10).parent == tmp_path / "poinc_dir"
    assert paths.in_eq.parent == tmp_path


def test_namelists_are_the_three_the_runner_edits(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)

    assert [p.name for p in paths.namelists] == ["in_eq", "in_main", "in_main_r"]


def test_flux_surface_psi_uses_three_decimals(tmp_path):
    """Matches the convention poinc_diag.py:100 writes and reads."""
    paths = RunPaths(tmp_path, pad_width=6)

    assert "psi_0.010_" in paths.flux_surface(0.01, 0).name
    assert "psi_0.950_" in paths.flux_surface(0.95, 0).name
