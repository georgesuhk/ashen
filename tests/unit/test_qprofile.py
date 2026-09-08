"""ashen.diagnostics.qprofile -- q-profile gathering and rational-surface
crossing search."""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics.qprofile import (
    find_rational_surfaces,
    rational_surface_matches,
    read_qprofile,
    run_qprofile_step,
    track_branches,
)
from ashen.jorek2 import Jorek2Run, MissingRestartError
from ashen.paths import RunPaths


# --- find_rational_surfaces --------------------------------------------------------


def test_finds_single_crossing():
    psi_n = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    crossings = find_rational_surfaces(psi_n, q, 1.5)
    assert len(crossings) == 1
    assert crossings[0] == pytest.approx(0.417, abs=1e-2)


def test_finds_multiple_crossings_for_reversed_shear():
    psi_n = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.array([2.5, 1.8, 1.5, 1.8, 2.5])  # dips below 2.0 and back up
    crossings = find_rational_surfaces(psi_n, q, 2.0)
    assert len(crossings) == 2


def test_no_crossing_returns_empty():
    psi_n = np.array([0.0, 0.5, 1.0])
    q = np.array([1.0, 1.2, 1.4])
    assert find_rational_surfaces(psi_n, q, 5.0) == []


def test_exact_sample_match_is_a_crossing():
    psi_n = np.array([0.0, 0.5, 1.0])
    q = np.array([1.0, 1.5, 2.0])
    crossings = find_rational_surfaces(psi_n, q, 1.5)
    assert crossings == pytest.approx([0.5])


# --- rational_surface_matches --------------------------------------------------------


def test_single_crossing_snaps_to_nearest_traced_value():
    psi_n = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.array([1.0, 1.3, 1.6, 1.9, 2.2])  # crosses 1.5 near psi_n=0.42
    traced = [0.1, 0.4, 0.6, 0.9]
    matches = rational_surface_matches(psi_n, q, [(2, 3, "red")], traced)
    assert matches == {0.4: "red"}


def test_multiple_crossings_all_get_the_same_mode_color():
    psi_n = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.array([2.5, 1.8, 1.5, 1.8, 2.5])  # crosses 2.0 twice
    traced = [0.1, 0.3, 0.7, 0.9]
    matches = rational_surface_matches(psi_n, q, [(1, 2, "blue")], traced)
    assert matches == {0.1: "blue", 0.9: "blue"}


def test_n_zero_entries_are_skipped():
    psi_n = np.array([0.0, 0.5, 1.0])
    q = np.array([1.0, 1.5, 2.0])
    traced = [0.5]
    matches = rational_surface_matches(psi_n, q, [(0, 1, "red")], traced)
    assert matches == {}


def test_empty_traced_psi_n_returns_empty():
    psi_n = np.array([0.0, 0.5, 1.0])
    q = np.array([1.0, 1.5, 2.0])
    matches = rational_surface_matches(psi_n, q, [(2, 1, "red")], [])
    assert matches == {}


def test_multiple_modes_can_map_to_different_traced_values():
    psi_n = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    traced = [0.1, 0.4, 0.6, 0.9]
    matches = rational_surface_matches(
        psi_n, q, [(2, 3, "red"), (1, 2, "green")], traced
    )
    assert matches[0.4] == "red"
    assert matches[0.9] == "green"


# --- read_qprofile ------------------------------------------------------------------


def test_read_qprofile(tmp_path):
    path = tmp_path / "qprofile_s000100.dat"
    path.write_text(
        "# Psi_n q\n"
        "# time step #000100\n"
        "0.1 1.0\n"
        "0.5 1.5\n"
        "0.9 2.0\n"
        "\n",
        encoding="utf-8",
    )
    psi_n, q = read_qprofile(path)
    assert psi_n.tolist() == [0.1, 0.5, 0.9]
    assert q.tolist() == [1.0, 1.5, 2.0]


def test_read_qprofile_handles_the_real_headerless_output(tmp_path):
    """The real ``jorek2_postproc`` output has no column names on the header
    line at all -- ``exec_commands.f90::qprofile`` sets ``n_expr = 0`` right
    before naming the two columns, so ``write_ascii_header``'s loop never
    runs and the line is just ``# `` with nothing after it. Column order
    (Psi_n, then q) is still reliable; this is what actually ships from the
    HPC, not the with-names variant above."""
    path = tmp_path / "qprofile_s000100.dat"
    path.write_text(
        "# \n"
        "# time step #000100\n"
        "0.1 1.0\n"
        "0.5 1.5\n"
        "0.9 2.0\n"
        "\n",
        encoding="utf-8",
    )
    psi_n, q = read_qprofile(path)
    assert psi_n.tolist() == [0.1, 0.5, 0.9]
    assert q.tolist() == [1.0, 1.5, 2.0]


# --- run_qprofile_step: in-place jorek2_postproc invocation, mirrors
# ashen.jorek2.run_zero_d's own test shape (test_jorek2.py). ------------------------


# No happy-path test here: run_qprofile_step hardcodes the exe name
# "jorek2_postproc" with no extension (POSTPROC_TOOL), the same as
# ashen.jorek2.run_zero_d's own "jorek2_postproc" -- on Windows an
# extensionless file cannot be executed, so (mirroring test_jorek2.py's
# test_run_zero_d_missing_restart_raises) only the pre-execution checks are
# exercised here; the actual subprocess invocation is identical to
# run_zero_d's already-covered shape.


def test_run_qprofile_step_missing_restart_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "jorek2_postproc").write_text("x", encoding="utf-8")
    namelist = run_dir / "in_main"
    namelist.write_text("&in1\n&end\n", encoding="utf-8")
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=namelist, pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)

    with pytest.raises(MissingRestartError):
        run_qprofile_step(run, 100, paths)


def test_run_qprofile_step_missing_exe_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000100.h5").write_bytes(b"fake-h5-data")
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)
    with pytest.raises(FileNotFoundError):
        run_qprofile_step(run, 100, paths)


# --- track_branches: following surfaces across steps ---------------------------


def test_track_branches_follows_one_drifting_surface():
    branches = track_branches({100: [0.70], 200: [0.66], 300: [0.61]})
    assert branches == [{100: 0.70, 200: 0.66, 300: 0.61}]


def test_track_branches_keeps_reversed_shear_pair_apart():
    """Two crossings of the same q, drifting independently, stay two branches."""
    branches = track_branches({100: [0.30, 0.80], 200: [0.34, 0.78]})
    assert branches == [{100: 0.30, 200: 0.34}, {100: 0.80, 200: 0.78}]


def test_track_branches_survives_a_surface_vanishing():
    """Regression for the reason this exists: pairing crossings by their rank
    within a step would re-label the outer surface as the inner one the moment
    the inner one merges away, turning its band into a spurious sweep across
    the whole domain."""
    branches = track_branches({100: [0.30, 0.80], 200: [0.34, 0.78], 300: [0.76]})

    assert branches == [{100: 0.30, 200: 0.34}, {100: 0.80, 200: 0.78, 300: 0.76}]
    outer = branches[1]
    assert max(outer.values()) - min(outer.values()) < 0.05  # not 0.30 -> 0.76


def test_track_branches_of_nothing_is_empty():
    assert track_branches({}) == []


def test_track_branches_follows_a_lone_surface_past_max_jump():
    """max_jump separates competing surfaces; it must not split a single one.

    Restarts saved far apart in time can show one surface moving further than
    the limit in one hop, and there is nothing it could be confused with.
    """
    branches = track_branches({100: [0.50], 200: [0.25]}, max_jump=0.1)
    assert branches == [{100: 0.50, 200: 0.25}]


def test_track_branches_does_not_pair_across_a_gap():
    """A branch that died two steps ago is not resurrected by a new crossing
    appearing elsewhere -- only a surface alive at the previous step is a
    candidate for the unambiguous pairing."""
    branches = track_branches({100: [0.20], 200: [], 300: [0.90]}, max_jump=0.1)
    assert branches == [{100: 0.20}, {300: 0.90}]


def test_track_branches_still_splits_two_ambiguous_surfaces():
    """Two crossings both beyond max_jump stay two new branches -- the
    leftover pairing only applies when there is exactly one of each."""
    branches = track_branches({100: [0.20, 0.25], 200: [0.80, 0.90]}, max_jump=0.1)
    assert len(branches) == 4
