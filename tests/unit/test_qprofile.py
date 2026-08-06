"""ashen.diagnostics.qprofile -- q-profile gathering and rational-surface
crossing search."""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics.qprofile import find_rational_surfaces, read_qprofile, run_qprofile_step
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
