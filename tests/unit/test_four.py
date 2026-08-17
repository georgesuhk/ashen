"""ashen.diagnostics.four -- jorek2_four decomposition, gathering and parsing.

Driven by a stub ``jorek2_four`` executable, so no JOREK is needed. The stub
reads the optional ``four_params.nml`` for ``nstpts``/``nTht`` (falling back
to the tool's own defaults if absent) and writes one
``{variable}_modes_n{n:03d}`` file per variable in ``("Psi", "u")`` and
toroidal mode ``n`` in ``(0, 1)``, each with ``nTht/2 + 1`` poloidal-mode
blocks of ``nstpts`` rows -- the same block/header shape
``jorek2_four.f90:99-119`` writes.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from ashen.diagnostics import four
from ashen.diagnostics import four_cache as fc
from ashen.jorek2 import Jorek2Error, Jorek2Run
from ashen.paths import RunPaths

pytest.importorskip("h5py")

_EXT = ".cmd" if os.name == "nt" else ""

_FOUR = """
import re, sys

sys.stdin.read()

nstpts, ntht = 30, 32
try:
    text = open("four_params.nml").read()
    m = re.search(r"nstpts\\s*=\\s*(\\d+)", text)
    if m: nstpts = int(m.group(1))
    m = re.search(r"n[Tt]ht\\s*=\\s*(\\d+)", text)
    if m: ntht = int(m.group(1))
except FileNotFoundError:
    pass

for var in ("Psi", "u"):
    for n in (0, 1):
        lines = []
        for l in range(ntht // 2 + 1):
            lines.append("# %3d:   m=%3d, n=%3d" % (l, l, n))
            for k in range(nstpts):
                psi_n = k / max(nstpts - 1, 1)
                lines.append("  %.7e  %.7e  %.7e  %.7e  %.7e" % (
                    psi_n, float(l), float(n), psi_n * (l + 1), 0.0
                ))
            lines.append("")
            lines.append("")
        with open("%s_modes_n%03d" % (var, n), "w") as f:
            f.write("\\n".join(lines) + "\\n")
"""


def _install(directory: Path, name: str, payload: str) -> None:
    src = directory / f"_{name}_payload.py"
    src.write_text(payload, encoding="utf-8")
    exe = directory / f"{name}{_EXT}"
    if os.name == "nt":
        exe.write_text(f'@echo off\r\n"{sys.executable}" "{src}"\r\n', encoding="utf-8")
    else:
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{src}"\n', encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exe_dir = run_dir / "exe"
    exe_dir.mkdir()
    _install(exe_dir, "jorek2_four", _FOUR)
    monkeypatch.setattr(four, "FOUR_TOOL", f"jorek2_four{_EXT}")

    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000100.h5").write_bytes(b"fake")

    return Jorek2Run(
        run_dir=run_dir, exe_dir=run_dir,
        namelist=run_dir / "in_main", pad_width=6, profiles=(),
    )


@pytest.fixture
def paths(run):
    return RunPaths(run.run_dir, pad_width=6)


# --- four_params_nml ---------------------------------------------------------------------


def test_four_params_nml_format():
    text = four.four_params_nml(
        nstpts=50, ntht=16, nmaxsteps=999, deltaphi=0.25,
        nsmallsteps=2, rad_range=(0.01, 0.98),
    )
    assert text.startswith("&four_params")
    assert "nstpts = 50" in text
    assert "nTht = 16" in text
    assert "nmaxsteps = 999" in text
    assert "deltaphi = 0.25" in text
    assert "nsmallsteps = 2" in text
    assert "rad_range = 0.01, 0.98" in text
    assert text.strip().endswith("/")


# --- _parse_four_file ---------------------------------------------------------------------


def test_parse_four_file_extracts_blocks_in_order(tmp_path):
    path = tmp_path / "Psi_modes_n000"
    path.write_text(
        "# 0:   m=0, n=0\n"
        "0.0 1.0 2.0 3.0 4.0\n"
        "0.5 1.0 2.0 3.0 4.0\n"
        "\n\n"
        "# 1:   m=1, n=0\n"
        "0.0 5.0 6.0 7.0 8.0\n"
        "\n\n",
        encoding="utf-8",
    )
    blocks = four._parse_four_file(path, expected_n=0)
    assert [m for m, _ in blocks] == [0, 1]
    np.testing.assert_allclose(blocks[0][1][:, 0], [0.0, 0.5])
    np.testing.assert_allclose(blocks[1][1][:, 0], [0.0])


def test_parse_four_file_rejects_out_of_order_header(tmp_path):
    path = tmp_path / "Psi_modes_n000"
    path.write_text("# 1:   m=1, n=0\n0.0 1.0 2.0 3.0 4.0\n\n\n", encoding="utf-8")
    with pytest.raises(Jorek2Error, match="out of order"):
        four._parse_four_file(path, expected_n=0)


def test_parse_four_file_rejects_n_mismatch(tmp_path):
    path = tmp_path / "Psi_modes_n000"
    path.write_text("# 0:   m=0, n=1\n0.0 1.0 2.0 3.0 4.0\n\n\n", encoding="utf-8")
    with pytest.raises(Jorek2Error, match="filename says"):
        four._parse_four_file(path, expected_n=0)


def test_parse_four_file_rejects_unrecognised_header(tmp_path):
    path = tmp_path / "Psi_modes_n000"
    path.write_text("# garbage\n0.0 1.0 2.0 3.0 4.0\n\n\n", encoding="utf-8")
    with pytest.raises(Jorek2Error, match="unrecognised header"):
        four._parse_four_file(path, expected_n=0)


# --- run_four_step / run_four_scan --------------------------------------------------------


def test_run_four_step_writes_a_cache(run, paths):
    report = four.run_four_step(run, paths, 100, nstpts=5, ntht=4)
    assert report.cached is False
    assert report.n_records == 2 * 2 * 3  # 2 vars * 2 n * (4//2+1) m-blocks

    cache = fc.read_cache(paths.four_cache(100))
    assert set(cache) == {
        (var, n, m) for var in ("Psi", "u") for n in (0, 1) for m in range(3)
    }
    record = cache[("Psi", 0, 1)]
    assert record.psi_n.shape == (5,)


def test_cached_step_is_skipped_without_force(run, paths):
    four.run_four_step(run, paths, 100, nstpts=5, ntht=4)
    # Break the stub so a second real invocation would fail -- proves the
    # cache gate actually skipped running the tool, not just re-ran it quietly.
    (run.run_dir / "exe" / f"jorek2_four{_EXT}").unlink()

    report = four.run_four_step(run, paths, 100, nstpts=5, ntht=4)
    assert report.cached is True
    assert report.n_records == 12


def test_force_reruns_even_with_a_cache(run, paths):
    four.run_four_step(run, paths, 100, nstpts=5, ntht=4)
    report = four.run_four_step(run, paths, 100, nstpts=5, ntht=4, force=True)
    assert report.cached is False


def test_case_params_reach_four_params_nml(run, paths):
    four.run_four_step(run, paths, 100, nstpts=7, ntht=4)
    record = fc.read_cache(paths.four_cache(100))[("Psi", 0, 0)]
    assert record.psi_n.shape == (7,)


def test_scratch_directory_is_removed(run, paths):
    four.run_four_step(run, paths, 100, nstpts=5, ntht=4)
    assert not list(paths.four_dir.glob("_scratch*"))


def test_scan_reports_one_entry_per_step_in_order(run, paths):
    (run.run_dir / "jorek000200.h5").write_bytes(b"fake")
    reports = four.run_four_scan(run, paths, [100, 200], nstpts=5, ntht=4, n_workers=1)
    assert [r.step for r in reports] == [100, 200]
    assert all(r.n_records == 12 for r in reports)
    assert paths.four_cache(200).is_file()


def test_scan_progress_callback_fires_once_per_step(run, paths):
    (run.run_dir / "jorek000200.h5").write_bytes(b"fake")
    seen = []
    four.run_four_scan(
        run, paths, [100, 200], nstpts=5, ntht=4, n_workers=1,
        on_progress=lambda done, total, report: seen.append((done, total, report.step)),
    )
    assert seen == [(1, 2, 100), (2, 2, 200)]
