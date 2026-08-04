"""End-to-end: run_poincare_step doing only the work it hasn't already done.

Driven by stub ``jorek2_postproc`` / ``jorek2_poincare`` executables, so no
JOREK is needed. The stub Poincare tool records every ``stpts`` it is handed,
which is what lets these tests assert the *quantity* of work -- how many lines
were traced and with how many turns each -- rather than only the end state.
It also deliberately emits its output blocks in **reverse** order to keep the
OpenMP-ordering assumption exercised on the real code path (see
``test_poincare_demux.py`` for that in isolation).

These are the behaviours George asked for:

- adding a psi_n costs one psi_n's worth of lines, not a rescan
- raising n_turns costs the difference, resumed from the last puncture
- re-running an unchanged request costs nothing at all
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from ashen.diagnostics import poincare as poinc
from ashen.diagnostics import poincare_cache as pc
from ashen.jorek2 import Jorek2Run
from ashen.paths import RunPaths

pytest.importorskip("h5py")

_EXT = ".cmd" if os.name == "nt" else ""

# A stub jorek2_postproc: parses the control script for `for step N do` and
# `fluxsurface PSI`, and writes a flux surface of evenly spaced (R, Z) points.
_POSTPROC = """
import re, sys, os
script = sys.stdin.read()
step = int(re.search(r"for step (\\d+)", script).group(1))
psi = float(re.search(r"fluxsurface ([0-9.eE+-]+)", script).group(1))
os.makedirs("postproc", exist_ok=True)
width = int(os.environ["STUB_PAD_WIDTH"])
name = "postproc/fluxsurface_at_psi_%.3f_s%s.dat" % (psi, str(step).zfill(width))
with open(name, "w") as f:
    for i in range(32):
        f.write("  %.6f  %.6f\\n" % (1.5 + psi + i * 0.01, 0.2 * i - 0.3))
"""

# A stub jorek2_poincare: reads stpts, appends it to a JSON log for the test to
# inspect, and emits one block per line with the requested number of turns --
# in reverse order, with matching "=> Line" messages.
_POINCARE = """
import json, os, sys
sys.stdin.read()
rows = []
with open("stpts") as f:
    lines = [l for l in f.read().splitlines() if l.strip()]
n_lines = int(lines[1])
for raw in lines[3:3 + n_lines]:
    nr, R, Z, phi, n_turns = raw.split()
    rows.append({"nr": int(nr), "R": float(R), "Z": float(Z),
                 "phi": float(phi), "n_turns": int(n_turns)})

log = os.environ["STUB_LOG"]
existing = json.load(open(log)) if os.path.exists(log) else []
existing.append(rows)
json.dump(existing, open(log, "w"))

cut = float(os.environ.get("STUB_TERMINATE_ABOVE", "1e9"))
rz, rt, msgs = [], [], []
for row in reversed(rows):
    n = row["n_turns"] if row["R"] < cut else 3
    rz.append("\\n".join("  %.8e  %.8e" % (row["R"] + i, row["Z"] + i) for i in range(n)))
    rt.append("\\n".join("  %.8e  %.8e" % (0.5 + i * 1e-6, 0.1 + i * 1e-6) for i in range(n)))
    msgs.append(" => Line%6d:%6d points" % (row["nr"], n))

open("poinc_R-Z.dat", "w").write(" # R Z\\n" + "\\n\\n\\n".join(rz) + "\\n\\n\\n")
open("poinc_rho-theta.dat", "w").write(" # rho theta\\n" + "\\n\\n\\n".join(rt) + "\\n\\n\\n")
sys.stdout.write("\\n".join(msgs))
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
def scan(tmp_path, monkeypatch):
    """A run folder with stub tools, plus a handle on what they were asked to do."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # run_tool invokes exe_dir/<tool> verbatim and Windows cannot execute an
    # extensionless file, so the stubs carry .cmd there and the module's tool
    # names are pointed at them.
    _install(run_dir, "jorek2_postproc", _POSTPROC)
    _install(run_dir, "jorek2_poincare", _POINCARE)
    monkeypatch.setattr(poinc, "POSTPROC_TOOL", f"jorek2_postproc{_EXT}")
    monkeypatch.setattr(poinc, "POINCARE_TOOL", f"jorek2_poincare{_EXT}")

    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000200.h5").write_bytes(b"fake")

    log = tmp_path / "stpts_log.json"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_PAD_WIDTH", "6")

    run = Jorek2Run(
        run_dir=run_dir, exe_dir=run_dir,
        namelist=run_dir / "in_main", pad_width=6, profiles=(),
    )
    return run, RunPaths(run_dir, pad_width=6), log


def batches(log: Path) -> list[list[dict]]:
    """Every stpts the tool was handed, in order."""
    return json.loads(log.read_text(encoding="utf-8")) if log.exists() else []


def step_once(scan, psi_n_list, n_turns, ang=2, **kw):
    run, paths, _ = scan
    return poinc.run_poincare_step(
        run, paths, 200, psi_n_list,
        ang_sample_freq=ang, n_turns=n_turns, **kw,
    )


# --- batching -----------------------------------------------------------------------


def test_a_whole_step_is_one_invocation(scan):
    """The point of the rewrite: 3 psi_n x 4 samples used to be 12 separate
    processes, each re-copying the restart file."""
    _, _, log = scan
    report = step_once(scan, [0.2, 0.3, 0.4], 100, ang=4)

    assert len(batches(log)) == 1
    assert len(batches(log)[0]) == 12
    assert (report.traced, report.extended, report.cached) == (12, 0, 0)


def test_all_lines_land_in_the_cache_with_the_right_lengths(scan):
    run, paths, _ = scan
    step_once(scan, [0.2, 0.3], 50, ang=2)

    records = pc.read_cache(paths.poincare_cache(200))
    assert len(records) == 4
    assert all(r.n_points == 50 and r.n_turns == 50 for r in records.values())
    assert all(r.n_segments == 1 and not r.terminated for r in records.values())


def test_blocks_are_matched_to_lines_not_to_positions(scan):
    """The stub emits its blocks reversed. Each cached line's first R sample is
    its own start point, so a positional mis-assignment would swap them."""
    run, paths, _ = scan
    step_once(scan, [0.2, 0.3], 20, ang=2)

    for key, record in pc.read_cache(paths.poincare_cache(200)).items():
        assert record.R[0] == pytest.approx(key.R, rel=1e-6)


# --- re-running costs nothing ---------------------------------------------------------


def test_an_unchanged_request_traces_nothing(scan):
    _, _, log = scan
    step_once(scan, [0.2, 0.3], 50)
    report = step_once(scan, [0.2, 0.3], 50)

    assert len(batches(log)) == 1  # the tool was not invoked a second time
    assert (report.traced, report.extended) == (0, 0)
    assert report.cached == 4


def test_force_discards_and_retraces(scan):
    _, _, log = scan
    step_once(scan, [0.2], 50)
    report = step_once(scan, [0.2], 50, force=True)

    assert len(batches(log)) == 2
    assert report.traced == 2


# --- widening psi_n_in -----------------------------------------------------------------


def test_adding_a_psi_n_traces_only_the_new_positions(scan):
    """George's first complaint: going from [0.1, 0.2] to [0.1, 0.2, 0.3]."""
    _, paths, log = scan
    step_once(scan, [0.1, 0.2], 50)
    report = step_once(scan, [0.1, 0.2, 0.3], 50)

    assert report.traced == 2  # one new psi_n x ang_sample_freq=2
    assert report.cached == 4
    second = batches(log)[1]
    assert len(second) == 2
    assert all(row["n_turns"] == 50 for row in second)
    assert len(pc.read_cache(paths.poincare_cache(200))) == 6


def test_existing_lines_are_untouched_when_a_psi_n_is_added(scan):
    _, paths, _ = scan
    step_once(scan, [0.1], 50)
    before = pc.read_cache(paths.poincare_cache(200))

    step_once(scan, [0.1, 0.9], 50)
    after = pc.read_cache(paths.poincare_cache(200))

    for key, record in before.items():
        np.testing.assert_array_equal(after[key].R, record.R)
        assert after[key].n_turns == record.n_turns


def test_dropping_a_psi_n_does_not_delete_it(scan):
    """Narrowing a request is not a request to throw data away."""
    _, paths, log = scan
    step_once(scan, [0.1, 0.2], 50)
    report = step_once(scan, [0.1], 50)

    assert (report.traced, report.extended) == (0, 0)
    assert len(batches(log)) == 1
    assert len(pc.read_cache(paths.poincare_cache(200))) == 4


# --- extending n_turns -------------------------------------------------------------------


def test_raising_n_turns_traces_only_the_difference(scan):
    """George's second complaint: 'if I just want to trace another 1000 turns
    I also have to start all over'."""
    _, paths, log = scan
    step_once(scan, [0.2], 1000)
    report = step_once(scan, [0.2], 3000)

    assert (report.traced, report.extended) == (0, 2)
    assert all(row["n_turns"] == 2000 for row in batches(log)[1])

    for record in pc.read_cache(paths.poincare_cache(200)).values():
        assert record.n_turns == 3000
        assert record.n_points == 3000
        assert record.n_segments == 2


def test_an_extension_resumes_from_the_last_puncture(scan):
    """Not from the original start point -- every puncture lies on the same
    toroidal plane, so the last one is a valid continuation."""
    _, paths, log = scan
    step_once(scan, [0.2], 10)
    cached = pc.read_cache(paths.poincare_cache(200))
    step_once(scan, [0.2], 20)

    resumed = sorted(row["R"] for row in batches(log)[1])
    last_punctures = sorted(float(r.R[-1]) for r in cached.values())
    original_starts = sorted(k.R for k in cached)

    assert resumed == pytest.approx(last_punctures, rel=1e-6)
    # The stub advances R by 1 per turn, so after 10 turns the two are far apart.
    assert resumed != pytest.approx(original_starts, rel=1e-3)


def test_the_original_samples_survive_an_extension(scan):
    _, paths, _ = scan
    step_once(scan, [0.2], 10)
    before = pc.read_cache(paths.poincare_cache(200))
    step_once(scan, [0.2], 25)
    after = pc.read_cache(paths.poincare_cache(200))

    for key, record in before.items():
        np.testing.assert_array_equal(after[key].R[:10], record.R)


def test_lowering_n_turns_traces_nothing(scan):
    _, _, log = scan
    step_once(scan, [0.2], 1000)
    report = step_once(scan, [0.2], 500)

    assert (report.traced, report.extended) == (0, 0)
    assert len(batches(log)) == 1


def test_widening_and_extending_at_once_is_one_batch(scan):
    """A mixed batch is exactly what per-line n_turns (stpts column 5) is for
    -- new lines want the full count, resumed ones only the shortfall."""
    _, _, log = scan
    step_once(scan, [0.2], 100)
    report = step_once(scan, [0.2, 0.5], 300)

    assert (report.traced, report.extended) == (2, 2)
    assert len(batches(log)) == 2
    turns = sorted(row["n_turns"] for row in batches(log)[1])
    assert turns == [200, 200, 300, 300]  # 2 resumed shortfalls, 2 fresh


# --- lines that leave the mesh -------------------------------------------------------------


def test_a_terminated_line_is_recorded_and_never_resumed(scan, monkeypatch):
    """The stub cuts any line starting above R=1.95 short at 3 points, standing
    in for `exit L_IT` when a field line leaves the mesh. At psi_n=0.2 the four
    sampled start points are R = 1.70, 1.80, 1.90, 2.01, so exactly one is
    cut short."""
    _, paths, log = scan
    monkeypatch.setenv("STUB_TERMINATE_ABOVE", "1.95")
    step_once(scan, [0.2], 100, ang=4)

    records = pc.read_cache(paths.poincare_cache(200))
    assert sum(r.terminated for r in records.values()) == 1

    report = step_once(scan, [0.2], 500, ang=4)
    assert report.extended == 3  # the terminated line is left alone
    assert all(row["n_turns"] == 400 for row in batches(log)[1])
    assert len(batches(log)[1]) == 3


# --- flux surfaces ---------------------------------------------------------------------------


def test_flux_surfaces_are_not_regenerated_for_cached_psi_n(scan):
    """The cache records the sampled positions, which was the only reason to
    keep the (large) flux-surface file around."""
    run, paths, _ = scan
    step_once(scan, [0.2, 0.3], 50)
    step_once(scan, [0.2, 0.3, 0.4], 50)

    assert not list(paths.postproc_dir.glob("fluxsurface_*"))


def test_flux_surfaces_are_cleaned_up(scan):
    run, paths, _ = scan
    step_once(scan, [0.2], 50)
    assert not list(paths.postproc_dir.glob("fluxsurface_*"))


def test_scratch_directory_is_removed(scan):
    run, paths, _ = scan
    step_once(scan, [0.2], 50)
    assert not list(paths.poinc_dir.glob("_scratch*"))


# --- the scan wrapper -------------------------------------------------------------------------


def test_scan_reports_one_entry_per_step(scan):
    run, paths, _ = scan
    (run.run_dir / "jorek000400.h5").write_bytes(b"fake")

    reports = poinc.run_poincare_scan(
        run, paths, [200, 400], [0.2],
        ang_sample_freq=2, n_turns=50, n_workers=1, omp_threads=2,
    )

    assert [r.step for r in reports] == [200, 400]
    assert all(r.traced == 2 for r in reports)
    assert paths.poincare_cache(400).is_file()


def test_each_step_gets_its_own_cache_file(scan):
    """One file per step is what makes step-level process parallelism safe
    without any locking."""
    run, paths, _ = scan
    (run.run_dir / "jorek000400.h5").write_bytes(b"fake")
    poinc.run_poincare_scan(
        run, paths, [200, 400], [0.2],
        ang_sample_freq=2, n_turns=50, n_workers=1, omp_threads=1,
    )
    assert paths.poincare_cache(200) != paths.poincare_cache(400)
    assert len(list(paths.poinc_dir.glob("poinc_s*.h5"))) == 2
