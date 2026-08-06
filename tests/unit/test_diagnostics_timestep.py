"""ashen.diagnostics.timestep -- step_time, exercised against a stand-in
run_zero_d (no real jorek2_postproc needed)."""

from __future__ import annotations

import pytest

from ashen.diagnostics import timestep as timestep_mod
from ashen.diagnostics.timestep import step_time
from ashen.jorek2 import Jorek2Run
from ashen.paths import RunPaths


@pytest.fixture
def run(tmp_path) -> Jorek2Run:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path / "run", pad_width=6)


def _stub_run_zero_d(monkeypatch, paths, *, time_si: float, time_jorek: float):
    """Writes both unit-system caches directly (skipping subprocess/jorek2_postproc
    entirely) and returns their paths, matching run_zero_d's own return
    convention."""
    def fake(run, step, paths_arg, *, si_units=True):
        out = paths_arg.zero_d(step, si_units=si_units)
        out.parent.mkdir(parents=True, exist_ok=True)
        value = time_si if si_units else time_jorek
        out.write_text(f"Time\n{value}\n", encoding="utf-8")
        return out

    monkeypatch.setattr(timestep_mod, "run_zero_d", fake)


def test_step_time_reads_both_unit_systems(run, paths, monkeypatch):
    _stub_run_zero_d(monkeypatch, paths, time_si=1.5e-4, time_jorek=452.3)

    result = step_time(run, paths, 100)

    assert result.step == 100
    assert result.time_si == pytest.approx(1.5e-4)
    assert result.time_jorek == pytest.approx(452.3)


def test_step_time_writes_distinct_cache_files(run, paths, monkeypatch):
    _stub_run_zero_d(monkeypatch, paths, time_si=1.0, time_jorek=2.0)

    step_time(run, paths, 100)

    assert paths.zero_d(100, si_units=True).is_file()
    assert paths.zero_d(100, si_units=False).is_file()
    assert paths.zero_d(100, si_units=True) != paths.zero_d(100, si_units=False)
