"""Tests for ashen.cli.analyse's zeroD gathering: cache-gating, fan-out, and
the missing-restart warn-and-continue behaviour.

Exercises ``_gather_zero_d`` directly (not the full CLI) since it needs only
a ``Jorek2Run``/``RunPaths`` pair and a stand-in ``run_zero_d`` -- no real
JOREK, no cases.toml plumbing.
"""

from __future__ import annotations

import pytest

from ashen.cases import Case
from ashen.cli import analyse as analyse_cli
from ashen.jorek2 import Jorek2Run, MissingRestartError
from ashen.paths import RunPaths


@pytest.fixture
def jrun_and_paths(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)
    return run, paths


def test_missing_restart_is_warned_and_skipped(jrun_and_paths, monkeypatch):
    """A step with no restart file yet must not abort the rest of the range --
    ports the behaviour George asked for: warn and move on to the next step."""
    run, paths = jrun_and_paths

    def fake_run_zero_d(jrun, step, paths):
        if step == 200:
            raise MissingRestartError(f"restart file not found: step {step}")
        paths.zero_d(step).parent.mkdir(parents=True, exist_ok=True)
        paths.zero_d(step).write_text("Time Energy\n1.0 1.0\n", encoding="utf-8")

    monkeypatch.setattr(analyse_cli, "run_zero_d", fake_run_zero_d)

    with pytest.warns(UserWarning, match="skipping zerod step 200"):
        analyse_cli._gather_zero_d(run, paths, [100, 200, 300], force=False, n_workers=1)

    assert paths.zero_d(100).is_file()
    assert not paths.zero_d(200).is_file()
    assert paths.zero_d(300).is_file()


def test_cached_steps_are_not_recomputed(jrun_and_paths, monkeypatch, capsys):
    run, paths = jrun_and_paths
    paths.zero_d(100).parent.mkdir(parents=True, exist_ok=True)
    paths.zero_d(100).write_text("Time Energy\n1.0 1.0\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        analyse_cli, "run_zero_d", lambda jrun, step, paths: calls.append(step)
    )

    analyse_cli._gather_zero_d(run, paths, [100, 200], force=False, n_workers=1)

    assert calls == [200]
    assert "[cached]" in capsys.readouterr().out


def test_force_recomputes_cached_steps(jrun_and_paths, monkeypatch):
    run, paths = jrun_and_paths
    paths.zero_d(100).parent.mkdir(parents=True, exist_ok=True)
    paths.zero_d(100).write_text("Time Energy\n1.0 1.0\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        analyse_cli, "run_zero_d", lambda jrun, step, paths: calls.append(step)
    )

    analyse_cli._gather_zero_d(run, paths, [100], force=True, n_workers=1)

    assert calls == [100]


# --- _gather_qprofile: same cache-gating/fan-out shape, mirrored onto the
# q-profile gather that rides along with `--diag four`. ------------------


def test_qprofile_missing_restart_is_warned_and_skipped(jrun_and_paths, monkeypatch):
    run, paths = jrun_and_paths

    def fake_run_qprofile_step(jrun, step, paths):
        if step == 200:
            raise MissingRestartError(f"restart file not found: step {step}")
        paths.qprofile(step).parent.mkdir(parents=True, exist_ok=True)
        paths.qprofile(step).write_text("# Psi_n q\n# time step #000100\n0.5 1.0\n\n", encoding="utf-8")

    monkeypatch.setattr(
        analyse_cli.qprofile_diag, "run_qprofile_step", fake_run_qprofile_step
    )

    with pytest.warns(UserWarning, match="skipping qprofile step 200"):
        analyse_cli._gather_qprofile(run, paths, [100, 200, 300], force=False, n_workers=1)

    assert paths.qprofile(100).is_file()
    assert not paths.qprofile(200).is_file()
    assert paths.qprofile(300).is_file()


def test_qprofile_cached_steps_are_not_recomputed(jrun_and_paths, monkeypatch, capsys):
    run, paths = jrun_and_paths
    paths.qprofile(100).parent.mkdir(parents=True, exist_ok=True)
    paths.qprofile(100).write_text("# Psi_n q\n# time step #000100\n0.5 1.0\n\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        analyse_cli.qprofile_diag, "run_qprofile_step",
        lambda jrun, step, paths: calls.append(step),
    )

    analyse_cli._gather_qprofile(run, paths, [100, 200], force=False, n_workers=1)

    assert calls == [200]
    assert "[cached]" in capsys.readouterr().out


def test_qprofile_force_recomputes_cached_steps(jrun_and_paths, monkeypatch):
    run, paths = jrun_and_paths
    paths.qprofile(100).parent.mkdir(parents=True, exist_ok=True)
    paths.qprofile(100).write_text("# Psi_n q\n# time step #000100\n0.5 1.0\n\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        analyse_cli.qprofile_diag, "run_qprofile_step",
        lambda jrun, step, paths: calls.append(step),
    )

    analyse_cli._gather_qprofile(run, paths, [100], force=True, n_workers=1)

    assert calls == [100]


# --- poincare_highlight: qprofile gathered alongside poincare --------------------


@pytest.fixture
def poincare_case_and_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "myrun"
    run_dir.mkdir()
    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000100.h5").write_bytes(b"")
    (run_dir / "real_psi_edge.dat").write_text("1.0\n", encoding="utf-8")
    return run_dir


def test_poincare_highlight_gathers_qprofile_alongside_poincare(
    poincare_case_and_run_dir, monkeypatch
):
    run_dir = poincare_case_and_run_dir
    case = Case(
        name="myrun", steps=[100], psi_n_in=[0.5],
        poincare_highlight=True, poincare_highlight_modes=[[2, 1]],
        poincare_highlight_colors=["red"],
    )

    calls = []
    monkeypatch.setattr(analyse_cli, "run_zero_d", lambda *a, **k: None)
    monkeypatch.setattr(
        analyse_cli, "_gather_qprofile",
        lambda *a, **k: calls.append("qprofile"),
    )
    monkeypatch.setattr(
        analyse_cli.poincare_diag, "run_poincare_scan",
        lambda *a, **k: calls.append("poincare") or [],
    )

    analyse_cli._run_case(case, diags=["poincare"], force=False, n_workers=1, omp_threads=1)

    assert calls == ["qprofile", "poincare"]


def test_poincare_without_highlight_does_not_gather_qprofile(
    poincare_case_and_run_dir, monkeypatch
):
    run_dir = poincare_case_and_run_dir
    case = Case(name="myrun", steps=[100], psi_n_in=[0.5])

    calls = []
    monkeypatch.setattr(analyse_cli, "run_zero_d", lambda *a, **k: None)
    monkeypatch.setattr(
        analyse_cli, "_gather_qprofile",
        lambda *a, **k: calls.append("qprofile"),
    )
    monkeypatch.setattr(
        analyse_cli.poincare_diag, "run_poincare_scan",
        lambda *a, **k: [],
    )

    analyse_cli._run_case(case, diags=["poincare"], force=False, n_workers=1, omp_threads=1)

    assert calls == []


# --- profiles: _run_case's per-mode never-succeeded warning ----------------------


@pytest.fixture
def case_and_run_dir(tmp_path, monkeypatch):
    """A minimal on-disk run folder + Case, real enough for _run_case's
    Path.cwd()/case.name existence check, with gather_profiles itself
    always monkeypatched by the individual test."""
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "myrun"
    run_dir.mkdir()
    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000100.h5").write_bytes(b"")
    (run_dir / "jorek000200.h5").write_bytes(b"")
    case = Case(name="myrun", steps=[100, 200], vars=["currdens"])
    return case, run_dir


def test_profiles_warns_when_a_mode_never_succeeds(case_and_run_dir, monkeypatch, capsys):
    case, run_dir = case_and_run_dir
    monkeypatch.setattr(
        analyse_cli.profiles_diag, "gather_profiles",
        lambda *a, **k: {"midplane": 2, "average": 0},
    )

    analyse_cli._run_case(case, diags=["profiles"], force=False, n_workers=1, omp_threads=1)

    out = capsys.readouterr().out
    assert "tor_mode 'average' produced no profiles at all" in out
    assert "tor_mode 'midplane' produced no profiles" not in out


def test_profiles_average_failure_hint_only_shown_for_average(case_and_run_dir, monkeypatch, capsys):
    case, run_dir = case_and_run_dir
    monkeypatch.setattr(
        analyse_cli.profiles_diag, "gather_profiles",
        lambda *a, **k: {"average": 0},
    )

    analyse_cli._run_case(case, diags=["profiles"], force=False, n_workers=1, omp_threads=1)

    assert "traces field lines and dies" in capsys.readouterr().out


def test_profiles_no_warning_when_every_mode_succeeds(case_and_run_dir, monkeypatch, capsys):
    case, run_dir = case_and_run_dir
    monkeypatch.setattr(
        analyse_cli.profiles_diag, "gather_profiles",
        lambda *a, **k: {"midplane": 2, "average": 1},
    )

    analyse_cli._run_case(case, diags=["profiles"], force=False, n_workers=1, omp_threads=1)

    assert "produced no profiles" not in capsys.readouterr().out


def test_profiles_passes_case_knobs_through_to_gather_profiles(case_and_run_dir, monkeypatch):
    case, run_dir = case_and_run_dir
    captured = {}

    def fake_gather(jrun, paths, steps, variables, **kwargs):
        captured.update(kwargs)
        captured["steps"] = steps
        captured["variables"] = variables
        return {m: 1 for m in kwargs["tor_modes"]}

    monkeypatch.setattr(analyse_cli.profiles_diag, "gather_profiles", fake_gather)

    analyse_cli._run_case(case, diags=["profiles"], force=False, n_workers=1, omp_threads=1)

    assert captured["coords_var"] == case.coords_var
    assert captured["tor_modes"] == case.tor_mode
    assert captured["surfaces"] == case.profile_surfaces
    assert captured["rad_range"] == tuple(case.profile_rad_range)
    assert captured["nmaxsteps"] == case.profile_nmaxsteps
    assert captured["deltaphi"] == case.profile_deltaphi
    assert captured["steps"] == case.steps
    assert captured["variables"] == case.vars
