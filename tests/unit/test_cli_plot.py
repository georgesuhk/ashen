"""ashen.cli.plot -- end-to-end against a synthetic run folder + cases.toml,
no JOREK needed. Exercises the wiring: cases.toml -> RunPaths -> the
Poincare cache -> the plotting functions -> files on disk.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ashen.cli import plot as plot_cli
from ashen.diagnostics import four_cache as four_cache_mod
from ashen.diagnostics import poincare_cache as pc
from ashen.paths import RunPaths, write_float

pytest.importorskip("h5py")


def _write_four_cache(run_dir, step, *, records):
    paths = RunPaths(run_dir, pad_width=6)
    four_cache_mod.write_cache(
        paths.four_cache(step), step=step, pad_width=6, records=records
    )


def _four_record(variable, n, m, *, real_peak):
    psi_n = np.linspace(0.0, 1.0, 4)
    real = np.array([0.1, real_peak, 0.2, 0.05], dtype=np.float32)
    return four_cache_mod.FourRecord(
        variable=variable, n=n, m=m, psi_n=psi_n, real=real, imag=np.zeros(4, dtype=np.float32)
    )


def _write_cache(run_dir, step, *, pad_width=6):
    path = run_dir / "poinc_dir" / f"poinc_s{step:0{pad_width}d}.h5"
    with pc.open_cache(path, step=step, pad_width=pad_width) as h:
        for psi_n, R in [(0.2, 1.7), (0.5, 1.8)]:
            key = pc.LineKey(psi_n=psi_n, R=R, Z=0.0, phi=0.0)
            n = 10
            pc.append_line(
                h, key,
                {
                    "R": np.full(n, R, dtype=np.float32),
                    "Z": np.zeros(n, dtype=np.float32),
                    "rho": np.sqrt(np.full(n, psi_n, dtype=np.float32)),
                    "theta": np.zeros(n, dtype=np.float32),
                },
                n_turns=n, terminated=False,
            )
    return path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    run_dir = tmp_path / "qa2.1_g2.3" / "eta1e-3_RE"
    run_dir.mkdir(parents=True)
    (run_dir / "jorek000100.h5").write_bytes(b"")
    (run_dir / "jorek000200.h5").write_bytes(b"")
    write_float(run_dir / "real_psi_edge.dat", 1.0)
    (run_dir / "log").write_text("R_axis = 1.363245\n", encoding="utf-8")
    (run_dir / "postproc").mkdir()
    for step, t in [(100, 1e-4), (200, 2e-4)]:
        (run_dir / "postproc" / f"zeroD_quantities_s{step:06d}.dat").write_text(
            f"Time Energy\n{t} 1.0\n", encoding="utf-8"
        )
        _write_cache(run_dir, step)

    cases_toml = tmp_path / "cases.toml"
    cases_toml.write_text(
        '[cases.test]\n'
        'folder = "qa2.1_g2.3/eta1e-3_RE"\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return run_dir


def test_list_shows_defined_cases(campaign, capsys):
    assert plot_cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "test:" in out


def test_poincare_diag_writes_one_file_per_step(campaign):
    assert plot_cli.main(["--case", "test", "--diag", "poincare"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert (campaign / "poinc_dir" / "200_poincare.png").is_file()


def test_connection_length_diag_writes_lc_and_lctt(campaign):
    assert plot_cli.main(["--case", "test", "--diag", "connection_length"]) == 0
    files = list((campaign / "poinc_dir").glob("L*_*.png"))
    assert any(f.name.startswith("LC_") for f in files)
    assert any(f.name.startswith("LCTT_") for f in files)


def test_step_filter_restricts_poincare_output(campaign):
    assert plot_cli.main(["--case", "test", "--diag", "poincare", "--step", "100"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert not (campaign / "poinc_dir" / "200_poincare.png").is_file()


def test_unknown_case_is_an_error(campaign, capsys):
    assert plot_cli.main(["--case", "does_not_exist"]) == 1
    assert "unknown case" in capsys.readouterr().out


def test_missing_run_folder_is_reported(tmp_path, monkeypatch):
    (tmp_path / "cases.toml").write_text(
        '[cases.ghost]\nfolder = "nope"\nsteps = [1]\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert plot_cli.main(["--case", "ghost"]) == 1


def test_missing_log_is_reported_not_raised(campaign, capsys):
    """r_axis raises LogfileError on a missing/unparseable log -- the CLI
    must catch it and continue, not crash the whole run."""
    (campaign / "log").write_text("no axis here\n", encoding="utf-8")
    assert plot_cli.main(["--case", "test", "--diag", "connection_length"]) == 0
    assert "error" in capsys.readouterr().out.lower()


def test_default_diags_run_both(campaign):
    assert plot_cli.main(["--case", "test"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert list((campaign / "poinc_dir").glob("LC_*.png"))


# --- psi range restriction for connection_length ----------------------------------------


def _spy_on_matrix_targets(monkeypatch, captured):
    """Records the psi_n_targets connection_length_matrix is actually called
    with, so the tests can assert on what got filtered without reaching into
    the plotted figure itself."""
    original = plot_cli.connection_length_matrix

    def spy(records_by_step, steps, psi_n_targets, **kwargs):
        captured["targets"] = psi_n_targets
        return original(records_by_step, steps, psi_n_targets, **kwargs)

    monkeypatch.setattr(plot_cli, "connection_length_matrix", spy)


def test_cli_psi_range_filters_plotted_psi_n(campaign, monkeypatch):
    captured = {}
    _spy_on_matrix_targets(monkeypatch, captured)
    assert plot_cli.main(
        ["--case", "test", "--diag", "connection_length", "--psi-range", "0.4", "0.6"]
    ) == 0
    assert captured["targets"] == [0.5]


def test_cli_psi_range_with_no_matches_reports_error(campaign, capsys):
    assert plot_cli.main(
        ["--case", "test", "--diag", "connection_length", "--psi-range", "0.6", "0.9"]
    ) == 0
    assert "no psi_n_in within range" in capsys.readouterr().out


def test_case_lc_psi_n_in_is_used_without_cli_override(campaign, monkeypatch):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases.test]\n'
        'folder = "qa2.1_g2.3/eta1e-3_RE"\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        'lc_psi_n_in = [0.5]\n',
        encoding="utf-8",
    )
    captured = {}
    _spy_on_matrix_targets(monkeypatch, captured)
    assert plot_cli.main(["--case", "test", "--diag", "connection_length"]) == 0
    assert captured["targets"] == [0.5]


def test_cli_psi_range_overrides_case_lc_psi_n_in(campaign, monkeypatch):
    """--psi-range further narrows whatever lc_psi_n_in already resolved to,
    rather than replacing it outright."""
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases.test]\n'
        'folder = "qa2.1_g2.3/eta1e-3_RE"\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        'lc_psi_n_in = [0.2, 0.5]\n',
        encoding="utf-8",
    )
    captured = {}
    _spy_on_matrix_targets(monkeypatch, captured)
    assert plot_cli.main(
        ["--case", "test", "--diag", "connection_length", "--psi-range", "0.4", "0.6"]
    ) == 0
    assert captured["targets"] == [0.5]


# --- jorek2_four mode-amplitude time series ---------------------------------------------


def test_four_diag_writes_one_file_per_variable(campaign):
    _write_four_cache(campaign, 100, records=[
        _four_record("Psi", 0, 1, real_peak=1.0),
        _four_record("u", 0, 0, real_peak=2.0),
    ])
    _write_four_cache(campaign, 200, records=[
        _four_record("Psi", 0, 1, real_peak=1.5),
        _four_record("u", 0, 0, real_peak=2.5),
    ])
    assert plot_cli.main(["--case", "test", "--diag", "four"]) == 0
    # The campaign fixture writes a zeroD cache for every step, so both the
    # step-indexed and true-time variants are written.
    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()
    assert (campaign / "four_dir" / "u_modes_step.png").is_file()
    assert (campaign / "four_dir" / "Psi_modes_time.png").is_file()
    assert (campaign / "four_dir" / "u_modes_time.png").is_file()


def test_four_diag_with_no_cache_reports_and_does_not_crash(campaign, capsys):
    assert plot_cli.main(["--case", "test", "--diag", "four"]) == 0
    assert "no jorek2_four cache found" in capsys.readouterr().out


def test_four_diag_skips_time_variant_without_zerod_cache(campaign, capsys):
    """A step missing from the zeroD cache must not produce a partial/wrong
    time axis -- the time variant is skipped outright, with the step variant
    still written."""
    (campaign / "postproc" / "zeroD_quantities_s000200.dat").unlink()
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 0, 1, real_peak=1.5)])

    assert plot_cli.main(["--case", "test", "--diag", "four"]) == 0

    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "Psi_modes_time.png").exists()
    assert "skipping time-axis four-mode plots" in capsys.readouterr().out


def test_four_modes_are_m_n_pairs_not_n_m(campaign, monkeypatch):
    """case.four_modes entries are [m, n] (poloidal, toroidal) -- [3, 2]
    means m=3, n=2. The diagnostics layer's modes= filter is (n, m), so the
    CLI must swap the pair, not pass it through as-is."""
    captured = {}
    original = plot_cli.max_amplitude_series

    def spy(paths, steps, **kwargs):
        captured["modes"] = kwargs.get("modes")
        return original(paths, steps, **kwargs)

    monkeypatch.setattr(plot_cli, "max_amplitude_series", spy)

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases.test]\n'
        'folder = "qa2.1_g2.3/eta1e-3_RE"\n'
        'steps = [100, 200]\n'
        'four_modes = [[3, 2]]\n',  # m=3, n=2
        encoding="utf-8",
    )
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 2, 3, real_peak=1.0)])

    assert plot_cli.main(["--case", "test", "--diag", "four"]) == 0
    assert captured["modes"] == [(2, 3)]  # (n, m)


def test_four_vars_filters_which_files_are_written(campaign):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases.test]\n'
        'folder = "qa2.1_g2.3/eta1e-3_RE"\n'
        'steps = [100, 200]\n'
        'four_vars = ["Psi"]\n',
        encoding="utf-8",
    )
    _write_four_cache(campaign, 100, records=[
        _four_record("Psi", 0, 1, real_peak=1.0),
        _four_record("u", 0, 0, real_peak=2.0),
    ])
    assert plot_cli.main(["--case", "test", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "u_modes_step.png").exists()
    assert (campaign / "four_dir" / "Psi_modes_time.png").is_file()
    assert not (campaign / "four_dir" / "u_modes_time.png").exists()
