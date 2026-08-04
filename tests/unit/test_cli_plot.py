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
from ashen.diagnostics import poincare_cache as pc
from ashen.paths import write_float

pytest.importorskip("h5py")


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
