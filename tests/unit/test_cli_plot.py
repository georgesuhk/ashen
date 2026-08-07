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
from ashen.diagnostics import profiles as profiles_mod
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


def _write_crossing_cache(run_dir, step, *, pad_width=6):
    """A cache whose lines actually cross psi_n=1 (unlike `_write_cache`,
    whose constant-psi_n lines never do) -- for theta_hist tests."""
    path = run_dir / "poinc_dir" / f"poinc_s{step:0{pad_width}d}.h5"
    with pc.open_cache(path, step=step, pad_width=pad_width) as h:
        for psi_n, R, theta in [(0.2, 1.7, 0.5), (0.5, 1.8, -0.5)]:
            key = pc.LineKey(psi_n=psi_n, R=R, Z=0.0, phi=0.0)
            n = 3
            pc.append_line(
                h, key,
                {
                    "R": np.full(n, R, dtype=np.float32),
                    "Z": np.zeros(n, dtype=np.float32),
                    "rho": np.sqrt(np.array([0.3, 1.5, 1.6], dtype=np.float32)),
                    "theta": np.array([0.0, theta, theta + 0.1], dtype=np.float32),
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
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return run_dir


def test_list_shows_defined_cases(campaign, capsys):
    assert plot_cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "qa2.1_g2.3/eta1e-3_RE (" in out


def test_poincare_diag_writes_one_file_per_step(campaign):
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert (campaign / "poinc_dir" / "200_poincare.png").is_file()


def test_connection_length_diag_writes_lc_and_lctt(campaign):
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length"]) == 0
    files = list((campaign / "poinc_dir").glob("L*_*.png"))
    assert any(f.name.startswith("LC_") for f in files)
    assert any(f.name.startswith("LCTT_") for f in files)


def test_step_filter_restricts_poincare_output(campaign):
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare", "--step", "100"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert not (campaign / "poinc_dir" / "200_poincare.png").is_file()


# --- poincare rational-surface highlight -------------------------------------------


def _write_btor_profile(run_dir, *, psi_n, btor, step=0, pad_width=6):
    """The step-0 midplane-outer Btor profile delta_b_over_b normalises
    against -- see ashen.diagnostics.profiles.edge_toroidal_field."""
    paths = RunPaths(run_dir, pad_width=pad_width)
    cache = paths.profile_cache("Psi_N", "Btor", step, "midplane outer")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=np.asarray(psi_n, dtype=float), y=np.asarray(btor, dtype=float))


def _write_qprofile_cache(run_dir, step, *, psi_n, q, pad_width=6):
    paths = RunPaths(run_dir, pad_width=pad_width)
    lines = ["# Psi_n q", f"# time step #{step:0{pad_width}d}"]
    lines += [f"{p} {v}" for p, v in zip(psi_n, q)]
    paths.qprofile(step).parent.mkdir(parents=True, exist_ok=True)
    paths.qprofile(step).write_text("\n".join(lines) + "\n\n", encoding="utf-8")


def _add_highlight_case(tmp_path, *, modes, colors):
    cases_toml = tmp_path / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        'poincare_highlight = true\n'
        f'poincare_highlight_modes = {modes}\n'
        f'poincare_highlight_colors = {colors}\n',
        encoding="utf-8",
    )


def test_poincare_highlight_colors_the_matched_line(campaign, monkeypatch):
    # q = 1 + 2*psi_n crosses q=2.0 (m=2, n=1) exactly at psi_n=0.5, which is
    # one of the two traced surfaces -- an exact match, not just "nearest".
    _write_qprofile_cache(
        campaign, 100, psi_n=[0.0, 0.25, 0.5, 0.75, 1.0], q=[1.0, 1.5, 2.0, 2.5, 3.0]
    )
    _write_qprofile_cache(
        campaign, 200, psi_n=[0.0, 0.25, 0.5, 0.75, 1.0], q=[1.0, 1.5, 2.0, 2.5, 3.0]
    )
    _add_highlight_case(campaign.parent.parent, modes=[[2, 1]], colors=["red"])

    captured = {}
    real_draw = plot_cli.plot_poincare_step

    def spy(records, out, **kwargs):
        captured[out.name] = kwargs.get("highlight")
        return real_draw(records, out, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_poincare_step", spy)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare"]) == 0
    assert captured["100_poincare.png"] == {0.5: "red"}
    assert captured["200_poincare.png"] == {0.5: "red"}


def test_poincare_highlight_is_not_rescaled_by_real_psi_edge(campaign, monkeypatch):
    """Regression: `LineKey.psi_n` and the qprofile's `Psi_n` column are both
    already JOREK-normalised psi_n -- `fluxsurface` takes [0,1] and inverts
    `get_psi_n` (exec_commands.f90:3063-3068) -- so `real_psi_edge` must NOT
    be applied a second time at plot time. cli/analyse.py already applies it
    once when turning case.psi_n_in into traced positions.

    real_psi_edge is deliberately != 1 so the correct and buggy behaviours
    differ: traced surfaces are 0.2 and 0.5, and q crosses 2.0 exactly at
    0.5. Correct -> highlights 0.5. The old double-normalised code divided
    the grid to {0.4, 1.0}, snapped the 0.5 crossing to 0.4, and highlighted
    0.2 instead.
    """
    write_float(campaign / "real_psi_edge.dat", 0.5)
    _write_qprofile_cache(campaign, 100, psi_n=[0.0, 0.5, 1.0], q=[1.0, 2.0, 3.0])
    _write_qprofile_cache(campaign, 200, psi_n=[0.0, 0.5, 1.0], q=[1.0, 2.0, 3.0])
    _add_highlight_case(campaign.parent.parent, modes=[[2, 1]], colors=["red"])

    captured = {}
    real_draw = plot_cli.plot_poincare_step

    def spy(records, out, **kwargs):
        captured[out.name] = kwargs.get("highlight")
        return real_draw(records, out, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_poincare_step", spy)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare"]) == 0
    assert captured["100_poincare.png"] == {0.5: "red"}
    assert captured["200_poincare.png"] == {0.5: "red"}


def test_poincare_highlight_missing_qprofile_cache_is_skipped_not_crashed(
    campaign, capsys
):
    # No qprofile cache written for either step.
    _add_highlight_case(campaign.parent.parent, modes=[[2, 1]], colors=["red"])

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert "no qprofile cache" in capsys.readouterr().out


def test_poincare_highlight_false_never_reads_qprofile(campaign, monkeypatch):
    calls = []
    monkeypatch.setattr(
        plot_cli, "read_qprofile", lambda path: calls.append(path) or (np.array([]), np.array([]))
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare"]) == 0
    assert calls == []


def test_unknown_case_is_an_error(campaign, capsys):
    assert plot_cli.main(["--case", "does_not_exist"]) == 1
    assert "unknown case" in capsys.readouterr().out


def test_missing_run_folder_is_reported(tmp_path, monkeypatch):
    (tmp_path / "cases.toml").write_text(
        '[cases.ghost]\nsteps = [1]\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert plot_cli.main(["--case", "ghost"]) == 1


def test_missing_log_is_reported_not_raised(campaign, capsys):
    """r_axis raises LogfileError on a missing/unparseable log -- the CLI
    must catch it and continue, not crash the whole run."""
    (campaign / "log").write_text("no axis here\n", encoding="utf-8")
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length"]) == 0
    assert "error" in capsys.readouterr().out.lower()


def test_default_diags_run_both(campaign):
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert list((campaign / "poinc_dir").glob("LC_*.png"))


# --- zeroD auto-gathered on demand for a missing true-time axis -------------------------


def _fake_run_zero_d_writes_cache(run, step, paths, **kwargs):
    """Stands in for a real jorek2_postproc call: just writes the cache file
    a successful gather would have produced."""
    path = paths.zero_d(step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Time Energy\n{step * 1e-6} 1.0\n", encoding="utf-8")
    return path


def test_missing_zerod_is_reported_then_gathered(campaign, monkeypatch, capsys):
    (campaign / "postproc" / "zeroD_quantities_s000200.dat").unlink()
    monkeypatch.setattr(plot_cli, "run_zero_d", _fake_run_zero_d_writes_cache)

    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length"]
    ) == 0
    out = capsys.readouterr().out
    assert "zerod: missing for step(s) [200], gathering" in out
    assert "zerod: step 200 done" in out


def test_gathered_zerod_lets_the_true_time_figure_succeed(campaign, monkeypatch):
    """The whole point: a figure that used to be skipped for a missing
    zeroD cache is now actually drawn, using the freshly gathered value."""
    (campaign / "postproc" / "zeroD_quantities_s000200.dat").unlink()
    monkeypatch.setattr(plot_cli, "run_zero_d", _fake_run_zero_d_writes_cache)

    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length"]
    ) == 0
    assert list((campaign / "poinc_dir").glob("LCTT_*.png"))


def test_zerod_gather_failure_is_reported_and_skipped_not_crashed(
    campaign, monkeypatch, capsys
):
    """jorek2_postproc failing to gather one step (no executable symlinked,
    a genuinely missing restart, etc.) must not crash the whole plot
    invocation -- same tolerance as analyse's own zerod gathering."""
    (campaign / "postproc" / "zeroD_quantities_s000200.dat").unlink()

    def fake_run_zero_d(run, step, paths, **kwargs):
        raise FileNotFoundError("jorek2_postproc not found")

    monkeypatch.setattr(plot_cli, "run_zero_d", fake_run_zero_d)

    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length"]
    ) == 0
    out = capsys.readouterr().out
    assert "zerod: step 200 skipped (jorek2_postproc not found)" in out
    assert "skipping LCTT: no zeroD cache" in out
    assert not list((campaign / "poinc_dir").glob("LCTT_*.png"))


def test_no_zerod_gathering_when_cache_already_complete(campaign, monkeypatch, capsys):
    """Every requested step already has a zeroD cache (the campaign fixture
    writes both) -- run_zero_d must never be called, and nothing about
    gathering is printed."""
    called = []
    monkeypatch.setattr(
        plot_cli, "run_zero_d", lambda *a, **k: called.append(a) or None
    )

    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length"]
    ) == 0
    assert called == []
    assert "zerod:" not in capsys.readouterr().out


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
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length", "--psi-range", "0.4", "0.6"]
    ) == 0
    assert captured["targets"] == [0.5]


def test_cli_psi_range_with_no_matches_reports_error(campaign, capsys):
    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length", "--psi-range", "0.6", "0.9"]
    ) == 0
    assert "no psi_n_in within range" in capsys.readouterr().out


def test_case_lc_psi_n_in_is_used_without_cli_override(campaign, monkeypatch):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        'lc_psi_n_in = [0.5]\n',
        encoding="utf-8",
    )
    captured = {}
    _spy_on_matrix_targets(monkeypatch, captured)
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length"]) == 0
    assert captured["targets"] == [0.5]


def test_cli_psi_range_overrides_case_lc_psi_n_in(campaign, monkeypatch):
    """--psi-range further narrows whatever lc_psi_n_in already resolved to,
    rather than replacing it outright."""
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        'lc_psi_n_in = [0.2, 0.5]\n',
        encoding="utf-8",
    )
    captured = {}
    _spy_on_matrix_targets(monkeypatch, captured)
    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "connection_length", "--psi-range", "0.4", "0.6"]
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
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    # The campaign fixture writes a zeroD cache for every step, so both the
    # step-indexed and true-time variants are written.
    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()
    assert (campaign / "four_dir" / "u_modes_step.png").is_file()
    assert (campaign / "four_dir" / "Psi_modes_time.png").is_file()
    assert (campaign / "four_dir" / "u_modes_time.png").is_file()


def test_four_diag_with_no_cache_reports_and_does_not_crash(campaign, capsys):
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert "no jorek2_four cache found" in capsys.readouterr().out


def test_four_diag_skips_time_variant_without_zerod_cache(campaign, capsys):
    """A step missing from the zeroD cache must not produce a partial/wrong
    time axis -- the time variant is skipped outright, with the step variant
    still written."""
    (campaign / "postproc" / "zeroD_quantities_s000200.dat").unlink()
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 0, 1, real_peak=1.5)])

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0

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
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_modes = [[3, 2]]\n',  # m=3, n=2
        encoding="utf-8",
    )
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 2, 3, real_peak=1.0)])

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured["modes"] == [(2, 3)]  # (n, m)


def test_four_vars_filters_which_files_are_written(campaign):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["Psi"]\n',
        encoding="utf-8",
    )
    _write_four_cache(campaign, 100, records=[
        _four_record("Psi", 0, 1, real_peak=1.0),
        _four_record("u", 0, 0, real_peak=2.0),
    ])
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "u_modes_step.png").exists()
    assert (campaign / "four_dir" / "Psi_modes_time.png").is_file()
    assert not (campaign / "four_dir" / "u_modes_time.png").exists()


# --- four_growth_rate: fit + mark down each mode's growth rate --------------------


def test_four_growth_rate_off_by_default_writes_no_summary(campaign):
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 0, 1, real_peak=1.5)])
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert not (campaign / "four_dir" / "growth_rates.txt").exists()


def test_four_growth_rate_writes_a_summary_with_the_fitted_gamma(campaign):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_growth_rate = true\n',
        encoding="utf-8",
    )
    # campaign's zeroD cache: t(100)=1e-4 s, t(200)=2e-4 s -> dt=1e-4 s.
    # exp(gamma*dt) with gamma=5000 -> exp(0.5) ~= 1.64872.
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 2, 1, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 2, 1, real_peak=1.64872)])

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0

    growth_path = campaign / "four_dir" / "growth_rates.txt"
    assert growth_path.is_file()
    text = growth_path.read_text(encoding="utf-8")
    assert "Psi" in text
    # m=1, n=2 (from the record's m=2,n=1 args, i.e. n=2,m=1 -- wait see below)
    gamma_field = text.splitlines()[1].split()[3]
    assert float(gamma_field) == pytest.approx(5000.0, rel=1e-3)


def test_four_growth_rate_skipped_without_zerod_cache(campaign, capsys):
    (campaign / "postproc" / "zeroD_quantities_s000100.dat").unlink()
    (campaign / "postproc" / "zeroD_quantities_s000200.dat").unlink()
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_growth_rate = true\n',
        encoding="utf-8",
    )
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 0, 1, real_peak=1.5)])

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0

    assert not (campaign / "four_dir" / "growth_rates.txt").exists()
    assert "skipping growth-rate fit" in capsys.readouterr().out


def test_four_growth_steps_is_passed_through_as_step_range(campaign, monkeypatch):
    captured = {}
    original = plot_cli.growth_rate_series

    def spy(series, true_times, steps, **kwargs):
        captured["step_range"] = kwargs.get("step_range")
        return original(series, true_times, steps, **kwargs)

    monkeypatch.setattr(plot_cli, "growth_rate_series", spy)

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_growth_rate = true\n'
        'four_growth_steps = [100, 150]\n',
        encoding="utf-8",
    )
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 0, 1, real_peak=1.5)])

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured["step_range"] == (100, 150)


# --- four_quantities: max vs. rational-surface amplitude --------------------------


def _spy_on_plot_mode_amplitudes(monkeypatch):
    captured = []
    original = plot_cli.plot_mode_amplitudes

    def spy(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_mode_amplitudes", spy)
    return captured


def test_four_quantities_defaults_to_max_only(campaign, monkeypatch):
    captured = _spy_on_plot_mode_amplitudes(monkeypatch)
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=1.5)])

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured
    for kwargs in captured:
        assert kwargs.get("rational_series") is None
        assert kwargs.get("ylabel") is None
        assert kwargs.get("label_suffix") == ""


def test_four_quantities_rational_surface_only_has_no_overlay_and_custom_ylabel(
    campaign, monkeypatch
):
    _write_qprofile_cache(campaign, 100, psi_n=[0.0, 0.5, 1.0], q=[1.0, 2.0, 3.0])
    _write_qprofile_cache(campaign, 200, psi_n=[0.0, 0.5, 1.0], q=[1.0, 2.0, 3.0])
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=1.5)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_quantities = ["rational_surface"]\n',
        encoding="utf-8",
    )
    captured = _spy_on_plot_mode_amplitudes(monkeypatch)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured
    for kwargs in captured:
        assert kwargs.get("rational_series") is None
        assert kwargs.get("ylabel") == "|Psi| @ rational surface"
        assert kwargs.get("label_suffix") == " @ rational surface"
    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()


def test_four_quantities_both_reproduces_max_solid_and_rational_dashed(campaign, monkeypatch):
    _write_qprofile_cache(campaign, 100, psi_n=[0.0, 0.5, 1.0], q=[1.0, 2.0, 3.0])
    _write_qprofile_cache(campaign, 200, psi_n=[0.0, 0.5, 1.0], q=[1.0, 2.0, 3.0])
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=1.5)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_quantities = ["max", "rational_surface"]\n',
        encoding="utf-8",
    )
    captured = _spy_on_plot_mode_amplitudes(monkeypatch)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured
    for kwargs in captured:
        assert kwargs.get("rational_series") is not None
        assert kwargs.get("ylabel") is None
        assert kwargs.get("label_suffix") == ""


def test_four_quantities_rational_surface_only_with_no_resonant_modes_reports_and_skips(
    campaign, capsys
):
    # n=0: q=m/n is undefined, so there is no rational surface at all.
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_quantities = ["rational_surface"]\n',
        encoding="utf-8",
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert not (campaign / "four_dir" / "Psi_modes_step.png").exists()
    assert "no rational-surface data to plot" in capsys.readouterr().out


# --- delta_b_over_b: derived from Psi, requires a step-0 Btor profile -------------


def test_delta_b_over_b_writes_a_figure_and_uses_it_over_raw_psi(campaign, monkeypatch):
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    _write_btor_profile(campaign, psi_n=[0.0, 1.0], btor=[3.0, 2.0])
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b_over_b"]\n',
        encoding="utf-8",
    )
    captured_calls = []
    original = plot_cli.plot_mode_amplitudes

    def spy(x, series, variable, out_path, **kwargs):
        captured_calls.append(series)
        return original(x, series, variable, out_path, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_mode_amplitudes", spy)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "delta_b_over_b_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "Psi_modes_step.png").exists()

    # b_ref = Btor at psi_n=1 (step 0) = 2.0; scale = m/(r_axis**2*b_ref) = 2/(4*2) = 0.25
    assert captured_calls
    np.testing.assert_allclose(captured_calls[0][("delta_b_over_b", 1, 2)], [1.0, 2.0])


def test_delta_b_over_b_not_requested_leaves_series_unaffected(campaign, monkeypatch):
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    _write_btor_profile(campaign, psi_n=[0.0, 1.0], btor=[3.0, 2.0])
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])
    captured = _spy_on_plot_mode_amplitudes(monkeypatch)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "delta_b_over_b_modes_step.png").exists()


def test_delta_b_over_b_missing_btor_profile_skips_with_message(campaign, capsys):
    # log has R_axis (from the campaign fixture) but no Btor profile cached,
    # and no step-0 restart file either -- auto-gathering can't succeed.
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b_over_b"]\n',
        encoding="utf-8",
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert "skipping delta_b_over_b" in capsys.readouterr().out
    assert not (campaign / "four_dir" / "delta_b_over_b_modes_step.png").exists()


def test_delta_b_over_b_auto_gathers_missing_btor_profile(campaign, monkeypatch):
    """The main ask this covers: plot shouldn't require a separate `analyse
    --diag profiles` pass just for delta_b_over_b's reference field -- it
    gathers that one (step, Btor, midplane outer) profile itself on demand."""
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    (campaign / "jorek000000.h5").write_bytes(b"")  # step-0 restart, for the auto-gather
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b_over_b"]\n',
        encoding="utf-8",
    )

    def fake_extract(run, step, var, coords_var, **kwargs):
        assert step == 0
        assert var == "Btor"
        assert kwargs["tor_mode"] == "midplane outer"
        return np.array([0.0, 1.0]), np.array([3.0, 2.0])

    monkeypatch.setattr(profiles_mod, "extract_profile", fake_extract)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "delta_b_over_b_modes_step.png").is_file()

    paths = RunPaths(campaign, pad_width=6)
    assert paths.profile_cache("Psi_N", "Btor", 0, "midplane outer").is_file()


def test_delta_b_over_b_auto_gather_missing_step_zero_restart_skips_gracefully(
    campaign, capsys
):
    """A genuinely missing step-0 restart must be reported and skipped, not
    crash the whole plot command the way an uncaught exception would."""
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    # No jorek000000.h5 -- the auto-gather has nothing to run against.
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b", "delta_b_over_b"]\n',
        encoding="utf-8",
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert "skipping delta_b_over_b" in capsys.readouterr().out
    assert not (campaign / "four_dir" / "delta_b_over_b_modes_step.png").exists()
    # delta_b (which doesn't need Btor) must still be drawn -- the failure
    # is scoped to delta_b_over_b, not the whole plot invocation.
    assert (campaign / "four_dir" / "delta_b_modes_step.png").is_file()


def test_delta_b_over_b_alongside_explicit_psi_keeps_both(campaign, monkeypatch):
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    _write_btor_profile(campaign, psi_n=[0.0, 1.0], btor=[3.0, 2.0])
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["Psi", "delta_b_over_b"]\n',
        encoding="utf-8",
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "Psi_modes_step.png").is_file()
    assert (campaign / "four_dir" / "delta_b_over_b_modes_step.png").is_file()


# --- delta_b: derived from Psi, only needs R_axis (no Btor profile) ---------------


def test_delta_b_writes_a_figure_and_uses_it_over_raw_psi(campaign, monkeypatch):
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b"]\n',
        encoding="utf-8",
    )
    captured_calls = []
    original = plot_cli.plot_mode_amplitudes

    def spy(x, series, variable, out_path, **kwargs):
        captured_calls.append(series)
        return original(x, series, variable, out_path, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_mode_amplitudes", spy)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "delta_b_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "Psi_modes_step.png").exists()

    # scale = m / r_axis**2 = 2 / 4 = 0.5 -- no Btor profile needed.
    assert captured_calls
    np.testing.assert_allclose(captured_calls[0][("delta_b", 1, 2)], [2.0, 4.0])


def test_delta_b_and_delta_b_over_b_both_requested_write_both_figures(campaign):
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    _write_btor_profile(campaign, psi_n=[0.0, 1.0], btor=[3.0, 2.0])
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b", "delta_b_over_b"]\n',
        encoding="utf-8",
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert (campaign / "four_dir" / "delta_b_modes_step.png").is_file()
    assert (campaign / "four_dir" / "delta_b_over_b_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "Psi_modes_step.png").exists()


def test_delta_b_missing_r_axis_skips_both_derived_vars_with_one_message(campaign, capsys):
    (campaign / "log").write_text("no axis info here\n", encoding="utf-8")
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b", "delta_b_over_b"]\n',
        encoding="utf-8",
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    out = capsys.readouterr().out
    assert "skipping delta_b, delta_b_over_b" in out
    assert not (campaign / "four_dir" / "delta_b_modes_step.png").exists()
    assert not (campaign / "four_dir" / "delta_b_over_b_modes_step.png").exists()


def test_delta_b_survives_missing_btor_profile_while_delta_b_over_b_is_skipped(
    campaign, capsys
):
    # log has R_axis (from the campaign fixture) but no Btor profile cached.
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b", "delta_b_over_b"]\n',
        encoding="utf-8",
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert "skipping delta_b_over_b" in capsys.readouterr().out
    assert (campaign / "four_dir" / "delta_b_modes_step.png").is_file()
    assert not (campaign / "four_dir" / "delta_b_over_b_modes_step.png").exists()


def test_delta_b_caption_reports_the_peak_across_steps_and_modes(campaign, monkeypatch):
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    _write_four_cache(
        campaign, 100,
        records=[_four_record("Psi", 1, 2, real_peak=4.0), _four_record("Psi", 1, 3, real_peak=1.0)],
    )
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b"]\n',
        encoding="utf-8",
    )
    captured = _spy_on_plot_mode_amplitudes(monkeypatch)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured
    # peak = m=3 * real_peak=1.0 / r_axis**2 = 3/4 = 0.75; m=2*8/4=4.0 is the true max.
    for kwargs in captured:
        assert kwargs.get("caption") == "max \N{GREEK SMALL LETTER DELTA}B = 4 T"


def test_delta_b_over_b_caption_uses_the_normalised_label(campaign, monkeypatch):
    (campaign / "log").write_text("R_axis = 2.0\n", encoding="utf-8")
    _write_btor_profile(campaign, psi_n=[0.0, 1.0], btor=[3.0, 2.0])
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 1, 2, real_peak=4.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 1, 2, real_peak=8.0)])

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'four_vars = ["delta_b_over_b"]\n',
        encoding="utf-8",
    )
    captured = _spy_on_plot_mode_amplitudes(monkeypatch)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured
    # b_ref = Btor at psi_n=1 = 2; scale = m/(r_axis**2*b_ref) = 2/(4*2) = 0.25; peak = 8*0.25 = 2.
    for kwargs in captured:
        assert kwargs.get("caption") == "max \N{GREEK SMALL LETTER DELTA}B/B = 2"


def test_raw_variables_get_no_caption(campaign, monkeypatch):
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])
    _write_four_cache(campaign, 200, records=[_four_record("Psi", 0, 1, real_peak=1.5)])
    captured = _spy_on_plot_mode_amplitudes(monkeypatch)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "four"]) == 0
    assert captured
    for kwargs in captured:
        assert kwargs.get("caption") is None


# --- per-diag steps override: default -> case -> case+diag tree -------------------


def test_per_diag_steps_override_restricts_poincare_only(campaign):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        '[cases."qa2.1_g2.3/eta1e-3_RE".poincare]\n'
        'steps = [100]\n',
        encoding="utf-8",
    )
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare"]) == 0
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert not (campaign / "poinc_dir" / "200_poincare.png").exists()


def test_cli_step_flag_overrides_per_diag_steps_override(campaign):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        '[cases."qa2.1_g2.3/eta1e-3_RE".poincare]\n'
        'steps = [100]\n',
        encoding="utf-8",
    )
    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare", "--step", "200"]
    ) == 0
    assert (campaign / "poinc_dir" / "200_poincare.png").is_file()
    assert not (campaign / "poinc_dir" / "100_poincare.png").exists()


def test_per_diag_steps_override_for_four_only_affects_four(campaign, monkeypatch):
    """A [cases.X.four] override must not change what a different diag
    (poincare) plots in the same invocation."""
    captured = {}
    original = plot_cli.max_amplitude_series

    def spy(paths, steps, **kwargs):
        captured["steps"] = steps
        return original(paths, steps, **kwargs)

    monkeypatch.setattr(plot_cli, "max_amplitude_series", spy)

    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        '[cases."qa2.1_g2.3/eta1e-3_RE".four]\n'
        'steps = [100]\n',
        encoding="utf-8",
    )
    _write_four_cache(campaign, 100, records=[_four_record("Psi", 0, 1, real_peak=1.0)])

    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "poincare", "--diag", "four"]
    ) == 0
    assert captured["steps"] == [100]
    # poincare still used the case's own (unoverridden) steps.
    assert (campaign / "poinc_dir" / "100_poincare.png").is_file()
    assert (campaign / "poinc_dir" / "200_poincare.png").is_file()


# --- radial profiles: current density (or anything else) vs psi_n -----------------


def _write_profile_cache(run_dir, coords_var, var, step, tor_mode, x, y):
    paths = RunPaths(run_dir, pad_width=6)
    cache = paths.profile_cache(coords_var, var, step, tor_mode)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=np.asarray(x), y=np.asarray(y))
    return cache


def test_profiles_diag_writes_one_file_per_var(campaign):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'coords_var = "Psi_N"\n'
        'vars = ["currdens", "T"]\n',
        encoding="utf-8",
    )
    for step in (100, 200):
        _write_profile_cache(
            campaign, "Psi_N", "currdens", step, "midplane", [0.1, 0.5, 0.9], [1.0, 2.0, 1.0],
        )
        _write_profile_cache(
            campaign, "Psi_N", "T", step, "midplane", [0.1, 0.5, 0.9], [3.0, 2.0, 1.0],
        )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "profiles"]) == 0

    assert (campaign / "poinc_dir" / "Psi_N_currdens_profile.png").is_file()
    assert (campaign / "poinc_dir" / "Psi_N_T_profile.png").is_file()


def test_profiles_diag_with_no_cache_reports_and_does_not_crash(campaign, capsys):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'vars = ["currdens"]\n',
        encoding="utf-8",
    )
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "profiles"]) == 0
    assert "no cached" in capsys.readouterr().out


def test_profiles_diag_with_no_vars_reports_and_does_not_crash(campaign, capsys):
    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "profiles"]) == 0
    assert "no vars configured" in capsys.readouterr().out


def test_profiles_diag_draws_every_configured_tor_mode(campaign, monkeypatch):
    """Two tor_modes in the case must both be read and passed to the plot,
    even when only one of them actually has a cache -- the missing one
    should show as an empty panel, not silently drop out."""
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'coords_var = "Psi_N"\n'
        'vars = ["currdens"]\n'
        'tor_mode = ["midplane", "average"]\n',
        encoding="utf-8",
    )
    _write_profile_cache(
        campaign, "Psi_N", "currdens", 100, "midplane", [0.1, 0.5], [1.0, 2.0],
    )
    # No cache for "average" -- simulates a run where the flux average died.

    captured = {}
    original = plot_cli.plot_profile_comparison

    def spy(series_by_mode, var, out_path, **kwargs):
        captured["modes"] = list(series_by_mode)
        captured["average_is_empty"] = series_by_mode.get("average") == {}
        return original(series_by_mode, var, out_path, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_profile_comparison", spy)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "profiles"]) == 0
    assert captured["modes"] == ["midplane", "average"]
    assert captured["average_is_empty"] is True


def test_profiles_diag_jgrad_expands_to_its_components(campaign):
    """Jgrad is a compound var (ashen-side, not a real jorek2_postproc
    expression) -- the cache lookup must be by its expanded component names,
    matching what gather_profiles actually wrote."""
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'coords_var = "Psi_N"\n'
        'vars = ["Jgrad"]\n',
        encoding="utf-8",
    )
    for var in ("currdens", "Btheta", "Btor", "r_minor"):
        _write_profile_cache(
            campaign, "Psi_N", var, 100, "midplane", [0.1, 0.5], [1.0, 2.0],
        )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "profiles"]) == 0
    assert (campaign / "poinc_dir" / "Psi_N_currdens_profile.png").is_file()


def test_profiles_diag_colors_by_true_time_when_zerod_available(campaign, monkeypatch):
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'coords_var = "Psi_N"\n'
        'vars = ["currdens"]\n',
        encoding="utf-8",
    )
    for step in (100, 200):
        _write_profile_cache(
            campaign, "Psi_N", "currdens", step, "midplane", [0.1], [1.0],
        )

    captured = {}
    original = plot_cli.plot_profile_comparison

    def spy(series_by_mode, var, out_path, **kwargs):
        captured["color_label"] = kwargs.get("color_label")
        captured["color_by"] = kwargs.get("color_by")
        return original(series_by_mode, var, out_path, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_profile_comparison", spy)

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "profiles"]) == 0
    assert captured["color_label"] == r"t [$\mu s$]"
    assert captured["color_by"] == {100: 100.0, 200: 200.0}  # 1e-4 s, 2e-4 s -> us


# --- theta_hist: field-line theta-crossing histograms -----------------------------


@pytest.fixture
def theta_campaign(tmp_path, monkeypatch):
    """A separate campaign (rather than reusing `campaign`) because its
    Poincare cache needs lines that actually cross psi_n=1 -- `campaign`'s
    `_write_cache` gives every line a constant psi_n, which never crosses."""
    run_dir = tmp_path / "qa2.1_g2.3" / "eta1e-3_RE"
    run_dir.mkdir(parents=True)
    write_float(run_dir / "real_psi_edge.dat", 1.0)
    (run_dir / "log").write_text("R_axis = 1.363245\n", encoding="utf-8")
    for step in (100, 200):
        (run_dir / f"jorek{step:06d}.h5").write_bytes(b"")
        _write_crossing_cache(run_dir, step)

    cases_toml = tmp_path / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'psi_n_in = [0.2, 0.5]\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return run_dir


def test_theta_hist_diag_writes_one_file_per_case(theta_campaign):
    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "theta_hist"]
    ) == 0
    assert (theta_campaign / "poinc_dir" / "theta_hist.png").is_file()


def test_theta_hist_reports_crossed_and_considered_counts(theta_campaign, capsys):
    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "theta_hist"]
    ) == 0
    out = capsys.readouterr().out
    # Both lines in _write_crossing_cache cross target_psi=1.0.
    assert "2 of 2 lines crossed" in out


def test_theta_hist_cli_overrides_take_precedence_over_case_config(theta_campaign, monkeypatch):
    captured = {}
    original = plot_cli.pooled_crossing_angles

    def spy(records_by_step, steps, **kwargs):
        captured["target_psi"] = kwargs.get("target_psi")
        captured["psi_n_range"] = kwargs.get("psi_n_range")
        return original(records_by_step, steps, **kwargs)

    monkeypatch.setattr(plot_cli, "pooled_crossing_angles", spy)

    assert plot_cli.main([
        "--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "theta_hist",
        "--theta_target_psi", "1.2", "--theta_psi_n_range", "0.1", "0.4",
    ]) == 0
    assert captured["target_psi"] == pytest.approx(1.2)
    assert captured["psi_n_range"] == (0.1, 0.4)


def test_theta_hist_no_poincare_cache_is_reported_not_crashed(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "qa2.1_g2.3" / "eta1e-3_RE"
    run_dir.mkdir(parents=True)
    write_float(run_dir / "real_psi_edge.dat", 1.0)
    (run_dir / "jorek000100.h5").write_bytes(b"")
    (tmp_path / "cases.toml").write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\nsteps = [100]\n', encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "theta_hist"]
    ) == 0
    assert "no Poincare cache" in capsys.readouterr().out
    assert not (run_dir / "poinc_dir" / "theta_hist.png").exists()


# --- cross-case comparisons ---------------------------------------------------------


@pytest.fixture
def comparison_campaign(tmp_path, monkeypatch):
    """Two runs under one campaign, grouped by a [comparisons.*] table."""
    for name in ("eta1e-3_RE", "eta1e-4_RE"):
        run_dir = tmp_path / "qa2.1_g2.3" / name
        run_dir.mkdir(parents=True)
        write_float(run_dir / "real_psi_edge.dat", 1.0)
        (run_dir / "log").write_text("R_axis = 1.363245\n", encoding="utf-8")
        (run_dir / "jorek000100.h5").write_bytes(b"")
        _write_crossing_cache(run_dir, 100)

    cases_toml = tmp_path / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.0\n'
        '[cases."qa2.1_g2.3/eta1e-4_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.0\n'
        '[comparisons.eta_scan]\n'
        'note = "resistivity scan"\n'
        'cases = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE"]\n'
        'x_tick_labels = ["1e-3", "1e-4"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_compare_theta_hist_writes_one_file_for_the_comparison(comparison_campaign):
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "theta_hist"]) == 0
    assert (comparison_campaign / "figures" / "eta_scan_theta_hist.png").is_file()


def test_compare_reports_per_case_crossing_counts(comparison_campaign, capsys):
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "theta_hist"]) == 0
    out = capsys.readouterr().out
    assert "qa2.1_g2.3/eta1e-3_RE: 2 of 2 lines crossed" in out
    assert "qa2.1_g2.3/eta1e-4_RE: 2 of 2 lines crossed" in out


def test_compare_unknown_comparison_is_an_error(comparison_campaign, capsys):
    assert plot_cli.main(["--compare", "does_not_exist"]) == 1
    assert "unknown comparison" in capsys.readouterr().out


def test_compare_diag_without_a_comparison_renderer_is_reported_and_skipped(
    comparison_campaign, capsys
):
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "profiles"]) == 0
    out = capsys.readouterr().out
    assert "no comparison renderer" in out
    assert not (comparison_campaign / "figures").exists()


def test_list_comparisons(comparison_campaign, capsys):
    assert plot_cli.main(["--list-comparisons"]) == 0
    out = capsys.readouterr().out
    assert "eta_scan (2 cases)" in out
    assert "resistivity scan" in out


def test_compare_step_override_applies_to_every_member(comparison_campaign, monkeypatch):
    _write_crossing_cache(comparison_campaign / "qa2.1_g2.3" / "eta1e-3_RE", 200)
    _write_crossing_cache(comparison_campaign / "qa2.1_g2.3" / "eta1e-4_RE", 200)

    captured = []
    original = plot_cli.pooled_crossing_angles

    def spy(records_by_step, steps, **kwargs):
        captured.append(list(steps))
        return original(records_by_step, steps, **kwargs)

    monkeypatch.setattr(plot_cli, "pooled_crossing_angles", spy)

    assert plot_cli.main(
        ["--compare", "eta_scan", "--diag", "theta_hist", "--step", "200"]
    ) == 0
    assert captured == [[200], [200]]


# --- wetted_fraction: comparison-only "scalar vs. scan parameter" plot ------------


def _add_x_values(comparison_campaign, *, x_values):
    cases_toml = comparison_campaign / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n'
        '[cases."qa2.1_g2.3/eta1e-4_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n'
        '[comparisons.eta_scan]\n'
        'note = "resistivity scan"\n'
        'cases = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE"]\n'
        'x_tick_labels = ["1e-3", "1e-4"]\n'
        f'x_values = {x_values}\n'
        'x_label = "$\\\\eta$"\n',
        encoding="utf-8",
    )


def _add_x_values_with_case_threshold(comparison_campaign, *, x_values, threshold):
    cases_toml = comparison_campaign / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n'
        f'theta_wetted_threshold = {threshold}\n'
        '[cases."qa2.1_g2.3/eta1e-4_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n'
        f'theta_wetted_threshold = {threshold}\n'
        '[comparisons.eta_scan]\n'
        'cases = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE"]\n'
        'x_tick_labels = ["1e-3", "1e-4"]\n'
        f'x_values = {x_values}\n',
        encoding="utf-8",
    )


def test_compare_wetted_fraction_writes_one_file_for_the_comparison(comparison_campaign):
    _add_x_values(comparison_campaign, x_values=[1e-3, 1e-4])
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    assert (comparison_campaign / "figures" / "eta_scan_wetted_fraction.png").is_file()


def test_compare_wetted_fraction_reports_per_case_fraction(comparison_campaign, capsys):
    _add_x_values(comparison_campaign, x_values=[1e-3, 1e-4])
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    out = capsys.readouterr().out
    assert "qa2.1_g2.3/eta1e-3_RE: wetted fraction" in out
    assert "qa2.1_g2.3/eta1e-4_RE: wetted fraction" in out


def test_compare_wetted_fraction_without_x_values_is_reported_and_skipped(
    comparison_campaign, capsys
):
    # comparison_campaign's default cases.toml has no x_values configured.
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    out = capsys.readouterr().out
    assert "no x_values configured" in out
    assert not (comparison_campaign / "figures").exists()


def test_compare_wetted_fraction_threshold_override(comparison_campaign, monkeypatch):
    _add_x_values(comparison_campaign, x_values=[1e-3, 1e-4])
    captured = []
    original = plot_cli.wetted_fraction

    def spy(counts, **kwargs):
        captured.append(kwargs.get("threshold"))
        return original(counts, **kwargs)

    monkeypatch.setattr(plot_cli, "wetted_fraction", spy)

    assert plot_cli.main(
        ["--compare", "eta_scan", "--diag", "wetted_fraction", "--theta_wetted_threshold", "0.02"]
    ) == 0
    assert captured == [0.02, 0.02]


def test_compare_wetted_fraction_default_threshold_is_one_over_bins(
    comparison_campaign, monkeypatch
):
    _add_x_values(comparison_campaign, x_values=[1e-3, 1e-4])
    captured = []
    original = plot_cli.wetted_fraction

    def spy(counts, **kwargs):
        captured.append(kwargs.get("threshold"))
        return original(counts, **kwargs)

    monkeypatch.setattr(plot_cli, "wetted_fraction", spy)

    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    assert captured == [pytest.approx(1 / 20), pytest.approx(1 / 20)]  # theta_bins=20


def test_compare_wetted_fraction_uses_case_theta_wetted_threshold(
    comparison_campaign, monkeypatch
):
    """A case's own theta_wetted_threshold is used when --theta_wetted_threshold
    is not given -- the config-in-cases.toml path being asked about."""
    _add_x_values_with_case_threshold(
        comparison_campaign, x_values=[1e-3, 1e-4], threshold=0.03,
    )
    captured = []
    original = plot_cli.wetted_fraction

    def spy(counts, **kwargs):
        captured.append(kwargs.get("threshold"))
        return original(counts, **kwargs)

    monkeypatch.setattr(plot_cli, "wetted_fraction", spy)

    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    assert captured == [pytest.approx(0.03), pytest.approx(0.03)]


def test_compare_wetted_fraction_cli_flag_overrides_case_config(
    comparison_campaign, monkeypatch
):
    """--theta_wetted_threshold on the command line outranks each case's own
    theta_wetted_threshold, same precedence as --theta_target_psi."""
    _add_x_values_with_case_threshold(
        comparison_campaign, x_values=[1e-3, 1e-4], threshold=0.03,
    )
    captured = []
    original = plot_cli.wetted_fraction

    def spy(counts, **kwargs):
        captured.append(kwargs.get("threshold"))
        return original(counts, **kwargs)

    monkeypatch.setattr(plot_cli, "wetted_fraction", spy)

    assert plot_cli.main(
        ["--compare", "eta_scan", "--diag", "wetted_fraction", "--theta_wetted_threshold", "0.05"]
    ) == 0
    assert captured == [pytest.approx(0.05), pytest.approx(0.05)]


# --- comparison-level overrides: uniform analysis params across every member -----


def _add_x_values_with_comparison_overrides(comparison_campaign, *, x_values):
    """Members set their own (different) theta_target_psi/theta_bins/
    theta_wetted_threshold; the comparison's own settings must win over all
    of them uniformly."""
    cases_toml = comparison_campaign / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n'
        'theta_wetted_threshold = 0.01\n'
        '[cases."qa2.1_g2.3/eta1e-4_RE"]\n'
        'steps = [100]\n'
        'theta_target_psi = 1.02\n'
        'theta_bins = 50\n'
        'theta_wetted_threshold = 0.02\n'
        '[comparisons.eta_scan]\n'
        'cases = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE"]\n'
        'x_tick_labels = ["1e-3", "1e-4"]\n'
        f'x_values = {x_values}\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n'
        'theta_wetted_threshold = 0.04\n',
        encoding="utf-8",
    )


def test_compare_wetted_fraction_comparison_config_overrides_case_config(
    comparison_campaign, monkeypatch
):
    _add_x_values_with_comparison_overrides(comparison_campaign, x_values=[1e-3, 1e-4])
    captured = []
    original = plot_cli.wetted_fraction

    def spy(counts, **kwargs):
        captured.append(kwargs.get("threshold"))
        return original(counts, **kwargs)

    monkeypatch.setattr(plot_cli, "wetted_fraction", spy)

    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    # Both members get the comparison's 0.04, not their own 0.01/0.02.
    assert captured == [pytest.approx(0.04), pytest.approx(0.04)]


def test_compare_wetted_fraction_cli_flag_outranks_comparison_config(
    comparison_campaign, monkeypatch
):
    _add_x_values_with_comparison_overrides(comparison_campaign, x_values=[1e-3, 1e-4])
    captured = []
    original = plot_cli.wetted_fraction

    def spy(counts, **kwargs):
        captured.append(kwargs.get("threshold"))
        return original(counts, **kwargs)

    monkeypatch.setattr(plot_cli, "wetted_fraction", spy)

    assert plot_cli.main(
        ["--compare", "eta_scan", "--diag", "wetted_fraction", "--theta_wetted_threshold", "0.09"]
    ) == 0
    assert captured == [pytest.approx(0.09), pytest.approx(0.09)]


def test_compare_theta_hist_comparison_target_psi_overrides_case_target_psi(
    comparison_campaign, monkeypatch
):
    _add_x_values_with_comparison_overrides(comparison_campaign, x_values=[1e-3, 1e-4])
    captured = []
    original = plot_cli.pooled_crossing_angles

    def spy(records_by_step, steps, **kwargs):
        captured.append(kwargs.get("target_psi"))
        return original(records_by_step, steps, **kwargs)

    monkeypatch.setattr(plot_cli, "pooled_crossing_angles", spy)

    assert plot_cli.main(["--compare", "eta_scan", "--diag", "theta_hist"]) == 0
    # Both members get the comparison's 1.0, not their own 1.0/1.02.
    assert captured == [pytest.approx(1.0), pytest.approx(1.0)]


# --- warning when a comparison override shadows a case's own explicit setting ----


def test_compare_theta_hist_warns_when_comparison_shadows_case_target_psi(
    comparison_campaign, capsys
):
    _add_x_values_with_comparison_overrides(comparison_campaign, x_values=[1e-3, 1e-4])
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "theta_hist"]) == 0
    out = capsys.readouterr().out
    assert (
        "qa2.1_g2.3/eta1e-3_RE: comparison 'eta_scan' sets theta_target_psi=1.0, "
        "overriding this case's own theta_target_psi=1.0" in out
    )
    assert (
        "qa2.1_g2.3/eta1e-4_RE: comparison 'eta_scan' sets theta_target_psi=1.0, "
        "overriding this case's own theta_target_psi=1.02" in out
    )


def test_compare_wetted_fraction_warns_for_every_shadowed_field(
    comparison_campaign, capsys
):
    _add_x_values_with_comparison_overrides(comparison_campaign, x_values=[1e-3, 1e-4])
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    out = capsys.readouterr().out
    assert "sets theta_target_psi=1.0, overriding this case's own theta_target_psi=1.02" in out
    assert "sets theta_bins=20, overriding this case's own theta_bins=50" in out
    assert (
        "sets theta_wetted_threshold=0.04, overriding this case's own "
        "theta_wetted_threshold=0.02" in out
    )


def test_compare_wetted_fraction_no_warning_when_case_uses_the_default(
    comparison_campaign, capsys
):
    """A case that never set theta_target_psi/theta_bins (still at their
    dataclass defaults) has nothing meaningful being shadowed -- no warning,
    even though the comparison does override them."""
    cases_toml = comparison_campaign / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100]\n'
        '[cases."qa2.1_g2.3/eta1e-4_RE"]\n'
        'steps = [100]\n'
        '[comparisons.eta_scan]\n'
        'cases = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE"]\n'
        'x_tick_labels = ["1e-3", "1e-4"]\n'
        'x_values = [1e-3, 1e-4]\n'
        'theta_target_psi = 1.0\n'
        'theta_bins = 20\n',
        encoding="utf-8",
    )
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    out = capsys.readouterr().out
    assert "overriding this case's own" not in out


def test_compare_wetted_fraction_no_warning_for_a_field_with_a_cli_flag(
    comparison_campaign, capsys
):
    """A CLI flag already outranks both tiers for its own field, so its
    override is expected, not surprising -- no warning for that one field.
    Other fields (theta_target_psi, theta_bins) have no CLI flag here and
    still warn -- the suppression is per field, not all-or-nothing."""
    _add_x_values_with_comparison_overrides(comparison_campaign, x_values=[1e-3, 1e-4])
    assert plot_cli.main(
        ["--compare", "eta_scan", "--diag", "wetted_fraction", "--theta_wetted_threshold", "0.09"]
    ) == 0
    out = capsys.readouterr().out
    assert "theta_wetted_threshold" not in out
    assert "theta_target_psi" in out  # no CLI flag for this one -- still warns


def test_compare_wetted_fraction_no_warning_when_every_field_has_a_cli_flag(
    comparison_campaign, capsys
):
    _add_x_values_with_comparison_overrides(comparison_campaign, x_values=[1e-3, 1e-4])
    assert plot_cli.main([
        "--compare", "eta_scan", "--diag", "wetted_fraction",
        "--theta_target_psi", "1.0", "--theta_bins", "20", "--theta_wetted_threshold", "0.09",
    ]) == 0
    out = capsys.readouterr().out
    assert "overriding this case's own" not in out


def test_wetted_fraction_diag_is_comparison_only_per_case(theta_campaign, capsys):
    assert plot_cli.main(
        ["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "wetted_fraction"]
    ) == 0
    out = capsys.readouterr().out
    assert "comparison-only" in out
    assert not (theta_campaign / "figures").exists()


# --- wetted_fraction with datasets: overlaying several related scans -------------


@pytest.fixture
def dataset_comparison_campaign(tmp_path, monkeypatch):
    """Two independent scans ("normal" and "rho19"), each two runs, grouped
    into one datasets-style comparison sharing a resistivity x-axis."""
    for group in ("qa2.1_g2.3", "qa2.1_g2.3_rho19"):
        for name in ("eta1e-3_RE", "eta1e-4_RE"):
            run_dir = tmp_path / group / name
            run_dir.mkdir(parents=True)
            write_float(run_dir / "real_psi_edge.dat", 1.0)
            (run_dir / "jorek000100.h5").write_bytes(b"")
            _write_crossing_cache(run_dir, 100)

    cases_toml = tmp_path / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100]\ntheta_target_psi = 1.0\ntheta_bins = 20\n'
        '[cases."qa2.1_g2.3/eta1e-4_RE"]\n'
        'steps = [100]\ntheta_target_psi = 1.0\ntheta_bins = 20\n'
        '[cases."qa2.1_g2.3_rho19/eta1e-3_RE"]\n'
        'steps = [100]\ntheta_target_psi = 1.0\ntheta_bins = 20\n'
        '[cases."qa2.1_g2.3_rho19/eta1e-4_RE"]\n'
        'steps = [100]\ntheta_target_psi = 1.0\ntheta_bins = 20\n'
        '[comparisons.eta_scan]\n'
        'note = "resistivity scan, normal vs. rho19"\n'
        'x_values = [1e-3, 1e-4]\n'
        'x_label = "$\\\\eta$"\n'
        '[comparisons.eta_scan.datasets.normal]\n'
        'cases = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE"]\n'
        '[comparisons.eta_scan.datasets.rho19]\n'
        'cases = ["qa2.1_g2.3_rho19/eta1e-3_RE", "qa2.1_g2.3_rho19/eta1e-4_RE"]\n'
        'color = "tab:red"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_compare_wetted_fraction_datasets_writes_one_file(dataset_comparison_campaign):
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    assert (dataset_comparison_campaign / "figures" / "eta_scan_wetted_fraction.png").is_file()


def test_compare_wetted_fraction_datasets_reports_every_case(
    dataset_comparison_campaign, capsys
):
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    out = capsys.readouterr().out
    assert "qa2.1_g2.3/eta1e-3_RE: wetted fraction" in out
    assert "qa2.1_g2.3_rho19/eta1e-3_RE: wetted fraction" in out


def test_compare_wetted_fraction_dataset_without_x_values_is_reported_and_skipped(
    tmp_path, monkeypatch, capsys
):
    run_dir = tmp_path / "qa2.1_g2.3" / "eta1e-3_RE"
    run_dir.mkdir(parents=True)
    write_float(run_dir / "real_psi_edge.dat", 1.0)
    (run_dir / "jorek000100.h5").write_bytes(b"")
    _write_crossing_cache(run_dir, 100)
    (tmp_path / "cases.toml").write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100]\ntheta_target_psi = 1.0\ntheta_bins = 20\n'
        '[comparisons.eta_scan]\n'
        '[comparisons.eta_scan.datasets.normal]\n'
        'cases = ["qa2.1_g2.3/eta1e-3_RE"]\n',  # no x_values anywhere
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "wetted_fraction"]) == 0
    out = capsys.readouterr().out
    assert "no x_values" in out
    assert not (tmp_path / "figures").exists()


def test_compare_theta_hist_on_a_datasets_comparison_is_reported_and_skipped(
    dataset_comparison_campaign, capsys
):
    assert plot_cli.main(["--compare", "eta_scan", "--diag", "theta_hist"]) == 0
    out = capsys.readouterr().out
    assert "uses 'datasets'" in out
    assert not (dataset_comparison_campaign / "figures").exists()


def test_list_comparisons_reports_dataset_count(dataset_comparison_campaign, capsys):
    assert plot_cli.main(["--list-comparisons"]) == 0
    out = capsys.readouterr().out
    assert "2 datasets, 4 cases" in out


def test_compare_wetted_fraction_dataset_flag_restricts_to_selected(
    dataset_comparison_campaign, capsys
):
    assert plot_cli.main([
        "--compare", "eta_scan", "--diag", "wetted_fraction", "--dataset", "normal",
    ]) == 0
    out = capsys.readouterr().out
    assert "qa2.1_g2.3/eta1e-3_RE" in out
    assert "qa2.1_g2.3_rho19/eta1e-3_RE" not in out


def test_compare_wetted_fraction_dataset_flag_unknown_name_is_reported(
    dataset_comparison_campaign, capsys
):
    assert plot_cli.main([
        "--compare", "eta_scan", "--diag", "wetted_fraction", "--dataset", "ghost",
    ]) == 0
    out = capsys.readouterr().out
    assert "no dataset(s) ['ghost']" in out
    assert not (dataset_comparison_campaign / "figures").exists()


def test_profiles_diag_falls_back_to_step_index_without_zerod(campaign, capsys):
    (campaign / "postproc" / "zeroD_quantities_s000100.dat").unlink()
    (campaign / "postproc" / "zeroD_quantities_s000200.dat").unlink()
    cases_toml = campaign.parent.parent / "cases.toml"
    cases_toml.write_text(
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\n'
        'steps = [100, 200]\n'
        'coords_var = "Psi_N"\n'
        'vars = ["currdens"]\n',
        encoding="utf-8",
    )
    _write_profile_cache(
        campaign, "Psi_N", "currdens", 100, "midplane", [0.1], [1.0],
    )

    assert plot_cli.main(["--case", "qa2.1_g2.3/eta1e-3_RE", "--diag", "profiles"]) == 0
    assert "colouring profiles by step index" in capsys.readouterr().out
