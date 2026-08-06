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
