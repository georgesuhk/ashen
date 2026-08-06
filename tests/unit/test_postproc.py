"""ashen.postproc: control-script builders and output parsers."""

from __future__ import annotations

from ashen.postproc import (
    flux_surface_script,
    parse_macroscopic_vars,
    profile_script,
    qprofile_script,
    read_postproc_profile,
    read_zeroD,
    zero_d_script,
)


def test_profile_script_matches_legacy_shape():
    text = profile_script("in_main", 5000, "Psi_N", ["currdens", "Btheta"], 100, tor_mode="midplane")
    lines = text.splitlines()
    assert lines[0] == "namelist in_main"
    assert lines[2] == "for step 5000 do"
    assert "  expressions Psi_N currdens Btheta" in lines
    assert "  set linepoints 100" in lines
    assert "  midplane" in lines


def test_profile_script_accepts_single_string_var():
    text = profile_script("in_main", 1, "R", "currdens", 10)
    assert "  expressions R currdens" in text.splitlines()


def test_profile_script_midplane_outer_and_inner_pass_through_verbatim():
    """tor_mode is emitted as the literal command word -- 'midplane outer'/
    'midplane inner' aren't special-cased here, only in
    ashen.diagnostics.profiles._TOR_MODE_PREFIX."""
    text = profile_script("in_main", 1, "Psi_N", "currdens", 100, tor_mode="midplane outer")
    assert "  midplane outer" in text.splitlines()


def test_profile_script_average_uses_surfaces_not_linepoints():
    """average reads the 'surfaces' setting (exec_commands.f90:1771), not
    'linepoints' -- the legacy script emitted only linepoints, so an average
    run silently ignored n_points and traced at the built-in default."""
    text = profile_script(
        "in_main", 5000, "Psi_N", "currdens", 100, tor_mode="average",
        surfaces=250, rad_range=(0.01, 0.85), nmaxsteps=10000,
        deltaphi=0.05, nsmallsteps=5,
    )
    lines = text.splitlines()
    assert "  set linepoints 100" not in lines
    assert "  set surfaces 250" in lines
    assert "  set rad_range_min 0.01" in lines
    assert "  set rad_range_max 0.85" in lines
    assert "  set nmaxsteps 10000" in lines
    assert "  set deltaphi 0.05" in lines
    assert "  set nsmallsteps 5" in lines
    assert "  average" in lines


def test_profile_script_average_defaults_match_jorek2_postproc_own_fallback():
    """Unconfigured knobs should reproduce jorek2_postproc's own built-in
    defaults (jorek2_postproc.f90:44-51) -- an unconfigured average call
    behaves like a bare `average` with no `set` lines at all."""
    text = profile_script("in_main", 1, "Psi_N", "currdens", 100, tor_mode="average")
    lines = text.splitlines()
    assert "  set surfaces 100" in lines
    assert "  set rad_range_min 0.001" in lines
    assert "  set rad_range_max 0.999" in lines
    assert "  set nmaxsteps 2500" in lines
    assert "  set deltaphi 0.3" in lines
    assert "  set nsmallsteps 3" in lines


def test_flux_surface_script():
    text = flux_surface_script("in_main", 200, 0.5)
    assert text.splitlines() == [
        "namelist in_main",
        "set units 1",
        "for step 200 do",
        "  fluxsurface 0.5",
        "done",
    ]


def test_qprofile_script():
    text = qprofile_script("in_main", 200)
    assert text.splitlines() == [
        "namelist in_main",
        "set units 1",
        "for step 200 do",
        "  qprofile",
        "done",
    ]


def test_zero_d_script_preserves_step_argument_verbatim():
    # The legacy caller (data_jorek.py:719) passes a zero-padded string, not
    # the raw int -- this must not reformat it.
    text = zero_d_script("in_main", "005000")
    assert "for step 005000 do" in text.splitlines()


def test_read_zeroD(tmp_path):
    path = tmp_path / "zeroD_quantities_s000100.dat"
    path.write_text("Time Energy\n1.5e-3 2.0e6\n", encoding="utf-8")
    assert read_zeroD(path) == {"Time": 1.5e-3, "Energy": 2.0e6}


def test_read_postproc_profile(tmp_path):
    path = tmp_path / "exprs_midplane_s000100.dat"
    path.write_text(
        "# R currdens\n"
        "# time step #000100\n"
        "1.0 10.0\n"
        "1.5 20.0\n"
        "\n",
        encoding="utf-8",
    )
    headers, blocks = read_postproc_profile(path)
    assert headers == ["R", "currdens"]
    assert 100 in blocks
    assert blocks[100].tolist() == [[1.0, 10.0], [1.5, 20.0]]


def test_parse_macroscopic_vars(tmp_path):
    path = tmp_path / "macroscopic_vars.dat"
    path.write_text(
        "@energies\n"  # a section header with no colon -- skipped, not parsed
        "@energy_magnetic: 1.0D0 2.0D0 3.0D0\n"
        "@energy_magnetic: 2.0D0 4.0D0 6.0D0\n",
        encoding="utf-8",
    )
    out = parse_macroscopic_vars(path)
    assert set(out) == {"energy_magnetic"}
    assert out["energy_magnetic"]["t"].tolist() == [1.0, 2.0]
    assert out["energy_magnetic"]["y"].tolist() == [[2.0, 3.0], [4.0, 6.0]]
