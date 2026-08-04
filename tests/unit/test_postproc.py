"""ashen.postproc: control-script builders and output parsers."""

from __future__ import annotations

from ashen.postproc import (
    flux_surface_script,
    parse_macroscopic_vars,
    profile_script,
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


def test_flux_surface_script():
    text = flux_surface_script("in_main", 200, 0.5)
    assert text.splitlines() == [
        "namelist in_main",
        "set units 1",
        "for step 200 do",
        "  fluxsurface 0.5",
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
