"""ashen.cases: declarative cases.toml loading."""

from __future__ import annotations

import pytest

from ashen.cases import CasesError, load_cases


def _write(tmp_path, text):
    path = tmp_path / "cases.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_case_with_explicit_step_list(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases."qa2.1_g2.3/eta1e-3_RE"]
        folder = "qa2.1_g2.3/eta1e-3_RE"
        steps = [200, 400, 600]
        note = "internal kink investigation"
        """,
    )
    cases = load_cases(path)
    case = cases["qa2.1_g2.3/eta1e-3_RE"]
    assert case.folder == "qa2.1_g2.3/eta1e-3_RE"
    assert case.steps == [200, 400, 600]
    assert case.note == "internal kink investigation"


def test_case_with_range_steps(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        folder = "a"
        steps = { start = 200, stop = 800, step = 200 }
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps == [200, 400, 600]


def test_defaults_are_inherited_and_overridable(tmp_path):
    path = _write(
        tmp_path,
        """
        [defaults]
        n_turns = 1000
        ang_sample_freq = 8
        vars = ["Jgrad", "currdens"]

        [cases.a]
        folder = "a"
        steps = [1]

        [cases.b]
        folder = "b"
        steps = [1]
        n_turns = 500
        """,
    )
    cases = load_cases(path)
    assert cases["a"].n_turns == 1000
    assert cases["a"].vars == ["Jgrad", "currdens"]
    assert cases["b"].n_turns == 500
    assert cases["b"].vars == ["Jgrad", "currdens"]


def test_four_defaults_match_jorek2_four_own_fallback(tmp_path):
    """An unconfigured case must reproduce jorek2_four.f90:44-50's own
    defaults, so a case with no [four] knobs behaves like a bare run."""
    path = _write(tmp_path, '[cases.a]\nfolder = "a"\nsteps = [1]\n')
    case = load_cases(path)["a"]
    assert case.nstpts == 30
    assert case.ntht == 32
    assert case.nmaxsteps == 2500
    assert case.deltaphi == 0.3
    assert case.nsmallsteps == 3
    assert case.rad_range == [0.001, 0.999]


def test_four_params_are_overridable_per_case(tmp_path):
    path = _write(
        tmp_path,
        """
        [defaults]
        nstpts = 50
        rad_range = [0.01, 0.9]

        [cases.a]
        folder = "a"
        steps = [1]
        ntht = 64
        """,
    )
    case = load_cases(path)["a"]
    assert case.nstpts == 50
    assert case.rad_range == [0.01, 0.9]
    assert case.ntht == 64


def test_missing_folder_raises(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    with pytest.raises(CasesError, match="folder"):
        load_cases(path)


def test_missing_steps_raises(tmp_path):
    path = _write(tmp_path, '[cases.a]\nfolder = "a"\n')
    with pytest.raises(CasesError, match="steps"):
        load_cases(path)


def test_unknown_key_raises(tmp_path):
    path = _write(tmp_path, '[cases.a]\nfolder = "a"\nsteps = [1]\ntypo_field = 1\n')
    with pytest.raises(CasesError, match="typo_field"):
        load_cases(path)


def test_no_cases_raises(tmp_path):
    path = _write(tmp_path, '[defaults]\nn_turns = 1000\n')
    with pytest.raises(CasesError, match="no \\[cases"):
        load_cases(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(CasesError, match="not found"):
        load_cases(tmp_path / "nope.toml")


def test_malformed_toml_raises(tmp_path):
    path = _write(tmp_path, "not [ valid toml")
    with pytest.raises(CasesError, match="malformed"):
        load_cases(path)


def test_phi_start_defaults_to_zero(tmp_path):
    """The legacy diagnostic hardcoded phi_start = 0 (poinc_diag.py:119)."""
    path = _write(tmp_path, '[cases.a]\nfolder = "a"\nsteps = [200]\n')
    assert load_cases(path)["a"].phi_start == 0.0


def test_phi_start_is_settable(tmp_path):
    path = _write(
        tmp_path,
        '[defaults]\nphi_start = 0.7853981634\n\n'
        '[cases.a]\nfolder = "a"\nsteps = [200]\n',
    )
    assert load_cases(path)["a"].phi_start == pytest.approx(0.7853981634)
