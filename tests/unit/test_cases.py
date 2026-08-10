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
        steps = [200, 400, 600]
        note = "internal kink investigation"
        """,
    )
    cases = load_cases(path)
    case = cases["qa2.1_g2.3/eta1e-3_RE"]
    assert case.name == "qa2.1_g2.3/eta1e-3_RE"
    assert case.steps == [200, 400, 600]
    assert case.note == "internal kink investigation"


def test_case_with_range_steps(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = { start = 200, stop = 800, step = 200 }
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps == [200, 400, 600]


def test_case_with_steps_list_mixing_explicit_values_and_a_range(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [200, 400, { start = 1000, stop = 1600, step = 200 }]
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps == [200, 400, 1000, 1200, 1400]


def test_case_with_steps_list_mixing_two_ranges(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [{ start = 200, stop = 600, step = 200 }, { start = 1000, stop = 1400, step = 200 }]
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps == [200, 400, 1000, 1200]


def test_case_with_steps_list_overlapping_range_and_explicit_value_dedupes(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [200, 400, 600, { start = 600, stop = 1000, step = 200 }]
        """,
    )
    case = load_cases(path)["a"]
    # 600 appears both as an explicit entry and as the range's start -- must
    # collapse to one, not be processed twice.
    assert case.steps == [200, 400, 600, 800]


def test_case_with_steps_list_of_ranges_is_sorted_regardless_of_toml_order(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [{ start = 1000, stop = 1400, step = 200 }, 200, 400]
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps == [200, 400, 1000, 1200]


def test_diag_steps_override_accepts_a_mixed_list(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]

        [cases.a.four]
        steps = [200, 400, { start = 1000, stop = 1400, step = 200 }]
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps_for("four") == [200, 400, 1000, 1200]


def test_defaults_are_inherited_and_overridable(tmp_path):
    path = _write(
        tmp_path,
        """
        [defaults]
        n_turns = 1000
        ang_sample_freq = 8
        vars = ["Jgrad", "currdens"]

        [cases.a]
        steps = [1]

        [cases.b]
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
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
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
        steps = [1]
        ntht = 64
        """,
    )
    case = load_cases(path)["a"]
    assert case.nstpts == 50
    assert case.rad_range == [0.01, 0.9]
    assert case.ntht == 64


def test_lc_psi_n_in_defaults_to_none(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    assert load_cases(path)["a"].lc_psi_n_in is None


def test_psi_n_in_as_explicit_list(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\npsi_n_in = [0.1, 0.5, 0.9]\n',
    )
    assert load_cases(path)["a"].psi_n_in == [0.1, 0.5, 0.9]


def test_psi_n_in_range_with_step(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        psi_n_in = { start = 0.1, stop = 0.5, step = 0.1 }
        """,
    )
    got = load_cases(path)["a"].psi_n_in
    assert got == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


def test_psi_n_in_range_with_count(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        psi_n_in = { start = 0.0, stop = 1.0, n = 5 }
        """,
    )
    got = load_cases(path)["a"].psi_n_in
    assert got == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_psi_n_in_range_rejects_both_step_and_n(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        psi_n_in = { start = 0.0, stop = 1.0, step = 0.1, n = 5 }
        """,
    )
    with pytest.raises(CasesError, match="'step' and 'n'"):
        load_cases(path)


def test_psi_n_in_range_requires_step_or_n(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\npsi_n_in = { start = 0.0, stop = 1.0 }\n',
    )
    with pytest.raises(CasesError, match="needs 'step' or 'n'"):
        load_cases(path)


def test_psi_n_in_range_rejects_nonpositive_step(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\n'
        'psi_n_in = { start = 0.0, stop = 1.0, step = 0.0 }\n',
    )
    with pytest.raises(CasesError, match="step must be positive"):
        load_cases(path)


def test_psi_n_in_range_rejects_stop_before_start(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\n'
        'psi_n_in = { start = 1.0, stop = 0.0, step = 0.1 }\n',
    )
    with pytest.raises(CasesError, match="stop must be >= start"):
        load_cases(path)


def test_psi_n_in_range_missing_bounds_raises(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\npsi_n_in = { step = 0.1 }\n',
    )
    with pytest.raises(CasesError, match="missing"):
        load_cases(path)


def test_lc_psi_n_in_as_explicit_list(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\n'
        'psi_n_in = [0.1, 0.3, 0.5, 0.7, 0.9]\n'
        'lc_psi_n_in = [0.3, 0.7]\n',
    )
    assert load_cases(path)["a"].lc_psi_n_in == [0.3, 0.7]


def test_lc_psi_n_in_as_range(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        lc_psi_n_in = { start = 0.2, stop = 0.6, step = 0.2 }
        """,
    )
    got = load_cases(path)["a"].lc_psi_n_in
    assert got == pytest.approx([0.2, 0.4, 0.6])


def test_lc_psi_n_in_bounds_filter_selects_from_psi_n_in(tmp_path):
    """{min, max} filters the case's own (already-resolved) psi_n_in, rather
    than generating new values -- the common "just zoom into a sub-range"
    case."""
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\n'
        'psi_n_in = [0.1, 0.3, 0.5, 0.7, 0.9]\n'
        'lc_psi_n_in = { min = 0.25, max = 0.75 }\n',
    )
    assert load_cases(path)["a"].lc_psi_n_in == [0.3, 0.5, 0.7]


def test_lc_psi_n_in_bounds_filter_with_no_psi_n_in_is_empty(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nlc_psi_n_in = { min = 0.25, max = 0.75 }\n',
    )
    assert load_cases(path)["a"].lc_psi_n_in == []


def test_four_vars_and_modes_default_empty(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    case = load_cases(path)["a"]
    assert case.four_vars == []
    assert case.four_modes == []


def test_four_vars_and_modes_are_settable(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        four_vars = ["Psi", "u"]
        four_modes = [[0, 1], [1, 0]]
        """,
    )
    case = load_cases(path)["a"]
    assert case.four_vars == ["Psi", "u"]
    assert case.four_modes == [[0, 1], [1, 0]]


def test_four_modes_rejects_non_pair_entries(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nfour_modes = [[0, 1, 2]]\n',
    )
    with pytest.raises(CasesError, match="\\[m, n\\] pairs"):
        load_cases(path)


def test_four_growth_rate_defaults_off(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    case = load_cases(path)["a"]
    assert case.four_growth_rate is False
    assert case.four_growth_steps is None


def test_four_growth_rate_and_steps_are_settable(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        four_growth_rate = true
        four_growth_steps = [1000, 3000]
        """,
    )
    case = load_cases(path)["a"]
    assert case.four_growth_rate is True
    assert case.four_growth_steps == [1000, 3000]


def test_four_growth_steps_rejects_non_pair(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nfour_growth_steps = [1000]\n',
    )
    with pytest.raises(CasesError, match="start_step, end_step"):
        load_cases(path)


def test_four_growth_steps_rejects_start_after_end(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nfour_growth_steps = [3000, 1000]\n',
    )
    with pytest.raises(CasesError, match="must not be greater than"):
        load_cases(path)


# --- four_ylim / four_deconfinement_step -----------------------------------------


def test_four_ylim_and_deconfinement_step_default_empty(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    case = load_cases(path)["a"]
    assert case.four_ylim == {}
    assert case.four_deconfinement_step is None


def test_four_ylim_and_deconfinement_step_are_settable(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        four_ylim = { Psi = [1e-6, 1e-1], T = [1e-4, 1e1] }
        four_deconfinement_step = 1200
        """,
    )
    case = load_cases(path)["a"]
    assert case.four_ylim == {"Psi": [1e-6, 1e-1], "T": [1e-4, 1e1]}
    assert case.four_deconfinement_step == 1200


def test_four_ylim_rejects_non_table(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\nfour_ylim = [1, 2]\n')
    with pytest.raises(CasesError, match="must be a table"):
        load_cases(path)


def test_four_ylim_rejects_non_pair(tmp_path):
    path = _write(
        tmp_path, '[cases.a]\nsteps = [1]\nfour_ylim = { Psi = [1e-6] }\n'
    )
    with pytest.raises(CasesError, match=r"must be \[min, max\]"):
        load_cases(path)


def test_four_ylim_rejects_min_not_less_than_max(tmp_path):
    path = _write(
        tmp_path, '[cases.a]\nsteps = [1]\nfour_ylim = { Psi = [1.0, 1.0] }\n'
    )
    with pytest.raises(CasesError, match="min < max"):
        load_cases(path)


# --- poincare_highlight ---------------------------------------------------------


def test_poincare_highlight_defaults_off_and_empty(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    case = load_cases(path)["a"]
    assert case.poincare_highlight is False
    assert case.poincare_highlight_modes == []
    assert case.poincare_highlight_colors == []


def test_poincare_highlight_modes_and_colors_are_settable(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        poincare_highlight = true
        poincare_highlight_modes = [[3, 2], [2, 1]]
        poincare_highlight_colors = ["red", "blue"]
        """,
    )
    case = load_cases(path)["a"]
    assert case.poincare_highlight is True
    assert case.poincare_highlight_modes == [[3, 2], [2, 1]]
    assert case.poincare_highlight_colors == ["red", "blue"]


def test_poincare_highlight_modes_rejects_non_pair_entries(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\npoincare_highlight_modes = [[3, 2, 1]]\n',
    )
    with pytest.raises(CasesError, match="\\[m, n\\] pairs"):
        load_cases(path)


def test_poincare_highlight_modes_rejects_n_equals_zero(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\npoincare_highlight_modes = [[3, 0]]\n'
        'poincare_highlight_colors = ["red"]\n',
    )
    with pytest.raises(CasesError, match="n=0"):
        load_cases(path)


def test_poincare_highlight_colors_length_mismatch_raises(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\npoincare_highlight_modes = [[3, 2], [2, 1]]\n'
        'poincare_highlight_colors = ["red"]\n',
    )
    with pytest.raises(CasesError, match="same length"):
        load_cases(path)


def test_poincare_highlight_true_without_modes_raises(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\npoincare_highlight = true\n',
    )
    with pytest.raises(CasesError, match="poincare_highlight_modes"):
        load_cases(path)


# --- four_quantities -------------------------------------------------------------


def test_four_quantities_defaults_to_max(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    assert load_cases(path)["a"].four_quantities == ["max"]


def test_four_quantities_settable_to_rational_surface_only(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nfour_quantities = ["rational_surface"]\n',
    )
    assert load_cases(path)["a"].four_quantities == ["rational_surface"]


def test_four_quantities_settable_to_both(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nfour_quantities = ["max", "rational_surface"]\n',
    )
    assert load_cases(path)["a"].four_quantities == ["max", "rational_surface"]


def test_four_quantities_rejects_unknown_value(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nfour_quantities = ["bogus"]\n',
    )
    with pytest.raises(CasesError, match="unknown four_quantities"):
        load_cases(path)


def test_four_quantities_rejects_empty_list(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nfour_quantities = []\n',
    )
    with pytest.raises(CasesError, match="must not be empty"):
        load_cases(path)


# --- steps_for / diag_steps: default -> case -> case+diag override tree ----------


def test_steps_for_falls_back_to_case_steps_when_no_override(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [100, 200]\n')
    case = load_cases(path)["a"]
    assert case.steps_for("four") == [100, 200]
    assert case.steps_for("poincare") == [100, 200]


def test_steps_for_returns_diag_override_as_a_list(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [100, 200]
        [cases.a.four]
        steps = [100, 200, 300, 400]
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps_for("four") == [100, 200, 300, 400]
    assert case.steps_for("poincare") == [100, 200]


def test_diag_steps_override_accepts_a_range_spec(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [100]
        [cases.a.poincare]
        steps = { start = 200, stop = 800, step = 200 }
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps_for("poincare") == [200, 400, 600]


def test_diag_steps_override_table_rejects_unknown_key(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [100]
        [cases.a.four]
        steps = [100]
        four_modes = [[2, 1]]
        """,
    )
    with pytest.raises(CasesError, match="unknown key"):
        load_cases(path)


def test_unrecognised_nested_table_name_hits_the_general_unknown_key_error(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [100]
        [cases.a.not_a_diag]
        steps = [100]
        """,
    )
    with pytest.raises(CasesError, match="unknown key"):
        load_cases(path)


def test_defaults_diag_override_seeds_every_case_and_case_override_replaces_it(tmp_path):
    path = _write(
        tmp_path,
        """
        [defaults.four]
        steps = [1000, 2000]

        [cases.a]
        steps = [100]

        [cases.b]
        steps = [100]
        [cases.b.four]
        steps = [5000, 6000]
        """,
    )
    cases = load_cases(path)
    assert cases["a"].steps_for("four") == [1000, 2000]
    assert cases["b"].steps_for("four") == [5000, 6000]


def test_case_name_is_its_own_run_folder(tmp_path):
    path = _write(
        tmp_path,
        '[cases."qa2.1_g2.3/eta1e-3_RE"]\nsteps = [1]\n',
    )
    case = load_cases(path)["qa2.1_g2.3/eta1e-3_RE"]
    assert case.name == "qa2.1_g2.3/eta1e-3_RE"


def test_folder_key_is_rejected(tmp_path):
    """There is no separate `folder` override -- a case's name is always its
    run folder, so a leftover `folder = ...` line must fail loudly rather
    than silently doing nothing."""
    path = _write(
        tmp_path,
        '[cases.rerun]\nfolder = "qa2.1_g2.3/eta1e-3_RE"\nsteps = [1]\n',
    )
    with pytest.raises(CasesError, match="folder"):
        load_cases(path)


def test_missing_steps_raises(tmp_path):
    path = _write(tmp_path, '[cases.a]\n')
    with pytest.raises(CasesError, match="steps"):
        load_cases(path)


def test_unknown_key_raises(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\ntypo_field = 1\n')
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
    path = _write(tmp_path, '[cases.a]\nsteps = [200]\n')
    assert load_cases(path)["a"].phi_start == 0.0


def test_phi_start_is_settable(tmp_path):
    path = _write(
        tmp_path,
        '[defaults]\nphi_start = 0.7853981634\n\n'
        '[cases.a]\nsteps = [200]\n',
    )
    assert load_cases(path)["a"].phi_start == pytest.approx(0.7853981634)


# --- tor_mode: normalised to a list, validated against the known set -------------


def test_tor_mode_defaults_to_midplane_list(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    assert load_cases(path)["a"].tor_mode == ["midplane"]


def test_tor_mode_bare_string_is_normalised_to_a_list(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\ntor_mode = "average"\n')
    assert load_cases(path)["a"].tor_mode == ["average"]


def test_tor_mode_list_of_several_modes_passes_through(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\n'
        'tor_mode = ["midplane outer", "average"]\n',
    )
    assert load_cases(path)["a"].tor_mode == ["midplane outer", "average"]


def test_tor_mode_rejects_unknown_mode(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\ntor_mode = "sideways"\n')
    with pytest.raises(CasesError, match="unknown tor_mode"):
        load_cases(path)


def test_tor_mode_rejects_unknown_mode_in_a_list(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\ntor_mode = ["midplane", "bogus"]\n',
    )
    with pytest.raises(CasesError, match="unknown tor_mode"):
        load_cases(path)


# --- profile_* knobs: average-only field-line-tracing overrides -----------------


def test_profile_knobs_default_to_jorek_own_defaults(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    case = load_cases(path)["a"]
    assert case.profile_surfaces == 100
    assert case.profile_rad_range == [0.001, 0.999]
    assert case.profile_nmaxsteps == 2500
    assert case.profile_deltaphi == pytest.approx(0.3)


def test_profile_knobs_are_settable(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        profile_surfaces = 250
        profile_rad_range = [0.01, 0.85]
        profile_nmaxsteps = 10000
        profile_deltaphi = 0.05
        """,
    )
    case = load_cases(path)["a"]
    assert case.profile_surfaces == 250
    assert case.profile_rad_range == [0.01, 0.85]
    assert case.profile_nmaxsteps == 10000
    assert case.profile_deltaphi == pytest.approx(0.05)


def test_profile_rad_range_rejects_non_pair(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nprofile_rad_range = [0.5]\n',
    )
    with pytest.raises(CasesError, match="\\[min, max\\]"):
        load_cases(path)


def test_profile_rad_range_rejects_min_not_less_than_max(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nprofile_rad_range = [0.9, 0.1]\n',
    )
    with pytest.raises(CasesError, match="0 <= min < max <= 1"):
        load_cases(path)


def test_profile_rad_range_rejects_out_of_unit_range(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\nprofile_rad_range = [-0.1, 0.9]\n',
    )
    with pytest.raises(CasesError, match="0 <= min < max <= 1"):
        load_cases(path)


# --- theta_* knobs: field-line theta-crossing histogram --------------------------


def test_theta_knobs_default(tmp_path):
    path = _write(tmp_path, '[cases.a]\nsteps = [1]\n')
    case = load_cases(path)["a"]
    assert case.theta_target_psi == pytest.approx(1.05)
    assert case.theta_bins == 500
    assert case.theta_psi_n_range is None
    assert case.theta_wetted_threshold is None


def test_theta_knobs_are_settable(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [1]
        theta_target_psi = 1.1
        theta_bins = 1000
        theta_psi_n_range = [0.2, 0.8]
        theta_wetted_threshold = 0.002
        """,
    )
    case = load_cases(path)["a"]
    assert case.theta_target_psi == pytest.approx(1.1)
    assert case.theta_bins == 1000
    assert case.theta_psi_n_range == [0.2, 0.8]
    assert case.theta_wetted_threshold == pytest.approx(0.002)


def test_theta_wetted_threshold_rejects_non_positive(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\ntheta_wetted_threshold = 0\n',
    )
    with pytest.raises(CasesError, match="must be positive"):
        load_cases(path)


def test_theta_psi_n_range_rejects_non_pair(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\ntheta_psi_n_range = [0.5]\n',
    )
    with pytest.raises(CasesError, match="\\[min, max\\]"):
        load_cases(path)


def test_theta_psi_n_range_rejects_min_not_less_than_max(tmp_path):
    path = _write(
        tmp_path,
        '[cases.a]\nsteps = [1]\ntheta_psi_n_range = [0.8, 0.2]\n',
    )
    with pytest.raises(CasesError, match="min < max"):
        load_cases(path)


def test_theta_hist_steps_override(tmp_path):
    path = _write(
        tmp_path,
        """
        [cases.a]
        steps = [100, 200, 300]
        [cases.a.theta_hist]
        steps = [200]
        """,
    )
    case = load_cases(path)["a"]
    assert case.steps_for("theta_hist") == [200]
    assert case.steps_for("poincare") == [100, 200, 300]
