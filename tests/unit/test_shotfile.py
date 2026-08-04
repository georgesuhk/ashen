"""Tests for shotfile loading and validation.

The central tension this module resolves: shotfiles must stay able to compute
values inline (George's explicit requirement), but a typo in an optional
field must still be caught rather than silently defaulting. See
shotfile.py's module docstring.
"""

from __future__ import annotations

import warnings

import pytest

from ashen.shotfile import ShotfileError, ShotParams, load_shotfile

MINIMAL = """
qa = 2.1
g = 2.3
eta = 1e-3
tstep_n = [0.03]
nstep_n = [3000]
nout = 100
exe = "jorek_model600_ntor1"
jobscript = "23h"
ffprime_method = "file"
T_method = "file"
rho_method = "const"
bnd_method = "file"
rho_const = 1e18
bnd_file = "boundary.dat"
"""


def write_shotfile(tmp_path, body: str, name="shotfile.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --- happy path ----------------------------------------------------------------


def test_loads_minimal_shotfile(tmp_path):
    params = load_shotfile(write_shotfile(tmp_path, MINIMAL))

    assert params.qa == 2.1
    assert params.eta == 1e-3
    assert params.tstep_n == [0.03]


def test_defaults_apply_when_omitted(tmp_path):
    params = load_shotfile(write_shotfile(tmp_path, MINIMAL))

    assert params.extend_bnd is True
    assert params.extend_ratio == 1.2
    assert params.with_refluid is False
    assert params.castor_suffix == "DIIID"
    assert params.namelist_options == {}


def test_the_real_shotfile_loads(repo_root):
    """The actual qa2.1_g2.3/eta1e-3_RE/shotfile.py, unmodified."""
    path = (
        repo_root
        / "Columbia" / "NL_kinks" / "qa2.1_g2.3" / "eta1e-3_RE" / "shotfile.py"
    )
    if not path.exists():
        pytest.skip("real shotfile not present in this checkout")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = load_shotfile(path)

    assert params.qa == 2.1
    assert params.g == 2.3
    assert params.with_refluid is True
    assert params.re_initialize == 2
    assert params.castor_params["scan_folder"] == "eta1e-6_rf"
    # castor_master_folder was migrated away (see the refactor plan's "Gap
    # found and fixed" note) in favour of castor_params["machine"], resolved
    # against site.toml's castor_root -- so it's no longer present to warn
    # about, but shot_folder/template_folder still are.
    assert params.castor_params["machine"] == "DIIID_low_pres"
    # n0 = 1e18 is a scratch variable feeding rho_const -- must not raise.
    deprecated = {str(w.message) for w in caught}
    assert any("shot_folder" in msg for msg in deprecated)
    assert any("template_folder" in msg for msg in deprecated)


@pytest.fixture
def repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


# --- scratch variables must not be flagged --------------------------------------


def test_unrelated_scratch_variable_is_allowed(tmp_path):
    """n0 used only to compute rho_const -- the pattern the real shotfile uses."""
    body = MINIMAL + "\nn0 = 1e18\nrho_const = n0\n"

    params = load_shotfile(write_shotfile(tmp_path, body))

    assert params.rho_const == 1e18


def test_arbitrary_helper_names_are_allowed(tmp_path):
    body = MINIMAL + "\nsome_local_note = 'internal kink investigation'\n"

    load_shotfile(write_shotfile(tmp_path, body))  # must not raise


def test_imports_and_functions_are_ignored(tmp_path):
    body = "import os\nfrom pathlib import Path\n\ndef helper():\n    return 1\n\n" + MINIMAL

    load_shotfile(write_shotfile(tmp_path, body))  # must not raise


# --- the typo catcher ------------------------------------------------------------


def test_typo_of_an_optional_field_is_caught(tmp_path):
    body = MINIMAL + "\nfreebondary = False\n"

    with pytest.raises(ShotfileError, match="freebondary.*freeboundary"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_typo_of_a_required_field_is_caught_by_the_same_fuzzy_match(tmp_path):
    """The fuzzy matcher doesn't care whether the field is required or
    optional -- a close-spelling typo of a required field is caught the same
    way, with an even more precise message than a bare 'missing' would give."""
    body = MINIMAL.replace("eta = 1e-3", "etaa = 1e-3")

    with pytest.raises(ShotfileError, match="etaa.*eta"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_required_field_omitted_entirely_reports_missing(tmp_path):
    """No typo attempt at all (not just a near-miss) -- the fallback path."""
    body = MINIMAL.replace("eta = 1e-3\n", "")

    with pytest.raises(ShotfileError, match="missing required.*eta"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_unrelated_short_name_is_not_flagged_as_a_typo(tmp_path):
    """A name with no close match to any real field is scratch, not a typo."""
    body = MINIMAL + "\nn0 = 1e18\n"

    load_shotfile(write_shotfile(tmp_path, body))  # must not raise


# --- deprecated fields -----------------------------------------------------------


def test_deprecated_fields_warn_but_still_load(tmp_path):
    body = MINIMAL + '\nshot_folder = "/old/path"\ntemplate_folder = "/old/tmpl"\n'

    with pytest.warns(UserWarning, match="shot_folder"):
        params = load_shotfile(write_shotfile(tmp_path, body))

    assert params.qa == 2.1  # loaded despite the deprecated fields


def test_deprecated_fields_do_not_trigger_the_typo_checker(tmp_path):
    """shot_folder/template_folder/castor_master_folder are removed before
    the fuzzy-match pass runs, so they warn once, not warn-then-also-error."""
    body = MINIMAL + '\ncastor_master_folder = "/old"\n'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_shotfile(write_shotfile(tmp_path, body))  # must not raise


# --- cross-field requirements ----------------------------------------------------


def test_castor_params_required_when_any_method_is_castor(tmp_path):
    body = MINIMAL.replace('bnd_method = "file"', 'bnd_method = "castor"')

    with pytest.raises(ShotfileError, match="castor_params"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_castor_params_not_required_when_no_method_is_castor(tmp_path):
    load_shotfile(write_shotfile(tmp_path, MINIMAL))  # all "file"/"const" -- fine


def test_castor_params_needs_machine_or_machine_folder(tmp_path):
    """Neither "machine" (subfolder name, resolved against site.toml's
    castor_root) nor the legacy absolute "machine_folder" is present --
    should fail loudly at load time, not with a KeyError deep inside
    runner.py's prepare_run."""
    body = MINIMAL.replace(
        'bnd_method = "file"', 'bnd_method = "castor"'
    ) + 'castor_params = {"scan_folder": "eta1e-6_rf", "qa": 2.1, "g": 2.3}\n'

    with pytest.raises(ShotfileError, match="machine"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_castor_params_machine_key_is_sufficient(tmp_path):
    body = MINIMAL.replace(
        'bnd_method = "file"', 'bnd_method = "castor"'
    ) + (
        'castor_params = {"machine": "DIIID_low_pres", '
        '"scan_folder": "eta1e-6_rf", "qa": 2.1, "g": 2.3}\n'
    )

    params = load_shotfile(write_shotfile(tmp_path, body))
    assert params.castor_params["machine"] == "DIIID_low_pres"


def test_rho_const_required_for_const_method(tmp_path):
    body = MINIMAL.replace("rho_const = 1e18\n", "")

    with pytest.raises(ShotfileError, match="rho_const"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_bnd_file_required_for_file_method(tmp_path):
    body = MINIMAL.replace('bnd_file = "boundary.dat"\n', "")

    with pytest.raises(ShotfileError, match="bnd_file"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_re_fields_required_when_with_refluid(tmp_path):
    body = MINIMAL + "\nwith_refluid = True\nexe = \"jorek_model600_RE\"\n"

    with pytest.raises(ShotfileError, match="re_initialize"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_re_fields_satisfied_when_with_refluid(tmp_path):
    body = MINIMAL + """
with_refluid = True
exe = "jorek_model600_RE"
re_initialize = 2
initial_re_current_fraction = 1
vpar_re_sign = 1
re_adv_fact = 0.01
Dre_num = 1e-12
Dre_par = 1e-6
"""

    params = load_shotfile(write_shotfile(tmp_path, body))

    assert params.re_initialize == 2


def test_with_refluid_without_re_in_exe_name_warns(tmp_path):
    """Soft check, ported as a warning (not an error) -- matches old behaviour."""
    body = MINIMAL + """
with_refluid = True
re_initialize = 2
initial_re_current_fraction = 1
vpar_re_sign = 1
re_adv_fact = 0.01
Dre_num = 1e-12
Dre_par = 1e-6
"""
    # exe stays "jorek_model600_ntor1" -- no "RE" in the name

    with pytest.warns(UserWarning, match="RE"):
        load_shotfile(write_shotfile(tmp_path, body))


# --- missing required fields ------------------------------------------------------


def test_missing_required_field_names_it(tmp_path):
    body = MINIMAL.replace("qa = 2.1\n", "")

    with pytest.raises(ShotfileError, match="qa"):
        load_shotfile(write_shotfile(tmp_path, body))


def test_multiple_missing_fields_all_named(tmp_path):
    body = MINIMAL.replace("qa = 2.1\n", "").replace("g = 2.3\n", "")

    with pytest.raises(ShotfileError) as excinfo:
        load_shotfile(write_shotfile(tmp_path, body))

    assert "qa" in str(excinfo.value)
    assert "g" in str(excinfo.value)


# --- ShotParams direct construction (bypassing the loader) -----------------------


def test_shotparams_validates_even_when_constructed_directly():
    """__post_init__ runs regardless of how the object is built."""
    with pytest.raises(ShotfileError, match="rho_const"):
        ShotParams(
            qa=2.1, g=2.3, eta=1e-3, tstep_n=[0.03], nstep_n=[3000], nout=100,
            exe="x", jobscript="y", ffprime_method="file", T_method="file",
            rho_method="const", bnd_method="file", bnd_file="b.dat",
        )
