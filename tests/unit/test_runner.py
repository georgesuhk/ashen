"""Tests for run-folder preparation and job submission.

Dry-run tests need no privileges and run everywhere -- they exercise every
computation prepare_run does (psi, profiles, boundary, namelist edits) without
touching disk, so they catch the same bugs a real run would. Real-write tests
use `symlinks_maybe_bypassed` (see conftest.py), which falls back to plain
copies on a machine without symlink privileges -- so the writing logic itself
is still verified everywhere; only tests that specifically assert
symlink-ness use `require_symlinks` and are skipped where unsupported.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ashen.config import SiteConfigError
from ashen.namelist import effective_fields
from ashen.paths import read_float
from ashen.runner import (
    prepare_run,
    submit_eq,
    submit_main,
    submit_restart,
    submit_starwall,
)
from ashen.shotfile import ShotfileError


# --- dry run: pure computation, no privileges needed ----------------------------


def test_dry_run_touches_nothing(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=True)

    assert not run_dir.exists()


def test_dry_run_logs_every_stage(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign

    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    joined = "\n".join(result.actions)
    assert "mkdir" in joined
    assert "copy" in joined
    assert "symlink" in joined
    assert "set_fields" in joined
    assert "set_boundary_block" in joined


def test_dry_run_writes_boundary_to_all_three_namelists(synthetic_campaign, tmp_path):
    """The confirmed fix (item 4 in the plan): not just in_eq anymore."""
    site, template_dir, params = synthetic_campaign

    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    boundary_actions = [a for a in result.actions if "set_boundary_block" in a]
    assert len(boundary_actions) == 3
    assert any("in_eq" in a for a in boundary_actions)
    assert any("in_main_r" in a for a in boundary_actions)
    assert any("in_main" in a and "in_main_r" not in a for a in boundary_actions)


def test_dry_run_computes_a_real_psi_edge(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign

    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    assert 0 < result.real_psi_edge < 1


def test_castor_params_machine_resolves_against_site_castor_root(synthetic_campaign, tmp_path):
    """castor_params["machine"] (just the subfolder name) should resolve to
    the exact same machine_folder as the legacy explicit absolute path --
    see the refactor plan's "Gap found and fixed" note on
    castor_master_folder. synthetic_campaign's fixture data lives at
    site.castor_root / "TESTMACHINE", matching castor_params["machine_folder"]
    used elsewhere in this file.
    """
    site, template_dir, params = synthetic_campaign
    legacy_machine_folder = params.castor_params["machine_folder"]
    assert legacy_machine_folder == str(site.castor_root / "TESTMACHINE")

    new_style_params = dataclasses.replace(
        params,
        castor_params={
            k: v for k, v in params.castor_params.items() if k != "machine_folder"
        }
        | {"machine": "TESTMACHINE"},
    )

    legacy_result = prepare_run(params, site, tmp_path / "legacy", dry_run=True)
    new_style_result = prepare_run(new_style_params, site, tmp_path / "new_style", dry_run=True)

    assert new_style_result.real_psi_edge == pytest.approx(legacy_result.real_psi_edge)


def test_castor_params_explicit_machine_folder_wins_over_machine(synthetic_campaign, tmp_path):
    """If both are present, the explicit absolute path takes precedence --
    lets an existing shotfile keep working exactly as before even if
    "machine" is added alongside it for some other reason."""
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(
        params,
        castor_params={**params.castor_params, "machine": "does-not-exist-anywhere"},
    )

    # Would raise (missing CASTOR3D files) if "machine" won instead of
    # "machine_folder".
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)
    assert 0 < result.real_psi_edge < 1


def test_dry_run_still_validates_missing_starwall_response(synthetic_campaign, tmp_path):
    """Validation happens before any side effect, but this specific check
    (starwall response existence) is itself a side-effect-adjacent read that
    happens during the disk-mutating phase -- confirm it still fires."""
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(params, qa=9.9)  # no starwall response for qa=9.9

    with pytest.raises(FileNotFoundError, match="starwall"):
        prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_freeboundary_false_skips_the_starwall_requirement(synthetic_campaign, tmp_path):
    """The fix for bug #1: freeboundary is a real bool now, so False is
    actually respected (the old code tested truthiness of the string '.f.',
    which is always truthy)."""
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(params, qa=9.9, freeboundary=False)

    # must NOT raise, unlike the qa=9.9 + freeboundary=True case above
    prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_allow_other_starwall_skips_the_requirement_too(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(params, qa=9.9, allow_other_starwall=True)

    prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_first_prepare_needs_no_flag(synthetic_campaign, tmp_path):
    """run_dir is always the cwd, so it always already exists by the time
    prepare_run runs (you have to cd into it first) -- must not raise
    against a fresh (existing but unpopulated) directory, the normal
    first-time-use case."""
    site, template_dir, params = synthetic_campaign

    prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_second_prepare_always_succeeds(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    """No --replace gate: prepare_run always repopulates a run folder in
    place, so calling it again against an already-prepared directory must
    not raise."""
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    prepare_run(params, site, run_dir, dry_run=False)  # first prepare succeeds

    prepare_run(params, site, run_dir, dry_run=False)  # must not raise


def test_unimplemented_ffprime_method_raises_before_any_write(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(params, ffprime_method="file")
    run_dir = tmp_path / "rundir"

    with pytest.raises(NotImplementedError, match="ffprime_method"):
        prepare_run(params, site, run_dir, dry_run=True)

    assert not run_dir.exists()


def test_unimplemented_rho_method_raises(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(params, rho_method="prof")

    with pytest.raises(NotImplementedError, match="rho_method"):
        prepare_run(params, site, tmp_path / "rundir", dry_run=True)


# --- namelist_options ---------------------------------------------------------

MODEL_SOURCE = """\
subroutine initialise_parameters(my_id, filename)
implicit none
namelist /in1/  tstep, nstep, eta, visco,                          &
                central_density, freeboundary,                     &
                Dre_num, Dre_par
end subroutine
"""


def _write_model(site, model_number="600", with_refluid=False):
    root = site.jorek_re if with_refluid else site.jorek
    model_dir = root / "models" / f"model{model_number}"
    model_dir.mkdir(parents=True)
    (model_dir / "initialise_parameters.f90").write_text(MODEL_SOURCE, encoding="utf-8")


def test_namelist_options_unknown_parameter_raises_before_any_write(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    _write_model(site)
    params = dataclasses.replace(
        params, exe="jorek_model600_fixed_T_rho", namelist_options={"not_a_real_param": 1.0}
    )
    run_dir = tmp_path / "rundir"

    with pytest.raises(ShotfileError, match="not_a_real_param"):
        prepare_run(params, site, run_dir, dry_run=True)

    assert not run_dir.exists()


def test_namelist_options_known_parameter_is_applied(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    _write_model(site)
    params = dataclasses.replace(
        params, exe="jorek_model600_fixed_T_rho", namelist_options={"visco": 5e-7}
    )
    run_dir = tmp_path / "rundir"
    (site.exe / "jorek_model600_fixed_T_rho").write_text("#!/bin/sh\n")

    prepare_run(params, site, run_dir, dry_run=False)

    for name in ("in_eq", "in_main", "in_main_r"):
        assert effective_fields(run_dir / name)["visco"] == pytest.approx(5e-7)


def test_namelist_options_without_model_number_in_exe_raises(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(params, namelist_options={"visco": 5e-7})

    with pytest.raises(ShotfileError, match="model number"):
        prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_namelist_options_missing_model_source_raises(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(
        params, exe="jorek_model999_fixed_T_rho", namelist_options={"visco": 5e-7}
    )

    with pytest.raises(ShotfileError, match="model999"):
        prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_namelist_options_empty_dict_needs_no_model_source(synthetic_campaign, tmp_path):
    """The common case (no namelist_options at all) must not require a
    models/ tree to exist -- most dev clones won't have one."""
    site, template_dir, params = synthetic_campaign

    prepare_run(params, site, tmp_path / "rundir", dry_run=True)  # must not raise


# --- starwall_options -----------------------------------------------------------

STARWALL_INPUT_SOURCE = """\
subroutine input
namelist / params / i_response, n_harm, n_tor, nv, delta, n_points, nwall, iwall
end subroutine
"""

STARWALL_WALL_SOURCE = """\
namelist / params_wall / nwu, nwv, mn_w, n_w, m_w, rc_w, rs_w, zc_w, zs_w, eta_thin_w
"""


def _with_starwall_source(site, tmp_path):
    """A synthetic_campaign site has no 'starwall' path -- add one pointing
    at a minimal fake checkout, mirroring the two real namelist sources."""
    starwall_dir = tmp_path / "starwall.git" / "src_3d"
    starwall_dir.mkdir(parents=True)
    (starwall_dir / "input.f90").write_text(STARWALL_INPUT_SOURCE, encoding="utf-8")
    (starwall_dir / "surface_wall.f90").write_text(STARWALL_WALL_SOURCE, encoding="utf-8")
    return dataclasses.replace(
        site, paths={**site.paths, "starwall": starwall_dir.parent}
    )


def test_starwall_options_needs_site_starwall_path(synthetic_campaign, tmp_path):
    """synthetic_campaign's site has no 'starwall' key -- must fail with a
    clear site-config error, not an AttributeError or KeyError."""
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(params, starwall_options={"i_response": 1})

    with pytest.raises(SiteConfigError, match="starwall"):
        prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_starwall_options_unknown_parameter_raises_before_any_write(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    site = _with_starwall_source(site, tmp_path)
    params = dataclasses.replace(params, starwall_options={"not_a_real_param": 1})
    run_dir = tmp_path / "rundir"

    with pytest.raises(ShotfileError, match="not_a_real_param"):
        prepare_run(params, site, run_dir, dry_run=True)

    assert not run_dir.exists()


def test_starwall_options_known_parameter_from_either_group_is_accepted(synthetic_campaign, tmp_path):
    """i_response is /params/, eta_thin_w is /params_wall/ -- both must
    validate, since starwall_options isn't split by group."""
    site, template_dir, params = synthetic_campaign
    site = _with_starwall_source(site, tmp_path)
    params = dataclasses.replace(
        params, starwall_options={"i_response": 1, "eta_thin_w": 1e-4}
    )

    # Only i_response exists in the fixture's input_starwall template (see
    # conftest.py); eta_thin_w would need create_missing, which starwall
    # writes don't support (see prepare_run's comment) -- so this specific
    # combination must fail at the *write* stage, past validation.
    with pytest.raises(Exception, match="eta_thin_w"):
        prepare_run(params, site, tmp_path / "rundir", dry_run=False)


def test_starwall_options_known_parameter_is_applied(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    site = _with_starwall_source(site, tmp_path)
    params = dataclasses.replace(params, starwall_options={"i_response": 0})
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    assert effective_fields(run_dir / "input_starwall")["i_response"] == pytest.approx(0)


def test_bnd_method_castor_alone_no_longer_name_errors(synthetic_campaign, tmp_path):
    """Regression for bug #5: the old code only checked ffprime/T/rho for
    'castor' membership, so bnd_method='castor' alone (with the others
    something else) raised NameError on castor_dir. Broadened here to check
    all four methods."""
    site, template_dir, params = synthetic_campaign
    # ffprime/T are still "castor" here since "file" is unimplemented and
    # would raise first -- this specifically tests that including bnd_method
    # in the castor-membership check doesn't regress the normal path.
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)
    assert result.real_psi_edge > 0


# --- real writes -------------------------------------------------------------
#
# Content-verification tests use `symlinks_maybe_bypassed`, which runs real
# symlinks where supported and falls back to plain copies otherwise -- so the
# namelist/profile/boundary-writing logic (what actually matters for
# correctness) is exercised on every machine, not only ones with symlink
# privileges. Tests that specifically assert symlink-ness use
# `require_symlinks` instead and are skipped where unsupported.


def test_real_run_symlinks_are_real_symlinks(synthetic_campaign, tmp_path, require_symlinks):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    assert (run_dir / "exe").is_symlink()
    assert (run_dir / "util").is_symlink()


def test_real_run_populates_the_folder(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    assert (run_dir / "in_eq").exists()
    assert (run_dir / "in_main").exists()
    assert (run_dir / "in_main_r").exists()
    assert (run_dir / "ffprime_prof.dat").exists()
    assert (run_dir / "T_prof.dat").exists()
    assert (run_dir / "rho_prof.dat").exists()
    assert (run_dir / "real_psi_edge.dat").exists()


def test_real_run_namelists_carry_the_eta_value(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    for name in ("in_eq", "in_main", "in_main_r"):
        fields = effective_fields(run_dir / name)
        assert fields["eta"] == pytest.approx(params.eta)


def test_real_run_boundary_lands_in_all_three_namelists(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    for name in ("in_eq", "in_main", "in_main_r"):
        fields = effective_fields(run_dir / name)
        assert "r_boundary(1)" in fields
        assert fields["n_boundary"] == pytest.approx(50)


def test_real_run_all_three_namelists_get_the_same_boundary(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    values = [
        effective_fields(run_dir / name)["r_boundary(1)"]
        for name in ("in_eq", "in_main", "in_main_r")
    ]
    assert values[0] == values[1] == values[2]


def test_real_run_real_psi_edge_is_readable(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    result = prepare_run(params, site, run_dir, dry_run=False)

    assert read_float(run_dir / "real_psi_edge.dat") == pytest.approx(result.real_psi_edge)


def test_real_run_profile_files_have_the_right_row_count(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    for name in ("ffprime_prof.dat", "T_prof.dat", "rho_prof.dat"):
        data = np.loadtxt(run_dir / name)
        assert data.shape == (200, 2)


def test_real_run_with_refluid_inserts_re_fields(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    params = dataclasses.replace(
        params,
        with_refluid=True,
        exe="jorek_test_exe_RE",
        re_initialize=2,
        initial_re_current_fraction=1,
        vpar_re_sign=1,
        re_adv_fact=0.01,
        Dre_num=1e-12,
        Dre_par=1e-6,
    )
    (site.exe / "jorek_test_exe_RE").write_text("#!/bin/sh\n")
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir, dry_run=False)

    fields = effective_fields(run_dir / "in_eq")
    assert fields["re_initialize"] == pytest.approx(2)
    assert fields["dre_par"] == pytest.approx(1e-6)


def test_real_run_empty_preexisting_folder_succeeds(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    """The normal first-time case: a user mkdir's an empty run folder
    themselves before invoking the tool."""
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    run_dir.mkdir()

    prepare_run(params, site, run_dir, dry_run=False)  # must not raise

    assert (run_dir / "in_eq").exists()



# --- submission ------------------------------------------------------------------


def test_submit_eq_dry_run_returns_command_without_running(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    command = submit_eq(result.paths, site, params, dry_run=True)

    assert "in_eq" in command
    assert "log_eq" in command
    assert params.exe in command


def test_submit_main_batch_uses_qa_g_eta_in_jobname(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    command = submit_main(result.paths, site, params, interactive=False, dry_run=True)

    assert "submit_jorek.sh" in command
    assert f"{params.g:.1f}_{params.eta:g}" in command
    assert "in_main" in command and "in_main_r" not in command


def test_submit_main_interactive_pipes_to_log_not_log_eq(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    command = submit_main(result.paths, site, params, interactive=True, dry_run=True)

    assert "| tee log" in command
    assert "log_eq" not in command


def test_submit_restart_targets_in_main_r(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    command = submit_restart(result.paths, site, params, dry_run=True)

    assert "in_main_r" in command


def test_submit_starwall_equilibrium_uses_log_eq_not_log(synthetic_campaign, tmp_path):
    """Fix for bug #3: the old code clobbered the main run's log here."""
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    commands = submit_starwall(result.paths, site, params, dry_run=True)

    eq_command = commands[0]
    assert "log_eq" in eq_command
    assert "| tee log " not in eq_command  # not the bare "log" the old bug used


def test_submit_starwall_second_stage_runs_starwall_binary(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    commands = submit_starwall(result.paths, site, params, dry_run=True)

    assert "STARWALL_JOREK_Linux" in commands[1]
    assert "input_starwall" in commands[1]


def test_submit_starwall_both_stages_use_interactive_prelude(synthetic_campaign, tmp_path):
    """Both stages run in the foreground (run_jorek.py:369-395 loads
    impi-interactive for both), neither is a queued batch job -- the
    STARWALL stage previously used batch_prelude by mistake."""
    site, template_dir, params = synthetic_campaign
    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    commands = submit_starwall(result.paths, site, params, dry_run=True)

    assert commands[0].startswith(site.launch.interactive_prelude)
    assert commands[1].startswith(site.launch.interactive_prelude)
    assert site.launch.batch_prelude not in commands[1]


def test_submit_starwall_eq_stage_does_not_check_exit_status(
    synthetic_campaign, tmp_path, symlinks_maybe_bypassed, monkeypatch
):
    """The equilibrium stage is expected to abort here -- with no
    starwall-response.dat yet (prepare_run(..., run_sw=True) skips
    symlinking one in), JOREK exports the wall geometry STARWALL needs and
    then aborts trying to read the response it doesn't have. That expected
    failure must not stop the STARWALL stage from running afterward."""
    import ashen.runner as runner_module

    checks = []

    def fake_run(*args, **kwargs):
        checks.append(kwargs["check"])

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    result = prepare_run(params, site, run_dir, dry_run=False, run_sw=True)
    (run_dir / "starwall-response.dat").write_text("computed response\n")

    submit_starwall(result.paths, site, params, dry_run=False)

    assert checks == [False, True]


def test_submit_starwall_archives_the_response(synthetic_campaign, tmp_path, symlinks_maybe_bypassed, monkeypatch):
    """Archiving (copy + remove) happens unconditionally after the two launch
    commands, which this mocks out -- they need a POSIX shell with `module`
    and real JOREK/STARWALL binaries, neither available in this test
    environment (same constraint as the legacy pipeline, see CLAUDE.md).

    prepare_run is called with run_sw=True, matching how the CLI wires
    --run_sw through: this is the case where STARWALL is about to *generate*
    the response, not consume an existing one -- see
    test_prepare_run_skips_starwall_symlink_when_run_sw for the regression
    this guards (a real bug found running the real symlink path on the HPC;
    the Windows dev clone's copy-based symlink bypass couldn't catch it)."""
    import ashen.runner as runner_module

    monkeypatch.setattr(runner_module.subprocess, "run", lambda *a, **k: None)

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    result = prepare_run(params, site, run_dir, dry_run=False, run_sw=True)
    (run_dir / "starwall-response.dat").write_text("computed response\n")

    submit_starwall(result.paths, site, params, dry_run=False)

    archived = (
        template_dir / "symlink" / "starwall"
        / f"starwall-response_qa{params.qa:.1f}_g{params.g:.3f}.dat"
    )
    assert archived.read_text() == "computed response\n"
    assert not (run_dir / "starwall-response.dat").exists()


def test_prepare_run_skips_starwall_symlink_when_run_sw(
    synthetic_campaign, tmp_path, symlinks_maybe_bypassed
):
    """Regression for a real bug found running with real POSIX symlinks on
    the HPC (the Windows copy-based symlink bypass masked it): without
    run_sw=True, prepare_run symlinks starwall-response.dat straight to the
    archived file (site.template/symlink/starwall/...), so a later
    submit_starwall's copy2-onto-itself raises shutil.SameFileError. Mirrors
    run_jorek.py:146's `if freeboundary and not run_sw_flag:` guard, which
    the initial port dropped."""
    site, _, params = synthetic_campaign

    run_dir_generating = tmp_path / "rundir_sw"
    prepare_run(params, site, run_dir_generating, dry_run=False, run_sw=True)
    assert not (run_dir_generating / "starwall-response.dat").exists()

    run_dir_consuming = tmp_path / "rundir_main"
    prepare_run(params, site, run_dir_consuming, dry_run=False, run_sw=False)
    assert (run_dir_consuming / "starwall-response.dat").exists()


# --- profiles and boundary without CASTOR3D -----------------------------------


def _current_params(params, run_dir, **changes):
    """synthetic_campaign's params with a current profile, flat T and a
    boundary file in run_dir, and no CASTOR3D source at all."""
    run_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 1, 101)
    np.savetxt(run_dir / "j_prof.dat", np.column_stack((x, 3.0e6 * (1 - x) ** 2)))
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    np.savetxt(
        run_dir / "bnd.dat",
        np.column_stack((1.5 + 0.5 * np.cos(theta), 0.6 * np.sin(theta))),
    )
    fields = dict(
        ffprime_method="current", T_method="const", bnd_method="file",
        current_file="j_prof.dat", current_R0=1.5, T_const=100.0,
        bnd_file="bnd.dat", bnd_file_is_plasma=True, castor_params=None,
    )
    return dataclasses.replace(params, **(fields | changes))


def test_current_method_needs_no_castor(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    from ashen.physics import MU_0

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _current_params(params, run_dir)

    result = prepare_run(params, site, run_dir)

    assert result.real_psi_edge == pytest.approx(1 / params.extend_ratio)
    ffprime = np.loadtxt(run_dir / "ffprime_prof.dat")
    assert ffprime.shape == (200, 2)
    assert ffprime[0, 0] == 0.0 and ffprime[-1, 0] == pytest.approx(1.0)
    assert ffprime[0, 1] == pytest.approx(-MU_0 * 1.5 * 3.0e6)
    # the plasma edge sits at real_psi_edge; beyond it, vacuum
    assert np.all(ffprime[ffprime[:, 0] >= result.real_psi_edge, 1] == pytest.approx(0.0, abs=1e-12))
    assert np.all(ffprime[:, 1] <= 1e-12)

    # the profile JOREK reads is the one asked for, at the plasma's own psi_N
    x = ffprime[:, 0] / result.real_psi_edge
    inside = x <= 1
    np.testing.assert_allclose(
        ffprime[inside, 1], -MU_0 * 1.5 * 3.0e6 * (1 - x[inside]) ** 2, atol=2e-3 * MU_0 * 1.5 * 3.0e6
    )

    t_prof = np.loadtxt(run_dir / "T_prof.dat")
    assert np.all(t_prof[:, 1] == pytest.approx(100.0 * 1.602176634e-19 * MU_0 * 1e18))


def test_plasma_boundary_file_is_expanded(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _current_params(params, run_dir)

    prepare_run(params, site, run_dir)

    np.testing.assert_allclose(
        np.loadtxt(run_dir / "original_bnd.dat"), np.loadtxt(run_dir / "bnd.dat")
    )
    text = (run_dir / "in_bnd").read_text()
    assert text.splitlines()[0].strip() == "n_boundary = 50"
    # outboard midplane: 1.5 + 0.5 * extend_ratio
    assert f"{1.5 + 0.5 * params.extend_ratio:.2f}" in text
    assert "psi_boundary(  1) = 0.00" in text


def test_boundary_file_as_domain_is_used_as_given(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _current_params(params, run_dir, bnd_file_is_plasma=False, psi_bnd=0.25)

    result = prepare_run(params, site, run_dir, dry_run=True)

    assert any("in_bnd (120 points)" in a for a in result.actions)


def test_no_extension_leaves_psi_n_alone(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _current_params(params, run_dir, extend_bnd=False)

    result = prepare_run(params, site, run_dir, dry_run=True)

    assert result.real_psi_edge == 1.0


def test_current_on_r_over_a(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    """A flat j(r/a) is flat in psi_N too, whatever the mapping."""
    from ashen.physics import MU_0

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _current_params(params, run_dir, current_coord="rho", extend_bnd=False)
    x = np.linspace(0, 1, 51)
    np.savetxt(run_dir / "j_prof.dat", np.column_stack((x, np.full_like(x, 2.0e6))))

    prepare_run(params, site, run_dir)

    ffprime = np.loadtxt(run_dir / "ffprime_prof.dat")
    np.testing.assert_allclose(ffprime[:, 1], -MU_0 * 1.5 * 2.0e6, rtol=1e-12)


def test_edge_current_into_vacuum_warns(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _current_params(params, run_dir)
    x = np.linspace(0, 1, 51)
    np.savetxt(run_dir / "j_prof.dat", np.column_stack((x, 2.0e6 - 1.0e6 * x)))

    with pytest.warns(UserWarning, match="plasma edge"):
        prepare_run(params, site, run_dir, dry_run=True)


def test_bad_current_file_is_a_shotfile_error(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _current_params(params, run_dir)
    np.savetxt(run_dir / "j_prof.dat", np.column_stack(([0.0, 0.5, 0.9], [2e6, 1e6, 0.0])))

    with pytest.raises(ShotfileError, match="0 to 1"):
        prepare_run(params, site, run_dir, dry_run=True)


def test_current_method_fields_are_required(synthetic_campaign):
    site, template_dir, params = synthetic_campaign
    with pytest.raises(ShotfileError, match="current_file"):
        dataclasses.replace(params, ffprime_method="current")
    with pytest.raises(ShotfileError, match="current_q0, current_li, current_qa"):
        dataclasses.replace(params, ffprime_method="q_li")
    with pytest.raises(ShotfileError, match="T_const"):
        dataclasses.replace(params, T_method="const")


# --- current profile from q0, l_i, q_edge ---------------------------------------


def _q_li_params(params, run_dir, template_dir, *, f0=3.0, **changes):
    """_current_params, with the profile made from q0/l_i/q_edge and F0 in
    the template's in_eq (the boundary is an ellipse: R0 1.5, a 0.5, kappa 1.2)."""
    in_eq = template_dir / "copy" / "in_eq"
    text = in_eq.read_text()
    if "F0" not in text:
        in_eq.write_text(text.replace("&end", f" F0 = {f0}\n&end", 1))
    fields = dict(
        ffprime_method="q_li", current_file=None, current_R0=None,
        current_q0=1.1, current_li=1.2, current_qa=3.0,
        freeboundary=False,
    )
    return _current_params(params, run_dir, **(fields | changes))


def test_q_li_writes_the_profile_it_was_asked_for(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    from ashen import current_profile as cur
    from ashen.physics import MU_0

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _q_li_params(params, run_dir, template_dir)

    result = prepare_run(params, site, run_dir)

    # j0 from q0 on the axis of the ellipse
    j0 = (3.0 / 1.5) * (1 + 1.2**2) / (MU_0 * 1.5 * 1.2 * 1.1)
    ffprime = np.loadtxt(run_dir / "ffprime_prof.dat")
    assert ffprime[0, 1] == pytest.approx(-MU_0 * 1.5 * j0, rel=1e-6)
    assert np.all(ffprime[ffprime[:, 0] >= result.real_psi_edge, 1] == pytest.approx(0.0, abs=1e-9))

    # j_prof.dat records it, and reads back as a "current" profile
    text = (run_dir / "j_prof.dat").read_text()
    assert "q0 = 1.1, l_i = 1.2, qa = 3" in text
    psi_n, j = cur.load_current_profile(run_dir / "j_prof.dat")
    assert j[0] == pytest.approx(j0, rel=1e-9) and j[-1] == 0.0
    assert psi_n[0] == 0.0 and psi_n[-1] == pytest.approx(1.0)


def test_q_li_profile_is_reusable_as_a_current_file(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    """ffprime_method="current" on the j_prof.dat that "q_li" wrote gives the
    same ffprime_prof.dat."""
    site, template_dir, params = synthetic_campaign
    first = tmp_path / "first"
    prepare_run(_q_li_params(params, first, template_dir), site, first)

    second = tmp_path / "second"
    again = _current_params(params, second, current_R0=None)
    (second / "j_prof.dat").write_text((first / "j_prof.dat").read_text())
    prepare_run(again, site, second)

    np.testing.assert_allclose(
        np.loadtxt(second / "ffprime_prof.dat"), np.loadtxt(first / "ffprime_prof.dat"),
        rtol=1e-9, atol=1e-12,
    )


def test_q_li_out_of_window_is_a_shotfile_error(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _q_li_params(params, run_dir, template_dir, current_q0=0.2)

    with pytest.raises(ShotfileError, match="outside what l_i"):
        prepare_run(params, site, run_dir, dry_run=True)


def test_q_li_needs_a_plasma_boundary(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _q_li_params(params, run_dir, template_dir, bnd_file_is_plasma=False)

    with pytest.raises(ShotfileError, match="plasma boundary"):
        prepare_run(params, site, run_dir, dry_run=True)


def test_q_li_without_f0_says_so(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _q_li_params(params, run_dir, template_dir)
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text("\n".join(l for l in in_eq.read_text().splitlines() if "F0" not in l) + "\n")

    with pytest.raises(ShotfileError, match="needs F0"):
        prepare_run(params, site, run_dir, dry_run=True)


def test_q_li_with_castor_boundary_and_psi(synthetic_campaign, tmp_path):
    """Only the current comes from q0/l_i/q_edge; grid, T and boundary stay CASTOR3D's."""
    site, template_dir, params = synthetic_campaign
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    params = dataclasses.replace(
        params, ffprime_method="q_li", current_q0=1.1, current_li=1.2, current_qa=3.0,
        freeboundary=False,
    )

    result = prepare_run(params, site, tmp_path / "rundir", dry_run=True)

    assert any("j_prof.dat" in a for a in result.actions)

def test_rho_profile_is_in_units_of_central_density(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    """JOREK's rho is n / (central_density * 1e20): rho_const everywhere is 1,
    not rho_const / 1e20 a second time."""

def test_boundary_points_are_not_rounded_to_centimetres(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    """R and Z keep 6 decimals; ".2f" used to put 1 cm kinks in the boundary."""
    from ashen import boundary as bnd_mod
    from ashen.namelist import read_boundary_points

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"

    prepare_run(params, site, run_dir)

    rho = np.loadtxt(run_dir / "rho_prof.dat")
    np.testing.assert_allclose(rho[:, 1], 1.0, rtol=1e-12)
    fields = effective_fields(run_dir / "in_eq")
    assert float(str(fields["central_density"]).lower().replace("d", "e")) == pytest.approx(
        params.rho_const / 1e20
    )

    raw = np.loadtxt(run_dir / "original_bnd.dat")
    R0, Z0 = bnd_mod.boundary_center(raw)
    exact = bnd_mod.downsample_boundary(
        bnd_mod.expand_boundary(raw, R0, Z0, scale=params.extend_ratio), 50
    )
    for name in ("in_bnd", "in_eq", "in_main", "in_main_r"):
        written = np.array(read_boundary_points(run_dir / name))
        np.testing.assert_allclose(written, exact, atol=5.1e-7)
    assert "psi_boundary(  1) = " in (run_dir / "in_bnd").read_text()


# --- a campaign boundary, and the STARWALL response named after it -------------


def _template_params(params, run_dir, template_dir, **changes):
    """A "q_li" run on the campaign boundary boundary1.dat (an ellipse: R0 1.5,
    a 0.5, kappa 1.2), with no qa or g."""
    boundary_dir = template_dir / "symlink" / "boundary"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    np.savetxt(boundary_dir / "boundary1.dat",
               np.column_stack((1.5 + 0.5 * np.cos(theta), 0.6 * np.sin(theta))))
    fields = dict(bnd_method="template", bnd_file="boundary1.dat", bnd_file_is_plasma=False,
                  qa=None, g=None)
    return _q_li_params(params, run_dir, template_dir, **(fields | changes))


def test_starwall_response_is_named_after_the_boundary(synthetic_campaign, tmp_path):
    from ashen.runner import job_name, starwall_response_name

    site, template_dir, params = synthetic_campaign
    assert starwall_response_name(params) == "starwall-response_qa2.1_g2.300.dat"
    assert job_name(params) == "2.3_0.001"

    on_template = _template_params(params, tmp_path / "r", template_dir)
    assert starwall_response_name(on_template) == "starwall-response_boundary1_ext1.2.dat"
    assert job_name(on_template) == "boundary1_0.001"
    assert starwall_response_name(
        dataclasses.replace(on_template, extend_bnd=False)
    ) == "starwall-response_boundary1_noext.dat"
    assert starwall_response_name(
        dataclasses.replace(on_template, extend_ratio=1.35)
    ) == "starwall-response_boundary1_ext1.35.dat"
    # the plasma does not enter: another profile, same response
    assert starwall_response_name(
        dataclasses.replace(on_template, current_q0=1.4, current_li=0.9)
    ) == "starwall-response_boundary1_ext1.2.dat"


def test_qa_and_g_are_needed_only_off_a_campaign_boundary(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    assert _template_params(params, tmp_path / "r", template_dir).qa is None
    with pytest.raises(ShotfileError, match="missing required field.*qa, g"):
        dataclasses.replace(params, qa=None, g=None)
    with pytest.raises(ShotfileError, match="missing required field.*qa, g"):
        _q_li_params(params, tmp_path / "s", template_dir, qa=None, g=None)   # a local file boundary


def test_template_boundary_is_linked_expanded_and_used(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    from ashen.namelist import read_boundary_points
    from ashen.physics import MU_0

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _template_params(params, run_dir, template_dir)

    result = prepare_run(params, site, run_dir)

    source = template_dir / "symlink" / "boundary" / "boundary1.dat"
    np.testing.assert_allclose(np.loadtxt(run_dir / "boundary1.dat"), np.loadtxt(source))
    np.testing.assert_allclose(np.loadtxt(run_dir / "original_bnd.dat"), np.loadtxt(source))
    assert any("symlink" in a and "boundary1.dat" in a for a in result.actions)
    written = np.array(read_boundary_points(run_dir / "in_bnd"))
    assert len(written) == 50 and written[:, 0].max() == pytest.approx(1.5 + 0.5 * 1.2, abs=1e-3)
    # the current profile took R0, a, kappa from it
    j0 = (3.0 / 1.5) * (1 + 1.2**2) / (MU_0 * 1.5 * 1.2 * 1.1)
    assert np.loadtxt(run_dir / "ffprime_prof.dat")[0, 1] == pytest.approx(-MU_0 * 1.5 * j0, rel=1e-6)


def test_template_boundary_is_a_real_symlink(synthetic_campaign, tmp_path, require_symlinks):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    prepare_run(_template_params(params, run_dir, template_dir), site, run_dir)
    assert (run_dir / "boundary1.dat").is_symlink()
    prepare_run(_template_params(params, run_dir, template_dir), site, run_dir)     # again: no error
    assert (run_dir / "boundary1.dat").is_symlink()


def test_unknown_template_boundary_lists_what_is_there(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    params = _template_params(params, tmp_path / "rundir", template_dir, bnd_file="boundary9.dat")
    with pytest.raises(ShotfileError, match="no such campaign boundary.*There: boundary1.dat"):
        prepare_run(params, site, tmp_path / "rundir", dry_run=True)


def test_template_boundary_takes_a_name_not_a_path(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    with pytest.raises(ShotfileError, match="not a path"):
        _template_params(params, tmp_path / "r", template_dir, bnd_file="../elsewhere/b.dat")


def test_freeboundary_run_uses_the_boundary_s_response(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _template_params(params, run_dir, template_dir, freeboundary=True)

    with pytest.raises(FileNotFoundError, match="boundary1_ext1.2"):
        prepare_run(params, site, run_dir)

    (template_dir / "symlink" / "starwall" / "starwall-response_boundary1_ext1.2.dat").write_text("1 2\n")
    prepare_run(params, site, run_dir)
    assert (run_dir / "starwall-response.dat").read_text() == "1 2\n"

    # another current profile on the same boundary needs no new response
    other = tmp_path / "other"
    prepare_run(_template_params(params, other, template_dir, freeboundary=True,
                                 current_q0=1.3, current_li=1.0), site, other)
    assert (other / "starwall-response.dat").read_text() == "1 2\n"


def test_starwall_run_archives_under_the_boundary_s_name(synthetic_campaign, tmp_path):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _template_params(params, run_dir, template_dir, freeboundary=True)
    result = prepare_run(params, site, run_dir, dry_run=True, run_sw=True)

    commands = submit_starwall(result.paths, site, params, dry_run=True)

    assert "starwall-response_boundary1_ext1.2.dat" in commands[-1]


# --- T from a file -------------------------------------------------------------------


def _t_file_params(params, run_dir, template_dir, **changes):
    run_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 1, 41)
    np.savetxt(run_dir / "T.dat", np.column_stack((x, 20.0 + 480.0 * (1 - x) ** 2)))
    fields = dict(T_method="file", T_file="T.dat", T_const=None)
    return _q_li_params(params, run_dir, template_dir, **(fields | changes))


def test_t_file_is_converted_and_held_flat_in_the_vacuum(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    from ashen.physics import MU_0

    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    result = prepare_run(_t_file_params(params, run_dir, template_dir), site, run_dir)

    to_jorek = 1.602176634e-19 * MU_0 * 1e18            # eV -> JOREK, n_0 = rho_const = 1e18
    t_prof = np.loadtxt(run_dir / "T_prof.dat")
    assert t_prof.shape == (200, 2)
    assert t_prof[0, 1] == pytest.approx(500.0 * to_jorek, rel=1e-9)
    outside = t_prof[:, 0] >= result.real_psi_edge
    np.testing.assert_allclose(t_prof[outside, 1], 20.0 * to_jorek, rtol=1e-9)
    x = t_prof[~outside, 0] / result.real_psi_edge       # the plasma's own psi_N
    np.testing.assert_allclose(
        t_prof[~outside, 1], (20.0 + 480.0 * (1 - x) ** 2) * to_jorek, rtol=2e-3
    )
    assert np.all(t_prof[:, 1] > 0)


def test_t_file_without_extension(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    prepare_run(_t_file_params(params, run_dir, template_dir, extend_bnd=False), site, run_dir)
    t_prof = np.loadtxt(run_dir / "T_prof.dat")
    assert t_prof[0, 1] / t_prof[-1, 1] == pytest.approx(500.0 / 20.0, rel=1e-9)


@pytest.mark.parametrize("rows, message", [
    ([(0.0, 100.0), (0.5, 50.0), (0.9, 10.0)], "0 to 1"),
    ([(0.0, 100.0), (0.5, 0.0), (1.0, 10.0)], "positive"),
    ([(0.0, 100.0), (0.6, 50.0), (0.5, 40.0), (1.0, 10.0)], "increase"),
])
def test_bad_t_file_is_a_shotfile_error(synthetic_campaign, tmp_path, rows, message):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    params = _t_file_params(params, run_dir, template_dir)
    np.savetxt(run_dir / "T.dat", np.array(rows))
    with pytest.raises(ShotfileError, match=message):
        prepare_run(params, site, run_dir, dry_run=True)


def test_t_file_is_required_for_the_method(synthetic_campaign):
    site, template_dir, params = synthetic_campaign
    with pytest.raises(ShotfileError, match="T_file"):
        dataclasses.replace(params, T_method="file")
