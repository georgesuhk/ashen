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


def test_first_prepare_never_needs_replace(synthetic_campaign, tmp_path):
    """The corrected fix for bug #2: run_dir is always the cwd, so it always
    already exists by the time prepare_run runs (you have to cd into it
    first) -- bare existence can't be what --replace gates, or the *default*
    case would fail every time, which is worse than the original bug. What
    matters is whether the folder was already populated by a previous run
    (checked via in_eq), not whether the directory itself exists."""
    site, template_dir, params = synthetic_campaign

    # must not raise, with no --replace, against a fresh (existing but
    # unpopulated) directory -- the normal first-time-use case.
    prepare_run(params, site, tmp_path / "rundir", dry_run=True, replace=False)


def test_second_prepare_without_replace_is_refused(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    prepare_run(params, site, run_dir, dry_run=False)  # first prepare succeeds

    with pytest.raises(FileExistsError, match="in_eq"):
        prepare_run(params, site, run_dir, dry_run=False, replace=False)


def test_second_prepare_with_replace_succeeds(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    prepare_run(params, site, run_dir, dry_run=False)

    prepare_run(params, site, run_dir, dry_run=False, replace=True)  # must not raise


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


def test_real_run_empty_preexisting_folder_does_not_need_replace(synthetic_campaign, tmp_path, symlinks_maybe_bypassed):
    """The normal first-time case: a user mkdir's an empty run folder
    themselves before invoking the tool. Must succeed with no --replace --
    see test_second_prepare_without_replace_is_refused for the case that
    should actually fail (a folder already populated by a prior run)."""
    site, template_dir, params = synthetic_campaign
    run_dir = tmp_path / "rundir"
    run_dir.mkdir()

    prepare_run(params, site, run_dir, dry_run=False, replace=False)  # must not raise

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
