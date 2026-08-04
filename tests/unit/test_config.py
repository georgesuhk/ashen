"""Tests for site discovery and path resolution.

The headline property is rename-independence: renaming or moving a campaign
must not require editing anything. ``test_rename_independence`` is the
regression test for the whole point of the package.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ashen.config import (
    ENV_VAR,
    REQUIRED_PATHS,
    Diagnostics,
    Site,
    SiteConfigError,
    find_site_file,
    load_site,
)

SITE_BODY = """\
[paths]
exe        = "./exe"
template   = "./template"
jobscripts = "../jobscripts"
jorek      = "../jorek"
jorek_re   = "../jorek_RE"
castor_root = "/elsewhere/castor3d"

[launch]
interactive_prelude = "module load impi-interactive/"
batch_prelude       = "module unload impi-interactive/1.0"
mpirun              = "mpirun -n {n}"
n_jorek             = 4
n_starwall          = 32
"""


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Never let a real $ASHEN_SITE leak into the tests."""
    monkeypatch.delenv(ENV_VAR, raising=False)


def build_campaign(parent: Path, name: str = "NL_kinks") -> Path:
    """Create a campaign tree mirroring the real Columbia/ layout."""
    campaign = parent / name
    (campaign / "exe").mkdir(parents=True)
    (campaign / "template").mkdir()
    (campaign / "qa2.1_g2.3" / "eta1e-3_RE").mkdir(parents=True)
    (parent / "jobscripts").mkdir()
    (parent / "jorek").mkdir()
    (parent / "jorek_RE").mkdir()
    (campaign / "site.toml").write_text(SITE_BODY, encoding="utf-8")
    return campaign


# --- discovery ---------------------------------------------------------------


def test_upward_search_finds_nearest_site(tmp_path):
    campaign = build_campaign(tmp_path)
    run_dir = campaign / "qa2.1_g2.3" / "eta1e-3_RE"

    found = find_site_file(start=run_dir)

    assert found == (campaign / "site.toml").resolve()


def test_nearest_site_wins_over_outer_one(tmp_path):
    outer = build_campaign(tmp_path, name="outer")
    inner = build_campaign(outer, name="inner")

    found = find_site_file(start=inner / "qa2.1_g2.3" / "eta1e-3_RE")

    assert found == (inner / "site.toml").resolve()


def test_env_var_overrides_upward_search(tmp_path, monkeypatch):
    campaign = build_campaign(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    explicit = other / "site.toml"
    explicit.write_text(SITE_BODY, encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(explicit))

    found = find_site_file(start=campaign / "qa2.1_g2.3")

    assert found == explicit.resolve()


def test_env_var_pointing_at_nothing_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "nope.toml"))

    with pytest.raises(SiteConfigError, match=ENV_VAR):
        find_site_file(start=tmp_path)


def test_missing_site_lists_where_it_looked(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()

    with pytest.raises(SiteConfigError) as excinfo:
        find_site_file(start=bare)

    assert "site.toml" in str(excinfo.value)
    assert str(bare) in str(excinfo.value)


# --- resolution --------------------------------------------------------------


def test_relative_paths_resolve_against_the_config_not_the_cwd(tmp_path, monkeypatch):
    campaign = build_campaign(tmp_path)
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    site = load_site(campaign / "site.toml")

    assert site.exe == campaign / "exe"
    assert site.template == campaign / "template"


def test_parent_relative_paths_escape_the_campaign(tmp_path):
    campaign = build_campaign(tmp_path)

    site = load_site(campaign / "site.toml")

    assert site.jobscripts == tmp_path / "jobscripts"
    assert site.jorek_re == tmp_path / "jorek_RE"


def test_absolute_paths_are_left_alone(tmp_path):
    campaign = build_campaign(tmp_path)

    site = load_site(campaign / "site.toml")

    assert site.castor_root.is_absolute()
    assert site.castor_root.name == "castor3d"


def test_jorek_util_follows_the_refluid_branch(tmp_path):
    """Mirrors run_jorek.py:154-159 -- RE runs link the jorek_RE util tree."""
    campaign = build_campaign(tmp_path)
    site = load_site(campaign / "site.toml")

    assert site.jorek_util(with_refluid=True) == tmp_path / "jorek_RE" / "util"
    assert site.jorek_util(with_refluid=False) == tmp_path / "jorek" / "util"


def test_missing_reports_absent_targets(tmp_path):
    campaign = build_campaign(tmp_path)

    site = load_site(campaign / "site.toml")

    # castor_root is the only one that does not exist in the fixture tree.
    assert site.missing() == ["castor_root"]


# --- the headline property ---------------------------------------------------


def test_rename_independence(tmp_path):
    """Renaming the campaign root must require editing nothing.

    This is the regression test for the entire reason the package exists: the
    old code hardcoded the campaign name in run_jorek.py *and* in every
    shotfile, so renaming NL_kink -> NL_kinks silently broke every run.
    """
    campaign = build_campaign(tmp_path, name="NL_kink")
    before = load_site(start=campaign / "qa2.1_g2.3" / "eta1e-3_RE")
    assert before.exe == campaign / "exe"

    renamed = tmp_path / "NL_kinks"
    campaign.rename(renamed)
    after = load_site(start=renamed / "qa2.1_g2.3" / "eta1e-3_RE")

    # Campaign-relative paths tracked the rename, with no file edited.
    assert after.exe == renamed / "exe"
    assert after.template == renamed / "template"
    assert after.exe != before.exe

    # Paths outside the campaign are unaffected by it.
    assert after.jobscripts == before.jobscripts
    assert after.jorek_re == before.jorek_re
    assert after.castor_root == before.castor_root


def test_whole_campaign_can_move(tmp_path):
    """Same property, one level up: moving the tree also needs no edits."""
    first = tmp_path / "first"
    first.mkdir()
    campaign = build_campaign(first)
    site = load_site(start=campaign / "qa2.1_g2.3")
    assert site.exe == campaign / "exe"

    second = tmp_path / "second"
    first.rename(second)

    moved = load_site(start=second / "NL_kinks" / "qa2.1_g2.3")
    assert moved.exe == second / "NL_kinks" / "exe"
    assert moved.jorek_re == second / "jorek_RE"


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize("dropped", REQUIRED_PATHS)
def test_missing_required_key_names_the_file_and_the_key(tmp_path, dropped):
    campaign = build_campaign(tmp_path)
    site_file = campaign / "site.toml"
    kept = [
        line
        for line in SITE_BODY.splitlines()
        if not line.startswith(dropped)
    ]
    site_file.write_text("\n".join(kept), encoding="utf-8")

    with pytest.raises(SiteConfigError) as excinfo:
        load_site(site_file)

    message = str(excinfo.value)
    assert dropped in message
    assert str(site_file) in message


def test_unknown_launch_key_is_an_error(tmp_path):
    """Typos in site.toml must fail loudly, not be silently ignored."""
    campaign = build_campaign(tmp_path)
    site_file = campaign / "site.toml"
    site_file.write_text(SITE_BODY + '\nn_jorekk = 8\n', encoding="utf-8")

    with pytest.raises(SiteConfigError, match="n_jorekk"):
        load_site(site_file)


def test_unknown_section_is_an_error(tmp_path):
    campaign = build_campaign(tmp_path)
    site_file = campaign / "site.toml"
    site_file.write_text(SITE_BODY + '\n[pathz]\nexe = "./exe"\n', encoding="utf-8")

    with pytest.raises(SiteConfigError, match="pathz"):
        load_site(site_file)


def test_malformed_toml_names_the_file(tmp_path):
    campaign = build_campaign(tmp_path)
    site_file = campaign / "site.toml"
    site_file.write_text("[paths\nexe = ", encoding="utf-8")

    with pytest.raises(SiteConfigError) as excinfo:
        load_site(site_file)

    assert str(site_file) in str(excinfo.value)


def test_unknown_path_key_lookup_lists_the_known_ones(tmp_path):
    campaign = build_campaign(tmp_path)
    site = load_site(campaign / "site.toml")

    with pytest.raises(SiteConfigError, match="exe"):
        site.path("nonexistent")


# --- launch ------------------------------------------------------------------


def test_launch_defaults_apply_when_section_absent(tmp_path):
    campaign = build_campaign(tmp_path)
    site_file = campaign / "site.toml"
    body = SITE_BODY.split("[launch]")[0]
    site_file.write_text(body, encoding="utf-8")

    site = load_site(site_file)

    assert site.launch.n_jorek == 4
    assert site.launch.mpirun_cmd(32) == "mpirun -n 32"


def test_mpirun_cmd_substitutes_rank_count(tmp_path):
    campaign = build_campaign(tmp_path)
    site = load_site(campaign / "site.toml")

    assert site.launch.mpirun_cmd(site.launch.n_starwall) == "mpirun -n 32"


# --- describe ----------------------------------------------------------------


def test_describe_mentions_source_and_flags_missing(tmp_path):
    campaign = build_campaign(tmp_path)
    site = load_site(campaign / "site.toml")

    text = site.describe()

    assert str(site.source) in text
    assert "castor_root" in text
    assert "(missing)" in text


# --- [diagnostics] -----------------------------------------------------------


def test_diagnostics_defaults_when_the_table_is_absent(tmp_path):
    """Every site.toml written before Phase 4b must stay valid."""
    site = load_site(build_campaign(tmp_path) / "site.toml")

    assert site.diagnostics.n_workers == 0
    assert site.diagnostics.omp_threads == 0


def test_diagnostics_values_are_read(tmp_path):
    site_file = build_campaign(tmp_path) / "site.toml"
    site_file.write_text(
        SITE_BODY + "\n[diagnostics]\nn_workers = 3\nomp_threads = 5\n",
        encoding="utf-8",
    )

    site = load_site(site_file)

    assert (site.diagnostics.n_workers, site.diagnostics.omp_threads) == (3, 5)


def test_unknown_diagnostics_key_is_named(tmp_path):
    site_file = build_campaign(tmp_path) / "site.toml"
    site_file.write_text(
        SITE_BODY + "\n[diagnostics]\nn_threads = 3\n", encoding="utf-8"
    )

    with pytest.raises(SiteConfigError, match="n_threads"):
        load_site(site_file)


def test_explicit_diagnostics_values_are_used_verbatim():
    assert Diagnostics(n_workers=2, omp_threads=4).resolve(cpu_count=64) == (2, 4)


def test_derived_split_fills_the_machine():
    """omp_threads caps at 8, then workers take the rest."""
    assert Diagnostics().resolve(cpu_count=32) == (4, 8)
    assert Diagnostics().resolve(cpu_count=4) == (1, 4)
    assert Diagnostics().resolve(cpu_count=1) == (1, 1)


def test_one_explicit_value_constrains_the_other():
    assert Diagnostics(omp_threads=16).resolve(cpu_count=32) == (2, 16)
    assert Diagnostics(n_workers=2).resolve(cpu_count=32) == (2, 8)


def test_oversubscription_warns_but_is_allowed():
    """A login node's cpu_count is not the batch allocation, so an explicit
    pair is not something to refuse."""
    with pytest.warns(UserWarning, match="oversubscribe"):
        assert Diagnostics(n_workers=8, omp_threads=8).resolve(cpu_count=4) == (8, 8)
