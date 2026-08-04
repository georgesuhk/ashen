"""Tests for the run_jorek CLI, calling main() directly rather than spawning
a subprocess -- this is exactly why bin/run_jorek is a thin shim over
ashen.cli.run_jorek.main.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from ashen.cli.run_jorek import main

SHOTFILE_BODY = """
qa = 2.1
g = 2.3
eta = 1e-3
tstep_n = [0.03]
nstep_n = [3000]
nout = 100
exe = "jorek_test_exe"
jobscript = "23h"
ffprime_method = "castor"
T_method = "castor"
rho_method = "const"
bnd_method = "castor"
rho_const = 1e18
castor_suffix = "TEST"
castor_params = {{
    "machine_folder": {machine_folder!r},
    "scan_folder": "eta1e-6_rf",
    "qa": 2.1,
    "g": 2.3,
}}
"""


def _write_site_toml(site, path: Path) -> Path:
    lines = ["[paths]"]
    for key, value in site.paths.items():
        lines.append(f'{key} = {str(value)!r}')
    lines.append("[launch]")
    lines.append(f'interactive_prelude = {site.launch.interactive_prelude!r}')
    lines.append(f'batch_prelude = {site.launch.batch_prelude!r}')
    lines.append(f'mpirun = {site.launch.mpirun!r}')
    lines.append(f'n_jorek = {site.launch.n_jorek}')
    lines.append(f'n_starwall = {site.launch.n_starwall}')
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def cli_campaign(synthetic_campaign, tmp_path, monkeypatch):
    """synthetic_campaign, plus a real site.toml and shotfile.py on disk, cwd
    set to the run folder -- the actual shape a user invokes the CLI from."""
    site, template_dir, params = synthetic_campaign

    site_toml = tmp_path / "campaign" / "site.toml"
    _write_site_toml(site, site_toml)
    # sanity: must be parseable by the real loader, not just written
    tomllib.loads(site_toml.read_text(encoding="utf-8"))

    run_dir = tmp_path / "campaign" / "qa2.1_g2.3" / "rundir"
    run_dir.mkdir(parents=True)
    (run_dir / "shotfile.py").write_text(
        SHOTFILE_BODY.format(machine_folder=params.castor_params["machine_folder"]),
        encoding="utf-8",
    )

    monkeypatch.chdir(run_dir)
    monkeypatch.delenv("ASHEN_SITE", raising=False)
    return run_dir


# --- --show-config -----------------------------------------------------------


def test_show_config_finds_the_site_toml(cli_campaign, capsys):
    code = main(["--show-config"])

    out = capsys.readouterr().out
    assert code == 0
    assert "site.toml" in out
    assert "exe" in out


def test_show_config_missing_site_reports_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASHEN_SITE", raising=False)

    code = main(["--show-config"])

    assert code == 1
    assert "error" in capsys.readouterr().out.lower()


# --- --help / no args --------------------------------------------------------


def test_no_args_prints_help_and_exits_zero(cli_campaign, capsys):
    code = main([])

    assert code == 0
    assert "usage" in capsys.readouterr().out.lower()


# --- dry run -----------------------------------------------------------------


def test_dry_run_prepares_without_writing(cli_campaign, capsys):
    code = main(["shotfile.py", "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Dry run" in out
    assert not (cli_campaign / "in_eq").exists()


def test_dry_run_reports_planned_boundary_writes(cli_campaign, capsys):
    main(["shotfile.py", "--dry-run"])

    out = capsys.readouterr().out
    assert out.count("set_boundary_block") == 3


def test_missing_shotfile_field_reports_error_not_traceback(cli_campaign, capsys):
    (cli_campaign / "shotfile.py").write_text("qa = 2.1\n", encoding="utf-8")

    code = main(["shotfile.py", "--dry-run"])

    assert code == 1
    assert "error" in capsys.readouterr().out.lower()


# --- real prepare (uses the symlink-bypass campaign) --------------------------


def test_real_prepare_populates_the_folder(cli_campaign, capsys, symlinks_maybe_bypassed):
    code = main(["shotfile.py"])

    assert code == 0
    assert (cli_campaign / "in_eq").exists()
    assert (cli_campaign / "ffprime_prof.dat").exists()
    out = capsys.readouterr().out
    assert "populated at" in out


def test_replace_flag_is_threaded_through(cli_campaign, symlinks_maybe_bypassed):
    main(["shotfile.py"])  # first prepare

    code = main(["shotfile.py", "--replace"])  # must not fail on existing folder

    assert code == 0


def test_without_replace_second_prepare_fails(cli_campaign, symlinks_maybe_bypassed):
    main(["shotfile.py"])

    code = main(["shotfile.py"])  # no --replace, folder already populated

    assert code == 1
