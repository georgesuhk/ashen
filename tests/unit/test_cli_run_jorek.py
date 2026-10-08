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
    assert "error" in capsys.readouterr().err.lower()


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
    assert "error" in capsys.readouterr().err.lower()


# --- real prepare (uses the symlink-bypass campaign) --------------------------


def test_real_prepare_populates_the_folder(cli_campaign, capsys, symlinks_maybe_bypassed):
    code = main(["shotfile.py"])

    assert code == 0
    assert (cli_campaign / "in_eq").exists()
    assert (cli_campaign / "ffprime_prof.dat").exists()
    out = capsys.readouterr().out
    assert "populated at" in out


def test_second_prepare_always_succeeds(cli_campaign, symlinks_maybe_bypassed):
    """No --replace flag: preparing an already-populated folder again must
    not fail."""
    main(["shotfile.py"])  # first prepare

    code = main(["shotfile.py"])

    assert code == 0


# --- the viewer notebook is written even when the run is refused ----------------


def test_notebook_is_written_even_when_prepare_run_refuses(cli_campaign, capsys):
    """A q0 outside the reachable window stops run_jorek; the viewer, which
    is how that gets fixed, must be there anyway."""
    shot = cli_campaign / "shotfile.py"
    shot.write_text(
        shot.read_text()
        + '\nffprime_method = "q_li"\ncurrent_q0 = 0.2\ncurrent_li = 1.2\ncurrent_qa = 3.0\n'
        + 'bnd_method = "castor"\n'
    )

    code = main(["shotfile.py"])

    captured = capsys.readouterr()
    assert code == 1
    assert (cli_campaign / "case_viewer.ipynb").is_file()
    assert "viewer notebook written" in captured.out
    assert not (cli_campaign / "ffprime_prof.dat").exists()


def test_notebook_is_written_even_when_the_shotfile_does_not_load(cli_campaign, capsys):
    (cli_campaign / "shotfile.py").write_text("eta = 1e-3\n")      # most fields missing

    assert main(["shotfile.py"]) == 1
    assert (cli_campaign / "case_viewer.ipynb").is_file()


def test_dry_run_still_writes_no_notebook(cli_campaign):
    assert main(["shotfile.py", "--dry-run"]) == 0
    assert not (cli_campaign / "case_viewer.ipynb").exists()


def test_window_message_speaks_of_qa(cli_campaign, capsys):
    shot = cli_campaign / "shotfile.py"
    shot.write_text(
        shot.read_text()
        + '\nffprime_method = "q_li"\ncurrent_q0 = 0.2\ncurrent_li = 1.2\ncurrent_qa = 3.0\n'
    )
    in_eq = cli_campaign.parents[1] / "template" / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))

    main(["shotfile.py"])

    err = capsys.readouterr().err
    assert "qa = 3" in err and "q_edge" not in err and "q0 from" in err


# --- --run_r --keep-inputs ------------------------------------------------------


@pytest.fixture
def submitted(monkeypatch):
    """The shell commands a stage would run, instead of running them."""
    from ashen import runner

    commands = []
    monkeypatch.setattr(
        runner, "_run",
        lambda command, cwd, *, dry_run, check=True: commands.append(command) or command,
    )
    return commands


def _snapshot(run_dir: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes() for p in sorted(run_dir.iterdir())
        if p.is_file() and not p.is_symlink()
    }


def _old_folder(run_dir: Path) -> None:
    """Prepare the folder, then put in_main_r's density back to what an
    ashen from before the rho fix wrote (rho_const / 1e20 / 1e20... i.e. 1e-4)."""
    from ashen import namelist as nml

    assert main(["shotfile.py"]) == 0
    nml.set_fields([run_dir / "in_main_r"], {"central_density": 1e-4})


def test_run_r_keep_inputs_touches_only_the_run_length_in_in_main_r(
    cli_campaign, submitted, symlinks_maybe_bypassed, capsys
):
    from ashen import namelist as nml

    _old_folder(cli_campaign)
    shotfile = cli_campaign / "shotfile.py"
    shotfile.write_text(shotfile.read_text().replace("nstep_n = [3000]", "nstep_n = [7000]"))
    before = _snapshot(cli_campaign)

    assert main(["shotfile.py", "--run_r", "--keep-inputs"]) == 0

    after = _snapshot(cli_campaign)
    assert {name for name in after if after[name] != before.get(name)} == {"in_main_r"}
    in_main_r = cli_campaign / "in_main_r"
    assert int(nml.read_field(in_main_r, "nstep_n")) == 7000
    assert float(nml.read_field(in_main_r, "central_density")) == 1e-4      # not renormalised
    assert int(nml.read_field(cli_campaign / "in_main", "nstep_n")) == 3000
    assert len(submitted) == 1 and "./in_main_r log" in submitted[0]
    assert "inputs kept" in capsys.readouterr().out


def test_plain_run_r_refuses_a_folder_whose_density_it_would_change(
    cli_campaign, submitted, symlinks_maybe_bypassed, capsys
):
    _old_folder(cli_campaign)
    before = _snapshot(cli_campaign)

    assert main(["shotfile.py", "--run_r"]) == 1

    assert _snapshot(cli_campaign) == before and submitted == []
    err = capsys.readouterr().err
    assert "central_density = 0.0001" in err and "0.01" in err and "--keep-inputs" in err


def test_plain_run_r_still_prepares_and_submits_a_current_folder(
    cli_campaign, submitted, symlinks_maybe_bypassed
):
    assert main(["shotfile.py"]) == 0
    assert main(["shotfile.py", "--run_r"]) == 0
    assert len(submitted) == 1 and "./in_main_r log" in submitted[0]


def test_keep_inputs_needs_run_r_alone_and_a_prepared_folder(cli_campaign, submitted, capsys):
    assert main(["shotfile.py", "--keep-inputs"]) == 1
    assert main(["shotfile.py", "--run", "--run_r", "--keep-inputs"]) == 1
    assert "--keep-inputs goes with --run_r alone" in capsys.readouterr().err

    assert main(["shotfile.py", "--run_r", "--keep-inputs"]) == 1          # never prepared
    assert "has not been prepared" in capsys.readouterr().err
    assert submitted == []


def test_keep_inputs_dry_run_writes_nothing(cli_campaign, submitted, symlinks_maybe_bypassed, capsys):
    _old_folder(cli_campaign)
    before = _snapshot(cli_campaign)
    assert main(["shotfile.py", "--run_r", "--keep-inputs", "--dry-run"]) == 0
    assert _snapshot(cli_campaign) == before
    out = capsys.readouterr().out
    assert "set_fields ['in_main_r']" in out and "would run:" in out
