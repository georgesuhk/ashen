"""ashen.scan and run_jorek --scan -- several run folders from one base shotfile."""

from __future__ import annotations

import tomllib

import numpy as np
import pytest

from ashen import scan as scan_mod
from ashen.cases import load_cases
from ashen.cli.run_jorek import main
from ashen.comparisons import load_comparisons
from ashen.scan import ScanError, format_value, load_scan, plan_scan
from ashen.shotfile import load_shotfile

BASE = """qa = 2.1
g = 2.3
n0 = 1e18
eta = 1e-3   # scanned
tstep_n = [0.03]
nstep_n = [10]
nout = 1
exe = "jorek_test_exe"
jobscript = "23h"
freeboundary = False
rho_const = n0
ffprime_method = "q_li"
current_q0 = 1.0
current_li = 1.0
current_qa = 2.1
T_method = "const"
T_const = 100.0
rho_method = "const"
bnd_method = "file"
bnd_file = "plasma_bnd.dat"
bnd_file_is_plasma = True
"""

SCAN = """base   = "base_shotfile.py"
folder = "qa2.1_li1.0_q01.0"
name   = "eta{eta}"
copy   = ["plasma_bnd.dat"]

[vary]
eta = [1e-3, 1e-4]
"""


@pytest.mark.parametrize("value, text", [
    (1e-3, "1e-3"), (2.5e-4, "2.5e-4"), (1e-10, "1e-10"), (1e5, "1e5"), (0.03, "0.03"),
    (2.1, "2.1"), (2.0, "2.0"), (0.0, "0.0"), (3, "3"), ("RE", "RE"), (True, "True"),
])
def test_format_value(value, text):
    assert format_value(value) == text


@pytest.fixture
def campaign(synthetic_campaign, monkeypatch):
    """synthetic_campaign's tree with a site.toml, a base shotfile, a plasma
    boundary and F0 in the template: what a scan needs. Returns its root."""
    site, template_dir, _ = synthetic_campaign
    root = site.root
    (root / "site.toml").write_text(
        '[paths]\nexe = "./exe"\ntemplate = "./template"\njobscripts = "./jobscripts"\n'
        'jorek = "./jorek"\njorek_re = "./jorek_re"\ncastor_root = "./castor"\n'
        '[launch]\ninteractive_prelude = ""\nbatch_prelude = ""\nmpirun = "mpirun -n {n}"\n'
        "n_jorek = 1\nn_starwall = 1\n"
    )
    in_eq = template_dir / "copy" / "in_eq"
    in_eq.write_text(in_eq.read_text().replace("&end", " F0 = 3.0\n&end", 1))
    (root / "base_shotfile.py").write_text(BASE)
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    np.savetxt(root / "plasma_bnd.dat",
               np.column_stack((1.5 + 0.5 * np.cos(theta), 0.6 * np.sin(theta))))
    (root / "scans").mkdir()
    (root / "scans" / "eta.toml").write_text(SCAN)
    monkeypatch.delenv("ASHEN_SITE", raising=False)
    return root


# --- reading a scan ----------------------------------------------------------------


def _scan(tmp_path, text):
    path = tmp_path / "s.toml"
    path.write_text(text)
    return load_scan(path)


def test_vary_gives_every_combination(tmp_path):
    scan = _scan(tmp_path, 'base = "b.py"\nfolder = "f"\nname = "eta{eta}_q{current_q0}"\n'
                           "[vary]\neta = [1e-3, 1e-4]\ncurrent_q0 = [0.9, 1.1, 1.3]\n")
    assert len(scan.runs) == 6 and scan.varied == ["eta", "current_q0"]
    assert scan.run_name(scan.runs[0]) == "eta1e-3_q0.9"
    assert scan.comparison == "s"


def test_runs_gives_hand_picked_combinations(tmp_path):
    scan = _scan(tmp_path, 'base = "b.py"\nfolder = "f"\nname = "eta{eta}_q{current_q0}"\n'
                           "[[runs]]\neta = 1e-3\ncurrent_q0 = 0.9\n[[runs]]\neta = 1e-5\ncurrent_q0 = 1.3\n")
    assert [scan.run_name(r) for r in scan.runs] == ["eta1e-3_q0.9", "eta1e-5_q1.3"]


def test_a_format_spec_in_name_is_honoured(tmp_path):
    scan = _scan(tmp_path, 'base = "b.py"\nfolder = "f"\nname = "eta{eta:.0e}"\n[vary]\neta = [1e-3]\n')
    assert scan.run_name(scan.runs[0]) == "eta1e-03"


@pytest.mark.parametrize("text, message", [
    ('folder = "f"\nname = "n"\n[vary]\neta = [1]\n', "base is required"),
    ('base = "b"\nfolder = "f"\nname = "n{eta}"\n', "either"),
    ('base = "b"\nfolder = "f"\nname = "n{eta}"\n[vary]\neta = [1]\n[[runs]]\neta = 2\n', "either"),
    ('base = "b"\nfolder = "f"\nname = "n{visco}"\n[vary]\nvisco = [1e-6]\n', "not shotfile fields"),
    ('base = "b"\nfolder = "f"\nname = "same"\n[vary]\neta = [1e-3, 1e-4]\n', "same folder"),
    ('base = "b"\nfolder = "f"\nname = "q{current_q0}"\n[vary]\neta = [1e-3]\n', "does not vary"),
    ('base = "b"\nfolder = "f"\nname = "n{eta}"\nwhat = 1\n[vary]\neta = [1]\n', "unknown key"),
    ('base = "b"\nfolder = "f"\nname = "a/{eta}"\n[vary]\neta = [1]\n', "folder name"),
])
def test_bad_scans_are_refused(tmp_path, text, message):
    with pytest.raises(ScanError, match=message):
        _scan(tmp_path, text)


# --- planning --------------------------------------------------------------------


def test_plan_rewrites_only_the_varied_line(campaign):
    plan = plan_scan(load_scan(campaign / "scans" / "eta.toml"), campaign)
    assert [r.name for r in plan] == ["qa2.1_li1.0_q01.0/eta1e-3", "qa2.1_li1.0_q01.0/eta1e-4"]
    assert all(r.status == "new" for r in plan)
    for run, value in zip(plan, ("0.001", "0.0001")):
        changed = [(a, b) for a, b in zip(BASE.splitlines(), run.shotfile_text.splitlines()) if a != b]
        assert changed == [("eta = 1e-3   # scanned", f"eta = {value}   # scanned")]


def test_plan_refuses_to_vary_a_computed_line(campaign):
    (campaign / "scans" / "rho.toml").write_text(
        'base = "base_shotfile.py"\nfolder = "f"\nname = "n{rho_const}"\n[vary]\nrho_const = [1e18, 1e19]\n'
    )
    with pytest.raises(ScanError, match="rho_const = n0 is computed"):
        plan_scan(load_scan(campaign / "scans" / "rho.toml"), campaign)


def test_plan_needs_the_base_and_the_copied_files(campaign):
    scan = load_scan(campaign / "scans" / "eta.toml")
    (campaign / "plasma_bnd.dat").unlink()
    with pytest.raises(ScanError, match="plasma_bnd.dat"):
        plan_scan(scan, campaign)
    (campaign / "base_shotfile.py").unlink()
    with pytest.raises(ScanError, match="base shotfile"):
        plan_scan(scan, campaign)


# --- run_jorek --scan ------------------------------------------------------------


def test_preview_writes_nothing(campaign, capsys):
    assert main(["--scan", str(campaign / "scans" / "eta.toml")]) == 0
    out = capsys.readouterr().out
    assert "eta = 1e-3" in out and "eta = 1e-4" in out and "Nothing written" in out
    assert not (campaign / "qa2.1_li1.0_q01.0").exists()
    assert not (campaign / "cases.toml").exists()


def test_apply_creates_prepared_run_folders(campaign, tmp_path, monkeypatch, symlinks_maybe_bypassed, capsys):
    monkeypatch.chdir(tmp_path)                 # the campaign comes from the scan file, not from here
    assert main(["--scan", str(campaign / "scans" / "eta.toml"), "--apply"]) == 0

    for name, eta in (("eta1e-3", 1e-3), ("eta1e-4", 1e-4)):
        run = campaign / "qa2.1_li1.0_q01.0" / name
        assert load_shotfile(run / "shotfile.py").eta == eta
        for made in ("plasma_bnd.dat", "ffprime_prof.dat", "j_prof.dat", "in_eq", "case_viewer.ipynb"):
            assert (run / made).is_file(), made
    from ashen.namelist import read_field
    assert read_field(campaign / "qa2.1_li1.0_q01.0" / "eta1e-4" / "in_eq", "eta", float) == 1e-4


def test_apply_adds_cases_and_a_comparison(campaign, symlinks_maybe_bypassed, monkeypatch):
    main(["--scan", str(campaign / "scans" / "eta.toml"), "--apply"])
    monkeypatch.chdir(campaign)

    cases = load_cases(campaign / "cases.toml")
    assert list(cases) == ["qa2.1_li1.0_q01.0/eta1e-3", "qa2.1_li1.0_q01.0/eta1e-4"]
    assert cases["qa2.1_li1.0_q01.0/eta1e-4"].steps == []          # no restarts yet
    comparison = load_comparisons(campaign / "cases.toml", cases)["eta"]
    assert comparison.cases == list(cases) and comparison.x_values == [1e-3, 1e-4]

    # once restarts exist, first_last picks them up with no edit to cases.toml
    run = campaign / "qa2.1_li1.0_q01.0" / "eta1e-4"
    for step in (0, 200, 400):
        (run / f"jorek{step:06d}.h5").write_bytes(b"")
    assert load_cases(campaign / "cases.toml")["qa2.1_li1.0_q01.0/eta1e-4"].steps == [0, 400]


def test_extending_a_scan_touches_only_the_new_run(campaign, symlinks_maybe_bypassed, capsys):
    scan_file = campaign / "scans" / "eta.toml"
    main(["--scan", str(scan_file), "--apply"])
    old = campaign / "qa2.1_li1.0_q01.0" / "eta1e-3"
    (old / "in_eq").write_text("touched by a running job\n")
    cases_before = (campaign / "cases.toml").read_text()
    capsys.readouterr()

    scan_file.write_text(SCAN.replace("[1e-3, 1e-4]", "[1e-3, 1e-4, 1e-5]"))
    assert main(["--scan", str(scan_file), "--apply"]) == 0

    out = capsys.readouterr().out
    assert (old / "in_eq").read_text() == "touched by a running job\n"      # not re-prepared
    assert (campaign / "qa2.1_li1.0_q01.0" / "eta1e-5" / "ffprime_prof.dat").is_file()
    assert out.count("prepared ") == 1 and "already there: not touched" in out
    text = (campaign / "cases.toml").read_text()
    assert text.startswith(cases_before)                                     # append only
    assert text.count('[cases."qa2.1_li1.0_q01.0/eta1e-5"]') == 1
    assert "[comparisons.eta] is already in cases.toml" in out and "1e-05" in out
    assert tomllib.loads(text)["comparisons"]["eta"]["x_values"] == [1e-3, 1e-4]   # not edited


def test_a_different_shotfile_is_left_alone_unless_forced(campaign, symlinks_maybe_bypassed, capsys):
    scan_file = campaign / "scans" / "eta.toml"
    main(["--scan", str(scan_file), "--apply"])
    shot = campaign / "qa2.1_li1.0_q01.0" / "eta1e-3" / "shotfile.py"
    shot.write_text(shot.read_text().replace("nout = 1", "nout = 7"))
    capsys.readouterr()

    main(["--scan", str(scan_file), "--apply"])
    assert "left alone" in capsys.readouterr().out and "nout = 7" in shot.read_text()

    main(["--scan", str(scan_file), "--apply", "--force"])
    assert "nout = 1" in shot.read_text()


def test_stage_flags_launch_only_the_new_runs(campaign, symlinks_maybe_bypassed, monkeypatch):
    launched = []
    monkeypatch.setattr(
        "ashen.cli.run_jorek.submit_main",
        lambda paths, site, params, *, interactive, dry_run=False: launched.append(
            (paths.run_dir.name, interactive)
        ),
    )
    scan_file = campaign / "scans" / "eta.toml"

    main(["--scan", str(scan_file), "--apply", "--run"])
    assert launched == [("eta1e-3", False), ("eta1e-4", False)]

    scan_file.write_text(SCAN.replace("[1e-3, 1e-4]", "[1e-3, 1e-4, 1e-5]"))
    main(["--scan", str(scan_file), "--apply", "--run"])
    assert launched[2:] == [("eta1e-5", False)]


def test_cases_false_leaves_cases_toml_alone(campaign, symlinks_maybe_bypassed):
    scan_file = campaign / "scans" / "eta.toml"
    scan_file.write_text("cases = false\n" + SCAN)
    main(["--scan", str(scan_file), "--apply"])
    assert not (campaign / "cases.toml").exists()


def test_a_run_that_cannot_be_prepared_is_reported_and_the_rest_go_on(campaign, symlinks_maybe_bypassed, capsys):
    scan_file = campaign / "scans" / "q0.toml"
    scan_file.write_text('base = "base_shotfile.py"\nfolder = "q0scan"\nname = "q{current_q0}"\n'
                         'copy = ["plasma_bnd.dat"]\n[vary]\ncurrent_q0 = [0.2, 1.0]\n')
    assert main(["--scan", str(scan_file), "--apply"]) == 1
    assert "outside what l_i" in capsys.readouterr().err
    assert (campaign / "q0scan" / "q1.0" / "ffprime_prof.dat").is_file()


def test_scan_flags_are_checked(campaign, capsys):
    assert main(["--apply"]) == 1
    assert main(["--scan", str(campaign / "scans" / "eta.toml"), "shotfile.py"]) == 1


def test_first_last_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cases.toml").write_text('[cases.a]\nsteps = { first_last = true }\n'
                                         '[cases.b]\nsteps = { first_last = true }\n')
    (tmp_path / "a").mkdir()
    for step in (200, 400, 600):
        (tmp_path / "a" / f"jorek{step:06d}.h5").write_bytes(b"")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "jorek000000.h5").write_bytes(b"")
    cases = load_cases(tmp_path / "cases.toml")
    assert cases["a"].steps == [200, 600] and cases["b"].steps == [0]

    (tmp_path / "cases.toml").write_text('[cases.a]\nsteps = { first_last = true, step = 2 }\n')
    from ashen.cases import CasesError
    with pytest.raises(CasesError, match="first_last"):
        load_cases(tmp_path / "cases.toml")
