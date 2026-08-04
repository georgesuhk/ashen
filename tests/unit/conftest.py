"""Shared fixtures for runner.py tests: a synthetic site + template tree.

Built once here rather than per-test because prepare_run needs a fairly
complete campaign layout (template/copy, template/symlink/*, a CASTOR3D
cotrans tree) to run at all -- assembling that inline in every test would
dwarf the test itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashen.config import Launch, Site
from ashen.shotfile import ShotParams
from support import install_symlink_bypass, symlinks_supported


@pytest.fixture
def require_symlinks(tmp_path):
    """Skip a test on machines that can't create symlinks (e.g. Windows
    without Developer Mode). Shared by test_fs.py and test_runner.py.
    """
    if not symlinks_supported(tmp_path):
        pytest.skip("symlinks not permitted on this machine (no Developer Mode / not root)")


@pytest.fixture
def symlinks_maybe_bypassed(monkeypatch, tmp_path):
    """Runs real symlink creation where supported; on a machine that can't
    (this Windows box without Developer Mode), replaces ashen.fs's symlink
    functions with plain-copy equivalents instead (see tests/support.py).

    Exists so prepare_run's namelist/profile/boundary-writing logic -- the
    part that actually matters for correctness -- gets exercised on every
    machine, not just ones with symlink privileges. It intentionally does NOT
    verify real symlinks are created (require_symlinks tests do that); it
    verifies everything downstream of "the folder got populated somehow".
    """
    if symlinks_supported(tmp_path):
        return  # real symlinks work here -- nothing to bypass
    install_symlink_bypass(monkeypatch)


def _write_two_col(path, x, y):
    with open(path, "w", encoding="utf-8") as f:
        for xi, yi in zip(x, y):
            f.write(f"   {xi:.8E}     {yi:.8E}\n")


NAMELIST_BODY = """\
&in1
 restart = {restart}
 eta = 1.d-6
 central_density = 1.d-2
 freeboundary = .t.
 tstep_n = 1
 nstep_n = 1200
 nout = 50
 R_geo = 1.368
 ffprime_file = 'ffprime_prof.dat'
 T_file       = 'T_prof.dat'
 rho_file     = 'rho_prof.dat'
&end
"""


@pytest.fixture
def synthetic_campaign(tmp_path):
    """Builds a full synthetic site + template + CASTOR3D fixture tree.

    Returns (site, template_dir, params) where params.castor_suffix="TEST"
    and params.exe names a file that exists under site.exe.
    """
    root = tmp_path / "campaign"
    exe_dir = root / "exe"
    jobscripts_dir = root / "jobscripts"
    jorek_dir = root / "jorek"
    jorek_re_dir = root / "jorek_re"
    template_dir = root / "template"
    castor_root = root / "castor"

    for d in (exe_dir, jobscripts_dir, template_dir, castor_root):
        d.mkdir(parents=True)
    (jorek_dir / "util").mkdir(parents=True)
    (jorek_re_dir / "util").mkdir(parents=True)
    (exe_dir / "jorek_test_exe").write_text("#!/bin/sh\n")
    (jobscripts_dir / "23h").write_text("#!/bin/sh\n")

    copy_dir = template_dir / "copy"
    copy_dir.mkdir()
    (copy_dir / "in_eq").write_text(NAMELIST_BODY.format(restart=".f."))
    (copy_dir / "in_main").write_text(NAMELIST_BODY.format(restart=".f."))
    (copy_dir / "in_main_r").write_text(NAMELIST_BODY.format(restart=".t."))
    (copy_dir / "input_starwall").write_text("&PARAMS\n i_response=2\n&end\n")

    for sub in ("base", "RE", "standard", "starwall"):
        (template_dir / "symlink" / sub).mkdir(parents=True)
    (template_dir / "symlink" / "base" / "submit_jorek.sh").write_text("#!/bin/sh\n")
    (template_dir / "symlink" / "base" / "stpts").write_text("# stpts\n")
    (template_dir / "symlink" / "RE" / "jorek2_poincare").write_text("x")
    (template_dir / "symlink" / "RE" / "jorek2_postproc").write_text("x")
    (template_dir / "symlink" / "standard" / "jorek2_poincare").write_text("x")
    (template_dir / "symlink" / "standard" / "jorek2_postproc").write_text("x")
    (template_dir / "symlink" / "starwall" / "starwall-response_qa2.1_g2.300.dat").write_text(
        "1.0 2.0\n"
    )

    # --- synthetic CASTOR3D fixtures, same shape as test_profiles.py's ---
    machine = castor_root / "TESTMACHINE"
    cotrans_dir = machine / "equilib" / "cotrans" / "qa2.1" / "a1.00_g2.300"
    castor_dir = machine / "castor3d" / "eta1e-6_rf" / "qa2.1" / "a1.00_g2.300"
    cotrans_dir.mkdir(parents=True)
    castor_dir.mkdir(parents=True)

    n = 150
    index = np.arange(n)
    psi = 1.0 * (1 - np.exp(-index / 100.0))
    _write_two_col(cotrans_dir / "xn_fpol_stor0_TEST", index, psi)

    jpol = 1.0e7 * np.exp(-index / 300.0)
    jpol_full = np.concatenate([[jpol[0]], jpol])
    _write_two_col(cotrans_dir / "xn_hjpol_stor0_TEST", np.arange(n + 1), jpol_full)

    pres = 100.0 * (1 - index / n)
    pres_full = np.concatenate([[pres[0]], pres])
    _write_two_col(cotrans_dir / "xn_hpres_stor0_TEST", np.arange(n + 1), pres_full)

    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    R = 1.5 + 0.5 * np.cos(theta)
    Z = 0.5 * np.sin(theta)
    _write_two_col(castor_dir / "xm_plasma_0_TEST_n1", R, Z)

    site = Site(
        source=root / "site.toml",
        root=root,
        paths={
            "exe": exe_dir,
            "template": template_dir,
            "jobscripts": jobscripts_dir,
            "jorek": jorek_dir,
            "jorek_re": jorek_re_dir,
            "castor_root": castor_root,
        },
        launch=Launch(
            interactive_prelude="module load test-interactive/",
            batch_prelude="module unload test-interactive/",
            mpirun="mpirun -n {n}",
            n_jorek=4,
            n_starwall=32,
        ),
    )

    params = ShotParams(
        qa=2.1, g=2.3, eta=1e-3, tstep_n=[0.03], nstep_n=[3000], nout=100,
        exe="jorek_test_exe", jobscript="23h",
        ffprime_method="castor", T_method="castor", rho_method="const",
        bnd_method="castor",
        rho_const=1e18,
        castor_suffix="TEST",
        castor_params={
            "machine_folder": str(machine),
            "scan_folder": "eta1e-6_rf",
            "qa": 2.1,
            "g": 2.3,
        },
    )

    return site, template_dir, params
