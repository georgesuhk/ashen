"""Test helpers shared between tests/unit and tests/golden.

Not itself a test module -- imported by both suites' conftest.py so the
symlink-privilege workaround (see below) isn't duplicated.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def symlinks_supported(tmp_path: Path) -> bool:
    """Probe whether this machine can create symlinks without elevation.

    False on Windows without Developer Mode enabled -- confirmed on the
    primary dev machine this project was built on.
    """
    probe_target = tmp_path / "_probe_target"
    probe_target.mkdir()
    probe_link = tmp_path / "_probe_link"
    try:
        probe_link.symlink_to(probe_target, target_is_directory=True)
        return True
    except OSError:
        return False
    finally:
        if probe_link.exists() or probe_link.is_symlink():
            probe_link.unlink()


def install_symlink_bypass(monkeypatch) -> None:
    """Replace ashen.fs's symlink functions with plain-copy equivalents.

    Used where a real machine's `fs.symlink_dir` would create a *dangling*
    symlink if the source doesn't exist yet (real symlinks don't require
    their target to exist) -- e.g. this dev clone has no `Columbia/jobscripts`
    or `Columbia/jorek` directory. A copy can't dangle, so a missing source
    becomes an empty directory instead, which is close enough for verifying
    prepare_run's namelist/profile/boundary-writing logic (the actual point
    of these tests) without needing every referenced directory to exist.
    """
    from ashen import runner as runner_module

    def fake_symlink_dir(src, dst, link_name=None):
        src, dst = Path(src), Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / (link_name or src.name)
        if target.exists():
            shutil.rmtree(target)
        if src.exists():
            shutil.copytree(src, target)
        else:
            target.mkdir()
        return target

    def fake_symlink_files_in(src, dst):
        src, dst = Path(src), Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        created = []
        if not src.exists():
            return created
        for entry in src.iterdir():
            if entry.is_file():
                target = dst / entry.name
                shutil.copy2(entry, target)
                created.append(target)
        return created

    def fake_symlink_file(src, dst):
        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.write_text("")
        return dst

    monkeypatch.setattr(runner_module.fs, "symlink_dir", fake_symlink_dir)
    monkeypatch.setattr(runner_module.fs, "symlink_files_in", fake_symlink_files_in)
    monkeypatch.setattr(runner_module.fs, "symlink_file", fake_symlink_file)


# --- JOREK main-run logs, for ashen.log_archive / tstep_history -------------

#: Header lines as JOREK's mod_log_params.f90 prints them (the subset the
#: time-step record reads), with this run's settings.
JOREK_LOG_HEADER = """\
  GIT revision        : v2.25.01.0-266-gda1ef6b4c-dirty
 n_tor          =             5
 n_plane        =             8
 time_evol_scheme       = "Gears"
 eta                    =   {eta:.4E}
 n_radial               =           60
 n_pol                  =           60
 R_geo                  =   1.3680E+00
 F0                     =   3.7000E+00
 central_density        =   1.0000E-02
 central_mass           =   2.0000E+00
 gmres                  =            T
 gmres_max_iter         =          {max_iter}
 gmres tolerance        =   1.0000E-08
 iter_precon            =           {iter_precon}
 gmres_m                =           20
 gmres_4                =   1.0000E+03
 use_mumps              =            F
 use_pastix             =            F
 use_strumpack          =            T
 re_adv_fact          =   1.0000E-01
 sqrt(mu0*rho0)       =   6.4836E-08
"""


def jorek_log_text(steps, *, first_index=1, t0=None, ending=None, eta=1e-5, max_iter=200,
                   iter_precon=40, killed_mid_step=False) -> str:
    """A JOREK main-run log: header, then one block per step.

    ``steps``: (dt, iterations, refactorised) per step. ``t0``: the restart
    time (a ``restart time :`` line), None for a run from t = 0. ``ending``:
    "gmresfail", "nan" or "sigterm" for the abort message after the last
    step. ``killed_mid_step``: a last banner with no solve after it."""
    out = [JOREK_LOG_HEADER.format(eta=eta, max_iter=max_iter, iter_precon=iter_precon)]
    if t0 is not None:
        out.append(f" *  restart time : {first_index - 1:6d} {t0:13.6f} *\n")
    previous = steps[0][0] if steps else 0.0
    for k, (dt, iters, refactor) in enumerate(steps):
        out.append(f" *   time step : {1:7d}{k + 1:7d}{first_index + k:7d}"
                   f"{dt:14.5f}{previous:14.5f}  *\n")
        out.append(" Solving MHD system using iterative solver\n")
        if refactor:
            out.append(" Elapsed time updating preconditioner (s):  0.27\n")
        out.append(f"{'Number of iterations: ':>32s}{iters:5d}\n")
        previous = dt
    if killed_mid_step:
        k = len(steps)
        out.append(f" *   time step : {1:7d}{k + 1:7d}{first_index + k:7d}"
                   f"{previous:14.5f}{previous:14.5f}  *\n")
    if ending == "gmresfail":
        out.append(f">>>>> NO CONVERGENCE AFTER {steps[-1][1]:06d} ITERATIONS. ABORTING <<<<<\n")
    elif ending == "nan":
        out.append(" >>>>> NaNs DETECTED: EXITING THE CODE <<<<<\n")
    elif ending == "sigterm":
        out.append(" >>>>> SIGTERM RECEIVED: EXITING THE CODE <<<<<\n")
    return "".join(out)
