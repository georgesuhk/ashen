"""Populating a run folder and submitting JOREK jobs.

Turns the straight-line ``Columbia/run_jorek.py`` script into functions:
``prepare_run`` (folder population, namelist writes, profile generation) and
``submit_*`` (the five job-launch stages). Everything that can be decided
before touching disk is decided first; nothing is written until validation
has passed.

**Fixes applied here, all confirmed with George (see the refactor plan's
"Behaviour changes to confirm" list):**

- ``freeboundary`` is a real ``bool`` on :class:`~ashen.shotfile.ShotParams`
  (not the string ``".t."``/``".f."`` the old code tested truthiness of), so
  the starwall-response requirement now actually respects fixed-boundary runs.
- ``--replace`` now genuinely allows overwriting an existing folder (the old
  ``exist_ok=(not replace_flag)`` had this backwards).
- ``--run_sw`` tees its equilibrium stage into ``log_eq``, not ``log`` --
  the old code clobbered the main run's log.
- The regenerated boundary is written to **all three** namelists
  (``in_eq``, ``in_main``, ``in_main_r``), not just ``in_eq``.
- ``castor_dir``/``psi`` are computed whenever *any* of
  ffprime/T/rho/bnd-method is ``"castor"`` (the old code only checked
  ffprime/T/rho, so ``bnd_method="castor"`` alone raised ``NameError``).
- ``prepare_run``'s ``run_sw`` parameter mirrors ``run_jorek.py:146``'s
  ``not run_sw_flag`` guard on symlinking the archived STARWALL response in
  -- found missing during the initial port when real POSIX symlinks on the
  HPC (not this dev clone's copy-based bypass) turned it into a
  ``shutil.SameFileError`` in :func:`submit_starwall`. See that function's
  docstring.
- ``castor_params["machine_folder"]`` no longer has to be a hand-built
  absolute path in every shotfile: ``castor_params["machine"]`` (just the
  subfolder name, e.g. ``"DIIID_low_pres"``) is now joined onto
  ``site.toml``'s ``castor_root``. ``"machine_folder"`` still wins if a
  shotfile sets it explicitly, for backward compatibility.

**Not fixed here -- see ``KNOWN_ISSUES.md``:** the T-profile grid and
density-independence issues in :func:`ashen.profiles.get_t_profile_from_castor`
are reproduced exactly.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ashen import boundary as bnd_mod
from ashen import fs
from ashen import namelist as nml
from ashen import profiles as prof_mod
from ashen.castor_io import load_two_col_data
from ashen.config import Site
from ashen.paths import RunPaths, write_float
from ashen.shotfile import ShotfileError, ShotParams

__all__ = [
    "PreparedRun",
    "prepare_run",
    "submit_eq",
    "submit_main",
    "submit_restart",
    "submit_starwall",
]

#: Points in the downsampled plasma boundary written to in_eq/in_bnd.
#: Matches run_jorek.py:113's `bnd_points = 50`.
BOUNDARY_POINTS = 50

#: Points each resampled profile (ffprime/T/rho) is written out at.
#: Matches run_jorek.py:114's `n_prof_points = 200`.
PROFILE_POINTS = 200


@dataclass(frozen=True)
class PreparedRun:
    """What :func:`prepare_run` did or planned to do.

    ``actions`` is a human-readable log of every disk-mutating step, useful
    on its own for ``--dry-run`` (nothing in it happened) or as a record of
    what did.
    """

    paths: RunPaths
    real_psi_edge: float
    actions: list[str]


class _Disk:
    """Executes or merely records disk-mutating steps, for ``--dry-run``.

    Pure computation (psi grids, profiles, boundaries) happens unconditionally
    before any of this is invoked -- only the parts with side effects are
    gated, so a dry run still validates everything a real run would.
    """

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.actions: list[str] = []

    def _log(self, message: str) -> None:
        self.actions.append(message)

    def mkdir(self, path: Path, *, exist_ok: bool) -> None:
        self._log(f"mkdir {path} (exist_ok={exist_ok})")
        if not self.dry_run:
            path.mkdir(parents=True, exist_ok=exist_ok)

    def copy_all_files(self, src: Path, dst: Path) -> None:
        self._log(f"copy {src}/* -> {dst}/")
        if not self.dry_run:
            fs.copy_all_files(src, dst)

    def symlink_dir(self, src: Path, dst: Path, link_name: str) -> None:
        self._log(f"symlink {dst}/{link_name} -> {src}")
        if not self.dry_run:
            fs.symlink_dir(src, dst, link_name=link_name)

    def symlink_files_in(self, src: Path, dst: Path) -> None:
        self._log(f"symlink files {src}/* -> {dst}/")
        if not self.dry_run:
            fs.symlink_files_in(src, dst)

    def symlink_file(self, src: Path, dst: Path) -> None:
        self._log(f"symlink {dst} -> {src}")
        if not self.dry_run:
            fs.symlink_file(src, dst)

    def set_fields(self, paths: list[Path], updates: dict, *, create_missing=False) -> None:
        self._log(
            f"set_fields {[p.name for p in paths]}: "
            f"{', '.join(f'{k}={v}' for k, v in updates.items())}"
        )
        if not self.dry_run:
            nml.set_fields(paths, updates, create_missing=create_missing)

    def set_boundary_block(self, path: Path, R, Z, psi, fmt: str) -> None:
        self._log(f"set_boundary_block {path} ({len(R)} points)")
        if not self.dry_run:
            nml.set_boundary_block(path, R, Z, psi, fmt)

    def write_boundary_file(self, path: Path, R, Z, psi) -> None:
        self._log(f"write_boundary_file {path} ({len(R)} points)")
        if not self.dry_run:
            nml.write_boundary_file(path, R, Z, psi)

    def write_float(self, path: Path, value: float) -> None:
        self._log(f"write_float {path} = {value}")
        if not self.dry_run:
            write_float(path, value)

    def savetxt(self, path: Path, data: np.ndarray) -> None:
        self._log(f"savetxt {path} ({data.shape[0]} rows)")
        if not self.dry_run:
            np.savetxt(path, data, delimiter=" ")


def _uses_castor(params: ShotParams) -> bool:
    """Broadened from the old ``"castor" in [ffprime_method, T_method,
    rho_method]`` to also include ``bnd_method`` -- see module docstring."""
    return "castor" in (
        params.ffprime_method, params.T_method, params.rho_method, params.bnd_method
    )


def prepare_run(
    params: ShotParams,
    site: Site,
    run_dir: Path,
    *,
    replace: bool = False,
    dry_run: bool = False,
    run_sw: bool = False,
) -> PreparedRun:
    """Populate a run folder from a shotfile. Validates before writing anything.

    Mirrors ``Columbia/run_jorek.py``'s folder-population section (roughly
    lines 124-302), with the fixes listed in this module's docstring applied.

    ``run_sw`` mirrors ``run_jorek.py:146``'s ``not run_sw_flag`` guard: pass
    ``True`` when this same invocation will also call :func:`submit_starwall`
    to *generate* the archived response, not consume it. Without this, the
    archived response gets symlinked in as ``starwall-response.dat`` and then
    :func:`submit_starwall`'s archive step tries to copy that file onto
    itself, raising ``shutil.SameFileError`` -- caught on the HPC, where real
    symlinks exposed it; the Windows dev clone's copy-based symlink bypass
    could not.
    """
    run_dir = Path(run_dir).resolve()
    disk = _Disk(dry_run)
    paths = RunPaths(run_dir)

    if params.with_refluid and "RE" not in params.exe:
        import warnings
        warnings.warn(
            f"with_refluid=True but 'RE' not in exe ({params.exe!r})", stacklevel=2
        )

    # ---- pure computation first: psi, profiles, boundary --------------------
    castor_dir = None
    psi = None
    psi_n = None
    psi_n_export = None
    real_psi_edge = 1.0
    real_psi_edge_psi = None
    extended_idx_range = None

    if _uses_castor(params):
        cp = params.castor_params
        # "machine_folder" (an absolute path) wins if a shotfile still sets it
        # directly; otherwise "machine" (just the subfolder name, e.g.
        # "DIIID_low_pres") is joined onto site.toml's castor_root. See the
        # refactor plan's "Gap found and fixed" note on castor_master_folder.
        machine_folder = cp.get("machine_folder") or str(site.castor_root / cp["machine"])
        cotrans_dir = f"{machine_folder}/equilib/cotrans/qa{cp['qa']:.1f}/a1.00_g{cp['g']:.3f}"
        castor_dir = f"{machine_folder}/castor3d/{cp['scan_folder']}/qa{cp['qa']:.1f}/a1.00_g{cp['g']:.3f}"

        castor_psi = prof_mod.get_psi_from_castor(cotrans_dir, params.castor_suffix)
        psi, psi_n = castor_psi.psi, castor_psi.psi_n
        psi_n_export = psi_n

        if params.extend_bnd:
            extended = bnd_mod.extend_psi(psi, params.extend_ratio, params.extend_reso)
            psi_n_export = extended.psi_n
            real_psi_edge = extended.real_psi_edge
            real_psi_edge_psi = extended.psi
            extended_idx_range = extended.extended_idx_range

    # ---- rho -------------------------------------------------------------
    if params.rho_method == "const":
        rho_const_jorek = params.rho_const / 1e20  # JOREK normalisation
        rho_prof_export = [rho_const_jorek] * len(psi_n_export)
    else:
        raise NotImplementedError(f"rho_method={params.rho_method!r} not implemented")
    rho_data = np.column_stack(
        prof_mod.resample_profile(psi_n_export, rho_prof_export, PROFILE_POINTS)
    )

    # ---- ffprime -----------------------------------------------------------
    if params.ffprime_method == "castor":
        ffprime_prof = prof_mod.get_ffprime_profile_from_castor(
            psi, cotrans_dir, params.castor_suffix
        )
        if params.extend_bnd:
            ffprime_prof = bnd_mod.extend_prof(ffprime_prof, extended_idx_range)
    else:
        raise NotImplementedError(f"ffprime_method={params.ffprime_method!r} not implemented")
    ffprime_data = np.column_stack(
        prof_mod.resample_profile(psi_n_export, ffprime_prof, PROFILE_POINTS)
    )

    # ---- T -----------------------------------------------------------------
    if params.T_method == "castor":
        # rho_prof (list form) matches what the old code passed -- see
        # KNOWN_ISSUES.md for why its value doesn't actually matter here.
        t_prof = prof_mod.get_t_profile_from_castor(
            [rho_const_jorek], cotrans_dir, params.castor_suffix
        )
        if params.extend_bnd:
            t_prof = bnd_mod.extend_prof(t_prof, extended_idx_range)
    else:
        raise NotImplementedError(f"T_method={params.T_method!r} not implemented")
    t_data = np.column_stack(
        prof_mod.resample_profile(psi_n_export, t_prof, PROFILE_POINTS)
    )

    # ---- boundary ------------------------------------------------------------
    if params.bnd_method == "castor":
        raw = bnd_mod.read_boundary_from_castor(castor_dir, params.castor_suffix)
        if params.extend_bnd:
            R0, Z0 = bnd_mod.boundary_center(raw)
            expanded = bnd_mod.expand_boundary(raw, R0, Z0, scale=params.extend_ratio)
        else:
            expanded = raw
        bnd = bnd_mod.downsample_boundary(expanded, BOUNDARY_POINTS)
        original_bnd = raw
        psi_bnd = np.full(
            len(bnd), real_psi_edge_psi[-1] if params.extend_bnd else psi[-1]
        )
    elif params.bnd_method == "file":
        bnd_path = run_dir / params.bnd_file
        bnd = load_two_col_data(bnd_path)
        if psi is None:
            raise ShotfileError(
                "bnd_method='file' needs psi from a castor-sourced ffprime/T/"
                "rho method, matching the old code's coupling"
            )
        psi_bnd = np.full(len(bnd), psi[-1])
        original_bnd = bnd
    else:
        raise NotImplementedError(f"bnd_method={params.bnd_method!r} not implemented")

    # =========================================================================
    # From here on: side effects. Everything above was pure computation, so a
    # dry run has already exercised every failure mode a real run would hit.
    # =========================================================================

    # run_dir is always the caller's cwd (same as the old script), so it
    # *always already exists* by the time prepare_run runs -- you have to cd
    # into it first. Bare existence is therefore not a meaningful signal;
    # the old `exist_ok=(not replace_flag)` made --replace fail every time
    # (bug #2), and a naive `exist_ok=replace` inversion would make the
    # *default* case fail every time instead, which is worse. What actually
    # needs gating behind --replace is whether the folder looks like it was
    # already populated by a previous prepare_run -- checked via in_eq,
    # the first file that step writes.
    if paths.in_eq.exists() and not replace:
        raise FileExistsError(
            f"{run_dir} already contains a prepared run (in_eq exists); "
            "pass replace=True to overwrite"
        )
    disk.mkdir(run_dir, exist_ok=True)

    disk.copy_all_files(site.template / "copy", run_dir)
    disk.symlink_dir(site.exe, run_dir, "exe")
    disk.symlink_dir(site.jobscripts, run_dir, "jobscripts")
    disk.symlink_files_in(site.template / "symlink" / "base", run_dir)

    if params.freeboundary and not params.allow_other_starwall and not run_sw:
        starwall_src = (
            site.template / "symlink" / "starwall"
            / f"starwall-response_qa{params.qa:.1f}_g{params.g:.3f}.dat"
        )
        if not starwall_src.is_file():
            raise FileNotFoundError(
                f"No starwall response found for qa{params.qa:.1f}_g{params.g:.3f}"
                f" ({starwall_src}). Run submit_starwall first, or set "
                "allow_other_starwall=True."
            )
        disk.symlink_file(starwall_src, run_dir / "starwall-response.dat")

    disk.symlink_dir(site.jorek_util(params.with_refluid), run_dir, "util")
    model_symlinks = "RE" if params.with_refluid else "standard"
    disk.symlink_files_in(site.template / "symlink" / model_symlinks, run_dir)

    disk.set_fields(
        paths.namelists,
        {
            "eta": params.eta,
            "central_density": rho_const_jorek,
            "freeboundary": params.freeboundary,
        },
    )
    disk.set_fields(
        [paths.in_main, paths.in_main_r],
        {
            # Preserve the old string formatting exactly: run_jorek.py passed
            # str(list).strip("[]") through untouched, not a Fortran-double
            # literal -- see namelist.py's fortran_literal for why a Python
            # list would otherwise render differently (e.g. "3.0d-2").
            "tstep_n": str(params.tstep_n).strip("[]"),
            "nstep_n": str(params.nstep_n).strip("[]"),
            "nout": params.nout,
        },
    )

    if params.with_refluid:
        disk.set_fields(
            paths.namelists,
            {
                "re_initialize": params.re_initialize,
                "initial_re_current_fraction": params.initial_re_current_fraction,
                "vpar_re_sign": params.vpar_re_sign,
                "re_adv_fact": params.re_adv_fact,
                "Dre_num": params.Dre_num,
                "Dre_par": params.Dre_par,
                "Dre_iso": params.Dre_iso,
            },
            create_missing=True,
        )

    disk.write_float(paths.real_psi_edge, real_psi_edge)
    disk.savetxt(run_dir / "ffprime_prof.dat", ffprime_data)
    disk.savetxt(run_dir / "T_prof.dat", t_data)
    disk.savetxt(run_dir / "rho_prof.dat", rho_data)

    # FIX applied here (confirmed with George, see module docstring): the
    # boundary now goes into every namelist, not just in_eq.
    for target in paths.namelists:
        disk.set_boundary_block(target, bnd[:, 0], bnd[:, 1], psi_bnd, ".2f")
    disk.write_boundary_file(paths.in_bnd, bnd[:, 0], bnd[:, 1], psi_bnd)

    if params.extend_bnd:
        disk.savetxt(run_dir / "original_bnd.dat", original_bnd)

    return PreparedRun(paths=paths, real_psi_edge=real_psi_edge, actions=disk.actions)


# =============================================================================
# Submission
# =============================================================================


def _run(command: str, cwd: Path, *, dry_run: bool) -> str:
    if dry_run:
        return command
    subprocess.run(command, cwd=cwd, shell=True, check=True)
    return command


def submit_eq(paths: RunPaths, site: Site, params: ShotParams, *, dry_run: bool = False) -> str:
    """Interactive equilibrium run. Mirrors run_jorek.py:308-321 (--run_eq)."""
    command = (
        f"{site.launch.interactive_prelude}\n"
        f"{site.launch.mpirun_cmd(site.launch.n_jorek)} ./exe/{params.exe} < ./in_eq | tee log_eq"
    )
    return _run(command, paths.run_dir, dry_run=dry_run)


def submit_main(
    paths: RunPaths, site: Site, params: ShotParams, *, interactive: bool, dry_run: bool = False,
) -> str:
    """Main run, interactive (--run_i) or batch via sbatch (--run)."""
    if interactive:
        command = (
            f"{site.launch.interactive_prelude}\n"
            f"{site.launch.mpirun_cmd(site.launch.n_jorek)} ./exe/{params.exe} < ./in_main | tee log"
        )
    else:
        command = (
            f"{site.launch.batch_prelude}\n"
            f"./submit_jorek.sh jobscripts/{params.jobscript} ./exe/{params.exe} "
            f"./in_main log {params.g:.1f}_{params.eta:g}"
        )
    return _run(command, paths.run_dir, dry_run=dry_run)


def submit_restart(
    paths: RunPaths, site: Site, params: ShotParams, *, dry_run: bool = False,
) -> str:
    """Batch restart run. Mirrors run_jorek.py:352-364 (--run_r)."""
    command = (
        f"{site.launch.batch_prelude}\n"
        f"./submit_jorek.sh jobscripts/{params.jobscript} ./exe/{params.exe} "
        f"./in_main_r log {params.g:.1f}_{params.eta:g}"
    )
    return _run(command, paths.run_dir, dry_run=dry_run)


def submit_starwall(
    paths: RunPaths, site: Site, params: ShotParams, *, dry_run: bool = False,
) -> list[str]:
    """Equilibrium then STARWALL, then archive the response.

    FIX applied here (confirmed with George): the equilibrium stage tees into
    ``log_eq``, matching ``submit_eq``, not ``log`` -- the old code clobbered
    the main run's log with this stage's equilibrium output.

    Requires the run folder to have been prepared with
    ``prepare_run(..., run_sw=True)`` -- otherwise ``starwall-response.dat``
    is already a symlink to the exact archive path this function copies onto,
    and the copy raises ``shutil.SameFileError``.
    """
    commands = []

    eq_command = (
        f"{site.launch.interactive_prelude}\n"
        f"{site.launch.mpirun_cmd(site.launch.n_jorek)} ./exe/{params.exe} < ./in_eq | tee log_eq"
    )
    commands.append(_run(eq_command, paths.run_dir, dry_run=dry_run))

    sw_command = (
        f"{site.launch.batch_prelude}\n"
        f"{site.launch.mpirun_cmd(site.launch.n_starwall)} "
        f"./exe/STARWALL_JOREK_Linux ./input_starwall | tee log_sw"
    )
    commands.append(_run(sw_command, paths.run_dir, dry_run=dry_run))

    archive_target = (
        site.template / "symlink" / "starwall"
        / f"starwall-response_qa{params.qa:.1f}_g{params.g:.3f}.dat"
    )
    response = paths.run_dir / "starwall-response.dat"
    if dry_run:
        commands.append(f"copy {response} -> {archive_target}, then remove {response}")
    else:
        import shutil

        shutil.copy2(response, archive_target)
        response.unlink()
    return commands
