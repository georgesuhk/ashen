"""Field-line Poincare tracing via jorek2_poincare.

Ports ``castor3d/util/diagnostics/poinc_diag.py``. The parallel fan-out
(``ProcessPoolExecutor`` over every ``(psi_n, angular-sample)`` pair) is kept
-- it is what makes an O(1000)-turn Poincare scan tractable -- but staging and
execution now go through :func:`ashen.jorek2.run_tool` instead of
``poinc_diag.py``'s own copy of that logic.

**Not ported in this pass**: the plotting functions (``plot_poincare``,
``plot_field_line_diffusion``, ``plot_connection_length``,
``get_island_width``) in ``castor3d/util/data_jorek.py``. Those are
visualization on top of the ``.npz`` caches this module writes, not part of
the run-orchestration this pass targets, and several have known bugs
(hardcoded ``R0 = 1.36`` in four places, a missing
``plot_max_fieldline_pos``) that need their own confirmation pass -- see
``ashen/KNOWN_ISSUES.md``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from ashen.castor_io import load_two_col_data
from ashen.jorek2 import Jorek2Error, Jorek2Run, run_tool
from ashen.paths import RunPaths
from ashen.postproc import flux_surface_script

__all__ = ["trace_field_line", "run_poincare_step"]


def _sampled_rz(fs_file: Path, n_sample_freq: int) -> list[tuple[list[float], list[float]]]:
    """Ports ``poinc_diag.py:51`` ``get_sampled_RZ_at_psi_N``."""
    if n_sample_freq <= 0:
        raise ValueError("n_sample_freq must be greater than 0")
    data = load_two_col_data(fs_file)
    if len(data) == 0:
        return []
    idx = np.linspace(0, len(data) - 1, num=n_sample_freq, dtype=int)
    return [([float(data[i, 0])], [float(data[i, 1])]) for i in idx]


def _fieldline_block(R: list[float], Z: list[float], phi: list[float], n_turns: int) -> str:
    """Ports ``poinc_diag.py:19`` ``write_fieldline_block``, as text rather
    than a direct write -- :func:`ashen.jorek2.run_tool` stages it."""
    R_arr, Z_arr, phi_arr = np.asarray(R, float), np.asarray(Z, float), np.asarray(phi, float)
    if not (len(R_arr) == len(Z_arr) == len(phi_arr)):
        raise ValueError("R, Z, phi must have the same length")
    lines = ["# n_lines", f"   {len(R_arr)}", "# nr    R_start   Z_start    phi_start   n_turns"]
    for i, (r, z, p) in enumerate(zip(R_arr, Z_arr, phi_arr), start=1):
        lines.append(f"{i:4d}  {r:10.6f}  {z:10.6f}  {p:10.6f}  {int(n_turns):6d}")
    return "\n".join(lines) + "\n"


def _write_flux_surfaces(run: Jorek2Run, step: int, psi_n_list, namelist_name: str) -> None:
    """Ports ``run_single_t``'s flux-surface-generation loop
    (``poinc_diag.py:143-154``) -- runs ``jorek2_postproc`` in place, same as
    :func:`ashen.jorek2.run_zero_d`, since the output is meant to land
    directly under ``run_dir/postproc/``."""
    import subprocess

    exe = run.exe_dir / "jorek2_postproc"
    if not exe.is_file():
        raise FileNotFoundError(f"jorek2_postproc not found at {exe}")

    for psi_n in psi_n_list:
        script = flux_surface_script(namelist_name, step, psi_n)
        script_path = run.run_dir / "postproc_fs_script.in"
        script_path.write_text(script, encoding="utf-8")
        with open(script_path, encoding="utf-8") as stdin_file:
            result = subprocess.run(
                [str(exe)],
                stdin=stdin_file,
                cwd=run.run_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        if result.returncode != 0:
            raise Jorek2Error(
                f"jorek2_postproc exited {result.returncode} generating the "
                f"flux surface at psi_n={psi_n} for step {step} in {run.run_dir}: "
                f"{result.stderr.decode(errors='replace')}"
            )


def trace_field_line(
    run: Jorek2Run,
    step: int,
    psi_n: float,
    ang_idx: int,
    *,
    paths: RunPaths,
    n_sample_freq: int,
    n_turns: int,
    dest_dir: Path,
) -> dict[str, np.ndarray]:
    """One field line: samples a starting point on ``psi_n``'s flux surface
    and traces it with jorek2_poincare. Ports ``poinc_diag.py:74``
    ``run_single``.

    Unlike the legacy version, a tool failure or missing output raises
    :class:`ashen.jorek2.Jorek2Error` instead of being swallowed into an
    array of ``None`` -- see the ``jorek2`` module docstring.
    """
    fs_file = paths.flux_surface(psi_n, step)
    samples = _sampled_rz(fs_file, n_sample_freq)
    R, Z = samples[ang_idx]

    collected = run_tool(
        run,
        "jorek2_poincare",
        step=step,
        dest_dir=dest_dir,
        outputs=["poinc_rho-theta.dat", "poinc_R-Z.dat"],
        stdin_is_namelist=True,
        extra_files={"stpts": _fieldline_block(R, Z, [0.0], n_turns)},
    )
    psi_theta = np.loadtxt(collected["poinc_rho-theta.dat"])
    r_z = np.loadtxt(collected["poinc_R-Z.dat"])
    return {
        "psi_n": psi_theta[:, 0] ** 2,
        "theta": psi_theta[:, 1],
        "R": r_z[:, 0],
        "Z": r_z[:, 1],
    }


def run_poincare_step(
    run: Jorek2Run,
    paths: RunPaths,
    step: int,
    psi_n_list: list[float],
    *,
    ang_sample_freq: int,
    n_turns: int,
    n_workers: int,
) -> dict[str, np.ndarray]:
    """One restart step's full Poincare scan. Ports ``poinc_diag.py:135``
    ``run_single_t``: generates flux surfaces, traces every
    ``(psi_n, angular sample)`` pair in parallel, saves the four ``.npz``
    caches ``run_single_t`` wrote, and returns the same arrays.
    """
    _write_flux_surfaces(run, step, psi_n_list, run.namelist.name)

    scratch = paths.poinc_dir / f"_scratch_s{paths.step_str(step)}"
    tasks = [(psi_n, j) for psi_n in psi_n_list for j in range(ang_sample_freq)]
    trace_one = partial(
        trace_field_line,
        run,
        step,
        paths=paths,
        n_sample_freq=ang_sample_freq,
        n_turns=n_turns,
        dest_dir=scratch,
    )
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # map() over parallel positional iterables: ProcessPoolExecutor has
        # no starmap, and a lambda unpacking tasks wouldn't be picklable.
        results = list(executor.map(trace_one, (t[0] for t in tasks), (t[1] for t in tasks)))

    for psi_n in psi_n_list:
        fs_file = paths.flux_surface(psi_n, step)
        if fs_file.exists():
            fs_file.unlink()
    if scratch.is_dir() and not any(scratch.iterdir()):
        scratch.rmdir()

    n_psi = len(psi_n_list)
    psi_n_out = np.empty((n_psi, ang_sample_freq), dtype=object)
    theta_out = np.empty((n_psi, ang_sample_freq), dtype=object)
    R_out = np.empty((n_psi, ang_sample_freq), dtype=object)
    Z_out = np.empty((n_psi, ang_sample_freq), dtype=object)
    for k, (i, j) in enumerate((i, j) for i in range(n_psi) for j in range(ang_sample_freq)):
        psi_n_out[i, j] = results[k]["psi_n"]
        theta_out[i, j] = results[k]["theta"]
        R_out[i, j] = results[k]["R"]
        Z_out[i, j] = results[k]["Z"]

    paths.poinc_dir.mkdir(parents=True, exist_ok=True)
    t_str = paths.step_str(step)
    save_args = {"in_val": np.asarray(psi_n_list)}
    np.savez(paths.poinc_dir / f"poinc_t{t_str}_psi_n", out_val=psi_n_out, **save_args)
    np.savez(paths.poinc_dir / f"poinc_t{t_str}_theta", out_val=theta_out, **save_args)
    np.savez(paths.poinc_dir / f"poinc_t{t_str}_R", out_val=R_out, **save_args)
    np.savez(paths.poinc_dir / f"poinc_t{t_str}_Z", out_val=Z_out, **save_args)

    return {"psi_n": psi_n_out, "theta": theta_out, "R": R_out, "Z": Z_out}
