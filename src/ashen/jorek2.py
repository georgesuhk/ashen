"""One runner for JOREK's own postprocessing tools (jorek2_postproc,
jorek2_poincare, and -- not wired up yet -- jorek2vtk).

Today the "stage a scratch dir, run a jorek2_* tool, collect outputs" dance is
written out three separate times: ``poinc_diag.py:74`` ``run_single``,
``gather_profiles.py:17`` ``run_single_var``, and ``data_jorek.py:719``
``get_zeroDs_at_t`` (the last of which turns out not to need staging at all --
see :func:`run_zero_d`). :func:`run_tool` extracts the shared shape once,
using ``poinc_diag.py``'s version as the template since it already avoids
``shell=True`` by piping the control input through a real file object;
``gather_profiles.py:70`` used ``subprocess.run(f"{exe} < ...", shell=True)``,
which is POSIX-only.

**Fix applied here (flagged in the refactor plan's verification section):**
the legacy ``run_single`` wraps its output-parsing in a bare
``try/except: pass`` and returns an array of ``None``s on failure
(``poinc_diag.py:124-130``), silently discarding a tool crash. Here a
non-zero exit or a missing expected output raises :class:`Jorek2Error` naming
the tool, the step, and the run directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ashen.paths import RunPaths

__all__ = ["Jorek2Error", "Jorek2Run", "ToolResult", "run_tool", "run_zero_d"]


class Jorek2Error(RuntimeError):
    """A jorek2_* tool exited non-zero, or an expected output was missing."""


@dataclass(frozen=True)
class ToolResult:
    """What one :func:`run_tool` invocation produced.

    ``outputs`` maps each *requested* output path to where it was copied.
    ``stdout`` is empty unless ``capture_stdout=True`` -- it matters for
    ``jorek2_poincare``, whose per-line progress messages are the only way to
    tell which output block belongs to which field line (see
    :mod:`ashen.diagnostics.poincare`).
    """

    outputs: dict[str, Path]
    stdout: str = ""

    # Ergonomics: run_tool used to return the mapping directly, and most
    # callers only ever want that.
    def __getitem__(self, key: str) -> Path:
        return self.outputs[key]

    def __contains__(self, key: object) -> bool:
        return key in self.outputs


@dataclass(frozen=True)
class Jorek2Run:
    """Where to find one run's inputs for staging into a jorek2_* scratch dir.

    ``exe_dir`` is usually ``run_dir`` itself -- the jorek2_* tools are
    symlinked into every prepared run folder (see
    :func:`ashen.runner.prepare_run`) -- but is kept separate since nothing
    here requires that.
    """

    run_dir: Path
    exe_dir: Path
    namelist: Path
    pad_width: int
    profiles: tuple[str, ...] = ("T_prof.dat", "rho_prof.dat", "ffprime_prof.dat")

    def restart_path(self, step: int) -> Path:
        return self.run_dir / f"jorek{step:0{self.pad_width}d}.h5"


def run_tool(
    run: Jorek2Run,
    tool: str,
    *,
    step: int,
    dest_dir: Path | str,
    outputs: Sequence[str],
    stdin_text: str | None = None,
    stdin_is_namelist: bool = False,
    restart_name: str = "jorek_restart.h5",
    extra_files: dict[str, str] | None = None,
    copy_exe: bool = False,
    env: Mapping[str, str] | None = None,
    capture_stdout: bool = False,
) -> ToolResult:
    """Stage inputs for one jorek2_* invocation, run it, collect outputs.

    A fresh temporary directory gets: the restart file for ``step`` (named
    ``restart_name`` -- ``jorek2_poincare`` wants the fixed name
    ``jorek_restart.h5``, but ``jorek2_postproc`` (via
    :func:`ashen.diagnostics.profiles.extract_profile`) is called with the
    real padded filename, matching ``gather_profiles.py:46``), the namelist,
    this run's profile files, and anything in ``extra_files`` (e.g. a
    Poincaré ``stpts`` starting-point file).

    Exactly one of ``stdin_text`` (a control script, e.g. from
    :mod:`ashen.postproc`) or ``stdin_is_namelist=True`` (pipe the copied
    namelist itself -- what ``jorek2_poincare`` expects) must be given.

    Every path in ``outputs`` (relative to the scratch dir, e.g.
    ``"postproc/exprs_midplane_s005000.dat"``) is copied into ``dest_dir``
    before the scratch dir is discarded; the returned
    :class:`ToolResult` maps the *requested* path to where it landed. Raises
    :class:`Jorek2Error` if the tool exits non-zero or an expected output is
    missing -- see the module docstring for why that matters here
    specifically.

    ``env`` is merged over the parent environment for the child only. This is
    how ``OMP_NUM_THREADS`` gets set per invocation instead of being inherited
    from whatever the shell happens to have -- ``site.toml``'s
    ``interactive_prelude`` exports ``OMP_NUM_THREADS=10``, which used to leak
    into every one of the diagnostics' worker processes at once.

    ``capture_stdout`` returns the tool's stdout on the result rather than
    discarding it.
    """
    if stdin_text is None and not stdin_is_namelist:
        raise ValueError("pass stdin_text, or stdin_is_namelist=True")
    if stdin_text is not None and stdin_is_namelist:
        raise ValueError("pass exactly one of stdin_text and stdin_is_namelist")

    exe = run.exe_dir / tool
    if not exe.is_file():
        raise FileNotFoundError(f"{tool} not found at {exe}")
    restart_src = run.restart_path(step)
    if not restart_src.is_file():
        raise FileNotFoundError(f"restart file not found: {restart_src}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{tool}_") as tmp:
        workdir = Path(tmp)
        shutil.copy(restart_src, workdir / restart_name)
        namelist_copy = workdir / run.namelist.name
        shutil.copy(run.namelist, namelist_copy)
        for profile in run.profiles:
            src = run.run_dir / profile
            if src.is_file():
                shutil.copy(src, workdir)

        exe_invoke = exe
        if copy_exe:
            exe_invoke = workdir / tool
            shutil.copy(exe, exe_invoke)

        for name, content in (extra_files or {}).items():
            target = workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        if stdin_is_namelist:
            stdin_path = namelist_copy
        else:
            stdin_path = workdir / f"{tool}.in"
            stdin_path.write_text(stdin_text, encoding="utf-8")

        child_env = None
        if env:
            child_env = {**os.environ, **{k: str(v) for k, v in env.items()}}

        with open(stdin_path, encoding="utf-8") as stdin_file:
            result = subprocess.run(
                [str(exe_invoke)],
                stdin=stdin_file,
                cwd=workdir,
                stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=child_env,
            )
        if result.returncode != 0:
            raise Jorek2Error(
                f"{tool} exited {result.returncode} for step {step} in "
                f"{run.run_dir}: {result.stderr.decode(errors='replace')}"
            )

        collected: dict[str, Path] = {}
        for output in outputs:
            src = workdir / output
            if not src.is_file():
                raise Jorek2Error(
                    f"{tool} did not produce expected output {output!r} for "
                    f"step {step} in {run.run_dir}"
                )
            dst = dest_dir / Path(output).name
            shutil.copy(src, dst)
            collected[output] = dst

    stdout = ""
    if capture_stdout and result.stdout is not None:
        stdout = result.stdout.decode(errors="replace")
    return ToolResult(outputs=collected, stdout=stdout)


def run_zero_d(run: Jorek2Run, step: int, paths: RunPaths) -> Path:
    """zeroD_quantities for one step. Ports ``data_jorek.py:719``
    ``get_zeroDs_at_t``.

    Unlike :func:`run_tool`, this runs ``jorek2_postproc`` **in place** in
    ``run_dir`` rather than a scratch copy, matching the original -- there is
    nothing to stage, since JOREK resolves ``for step <t> do`` against the
    restart files already present in the run folder, and the output
    (``postproc/zeroD_quantities_s<step>.dat``) is meant to persist there as
    a cache, not be collected and discarded.
    """
    from ashen.postproc import zero_d_script

    exe = run.exe_dir / "jorek2_postproc"
    if not exe.is_file():
        raise FileNotFoundError(f"jorek2_postproc not found at {exe}")

    paths.postproc_dir.mkdir(parents=True, exist_ok=True)
    script_path = run.run_dir / "postproc_zeroD_script.in"
    script_path.write_text(
        zero_d_script(run.namelist.name, paths.step_str(step)), encoding="utf-8"
    )
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
            f"jorek2_postproc exited {result.returncode} for zeroD at step "
            f"{step} in {run.run_dir}: {result.stderr.decode(errors='replace')}"
        )

    out = paths.zero_d(step)
    if not out.is_file():
        raise Jorek2Error(f"zeroD_quantities not produced at {out}")
    return out
