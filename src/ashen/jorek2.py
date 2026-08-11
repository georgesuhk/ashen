"""One runner for JOREK's own postprocessing tools (jorek2_postproc,
jorek2_poincare, and -- not wired up yet -- jorek2vtk).

Legacy wrote the "stage scratch dir, run tool, collect outputs" dance 3x:
poinc_diag.py:74 run_single, gather_profiles.py:17 run_single_var,
data_jorek.py:719 get_zeroDs_at_t (which turns out not to need staging at
all -- see run_zero_d). run_tool extracts the shared shape once, using
poinc_diag.py's version as template since it already avoids shell=True by
piping control input through a real file object; gather_profiles.py:70
used subprocess.run(f"{exe} < ...", shell=True), POSIX-only.

Fix vs. legacy: run_single wraps output-parsing in a bare try/except: pass
and returns an array of Nones on failure (poinc_diag.py:124-130), silently
discarding a tool crash. Here a non-zero exit or missing expected output
raises Jorek2Error naming the tool, step, and run directory.
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

__all__ = [
    "Jorek2Error", "MissingRestartError", "Jorek2Run", "ToolResult", "run_tool", "run_zero_d",
]


class Jorek2Error(RuntimeError):
    """A jorek2_* tool exited non-zero, or an expected output was missing."""


class MissingRestartError(FileNotFoundError):
    """The restart file for a requested step doesn't exist.

    Distinguished from a bare FileNotFoundError (e.g. missing executable)
    so per-step gather loops can catch this specifically, warn, and move
    on instead of aborting the whole scan.
    """


@dataclass(frozen=True)
class ToolResult:
    """What one run_tool invocation produced.

    outputs maps each requested output path to where it was copied. stdout
    is empty unless capture_stdout=True -- matters for jorek2_poincare,
    whose per-line progress messages are the only way to tell which output
    block belongs to which field line (diagnostics.poincare).

    stderr is always captured (quoted on non-zero exit) and kept here on
    success too: which stream a Fortran write(*,...) lands on isn't
    guaranteed across builds/launchers, so a caller scraping the tool's
    log should use `log`, not stdout alone.
    """

    outputs: dict[str, Path]
    stdout: str = ""
    stderr: str = ""

    @property
    def log(self) -> str:
        """Both captured streams, for scraping the tool's progress messages.
        Concatenated, not interleaved -- true interleaving is lost once
        they're two pipes; every consumer here matches per-line patterns,
        not cross-stream ordering."""
        return self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr

    # Ergonomics: run_tool used to return the mapping directly, and most
    # callers only ever want that.
    def __getitem__(self, key: str) -> Path:
        return self.outputs[key]

    def __contains__(self, key: object) -> bool:
        return key in self.outputs


@dataclass(frozen=True)
class Jorek2Run:
    """Where to find one run's inputs for staging into a jorek2_* scratch dir.

    exe_dir is usually run_dir itself (jorek2_* tools are symlinked into
    every prepared run folder, runner.prepare_run) but kept separate since
    nothing here requires that.
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
    outputs: Sequence[str] = (),
    output_glob: str | None = None,
    stdin_text: str | None = None,
    stdin_is_namelist: bool = False,
    restart_name: str = "jorek_restart.h5",
    extra_files: dict[str, str] | None = None,
    copy_exe: bool = False,
    env: Mapping[str, str] | None = None,
    capture_stdout: bool = False,
) -> ToolResult:
    """Stage inputs for one jorek2_* invocation, run it, collect outputs.

    A fresh temp dir gets: the restart file for `step` (named restart_name
    -- jorek2_poincare wants fixed name jorek_restart.h5, but
    jorek2_postproc, via diagnostics.profiles.extract_profile, is called
    with the real padded filename, matching gather_profiles.py:46), the
    namelist, this run's profile files, and anything in extra_files (e.g.
    a Poincare stpts starting-point file).

    Exactly one of stdin_text (a control script, e.g. from ashen.postproc)
    or stdin_is_namelist=True (pipe the copied namelist itself -- what
    jorek2_poincare expects) must be given.

    Every path in `outputs` (relative to the scratch dir, e.g.
    "postproc/exprs_midplane_s005000.dat") is copied into dest_dir before
    the scratch dir is discarded; the returned ToolResult maps the
    requested path to where it landed. Raises Jorek2Error if the tool
    exits non-zero or an expected output is missing (see module docstring).

    output_glob, if given, additionally collects every top-level file
    matching that pattern -- for a tool like jorek2_four whose output
    filenames depend on the model (which variables it carries) and the
    run's namelist (toroidal harmonic count), so they can't be listed in
    `outputs` ahead of time. Collected files are keyed by name in the same
    ToolResult.outputs mapping. Raises Jorek2Error if nothing matches -- a
    tool that "succeeded" but produced none of its expected output is
    exactly the silent failure this function exists to prevent.

    env is merged over the parent environment for the child only -- how
    OMP_NUM_THREADS gets set per invocation instead of inherited from
    whatever the shell has (site.toml's interactive_prelude exports
    OMP_NUM_THREADS=10, which used to leak into every worker process at once).

    capture_stdout returns the tool's stdout on the result instead of
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
        raise MissingRestartError(f"restart file not found: {restart_src}")

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

        if output_glob is not None:
            matches = sorted(p for p in workdir.glob(output_glob) if p.is_file())
            if not matches:
                raise Jorek2Error(
                    f"{tool} produced no output matching {output_glob!r} for "
                    f"step {step} in {run.run_dir}"
                )
            for src in matches:
                dst = dest_dir / src.name
                shutil.copy(src, dst)
                collected[src.name] = dst

    stdout = ""
    if capture_stdout and result.stdout is not None:
        stdout = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace") if result.stderr is not None else ""
    return ToolResult(outputs=collected, stdout=stdout, stderr=stderr)


def run_zero_d(
    run: Jorek2Run, step: int, paths: RunPaths, *, si_units: bool = True
) -> Path:
    """zeroD_quantities for one step. Ports data_jorek.py:719 get_zeroDs_at_t.

    si_units=True (default) runs jorek2_postproc in place in run_dir, not a
    scratch copy -- nothing to stage, since JOREK resolves `for step <t>
    do` against restart files already in the run folder, and the output
    (postproc/zeroD_quantities_s<step>.dat) is meant to persist as a
    cache, not be collected and discarded.

    Control script is written to a unique temp file, not a fixed name in
    run_dir -- analyse's zerod gathering fans out across processes sharing
    run_dir, and a fixed name would let one step's process overwrite
    another's script before it's read (same race poincare.
    _write_flux_surface was fixed for).

    si_units=False runs via run_tool's scratch copy instead: JOREK writes
    the same one fixed filename regardless of units mode
    (exec_commands.f90's zeroD_quantities hardcodes it, unaffected by the
    units toggle), so running in place would silently clobber -- then move
    away -- whatever SI cache the default call already left there. A
    scratch copy never touches run_dir/postproc/ at all; only the final
    result is copied out, to RunPaths.zero_d's si_units=False path, which
    the SI variant never uses.
    """
    from ashen.postproc import zero_d_script

    if not si_units:
        step_str = paths.step_str(step)
        scratch = paths.postproc_dir / f"_scratch_zeroD_jorek_s{step_str}"
        result = run_tool(
            run, "jorek2_postproc", step=step, dest_dir=scratch,
            outputs=[f"postproc/zeroD_quantities_s{step_str}.dat"],
            stdin_text=zero_d_script(run.namelist.name, step_str, si_units=False),
            restart_name=run.restart_path(step).name,
            copy_exe=True,
        )
        produced = result[f"postproc/zeroD_quantities_s{step_str}.dat"]
        out = paths.zero_d(step, si_units=False)
        paths.postproc_dir.mkdir(parents=True, exist_ok=True)
        if produced != out:
            shutil.move(str(produced), str(out))
        if scratch.is_dir() and not any(scratch.iterdir()):
            scratch.rmdir()
        return out

    exe = run.exe_dir / "jorek2_postproc"
    if not exe.is_file():
        raise FileNotFoundError(f"jorek2_postproc not found at {exe}")
    restart_src = run.restart_path(step)
    if not restart_src.is_file():
        raise MissingRestartError(f"restart file not found: {restart_src}")

    paths.postproc_dir.mkdir(parents=True, exist_ok=True)
    fd, script_name = tempfile.mkstemp(
        prefix="postproc_zeroD_script_", suffix=".in", dir=run.run_dir
    )
    script_path = Path(script_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(zero_d_script(run.namelist.name, paths.step_str(step), si_units=True))
        with open(script_path, encoding="utf-8") as stdin_file:
            result = subprocess.run(
                [str(exe)],
                stdin=stdin_file,
                cwd=run.run_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
    finally:
        script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise Jorek2Error(
            f"jorek2_postproc exited {result.returncode} for zeroD at step "
            f"{step} in {run.run_dir}: {result.stderr.decode(errors='replace')}"
        )

    out = paths.zero_d(step, si_units=True)
    if not out.is_file():
        raise Jorek2Error(f"zeroD_quantities not produced at {out}")
    return out
