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
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ashen.paths import RunPaths, step_name_variants

__all__ = [
    "Jorek2Error", "MissingRestartError", "Jorek2Run", "ToolResult", "run_tool", "run_zero_d",
    "TOOL_OUTPUT_ENV", "enable_tool_output", "tool_output_enabled",
]

#: Set to turn on echoing of every jorek2_* tool's stdout/stderr. An
#: environment variable rather than a module global because per-step gathers
#: fan out across a ProcessPoolExecutor: a global set in the parent survives
#: a fork but not a spawn (Windows), while the environment is inherited by
#: both. Also usable directly, without a CLI flag, from a notebook or a
#: jobscript.
TOOL_OUTPUT_ENV = "ASHEN_TOOL_OUTPUT"

_FALSEY = {"", "0", "false", "no", "off"}


def tool_output_enabled() -> bool:
    """Whether jorek2_* child output should be echoed to stderr."""
    return os.environ.get(TOOL_OUTPUT_ENV, "").strip().lower() not in _FALSEY


def enable_tool_output() -> None:
    """Turn echoing on for this process and every child it spawns."""
    os.environ[TOOL_OUTPUT_ENV] = "1"


def _announce_tool(tool: str, step: int, exe: Path, cwd: Path) -> None:
    """Header printed before a tool is launched with its streams inherited.

    Printed *before* the launch, not after: the case this exists for is a
    tool that hangs, where anything printed afterwards never arrives. The
    header is also what tells you the exe was found and the process really
    started -- a hang before any tool output is a different problem from a
    hang partway through one.
    """
    print(f"--- {tool} step {step}", file=sys.stderr)
    print(f"    exe {exe}", file=sys.stderr)
    print(f"    cwd {cwd}", file=sys.stderr, flush=True)


def _tee(stream, chunks: list[str]) -> None:
    """Drain one child stream, echoing each line as it arrives and keeping it.

    Line-buffered rather than read-to-end: the whole point is to see output
    from a tool that has not exited yet. Both streams go to *stderr*
    regardless of which one they came from, so that a caller parsing our own
    stdout is unaffected by the echo.

    Reading both streams concurrently (one thread each, in _launch) is not
    optional once both are pipes: a tool that fills the stderr pipe buffer
    while we block reading stdout would deadlock, which is the failure mode
    subprocess.communicate exists to avoid and that we lose by draining
    incrementally.
    """
    for raw in iter(stream.readline, b""):
        text = raw.decode(errors="replace")
        chunks.append(text)
        sys.stderr.write(text)
        sys.stderr.flush()
    stream.close()


@dataclass(frozen=True)
class _Completed:
    returncode: int
    stdout: str
    stderr: str


def _launch(
    argv: list[str],
    *,
    stdin_file,
    cwd: Path,
    env: Mapping[str, str] | None,
    capture_stdout: bool,
    echo: bool,
) -> _Completed:
    """Run one tool, in whichever of three stream modes applies.

    - not echoing: stdout to a pipe only if the caller wants it, else
      discarded; stderr piped so a non-zero exit can quote it.
    - echoing, output not wanted back: inherit this process's streams. The
      cheapest live passthrough there is -- no decoding, no threads, and the
      tool's own buffering is all that stands between it and the terminal.
    - echoing *and* wanted back (jorek2_poincare, whose progress messages
      are parsed to demux field lines): tee. Piped, drained line by line in
      a thread per stream, echoed as it arrives and kept for the caller.
    """
    if echo and capture_stdout:
        proc = subprocess.Popen(
            argv, stdin=stdin_file, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out_chunks: list[str] = []
        err_chunks: list[str] = []
        threads = [
            threading.Thread(target=_tee, args=(proc.stdout, out_chunks), daemon=True),
            threading.Thread(target=_tee, args=(proc.stderr, err_chunks), daemon=True),
        ]
        for thread in threads:
            thread.start()
        returncode = proc.wait()
        for thread in threads:
            thread.join()
        return _Completed(returncode, "".join(out_chunks), "".join(err_chunks))

    if echo:
        result = subprocess.run(argv, stdin=stdin_file, cwd=cwd, env=env)
        return _Completed(result.returncode, "", "")

    result = subprocess.run(
        argv, stdin=stdin_file, cwd=cwd, env=env,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return _Completed(
        result.returncode,
        result.stdout.decode(errors="replace") if result.stdout is not None else "",
        result.stderr.decode(errors="replace") if result.stderr is not None else "",
    )


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



def _first_existing_variant(workdir: Path, output: str, step: int) -> Path | None:
    """The step-padding variant of ``output`` that the tool actually wrote.

    ``output`` is a path relative to the scratch dir; only its filename is
    re-padded, never its directory.
    """
    rel = Path(output)
    for name in step_name_variants(rel.name, step):
        candidate = workdir / rel.with_name(name)
        if candidate.is_file():
            return candidate
    return None


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
    exe_subdir: str | None = None,
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

    exe_subdir, if given, looks the tool up under exe_dir/<exe_subdir>
    instead of exe_dir itself -- e.g. jorek2_four uses "exe", the folder
    prepare_run symlinks to site.exe, rather than relying on a per-tool
    top-level symlink like the other jorek2_* tools.
    """
    if stdin_text is None and not stdin_is_namelist:
        raise ValueError("pass stdin_text, or stdin_is_namelist=True")
    if stdin_text is not None and stdin_is_namelist:
        raise ValueError("pass exactly one of stdin_text and stdin_is_namelist")

    exe_dir = run.exe_dir / exe_subdir if exe_subdir else run.exe_dir
    exe = exe_dir / tool
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

        # Echoing shows the tool's output as it is produced rather than
        # replaying it afterwards: a captured stream only reaches the
        # terminal once the child exits, which is exactly the output you do
        # not get from a tool that hangs. See _launch for the three stream
        # modes; a caller that also wants the output back (capture_stdout)
        # gets it teed, not taken away.
        echo = tool_output_enabled()
        if echo:
            _announce_tool(tool, step, exe_invoke, workdir)
        with open(stdin_path, encoding="utf-8") as stdin_file:
            result = _launch(
                [str(exe_invoke)],
                stdin_file=stdin_file,
                cwd=workdir,
                env=child_env,
                capture_stdout=capture_stdout,
                echo=echo,
            )
        if result.returncode != 0:
            # Inheriting the streams leaves no captured stderr to quote --
            # and it is already on screen. Teeing does keep it, so that
            # path quotes as usual.
            detail = result.stderr or "see its output above"
            raise Jorek2Error(
                f"{tool} exited {result.returncode} for step {step} in "
                f"{run.run_dir}: {detail}"
            )

        collected: dict[str, Path] = {}
        for output in outputs:
            src = workdir / output
            if not src.is_file():
                # The tool may have padded the step to a different width than
                # this run's restarts use -- that is a property of the
                # postproc binary, not of the run (paths.JOREK_PAD_WIDTHS).
                src = _first_existing_variant(workdir, output, step)
            if src is None:
                raise Jorek2Error(
                    f"{tool} did not produce expected output {output!r} for "
                    f"step {step} in {run.run_dir} (nor under any other step "
                    f"padding: {', '.join(step_name_variants(Path(output).name, step))})"
                )
            # Named by what was *asked for*, not by what the tool wrote, so
            # everything downstream of here sees one spelling.
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

    # Both are empty when the streams were inherited rather than captured
    # (_launch's echo-only mode); the caller gets nothing back because the
    # user got the live output instead.
    stdout = result.stdout if capture_stdout else ""
    return ToolResult(outputs=collected, stdout=stdout, stderr=result.stderr)


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
        echo = tool_output_enabled()
        if echo:
            _announce_tool("jorek2_postproc", step, exe, run.run_dir)
        with open(script_path, encoding="utf-8") as stdin_file:
            result = _launch(
                [str(exe)],
                stdin_file=stdin_file,
                cwd=run.run_dir,
                env=None,
                capture_stdout=False,
                echo=echo,
            )
    finally:
        script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = result.stderr or "see its output above"
        raise Jorek2Error(
            f"jorek2_postproc exited {result.returncode} for zeroD at step "
            f"{step} in {run.run_dir}: {detail}"
        )

    out = paths.zero_d(step, si_units=True)
    if not out.is_file():
        raise Jorek2Error(f"zeroD_quantities not produced at {out}")
    return out
