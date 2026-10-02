"""Staging and running a ptrace: the executable a case's ptrace_exe names,
unmodified, against that case's restarts, as its ptrace_* fields configure
it (ashen.cases). Whatever it is called: ashen goes by what it writes --
its files and log lines (ashen.particle_programs) -- not by its name.

Each ptrace runs in its own folder, ``<run>/ptrace/<executable name>/``.
Restarts, the namelist and profile files are symlinked in, relative to the
folder, so the run can still be moved or renamed; the restart links are
removed again once the program exits, leaving only what it produced. A particle file for a program that
reads one is *copied*: the program overwrites part_restart.h5 with its
final state, and through a symlink that would clobber the original.

**How the restarts are linked.** The programs find their fields through
``read_jorek_fields_interp_linear`` (``mod_fields_linear.f90``), which:

- after file ``i`` looks for ``i+1 .. i+20`` only, so restarts written
  every 200 steps are never found;
- builds each name with ``rst_file_ind_fmt(1)`` alone (6 digits in
  jorek_RE), with no fallback to the other width;
- for ex6/ex7, picks its first file with ``last_file_before_time``, which
  lists ``jorek[0-9]*.h5`` through ``grep -o '[0-9]\\{5\\}'`` and opens
  ``'jorek'//i0.5//'.h5'`` -- 5 digits only.

So the chosen restarts are linked in as a consecutive sequence
(``index 0 = start_step, 1 = the next restart, ...``) under *both* widths,
``jorek00003.h5`` and ``jorek000003.h5``. Each file carries its own time
(``t_now``), so the renumbering loses nothing. The extra width does not
confuse ``last_file_before_time``: a 6-digit ``jorek0000NM.h5`` greps to
``0000N`` and sorts right after ``jorek0000N.h5``, so the numbers it
bisects over stay in ascending order. The start restart is also linked as
``jorek_restart.h5``, the name the reader uses for a frozen field (``i=-1``).

A case's ptrace_inputs are copied in under their own names, like a
particle file. A case's ptrace_<setting> keys (ptrace_dt, ptrace_initialiser,
...), over ashen's defaults for the rest, are written into
ptrace_settings.nml -- the one settings file ptrace_gc reads, and so the
record, kept in the trace folder, of what that trace ran with.

Which restarts: ptrace_start_step to ptrace_end_step, each defaulting to
the run's first and last restart (traced_steps).

A trace runs here (plan.job None: mpirun with the case's ptrace_n_mpi), or
is queued with a jobscript from site.toml's jobscripts folder, which is
then part of its fingerprint in ptrace_n_mpi's place (its srun sets the
ranks). A queued trace keeps its restart links until poll_job finds its
job ended and concludes it.

A trace is cached like the other gathers: it reruns only if its settings,
restarts, particle file or executable changed since it last completed, or
under --force.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ashen.cases import Case
from ashen.config import Site
from ashen.jorek2 import tool_output_enabled
from ashen.padding import JOREK_PAD_WIDTHS, restart_name, restart_steps
from ashen.paths import RunPaths
from ashen.particle_programs import (
    LOST_MARKER,
    NOTE_MARKER,
    OUTPUT_FILES,
    RETIRED_SETTINGS_FILES,
    SETTINGS_FILE,
    PtraceSettingsError,
    describe_settings,
    resolve_settings,
    settings_namelist,
)

__all__ = [
    "LOG_FILE",
    "META_FILE",
    "QueuedJob",
    "PARTICLES_FILE",
    "PtraceError",
    "PtracePlan",
    "PtraceResult",
    "is_current",
    "last_error",
    "last_jobscript",
    "poll_job",
    "run_path",
    "ptrace_dir",
    "ptrace_exe_path",
    "plan_ptrace",
    "run_ptrace",
    "stage",
    "traced_start",
    "traced_steps",
]

#: Folder, under the run folder, that holds one subfolder per trace.
TRACE_DIR = "ptrace"

#: The particle file the programs read at the start and write at the end.
PARTICLES_FILE = "part_restart.h5"

#: Files ashen itself writes into the ptrace folder.
META_FILE = "ptrace_meta.json"
LOG_FILE = "ptrace.log"

#: The start restart, under the name the field reader uses for a frozen field.
STATIC_RESTART = "jorek_restart.h5"

#: The restart whose current profile ptrace_gc's current_pdf_simple samples
#: (ptrace_pdf_step, else the start step).
PDF_RESTART = "jorek_pdf.h5"

#: Left behind by last_file_before_time if it is interrupted mid-listing.
_FILENUMS_GLOB = ".jorek_filenums.*"


class PtraceError(RuntimeError):
    """A trace that cannot be staged, or whose program failed."""


@dataclass(frozen=True)
class PtracePlan:
    """Everything a trace will do, decided before anything is written."""

    case: Case
    run_dir: Path
    work_dir: Path
    exe: Path
    #: Restart steps traced through, in order; the i-th is linked as index i.
    steps: list[int]
    #: (name in work_dir, file it points at): restarts, namelist, profiles.
    links: list[tuple[str, Path]]
    #: (name in work_dir, file copied there): the particle file, if any.
    copies: list[tuple[str, Path]]
    #: The namelist piped to the program on stdin (a name in `links`).
    namelist: str
    #: Shell script run in work_dir.
    command: str
    omp_threads: int
    #: Changes whenever anything that affects the result changes.
    fingerprint: str
    #: (name in work_dir, text): files ashen writes there itself -- the
    #: trace's settings as ptrace_settings.nml.
    writes: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    #: Every ptrace_gc setting the trace runs with (resolve_settings): the
    #: case's ptrace_<setting> keys over the defaults. Empty for a case that
    #: sets none -- a program that reads no settings file.
    settings: dict = dataclasses.field(default_factory=dict)
    #: The jobscript (a file in site.toml's jobscripts folder, e.g. "2h")
    #: the trace is queued with, or None to run it here (`ptrace --run_i`).
    job: str | None = None

    @property
    def restart_links(self) -> list[str]:
        """The links to JOREK restarts -- only needed while the program
        runs, so removed again afterwards (see run_ptrace)."""
        return [name for name, _ in self.links if name.startswith("jorek")]

    def describe_settings(self) -> list[str]:
        """The trace's settings, one "name = value" line each, defaults
        marked -- what `ptrace` prints when it runs or queues the trace."""
        if not self.settings:
            return []
        return describe_settings(self.settings, self.case.ptrace_settings)

    def describe(self) -> list[str]:
        """Human-readable plan, for --dry-run."""
        lines = [f"folder   {self.work_dir}", f"exe      {self.exe}"]
        restarts = set(self.restart_links)
        lines += [
            f"link     {name} -> {target}" + ("  (removed after the run)" if name in restarts else "")
            for name, target in self.links
        ]
        lines += [f"copy     {name} <- {source}" for name, source in self.copies]
        for name, text in self.writes:
            lines += [f"write    {name}:"]
            # The settings as a person reads them; the file spells out
            # every marker of a per-marker list.
            shown = self.describe_settings() if name == SETTINGS_FILE else [
                line for line in text.splitlines() if not line.startswith("!")
            ]
            lines += [f"           {line}" for line in shown]
        lines += ["queue:" if self.job else "run:"]
        lines += [f"           {line}" for line in self.command.splitlines()]
        return lines


@dataclass(frozen=True)
class PtraceResult:
    """What run_ptrace did."""

    #: False when the trace was already current and nothing ran.
    ran: bool
    #: The program stopped at a lost particle (ex7_jorek's LOST_MARKER).
    lost: bool = False
    #: Log lines the program marks as worth seeing (NOTE_MARKER), e.g.
    #: ptrace_gc stopping at the last restart before t_span was done -- and
    #: ashen's own, e.g. that no part_restart.h5 was written.
    notes: tuple[str, ...] = ()
    #: The SLURM job the trace was queued as (`ptrace --run`), if it was.
    job_id: str | None = None


def traced_steps(case: Case, available: list[int]) -> list[int]:
    """The restart steps a case traces through, in order: from
    ptrace_start_step to ptrace_end_step, which default to the first and
    last of `available` (the run's restarts)."""
    if not available:
        raise PtraceError(f"case {case.name!r}: the run has no restarts (jorek<step>.h5)")
    start, end = case.ptrace_start_step, case.ptrace_end_step
    if start is None:
        start = available[0]
    if start not in available:
        nearby = [s for s in available if abs(s - start) <= 1000][:10]
        raise PtraceError(
            f"case {case.name!r}: no restart for ptrace_start_step {start}"
            + (f"; nearby: {nearby}" if nearby else "")
        )
    return [s for s in available if s >= start and (end is None or s <= end)]


def traced_start(case: Case, run_dir: Path) -> int | None:
    """The step a case's trace started at, for the plots: ptrace_start_step,
    else the first step its last trace actually ran from (ptrace_meta.json),
    else the run's first restart. None if the run has no restarts."""
    if case.ptrace_start_step is not None:
        return case.ptrace_start_step
    steps = _read_meta(ptrace_dir(case, run_dir)).get("steps")
    if steps:
        return int(steps[0])
    available = restart_steps(run_dir)
    return available[0] if available else None


def _file_stamp(path: Path | None) -> list[int] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return [stat.st_mtime_ns, stat.st_size]


#: Case fields that decide a trace's result -- the fingerprint's input.
#: ptrace_omp_threads is left out: it changes how the work is spread, not the
#: answer. ptrace_n_mpi stays in: re_gc samples its particle count per rank.
_RESULT_FIELDS = (
    "ptrace_exe", "ptrace_start_step", "ptrace_end_step", "ptrace_particles",
    "ptrace_inputs", "ptrace_n_mpi", "namelist", "ptrace_settings", "ptrace_pdf_step",
)


def run_path(run_dir: Path, path: str | Path) -> Path:
    """A ptrace_* path: relative to the run folder unless absolute,
    normalised so "../" works. Every ptrace_* path resolves this one way."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(os.path.normpath(Path(run_dir) / path))
    return path


def ptrace_exe_path(case: Case, run_dir: Path) -> Path:
    """The executable case.ptrace_exe names (see run_path)."""
    if case.ptrace_exe is None:
        raise PtraceError(f"case {case.name!r} has no ptrace_exe")
    return run_path(run_dir, case.ptrace_exe)


def ptrace_dir(case: Case, run_dir: Path) -> Path:
    """The folder a case's ptrace runs in, and where its outputs land:
    ``<run>/ptrace/<executable filename>/``."""
    return Path(run_dir) / TRACE_DIR / ptrace_exe_path(case, run_dir).name


def plan_ptrace(
    case: Case, run_dir: Path, site: Site, *, omp_threads: int, job: str | None = None,
) -> PtracePlan:
    """Decide everything about a case's ptrace without touching disk.

    omp_threads is the fallback for a case that leaves ptrace_omp_threads at 0.
    job names a jobscript in site.toml's jobscripts folder to queue the trace
    with (`ptrace --run`); None runs it here, with the case's ptrace_n_mpi
    ranks and ptrace_omp_threads threads (`ptrace --run_i`).
    """
    run_dir = Path(run_dir)
    exe = ptrace_exe_path(case, run_dir)
    steps = traced_steps(case, restart_steps(run_dir))
    paths = RunPaths.detect(run_dir)
    links = [
        (restart_name(index, width), paths.restart(step))
        for index, step in enumerate(steps)
        for width in JOREK_PAD_WIDTHS
    ]
    links.append((STATIC_RESTART, paths.restart(steps[0])))
    pdf_step = case.ptrace_pdf_step if case.ptrace_pdf_step is not None else steps[0]
    if not paths.restart(pdf_step).is_file():
        raise PtraceError(f"case {case.name!r}: no restart for ptrace_pdf_step {pdf_step}")
    links.append((PDF_RESTART, paths.restart(pdf_step)))

    namelist = run_dir / case.namelist
    if not namelist.is_file():
        raise PtraceError(f"case {case.name!r}: namelist {namelist} not found")
    links.append((namelist.name, namelist))
    links += [
        (name, run_dir / name)
        for name in paths.profile_files
        if (run_dir / name).is_file()
    ]

    copies = []
    particles = None
    if case.ptrace_particles is not None:
        particles = run_path(run_dir, case.ptrace_particles)
        if not particles.is_file():
            raise PtraceError(
                f"case {case.name!r}: ptrace_particles {case.ptrace_particles!r} not found "
                f"(looked for {particles}; ptrace_* paths are relative to the run folder)"
            )
        copies.append((PARTICLES_FILE, particles))
    for entry in case.ptrace_inputs:
        source = run_path(run_dir, entry)
        if not source.is_file():
            raise PtraceError(
                f"case {case.name!r}: ptrace_inputs file {entry!r} not found "
                f"(looked for {source}; ptrace_* paths are relative to the run folder)"
            )
        copies.append((source.name, source))
    writes = []
    resolved: dict = {}
    if case.ptrace_settings:
        try:
            resolved = resolve_settings(case.ptrace_settings)
        except PtraceSettingsError as exc:
            raise PtraceError(f"case {case.name!r}: {exc}") from None
        writes.append((SETTINGS_FILE, settings_namelist(resolved, header=(
            f"case {case.name}: restart steps {steps[0]}..{steps[-1]} "
            f"({len(steps)}), linked from index 0; jorek_pdf.h5 is step {pdf_step}",
        ))))

    work_dir = ptrace_dir(case, run_dir)
    settings = {key: getattr(case, key) for key in _RESULT_FIELDS}
    # Every setting, defaults included: a default that changes in ashen
    # changes the result as much as a key the case sets.
    settings["ptrace_settings"] = resolved
    settings["ptrace_particles"] = str(particles) if particles else None
    settings["ptrace_inputs"] = [str(source) for name, source in copies if name != PARTICLES_FILE]
    fingerprint_input = {
        "settings": settings, "steps": steps, "exe": exe.name,
        "exe_stamp": _file_stamp(exe),
        "copy_stamps": [_file_stamp(source) for _, source in copies],
    }

    threads = case.ptrace_omp_threads or omp_threads
    if job is None:
        mpirun = site.launch.mpirun_cmd(case.ptrace_n_mpi)
        command = "\n".join(
            line for line in (
                site.launch.interactive_prelude,
                # After the prelude, which may export its own (site.example.toml's does).
                f"export OMP_NUM_THREADS={threads}",
                f"{mpirun} {exe} < {namelist.name}".strip(),
            ) if line
        )
    else:
        jobscript = _jobscript(site, job)
        # The jobscript's own srun decides the ranks, not ptrace_n_mpi -- and
        # re_gc samples its particle count per rank, so which jobscript (and
        # what is in it) is part of the result instead.
        settings["ptrace_n_mpi"] = None
        fingerprint_input["jobscript"] = [
            job, hashlib.sha256(jobscript.read_bytes()).hexdigest(),
        ]
        # The jobscripts take <exe> <stdin file> <log> [job name] and run
        # `srun ./"$exe" < ./"$input" | tee "$log"` in the submitting folder
        # (#SBATCH -D ./), so the exe goes in relative to the trace folder.
        command = "\n".join(
            line for line in (
                site.launch.batch_prelude,
                " ".join(shlex.quote(word) for word in (
                    "sbatch", "--parsable", str(jobscript),
                    os.path.relpath(exe, work_dir), namelist.name, LOG_FILE,
                    _job_name(case),
                )),
            ) if line
        )
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True).encode()
    ).hexdigest()

    return PtracePlan(
        case=case, run_dir=run_dir,
        work_dir=work_dir,
        exe=exe, steps=steps, links=links, copies=copies, namelist=namelist.name,
        command=command, omp_threads=threads, fingerprint=fingerprint,
        writes=writes, job=job, settings=resolved,
    )


def _jobscript(site: Site, job: str) -> Path:
    """The jobscript `ptrace --run -job <job>` queues with: a file in
    site.toml's jobscripts folder."""
    folder = site.jobscripts
    path = folder / job
    if Path(job).name != job or not path.is_file():
        known = sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
        raise PtraceError(
            f"no jobscript {job!r} in {folder}"
            + (f"; there: {', '.join(known)}" if known else " (site.toml's jobscripts)")
        )
    return path


def _job_name(case: Case) -> str:
    """The SLURM job name: "pt_" and the last part of the case's name."""
    tail = case.name.rstrip("/").rsplit("/", 1)[-1]
    return "pt_" + re.sub(r"[^A-Za-z0-9._-]", "_", tail)


def _read_meta(work_dir: Path) -> dict:
    try:
        return json.loads((work_dir / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_meta(work_dir: Path, meta: dict) -> None:
    (work_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _write_meta(
    plan: PtracePlan, *, complete: bool, lost: bool = False, notes: tuple[str, ...] = (),
    **extra,
) -> None:
    meta = {
        "fingerprint": plan.fingerprint,
        "complete": complete,
        "lost": lost,
        "notes": list(notes),
        "case": plan.case.name,
        "steps": plan.steps,
        "exe": plan.exe.name,
        "copied": [name for name, _ in plan.copies],
        "jobscript": plan.job,
        **extra,
    }
    _save_meta(plan.work_dir, meta)


def last_jobscript(case: Case, run_dir: Path) -> str | None:
    """The jobscript the case's last trace was queued with (None: run here,
    or never traced) -- the launch to compare the current settings against."""
    return _read_meta(ptrace_dir(case, run_dir)).get("jobscript")


def is_current(plan: PtracePlan) -> bool:
    """Whether this exact trace already ran to completion."""
    meta = _read_meta(plan.work_dir)
    return meta.get("fingerprint") == plan.fingerprint and meta.get("complete") is True


def _is_managed(entry: Path, plan: PtracePlan, previously_copied: list[str]) -> bool:
    """Whether ashen or the program put `entry` in the ptrace folder. Only
    these are cleared on restaging; anything else is left alone."""
    return (
        entry.is_symlink()
        or entry.name in (
            META_FILE, LOG_FILE, SETTINGS_FILE, *RETIRED_SETTINGS_FILES, *OUTPUT_FILES,
        )
        or entry.name in previously_copied
        or entry.name in dict(plan.copies)
        or entry.match("part_restart*.h5")
        or entry.match(_FILENUMS_GLOB)
    )


def stage(plan: PtracePlan) -> None:
    """Populate the ptrace folder, clearing only what a trace put there before.

    A leftover part_restart.h5 in particular must go: a program that reads
    one would otherwise continue the last trace instead of starting fresh.
    """
    work_dir = plan.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    previously_copied = _read_meta(work_dir).get("copied", [])
    for entry in work_dir.iterdir():
        if _is_managed(entry, plan, previously_copied):
            entry.unlink()
    for name, target in plan.links:
        (work_dir / name).symlink_to(os.path.relpath(target, work_dir))
    for name, source in plan.copies:
        shutil.copy2(source, work_dir / name)
    for name, text in plan.writes:
        (work_dir / name).write_text(text, encoding="utf-8")
    _write_meta(plan, complete=False, particles_stamp=_file_stamp(work_dir / PARTICLES_FILE))


def _log_tail(log: Path, n: int = 15) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n  ".join(lines[-n:])


def _launch(plan: PtracePlan) -> int:
    """Run the plan's command in its folder, writing everything to ptrace.log
    (and echoing it live under --tool-output). Returns the exit status."""
    echo = tool_output_enabled()
    with open(plan.work_dir / LOG_FILE, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            plan.command, shell=True, cwd=plan.work_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for raw in iter(proc.stdout.readline, b""):
            text = raw.decode(errors="replace")
            log.write(text)
            if echo:
                sys.stderr.write(text)
                sys.stderr.flush()
        proc.stdout.close()
        return proc.wait()


def _remove_restart_links(work_dir: Path) -> None:
    """The restart links are the program's input, not a result: leave the
    folder holding only what the run produced. (Every one is named jorek*;
    see PtracePlan.restart_links.)"""
    for entry in work_dir.glob("jorek*"):
        if entry.is_symlink():
            entry.unlink()


def _outcome(work_dir: Path) -> tuple[bool, tuple[str, ...], bool]:
    """(lost, notes, wrote_particles) from what a finished program left:
    its log, and a part_restart.h5 it wrote -- not one ashen copied in from
    ptrace_particles and the program left untouched (stage records that
    one's stamp)."""
    log = work_dir / LOG_FILE
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    stamp = _file_stamp(work_dir / PARTICLES_FILE)
    wrote = stamp is not None and stamp != _read_meta(work_dir).get("particles_stamp")
    return LOST_MARKER in text, _notes(text, NOTE_MARKER), wrote


def run_ptrace(plan: PtracePlan, *, force: bool = False) -> PtraceResult:
    """Stage and run one trace (plan.job None), or stage and queue it with
    plan.job's jobscript -- unless it is already current.

    Raises PtraceError if the executable is missing, the program exits
    non-zero, sbatch fails, or a job queued for this trace earlier is still
    queued or running (restaging would pull its files from under it, so not
    even --force does that). A program that stops at its first lost particle
    (ex7_jorek's LOST_MARKER) has done what it does, and is reported as
    lost; one that exits 0 without writing part_restart.h5 gets a note.
    A queued trace is concluded later, by poll_job.
    """
    queued = poll_job(plan.work_dir)
    if queued is not None and queued.running:
        raise PtraceError(
            f"case {plan.case.name!r}: job {queued.job_id} ({queued.state}) is still queued "
            f"or running for this trace; wait for it, or `scancel {queued.job_id}` first"
        )
    if not force and is_current(plan):
        meta = _read_meta(plan.work_dir)
        return PtraceResult(
            ran=False, lost=meta.get("lost", False), notes=tuple(meta.get("notes", ())),
            job_id=meta.get("job_id"),
        )
    if not plan.exe.is_file():
        raise PtraceError(
            f"case {plan.case.name!r}: ptrace_exe {plan.case.ptrace_exe!r} not found "
            f"(looked for {plan.exe}; ptrace_exe is relative to the run folder). "
            "Build it with `make <program>` in the JOREK checkout, with the same "
            "MODEL as this run, and copy the binary there."
        )
    stage(plan)
    if plan.job is not None:
        return _submit(plan)
    try:
        status = _launch(plan)
    finally:
        _remove_restart_links(plan.work_dir)
    log = plan.work_dir / LOG_FILE
    if status != 0:
        raise PtraceError(
            f"case {plan.case.name!r}: {plan.exe.name} exited {status}; "
            f"last lines of {log}:\n  {_log_tail(log)}"
        )
    lost, notes, wrote = _outcome(plan.work_dir)
    if not lost and not wrote:
        notes += (
            f"{plan.exe.name} exited 0 without writing {PARTICLES_FILE}, which every "
            f"JOREK particle program writes at the end -- see {log}",
        )
    _write_meta(plan, complete=True, lost=lost, notes=notes)
    return PtraceResult(ran=True, lost=lost, notes=notes)


def _submit(plan: PtracePlan) -> PtraceResult:
    """Queue a staged trace with sbatch. The restart links stay until
    poll_job finds the job finished: the job reads them."""
    proc = subprocess.run(
        plan.command, shell=True, cwd=plan.work_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # sbatch --parsable prints "<id>" or "<id>;<cluster>"; the prelude may print first.
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    job_id = lines[-1].split(";")[0] if lines else ""
    if proc.returncode != 0 or not job_id.isdigit():
        _remove_restart_links(plan.work_dir)
        raise PtraceError(
            f"case {plan.case.name!r}: sbatch failed (exit {proc.returncode}):\n  "
            + "\n  ".join(lines[-15:])
        )
    _write_meta(
        plan, complete=False, job_id=job_id,
        particles_stamp=_read_meta(plan.work_dir).get("particles_stamp"),
    )
    return PtraceResult(ran=True, job_id=job_id)


#: SLURM states of a job that has ended (squeue may still list one briefly).
_ENDED = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
})


@dataclass(frozen=True)
class QueuedJob:
    """A trace queued with `ptrace --run`, as poll_job found it."""

    job_id: str
    #: SLURM's state for it (PENDING, RUNNING, COMPLETED, TIMEOUT, ...), or
    #: "" once SLURM no longer knows the job.
    state: str
    #: Still queued or running: its folder must be left alone.
    running: bool
    #: Once it has ended: what it did, or why it is counted as failed.
    result: PtraceResult | None = None
    error: str | None = None


def _slurm(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        raise PtraceError(
            f"cannot ask SLURM about a queued trace: {command[0]} not found "
            "(run ptrace on the machine the job was queued from)"
        ) from None
    return proc.stdout if proc.returncode == 0 else ""


def _job_state(job_id: str) -> tuple[str, bool]:
    """(state, still running). squeue lists live jobs; sacct, if the site
    keeps accounting, says how an ended one ended."""
    state = _slurm(["squeue", "-h", "-j", job_id, "-o", "%T"]).strip().split("\n")[0].strip()
    if state and state not in _ENDED:
        return state, True
    if not state:
        try:
            out = _slurm(["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "State"])
        except PtraceError:
            out = ""
        words = out.split()
        # "CANCELLED by 1234" -> CANCELLED
        state = words[0] if words else ""
    return state, False


def poll_job(work_dir: Path) -> QueuedJob | None:
    """The job a trace was last queued as, concluding it if it has ended:
    restart links removed, its outcome written to ptrace_meta.json (so it
    is cached, or reported failed, from then on). None if the folder's last
    trace was not queued, or its job was concluded already.

    A queued trace's exit status is lost -- the jobscripts pipe the program
    through `tee` -- so it counts as done only if SLURM did not report it
    failed *and* it wrote part_restart.h5 (every JOREK particle program
    does, at the end) or stopped at a lost particle.
    """
    work_dir = Path(work_dir)
    meta = _read_meta(work_dir)
    job_id = meta.get("job_id")
    if not job_id or meta.get("complete") or meta.get("job_state") is not None:
        return None
    state, running = _job_state(job_id)
    if running:
        return QueuedJob(job_id=job_id, state=state, running=True)

    _remove_restart_links(work_dir)
    lost, notes, wrote = _outcome(work_dir)
    log = work_dir / LOG_FILE
    error = None
    if state and state != "COMPLETED":
        error = f"job {job_id} ended {state}"
        if state == "TIMEOUT":
            error += " (out of time: a longer jobscript, e.g. -job 23h, or a shorter trace)"
    elif not lost and not wrote:
        error = f"job {job_id} ended without the program writing {PARTICLES_FILE}"
    if error is not None:
        slurm_logs = sorted(p.name for p in work_dir.glob(f"*.{job_id}.*"))
        error += (
            f"; see {log}" + (f" and {', '.join(slurm_logs)}" if slurm_logs else "")
            + (f"; its last lines:\n  {_log_tail(log)}" if log.is_file() else "")
        )
    meta.update(
        complete=error is None, lost=lost, notes=list(notes),
        job_state=state or "ended", error=error,
    )
    _save_meta(work_dir, meta)
    result = None if error else PtraceResult(ran=True, lost=lost, notes=notes, job_id=job_id)
    return QueuedJob(job_id=job_id, state=state, running=False, result=result, error=error)


def last_error(work_dir: Path) -> str | None:
    """Why the folder's last queued trace was counted as failed, if it was."""
    return _read_meta(Path(work_dir)).get("error")


def _notes(log_text: str, marker: str) -> tuple[str, ...]:
    """The log lines carrying marker, marker and surrounding space removed."""
    return tuple(
        line.split(marker, 1)[1].strip()
        for line in log_text.splitlines() if marker in line
    )
