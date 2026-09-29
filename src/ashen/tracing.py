"""Staging and running a trace: the executable a case's trace_exe names,
unmodified, against that case's restarts, as its trace_* fields configure
it (ashen.cases). When the executable is one of JOREK's own particle
programs, what ashen knows about it applies too (ashen.particle_programs).

Each trace runs in its own folder, ``<run>/trace/<executable name>/``. Restarts, the
namelist and profile files are symlinked in, relative to the folder, so
the run can still be moved or renamed. A particle file for a program that
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
bisects over stay in ascending order.

A trace is cached like the other gathers: it reruns only if its settings,
restarts, particle file or executable changed since it last completed, or
under --force.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
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
from ashen.postproc import read_zeroD, zero_d_is_usable
from ashen.particle_programs import PROGRAMS, Program, program_for

__all__ = [
    "LOG_FILE",
    "META_FILE",
    "PARTICLES_FILE",
    "TraceError",
    "TracePlan",
    "TraceResult",
    "is_current",
    "plan_trace",
    "run_trace",
    "stage",
]

#: Folder, under the run folder, that holds one subfolder per trace.
TRACE_DIR = "trace"

#: The particle file the programs read at the start and write at the end.
PARTICLES_FILE = "part_restart.h5"

#: Files ashen itself writes into the trace folder.
META_FILE = "trace_meta.json"
LOG_FILE = "trace.log"

#: Left behind by last_file_before_time if it is interrupted mid-listing.
_FILENUMS_GLOB = ".jorek_filenums.*"


class TraceError(RuntimeError):
    """A trace that cannot be staged, or whose program failed."""


@dataclass(frozen=True)
class TracePlan:
    """Everything a trace will do, decided before anything is written."""

    case: Case
    program: Program
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
    #: Things worth knowing before running that don't stop it.
    warnings: list[str] = dataclasses.field(default_factory=list)

    def describe(self) -> list[str]:
        """Human-readable plan, for --dry-run."""
        lines = [f"folder   {self.work_dir}", f"exe      {self.exe}"]
        lines += [f"link     {name} -> {target}" for name, target in self.links]
        lines += [f"copy     {name} <- {source}" for name, source in self.copies]
        lines += ["run:"]
        lines += [f"           {line}" for line in self.command.splitlines()]
        return lines


@dataclass(frozen=True)
class TraceResult:
    """What run_trace did."""

    #: False when the trace was already current and nothing ran.
    ran: bool
    #: The program stopped at a lost particle (Program.stops_on_loss).
    lost: bool = False


def _select_steps(case: Case, program: Program, available: list[int]) -> list[int]:
    start, end = case.trace_start_step, case.trace_end_step
    if start not in available:
        nearby = [s for s in available if abs(s - start) <= 1000][:10]
        raise TraceError(
            f"case {case.name!r}: no restart for trace_start_step {start}"
            + (f"; nearby: {nearby}" if nearby else "")
        )
    steps = [s for s in available if s >= start and (end is None or s <= end)]
    if len(steps) < 2 and not program.holds_last_field:
        raise TraceError(
            f"case {case.name!r}: {program.name} aborts when it finds no next "
            f"restart, so it needs at least two from step {start}; found {steps}"
        )
    return steps


def _start_time_warnings(program: Program, paths: RunPaths, steps: list[int]) -> list[str]:
    """For a program with a hard-coded start time: whether the linked
    restarts span it, judged from the zeroD cache when there is one."""
    start = program.fixed_start_time
    if start is None:
        return []
    first, last = paths.zero_d(steps[0]), paths.zero_d(steps[-1])
    if not (zero_d_is_usable(first) and zero_d_is_usable(last)):
        return [
            f"{program.name} always starts at t = {start:g} s; cannot check the "
            f"restarts cover it without zeroD for steps {steps[0]} and {steps[-1]} "
            "(`analyse --diag zerod` gathers it)"
        ]
    t_first, t_last = read_zeroD(first)["Time"], read_zeroD(last)["Time"]
    if t_first <= start < t_last:
        return []
    return [
        f"{program.name} always starts at t = {start:g} s, but the linked restarts "
        f"span {t_first:g} .. {t_last:g} s (steps {steps[0]}..{steps[-1]}); its "
        "fields at the start will not be the ones it expects"
    ]


def _file_stamp(path: Path | None) -> list[int] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return [stat.st_mtime_ns, stat.st_size]


#: Case fields that decide a trace's result -- the fingerprint's input.
#: trace_omp_threads is left out: it changes how the work is spread, not the
#: answer. trace_n_mpi stays in: re_gc samples its particle count per rank.
_RESULT_FIELDS = (
    "trace_exe", "trace_start_step", "trace_end_step", "trace_particles",
    "trace_n_mpi", "namelist",
)

#: Every known program's outputs: cleared on restaging whichever executable
#: runs, so a stale diag file is never mistaken for, or appended to by, a
#: new trace.
_KNOWN_OUTPUTS = frozenset(name for p in PROGRAMS.values() for name in p.outputs)


def plan_trace(case: Case, run_dir: Path, site: Site, *, omp_threads: int) -> TracePlan:
    """Decide everything about a case's trace without touching disk.

    omp_threads is the fallback for a case that leaves trace_omp_threads at 0.
    """
    if case.trace_exe is None:
        raise TraceError(f"case {case.name!r} has no trace_exe")
    run_dir = Path(run_dir)
    exe = Path(case.trace_exe)
    if not exe.is_absolute():
        exe = Path(os.path.normpath(run_dir / exe))
    program = program_for(exe)
    paths = RunPaths.detect(run_dir)

    steps = _select_steps(case, program, restart_steps(run_dir))
    links = [
        (restart_name(index, width), paths.restart(step))
        for index, step in enumerate(steps)
        for width in JOREK_PAD_WIDTHS
    ]

    namelist = run_dir / case.namelist
    if not namelist.is_file():
        raise TraceError(f"case {case.name!r}: namelist {namelist} not found")
    links.append((namelist.name, namelist))
    links += [
        (name, run_dir / name)
        for name in paths.profile_files
        if (run_dir / name).is_file()
    ]

    copies = []
    particles = case.trace_particles
    if particles is not None:
        if not particles.is_file():
            raise TraceError(f"case {case.name!r}: trace_particles {particles} not found")
        copies.append((PARTICLES_FILE, particles))

    threads = case.trace_omp_threads or omp_threads
    mpirun = site.launch.mpirun_cmd(case.trace_n_mpi)
    command = "\n".join(
        line for line in (
            site.launch.interactive_prelude,
            # After the prelude, which may export its own (site.example.toml's does).
            f"export OMP_NUM_THREADS={threads}",
            f"{mpirun} {exe} < {namelist.name}".strip(),
        ) if line
    )

    settings = {key: getattr(case, key) for key in _RESULT_FIELDS}
    settings["trace_particles"] = str(particles) if particles else None
    fingerprint = hashlib.sha256(json.dumps(
        {"settings": settings, "steps": steps, "exe": exe.name,
         "exe_stamp": _file_stamp(exe), "particles_stamp": _file_stamp(particles)},
        sort_keys=True,
    ).encode()).hexdigest()

    return TracePlan(
        case=case, program=program, run_dir=run_dir,
        work_dir=run_dir / TRACE_DIR / exe.name,
        exe=exe, steps=steps, links=links, copies=copies, namelist=namelist.name,
        command=command, omp_threads=threads, fingerprint=fingerprint,
        warnings=_start_time_warnings(program, paths, steps),
    )


def _read_meta(work_dir: Path) -> dict:
    try:
        return json.loads((work_dir / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_meta(plan: TracePlan, *, complete: bool, lost: bool = False) -> None:
    meta = {
        "fingerprint": plan.fingerprint,
        "complete": complete,
        "lost": lost,
        "case": plan.case.name,
        "program": plan.program.name,
        "steps": plan.steps,
        "exe": plan.exe.name,
    }
    (plan.work_dir / META_FILE).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def is_current(plan: TracePlan) -> bool:
    """Whether this exact trace already ran to completion."""
    meta = _read_meta(plan.work_dir)
    return meta.get("fingerprint") == plan.fingerprint and meta.get("complete") is True


def _is_managed(entry: Path, plan: TracePlan) -> bool:
    """Whether ashen or the program put `entry` in the trace folder. Only
    these are cleared on restaging; anything else is left alone."""
    return (
        entry.is_symlink()
        or entry.name in (META_FILE, LOG_FILE, *_KNOWN_OUTPUTS, *plan.program.outputs)
        or entry.match("part_restart*.h5")
        or entry.match(_FILENUMS_GLOB)
    )


def stage(plan: TracePlan) -> None:
    """Populate the trace folder, clearing only what a trace put there before.

    A leftover part_restart.h5 in particular must go: a program that reads
    one would otherwise continue the last trace instead of starting fresh.
    """
    work_dir = plan.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    for entry in work_dir.iterdir():
        if _is_managed(entry, plan):
            entry.unlink()
    for name, target in plan.links:
        (work_dir / name).symlink_to(os.path.relpath(target, work_dir))
    for name, source in plan.copies:
        shutil.copy2(source, work_dir / name)
    _write_meta(plan, complete=False)


def _log_tail(log: Path, n: int = 15) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n  ".join(lines[-n:])


def _launch(plan: TracePlan) -> int:
    """Run the plan's command in its folder, writing everything to trace.log
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


def run_trace(plan: TracePlan, *, force: bool = False) -> TraceResult:
    """Stage and run one trace, unless it is already current.

    Raises TraceError if the executable is missing, the program exits
    non-zero, or it exits without writing its outputs -- except that a
    program which stops at its first lost particle (ex7_jorek) has done
    what it does, and is reported as lost rather than failed.
    """
    if not force and is_current(plan):
        return TraceResult(ran=False, lost=_read_meta(plan.work_dir).get("lost", False))
    spec = plan.program
    if not plan.exe.is_file():
        build = (
            f" Build it with `make {spec.name}` in the JOREK checkout, with the "
            f"same MODEL as this run, and copy the binary there."
            if spec.known else ""
        )
        raise TraceError(
            f"case {plan.case.name!r}: trace_exe {plan.case.trace_exe!r} not found "
            f"(looked for {plan.exe}; trace_exe is relative to the run folder).{build}"
        )
    stage(plan)
    status = _launch(plan)
    log = plan.work_dir / LOG_FILE
    if status != 0:
        raise TraceError(
            f"case {plan.case.name!r}: {plan.exe.name} exited {status}; "
            f"last lines of {log}:\n  {_log_tail(log)}"
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    lost = spec.stops_on_loss is not None and spec.stops_on_loss in text
    if not lost:
        missing = [name for name in spec.outputs if not (plan.work_dir / name).is_file()]
        if missing:
            raise TraceError(
                f"case {plan.case.name!r}: {plan.exe.name} exited 0 but did not "
                f"write {missing}; last lines of {log}:\n  {_log_tail(log)}"
            )
    _write_meta(plan, complete=True, lost=lost)
    return TraceResult(ran=True, lost=lost)
