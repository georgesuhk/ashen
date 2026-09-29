"""What ashen knows about JOREK's particle programs, for `bin/ptrace`.

`bin/ptrace` runs whatever executable a case's ptrace_exe names, unmodified,
against its restarts (ashen.ptracing). When that executable's filename is one
of JOREK's own ``particles/examples`` programs listed here, ashen also
applies what it knows about it: which files it writes, whether it reads a
particle file, its hard-coded start time, how it reports a lost particle.
Any other executable is run as-is (:func:`program_for`).

ashen does not change what a program computes -- its particles, energies,
time step and duration are fixed in its source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["PROGRAMS", "Program", "program_for"]


@dataclass(frozen=True)
class Program:
    """What ashen needs to know about one JOREK particle program. Every
    value here restates the program's own source; see each entry."""

    name: str
    summary: str
    #: Files it writes in its working folder, all expected after a clean run.
    outputs: tuple[str, ...]
    #: Starts from part_restart.h5 when that file exists, instead of
    #: generating its own particles.
    reads_particles: bool = False
    #: Start time [s] hard-coded in the source, or None when the start is
    #: the first restart's own time.
    fixed_start_time: float | None = None
    #: Stops at the first particle it loses (logging this line) rather than
    #: carrying on; its final-state output is then never written.
    stops_on_loss: str | None = None
    #: With a single restart, keeps that field frozen (True) rather than
    #: aborting when it finds no next restart (stop_at_end=.true.).
    holds_last_field: bool = False
    #: Files it reads from its working folder that the case must supply
    #: through ptrace_inputs.
    required_inputs: tuple[str, ...] = ()
    #: Its write_particle_diagnostics file (psi_n, R, Z, phi, ... over time),
    #: which `plot --diag particle_exits` reads. None = none known.
    diag_file: str | None = None
    #: Log lines containing this are things the user should see even when
    #: the run succeeds (e.g. it stopped before its time span was done);
    #: bin/ptrace repeats them after the run.
    note_marker: str | None = None
    #: One of PROGRAMS, as opposed to an executable ashen knows nothing about.
    known: bool = True


PROGRAMS = {
    program.name: program
    for program in (
        Program(
            name="re_gc_current_density_initialisation",
            summary="relativistic guiding centres (RK45), sampled from the current density",
            # part_diag.h5 (diag_filename), part_restart.h5 at the end;
            # part_restart*.h5 snapshots every write_step on the way.
            outputs=("part_diag.h5", "part_restart.h5"),
            diag_file="part_diag.h5",
            reads_particles=True,  # `inquire(file=part_restart.h5)`, line 110
            holds_last_field=True,  # abort_at_last_mhd_restart = .false.
        ),
        Program(
            name="ex6_jorek",
            summary="one relativistic full-orbit electron (volume-preserving pusher)",
            outputs=("diag.h5", "part_restart.h5"),
            diag_file="diag.h5",
            fixed_start_time=2.5e-3,  # `sim%time = 2.5d-3`, restart = .false.
        ),
        Program(
            name="ex7_jorek",
            summary="one relativistic guiding-centre electron (fixed-step RK4)",
            outputs=("diag.h5", "part_restart.h5"),
            diag_file="diag.h5",
            fixed_start_time=2.5e-3,  # `sim%time = 2.5d-3`, restart = .false.
            stops_on_loss="PARTICLE IS LOST, STOPPING",
        ),
        Program(
            name="ptrace_gc",
            summary="ashen's configurable guiding-centre tracer (fortran/ptrace_gc.f90): "
            "markers, energies, time span and snapshots from ptrace_params.nml",
            # ptrace_diag.h5 every diag_step; part_restart_s<step>_t<time>.h5 every
            # snapshot_step (if > 0) and part_restart.h5 at the end.
            outputs=("ptrace_diag.h5", "part_restart.h5"),
            diag_file="ptrace_diag.h5",
            # Runs on one restart; without hold_last_field it stops at the last
            # restart's time -- outputs written, a NOTE logged -- if t_span
            # reaches past it.
            holds_last_field=True,
            required_inputs=("ptrace_params.nml",),
            note_marker="ptrace_gc: NOTE:",
        ),
    )
}


def program_for(exe: Path | str) -> Program:
    """What ashen knows about the program at `exe`, recognised by filename.

    An unrecognised executable gets no program-specific handling: no
    expected outputs (a zero exit is success), no start-time check, a
    single restart allowed, and a particle file passed through if given.
    """
    name = Path(exe).name
    known = PROGRAMS.get(name)
    if known is not None:
        return known
    return Program(
        name=name,
        summary="not a program ashen knows; run as-is",
        outputs=(),
        reads_particles=True,
        holds_last_field=True,
        known=False,
    )

