"""The JOREK particle programs `bin/trace` can run.

Each is one of JOREK's own ``particles/examples`` programs, run unmodified
against a case's restarts (ashen.tracing; configured by a case's trace_*
fields, ashen.cases). ashen does not change what a program computes -- its
particles, energies, time step and duration are fixed in its source. What
ashen chooses is which restarts it sees, how it is launched, and (for a
program that starts from a particle file) which particles it starts from.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PROGRAMS", "Program"]


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


PROGRAMS = {
    program.name: program
    for program in (
        Program(
            name="re_gc_current_density_initialisation",
            summary="relativistic guiding centres (RK45), sampled from the current density",
            # part_diag.h5 (diag_filename), part_restart.h5 at the end;
            # part_restart*.h5 snapshots every write_step on the way.
            outputs=("part_diag.h5", "part_restart.h5"),
            reads_particles=True,  # `inquire(file=part_restart.h5)`, line 110
            holds_last_field=True,  # abort_at_last_mhd_restart = .false.
        ),
        Program(
            name="ex6_jorek",
            summary="one relativistic full-orbit electron (volume-preserving pusher)",
            outputs=("diag.h5", "part_restart.h5"),
            fixed_start_time=2.5e-3,  # `sim%time = 2.5d-3`, restart = .false.
        ),
        Program(
            name="ex7_jorek",
            summary="one relativistic guiding-centre electron (fixed-step RK4)",
            outputs=("diag.h5", "part_restart.h5"),
            fixed_start_time=2.5e-3,  # `sim%time = 2.5d-3`, restart = .false.
            stops_on_loss="PARTICLE IS LOST, STOPPING",
        ),
    )
}
