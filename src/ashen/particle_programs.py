"""What ashen knows about JOREK's particle programs, for `bin/ptrace`.

`bin/ptrace` runs whatever executable a case's ptrace_exe names, unmodified,
against its restarts (ashen.ptracing). When that executable's filename is one
of JOREK's own ``particles/examples`` programs listed here, ashen also
applies what it knows about it: which files it writes, whether it reads a
particle file, its hard-coded start time, how it reports a lost particle.
Any other executable is run as-is (:func:`program_for`).

ashen does not change what JOREK's own programs compute -- their particles,
energies, time step and duration are fixed in their source. ashen's own
ptrace_gc reads all of those from a &ptrace namelist, which a case can set
key by key (:data:`PTRACE_SETTINGS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OVERRIDES_FILE",
    "PROGRAMS",
    "PTRACE_SETTINGS",
    "PTRACE_SETTING_CHOICES",
    "Program",
    "overrides_namelist",
    "program_for",
]


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
    #: Its write_particle_diagnostics file (psi_n, R, Z, phi, ... over time).
    #: The plots don't go by this but by which of DIAG_FILES is in the ptrace
    #: folder, so a binary under any name is plotted the same.
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
            "markers (listed, or from the current profile), energies and snapshots "
            "from ptrace_params.nml and the case's ptrace_<setting> keys",
            # ptrace_diag.h5 every diag_step; part_restart_s<step>_t<time>.h5 every
            # snapshot_step (if > 0) and part_restart.h5 at the end.
            outputs=("ptrace_diag.h5", "part_restart.h5"),
            diag_file="ptrace_diag.h5",
            # Static fields need only one restart (with t_span > 0); by default
            # it runs to the last linked restart and stops there.
            holds_last_field=True,
            note_marker="ptrace_gc: NOTE:",
        ),
    )
}


#: ptrace_gc's &ptrace settings a case can set as ptrace_<name> in cases.toml,
#: and what each takes: "str", "bool", "int", "real", or "ints"/"reals" -- a
#: list, one value per marker (a single value is a list of one). ashen writes
#: them to OVERRIDES_FILE, which ptrace_gc reads after ptrace_params.nml.
#: restart_index is left out: ashen links the restarts from index 0.
PTRACE_SETTINGS = {
    "field_mode": "str",
    "hold_last_field": "bool",
    "t_span": "real",
    "dt": "real",
    "diag_step": "real",
    "snapshot_step": "real",
    "mass": "real",
    "initialiser": "str",
    "n_markers": "int",
    "R0": "reals",
    "Z0": "reals",
    "phi0": "reals",
    "E_kin_eV": "reals",
    "cos_pitch": "reals",
    "charge": "ints",
    "pdf_n_sub": "int",
    "pdf_n_phi": "int",
    "seed": "int",
}

#: Allowed values of the "str" settings that have a fixed set.
PTRACE_SETTING_CHOICES = {
    "field_mode": ("static", "evolving"),
    "initialiser": ("markers", "current_pdf_simple"),
}

#: The file ashen writes a case's PTRACE_SETTINGS into, as a &ptrace namelist.
OVERRIDES_FILE = "ptrace_overrides.nml"


def _fortran_value(value) -> str:
    """One namelist value: 'quoted' strings, .true./.false., d-exponent reals."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, float):
        text = repr(value)
        return text.replace("e", "d") if "e" in text else text
    return str(value)


def overrides_namelist(settings: dict) -> str:
    """The &ptrace namelist for OVERRIDES_FILE, settings in PTRACE_SETTINGS order."""
    lines = ["! Written by ashen from the case's ptrace_<setting> keys in cases.toml.",
             "! ptrace_gc reads it after ptrace_params.nml, so these values win.",
             "&ptrace"]
    for name in PTRACE_SETTINGS:
        if name not in settings:
            continue
        value = settings[name]
        values = value if isinstance(value, list) else [value]
        lines.append(f"  {name} = " + ", ".join(_fortran_value(v) for v in values))
    lines.append("/")
    return "\n".join(lines) + "\n"


#: Every diagnostics file a known program writes, in the order find_diag_file
#: prefers them.
DIAG_FILES = tuple(dict.fromkeys(
    program.diag_file for program in PROGRAMS.values() if program.diag_file
))


def find_diag_file(folder: Path | str) -> Path | None:
    """The particle diagnostics file in a ptrace folder, whatever the
    executable is called: the first of DIAG_FILES there, or None."""
    for name in DIAG_FILES:
        path = Path(folder) / name
        if path.is_file():
            return path
    return None


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

