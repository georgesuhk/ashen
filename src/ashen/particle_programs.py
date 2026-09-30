"""What ashen knows about the files and logs of JOREK's particle programs
and its own ptrace_gc, for `bin/ptrace` and the particle plots.

`bin/ptrace` runs whatever executable a case's ptrace_exe names, as-is
(ashen.ptracing). Nothing here depends on that executable's name: what
ashen reacts to is what the program writes -- the files in its folder
(:data:`OUTPUT_FILES`, :data:`DIAG_FILES`) and lines in its log
(:data:`NOTE_MARKER`, :data:`LOST_MARKER`). So a program built under any
name, e.g. ptrace_gc per model, is handled the same.

ptrace_gc (fortran/ptrace_gc.f90) reads its settings from a &ptrace
namelist, which a case can also set key by key (:data:`PTRACE_SETTINGS`).
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DIAG_FILES",
    "LOST_MARKER",
    "NOTE_MARKER",
    "OUTPUT_FILES",
    "OVERRIDES_FILE",
    "PTRACE_SETTINGS",
    "PTRACE_SETTING_CHOICES",
    "find_diag_file",
    "overrides_namelist",
]

#: write_particle_diagnostics files, in the order find_diag_file prefers
#: them: ptrace_gc's, re_gc_current_density_initialisation's, ex6/ex7's.
DIAG_FILES = ("ptrace_diag.h5", "part_diag.h5", "diag.h5")

#: Every output file the particle programs write under a fixed name --
#: cleared when a ptrace folder is restaged, so a stale one is never
#: mistaken for, or appended to by, a new trace. (Snapshots,
#: part_restart*.h5, are cleared by pattern.)
OUTPUT_FILES = (*DIAG_FILES, "part_restart.h5")

#: ptrace_gc marks log lines the user should see even when the run succeeds
#: (e.g. it stopped before t_span was done); bin/ptrace repeats them.
NOTE_MARKER = "ptrace_gc: NOTE:"

#: ex7_jorek logs this and stops at the first particle it loses, without
#: writing its final output -- what it does, not a failure.
LOST_MARKER = "PARTICLE IS LOST, STOPPING"


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


def find_diag_file(folder: Path | str) -> Path | None:
    """The particle diagnostics file in a ptrace folder, whatever the
    executable is called: the first of DIAG_FILES there, or None."""
    for name in DIAG_FILES:
        path = Path(folder) / name
        if path.is_file():
            return path
    return None
