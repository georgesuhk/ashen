"""What ashen knows about the files and logs of JOREK's particle programs
and its own ptrace_gc, for `bin/ptrace` and the particle plots.

`bin/ptrace` runs whatever executable a case's ptrace_exe names, as-is
(ashen.ptracing). Nothing here depends on that executable's name: what
ashen reacts to is what the program writes -- the files in its folder
(:data:`OUTPUT_FILES`, :data:`DIAG_FILES`) and lines in its log
(:data:`NOTE_MARKER`, :data:`LOST_MARKER`). So a program built under any
name, e.g. ptrace_gc per model, is handled the same.

ptrace_gc (fortran/ptrace_gc.f90) reads its settings from one &ptrace
namelist, :data:`SETTINGS_FILE`, which ashen writes into the trace folder
from the case's ptrace_<name> keys (:data:`PTRACE_SETTINGS`) over
:data:`PTRACE_DEFAULTS` -- every setting, so the file is also the record of
what that trace ran with (:func:`resolve_settings`,
:func:`settings_namelist`).
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "BOUNDARY_FILE",
    "DIAG_FILES",
    "LOST_MARKER",
    "NOTE_MARKER",
    "MAX_MARKERS",
    "OUTPUT_FILES",
    "PTRACE_DEFAULTS",
    "PTRACE_SETTINGS",
    "PTRACE_SETTING_CHOICES",
    "PtraceSettingsError",
    "RETIRED_SETTINGS_FILES",
    "SETTINGS_FILE",
    "boundary_file_text",
    "describe_settings",
    "energy_text",
    "find_diag_file",
    "resolve_settings",
    "settings_namelist",
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
#: them, over PTRACE_DEFAULTS, to SETTINGS_FILE. restart_index is not one of
#: them: ashen links the restarts from index 0, and writes it as 0.
PTRACE_SETTINGS = {
    "field_mode": "str",
    "hold_last_field": "bool",
    "t_span": "real",
    "dt": "real",
    "diag_step": "real",
    "snapshot_step": "real",
    "n_snapshots": "int",
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
    "seed": "int",
    "stop_when_stalled": "bool",
    "stall_rate_fraction": "real",
    "stall_min_lost": "real",
    "stall_min_time": "real",
    "stall_window": "real",
}

#: Allowed values of the "str" settings that have a fixed set.
PTRACE_SETTING_CHOICES = {
    "field_mode": ("static", "evolving"),
    "initialiser": ("markers", "current_pdf_simple"),
}

#: What a setting is when the case does not set it -- the same values
#: ptrace_gc.f90 declares (tests/unit/test_particle_programs.py holds the
#: two together). Not here, so required: n_markers, E_kin_eV and cos_pitch,
#: and R0 and Z0 for the 'markers' initialiser.
PTRACE_DEFAULTS = {
    "field_mode": "evolving",
    "hold_last_field": False,
    "t_span": 0.0,
    "dt": 1e-10,
    "diag_step": 1e-8,
    # 0: ptrace_gc spaces the snapshots itself, about n_snapshots of them.
    "snapshot_step": 0.0,
    "n_snapshots": 100,
    "mass": 5.48579909065e-4,
    "initialiser": "markers",
    "phi0": [0.0],
    "charge": [-1],
    "pdf_n_sub": 4,
    "seed": 1,
    # Stop before the end once the loss rate, over stall_window, has fallen
    # to stall_rate_fraction of its peak -- looked at only once
    # stall_min_lost of the markers has left or stall_min_time of the trace
    # has passed. Off unless the case asks. stall_window 0: a tenth of the trace.
    "stop_when_stalled": False,
    "stall_rate_fraction": 0.05,
    "stall_min_lost": 0.10,
    "stall_min_time": 0.5,
    "stall_window": 0.0,
}

#: The outline ptrace_gc's stop_when_stalled counts a marker as having left
#: the plasma by crossing: written into the trace folder from the run's
#: original_bnd.dat (the boundary before extend_bnd), as a line with n, then
#: n lines of R Z. Without it, a marker has left once it is off the grid.
BOUNDARY_FILE = "ptrace_boundary.dat"


def boundary_file_text(points) -> str:
    """BOUNDARY_FILE's text for an (N, 2) outline of (R, Z) [m]."""
    return f"{len(points)}\n" + "".join(f"{float(r)!r} {float(z)!r}\n" for r, z in points)


#: ptrace_gc's MAX_MARKERS: the length of its marker arrays.
MAX_MARKERS = 100000

#: The settings that are one value per marker.
_PER_MARKER = ("R0", "Z0", "phi0", "E_kin_eV", "cos_pitch", "charge")
#: Of those, where each marker starts -- only the 'markers' initialiser's.
_POSITIONS = ("R0", "Z0", "phi0")

#: The file ashen writes a trace's settings into, as a &ptrace namelist --
#: all of them, so it is also the record of what the trace ran with.
SETTINGS_FILE = "ptrace_settings.nml"

#: Files earlier versions read settings from. ptrace_gc no longer reads
#: them, so one in a case's ptrace_inputs would be silently ignored.
RETIRED_SETTINGS_FILES = ("ptrace_params.nml", "ptrace_overrides.nml")


class PtraceSettingsError(ValueError):
    """A case's ptrace_<setting> keys that do not make a valid trace."""


def resolve_settings(settings: dict) -> dict:
    """Every ptrace_gc setting a trace runs with, in PTRACE_SETTINGS order:
    the case's own (checked and normalised, as ashen.cases leaves them)
    over PTRACE_DEFAULTS.

    Snapshots: a snapshot_step the case sets is used as it is, and
    n_snapshots becomes 0; otherwise snapshot_step stays 0 and ptrace_gc
    spaces about n_snapshots of them over the trace.

    Per-marker settings: under 'markers' a single value stands for every
    marker and a list must have n_markers values (ptrace_gc itself would
    leave the markers past a short list at energy 0); under
    'current_pdf_simple' every marker shares one energy, pitch and charge,
    and R0/Z0/phi0 -- which it would ignore -- are refused.
    Raises PtraceSettingsError, naming the ptrace_<name> key at fault.
    """
    def bad(message: str) -> PtraceSettingsError:
        return PtraceSettingsError(message)

    merged = {**PTRACE_DEFAULTS, **settings}
    for name in ("n_markers", "E_kin_eV", "cos_pitch"):
        if name not in merged:
            raise bad(f"ptrace_{name} is not set, and has no default")
    n = merged["n_markers"]
    if not 1 <= n <= MAX_MARKERS:
        raise bad(f"ptrace_n_markers must be in 1..{MAX_MARKERS}, got {n}")
    for name, low in (("dt", 0.0), ("diag_step", 0.0), ("mass", 0.0)):
        if not merged[name] > low:
            raise bad(f"ptrace_{name} must be > 0, got {merged[name]}")
    for name in ("t_span", "snapshot_step"):
        if merged[name] < 0:
            raise bad(f"ptrace_{name} must be >= 0, got {merged[name]}")
    if merged["n_snapshots"] < 0:
        raise bad(f"ptrace_n_snapshots must be >= 0, got {merged['n_snapshots']}")
    # A snapshot_step the case sets is the spacing, whatever n_snapshots says
    # (0: only the final snapshot). Written as n_snapshots = 0, so the file
    # says the same thing to ptrace_gc, whose own default is 100.
    if "snapshot_step" in settings:
        merged["n_snapshots"] = 0
    if not 0 < merged["stall_rate_fraction"] < 1:
        raise bad("ptrace_stall_rate_fraction must be between 0 and 1 (0.05: stop once the "
                  f"loss rate is 5 % of its peak), got {merged['stall_rate_fraction']}")
    for name in ("stall_min_lost", "stall_min_time"):
        if not 0 <= merged[name] <= 1:
            raise bad(f"ptrace_{name} is a fraction, 0 to 1; got {merged[name]}")
    if merged["stall_window"] < 0:
        raise bad(f"ptrace_stall_window must be >= 0 [s], got {merged['stall_window']}")
    if merged["pdf_n_sub"] < 1:
        raise bad(f"ptrace_pdf_n_sub must be >= 1, got {merged['pdf_n_sub']}")

    by_position = merged["initialiser"] == "markers"
    if by_position:
        for name in ("R0", "Z0"):
            if name not in merged:
                raise bad(f"ptrace_{name} is not set: the 'markers' initialiser "
                          "places each marker at its R0, Z0, phi0")
    else:
        given = [f"ptrace_{name}" for name in _POSITIONS if name in settings]
        if given:
            raise bad(f"{', '.join(given)} set, but initialiser "
                      f"{merged['initialiser']!r} places the markers itself and "
                      "would ignore it; remove it, or use initialiser 'markers'")
    for name in _PER_MARKER:
        if name not in merged:
            continue
        values = list(merged[name])
        if by_position:
            if len(values) == 1:
                values = values * n
            elif len(values) != n:
                raise bad(f"ptrace_{name} has {len(values)} values for ptrace_n_markers = "
                          f"{n}; give one value for all markers, or one each")
        elif name in _POSITIONS:
            del merged[name]
            continue
        elif len(values) != 1:
            raise bad(f"ptrace_{name} has {len(values)} values, but initialiser "
                      f"{merged['initialiser']!r} gives every marker the same one")
        merged[name] = values
    if not all(e > 0 for e in merged["E_kin_eV"]):
        raise bad(f"ptrace_E_kin_eV must be > 0 [eV], got {_brief(merged['E_kin_eV'])}")
    if not all(abs(c) <= 1 for c in merged["cos_pitch"]):
        raise bad(f"ptrace_cos_pitch must be within -1..1, got {_brief(merged['cos_pitch'])}")
    return {name: merged[name] for name in PTRACE_SETTINGS if name in merged}


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


#: Values per line of a long per-marker list in SETTINGS_FILE.
_PER_LINE = 6


def settings_namelist(resolved: dict, *, header: tuple[str, ...] = ()) -> str:
    """SETTINGS_FILE's text: the &ptrace namelist of resolve_settings'
    result, after `header` as comment lines. A per-marker list that is one
    value repeated is written with a repeat count (1000*1.0d7); a longer
    one is wrapped."""
    lines = [f"! {line}" for line in (
        *header,
        "Written by ashen from the case's ptrace_<name> keys in cases.toml, over its",
        "defaults: every setting this trace ran with. Rewritten each time it is staged.",
    )]
    lines += ["&ptrace", "  restart_index = 0"]
    for name, value in resolved.items():
        if not isinstance(value, list):
            lines.append(f"  {name} = {_fortran_value(value)}")
        elif len(value) > 1 and all(v == value[0] for v in value):
            lines.append(f"  {name} = {len(value)}*{_fortran_value(value[0])}")
        else:
            texts = [_fortran_value(v) for v in value]
            rows = [", ".join(texts[i:i + _PER_LINE]) for i in range(0, len(texts), _PER_LINE)]
            lines.append(f"  {name} = " + (",\n" + " " * (len(name) + 5)).join(rows))
    lines.append("/")
    return "\n".join(lines) + "\n"


def energy_text(energy_eV: float) -> str:
    """An energy in eV as a person writes it: 10 MeV, 500 keV, 2.5 MeV,
    1.2 GeV, 30 eV. For what `ptrace` prints; files and folder names keep
    plain eV."""
    for scale, unit in ((1e9, "GeV"), (1e6, "MeV"), (1e3, "keV")):
        if abs(energy_eV) >= scale:
            return f"{energy_eV / scale:.6g} {unit}"
    return f"{energy_eV:.6g} eV"


def _brief(value, text=str) -> str:
    """A setting's value for a person: a per-marker list that is one value
    repeated as "N x value", a long one cut short. text writes one value."""
    if not isinstance(value, list):
        return str(value).lower() if isinstance(value, bool) else text(value)
    if len(value) == 1:
        return text(value[0])
    if all(v == value[0] for v in value):
        return f"{len(value)} x {text(value[0])}"
    if len(value) <= 6:
        return ", ".join(text(v) for v in value)
    return ", ".join(text(v) for v in value[:3]) + f", ... ({len(value)} values)"


def describe_settings(resolved: dict, given: dict) -> list[str]:
    """One line per setting of resolve_settings' result, for `ptrace` to
    print: "name = value", marked where it is the default rather than one
    of the case's own keys (`given`)."""
    width = max(len(name) for name in resolved)
    spaced_by_step = "snapshot_step" in given
    lines = []
    for name, value in resolved.items():
        text = _brief(value, energy_text) if name == "E_kin_eV" else _brief(value)
        note = "" if name in given else "   (default)"
        # The two snapshot settings are one choice: say which one is in force.
        if name == "snapshot_step" and not spaced_by_step:
            text = "from n_snapshots" if resolved["n_snapshots"] else "0.0: only the final snapshot"
        elif name == "n_snapshots" and spaced_by_step:
            text, note = "not used", "   (snapshot_step is set)"
        elif name == "n_snapshots" and value:
            text = f"about {value}, at round times"
        elif name.startswith("stall_") and not resolved["stop_when_stalled"]:
            continue  # the rule is off: its settings say nothing about this trace
        elif name == "stall_window" and not value:
            text = "a tenth of the trace"
        lines.append(f"{name:<{width}} = {text}{note}")
    return lines


def find_diag_file(folder: Path | str) -> Path | None:
    """The particle diagnostics file in a ptrace folder, whatever the
    executable is called: the first of DIAG_FILES there, or None."""
    for name in DIAG_FILES:
        path = Path(folder) / name
        if path.is_file():
            return path
    return None
