"""Control scripts for jorek2_postproc, and parsers for JOREK's text outputs.

Ports three near-identical string builders from ``basics.py``
(``write_jorek_postproc_script:25``, ``write_postproc_get_flux_surface:79``,
``write_postproc_get_zeroD_input:100``) and three readers from ``io.py``
(``read_jorek_postproc:198``, ``parse_macroscopic_vars:140``,
``read_zeroD:7``).

The ``*_script`` functions here return the script text rather than writing a
file directly -- :func:`ashen.jorek2.run_tool` stages it into the scratch
directory itself, so building the text is a pure, testable function.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: Fortran's fixed-width E/ES format drops the 'E' when a 3-digit exponent
#: would overflow the field it budgeted for a 2-digit one -- e.g.
#: "-1.114495214678738E-107" is written as "-1.114495214678738-107". Matches
#: a trailing 2-3 digit signed exponent glued directly onto a preceding
#: digit, so a real reinserted 'E' can be re-parsed by float().
_MISSING_EXPONENT_RE = re.compile(r"(?<=[0-9])([+-]\d{2,3})$")


def _parse_fortran_float(token: str) -> float:
    """``float()``, tolerating Fortran's dropped-'E' 3-digit-exponent quirk
    (see :data:`_MISSING_EXPONENT_RE`) -- raises the original
    :class:`ValueError` if the token still doesn't parse after that fix-up,
    so a genuinely malformed value isn't silently swallowed.
    """
    try:
        return float(token)
    except ValueError:
        fixed = _MISSING_EXPONENT_RE.sub(lambda m: "E" + m.group(1), token, count=1)
        return float(fixed)

__all__ = [
    "profile_script",
    "flux_surface_script",
    "qprofile_script",
    "zero_d_script",
    "read_zeroD",
    "read_postproc_profile",
    "parse_macroscopic_vars",
]


def profile_script(
    namelist: str,
    step: int,
    coords_var: str,
    variables: str | list[str],
    n_points: int,
    *,
    units: int = 1,
    tor_mode: str = "midplane",
    surfaces: int = 100,
    rad_range: tuple[float, float] = (0.001, 0.999),
    nmaxsteps: int = 2500,
    deltaphi: float = 0.3,
    nsmallsteps: int = 3,
) -> str:
    """Ports ``basics.py:25`` ``write_jorek_postproc_script``.

    **The legacy version emitted the wrong knob for ``average``.** The
    midplane family (``midplane``/``midplane outer``/``midplane inner``,
    and ``pol_line``/``tor_line``) reads ``linepoints``
    (``exec_commands.f90:1330``), but ``average`` reads ``surfaces``
    (``exec_commands.f90:1771``) plus the field-line-tracing parameters
    ``rad_range_min``/``rad_range_max``/``nmaxsteps``/``deltaphi``/
    ``nsmallsteps`` (``:1772-1777``). ``basics.py:25`` only ever wrote
    ``set linepoints``, so an ``average`` run silently ignored ``n_points``
    and traced at whatever the built-in defaults were
    (``jorek2_postproc.f90:44-51``).

    That matters beyond tidiness: ``rad_range_max`` defaults to 0.999, i.e.
    tracing right up to the separatrix, which is where ``trace_fieldlines``
    is most likely to fail -- and its failure is a hard Fortran ``stop``
    (``mod_straight_field_line.f90:518``) that kills the whole process, not
    an error return. Pulling ``rad_range`` in to the still-nested core is
    the main lever for getting a flux average out of a nonlinear run at all.
    """
    if isinstance(variables, str):
        variables = [variables]
    expressions = [coords_var, *variables]

    if tor_mode.split()[0] == "average":
        knobs = [
            f"  set surfaces {int(surfaces)}",
            f"  set rad_range_min {float(rad_range[0])}",
            f"  set rad_range_max {float(rad_range[1])}",
            f"  set nmaxsteps {int(nmaxsteps)}",
            f"  set deltaphi {float(deltaphi)}",
            f"  set nsmallsteps {int(nsmallsteps)}",
        ]
    else:
        knobs = [f"  set linepoints {int(n_points)}"]

    lines = [
        f"namelist {namelist}",
        f"set units {units}",
        f"for step {step} do",
        "",
        "  expressions " + " ".join(expressions),
        "  mark_coords 1",
        *knobs,
        f"  {tor_mode}",
        "done",
        "",
    ]
    return "\n".join(lines)


def flux_surface_script(namelist: str, step: int, psi_n: float, *, units: int = 1) -> str:
    """Ports ``basics.py:79`` ``write_postproc_get_flux_surface``."""
    lines = [
        f"namelist {namelist}",
        f"set units {units}",
        f"for step {step} do",
        f"  fluxsurface {psi_n}",
        "done",
        "",
    ]
    return "\n".join(lines)


def qprofile_script(namelist: str, step: int | str, *, units: int = 1) -> str:
    """The ``jorek2_postproc`` ``qprofile`` command, one step at a time.

    Same shape as :func:`flux_surface_script` -- ``qprofile`` writes
    ``Psi_n``/``q`` pairs to ``postproc/qprofile_s<step>.dat``
    (``exec_commands.f90::qprofile``), single-step naming matching
    :meth:`ashen.paths.RunPaths.qprofile`.
    """
    lines = [
        f"namelist {namelist}",
        f"set units {units}",
        f"for step {step} do",
        "  qprofile",
        "done",
        "",
    ]
    return "\n".join(lines)


def zero_d_script(namelist: str, step: int | str, *, si_units: bool = True) -> str:
    """Ports ``basics.py:100`` ``write_postproc_get_zeroD_input``.

    ``step`` is passed through as-is -- the legacy caller
    (``data_jorek.py:719`` ``get_zeroDs_at_t``) passes the zero-padded string
    form, not the raw int, so this preserves that rather than reformatting it.

    ``si_units=False`` emits ``jorek-units`` instead of ``si-units``
    (``exec_commands.f90``'s explicit opposite of the default toggle) --
    every ``0D_quantities`` column, including ``Time``, comes back in
    JOREK's own code units instead of SI. See
    :mod:`ashen.diagnostics.timestep` for the caller that wants both.
    """
    lines = [
        f"namelist {namelist}",
        "si-units" if si_units else "jorek-units",
        f"for step {step} do",
        "  zeroD_quantities",
        "done",
        "",
    ]
    return "\n".join(lines)


def read_zeroD(path: Path | str) -> dict[str, float]:
    """Ports ``io.py:7`` ``read_zeroD``.

    Raises :class:`ValueError` if the file doesn't carry both a header line
    and a data row. ``jorek2_postproc`` leaves an empty or header-only file
    behind when it's interrupted, or when it produces no output for a step --
    and such a file is indistinguishable from a good one by existence alone,
    so callers that probe with ``is_file()`` need this to fail loudly and
    catchably rather than with a bare ``IndexError`` from ``lines[1]``.

    Blank lines are ignored rather than counted, so a trailing newline or a
    blank line between the header and the data doesn't look like data.
    """
    path = Path(path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(
            f"{path}: expected a header line and a data row, found "
            f"{len(lines)} non-blank line(s) -- empty or truncated zeroD cache"
        )
    keys = lines[0].split()
    values = [_parse_fortran_float(v) for v in lines[1].split()]
    return dict(zip(keys, values))


def read_postproc_profile(path: Path | str) -> tuple[list[str], dict[int, np.ndarray]]:
    """Ports ``io.py:198`` ``read_jorek_postproc``.

    Returns ``(headers, {step: array})`` -- each array's columns are ordered
    per ``headers``, so a variable is selected via ``headers.index(name)``.
    """
    results: dict[int, np.ndarray] = {}
    with Path(path).open(encoding="utf-8") as f:
        headers = next(f).lstrip("#").split()
        for line in f:
            if line.startswith("# time step"):
                step = int(line.split("#")[-1])
                block = []
                for row in f:
                    if row.startswith("#") or not row.strip():
                        break
                    block.append([float(x) for x in row.split()])
                results[step] = np.array(block)
    return headers, results


def parse_macroscopic_vars(path: Path | str) -> dict[str, dict[str, np.ndarray]]:
    """Ports ``io.py:140`` ``parse_macroscopic_vars``."""
    times: dict[str, list[float]] = {}
    data: dict[str, list[list[float]]] = {}

    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("@") or ":" not in line:
                continue

            name, raw = line.split(":", 1)
            name = name[1:].strip()
            raw = raw.strip().replace("D", "E").replace("d", "e")

            try:
                vals = [float(x) for x in raw.split()]
            except ValueError:
                continue  # header/metadata line, e.g. "@energies:"

            times.setdefault(name, []).append(vals[0])
            data.setdefault(name, []).append(vals[1:])

    return {
        k: {"t": np.asarray(times[k], dtype=float), "y": np.asarray(data[k], dtype=float)}
        for k in data
    }
