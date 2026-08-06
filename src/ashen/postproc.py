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

from pathlib import Path

import numpy as np

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
) -> str:
    """Ports ``basics.py:25`` ``write_jorek_postproc_script``."""
    if isinstance(variables, str):
        variables = [variables]
    expressions = [coords_var, *variables]
    lines = [
        f"namelist {namelist}",
        f"set units {units}",
        f"for step {step} do",
        "",
        "  expressions " + " ".join(expressions),
        "  mark_coords 1",
        f"  set linepoints {n_points}",
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


def zero_d_script(namelist: str, step: int | str) -> str:
    """Ports ``basics.py:100`` ``write_postproc_get_zeroD_input``.

    ``step`` is passed through as-is -- the legacy caller
    (``data_jorek.py:719`` ``get_zeroDs_at_t``) passes the zero-padded string
    form, not the raw int, so this preserves that rather than reformatting it.
    """
    lines = [
        f"namelist {namelist}",
        "si-units",
        f"for step {step} do",
        "  zeroD_quantities",
        "done",
        "",
    ]
    return "\n".join(lines)


def read_zeroD(path: Path | str) -> dict[str, float]:
    """Ports ``io.py:7`` ``read_zeroD``."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    keys = lines[0].split()
    values = [float(v) for v in lines[1].split()]
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
