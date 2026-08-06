"""Declarative case definitions for ``bin/analyse``.

Replaces ``Columbia/NL_kinks/analysis.py:20-58`` -- ~25 successive
reassignments of the same two variables (``restart_times, run_folders``),
where only the last line takes effect and everything above is
commented-out-by-overwrite history, indistinguishable from live config -- and
its hand-edited ``diags`` list (``analysis.py:64-70``) with a ``cases.toml``
that preserves that history as named, listable entries instead of deleting it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Case", "CasesError", "load_cases"]

#: Case fields that come from [defaults] or a case table, not computed.
_CASE_KEYS = (
    "folder", "note", "psi_n_in", "n_turns", "ang_sample_freq", "phi_start",
    "vars", "coords_var", "tor_mode", "namelist", "n_points",
    "nstpts", "ntht", "nmaxsteps", "deltaphi", "nsmallsteps", "rad_range",
    "lc_psi_n_in", "four_vars", "four_modes", "four_growth_rate", "four_growth_steps",
)


class CasesError(RuntimeError):
    """Raised for a missing, malformed, or incomplete cases.toml."""


@dataclass(frozen=True)
class Case:
    name: str
    #: Defaults to ``name`` if not given explicitly -- see ``load_cases``.
    folder: str
    steps: list[int]
    note: str = ""
    #: Poincare requests, satisfied incrementally against the cache -- widening
    #: psi_n_in or raising n_turns costs only the increment, not a rescan.
    psi_n_in: list[float] = field(default_factory=list)
    n_turns: int = 1000
    ang_sample_freq: int = 8
    #: Toroidal angle every field line starts from, and therefore the plane the
    #: punctures land on. Was hardcoded to 0 in the legacy diagnostic.
    phi_start: float = 0.0
    vars: list[str] = field(default_factory=list)
    coords_var: str = "R"
    tor_mode: str = "midplane"
    namelist: str = "in_main"
    n_points: int = 100
    #: jorek2_four's own defaults (jorek2_four.f90:44-50) -- an unconfigured
    #: case reproduces exactly what a bare jorek2_four run would do.
    nstpts: int = 30
    ntht: int = 32
    nmaxsteps: int = 2500
    deltaphi: float = 0.3
    nsmallsteps: int = 3
    rad_range: list[float] = field(default_factory=lambda: [0.001, 0.999])
    #: Which of the gathered psi_n_in to plot for LC/LCTT -- None (default)
    #: plots every one. Plot-time only: does not affect what analyse gathers,
    #: and can only select/reorder *already-traced* surfaces (see
    #: _psi_from_spec's docstring) -- it cannot invent new ones.
    lc_psi_n_in: list[float] | None = None
    #: Which variables/modes `plot --diag four` draws -- plot-time only, does
    #: not affect what analyse gathers via jorek2_four. Empty (default) means
    #: every one found in the cache.
    four_vars: list[str] = field(default_factory=list)
    #: [m, n] pairs (poloidal, toroidal) -- e.g. [3, 2] is m=3, n=2.
    four_modes: list[list[int]] = field(default_factory=list)
    #: Fit + mark each drawn mode's exponential growth rate (gamma, 1/s,
    #: from d ln|amplitude| / dt) -- plot-time only, needs the zeroD cache
    #: for real time. Default off.
    four_growth_rate: bool = False
    #: [start_step, end_step] inclusive step range for the growth-rate fit
    #: window -- None (default) fits every requested step. Lets you pick the
    #: visually-linear region of the log-amplitude curve, since points near
    #: the noise floor or past saturation bias a whole-range fit.
    four_growth_steps: list[int] | None = None


def _steps_from_spec(spec: object, *, case_name: str, source: Path) -> list[int]:
    if isinstance(spec, list):
        return [int(s) for s in spec]
    if isinstance(spec, dict):
        missing = {"start", "stop"} - set(spec)
        if missing:
            raise CasesError(
                f"{source}: case {case_name!r} steps table missing {sorted(missing)}"
            )
        start, stop, step = spec["start"], spec["stop"], spec.get("step", 1)
        return list(range(int(start), int(stop), int(step)))
    raise CasesError(
        f"{source}: case {case_name!r} steps must be a list or a "
        f"{{start, stop, step}} table, got {spec!r}"
    )


def _psi_from_spec(
    spec: object, *, case_name: str, source: Path, field_name: str
) -> list[float]:
    """Resolve a psi_n_in / lc_psi_n_in spec into explicit values.

    Either an explicit list, or a ``{start, stop, step}`` / ``{start, stop,
    n}`` table generating an evenly spaced, *inclusive-of-stop* range via
    ``np.linspace`` -- computing the point count up front (rather than
    ``np.arange``) avoids float-accumulation drift landing just short of
    ``stop``. ``n`` mirrors the legacy ``np.linspace(min, max, 20))``
    convention at ``Columbia/NL_kinks/analysis.py:81``.

    A generated or listed value only produces real connection-length data at
    plot time if a field line was actually traced at that (quantised) psi_n
    during gathering -- ``connection_lengths_for_step`` matches exactly, with
    no interpolation. A value with no match comes back ``nan``, rendered as a
    visible black cell rather than silently missing or erroring.
    """
    if isinstance(spec, list):
        return [float(p) for p in spec]
    if isinstance(spec, dict):
        missing = {"start", "stop"} - set(spec)
        if missing:
            raise CasesError(
                f"{source}: case {case_name!r} {field_name} table missing {sorted(missing)}"
            )
        start, stop = float(spec["start"]), float(spec["stop"])
        if "step" in spec and "n" in spec:
            raise CasesError(
                f"{source}: case {case_name!r} {field_name} table cannot have "
                "both 'step' and 'n'"
            )
        if "step" in spec:
            step = float(spec["step"])
            if step <= 0:
                raise CasesError(
                    f"{source}: case {case_name!r} {field_name} step must be positive"
                )
            n = round((stop - start) / step) + 1
        elif "n" in spec:
            n = int(spec["n"])
        else:
            raise CasesError(
                f"{source}: case {case_name!r} {field_name} table needs 'step' or 'n'"
            )
        if n < 1:
            raise CasesError(
                f"{source}: case {case_name!r} {field_name} stop must be >= start"
            )
        return [float(p) for p in np.linspace(start, stop, n)]
    raise CasesError(
        f"{source}: case {case_name!r} {field_name} must be a list, or a "
        f"{{start, stop, step}}/{{start, stop, n}} table, got {spec!r}"
    )


def load_cases(path: Path | str) -> dict[str, Case]:
    """Parse ``cases.toml``. Every entry under ``[defaults]`` seeds every
    case, overridable per case -- mirrors how the physics params at
    ``analysis.py:80-104`` were shared globals with occasional per-run
    overrides scattered inline.
    """
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CasesError(f"{path}: not found") from None
    except tomllib.TOMLDecodeError as exc:
        raise CasesError(f"{path}: malformed TOML -- {exc}") from exc

    defaults = data.get("defaults", {})
    raw_cases = data.get("cases", {})
    if not raw_cases:
        raise CasesError(f"{path}: no [cases.*] entries defined")

    cases: dict[str, Case] = {}
    for name, raw in raw_cases.items():
        merged = {**defaults, **raw}

        # folder defaults to the case's own name -- [cases."qa2.1_g2.3/eta1e-3_RE"]
        # needs no separate folder = "qa2.1_g2.3/eta1e-3_RE" line, since the two
        # are the overwhelmingly common case. Still overridable when a case is
        # named for something other than its run folder (e.g. a `[defaults]`-only
        # rerun under a different label).
        merged.setdefault("folder", name)
        if "steps" not in merged:
            raise CasesError(f"{path}: case {name!r} has no 'steps'")
        steps = _steps_from_spec(merged.pop("steps"), case_name=name, source=path)

        if "psi_n_in" in merged:
            merged["psi_n_in"] = _psi_from_spec(
                merged["psi_n_in"], case_name=name, source=path, field_name="psi_n_in"
            )

        if "lc_psi_n_in" in merged:
            spec = merged["lc_psi_n_in"]
            if isinstance(spec, dict) and {"min", "max"} <= set(spec):
                lo, hi = float(spec["min"]), float(spec["max"])
                base = merged.get("psi_n_in", [])
                merged["lc_psi_n_in"] = [p for p in base if lo <= p <= hi]
            else:
                merged["lc_psi_n_in"] = _psi_from_spec(
                    spec, case_name=name, source=path, field_name="lc_psi_n_in"
                )

        if "four_modes" in merged:
            for mode in merged["four_modes"]:
                if not (isinstance(mode, list) and len(mode) == 2):
                    raise CasesError(
                        f"{path}: case {name!r} four_modes entries must be "
                        f"[m, n] pairs, got {mode!r}"
                    )
            merged["four_modes"] = [[int(m), int(n)] for m, n in merged["four_modes"]]

        if "four_growth_steps" in merged:
            spec = merged["four_growth_steps"]
            if not (isinstance(spec, list) and len(spec) == 2):
                raise CasesError(
                    f"{path}: case {name!r} four_growth_steps must be "
                    f"[start_step, end_step], got {spec!r}"
                )
            start, end = int(spec[0]), int(spec[1])
            if start > end:
                raise CasesError(
                    f"{path}: case {name!r} four_growth_steps start ({start}) "
                    f"must not be greater than end ({end})"
                )
            merged["four_growth_steps"] = [start, end]

        unknown = sorted(set(merged) - set(_CASE_KEYS))
        if unknown:
            raise CasesError(f"{path}: case {name!r} has unknown key(s): {unknown}")

        cases[name] = Case(name=name, steps=steps, **{k: v for k, v in merged.items()})

    return cases
