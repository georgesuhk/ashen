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
    "note", "psi_n_in", "n_turns", "ang_sample_freq", "phi_start",
    "vars", "coords_var", "tor_mode", "namelist", "n_points",
    "nstpts", "ntht", "nmaxsteps", "deltaphi", "nsmallsteps", "rad_range",
    "lc_psi_n_in", "four_vars", "four_modes", "four_growth_rate", "four_growth_steps",
    "four_max_delta_b", "four_ylim", "four_deconfinement_step", "four_deconfinement_caption",
    "profile_surfaces", "profile_rad_range", "profile_nmaxsteps", "profile_deltaphi",
    "poincare_highlight", "poincare_highlight_modes", "poincare_highlight_colors",
    "poincare_point_size",
    "four_quantities", "theta_target_psi", "theta_bins", "theta_psi_n_range",
    "theta_wetted_threshold",
)

#: Diag names recognised as [cases.NAME.<diag>] step-override tables -- the
#: union of both CLIs' DIAG_CHOICES (ashen.cli.analyse, ashen.cli.plot).
_DIAG_NAMES = ("zerod", "poincare", "profiles", "four", "connection_length", "theta_hist")


class CasesError(RuntimeError):
    """Raised for a missing, malformed, or incomplete cases.toml."""


@dataclass(frozen=True)
class Case:
    name: str
    steps: list[int]
    #: Per-diag overrides of `steps`, e.g. {"four": [1000, 1200, ...]} --
    #: populated from nested [cases.NAME.<diag>] tables in cases.toml. Read
    #: through `steps_for`, not directly: the innermost tier of the
    #: default -> case -> case+diag tree, `steps` itself being the middle.
    diag_steps: dict[str, list[int]] = field(default_factory=dict)
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
    #: Which postproc command(s) to cut the profile with. A bare string in
    #: cases.toml is normalised to a one-element list by ``load_cases``, so
    #: listing two (e.g. ``["midplane outer", "average"]``) gathers the same
    #: variables both ways for comparison. Note ``coords_var = "Psi_N"``
    #: needs ``midplane outer``, not bare ``midplane`` -- see
    #: ``ashen.diagnostics.profiles._TOR_MODE_PREFIX``.
    tor_mode: list[str] = field(default_factory=lambda: ["midplane"])
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
    #: Annotate the delta_b / delta_b_over_b figures with their peak value
    #: ("max dB/B = ..." in the lower-right corner) -- plot-time only, and
    #: only meaningful for those two derived variables. Default off, same as
    #: four_growth_rate: it's a figure-level number, useful when quoting a
    #: single stochasticity figure but noise on a plot read for its shape.
    four_max_delta_b: bool = False
    #: Per-variable y-axis bounds for `plot --diag four`, e.g.
    #: {"Psi" = [1e-6, 1e-1]} -- plot-time only, keyed on the same variable
    #: name used for four_vars/the output filename. A variable absent here
    #: keeps matplotlib's own auto-scaling.
    four_ylim: dict[str, list[float]] = field(default_factory=dict)
    #: Time step at which to draw a vertical dashed line labelled
    #: "deconfinement step"/"deconfinement time" on the four-mode figures --
    #: plot-time only, set manually (e.g. from a separate diagnostic), not
    #: computed here. Drawn on the step-axis figure at this step directly,
    #: and on the time-axis figure at this step's real time from the zeroD
    #: cache (gathered on demand if missing, same precedent as
    #: delta_b_over_b's Btor profile). None (default) draws nothing.
    four_deconfinement_step: int | None = None
    #: Add a "delta_b/delta_b_over_b at deconfinement" line to those figures'
    #: boxed caption -- opt-in like four_max_delta_b, and independent of it
    #: (both set draws both lines). Needs four_deconfinement_step, and only
    #: takes effect when that step exactly matches one of `four`'s own
    #: gathered steps (no interpolation, same rule as connection_length's
    #: psi_n matching) -- a mismatch is silent, not an error, same as a
    #: lc_psi_n_in value never actually traced.
    four_deconfinement_caption: bool = False
    #: Field-line-tracing knobs for the `average` tor_mode only; the midplane
    #: family uses n_points instead. Defaults match jorek2_postproc's own
    #: (jorek2_postproc.f90:44-51). Deliberately separate from the
    #: identically-named nstpts/nmaxsteps/deltaphi/rad_range above, which
    #: configure jorek2_four: same physical knobs, different consumer, and
    #: sharing them would make a jorek2_four tweak silently move the
    #: profiles. profile_rad_range is the main lever for keeping `average`
    #: alive on a nonlinear run -- pulling the outer bound in off the
    #: separatrix keeps tracing inside surfaces that still exist.
    profile_surfaces: int = 100
    profile_rad_range: list[float] = field(default_factory=lambda: [0.001, 0.999])
    profile_nmaxsteps: int = 2500
    profile_deltaphi: float = 0.3
    #: Draw the Poincare plot with only the field lines nearest each chosen
    #: rational surface in colour, everything else dimmed grey. Needs the
    #: qprofile cache -- `analyse --diag poincare` gathers it automatically
    #: when this is true, same as poincare implies zerod.
    poincare_highlight: bool = False
    #: [m, n] pairs, same convention as four_modes -- e.g. [3, 2] is m=3, n=2.
    poincare_highlight_modes: list[list[int]] = field(default_factory=list)
    #: Parallel to poincare_highlight_modes: poincare_highlight_colors[i] is
    #: the colour drawn for poincare_highlight_modes[i].
    poincare_highlight_colors: list[str] = field(default_factory=list)
    #: Marker area for each puncture, in points^2 (matplotlib's scatter `s`).
    #: Plot-time only. The 0.1 default suits a dense scan -- a short one, or
    #: one zoomed into a small region, usually wants this raised.
    poincare_point_size: float = 0.1
    #: Which amplitude quantity(ies) `plot --diag four` draws. "max" is the
    #: whole-domain max |amplitude| (today's only option); "rational_surface"
    #: is the value pinned to each mode's q=m/n resonant surface. Both
    #: together draws max solid with rational_surface dashed over it.
    four_quantities: list[str] = field(default_factory=lambda: ["max"])
    #: `plot --diag theta_hist`: the user-facing (plasma-fraction) psi_n a
    #: field line must cross to be counted -- ashen.diagnostics.
    #: theta_histogram.crossing_angles applies real_psi_edge to this once,
    #: at comparison time, never to the traced data (KNOWN_ISSUES.md #7).
    theta_target_psi: float = 1.05
    #: `plot --diag theta_hist`: histogram bins over (-pi, pi].
    theta_bins: int = 500
    #: `plot --diag theta_hist`: [min, max] user-facing psi_n_in bounds
    #: restricting which *starting* flux surfaces are counted -- None (default)
    #: counts every traced surface. Replaces the notebook's `i_lim` (a
    #: positional index into scan order, which silently changes meaning
    #: whenever psi_n_in is widened or reordered).
    theta_psi_n_range: list[float] | None = None
    #: `plot --diag wetted_fraction`: the theta_hist bin-count threshold a
    #: bin must exceed to count as "wetted" -- on the same scale as
    #: theta_histogram's output (a fraction-of-lines-per-bin, since bin
    #: weights sum to 1). None (default) falls back to 1/theta_bins at plot
    #: time -- the value a perfectly uniform distribution would put in every
    #: bin, so "wetted" means "above what uniform spreading would give".
    #: `--theta_wetted_threshold` on the plot command line outranks this the
    #: same way `--theta_target_psi` outranks theta_target_psi.
    theta_wetted_threshold: float | None = None

    def steps_for(self, diag: str) -> list[int]:
        """`steps`, unless `diag` has its own override in `diag_steps` --
        the case+diag tier of the default -> case -> case+diag tree."""
        return self.diag_steps.get(diag, self.steps)


def _steps_from_range_dict(spec: dict, *, case_name: str, source: Path) -> list[int]:
    missing = {"start", "stop"} - set(spec)
    if missing:
        raise CasesError(
            f"{source}: case {case_name!r} steps table missing {sorted(missing)}"
        )
    start, stop, step = spec["start"], spec["stop"], spec.get("step", 1)
    return list(range(int(start), int(stop), int(step)))


def _steps_from_spec(spec: object, *, case_name: str, source: Path) -> list[int]:
    """A plain list, a single ``{start, stop, step}`` range table, or a list
    mixing both -- e.g. ``[200, 400, {start = 400, stop = 2000, step = 200}]``
    for a dense early stretch plus a coarser range after it. Entries are
    unioned and returned sorted, so overlapping values (like the ``400``
    both an explicit entry and a range boundary might share) collapse to
    one rather than processing the same step twice.
    """
    if isinstance(spec, list):
        steps: set[int] = set()
        for item in spec:
            if isinstance(item, dict):
                steps.update(
                    _steps_from_range_dict(item, case_name=case_name, source=source)
                )
            else:
                steps.add(int(item))
        return sorted(steps)
    if isinstance(spec, dict):
        return _steps_from_range_dict(spec, case_name=case_name, source=source)
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

        # [cases.NAME.<diag>] tables, if present, override `steps` for that
        # one diag -- popped before the "steps" check below so their nested
        # dicts never reach the flat unknown-key check further down, and
        # before Case(**merged) since Case takes them as `diag_steps`, not
        # as fields named "four"/"poincare"/etc.
        diag_steps: dict[str, list[int]] = {}
        for diag in _DIAG_NAMES:
            sub = merged.pop(diag, None)
            if sub is None:
                continue
            if not isinstance(sub, dict):
                raise CasesError(
                    f"{path}: case {name!r} has a {diag!r} key that isn't a "
                    f"[cases.{name}.{diag}] table, got {sub!r}"
                )
            unknown_sub = sorted(set(sub) - {"steps"})
            if unknown_sub:
                raise CasesError(
                    f"{path}: case {name!r} [{diag}] override table has "
                    f"unknown key(s) {unknown_sub}; only 'steps' is "
                    "currently supported"
                )
            if "steps" in sub:
                diag_steps[diag] = _steps_from_spec(
                    sub["steps"], case_name=name, source=path
                )

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

        if "tor_mode" in merged:
            # A bare string stays valid in cases.toml (and is what every
            # existing config has); normalise here so downstream code only
            # ever sees a list. Validated up front rather than deep inside a
            # pool worker, where the traceback names neither the case nor the
            # file it came from.
            from ashen.diagnostics.profiles import TOR_MODES

            spec = merged["tor_mode"]
            modes = [spec] if isinstance(spec, str) else list(spec)
            unknown = [m for m in modes if m not in TOR_MODES]
            if unknown:
                raise CasesError(
                    f"{path}: case {name!r} has unknown tor_mode(s) {unknown}; "
                    f"expected any of {list(TOR_MODES)}"
                )
            merged["tor_mode"] = modes

        if "profile_rad_range" in merged:
            spec = merged["profile_rad_range"]
            if not (isinstance(spec, list) and len(spec) == 2):
                raise CasesError(
                    f"{path}: case {name!r} profile_rad_range must be "
                    f"[min, max], got {spec!r}"
                )
            lo, hi = float(spec[0]), float(spec[1])
            if not 0.0 <= lo < hi <= 1.0:
                raise CasesError(
                    f"{path}: case {name!r} profile_rad_range must satisfy "
                    f"0 <= min < max <= 1, got [{lo}, {hi}]"
                )
            merged["profile_rad_range"] = [lo, hi]

        if "four_modes" in merged:
            for mode in merged["four_modes"]:
                if not (isinstance(mode, list) and len(mode) == 2):
                    raise CasesError(
                        f"{path}: case {name!r} four_modes entries must be "
                        f"[m, n] pairs, got {mode!r}"
                    )
            merged["four_modes"] = [[int(m), int(n)] for m, n in merged["four_modes"]]

        if "poincare_highlight_modes" in merged:
            for mode in merged["poincare_highlight_modes"]:
                if not (isinstance(mode, list) and len(mode) == 2):
                    raise CasesError(
                        f"{path}: case {name!r} poincare_highlight_modes entries "
                        f"must be [m, n] pairs, got {mode!r}"
                    )
            modes = [[int(m), int(n)] for m, n in merged["poincare_highlight_modes"]]
            zero_n = [mode for mode in modes if mode[1] == 0]
            if zero_n:
                raise CasesError(
                    f"{path}: case {name!r} poincare_highlight_modes has n=0 "
                    f"entries {zero_n}; q=m/n is undefined for n=0"
                )
            merged["poincare_highlight_modes"] = modes

        if "poincare_highlight_colors" in merged or "poincare_highlight_modes" in merged:
            modes = merged.get("poincare_highlight_modes", [])
            colors = merged.get("poincare_highlight_colors", [])
            if len(colors) != len(modes):
                raise CasesError(
                    f"{path}: case {name!r} poincare_highlight_colors "
                    f"({len(colors)}) must have the same length as "
                    f"poincare_highlight_modes ({len(modes)})"
                )

        if merged.get("poincare_highlight") and not merged.get("poincare_highlight_modes"):
            raise CasesError(
                f"{path}: case {name!r} has poincare_highlight = true but no "
                "poincare_highlight_modes/poincare_highlight_colors configured"
            )

        if "four_quantities" in merged:
            spec = merged["four_quantities"]
            quantities = [spec] if isinstance(spec, str) else list(spec)
            unknown_q = [q for q in quantities if q not in ("max", "rational_surface")]
            if unknown_q:
                raise CasesError(
                    f"{path}: case {name!r} has unknown four_quantities {unknown_q}; "
                    "expected 'max' and/or 'rational_surface'"
                )
            if not quantities:
                raise CasesError(
                    f"{path}: case {name!r} four_quantities must not be empty"
                )
            merged["four_quantities"] = quantities

        if "four_ylim" in merged:
            spec = merged["four_ylim"]
            if not isinstance(spec, dict):
                raise CasesError(
                    f"{path}: case {name!r} four_ylim must be a table of "
                    f"variable -> [min, max], got {spec!r}"
                )
            ylim: dict[str, list[float]] = {}
            for var, bounds in spec.items():
                if not (isinstance(bounds, list) and len(bounds) == 2):
                    raise CasesError(
                        f"{path}: case {name!r} four_ylim[{var!r}] must be "
                        f"[min, max], got {bounds!r}"
                    )
                lo, hi = float(bounds[0]), float(bounds[1])
                if not lo < hi:
                    raise CasesError(
                        f"{path}: case {name!r} four_ylim[{var!r}] must satisfy "
                        f"min < max, got [{lo}, {hi}]"
                    )
                ylim[var] = [lo, hi]
            merged["four_ylim"] = ylim

        if "four_deconfinement_step" in merged:
            merged["four_deconfinement_step"] = int(merged["four_deconfinement_step"])

        if "theta_psi_n_range" in merged:
            spec = merged["theta_psi_n_range"]
            if not (isinstance(spec, list) and len(spec) == 2):
                raise CasesError(
                    f"{path}: case {name!r} theta_psi_n_range must be "
                    f"[min, max], got {spec!r}"
                )
            lo, hi = float(spec[0]), float(spec[1])
            if not lo < hi:
                raise CasesError(
                    f"{path}: case {name!r} theta_psi_n_range must satisfy "
                    f"min < max, got [{lo}, {hi}]"
                )
            merged["theta_psi_n_range"] = [lo, hi]

        if "theta_wetted_threshold" in merged:
            threshold = float(merged["theta_wetted_threshold"])
            if threshold <= 0:
                raise CasesError(
                    f"{path}: case {name!r} theta_wetted_threshold must be "
                    f"positive, got {threshold}"
                )
            merged["theta_wetted_threshold"] = threshold

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

        cases[name] = Case(
            name=name, steps=steps, diag_steps=diag_steps,
            **{k: v for k, v in merged.items()},
        )

    return cases
