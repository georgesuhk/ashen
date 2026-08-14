"""Declarative case definitions for `bin/analyse`.

Replaces `analysis.py:20-58`'s ~25 stacked reassignments of the same two
vars (only the last took effect; the rest were dead history indistinguishable
from live config) with named, listable `cases.toml` entries.
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
    "lc_psi_n_in", "four_vars", "modes", "mode_colors", "four_growth_rate", "four_growth_steps",
    "four_max_delta_b", "four_ylim", "four_deconfinement_step", "four_deconfinement_caption",
    "profile_surfaces", "profile_rad_range", "profile_nmaxsteps", "profile_deltaphi",
    "poincare_highlight", "poincare_point_size", "mark_rational",
    "four_quantities", "theta_target_psi", "theta_bins", "theta_psi_n_range",
    "theta_wetted_threshold",
)

#: [cases.NAME.<diag>] step-override table names -- union of both CLIs' DIAG_CHOICES.
_DIAG_NAMES = ("zerod", "poincare", "profiles", "four", "connection_length", "theta_hist")


class CasesError(RuntimeError):
    """Raised for a missing, malformed, or incomplete cases.toml."""


@dataclass(frozen=True)
class Case:
    name: str
    steps: list[int]
    #: Per-diag `steps` override (e.g. {"four": [1000,...]}) from nested
    #: [cases.NAME.<diag>] tables. Read via steps_for: default -> case -> case+diag.
    diag_steps: dict[str, list[int]] = field(default_factory=dict)
    note: str = ""
    #: Poincare requests, satisfied incrementally: widening psi_n_in or
    #: raising n_turns costs only the increment, not a rescan.
    psi_n_in: list[float] = field(default_factory=list)
    n_turns: int = 1000
    ang_sample_freq: int = 8
    #: Toroidal start angle for every field line = puncture plane. Legacy
    #: diagnostic hardcoded 0.
    phi_start: float = 0.0
    vars: list[str] = field(default_factory=list)
    coords_var: str = "R"
    #: Postproc cut command(s); bare string -> 1-elem list via load_cases.
    #: >1 entries gathers the same vars multiple ways (e.g. ["midplane
    #: outer", "average"]). coords_var="Psi_N" needs "midplane outer", not
    #: bare "midplane" (see profiles._TOR_MODE_PREFIX).
    tor_mode: list[str] = field(default_factory=lambda: ["midplane"])
    namelist: str = "in_main"
    n_points: int = 100
    #: jorek2_four's own defaults (jorek2_four.f90:44-50); unconfigured ==
    #: bare jorek2_four run.
    nstpts: int = 30
    ntht: int = 32
    nmaxsteps: int = 2500
    deltaphi: float = 0.3
    nsmallsteps: int = 3
    rad_range: list[float] = field(default_factory=lambda: [0.001, 0.999])
    #: Which gathered psi_n_in to plot for LC/LCTT. Plot-time only, None=all;
    #: can only select/reorder already-traced surfaces (_psi_from_spec),
    #: never invents new ones.
    lc_psi_n_in: list[float] | None = None
    #: Which vars `plot --diag four` draws. Plot-time only; doesn't affect
    #: what analyse gathers. Empty=every var found in cache.
    four_vars: list[str] = field(default_factory=list)
    #: [m, n] pairs (poloidal, toroidal), e.g. [3, 2] = m=3, n=2. Shared by
    #: every mode-aware diag: which modes `plot --diag four` draws, which
    #: rational surfaces `poincare_highlight` colours in Poincare plots, and
    #: which `mark_rational` marks on radial profiles. One list, one
    #: convention -- where a colour is needed (poincare_highlight,
    #: mark_rational), it's auto-assigned from plotting.colors.
    #: DISCRETE_PALETTE by each mode's sorted (n, m) position, so the same
    #: mode gets the same colour on every figure that draws it, the same way
    #: `four`'s own mode-amplitude lines already are -- override individual
    #: modes via mode_colors below.
    modes: list[list[int]] = field(default_factory=list)
    #: Optional per-mode colour override, keyed "m,n" matching a `modes`
    #: entry, e.g. {"3,2" = "red"}. Only overrides the modes listed; every
    #: other mode keeps its auto-assigned DISCRETE_PALETTE colour. Affects
    #: poincare_highlight and mark_rational only -- `four`'s own
    #: mode-amplitude lines colour from whatever's actually in the jorek2_
    #: four cache, not from this.
    mode_colors: dict[str, str] = field(default_factory=dict)
    #: Fit+mark each mode's growth rate (gamma [1/s] = d ln|amp|/dt).
    #: Plot-time, needs zeroD cache for real time. Default off.
    four_growth_rate: bool = False
    #: [start_step, end_step] inclusive fit window; None=every requested
    #: step. Use to skip noise-floor/saturation bias in the fit.
    four_growth_steps: list[int] | None = None
    #: Caption delta_b/delta_b_over_b figures with their peak value
    #: ("max dB/B = ..."). Plot-time only. Default off.
    four_max_delta_b: bool = False
    #: Per-variable y-axis bounds for `plot --diag four`, e.g.
    #: {"Psi" = [1e-6, 1e-1]}. Plot-time only. Unlisted var = auto-scale.
    four_ylim: dict[str, list[float]] = field(default_factory=dict)
    #: Step marked with a vline on four-mode figures: step-axis draws it
    #: directly, time-axis draws its real time from the zeroD cache
    #: (gathered on demand, same precedent as delta_b_over_b's Btor
    #: profile). Manual, not computed. None (default) = no line.
    four_deconfinement_step: int | None = None
    #: Adds "delta_b[/b] at deconfinement = X% of max" to those figures'
    #: caption. Default on whenever four_deconfinement_step is set
    #: (opt *out*), independent of four_max_delta_b (both set = both
    #: lines). Exact step match only vs `four`'s gathered steps (no
    #: interpolation, same rule as connection_length's psi_n matching) --
    #: mismatch is silent.
    four_deconfinement_caption: bool = True
    #: `average` tor_mode tracing knobs (midplane family uses n_points
    #: instead). Defaults = jorek2_postproc's own (jorek2_postproc.f90:44-51).
    #: Separate from the identically-named jorek2_four knobs above --
    #: different consumer, so tuning one can't silently move the other.
    #: profile_rad_range's outer bound is the main lever for keeping
    #: `average` alive late in a nonlinear run.
    profile_surfaces: int = 100
    profile_rad_range: list[float] = field(default_factory=lambda: [0.001, 0.999])
    profile_nmaxsteps: int = 2500
    profile_deltaphi: float = 0.3
    #: Mark `modes`' q=m/n rational surfaces as vertical lines on `plot
    #: --diag profiles` (needs coords_var = "Psi_N"; skipped with a message
    #: otherwise). Auto-gathers the qprofile cache (jorek2_postproc) for any
    #: requested step that lacks one, in parallel under --n-workers -- same
    #: on-demand precedent as _ensure_zero_d. Default off; `--mark_rational`
    #: turns it on for this invocation regardless of the case's own setting.
    mark_rational: bool = False
    #: Colour only field lines near `modes`' rational surfaces, dim the
    #: rest. Needs qprofile cache (auto-gathered, same as poincare implies
    #: zerod).
    poincare_highlight: bool = False
    #: Puncture marker area (scatter `s`, pts^2). Plot-time. Raise for a
    #: short or zoomed-in scan.
    poincare_point_size: float = 0.1
    #: Amplitude quantity(ies) `plot --diag four` draws. "max"=domain-wide
    #: max|amp|; "rational_surface"=value at q=m/n. Both = max solid +
    #: rational_surface dashed overlay.
    four_quantities: list[str] = field(default_factory=lambda: ["max"])
    #: `--diag theta_hist`: user-facing psi_n a line must cross to count.
    #: real_psi_edge applied once at comparison time, never to traced data
    #: (KNOWN_ISSUES.md #7).
    theta_target_psi: float = 1.05
    #: `--diag theta_hist`: histogram bins over (-pi, pi].
    theta_bins: int = 500
    #: `--diag theta_hist`: [min, max] filter on starting psi_n_in. None=all
    #: traced. Replaces the legacy positional index i_lim.
    theta_psi_n_range: list[float] | None = None
    #: `--diag wetted_fraction`: bin-count threshold for "wetted" (same
    #: scale as theta_hist's per-bin fraction output). None -> 1/theta_bins
    #: at plot time. `--theta_wetted_threshold` CLI flag outranks this.
    theta_wetted_threshold: float | None = None

    def steps_for(self, diag: str) -> list[int]:
        """`steps` unless `diag` overrides it in `diag_steps` (case+diag tier)."""
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
    """A plain list, a `{start, stop, step}` range table, or a mix, e.g.
    `[200, 400, {start=400, stop=2000, step=200}]`. Unioned + sorted, so
    overlapping values (e.g. 400 as both explicit and a range boundary)
    collapse to one instead of duplicating the step.
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

    Explicit list, or a `{start, stop, step}` / `{start, stop, n}` table ->
    inclusive-of-stop `np.linspace` range (point count computed up front,
    not `np.arange`, to avoid float drift landing short of `stop`; `n`
    mirrors legacy `analysis.py:81`'s `np.linspace(min, max, 20)`).

    A value only yields real connection-length data if a field line was
    actually traced at that (quantised) psi_n -- exact match, no
    interpolation; a miss is `nan`, shown as a black cell, not an error.
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
    """Parse `cases.toml`. [defaults] seeds every case, overridable per
    case (mirrors legacy `analysis.py:80-104`'s shared-global params with
    scattered per-run overrides)."""
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

        # [cases.NAME.<diag>] step overrides -- popped before the unknown-key
        # check (nested dicts aren't flat fields) and before Case(**merged)
        # (Case takes them as diag_steps, not fields named "four"/"poincare").
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
            # Normalise bare string -> list here, not deep in a pool worker
            # where the traceback wouldn't name the case or file.
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

        if "modes" in merged:
            for mode in merged["modes"]:
                if not (isinstance(mode, list) and len(mode) == 2):
                    raise CasesError(
                        f"{path}: case {name!r} modes entries must be "
                        f"[m, n] pairs, got {mode!r}"
                    )
            merged["modes"] = [[int(m), int(n)] for m, n in merged["modes"]]

        if "mode_colors" in merged:
            spec = merged["mode_colors"]
            if not isinstance(spec, dict):
                raise CasesError(
                    f"{path}: case {name!r} mode_colors must be a table of "
                    f"'m,n' -> colour, got {spec!r}"
                )
            configured_modes = {tuple(mode) for mode in merged.get("modes", [])}
            parsed: dict[str, str] = {}
            for key, color in spec.items():
                parts = key.split(",")
                if len(parts) != 2:
                    raise CasesError(
                        f"{path}: case {name!r} mode_colors key {key!r} must be "
                        "'m,n' matching a modes entry, e.g. '3,2'"
                    )
                try:
                    m, n = int(parts[0].strip()), int(parts[1].strip())
                except ValueError:
                    raise CasesError(
                        f"{path}: case {name!r} mode_colors key {key!r} must be "
                        "'m,n' matching a modes entry, e.g. '3,2'"
                    ) from None
                if (m, n) not in configured_modes:
                    raise CasesError(
                        f"{path}: case {name!r} mode_colors key {key!r} has no "
                        f"matching entry in modes ({merged.get('modes', [])})"
                    )
                parsed[f"{m},{n}"] = str(color)
            merged["mode_colors"] = parsed

        if merged.get("poincare_highlight") and not merged.get("modes"):
            raise CasesError(
                f"{path}: case {name!r} has poincare_highlight = true but no "
                "modes configured"
            )

        if merged.get("mark_rational") and not merged.get("modes"):
            raise CasesError(
                f"{path}: case {name!r} has mark_rational = true but no "
                "modes configured"
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
