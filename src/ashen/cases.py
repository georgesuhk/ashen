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

from ashen.padding import restart_steps

__all__ = [
    "Case", "CasesError", "FOUR_QUANTITIES", "FOUR_RADIAL_QUANTITIES", "load_cases",
]

#: Accepted `four_quantities` entries. "max"/"rational_surface" are scalar
#: time series drawn onto one amplitude-vs-time axes; "radial" is the whole
#: |amp|(psi_n) eigenfunction and gets its own figure -- see Case.four_quantities.
FOUR_QUANTITIES = ("max", "rational_surface", "radial")

#: Accepted `four_delta_b_at` values -- see Case.four_delta_b_at.
FOUR_DELTA_B_AT = ("max", "edge")

#: Accepted `four_radial_quantity` values. Duplicates
#: diagnostics.four_modes.RADIAL_QUANTITIES rather than importing it into TOML
#: parsing; a unit test pins the two equal.
FOUR_RADIAL_QUANTITIES = ("abs", "real", "phase")

#: Case fields that come from [defaults] or a case table, not computed.
_CASE_KEYS = (
    "note", "psi_n_in", "n_turns", "ang_sample_freq", "phi_start",
    "vars", "coords_var", "tor_mode", "namelist", "n_points",
    "nstpts", "ntht", "nmaxsteps", "deltaphi", "nsmallsteps", "rad_range",
    "lc_psi_n_in", "four_vars", "modes", "mode_colors", "four_growth_rate", "four_growth_steps",
    "four_max_delta_b", "four_ylim", "four_radial_log", "four_radial_quantity",
    "four_delta_b_at",
    "four_deconfinement_step", "four_deconfinement_caption",
    "profile_surfaces", "profile_rad_range", "profile_nmaxsteps", "profile_deltaphi",
    "profile_cmap", "profile_ylim", "animate",
    "poincare_highlight", "poincare_point_size", "mark_rational",
    "four_quantities", "theta_target_psi", "theta_bins", "theta_psi_n_range",
    "theta_wetted_threshold",
    "ptrace_exe", "ptrace_start_step", "ptrace_end_step", "ptrace_particles",
    "ptrace_inputs", "ptrace_n_mpi", "ptrace_omp_threads",
    "ptrace_poincare", "ptrace_poincare_psi_n", "ptrace_poincare_n_turns",
    "ptrace_original_boundary", "ptrace_exit_psi_n", "ptrace_exit_bins",
    "ptrace_particle_color", "ptrace_settings", "ptrace_wetted_bins", "ptrace_pdf_step",
    "ptrace_wetted_density_range", "ptrace_initial_psi_n_range", "ptrace_loss_bins",
    "ptrace_wetted_counts", "ptrace_wetted_count_range",
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
    #: poincare_highlight, mark_rational and `four`'s mode-amplitude lines
    #: (not the radial eigenfunctions, which are coloured by step).
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
    #: Log y-axis on the `four_quantities = ["radial"]` eigenfunction
    #: figures. Independent of the time-series figures' scale, since an
    #: eigenfunction's shape reads better linear while amplitude growth
    #: across decades needs log. Plot-time only, default off (linear).
    #: `--four-radial-log` forces it on for one invocation; `--four-linear`
    #: forces both figure families linear regardless.
    four_radial_log: bool = False
    #: What the radial eigenfunction figures draw: "abs" (|amp|, default),
    #: "real" (signed, phase-aligned to the |amp| peak per step) or "phase"
    #: (radians, relative to the peak's). Plot-time only -- nothing to
    #: regather, the four cache already holds real and imag.
    #: `--four-radial-quantity` overrides it for one invocation.
    four_radial_quantity: str = "abs"
    #: Where delta_b and delta_b_over_b are taken. "max": the largest value
    #: over the radial grid. "edge": the value at the outermost radial point,
    #: the edge of JOREK's domain -- with an extended boundary, in the vacuum
    #: outside the plasma, as a probe there would see it.
    four_delta_b_at: str = "max"
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
    #: Colourmap for `plot --diag profiles`' time/step colourbar -- any
    #: matplotlib colormap name. Defaults to "turbo" (a perceptually-improved
    #: rainbow), not "viridis" like the rest of the package's colourbars --
    #: profiles are read by eye for "which step is this line", a task a
    #: rainbow's larger hue range makes easier than viridis's narrower one,
    #: even though viridis is the more broadly "correct" choice for a
    #: continuous quantity a reader might measure off the colour. Plot-time
    #: only. `--profile-cmap` overrides it for one invocation without
    #: editing the file.
    profile_cmap: str = "turbo"
    #: Per-variable y-axis bounds for `plot --diag profiles`, e.g.
    #: {"currdens" = [0, 5e5]}. Plot-time only. Unlisted var = auto-scale.
    #: currdens's auto-drawn gradient figure is keyed "<var>_grad", e.g.
    #: {"currdens_grad" = [0, 1e7]} -- it's a separate figure/quantity from
    #: currdens itself, so it needs its own entry, same convention as
    #: four_ylim keying by the variable each bound applies to.
    profile_ylim: dict[str, list[float]] = field(default_factory=dict)
    #: Also write an animated GIF of `plot --diag profiles`' time evolution
    #: alongside the static PNG -- one frame per restart step, each panel
    #: showing that step's curve alone. Skipped (with a message) for a
    #: figure with fewer than two steps, since a one-frame "animation"
    #: isn't one. Plot-time only, default off; `--animate` turns it on for
    #: this invocation regardless of the case's own setting.
    animate: bool = False
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
    #: rational_surface dashed overlay. Those two are scalar-per-step time
    #: series; "radial" is the odd one out -- the whole |amp|(psi_n)
    #: eigenfunction, one panel per mode, drawn to its own figure rather
    #: than onto the time-series axes.
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
    #: `bin/ptrace`: the executable to run against this run's restarts,
    #: relative to the run folder (e.g. "./exe/ex7_jorek"). None = the case
    #: has no ptrace. Run unmodified; a filename that is one of JOREK's own
    #: particle programs also gets what ashen knows about it
    #: (ashen.particle_programs).
    ptrace_exe: str | None = None
    #: `bin/ptrace`: first restart step the program sees. None = the run's
    #: first restart (ashen.ptracing.traced_steps). re_gc starts at this
    #: step's own time; ex6/ex7 always start at 2.5 ms and pick from the
    #: restarts from here on.
    ptrace_start_step: int | None = None
    #: `bin/ptrace`: last restart step the program sees. None = the run's
    #: last restart. ptrace_gc traces up to this step's time (unless t_span
    #: says otherwise).
    ptrace_end_step: int | None = None
    #: `bin/ptrace`: the restart step whose current profile ptrace_gc's
    #: current_pdf_simple initialiser samples, linked in as jorek_pdf.h5.
    #: None = ptrace_start_step's. Any step of the run, inside the traced
    #: range or not.
    ptrace_pdf_step: int | None = None
    #: `bin/ptrace`: a JOREK particle file to start from, copied in as
    #: part_restart.h5 (re_gc reads it instead of sampling the current
    #: density; ex6/ex7 and ptrace_gc ignore it). Relative to the run
    #: folder, like ptrace_exe.
    ptrace_particles: str | None = None
    #: `bin/ptrace`: files copied into the trace folder under their own names
    #: before the program runs, for a program that reads some. Relative to
    #: the run folder, like ptrace_exe. (Not ptrace_gc's settings: those are
    #: the ptrace_<setting> keys.)
    ptrace_inputs: list[str] = field(default_factory=list)
    #: `bin/ptrace`: MPI ranks. re_gc samples its particle count per rank.
    ptrace_n_mpi: int = 1
    #: `bin/ptrace`: OpenMP threads per rank; 0 = site.toml's [diagnostics].
    ptrace_omp_threads: int = 0
    #: `bin/ptrace`: ptrace_gc's &ptrace settings, from the case's
    #: ptrace_<setting> keys (ptrace_dt, ptrace_initialiser, ...; see
    #: particle_programs.PTRACE_SETTINGS) -- those the case sets; the rest
    #: are PTRACE_DEFAULTS. All of them are written to ptrace_settings.nml
    #: in the trace folder, the only settings ptrace_gc reads.
    ptrace_settings: dict = field(default_factory=dict)
    #: `plot --diag particles`: draw the Poincare punctures `analyse --diag
    #: poincare` cached, under each snapshot, from the traced restart nearest
    #: it in time. Implied by either key below.
    ptrace_poincare: bool = False
    #: Which of the cached field lines to draw, as psi_n_in values (same
    #: units and list/table spec as psi_n_in). None = every cached line.
    ptrace_poincare_psi_n: list[float] | None = None
    #: Draw at most this many punctures (turns) per line. None = all cached.
    ptrace_poincare_n_turns: int | None = None
    #: `plot --diag particles`, for a run prepared with extend_bnd: draw the
    #: plasma boundary before extension (original_bnd.dat) and mark particles
    #: outside it as crosses, though they are still on the grid.
    ptrace_original_boundary: bool = False
    #: `plot --diag particle_exits`: the psi_n a particle has left the
    #: plasma past -- psi_n as the program's diagnostics file has it (JOREK's
    #: (psi - psi_axis)/(psi_limit - psi_axis)), not scaled by real_psi_edge.
    #: `--exit-psi-n` overrides it.
    ptrace_exit_psi_n: float = 1.0
    #: `plot --diag particle_exits`: bins over each of theta and phi.
    ptrace_exit_bins: int = 72
    #: `plot --diag particle_loss`: bins in starting psi_n. Each needs
    #: enough particles for its lost fraction to mean something.
    ptrace_loss_bins: int = 40
    #: `plot --diag particle_exits`, `particle_wetted` and `particle_loss`: count only the
    #: markers that *started* with psi_n in [min, max] -- psi_n as in
    #: ptrace_exit_psi_n, at the first diagnostics time. None = all of them.
    #: The figures and numbers are then written under names carrying the
    #: range, beside the all-marker ones.
    ptrace_initial_psi_n_range: list[float] | None = None
    #: `plot --diag particle_wetted`: [n_l, n_phi] bins along the wall and
    #: around the torus (a single number: both). The 2D map needs many more
    #: hits than cells to mean much; the 1D profiles far fewer.
    ptrace_wetted_bins: list[int] = field(default_factory=lambda: [36, 36])
    #: `plot --diag particle_wetted`: [min, max] of the hit-density map's
    #: colour scale, in 1/m^2 (share of all hits per m^2 of wall). None =
    #: the data's own range. Set it to compare maps between cases.
    ptrace_wetted_density_range: list[float] | None = None
    #: `plot --diag particle_wetted`: draw how many particles hit (each
    #: cell, each bin) instead of fractions, as particle_wetted_counts.png.
    #: `--wetted-counts` turns it on from the command line.
    ptrace_wetted_counts: bool = False
    #: `plot --diag particle_wetted` with counts: [min, max] of the map's
    #: colour scale in particles per cell, apart from
    #: ptrace_wetted_density_range's. None = the data's own range.
    ptrace_wetted_count_range: list[float] | None = None
    #: `plot --diag particles`: the colour of the particles, any matplotlib
    #: colour ("red", "#ff8800", "tab:orange"). Red stands out against the
    #: Poincare plot's viridis.
    ptrace_particle_color: str = "red"

    def steps_for(self, diag: str) -> list[int]:
        """`steps` unless `diag` overrides it in `diag_steps` (case+diag tier)."""
        return self.diag_steps.get(diag, self.steps)


def _steps_from_range_dict(
    spec: dict, *, case_name: str, source: Path, run_dir: Path,
) -> list[int]:
    """A `{start, stop, step}` range table. With both ends it is
    range(start, stop, step), whether or not those restarts exist (yet).

    Leave out `start` and/or `stop` and the run's own restarts
    (`jorek<step>.h5` in run_dir) fill them in: from its first restart, up
    to and including its last. The steps are then picked *from the restarts
    that exist* -- those at start, start + step, start + 2 step, ... -- so
    `{step = 400}` needs no knowledge of the run's length and `{}` is every
    restart. A folder that is missing or has no restarts gives no steps; the
    entry points then report that case as they would anyway.
    """
    where = f"{source}: case {case_name!r} steps table"
    unknown = sorted(set(spec) - {"start", "stop", "step"})
    if unknown:
        raise CasesError(f"{where} has unknown key(s) {unknown}; it takes start, stop, step")
    try:
        start, stop = (None if spec.get(k) is None else int(spec[k]) for k in ("start", "stop"))
        step = int(spec.get("step", 1))
    except (TypeError, ValueError):
        raise CasesError(f"{where}: start, stop and step must be whole numbers, got {spec!r}") from None
    if start is not None and stop is not None:
        return list(range(start, stop, step))
    if step < 1:
        raise CasesError(f"{where}: step must be >= 1 when start or stop is left to the run")
    available = restart_steps(run_dir) if run_dir.is_dir() else []
    if not available:
        return []
    if start is None:
        start = available[0]
    return [
        s for s in available
        if s >= start and (s - start) % step == 0 and (stop is None or s < stop)
    ]


def _steps_from_spec(
    spec: object, *, case_name: str, source: Path, run_dir: Path,
) -> list[int]:
    """A plain list, a `{start, stop, step}` range table, or a mix, e.g.
    `[200, 400, {start=400, stop=2000, step=200}]`. Unioned + sorted, so
    overlapping values (e.g. 400 as both explicit and a range boundary)
    collapse to one instead of duplicating the step. A table without
    `start` or `stop` takes them from run_dir's restarts
    (_steps_from_range_dict).
    """
    if isinstance(spec, list):
        steps: set[int] = set()
        for item in spec:
            if isinstance(item, dict):
                steps.update(_steps_from_range_dict(
                    item, case_name=case_name, source=source, run_dir=run_dir,
                ))
            else:
                steps.add(int(item))
        return sorted(steps)
    if isinstance(spec, dict):
        return _steps_from_range_dict(
            spec, case_name=case_name, source=source, run_dir=run_dir,
        )
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


def _ylim_table_from_spec(
    spec: object, *, case_name: str, source: Path, field_name: str
) -> dict[str, list[float]]:
    """Validate/normalise a per-variable y-axis-bounds table (four_ylim,
    profile_ylim) -- a table of variable -> [min, max], min < max.
    """
    if not isinstance(spec, dict):
        raise CasesError(
            f"{source}: case {case_name!r} {field_name} must be a table of "
            f"variable -> [min, max], got {spec!r}"
        )
    ylim: dict[str, list[float]] = {}
    for var, bounds in spec.items():
        if not (isinstance(bounds, list) and len(bounds) == 2):
            raise CasesError(
                f"{source}: case {case_name!r} {field_name}[{var!r}] must be "
                f"[min, max], got {bounds!r}"
            )
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise CasesError(
                f"{source}: case {case_name!r} {field_name}[{var!r}] must satisfy "
                f"min < max, got [{lo}, {hi}]"
            )
        ylim[var] = [lo, hi]
    return ylim


def _ptrace_setting(name: str, kind: str, value: object, where: str) -> object:
    """One ptrace_<setting> value, checked against its PTRACE_SETTINGS kind
    and normalised (ints to floats for reals, a single value to a list)."""
    from ashen.particle_programs import PTRACE_SETTING_CHOICES

    def bad(expected: str):
        return CasesError(f"{where}: ptrace_{name} must be {expected}, got {value!r}")

    def is_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def is_real(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if kind == "str":
        choices = PTRACE_SETTING_CHOICES.get(name)
        if not isinstance(value, str) or (choices and value not in choices):
            raise bad(f"one of {list(choices)}" if choices else "a string")
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise bad("true or false")
        return value
    if kind == "int":
        if not is_int(value):
            raise bad("a whole number")
        return value
    if kind == "real":
        if not is_real(value):
            raise bad("a number")
        return float(value)
    values = value if isinstance(value, list) else [value]
    check, cast = (is_int, int) if kind == "ints" else (is_real, float)
    if not values or not all(check(v) for v in values):
        raise bad("a number or a list of numbers, one per marker"
                  if kind == "reals" else "a whole number or a list of them, one per marker")
    return [cast(v) for v in values]


def _collect_ptrace_settings(merged: dict, where: str) -> None:
    """Move every ptrace_<setting> key (PTRACE_SETTINGS; any case, as a
    Fortran namelist is) into merged["ptrace_settings"], checked."""
    from ashen.particle_programs import PTRACE_SETTINGS

    if "ptrace_settings" in merged:
        raise CasesError(
            f"{where}: set ptrace_gc's settings one key each (ptrace_dt, "
            "ptrace_initialiser, ...), not as ptrace_settings"
        )
    by_lower = {name.lower(): name for name in PTRACE_SETTINGS}
    settings: dict[str, object] = {}
    for key in [k for k in merged if k.startswith("ptrace_")]:
        name = by_lower.get(key[len("ptrace_"):].lower())
        if name is None:
            continue
        if name in settings:
            raise CasesError(f"{where} sets ptrace_{name} twice (names ignore case)")
        settings[name] = _ptrace_setting(name, PTRACE_SETTINGS[name], merged.pop(key), where)
    if settings:
        from ashen.particle_programs import PtraceSettingsError, resolve_settings

        try:
            resolve_settings(settings)
        except PtraceSettingsError as exc:
            raise CasesError(f"{where}: {exc}") from None
        merged["ptrace_settings"] = settings


def _check_ptrace_fields(merged: dict, *, case_name: str, source: Path) -> None:
    """Validate and normalise a case's ptrace_* fields in place."""
    where = f"{source}: case {case_name!r}"
    exe = merged.get("ptrace_exe")
    if exe is None:
        others = sorted(k for k in merged if k.startswith("ptrace_"))
        if others:
            raise CasesError(f"{where} sets {others} but no ptrace_exe")
        return
    _collect_ptrace_settings(merged, where)
    if not isinstance(exe, str) or not exe.strip():
        raise CasesError(f"{where}: ptrace_exe must be a path, got {exe!r}")

    start = None
    if merged.get("ptrace_start_step") is not None:
        start = int(merged["ptrace_start_step"])
        if start < 0:
            raise CasesError(f"{where}: ptrace_start_step must be >= 0, got {start}")
        merged["ptrace_start_step"] = start
    if merged.get("ptrace_pdf_step") is not None:
        pdf_step = merged["ptrace_pdf_step"]
        if isinstance(pdf_step, bool) or not isinstance(pdf_step, int) or pdf_step < 0:
            raise CasesError(f"{where}: ptrace_pdf_step must be a step >= 0, got {pdf_step!r}")
    if merged.get("ptrace_end_step") is not None:
        end = int(merged["ptrace_end_step"])
        if start is not None and end < start:
            raise CasesError(
                f"{where}: ptrace_end_step ({end}) is before ptrace_start_step ({start})"
            )
        merged["ptrace_end_step"] = end

    if merged.get("ptrace_particles") is not None:
        particles = merged["ptrace_particles"]
        if not isinstance(particles, str) or not particles.strip():
            raise CasesError(f"{where}: ptrace_particles must be a path, got {particles!r}")

    if "ptrace_inputs" in merged:
        spec = merged["ptrace_inputs"]
        if isinstance(spec, str):
            spec = [spec]
        if not (isinstance(spec, list) and all(isinstance(x, str) and x.strip() for x in spec)):
            raise CasesError(f"{where}: ptrace_inputs must be a list of paths, got {spec!r}")
        names = [Path(x).name for x in spec]
        clashes = sorted({n for n in names if names.count(n) > 1})
        if clashes:
            raise CasesError(
                f"{where}: ptrace_inputs are copied in under their own names, but "
                f"{clashes} appear more than once"
            )
        if "part_restart.h5" in names:
            raise CasesError(
                f"{where}: part_restart.h5 in ptrace_inputs -- starting particles "
                "go in ptrace_particles"
            )
        from ashen.particle_programs import RETIRED_SETTINGS_FILES, SETTINGS_FILE

        settings_files = sorted(set(names) & {SETTINGS_FILE, *RETIRED_SETTINGS_FILES})
        if settings_files:
            raise CasesError(
                f"{where}: {settings_files} in ptrace_inputs -- ptrace_gc no longer reads "
                "a settings file of yours. Set each setting as a ptrace_<name> key "
                "(ptrace_dt, ptrace_n_markers, ...; in [defaults] to share them); ashen "
                f"writes them all to {SETTINGS_FILE} in the trace folder"
            )
        merged["ptrace_inputs"] = list(spec)

    n_mpi = int(merged.get("ptrace_n_mpi", 1))
    omp_threads = int(merged.get("ptrace_omp_threads", 0))
    if n_mpi < 1 or omp_threads < 0:
        raise CasesError(f"{where}: ptrace_n_mpi must be >= 1 and ptrace_omp_threads >= 0")
    if "ptrace_n_mpi" in merged:
        merged["ptrace_n_mpi"] = n_mpi
    if "ptrace_omp_threads" in merged:
        merged["ptrace_omp_threads"] = omp_threads

    if "ptrace_exit_psi_n" in merged:
        value = merged["ptrace_exit_psi_n"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise CasesError(f"{where}: ptrace_exit_psi_n must be a number > 0, got {value!r}")
        merged["ptrace_exit_psi_n"] = float(value)
    if merged.get("ptrace_initial_psi_n_range") is not None:
        value = merged["ptrace_initial_psi_n_range"]
        if not (
            isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            and 0 <= value[0] < value[1]
        ):
            raise CasesError(
                f"{where}: ptrace_initial_psi_n_range must be [min, max] in psi_n "
                f"with 0 <= min < max, got {value!r}"
            )
        merged["ptrace_initial_psi_n_range"] = [float(v) for v in value]
    if "ptrace_loss_bins" in merged:
        value = merged["ptrace_loss_bins"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CasesError(f"{where}: ptrace_loss_bins must be a whole number >= 1, got {value!r}")
    if "ptrace_exit_bins" in merged:
        value = merged["ptrace_exit_bins"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CasesError(f"{where}: ptrace_exit_bins must be a whole number >= 1, got {value!r}")

    if "ptrace_wetted_bins" in merged:
        value = merged["ptrace_wetted_bins"]
        bins = [value, value] if not isinstance(value, list) else value
        if len(bins) != 2 or not all(
            isinstance(b, int) and not isinstance(b, bool) and b >= 1 for b in bins
        ):
            raise CasesError(
                f"{where}: ptrace_wetted_bins must be a whole number >= 1 or "
                f"[n_l, n_phi], got {value!r}"
            )
        merged["ptrace_wetted_bins"] = list(bins)

    for key, units in (("ptrace_wetted_density_range", "in 1/m^2"),
                       ("ptrace_wetted_count_range", "in particles")):
        if merged.get(key) is None:
            continue
        value = merged[key]
        if not (
            isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            and 0 <= value[0] < value[1]
        ):
            raise CasesError(
                f"{where}: {key} must be [min, max] {units} "
                f"with 0 <= min < max, got {value!r}"
            )
        merged[key] = [float(v) for v in value]

    if "ptrace_particle_color" in merged:
        from matplotlib.colors import is_color_like

        value = merged["ptrace_particle_color"]
        if not (isinstance(value, str) and is_color_like(value)):
            raise CasesError(
                f"{where}: ptrace_particle_color must be a matplotlib colour "
                f'name or "#rrggbb", got {value!r}'
            )

    for key in ("ptrace_poincare", "ptrace_original_boundary", "ptrace_wetted_counts"):
        if key in merged and not isinstance(merged[key], bool):
            raise CasesError(f"{where}: {key} must be true or false, got {merged[key]!r}")
    if merged.get("ptrace_poincare_psi_n") is not None:
        merged["ptrace_poincare_psi_n"] = _psi_from_spec(
            merged["ptrace_poincare_psi_n"], case_name=case_name, source=source,
            field_name="ptrace_poincare_psi_n",
        )
        merged["ptrace_poincare"] = True
    if merged.get("ptrace_poincare_n_turns") is not None:
        n_turns = merged["ptrace_poincare_n_turns"]
        if isinstance(n_turns, bool) or not isinstance(n_turns, int) or n_turns < 1:
            raise CasesError(
                f"{where}: ptrace_poincare_n_turns must be a whole number >= 1, got {n_turns!r}"
            )
        merged["ptrace_poincare"] = True


def load_cases(path: Path | str, *, run_root: Path | str | None = None) -> dict[str, Case]:
    """Parse `cases.toml`. [defaults] seeds every case, overridable per
    case (mirrors legacy `analysis.py:80-104`'s shared-global params with
    scattered per-run overrides).

    run_root is where the run folders are -- `run_root / <case name>` --
    for a steps table that leaves its start or stop to the run's restarts.
    Default: the current directory, as every entry point resolves them."""
    path = Path(path)
    run_root = Path(run_root) if run_root is not None else Path.cwd()
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
                    sub["steps"], case_name=name, source=path, run_dir=run_root / name,
                )

        if "steps" not in merged:
            raise CasesError(f"{path}: case {name!r} has no 'steps'")
        steps = _steps_from_spec(
            merged.pop("steps"), case_name=name, source=path, run_dir=run_root / name,
        )

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
            unknown_q = [q for q in quantities if q not in FOUR_QUANTITIES]
            if unknown_q:
                raise CasesError(
                    f"{path}: case {name!r} has unknown four_quantities {unknown_q}; "
                    f"expected any of {', '.join(repr(q) for q in FOUR_QUANTITIES)}"
                )
            if not quantities:
                raise CasesError(
                    f"{path}: case {name!r} four_quantities must not be empty"
                )
            merged["four_quantities"] = quantities

        if merged.get("four_delta_b_at", "max") not in FOUR_DELTA_B_AT:
            raise CasesError(
                f"{path}: case {name!r} has unknown four_delta_b_at "
                f"{merged['four_delta_b_at']!r}; expected one of "
                f"{', '.join(repr(q) for q in FOUR_DELTA_B_AT)}"
            )

        if merged.get("four_radial_quantity", "abs") not in FOUR_RADIAL_QUANTITIES:
            raise CasesError(
                f"{path}: case {name!r} has unknown four_radial_quantity "
                f"{merged['four_radial_quantity']!r}; expected one of "
                f"{', '.join(repr(q) for q in FOUR_RADIAL_QUANTITIES)}"
            )

        if "four_ylim" in merged:
            merged["four_ylim"] = _ylim_table_from_spec(
                merged["four_ylim"], case_name=name, source=path, field_name="four_ylim"
            )

        if "profile_ylim" in merged:
            merged["profile_ylim"] = _ylim_table_from_spec(
                merged["profile_ylim"], case_name=name, source=path, field_name="profile_ylim"
            )

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

        _check_ptrace_fields(merged, case_name=name, source=path)

        unknown = sorted(set(merged) - set(_CASE_KEYS))
        if unknown:
            raise CasesError(f"{path}: case {name!r} has unknown key(s): {unknown}")

        cases[name] = Case(
            name=name, steps=steps, diag_steps=diag_steps,
            **{k: v for k, v in merged.items()},
        )

    return cases
