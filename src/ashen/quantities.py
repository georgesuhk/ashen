"""Named per-run scalars, for plots whose axes are "one number per run".

Cross-run figures in cli/plot kept growing a bespoke extractor each:
_wetted_fraction_xy (cli/plot.py) and _delta_b_xy (cli/plot.py) are the
same shape -- loop the member cases, read one cache, reduce to one float,
report-and-skip whatever is missing -- with nothing shared but the loop.
A 2D scan map needs *three* such scalars at once, chosen independently, so
a third copy was not an option.

Each quantity is one name -> one unambiguous float. The over-time
reduction is part of the name (delta_b_over_b_max vs.
delta_b_over_b_at_deconfinement), not a separate `reduce` argument: a
figure axis labelled "delta_b/B" that silently means max on one run and
final-value on another is exactly the kind of drift this registry exists
to make impossible.

Nothing here runs a jorek2_* tool. A quantity that needs a cache topped up
asks its QuantityContext's injected callback to do it -- cli/plot owns
those (_ensure_zero_d, _ensure_qprofile), because topping up fans steps
out through cli._common.run_steps, and a library module must not import a
CLI (CLAUDE.md). With no callback supplied (a notebook, a unit test), a
missing cache is simply a missing value.

A missing value is None plus a one-line report, never an exception: one
run dropping out of a scan must not take the other nine points with it,
matching how every --compare renderer in cli/plot already behaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np

from ashen.cases import Case
from ashen.diagnostics.four_modes import (
    DELTA_B,
    DELTA_B_OVER_B,
    delta_b_over_b_series,
    delta_b_series,
    max_amplitude_series,
)
from ashen.diagnostics.poincare_cache import read_step
from ashen.diagnostics.qprofile import read_qprofile
from ashen.diagnostics.theta_histogram import (
    pooled_crossing_angles,
    theta_histogram,
    wetted_fraction as _frac_wetted,
)
from ashen.logfile import LogfileError, r_axis
from ashen.namelist import NamelistError, read_field
from ashen.paths import RunPaths, read_float
from ashen.postproc import read_zeroD

__all__ = [
    "EDGE_Q_CLAMP_TOL",
    "QUANTITIES",
    "Quantity",
    "QuantityContext",
    "QuantityError",
    "ZEROD_PREFIX",
    "describe_quantities",
    "is_known_quantity",
    "quantity",
    "quantity_names",
]

#: Escape hatch for a zeroD column with no named alias below, e.g.
#: "zerod:beta_n". Read at QuantityContext.eq_step, the same tier the
#: named q95/q99/li aliases use -- one documented reduction, so the name
#: stays unambiguous. Exists because zeroD's column set is decided by the
#: JOREK build, not by this package: a column nobody here has seen yet
#: should not need a code change to plot.
ZEROD_PREFIX = "zerod:"

#: How far (in psi_n) `edge_q`'s target may sit off the end of the cached
#: qprofile grid and still be answered by clamping to the nearest edge,
#: rather than treated as data the grid doesn't contain. See _edge_q.
EDGE_Q_CLAMP_TOL = 0.05


class QuantityError(RuntimeError):
    """Raised for an unknown quantity name -- a configuration mistake
    (cases.toml, or a --x-quantity typo), so it is fatal at resolve time.

    Distinct from a quantity that simply has no value for one run: that is
    None plus a report, because it is data, not configuration.
    """


@dataclass(frozen=True)
class QuantityContext:
    """Everything a quantity extractor may read, for exactly one run.

    One context type rather than a per-quantity argument list: a scan map
    chooses three quantities at runtime and cannot know which settings any
    of them will want. Settings a given quantity ignores cost nothing --
    the caller resolves the whole CLI > comparison > case > default chain
    once (cli/plot._quantity_context) and hands over the result, so the
    precedence logic stays in one place instead of once per quantity.
    """

    case: Case
    paths: RunPaths

    #: Steps this quantity may reduce over, already resolved by the caller
    #: through Quantity.steps_diag (see that field). Never empty for a
    #: time-reducing quantity; may be empty for an equilibrium one.
    steps: list[int] = field(default_factory=list)

    #: Step an equilibrium-type quantity (edge_q, q95, li, ...) reads.
    #: None -> `steps[0]`, the case's first plotted step, which for these
    #: runs is the initial equilibrium. Configurable because a restarted
    #: case's first *plotted* step is not necessarily its equilibrium.
    equilibrium_step: int | None = None

    #: psi_n the q-profile is interpolated at for `edge_q`. 1.0 = the
    #: separatrix. Deliberately separate from q95/q99, which are zeroD
    #: columns JOREK computed its own way -- see the registry.
    edge_q_psi_n: float = 1.0

    #: theta_hist family, for `wetted_fraction`. None -> the case's own
    #: field (Case.theta_*), resolved here, not by the caller.
    theta_target_psi: float | None = None
    theta_bins: int | None = None
    theta_psi_n_range: tuple[float, float] | None = None
    theta_wetted_threshold: float | None = None

    #: (m, n) -- cases.toml's `modes` convention -- for the delta_b_*_mode_*
    #: quantities only. None makes those quantities report and return None.
    delta_b_mode: tuple[int, int] | None = None

    #: Cache top-up, injected. See the module docstring. Each takes the
    #: steps it should make current; None = read-only.
    ensure_zero_d: Callable[[list[int]], None] | None = None
    ensure_qprofile: Callable[[list[int]], None] | None = None
    #: Btor at the plasma edge (delta_b_over_b_*), gathering it if needed.
    ensure_b_ref: Callable[[], float | None] | None = None

    #: Where a skip note goes. `print` matches every other --compare
    #: renderer; a test can capture it without capsys, and a notebook can
    #: silence it with `lambda _: None`.
    report: Callable[[str], None] = print

    @property
    def eq_step(self) -> int | None:
        """The step an equilibrium quantity reads: `equilibrium_step` if
        set, else the first of `steps`. None when neither is available --
        the caller reports that as a skip rather than guessing step 0,
        since a restarted run may have no step 0 on disk at all."""
        if self.equilibrium_step is not None:
            return self.equilibrium_step
        return self.steps[0] if self.steps else None


@dataclass(frozen=True)
class Quantity:
    """One named per-run scalar.

    `label` is a matplotlib mathtext axis label, same convention as
    Comparison.x_label (e.g. "$\\eta$ [$\\Omega \\cdot$ m]") -- carried
    with the quantity so a figure cannot be mislabelled by forgetting to
    set it, and overridable per comparison where a campaign wants its own
    wording.

    `log_scale` is the *default* axis scale for this quantity, not a
    mandate: eta spans decades and is unreadable linear, q and li are
    O(1). A comparison's log_x/log_y (or the CLI's) still overrides.

    `steps_diag` names the [cases.NAME.<diag>] steps override whose list
    this quantity reduces over -- "four" for the delta_b family,
    "theta_hist" for wetted_fraction, matching what _delta_b_xy and
    _wetted_fraction_xy in cli/plot.py already each hardcode. None means
    the quantity needs no step list beyond the case's own `steps` (every
    equilibrium quantity).
    """

    name: str
    label: str
    extract: Callable[[QuantityContext], float | None]
    log_scale: bool = False
    steps_diag: str | None = None
    note: str = ""


# --- the extractors ----------------------------------------------------------


def _eta(ctx: QuantityContext) -> float | None:
    """Resistivity, read from the run's own namelist.

    Not from x_values and not from the folder name: CLAUDE.md flags
    CASTOR3D's directory-name parsing as the hazard this project exists to
    avoid, and an x_values list is a hand-maintained parallel array that
    can drift from what the run actually solved. The namelist is what
    JOREK read, so it cannot drift.

    Resolved as run_dir / case.namelist, not RunPaths.in_main: in_main is
    hardcoded (paths.py) while Case.namelist is configurable, and a case
    running in_main_r must read eta out of in_main_r.
    """
    path = ctx.paths.run_dir / ctx.case.namelist
    try:
        return float(read_field(path, "eta", float))
    except (NamelistError, OSError, TypeError, ValueError) as exc:
        ctx.report(f"  {ctx.case.name}: no eta ({exc})")
        return None


def _edge_q(ctx: QuantityContext) -> float | None:
    """q interpolated to ctx.edge_q_psi_n from the equilibrium step's
    qprofile cache -- deliberately a *different* quantity from q95/q99,
    which are zeroD columns JOREK computed its own way. Registering both
    separately keeps a figure honest about which definition it plotted.

    jorek2_postproc writes the profile over the case's rad_range, whose
    default outer bound is 0.999 (cases.py) -- so the default target of
    1.0 lands just off the end of the grid on essentially every run.
    np.interp would clamp to the last point silently. Clamping within
    EDGE_Q_CLAMP_TOL is allowed and reported; beyond it the value is None,
    because "q at psi_n=1.2" from a grid ending at 0.999 is not a number
    this data contains. See KNOWN_ISSUES.md for the physics question this
    default raises.
    """
    case, paths = ctx.case, ctx.paths
    step = ctx.eq_step
    if step is None:
        ctx.report(f"  {case.name}: edge_q needs an equilibrium step, none available")
        return None
    if ctx.ensure_qprofile is not None:
        ctx.ensure_qprofile([step])
    path = paths.qprofile(step)
    if not path.is_file():
        ctx.report(
            f"  {case.name}: no qprofile cache for step {step} "
            "(run analyse --diag four)"
        )
        return None
    try:
        psi_n, q = read_qprofile(path)
    except (OSError, ValueError) as exc:
        ctx.report(f"  {case.name}: unreadable qprofile cache ({exc})")
        return None
    if psi_n.size == 0:
        ctx.report(f"  {case.name}: empty qprofile cache for step {step}")
        return None

    order = np.argsort(psi_n)
    psi_sorted, q_sorted = psi_n[order], q[order]
    target = ctx.edge_q_psi_n
    lo, hi = float(psi_sorted[0]), float(psi_sorted[-1])
    if target < lo - EDGE_Q_CLAMP_TOL or target > hi + EDGE_Q_CLAMP_TOL:
        ctx.report(
            f"  {case.name}: edge_q target psi_n={target} is outside the "
            f"qprofile grid [{lo:.3g}, {hi:.3g}] by more than "
            f"{EDGE_Q_CLAMP_TOL}, skipped"
        )
        return None

    value = float(np.interp(target, psi_sorted, q_sorted))
    if target > hi:
        ctx.report(
            f"  {case.name}: edge_q clamped to q({hi:.3g}) -- the qprofile "
            f"grid ends short of the requested psi_n={target}"
        )
    elif target < lo:
        ctx.report(
            f"  {case.name}: edge_q clamped to q({lo:.3g}) -- the qprofile "
            f"grid starts after the requested psi_n={target}"
        )
    return value


def _zerod_column(column: str) -> Callable[[QuantityContext], float | None]:
    """An extractor reading one zeroD column at ctx.eq_step.

    A factory rather than one function per column: q95, q99 and li3 differ
    only in the key, and zeroD's column set is a property of the JOREK
    build (see ZEROD_PREFIX), so the list here is a convenience layer over
    a generic reader, not the authority on what exists.

    A missing key reports the columns that ARE present. The names are not
    verified against a real zeroD file anywhere in this tree (no fixture
    has more than Time/Energy), so the first person to hit a spelling
    difference gets told the right spelling instead of an empty figure.
    """

    def _extract(ctx: QuantityContext) -> float | None:
        case, paths = ctx.case, ctx.paths
        step = ctx.eq_step
        if step is None:
            ctx.report(f"  {case.name}: {column} needs an equilibrium step, none available")
            return None
        if ctx.ensure_zero_d is not None:
            ctx.ensure_zero_d([step])
        try:
            values = read_zeroD(paths.zero_d(step))
        except (OSError, ValueError) as exc:
            ctx.report(f"  {case.name}: no usable zeroD cache for step {step} ({exc})")
            return None
        if column not in values:
            available = ", ".join(sorted(values)) or "(no columns)"
            ctx.report(
                f"  {case.name}: zeroD cache has no column {column!r}; "
                f"available: {available}"
            )
            return None
        return values[column]

    return _extract


def _t_final(ctx: QuantityContext) -> float | None:
    """Simulation time (s) at the last plotted step, from the zeroD cache.

    Useful as a scan-map axis in its own right (a run that deconfined
    early sits at a different x from one that ran to completion) and the
    cheapest possible smoke test that a case's zeroD cache is usable.
    """
    case, paths = ctx.case, ctx.paths
    if not ctx.steps:
        ctx.report(f"  {case.name}: t_final needs a step list, none available")
        return None
    step = ctx.steps[-1]
    if ctx.ensure_zero_d is not None:
        ctx.ensure_zero_d([step])
    try:
        values = read_zeroD(paths.zero_d(step))
    except (OSError, ValueError) as exc:
        ctx.report(f"  {case.name}: no usable zeroD cache for step {step} ({exc})")
        return None
    if "Time" not in values:
        ctx.report(f"  {case.name}: zeroD cache has no 'Time' column")
        return None
    return values["Time"]


def _wetted_fraction_extract(ctx: QuantityContext) -> float | None:
    """The pooled-over-steps wetted fraction: the per-case body of
    cli/plot.py's _wetted_fraction_xy, minus its loop, its x bookkeeping
    and its four _warn_if_case_value_shadowed calls.

    Those warnings stay in cli/plot: they are about *precedence* between a
    comparison and a case, which is configuration resolution, not
    extraction. By the time a context exists the winner is already
    decided -- ctx.theta_* are the resolved values.
    """
    case, paths = ctx.case, ctx.paths
    steps = ctx.steps
    if not steps:
        ctx.report(f"  {case.name}: wetted_fraction needs a step list, none available")
        return None
    try:
        real_psi_edge = read_float(paths.real_psi_edge)
    except OSError as exc:
        ctx.report(f"  {case.name}: no real_psi_edge.dat ({exc})")
        return None

    target = ctx.theta_target_psi if ctx.theta_target_psi is not None else case.theta_target_psi
    n_bins = ctx.theta_bins if ctx.theta_bins is not None else case.theta_bins
    if ctx.theta_psi_n_range is not None:
        theta_range = tuple(ctx.theta_psi_n_range)
    elif case.theta_psi_n_range is not None:
        theta_range = tuple(case.theta_psi_n_range)
    else:
        theta_range = None

    records_by_step = {step: read_step(paths, step) for step in steps}
    result = pooled_crossing_angles(
        records_by_step, steps,
        target_psi=target, real_psi_edge=real_psi_edge, psi_n_range=theta_range,
    )
    counts, _ = theta_histogram(result.angles, bins=n_bins)
    threshold = ctx.theta_wetted_threshold
    if threshold is None:
        threshold = case.theta_wetted_threshold
    if threshold is None:
        threshold = 1.0 / n_bins
    return _frac_wetted(counts, threshold=threshold)


def _peak_of_variable(series: dict, variable: str) -> float | None:
    """The largest finite value across every mode of ``variable`` in
    ``series`` -- ``None`` if there isn't one (all-nan or no matching key).

    Moved here from cli/plot.py: a pure numpy reduction over a
    four_modes series with no CLI content, and this module is now its only
    caller (via _delta_b_value) besides cli/plot's own four-mode caption,
    which imports it back from here.
    """
    arrays = [np.asarray(v, dtype=float) for (var, _, _), v in series.items() if var == variable]
    if not arrays:
        return None
    values = np.concatenate(arrays)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def _value_at_step(series: dict, variable: str, steps: list[int], step: int) -> float | None:
    """The largest finite value across every mode of ``variable`` at exactly
    ``step`` -- ``None`` if ``step`` isn't one of ``steps`` (no interpolation,
    same convention as connection_length's psi_n matching) or every mode is
    nan there. Moved here alongside :func:`_peak_of_variable` -- see its
    docstring."""
    if step not in steps:
        return None
    idx = steps.index(step)
    values = [v[idx] for (var, _, _), v in series.items() if var == variable]
    finite = [v for v in values if np.isfinite(v)]
    return float(max(finite)) if finite else None


def _delta_b_value(ctx: QuantityContext, *, variable: str, reduction: str) -> float | None:
    """The per-case body of cli/plot.py's _delta_b_xy, lifted verbatim
    minus its loop, its x bookkeeping and its Jorek2Run literal.

    `variable` is DELTA_B or DELTA_B_OVER_B; `reduction` is "max",
    "at_deconfinement" or "mode_max" -- cli/plot's --delta-b-quantity
    values ("max"/"deconfinement"/"mode") renamed to read as part of a
    quantity name, since that is now where they live.

    delta_b_over_b needs a reference field, which needs a jorek2_postproc
    call, which is why it arrives as ctx.ensure_b_ref rather than being
    gathered here (module docstring).
    """
    case, paths = ctx.case, ctx.paths
    steps = ctx.steps
    if not steps:
        ctx.report(f"  {case.name}: {variable} needs a step list, none available")
        return None

    modes_filter = (
        [(ctx.delta_b_mode[1], ctx.delta_b_mode[0])] if ctx.delta_b_mode is not None
        else ([(n, m) for m, n in case.modes] if case.modes else None)
    )
    series = max_amplitude_series(paths, steps, variables=["Psi"], modes=modes_filter)
    psi_only = {k: v for k, v in series.items() if k[0] == "Psi"}
    if not psi_only:
        ctx.report(
            f"  {case.name}: no jorek2_four Psi cache found, skipped "
            "(run analyse --diag four)"
        )
        return None

    try:
        r0 = r_axis(paths.log)
    except LogfileError as exc:
        ctx.report(f"  {case.name}: skipping ({exc})")
        return None

    if variable == DELTA_B_OVER_B:
        b_ref = ctx.ensure_b_ref() if ctx.ensure_b_ref is not None else None
        if b_ref is None:
            ctx.report(
                f"  {case.name}: skipping {DELTA_B_OVER_B} (no reference Btor "
                "at the plasma edge)"
            )
            return None
        converted = delta_b_over_b_series(psi_only, r_axis=r0, b_ref=b_ref)
    else:
        converted = delta_b_series(psi_only, r_axis=r0)

    if reduction == "max":
        value = _peak_of_variable(converted, variable)
    elif reduction == "at_deconfinement":
        if case.four_deconfinement_step is None:
            ctx.report(f"  {case.name}: no four_deconfinement_step set, skipped")
            return None
        value = _value_at_step(converted, variable, steps, case.four_deconfinement_step)
    else:  # "mode_max"
        if ctx.delta_b_mode is None:
            ctx.report(f"  {case.name}: {variable} mode_max needs a (m, n) mode, none given")
            return None
        n, m = ctx.delta_b_mode[1], ctx.delta_b_mode[0]
        arr = converted.get((variable, n, m))
        finite = arr[np.isfinite(arr)] if arr is not None else np.array([])
        value = float(np.max(finite)) if finite.size else None

    if value is None:
        ctx.report(
            f"  {case.name}: no {variable} value for the requested quantity "
            f"({reduction}), skipped"
        )
    return value


#: The two rational modes this project compares most often -- cases.toml's
#: own `modes = [[3, 2], [2, 1], [1, 1]]` convention: [m, n] pairs. Fixed
#: here rather than a --delta-b-mode-style parameter, because a named
#: quantity is one unambiguous scalar and "which two modes" is exactly the
#: kind of silent-drift knob this registry exists to make impossible. Add a
#: second pair of quantities (following this one's shape) if another mode
#: ratio is ever needed -- there is no generic "ratio of any two modes"
#: quantity by design.
_MODE_32 = (3, 2)  # (m, n)
_MODE_21 = (2, 1)  # (m, n)


def _mode_energy_fraction(ctx: QuantityContext, *, reduction: str) -> float | None:
    """(3,2) mode's magnetic energy as a fraction of (2,1)'s.

    Energy is taken proportional to delta_b^2 (Tesla^2, ashen.diagnostics.
    four_modes.delta_b_series) rather than the raw |Psi_mn| amplitude:
    delta_b already carries each mode's own m/r_axis**2 scale factor, so
    "3/2's energy relative to 2/1's" means what it says even though the two
    modes don't share a scale factor. r_axis itself cancels in the ratio,
    so this needs no b_ref (unlike delta_b_over_b) and never skips for
    lacking one.

    reduction is "max" (the largest instantaneous ratio over the requested
    steps -- when is (3,2) most competitive with (2,1)) or
    "at_deconfinement" (the ratio at the case's own four_deconfinement_step)
    -- same reduction-is-part-of-the-name convention as the delta_b family.
    """
    case, paths = ctx.case, ctx.paths
    steps = ctx.steps
    if not steps:
        ctx.report(f"  {case.name}: mode energy fraction needs a step list, none available")
        return None

    m32, n32 = _MODE_32
    m21, n21 = _MODE_21
    series = max_amplitude_series(
        paths, steps, variables=["Psi"], modes=[(n32, m32), (n21, m21)],
    )
    key32, key21 = ("Psi", n32, m32), ("Psi", n21, m21)
    if key32 not in series or key21 not in series:
        ctx.report(
            f"  {case.name}: no jorek2_four cache for mode (m={m32},n={n32}) or "
            f"(m={m21},n={n21}), skipped (run analyse --diag four)"
        )
        return None

    try:
        r0 = r_axis(paths.log)
    except LogfileError as exc:
        ctx.report(f"  {case.name}: skipping ({exc})")
        return None

    db32 = delta_b_series({key32: series[key32]}, r_axis=r0)[(DELTA_B, n32, m32)]
    db21 = delta_b_series({key21: series[key21]}, r_axis=r0)[(DELTA_B, n21, m21)]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (db32**2) / (db21**2)

    if reduction == "at_deconfinement":
        if case.four_deconfinement_step is None:
            ctx.report(f"  {case.name}: no four_deconfinement_step set, skipped")
            return None
        value = _value_at_step(
            {("ratio", n32, m32): ratio}, "ratio", steps, case.four_deconfinement_step
        )
    else:  # "max"
        finite = ratio[np.isfinite(ratio)]
        value = float(np.max(finite)) if finite.size else None

    if value is None:
        ctx.report(
            f"  {case.name}: no finite (3,2)/(2,1) energy fraction ({reduction}), skipped"
        )
    return value


# --- the registry --------------------------------------------------------

QUANTITIES: dict[str, Quantity] = {}


def _register(*quantities: Quantity) -> None:
    for q in quantities:
        QUANTITIES[q.name] = q


_register(
    Quantity(
        "eta", r"$\eta$ [$\Omega \cdot$ m]", _eta, log_scale=True,
        note="resistivity, read from the run's own namelist",
    ),
    Quantity(
        "edge_q", r"$q_{\mathrm{edge}}$", _edge_q,
        note="qprofile interpolated at edge_q_psi_n (default 1.0)",
    ),
    Quantity(
        "q95", r"$q_{95}$", _zerod_column("Q95"),
        note="zeroD column, at the equilibrium step",
    ),
    Quantity(
        "q99", r"$q_{99}$", _zerod_column("Q99"),
        note="zeroD column, at the equilibrium step",
    ),
    Quantity(
        "li", r"$l_i$", _zerod_column("li3"),
        note="zeroD li3, at the equilibrium step",
    ),
    Quantity(
        "t_final", r"$t_{\mathrm{final}}$ [s]", _t_final,
        note="zeroD Time at the last plotted step",
    ),
    Quantity(
        "wetted_fraction", "Wetted fraction", _wetted_fraction_extract,
        steps_diag="theta_hist",
        note="fraction of theta_hist bins above threshold, pooled over steps",
    ),
    Quantity(
        "delta_b_max", r"max $\delta B$ [T]",
        partial(_delta_b_value, variable=DELTA_B, reduction="max"),
        log_scale=True, steps_diag="four",
        note="domain-wide peak delta-B over every mode and step",
    ),
    Quantity(
        "delta_b_over_b_max", r"max $\delta B / B$",
        partial(_delta_b_value, variable=DELTA_B_OVER_B, reduction="max"),
        log_scale=True, steps_diag="four",
        note="domain-wide peak delta-B/B over every mode and step",
    ),
    Quantity(
        "delta_b_at_deconfinement", r"$\delta B$ at deconfinement [T]",
        partial(_delta_b_value, variable=DELTA_B, reduction="at_deconfinement"),
        log_scale=True, steps_diag="four",
        note="delta-B at the case's own four_deconfinement_step",
    ),
    Quantity(
        "delta_b_over_b_at_deconfinement", r"$\delta B / B$ at deconfinement",
        partial(_delta_b_value, variable=DELTA_B_OVER_B, reduction="at_deconfinement"),
        log_scale=True, steps_diag="four",
        note="delta-B/B at the case's own four_deconfinement_step",
    ),
    Quantity(
        "delta_b_mode_max", r"max $\delta B$ [T] (one mode)",
        partial(_delta_b_value, variable=DELTA_B, reduction="mode_max"),
        log_scale=True, steps_diag="four",
        note="one (m, n) mode's own peak delta-B (needs delta_b_mode)",
    ),
    Quantity(
        "delta_b_over_b_mode_max", r"max $\delta B / B$ (one mode)",
        partial(_delta_b_value, variable=DELTA_B_OVER_B, reduction="mode_max"),
        log_scale=True, steps_diag="four",
        note="one (m, n) mode's own peak delta-B/B (needs delta_b_mode)",
    ),
    Quantity(
        "energy_32_over_21_max", r"$W_{(3,2)} / W_{(2,1)}$",
        partial(_mode_energy_fraction, reduction="max"),
        log_scale=True, steps_diag="four",
        note="peak (delta_B_(3,2)/delta_B_(2,1))^2 over every requested step",
    ),
    Quantity(
        "energy_32_over_21_at_deconfinement", r"$W_{(3,2)} / W_{(2,1)}$ at deconfinement",
        partial(_mode_energy_fraction, reduction="at_deconfinement"),
        log_scale=True, steps_diag="four",
        note="(delta_B_(3,2)/delta_B_(2,1))^2 at the case's own four_deconfinement_step",
    ),
)


def quantity(name: str) -> Quantity:
    """The named quantity, or raise QuantityError listing the known names.

    Accepts the dynamic ZEROD_PREFIX form, built on demand with a label
    derived from the column name -- so is_known_quantity and this agree,
    and cases.toml validation can use either.
    """
    if name in QUANTITIES:
        return QUANTITIES[name]
    if name.startswith(ZEROD_PREFIX):
        column = name[len(ZEROD_PREFIX):]
        if not column:
            raise QuantityError(
                f"{name!r}: {ZEROD_PREFIX} needs a column name, e.g. "
                f"{ZEROD_PREFIX}beta_n"
            )
        return Quantity(
            name=name, label=column, extract=_zerod_column(column),
            note=f"zeroD column {column!r}, at the equilibrium step",
        )
    raise QuantityError(
        f"unknown quantity {name!r}; known: {', '.join(quantity_names())} "
        f"(or {ZEROD_PREFIX}<COLUMN> for any zeroD column)"
    )


def is_known_quantity(name: str) -> bool:
    """Whether `quantity(name)` would succeed -- for cases.toml validation,
    which must fail at parse time rather than mid-figure."""
    if name in QUANTITIES:
        return True
    return name.startswith(ZEROD_PREFIX) and len(name) > len(ZEROD_PREFIX)


def quantity_names() -> tuple[str, ...]:
    """Every registered name, sorted. Excludes the ZEROD_PREFIX form, which
    is open-ended."""
    return tuple(sorted(QUANTITIES))


def describe_quantities() -> list[tuple[str, str, str]]:
    """[(name, "log"|"linear", note), ...] for `plot --list-quantities`."""
    return [
        (name, "log" if q.log_scale else "linear", q.note)
        for name, q in sorted(QUANTITIES.items())
    ]
