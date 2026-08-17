"""`plot` entry point: figures from data `analyse` already gathered.

Separate command from `analyse` on purpose: gathering is slow/batch (run
once, cached); plotting is fast/iterative (re-run while tuning a figure).
Keeps re-plotting from ever touching the gathering path, and lets this CLI
carry display-only flags (--dpi, --log/--linear, --smooth) without cluttering
`analyse`.

Reads the same `cases.toml` (ashen.cases) and mostly just reads what
`analyse` already cached (ashen.diagnostics.poincare_cache,
ashen.postproc.read_zeroD). Two things are gathered here anyway, cheaply and
on demand, rather than requiring a prior `analyse` pass: `ensure_edge_
toroidal_field`'s Btor profile (delta_b_over_b only) and `_ensure_zero_d`'s
per-step zeroD cache (every true-time x-axis). Both are single-valued/cheap;
a whole diagnostic (Poincare, jorek2_four) is never auto-gathered here.

Diagnostics not ported from the legacy `data_jorek.py`/`gather_profiles.py`
(macroscopic traces, field-line diffusion, stochastic factor, the
never-implemented max_fieldline_pos): see KNOWN_ISSUES.md #4.
"""

from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np

from ashen.cases import Case, CasesError, load_cases
from ashen.comparisons import Comparison, load_comparisons
from ashen.config import SiteConfigError, load_site
from ashen.diagnostics.connection_length import connection_length_matrix
from ashen.diagnostics.four_modes import (
    DELTA_B,
    DELTA_B_OVER_B,
    delta_b_over_b_series,
    delta_b_series,
    format_growth_rates,
    growth_rate_series,
    max_amplitude_series,
    rational_surface_series,
)
from ashen.diagnostics.poincare_cache import read_step
from ashen.diagnostics.profiles import (
    ensure_edge_toroidal_field,
    expand_compound_vars,
    read_profile_series,
)
from ashen.diagnostics.qprofile import (
    find_rational_surfaces,
    rational_surface_matches,
    read_qprofile,
    run_qprofile_step,
)
from ashen.diagnostics.theta_histogram import (
    pooled_crossing_angles,
    theta_histogram,
    wetted_fraction,
)
from ashen.jorek2 import Jorek2Error, Jorek2Run, run_zero_d
from ashen.logfile import LogfileError, r_axis
from ashen.paths import RunPaths, read_float
from ashen.plotting.colors import DISCRETE_PALETTE
from ashen.plotting.connection_length import plot_connection_length_map
from ashen.plotting.four_modes import plot_mode_amplitudes
from ashen.plotting.poincare import plot_poincare_step
from ashen.plotting.profiles import animate_profile_comparison, plot_profile_comparison
from ashen.plotting.theta_histogram import plot_theta_histogram_grid
from ashen.plotting.wetted_fraction import plot_wetted_fraction_datasets, plot_wetted_fraction_vs_x
from ashen.postproc import read_zeroD

DIAG_CHOICES = (
    "poincare", "connection_length", "four", "profiles", "theta_hist", "wetted_fraction",
)

#: Diags with a registered --compare renderer -- asking for one without (e.g.
#: --compare X --diag profiles) is reported, not silently a no-op.
COMPARABLE_DIAGS = ("theta_hist", "wetted_fraction")

#: Diags that only make sense across several cases (there is no single-case
#: "vs. scan parameter" plot) -- valid under --compare, reported and skipped
#: under the plain per-case loop rather than attempting something meaningless.
#: wetted_fraction has a single-case renderer instead (its evolution across
#: the case's own steps), so it is not listed here.
COMPARISON_ONLY_DIAGS = ()


def _dpi_kwargs(dpi: int | None) -> dict:
    """{"dpi": dpi} if the CLI overrode it, else {} so the plotting
    function's own default applies -- reused by every per-diag renderer."""
    return {} if dpi is None else {"dpi": dpi}


#: four_vars entries that are derived from "Psi" at plot time rather than
#: read directly from the jorek2_four cache -- see _plot_four_modes.
PSI_DERIVED = {DELTA_B, DELTA_B_OVER_B}


def _mode_colors(
    modes: list[list[int]], overrides: dict[str, str] | None = None
) -> dict[tuple[int, int], str]:
    """{(n, m): color} for case.modes, sorted by (n, m) so the same mode
    gets the same colour on every figure that draws it -- poincare_highlight
    and mark_rational both need a colour per mode; `four` doesn't (its own
    draw_mode_amplitudes assigns colour from whatever's actually in its
    cache), but this sorted-index-into-DISCRETE_PALETTE convention matches
    it, same "same mode, same colour" intent.

    overrides (case.mode_colors, "m,n" -> colour) replaces the
    auto-assigned colour for the modes it names; every other mode keeps its
    DISCRETE_PALETTE slot. cases.load_cases already validated every key
    names a mode actually in `modes` and is a parseable "m,n" pair.
    """
    ordered = sorted((n, m) for m, n in modes)
    colors = {
        (n, m): DISCRETE_PALETTE[i % len(DISCRETE_PALETTE)] for i, (n, m) in enumerate(ordered)
    }
    for key, color in (overrides or {}).items():
        m, n = (int(part) for part in key.split(","))
        colors[(n, m)] = color
    return colors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot",
        description="Draw figures from data gathered by `analyse` for cases in cases.toml.",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("cases.toml"),
        help="path to cases.toml (default: ./cases.toml)",
    )
    parser.add_argument(
        "--case", action="append", dest="selected",
        help="case name to plot (repeatable; default: every case in the file)",
    )
    parser.add_argument("--list", action="store_true", help="list defined cases and exit")
    parser.add_argument(
        "--diag", action="append", dest="diags", choices=DIAG_CHOICES,
        help="which figure(s) to draw (repeatable; default: all)",
    )
    parser.add_argument(
        "--step", type=int, action="append", dest="steps",
        help="restart step to plot (repeatable; applies to every diag drawn "
        "this run, overriding any [cases.NAME.<diag>] steps override; "
        "default: each diag's own steps, or the case's steps)",
    )
    parser.add_argument("--dpi", type=int, default=None, help="override figure DPI")
    parser.add_argument(
        "--point-size", type=float, default=None,
        help="override the poincare marker area (matplotlib scatter s, in "
        "points^2; default: the case's poincare_point_size)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="poincare steps rendered concurrently (default: site.toml's "
        "[diagnostics] n_workers)",
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="connection_length: linear colour scale instead of the default log",
    )
    parser.add_argument(
        "--smooth", action="store_true",
        help="connection_length: smooth over time, ignoring confined (inf) cells",
    )
    parser.add_argument(
        "--psi-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="connection_length: restrict the plotted psi_n axis to [MIN, MAX] "
        "(applied on top of the case's lc_psi_n_in, or psi_n_in if unset)",
    )
    parser.add_argument(
        "--four-linear", action="store_true",
        help="four: linear amplitude scale instead of the default log",
    )
    parser.add_argument(
        "--theta_target_psi", type=float, default=None,
        help="theta_hist: override case/comparison theta_target_psi",
    )
    parser.add_argument(
        "--theta_bins", type=int, default=None,
        help="theta_hist: override case/comparison theta_bins",
    )
    parser.add_argument(
        "--theta_psi_n_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="theta_hist: override case/comparison theta_psi_n_range",
    )
    parser.add_argument(
        "--n-cols", type=int, default=None,
        help="theta_hist: panels per row (default: 4 for per-case mode, or "
        "the comparison's own n_cols for --compare)",
    )
    parser.add_argument(
        "--theta_wetted_threshold", type=float, default=None,
        help="wetted_fraction: bin-count threshold a theta_hist bin must "
        "exceed to count as 'wetted', overriding every case/comparison's "
        "theta_wetted_threshold (default per case: theta_wetted_threshold, "
        "or 1/theta_bins if that's unset -- the value a uniform distribution "
        "would put in every bin)",
    )
    parser.add_argument(
        "--mark_rational", action="store_true",
        help="profiles: mark case.modes' q=m/n rational surfaces as vertical "
        "lines (needs coords_var = 'Psi_N'); auto-gathers the qprofile cache "
        "for any step missing one, in parallel under --n-workers; turns this "
        "on for every case plotted, regardless of the case's own "
        "mark_rational setting",
    )
    parser.add_argument(
        "--profile-cmap", type=str, default=None,
        help="profiles: matplotlib colormap for the time/step colourbar "
        "(default: the case's profile_cmap, or 'turbo' if unset)",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="profiles: also write an animated GIF of the time evolution "
        "alongside the static PNG, one frame per restart step (skipped, "
        "with a message, for fewer than two steps); turns this on for "
        "every case plotted, regardless of the case's own animate setting",
    )
    parser.add_argument(
        "--compare", action="append", dest="comparisons",
        help="draw a [comparisons.NAME] figure instead of per-case figures "
        "(repeatable; only diags with a comparison renderer are valid here)",
    )
    parser.add_argument(
        "--list-comparisons", action="store_true",
        help="list defined comparisons and exit",
    )
    parser.add_argument(
        "--dataset", action="append", dest="datasets_selected",
        help="wetted_fraction: which dataset(s) to draw from a datasets-style "
        "comparison (repeatable; default: every dataset)",
    )
    parser.add_argument("--site", type=Path, default=None, help="explicit site.toml")
    parser.add_argument(
        "--show-config", action="store_true",
        help="print where site.toml was found and what each key resolved to",
    )
    return parser


def _render_poincare_step(
    step: int, records, highlight: dict[float, str] | None, *, paths: RunPaths, kwargs: dict
) -> Path:
    """Module-level so it can be pickled for :class:`ProcessPoolExecutor`."""
    out = paths.figures_dir / f"{step}_poincare.png"
    step_kwargs = dict(kwargs)
    if highlight:
        step_kwargs["highlight"] = highlight
    plot_poincare_step(records, out, title=f"t={step}", **step_kwargs)
    return out


def _rational_highlight_for_step(
    case: Case, paths: RunPaths, step: int, records
) -> dict[float, str] | None:
    """{psi_n: color} for traced field lines nearest case's rational
    surfaces at this step. None if highlighting off or no qprofile cache yet.

    DO NOT rescale by real_psi_edge here. Both sides are already JOREK-grid
    psi_n: LineKey.psi_n is what's handed to fluxsurface (exec_commands.f90:
    3063-3068, psi_axis+psi_n*(psi_bnd-psi_axis)); the q-profile's first
    column is get_psi_n(...), its exact inverse. real_psi_edge (plasma-
    fraction -> JOREK-grid, boundary.py::extend_psi) was already applied
    once in cli/analyse.py when tracing case.psi_n_in; applying it again
    here double-normalises and shifts every match -- same trap as
    KNOWN_ISSUES.md #7's connection-length threshold bug.
    """
    if not case.poincare_highlight:
        return None
    q_path = paths.qprofile(step)
    if not q_path.is_file():
        print(f"  step {step}: no qprofile cache, skipping rational-surface highlight "
              "(run analyse --diag poincare with poincare_highlight, or --diag four)")
        return None

    traced_psi_n = sorted({key.psi_n for key in records})
    if not traced_psi_n:
        return None

    psi_n_q, q = read_qprofile(q_path)
    colors = _mode_colors(case.modes, case.mode_colors)
    modes = [(n, m, colors[(n, m)]) for m, n in case.modes]
    return rational_surface_matches(psi_n_q, q, modes, traced_psi_n)


def _plot_poincare(
    case: Case, paths: RunPaths, steps: list[int], *, dpi: int | None,
    n_workers: int = 1, point_size: float | None = None,
) -> None:
    kwargs = _dpi_kwargs(dpi)
    # CLI flag beats the case field, same precedence --dpi already has.
    kwargs["s"] = case.poincare_point_size if point_size is None else point_size

    to_render: list[tuple[int, dict, dict[float, str] | None]] = []
    for step in steps:
        records = read_step(paths, step)
        if not records:
            print(f"  step {step}: no Poincare cache, skipped")
            continue
        highlight = _rational_highlight_for_step(case, paths, step, records)
        to_render.append((step, records, highlight))

    if not to_render:
        return

    if n_workers <= 1 or len(to_render) <= 1:
        for step, records, highlight in to_render:
            out = _render_poincare_step(step, records, highlight, paths=paths, kwargs=kwargs)
            print(f"  step {step}: {out}")
        return

    one = partial(_render_poincare_step, paths=paths, kwargs=kwargs)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(one, step, records, highlight): step
            for step, records, highlight in to_render
        }
        outputs: dict[int, Path] = {}
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    for step, _, _ in to_render:
        print(f"  step {step}: {outputs[step]}")


def _plot_connection_length(
    case: Case,
    paths: RunPaths,
    steps: list[int],
    *,
    log: bool,
    smooth: bool,
    dpi: int | None,
    psi_range: tuple[float, float] | None = None,
    n_workers: int = 1,
) -> None:
    real_psi_edge = read_float(paths.real_psi_edge)
    try:
        R0 = r_axis(paths.log)
    except LogfileError as exc:
        print(f"  error: {exc}")
        return

    psi_n_in = case.lc_psi_n_in if case.lc_psi_n_in is not None else case.psi_n_in
    if psi_range is not None:
        lo, hi = psi_range
        psi_n_in = [p for p in psi_n_in if lo <= p <= hi]
        if not psi_n_in:
            print(f"  error: no psi_n_in within range {psi_range}")
            return

    psi_n_targets = [p * real_psi_edge for p in psi_n_in]
    records_by_step = {step: read_step(paths, step) for step in steps}
    matrix = connection_length_matrix(
        records_by_step, steps, psi_n_targets, real_psi_edge=real_psi_edge, R0=R0
    )

    true_times = _read_true_times(case, paths, steps, n_workers=n_workers)

    kwargs = _dpi_kwargs(dpi)
    for plot_true_times in (True, False):
        if plot_true_times and true_times is None:
            print("  skipping LCTT: no zeroD cache for one or more steps (run analyse --diag zerod)")
            continue
        out = plot_connection_length_map(
            matrix, steps, psi_n_in, paths.figures_dir,
            true_times=true_times, plot_true_times=plot_true_times,
            log=log, smooth=smooth, **kwargs,
        )
        print(f"  {out}")


def _zero_d_is_usable(path: Path) -> bool:
    """Whether this step's zeroD cache exists AND actually parses.

    Existence alone isn't enough: an interrupted jorek2_postproc leaves an
    empty/header-only file, blocking its own re-gathering forever (the
    file is there, so nothing regenerates it) and silently losing every
    true-time figure's x-axis. Treating an unparseable cache as absent
    makes that self-healing.
    """
    if not path.is_file():
        return False
    try:
        read_zeroD(path)
    except (OSError, ValueError):
        return False
    return True


def _ensure_zero_d(
    case: Case, paths: RunPaths, steps: list[int], *, n_workers: int = 1
) -> None:
    """Gather the zeroD cache for whichever `steps` lack a usable one --
    absent or unparseable (_zero_d_is_usable).

    Every true-time x-axis goes through _read_true_times, which used to
    just skip on "no zeroD cache", needing a separate `analyse --diag
    zerod` pass first. zeroD is cheap (one jorek2_postproc call/step, no
    tracing), so it's gathered here on demand instead, same precedent as
    ensure_edge_toroidal_field's Btor profile.

    A step whose restart doesn't exist (MissingRestartError, a
    FileNotFoundError subclass -- same tolerance analyse's own zerod
    gathering has for a run in progress) or whose jorek2_postproc call
    fails otherwise (Jorek2Error, or bare FileNotFoundError if the exe
    isn't symlinked in) is reported and skipped, not raised -- one step's
    failure only drops that step from the eventual axis, doesn't abort
    every other diag this invocation also asked for.

    n_workers > 1 fans missing steps out across processes, same shape as
    _plot_poincare's rendering pool -- each zeroD call is its own
    jorek2_postproc invocation, independent of every other step.
    """
    missing = [step for step in steps if not _zero_d_is_usable(paths.zero_d(step))]
    if not missing:
        return

    print(f"  zerod: missing or unreadable for step(s) {missing}, gathering")
    jrun = Jorek2Run(
        run_dir=paths.run_dir, exe_dir=paths.run_dir,
        namelist=paths.run_dir / case.namelist, pad_width=paths.pad_width,
    )

    if n_workers <= 1 or len(missing) <= 1:
        for step in missing:
            try:
                run_zero_d(jrun, step, paths)
            except (FileNotFoundError, Jorek2Error) as exc:
                print(f"  zerod: step {step} skipped ({exc})")
            else:
                print(f"  zerod: step {step} done")
        return

    one = partial(run_zero_d, jrun, paths=paths)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(one, step): step for step in missing}
        for future in as_completed(futures):
            step = futures[future]
            try:
                future.result()
            except (FileNotFoundError, Jorek2Error) as exc:
                print(f"  zerod: step {step} skipped ({exc})")
                continue
            print(f"  zerod: step {step} done")


def _ensure_qprofile(
    case: Case, paths: RunPaths, steps: list[int], *, n_workers: int = 1
) -> None:
    """Gather the qprofile cache (jorek2_postproc) for whichever `steps`
    lack one -- same on-demand shape as _ensure_zero_d, swapping in
    qprofile.run_qprofile_step.

    A step whose restart doesn't exist, or whose jorek2_postproc call
    otherwise fails, is reported and skipped rather than raised: one
    step missing its rational-surface line shouldn't abort the whole
    profile figure.
    """
    missing = [step for step in steps if not paths.qprofile(step).is_file()]
    if not missing:
        return

    print(f"  qprofile: missing for step(s) {missing}, gathering")
    jrun = Jorek2Run(
        run_dir=paths.run_dir, exe_dir=paths.run_dir,
        namelist=paths.run_dir / case.namelist, pad_width=paths.pad_width,
    )

    if n_workers <= 1 or len(missing) <= 1:
        for step in missing:
            try:
                run_qprofile_step(jrun, step, paths)
            except (FileNotFoundError, Jorek2Error) as exc:
                print(f"  qprofile: step {step} skipped ({exc})")
            else:
                print(f"  qprofile: step {step} done")
        return

    one = partial(run_qprofile_step, jrun, paths=paths)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(one, step): step for step in missing}
        for future in as_completed(futures):
            step = futures[future]
            try:
                future.result()
            except (FileNotFoundError, Jorek2Error) as exc:
                print(f"  qprofile: step {step} skipped ({exc})")
                continue
            print(f"  qprofile: step {step} done")


def _rational_lines_for_step(
    case: Case, paths: RunPaths, step: int
) -> list[tuple[float, str, str]] | None:
    """[(psi_n, color, label), ...] for case.modes' q=m/n crossings in this
    step's qprofile cache. None if the cache is missing (caller already
    tried _ensure_qprofile). n=0 entries are skipped (m/0 undefined) --
    `modes` is shared with `four`, where an n=0 axisymmetric component is
    valid.

    label is "n=<n>, m=<m>" (same convention as draw_mode_amplitudes'
    legend labels) -- draw_profile_family shows one legend entry per label,
    even though a reversed-shear q-profile can produce several crossings
    (and therefore several lines) for the same mode.
    """
    q_path = paths.qprofile(step)
    if not q_path.is_file():
        return None
    psi_n_q, q = read_qprofile(q_path)
    colors = _mode_colors(case.modes, case.mode_colors)
    lines: list[tuple[float, str, str]] = []
    for m, n in case.modes:
        if n == 0:
            continue
        color = colors[(n, m)]
        label = f"n={n}, m={m}"
        for crossing in find_rational_surfaces(psi_n_q, q, m / n):
            lines.append((crossing, color, label))
    return lines


def _read_true_times(
    case: Case, paths: RunPaths, steps: list[int], *, n_workers: int = 1
) -> list[float] | None:
    """Each step's true time (seconds) from the zeroD cache, gathering
    missing ones first (_ensure_zero_d, in parallel if n_workers > 1).
    Still None if any requested step's cache is missing/unreadable after
    that -- a genuinely-missing restart can't be gathered, and a partial
    time axis is worse than none.

    ValueError covers a cache that's empty/truncated or unparseable
    (postproc.read_zeroD) -- the same case _zero_d_is_usable already tried
    to re-gather; kept here so a failed re-gather degrades to skipping the
    time axis rather than aborting every other diag.
    """
    _ensure_zero_d(case, paths, steps, n_workers=n_workers)
    true_times = []
    for step in steps:
        try:
            true_times.append(read_zeroD(paths.zero_d(step))["Time"])
        except (OSError, KeyError, ValueError):
            return None
    return true_times


def _peak_of_variable(series: dict, variable: str) -> float | None:
    """The largest finite value across every mode of ``variable`` in
    ``series`` -- ``None`` if there isn't one (all-nan or no matching key).
    Used to caption a figure with its peak delta-B/delta-B-over-B, a single
    figure-level number that isn't tied to any one mode's legend entry.
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
    nan there. ``series`` values are one entry per ``steps`` index (see
    :func:`~ashen.diagnostics.four_modes.max_amplitude_series`), so ``step``
    is resolved to that index positionally.
    """
    if step not in steps:
        return None
    idx = steps.index(step)
    values = [v[idx] for (var, _, _), v in series.items() if var == variable]
    finite = [v for v in values if np.isfinite(v)]
    return float(max(finite)) if finite else None


def _plot_four_modes(
    case: Case, paths: RunPaths, steps: list[int], *, log: bool, dpi: int | None,
    n_workers: int = 1,
) -> None:
    want_max = "max" in case.four_quantities
    want_rational = "rational_surface" in case.four_quantities

    # delta_b/delta_b_over_b are derived from the "Psi" cache variable, not a
    # raw jorek2_four output -- only computed when explicitly requested
    # (never picked up by an unrestricted four_vars = [] "plot everything").
    # "Psi" is added to the cache fetch if the caller didn't already ask for
    # it, then stripped back out below so it doesn't appear as its own panel.
    requested_vars = list(case.four_vars) if case.four_vars else None
    requested_derived = set(requested_vars or []) & PSI_DERIVED
    fetch_vars = requested_vars
    if requested_derived:
        fetch_vars = [v for v in requested_vars if v not in PSI_DERIVED]
        if "Psi" not in fetch_vars:
            fetch_vars.append("Psi")

    # case.modes entries are [m, n] pairs (user-facing); the diagnostics
    # layer's ModeKey/modes filter is (n, m), matching FourRecord's own
    # (variable, n, m) field order -- swap here, at the one point they meet.
    # Computed unconditionally: it's the only place mode/variable keys are
    # discovered from the cache (cheap -- just reading what's on disk), and
    # is itself the primary series whenever "max" is selected.
    series = max_amplitude_series(
        paths, steps,
        variables=fetch_vars,
        modes=[(n, m) for m, n in case.modes] if case.modes else None,
    )
    if not series:
        print("  no jorek2_four cache found for any requested step/variable/mode "
              "(run analyse --diag four)")
        return

    # Pin each n!=0 mode's amplitude to its q=m/n rational surface, using
    # whatever q-profile cache `analyse --diag four` already gathered
    # alongside it. Silently empty (not an error) if that cache is missing --
    # e.g. plotting a case gathered before this feature existed -- so a
    # "max"-only plot is unaffected, and a rational-only one just reports
    # nothing to draw (below).
    rational = {}
    if want_rational:
        rational_modes = {(n, m) for _, n, m in series if n != 0}
        if rational_modes:
            rational = rational_surface_series(
                paths, steps, sorted(rational_modes), variables=fetch_vars
            )

    if requested_derived:
        remaining = set(requested_derived)
        try:
            r0 = r_axis(paths.log)
        except LogfileError as exc:
            print(f"  skipping {', '.join(sorted(remaining))}: {exc}")
            remaining = set()

        b_ref = None
        if DELTA_B_OVER_B in remaining:
            jrun = Jorek2Run(
                run_dir=paths.run_dir, exe_dir=paths.run_dir,
                namelist=paths.run_dir / case.namelist, pad_width=paths.pad_width,
            )
            try:
                b_ref = ensure_edge_toroidal_field(jrun, paths)
            except (FileNotFoundError, Jorek2Error) as exc:
                # FileNotFoundError also covers MissingRestartError (a
                # subclass) -- e.g. jorek2_postproc itself isn't symlinked
                # into this run folder. Caught here, not left to `main`'s
                # outer per-case handler, so this only skips
                # delta_b_over_b rather than aborting every other diag
                # this plot invocation was also asked to draw.
                print(f"  skipping {DELTA_B_OVER_B}: {exc}")
                remaining.discard(DELTA_B_OVER_B)
            else:
                if b_ref is None:
                    print(f"  skipping {DELTA_B_OVER_B}: could not gather the Btor "
                          "profile at the plasma edge for step 0 (jorek2_postproc "
                          "produced no output -- see the warning above)")
                    remaining.discard(DELTA_B_OVER_B)

        if remaining:
            psi_keys = {k for k in series if k[0] == "Psi"}
            psi_only = {k: series[k] for k in psi_keys}
            if DELTA_B in remaining:
                series.update(delta_b_series(psi_only, r_axis=r0))
            if DELTA_B_OVER_B in remaining:
                series.update(delta_b_over_b_series(psi_only, r_axis=r0, b_ref=b_ref))
            if rational:
                psi_rational_keys = {k for k in rational if k[0] == "Psi"}
                psi_rational_only = {k: rational[k] for k in psi_rational_keys}
                if DELTA_B in remaining:
                    rational.update(delta_b_series(psi_rational_only, r_axis=r0))
                if DELTA_B_OVER_B in remaining:
                    rational.update(
                        delta_b_over_b_series(psi_rational_only, r_axis=r0, b_ref=b_ref)
                    )

        # Drop the raw Psi keys unless Psi itself was explicitly requested --
        # they were only fetched as the input to the conversion above.
        if requested_vars is not None and "Psi" not in requested_vars:
            series = {k: v for k, v in series.items() if k[0] != "Psi"}
            rational = {k: v for k, v in rational.items() if k[0] != "Psi"}

    # Primary (solid) series is max whenever selected -- both-selected
    # reproduces the original solid-max/dashed-rational-overlay look;
    # rational-only makes rational itself the (solid) primary, with no
    # overlay, labelled to say what it actually is.
    primary_series = series if want_max else rational
    overlay_series = rational if (want_max and want_rational) else None
    if want_max:
        ylabel_suffix, label_suffix = None, ""
    else:
        ylabel_suffix, label_suffix = " @ rational surface", " @ rational surface"

    if not primary_series:
        print("  no rational-surface data to plot (need the qprofile cache "
              "and at least one n != 0 mode; run analyse --diag four)")
        return

    # Both x-axis variants are always written (unlike connection_length's
    # LC/LCTT, which are separate figures too, but this mirrors that split
    # rather than picking one): "step" always works, "time" only if the
    # zeroD cache covers every requested step.
    variants = [("step", list(steps), "Time step")]
    true_times = _read_true_times(case, paths, steps, n_workers=n_workers)
    if true_times is not None:
        variants.append(("time", [t * 1e6 for t in true_times], r"t [$\mu s$]"))
    else:
        print("  skipping time-axis four-mode plots: no zeroD cache for one or "
              "more steps (run analyse --diag zerod)")

    # Growth rate (gamma, 1/s) is a physical quantity fit against real time,
    # so it needs true_times regardless of which x-axis variant it ends up
    # labelled on -- fit against whichever series is primary, since gamma
    # should describe the quantity actually on the plot.
    growth_fits = {}
    if case.four_growth_rate:
        if true_times is None:
            print("  skipping growth-rate fit: no zeroD cache for one or more "
                  "steps (run analyse --diag zerod)")
        else:
            growth_fits = growth_rate_series(
                primary_series, true_times, steps,
                step_range=tuple(case.four_growth_steps) if case.four_growth_steps else None,
            )
            if growth_fits:
                growth_path = paths.four_dir / "growth_rates.txt"
                growth_path.parent.mkdir(parents=True, exist_ok=True)
                growth_path.write_text(format_growth_rates(growth_fits), encoding="utf-8")
                print(f"  {growth_path}")
            else:
                print("  no growth-rate fit possible (need >=2 positive-amplitude "
                      "points per mode in the fit window)")

    # The deconfinement step's real time, gathered on demand (same precedent
    # as delta_b_over_b's Btor profile above) -- it need not be one of the
    # requested `steps`, so it's resolved once here rather than threaded
    # through `variants`. None if unset, or if the step's zeroD cache can't
    # be gathered (e.g. its restart doesn't exist).
    deconfinement_time_us = None
    if case.four_deconfinement_step is not None:
        step_time = _read_true_times(
            case, paths, [case.four_deconfinement_step], n_workers=n_workers
        )
        if step_time is None:
            print(f"  skipping deconfinement-time marker: no zeroD cache for step "
                  f"{case.four_deconfinement_step} (run analyse --diag zerod)")
        else:
            deconfinement_time_us = step_time[0] * 1e6

    kwargs = _dpi_kwargs(dpi)
    for suffix, x, xlabel in variants:
        vline = None
        if suffix == "step" and case.four_deconfinement_step is not None:
            vline = (case.four_deconfinement_step, "deconfinement step")
        elif suffix == "time" and deconfinement_time_us is not None:
            vline = (deconfinement_time_us, "deconfinement time")
        for variable in sorted({var for var, _, _ in primary_series}):
            out = paths.four_dir / f"{variable}_modes_{suffix}.png"
            caption_lines = []
            if variable == DELTA_B_OVER_B:
                ylabel = f"\N{GREEK SMALL LETTER DELTA}B/B{ylabel_suffix or ''}"
                peak = _peak_of_variable(primary_series, variable)
                if case.four_max_delta_b and peak is not None:
                    caption_lines.append(f"max \N{GREEK SMALL LETTER DELTA}B/B = {peak:.3g}")
                if case.four_deconfinement_caption and case.four_deconfinement_step is not None:
                    at_deconf = _value_at_step(
                        primary_series, variable, steps, case.four_deconfinement_step
                    )
                    if at_deconf is not None and peak:
                        caption_lines.append(
                            f"\N{GREEK SMALL LETTER DELTA}B/B at deconfinement = "
                            f"{at_deconf / peak * 100:.3g}% of max"
                        )
            elif variable == DELTA_B:
                ylabel = f"\N{GREEK SMALL LETTER DELTA}B [T]{ylabel_suffix or ''}"
                peak = _peak_of_variable(primary_series, variable)
                if case.four_max_delta_b and peak is not None:
                    caption_lines.append(f"max \N{GREEK SMALL LETTER DELTA}B = {peak:.3g} T")
                if case.four_deconfinement_caption and case.four_deconfinement_step is not None:
                    at_deconf = _value_at_step(
                        primary_series, variable, steps, case.four_deconfinement_step
                    )
                    if at_deconf is not None and peak:
                        caption_lines.append(
                            f"\N{GREEK SMALL LETTER DELTA}B at deconfinement = "
                            f"{at_deconf / peak * 100:.3g}% of max"
                        )
            else:
                ylabel = f"|{variable}|{ylabel_suffix}" if ylabel_suffix else None
            caption = "\n".join(caption_lines) if caption_lines else None
            ylim = case.four_ylim.get(variable)
            plot_mode_amplitudes(
                x, primary_series, variable, out, rational_series=overlay_series or None,
                growth_fits=growth_fits or None, log=log, xlabel=xlabel,
                ylabel=ylabel, label_suffix=label_suffix, caption=caption,
                ylim=(ylim[0], ylim[1]) if ylim else None, vline=vline, **kwargs,
            )
            print(f"  {out}")


#: expand_compound_vars-produced var whose radial gradient is worth its own
#: figure by default -- q=2/1 tearing-mode onset shows up as a sharpening
#: peak in dcurrdens/dcoords_var well before it's obvious in currdens
#: itself, so it's drawn automatically alongside currdens rather than
#: needing a separate vars entry.
_AUTO_GRADIENT_VARS = ("currdens",)


def _gradient_series_by_mode(
    series_by_mode: Mapping[str, Mapping[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """|d(y)/d(x)| per step, via np.gradient on the same x grid already
    cached -- no new jorek2_postproc gather, just a derivative of what
    read_profile_series returned. Absolute value, same convention as the
    legacy gather_profiles.py::plot_postproc_profs Jgrad panel -- what
    matters here is the gradient's magnitude (how sharply peaked the
    current profile is), not its sign. A step with fewer than two points
    can't be differentiated and is dropped from that mode's series, not
    raised.
    """
    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for mode, series in series_by_mode.items():
        grad_series = {}
        for step, (x, y) in series.items():
            if len(x) < 2:
                continue
            grad_series[step] = (x, np.abs(np.gradient(y, x)))
        result[mode] = grad_series
    return result


def _draw_profile_variant(
    out_stem: Path,
    series_by_mode: Mapping[str, Mapping[int, tuple[np.ndarray, np.ndarray]]],
    ylabel: str,
    *,
    color_by: Mapping[int, float] | None,
    color_label: str,
    time_by_step: Mapping[int, float] | None,
    xlabel: str,
    rational_lines: list[tuple[float, str, str]] | None,
    cmap: str,
    animate: bool,
    kwargs: dict,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draws one variable's static PNG (and, if animate, its GIF) -- the
    shared tail of _plot_profiles' per-variable rendering, reused for both
    a directly-cached variable and a derived one (_gradient_series_by_mode).
    """
    out = out_stem.with_suffix(".png")
    plot_profile_comparison(
        series_by_mode, ylabel, out,
        color_by=color_by, color_label=color_label,
        xlabel=xlabel, rational_lines=rational_lines,
        cmap=cmap, ylim=ylim, **kwargs,
    )
    print(f"  {out}")

    if animate:
        gif_out = out.with_suffix(".gif")
        animated = animate_profile_comparison(
            series_by_mode, ylabel, gif_out,
            color_by=color_by, color_label=color_label, time_by_step=time_by_step,
            xlabel=xlabel, rational_lines=rational_lines,
            cmap=cmap, ylim=ylim, **kwargs,
        )
        if animated is None:
            print(f"  {ylabel!r}: fewer than two steps, skipping animation")
        else:
            print(f"  {gif_out}")


def _plot_profiles(
    case: Case, paths: RunPaths, steps: list[int], *, dpi: int | None, n_workers: int = 1,
    mark_rational: bool = False, cmap: str | None = None, animate: bool = False,
) -> None:
    """One figure per variable: a panel per ``tor_mode``, a curve per step.

    Colour carries true time where the zeroD cache allows it, falling back to
    step index -- the same all-or-nothing rule as ``_plot_four_modes``' time
    axis, since a colour scale mixing seconds and step indices would be worse
    than one honest unit. When ``animate``, each GIF frame's title always
    states both the step and the true time as text (``time_by_step``,
    animate_profile_comparison), independent of which of the two the
    colourbar itself ended up keyed on.
    """
    variables = expand_compound_vars(case.vars)
    if not variables:
        print("  no vars configured for this case; nothing to plot")
        return

    true_times = _read_true_times(case, paths, steps, n_workers=n_workers)
    if true_times is not None:
        color_by = {step: t * 1e6 for step, t in zip(steps, true_times)}
        color_label = r"t [$\mu s$]"
        # Kept separately (seconds, not the display-unit color_by) so
        # animate's per-frame step+time text always has real time available,
        # even in the "colour by step index" branch below.
        time_by_step = dict(zip(steps, true_times))
    else:
        color_by = None
        color_label = "Time step"
        time_by_step = None
        print("  no zeroD cache for one or more steps (run analyse --diag zerod); "
              "colouring profiles by step index")

    # One reference step's rational-surface positions, shared across every
    # variable's figure -- q shifts only slightly step to step, and a single
    # consistent set of lines is more legible than one line set per figure.
    rational_lines: list[tuple[float, str, str]] | None = None
    if mark_rational:
        if case.coords_var != "Psi_N":
            print("  mark_rational needs coords_var = 'Psi_N', skipping "
                  f"(this case's coords_var is {case.coords_var!r})")
        elif not case.modes:
            print("  mark_rational is on but no modes configured, skipping")
        else:
            _ensure_qprofile(case, paths, steps, n_workers=n_workers)
            rational_lines = _rational_lines_for_step(case, paths, steps[0])
            if rational_lines is None:
                print(f"  mark_rational: no qprofile cache for step {steps[0]}, skipping")

    kwargs = _dpi_kwargs(dpi)
    for var in variables:
        series_by_mode = {
            mode: read_profile_series(paths, steps, case.coords_var, var, mode)
            for mode in case.tor_mode
        }
        if not any(series_by_mode.values()):
            print(f"  no cached {var!r} profiles (run analyse --diag profiles)")
            continue

        resolved_cmap = cmap if cmap is not None else case.profile_cmap
        out_stem = paths.profile_figures_dir / f"{case.coords_var}_{var}_profile"
        var_ylim = case.profile_ylim.get(var)
        _draw_profile_variant(
            out_stem, series_by_mode, var,
            color_by=color_by, color_label=color_label, time_by_step=time_by_step,
            xlabel=case.coords_var, rational_lines=rational_lines,
            cmap=resolved_cmap, animate=animate, kwargs=kwargs,
            ylim=(var_ylim[0], var_ylim[1]) if var_ylim else None,
        )

        if var in _AUTO_GRADIENT_VARS:
            grad_series_by_mode = _gradient_series_by_mode(series_by_mode)
            if any(grad_series_by_mode.values()):
                grad_var = f"{var}_grad"
                grad_ylim = case.profile_ylim.get(grad_var)
                grad_out_stem = (
                    paths.profile_figures_dir / f"{case.coords_var}_{var}_grad_profile"
                )
                _draw_profile_variant(
                    grad_out_stem, grad_series_by_mode, f"|d({var})/d({case.coords_var})|",
                    color_by=color_by, color_label=color_label, time_by_step=time_by_step,
                    xlabel=case.coords_var, rational_lines=rational_lines,
                    cmap=resolved_cmap, animate=animate, kwargs=kwargs,
                    ylim=(grad_ylim[0], grad_ylim[1]) if grad_ylim else None,
                )


def _plot_theta_hist(
    case: Case,
    paths: RunPaths,
    steps: list[int],
    *,
    target_psi: float | None,
    bins: int | None,
    psi_range: tuple[float, float] | None,
    n_cols: int | None,
    dpi: int | None,
) -> None:
    """One panel per step, pooled into a single case's own figure."""
    real_psi_edge = read_float(paths.real_psi_edge)
    target = target_psi if target_psi is not None else case.theta_target_psi
    n_bins = bins if bins is not None else case.theta_bins
    if psi_range is not None:
        theta_range = psi_range
    elif case.theta_psi_n_range is not None:
        theta_range = tuple(case.theta_psi_n_range)
    else:
        theta_range = None

    panels = []
    for step in steps:
        records = read_step(paths, step)
        if not records:
            print(f"  step {step}: no Poincare cache, skipped")
            continue
        result = pooled_crossing_angles(
            {step: records}, [step],
            target_psi=target, real_psi_edge=real_psi_edge, psi_n_range=theta_range,
        )
        print(f"  step {step}: {result.n_crossed} of {result.n_considered} lines crossed")
        panels.append((f"t={step}", result.angles))

    if not panels:
        return

    kwargs = _dpi_kwargs(dpi)
    out = plot_theta_histogram_grid(
        panels, paths.figures_dir / "theta_hist.png",
        bins=n_bins, n_cols=n_cols or 4, **kwargs,
    )
    print(f"  {out}")


def _plot_wetted_fraction(
    case: Case,
    paths: RunPaths,
    steps: list[int],
    *,
    target_psi: float | None,
    bins: int | None,
    psi_range: tuple[float, float] | None,
    threshold: float | None,
    dpi: int | None,
) -> None:
    """A single case's own wetted fraction evolution: one point per step
    (not pooled across them), unlike the --compare renderer's one pooled
    point per case. x-axis is the step number, not log-scaled -- unlike
    plot_wetted_fraction_vs_x's default scan-parameter axis."""
    real_psi_edge = read_float(paths.real_psi_edge)
    target = target_psi if target_psi is not None else case.theta_target_psi
    n_bins = bins if bins is not None else case.theta_bins
    if psi_range is not None:
        theta_range = psi_range
    elif case.theta_psi_n_range is not None:
        theta_range = tuple(case.theta_psi_n_range)
    else:
        theta_range = None
    case_threshold = threshold if threshold is not None else case.theta_wetted_threshold
    if case_threshold is None:
        case_threshold = 1.0 / n_bins

    xs: list[float] = []
    ys: list[float] = []
    for step in steps:
        records = read_step(paths, step)
        if not records:
            print(f"  step {step}: no Poincare cache, skipped")
            continue
        result = pooled_crossing_angles(
            {step: records}, [step],
            target_psi=target, real_psi_edge=real_psi_edge, psi_n_range=theta_range,
        )
        counts, _ = theta_histogram(result.angles, bins=n_bins)
        fraction = wetted_fraction(counts, threshold=case_threshold)
        print(f"  step {step}: wetted fraction = {fraction:.3g}")
        xs.append(step)
        ys.append(fraction)

    if not xs:
        return

    kwargs = _dpi_kwargs(dpi)
    out = plot_wetted_fraction_vs_x(
        xs, ys, paths.figures_dir / "wetted_fraction.png",
        xlabel="Step", log_x=False, **kwargs,
    )
    print(f"  {out}")


def _first_not_none(*values):
    """The first non-``None`` value -- the shared shape of every "CLI flag >
    comparison setting > case setting > built-in default" precedence chain
    used under ``--compare``."""
    for v in values:
        if v is not None:
            return v
    return None


def _resolve_theta_range(
    cli_range: tuple[float, float] | None,
    comparison_range: list[float] | None,
    case_range: list[float] | None,
) -> tuple[float, float] | None:
    chosen = _first_not_none(cli_range, comparison_range, case_range)
    return tuple(chosen) if chosen is not None else None


#: Each theta_* field's declared default, read off Case itself rather than
#: hardcoded a second time here -- used only to detect "this case explicitly
#: set a non-default value that a comparison override is about to shadow".
_CASE_THETA_DEFAULTS = {
    f.name: f.default
    for f in dataclasses.fields(Case)
    if f.name in (
        "theta_target_psi", "theta_bins", "theta_psi_n_range", "theta_wetted_threshold",
    )
}


def _warn_if_case_value_shadowed(
    field_name: str, *, cli_value, comparison: Comparison, case: Case, case_name: str,
) -> None:
    """Warn when a case's own non-default setting for `field_name` is about
    to be silently overridden by `comparison`'s setting for the same field.

    Only fires when no CLI flag was given for this field: a CLI flag already
    outranks both the comparison and the case, so it overriding either is
    the documented, unsurprising behaviour -- nothing to warn about there.
    """
    if cli_value is not None:
        return
    comparison_value = getattr(comparison, field_name)
    if comparison_value is None:
        return
    case_value = getattr(case, field_name)
    if case_value == _CASE_THETA_DEFAULTS[field_name]:
        return
    print(
        f"  {case_name}: comparison {comparison.name!r} sets {field_name}="
        f"{comparison_value!r}, overriding this case's own {field_name}={case_value!r}"
    )


def _compare_theta_hist(
    comparison: Comparison,
    cases: dict[str, Case],
    *,
    target_psi: float | None,
    bins: int | None,
    psi_range: tuple[float, float] | None,
    n_cols: int | None,
    dpi: int | None,
    steps: list[int] | None,
) -> None:
    """One panel per member case, each pooling that case's own theta_hist
    steps (or `--step`, if given, applied uniformly across every member).

    `target_psi`/`bins`/`psi_range` are already the CLI's values (or None);
    each resolves through `comparison`'s own setting, then the member case's,
    before falling back to a default -- see `Comparison`'s docstring for why
    the comparison tier exists (a scan is only apples-to-apples if every
    point was computed the same way).

    `datasets`-style comparisons aren't supported here -- reported and
    skipped rather than silently drawing an empty grid.
    """
    if comparison.datasets:
        print(
            f"  comparison {comparison.name!r} uses 'datasets'; theta_hist "
            "doesn't support grouped datasets, use a flat 'cases' comparison "
            "instead, skipped"
        )
        return

    panels = []
    for label, case_name in comparison.labelled_cases():
        case = cases[case_name]
        run_dir = Path.cwd() / case.name
        if not run_dir.is_dir():
            print(f"  {case_name}: no such folder {run_dir}, skipped")
            continue
        paths = RunPaths.detect(run_dir)
        real_psi_edge = read_float(paths.real_psi_edge)

        _warn_if_case_value_shadowed(
            "theta_target_psi", cli_value=target_psi, comparison=comparison,
            case=case, case_name=case_name,
        )
        _warn_if_case_value_shadowed(
            "theta_psi_n_range", cli_value=psi_range, comparison=comparison,
            case=case, case_name=case_name,
        )
        target = _first_not_none(
            target_psi, comparison.theta_target_psi, case.theta_target_psi
        )
        theta_range = _resolve_theta_range(
            psi_range, comparison.theta_psi_n_range, case.theta_psi_n_range
        )

        case_steps = steps or case.steps_for("theta_hist")
        records_by_step = {step: read_step(paths, step) for step in case_steps}
        result = pooled_crossing_angles(
            records_by_step, case_steps,
            target_psi=target, real_psi_edge=real_psi_edge, psi_n_range=theta_range,
        )
        print(f"  {case_name}: {result.n_crossed} of {result.n_considered} lines crossed")
        panels.append((label, result.angles))

    if not panels:
        return

    n_bins = _first_not_none(bins, comparison.theta_bins) or 500
    out_dir = Path.cwd() / "figures"
    kwargs = _dpi_kwargs(dpi)
    out = plot_theta_histogram_grid(
        panels, out_dir / f"{comparison.name}_theta_hist.png",
        bins=n_bins, n_cols=n_cols or comparison.n_cols, **kwargs,
    )
    print(f"  {out}")


def _wetted_fraction_xy(
    labelled_cases: list[tuple[str, str]],
    x_by_case: dict[str, float],
    cases: dict[str, Case],
    comparison: Comparison,
    *,
    target_psi: float | None,
    bins: int | None,
    psi_range: tuple[float, float] | None,
    threshold: float | None,
    steps: list[int] | None,
) -> tuple[list[float], list[float]]:
    """One (x, wetted fraction) point per case in `labelled_cases` -- the
    per-case computation shared by both a flat comparison's single series
    and each dataset's series within a `datasets`-style comparison.

    `target_psi`/`bins`/`psi_range`/`threshold` are the CLI's values (or
    None); each resolves through `comparison`'s own setting, then the member
    case's, before falling back to a default -- see `Comparison`'s
    docstring. A scan is only a meaningful comparison if every point was
    computed with the same target/bins/threshold, which is exactly what the
    comparison tier is for.
    """
    xs: list[float] = []
    ys: list[float] = []
    for _, case_name in labelled_cases:
        case = cases[case_name]
        run_dir = Path.cwd() / case.name
        if not run_dir.is_dir():
            print(f"  {case_name}: no such folder {run_dir}, skipped")
            continue
        paths = RunPaths.detect(run_dir)
        real_psi_edge = read_float(paths.real_psi_edge)

        _warn_if_case_value_shadowed(
            "theta_target_psi", cli_value=target_psi, comparison=comparison,
            case=case, case_name=case_name,
        )
        _warn_if_case_value_shadowed(
            "theta_bins", cli_value=bins, comparison=comparison,
            case=case, case_name=case_name,
        )
        _warn_if_case_value_shadowed(
            "theta_psi_n_range", cli_value=psi_range, comparison=comparison,
            case=case, case_name=case_name,
        )
        _warn_if_case_value_shadowed(
            "theta_wetted_threshold", cli_value=threshold, comparison=comparison,
            case=case, case_name=case_name,
        )
        target = _first_not_none(
            target_psi, comparison.theta_target_psi, case.theta_target_psi
        )
        n_bins = _first_not_none(bins, comparison.theta_bins, case.theta_bins)
        theta_range = _resolve_theta_range(
            psi_range, comparison.theta_psi_n_range, case.theta_psi_n_range
        )

        case_steps = steps or case.steps_for("theta_hist")
        records_by_step = {step: read_step(paths, step) for step in case_steps}
        result = pooled_crossing_angles(
            records_by_step, case_steps,
            target_psi=target, real_psi_edge=real_psi_edge, psi_n_range=theta_range,
        )
        counts, _ = theta_histogram(result.angles, bins=n_bins)
        case_threshold = _first_not_none(
            threshold, comparison.theta_wetted_threshold, case.theta_wetted_threshold,
        )
        if case_threshold is None:
            case_threshold = 1.0 / n_bins
        fraction = wetted_fraction(counts, threshold=case_threshold)
        print(f"  {case_name}: wetted fraction = {fraction:.3g}")
        xs.append(x_by_case[case_name])
        ys.append(fraction)

    return xs, ys


def _compare_wetted_fraction(
    comparison: Comparison,
    cases: dict[str, Case],
    *,
    target_psi: float | None,
    bins: int | None,
    psi_range: tuple[float, float] | None,
    threshold: float | None,
    dpi: int | None,
    steps: list[int] | None,
    dataset_names: list[str] | None = None,
) -> None:
    """The fraction of each case's (pooled) theta_hist bins exceeding a
    threshold, plotted against a numeric x-axis -- e.g. wetted fraction vs.
    eta. Needs `x_values` configured somewhere; unlike theta_hist there is no
    meaningful figure without one, so a series missing it is skipped.

    A `datasets` comparison draws one overlaid, legend-labelled series per
    dataset (`plot_wetted_fraction_datasets`), optionally restricted to
    `dataset_names`; a flat comparison draws its one series as before
    (`plot_wetted_fraction_vs_x`).
    """
    out_dir = Path.cwd() / "figures"
    kwargs = _dpi_kwargs(dpi)

    if comparison.datasets:
        chosen = comparison.datasets
        if dataset_names is not None:
            unknown = [n for n in dataset_names if n not in comparison.datasets]
            if unknown:
                print(
                    f"  comparison {comparison.name!r} has no dataset(s) {unknown}; "
                    f"known: {list(comparison.datasets)}"
                )
                return
            chosen = {n: comparison.datasets[n] for n in dataset_names}

        series: list[tuple[str, list[float], list[float]]] = []
        colors: list[str | None] = []
        for ds_name, ds in chosen.items():
            if ds.x_values is None:
                print(
                    f"  dataset {ds_name!r} of comparison {comparison.name!r} has no "
                    "x_values (and the comparison sets none to fall back on); "
                    "wetted_fraction needs one numeric value per case, skipped"
                )
                continue
            x_by_case = dict(zip(ds.cases, ds.x_values))
            xs, ys = _wetted_fraction_xy(
                ds.labelled_cases(), x_by_case, cases, comparison,
                target_psi=target_psi, bins=bins, psi_range=psi_range,
                threshold=threshold, steps=steps,
            )
            if not xs:
                continue
            series.append((ds.series_label, xs, ys))
            colors.append(ds.color)

        if not series:
            return
        out = plot_wetted_fraction_datasets(
            series, out_dir / f"{comparison.name}_wetted_fraction.png",
            xlabel=comparison.x_label, colors=colors, **kwargs,
        )
        print(f"  {out}")
        return

    if comparison.x_values is None:
        print(
            f"  comparison {comparison.name!r} has no x_values configured; "
            "wetted_fraction needs one numeric value per case, skipped"
        )
        return

    x_by_case = dict(zip(comparison.cases, comparison.x_values))
    xs, ys = _wetted_fraction_xy(
        comparison.labelled_cases(), x_by_case, cases, comparison,
        target_psi=target_psi, bins=bins, psi_range=psi_range,
        threshold=threshold, steps=steps,
    )
    if not xs:
        return

    out = plot_wetted_fraction_vs_x(
        xs, ys, out_dir / f"{comparison.name}_wetted_fraction.png",
        xlabel=comparison.x_label, **kwargs,
    )
    print(f"  {out}")


def _resolve_n_workers(args) -> int:
    """--n-workers if given, else site.toml's [diagnostics] n_workers if
    explicitly set, else 1 (serial).

    Unlike cli/analyse.py's _resolve_parallelism, an unset [diagnostics]
    (or no site.toml) does NOT derive a worker count from core count --
    rendering a handful of PNGs is cheap enough that auto-parallelising by
    default would spawn processes for no measurable benefit.
    """
    if args.n_workers is not None:
        return max(1, args.n_workers)
    try:
        diagnostics = load_site(args.site).diagnostics
    except SiteConfigError:
        return 1
    return max(1, diagnostics.n_workers) if diagnostics.n_workers else 1


def _run_case(
    case: Case,
    *,
    diags: list[str],
    steps: list[int] | None,
    log: bool,
    smooth: bool,
    dpi: int | None,
    n_workers: int = 1,
    psi_range: tuple[float, float] | None = None,
    four_log: bool = True,
    theta_target_psi: float | None = None,
    theta_bins: int | None = None,
    theta_psi_range: tuple[float, float] | None = None,
    theta_wetted_threshold: float | None = None,
    n_cols: int | None = None,
    point_size: float | None = None,
    mark_rational: bool = False,
    profile_cmap: str | None = None,
    animate: bool = False,
) -> None:
    run_dir = Path.cwd() / case.name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"case {case.name!r}: no such folder {run_dir}")
    paths = RunPaths.detect(run_dir)

    # --step (CLI) outranks everything below it in the default -> case ->
    # case+diag tree -- an explicit runtime request wins over any configured
    # override. Absent that, each diag falls back to its own steps_for(diag)
    # override (a [cases.NAME.<diag>] table) before the case's plain steps.
    if "poincare" in diags:
        _plot_poincare(
            case, paths, steps or case.steps_for("poincare"), dpi=dpi,
            n_workers=n_workers, point_size=point_size,
        )
    if "connection_length" in diags:
        # --psi-range (if given) further bounds-filters whichever list is
        # already in effect (case.lc_psi_n_in, or case.psi_n_in if unset).
        # connection_length's own steps only select which already-gathered
        # poincare steps to plot (same "no interpolation" rule lc_psi_n_in
        # already has) -- it never gathers new data on its own.
        _plot_connection_length(
            case, paths, steps or case.steps_for("connection_length"),
            log=log, smooth=smooth, dpi=dpi, psi_range=psi_range, n_workers=n_workers,
        )
    if "four" in diags:
        _plot_four_modes(
            case, paths, steps or case.steps_for("four"), log=four_log, dpi=dpi,
            n_workers=n_workers,
        )
    if "profiles" in diags:
        _plot_profiles(
            case, paths, steps or case.steps_for("profiles"), dpi=dpi, n_workers=n_workers,
            mark_rational=mark_rational or case.mark_rational, cmap=profile_cmap,
            animate=animate or case.animate,
        )
    if "theta_hist" in diags:
        _plot_theta_hist(
            case, paths, steps or case.steps_for("theta_hist"),
            target_psi=theta_target_psi, bins=theta_bins, psi_range=theta_psi_range,
            n_cols=n_cols, dpi=dpi,
        )
    if "wetted_fraction" in diags:
        _plot_wetted_fraction(
            case, paths, steps or case.steps_for("wetted_fraction"),
            target_psi=theta_target_psi, bins=theta_bins, psi_range=theta_psi_range,
            threshold=theta_wetted_threshold, dpi=dpi,
        )
    comparison_only = [d for d in diags if d in COMPARISON_ONLY_DIAGS]
    if comparison_only:
        print(f"  diag(s) {comparison_only} are comparison-only (use --compare), skipped")


def _run_comparisons(
    names: list[str],
    comparisons: dict[str, Comparison],
    cases: dict[str, Case],
    *,
    diags: list[str],
    steps: list[int] | None,
    dpi: int | None,
    theta_target_psi: float | None,
    theta_bins: int | None,
    theta_psi_range: tuple[float, float] | None,
    n_cols: int | None,
    wetted_threshold: float | None,
    dataset_names: list[str] | None,
) -> int:
    unknown = [name for name in names if name not in comparisons]
    if unknown:
        print(f"error: unknown comparison(s) {unknown}; --list-comparisons to see defined ones")
        return 1

    comparable = [d for d in diags if d in COMPARABLE_DIAGS]
    not_comparable = [d for d in diags if d not in COMPARABLE_DIAGS]
    if not_comparable:
        print(f"  diag(s) {not_comparable} have no comparison renderer, skipped")
    if not comparable:
        return 0

    for name in names:
        print(f"==== compare: {name} ====")
        comparison = comparisons[name]
        if "theta_hist" in comparable:
            _compare_theta_hist(
                comparison, cases,
                target_psi=theta_target_psi, bins=theta_bins, psi_range=theta_psi_range,
                n_cols=n_cols, dpi=dpi, steps=steps,
            )
        if "wetted_fraction" in comparable:
            _compare_wetted_fraction(
                comparison, cases,
                target_psi=theta_target_psi, bins=theta_bins, psi_range=theta_psi_range,
                threshold=wetted_threshold, dpi=dpi, steps=steps,
                dataset_names=dataset_names,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_config:
        try:
            site = load_site(args.site)
        except SiteConfigError as exc:
            print(f"error: {exc}")
            return 1
        print(site.describe())
        return 0

    try:
        cases = load_cases(args.cases)
    except CasesError as exc:
        print(f"error: {exc}")
        return 1

    try:
        comparisons = load_comparisons(args.cases, cases)
    except CasesError as exc:
        print(f"error: {exc}")
        return 1

    if args.list:
        for name, case in cases.items():
            note = f" -- {case.note}" if case.note else ""
            print(f"{name} ({len(case.steps)} steps){note}")
        return 0

    if args.list_comparisons:
        for name, comparison in comparisons.items():
            note = f" -- {comparison.note}" if comparison.note else ""
            if comparison.datasets:
                n_cases = sum(len(ds.cases) for ds in comparison.datasets.values())
                size = f"{len(comparison.datasets)} datasets, {n_cases} cases"
            else:
                size = f"{len(comparison.cases)} cases"
            print(f"{name} ({size}){note}")
        return 0

    diags = args.diags or list(DIAG_CHOICES)
    dpi = args.dpi
    theta_target_psi = args.theta_target_psi
    theta_bins = args.theta_bins
    theta_psi_range = (
        tuple(args.theta_psi_n_range) if args.theta_psi_n_range is not None else None
    )
    n_cols = args.n_cols

    if args.comparisons:
        return _run_comparisons(
            args.comparisons, comparisons, cases,
            diags=diags, steps=args.steps, dpi=dpi,
            theta_target_psi=theta_target_psi, theta_bins=theta_bins,
            theta_psi_range=theta_psi_range, n_cols=n_cols,
            wetted_threshold=args.theta_wetted_threshold,
            dataset_names=args.datasets_selected,
        )

    selected = args.selected or list(cases)
    unknown = [name for name in selected if name not in cases]
    if unknown:
        print(f"error: unknown case(s) {unknown}; --list to see defined cases")
        return 1

    n_workers = _resolve_n_workers(args)
    psi_range = tuple(args.psi_range) if args.psi_range is not None else None

    for name in selected:
        print(f"==== {name} ====")
        try:
            _run_case(
                cases[name],
                diags=diags,
                steps=args.steps,
                log=not args.linear,
                smooth=args.smooth,
                dpi=dpi,
                n_workers=n_workers,
                psi_range=psi_range,
                four_log=not args.four_linear,
                theta_target_psi=theta_target_psi,
                theta_bins=theta_bins,
                theta_psi_range=theta_psi_range,
                theta_wetted_threshold=args.theta_wetted_threshold,
                n_cols=n_cols,
                point_size=args.point_size,
                mark_rational=args.mark_rational,
                profile_cmap=args.profile_cmap,
                animate=args.animate,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}")
            return 1

    return 0
