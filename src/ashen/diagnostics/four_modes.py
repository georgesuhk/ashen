"""Time evolution of jorek2_four mode amplitudes across restart steps.

Extracts, from the per-step caches `analyse --diag four` already wrote
(four_cache), the peak |amplitude| over the radial (psi_n) grid for each
requested (variable, n, m), one value per restart step -- what
plotting.four_modes draws as a time series. Pure/no matplotlib import,
like diagnostics.connection_length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ashen.diagnostics import four_cache as fc
from ashen.diagnostics.qprofile import find_rational_surfaces, read_qprofile
from ashen.paths import RunPaths
from ashen.postproc import read_zeroD, zero_d_is_usable

__all__ = [
    "ModeKey", "max_amplitude_series", "radial_amplitude_series",
    "RADIAL_QUANTITIES", "radial_values",
    "rational_surface_series",
    "DELTA_B", "DELTA_B_OVER_B", "delta_b_series",
    "edge_minor_radius", "effective_minor_radius", "minor_radius_profile",
    "GrowthFit", "fit_growth_rate", "growth_rate_series", "format_growth_rates",
]

#: (variable, toroidal mode n, poloidal mode m).
ModeKey = tuple[str, int, int]

#: Pseudo-variable names delta_b_series's output is
#: keyed under -- neither is a real jorek2_four cache variable, so callers
#: gate on these names to request a derived quantity rather than a raw one.
DELTA_B = "delta_b"
DELTA_B_OVER_B = "delta_b_over_b"


def _select_keys(
    per_step: Sequence[Mapping[ModeKey, fc.FourRecord]],
    variables: Sequence[str] | None,
    modes: Sequence[tuple[int, int]] | None,
) -> set[ModeKey]:
    """Every (variable, n, m) present in any step's cache, narrowed by the
    `variables`/`modes` filters -- None for either means "no filter", so the
    result is the union across `per_step`.

    Shared by max_amplitude_series and radial_amplitude_series so the two
    cannot disagree about which keys a given filter selects. `modes` entries
    are (n, m), matching FourRecord's own field order.
    """
    keys: set[ModeKey] = set()
    for records in per_step:
        keys.update(records)
    if variables is not None:
        wanted_vars = set(variables)
        keys = {k for k in keys if k[0] in wanted_vars}
    if modes is not None:
        wanted_modes = {(int(n), int(m)) for n, m in modes}
        keys = {k for k in keys if (k[1], k[2]) in wanted_modes}
    return keys


def max_amplitude_series(
    paths: RunPaths,
    steps: Sequence[int],
    *,
    variables: Sequence[str] | None = None,
    modes: Sequence[tuple[int, int]] | None = None,
) -> dict[ModeKey, np.ndarray]:
    """{(variable, n, m): amplitudes}, one value per `steps` entry.

    amplitudes[i] = max(record.abs) over the radial grid for that key at
    steps[i]; nan if that step has no cache, or lacks that key (e.g. an n
    the model doesn't produce) -- visibly missing, not silently dropped,
    same convention as connection_length.harmonic_connection_length.

    variables/modes filter which keys come back; None for either = union
    of every key found across the requested steps' caches.
    """
    per_step: list[dict[ModeKey, fc.FourRecord]] = [
        fc.read_cache(paths.four_cache(step)) for step in steps
    ]
    keys = _select_keys(per_step, variables, modes)

    series: dict[ModeKey, np.ndarray] = {}
    for key in keys:
        values = np.full(len(steps), np.nan)
        for i, records in enumerate(per_step):
            record = records.get(key)
            if record is not None and record.abs.size:
                values[i] = float(np.max(record.abs))
        series[key] = values
    return series


#: What radial_amplitude_series can return per psi_n point. "abs" is |c|;
#: "real" and "phase" are phase-aligned -- see radial_values.
RADIAL_QUANTITIES = ("abs", "real", "phase")


def _check_radial_quantity(quantity: str) -> None:
    if quantity not in RADIAL_QUANTITIES:
        raise ValueError(
            f"unknown radial quantity {quantity!r}; expected one of {RADIAL_QUANTITIES}"
        )


def radial_values(record: fc.FourRecord, quantity: str = "abs") -> np.ndarray:
    """One mode's radial profile as `quantity` (a RADIAL_QUANTITIES entry).

    jorek2_four's complex coefficient c(psi_n) has its phase measured from
    theta_star = 0, phi = 0, which drifts as the mode rotates -- so Re(c) as
    written flips sign and shape between steps for no physical reason. What
    *is* physical is how the phase varies across psi_n within one step (e.g.
    a tearing eigenfunction's sign change across its rational surface). So
    "real" and "phase" first rotate the whole profile by one reference phase
    phi0, taken where |c| peaks: Re(c e^{-i phi0}) is then positive at the
    peak and comparable across steps, and the phase is relative to the
    peak's, wrapped to (-pi, pi]. Phase is noise wherever |c| is near zero.
    """
    _check_radial_quantity(quantity)
    if quantity == "abs":
        return record.abs
    c = record.real + 1j * record.imag
    if c.size:
        c = c * np.exp(-1j * np.angle(c[np.argmax(np.abs(c))]))
    return c.real if quantity == "real" else np.angle(c)


def radial_amplitude_series(
    paths: RunPaths,
    steps: Sequence[int],
    *,
    variables: Sequence[str] | None = None,
    modes: Sequence[tuple[int, int]] | None = None,
    quantity: str = "abs",
) -> dict[ModeKey, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """{(variable, n, m): {step: (psi_n, values)}} -- each mode's radial
    eigenfunction, one curve per step, rather than the scalar per step
    max_amplitude_series reduces it to.

    This is the whole radial structure jorek2_four already wrote and every
    other consumer here discards: FourRecord carries psi_n/real/imag, and
    both max_amplitude_series and rational_surface_series collapse it to one
    number. Nothing is recomputed -- the arrays come straight off the cache.

    `quantity` defaults to abs; "real"/"phase" give the signed structure,
    phase-aligned per step so it stays comparable across steps -- see
    radial_values for why the raw real part is not.

    Missing data is *absent*, not nan -- a step with no cache, or whose cache
    lacks that key, simply has no entry for that step, and a key present in
    no step at all is omitted entirely. The scalar series here are
    fixed-length and aligned to `steps`, where nan is the only way to show a
    hole; a curve has no such alignment, so absence says it directly and the
    plotting layer draws nothing for that step.

    variables/modes filter which keys come back -- see _select_keys.
    """
    _check_radial_quantity(quantity)
    per_step: list[dict[ModeKey, fc.FourRecord]] = [
        fc.read_cache(paths.four_cache(step)) for step in steps
    ]
    keys = _select_keys(per_step, variables, modes)

    series: dict[ModeKey, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for key in keys:
        curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for step, records in zip(steps, per_step):
            record = records.get(key)
            if record is not None and record.psi_n.size:
                curves[int(step)] = (record.psi_n, radial_values(record, quantity))
        if curves:
            series[key] = curves
    return series


def rational_surface_series(
    paths: RunPaths,
    steps: Sequence[int],
    modes: Sequence[tuple[int, int]],
    *,
    variables: Sequence[str] | None = None,
) -> dict[ModeKey, np.ndarray]:
    """{(variable, n, m): amplitudes} pinned to the q=m/n rational surface,
    instead of max_amplitude_series's whole-domain max.

    Per step: the cached q-profile (qprofile.run_qprofile_step, gathered
    alongside --diag four) is searched for every psi_n where q crosses m/n;
    the four cache's amplitude is linearly interpolated onto each crossing
    and the largest kept. A reversed-shear profile can cross a given q more
    than once -- each is a distinct physical rational surface, only the
    strongest matters for "how hard is this helicity driven".

    Uses abs, not signed real part: a Fourier component's phase is an
    arbitrary toroidal-angle offset with no fixed sign convention across
    steps, so the signed real part flips as the mode rotates -- not a
    meaningful growth trace. abs also matches plotting.four_modes' default
    log-scale axis.

    n=0 has no rational surface (m/0 undefined), silently skipped. A step
    missing either cache, or whose q-profile never crosses the target, is
    nan -- same convention as max_amplitude_series.
    """
    wanted_modes = [(int(n), int(m)) for n, m in modes if int(n) != 0]

    per_step_four: list[dict[ModeKey, fc.FourRecord]] = [
        fc.read_cache(paths.four_cache(step)) for step in steps
    ]
    per_step_q: list[tuple[np.ndarray, np.ndarray] | None] = []
    for step in steps:
        q_path = paths.qprofile(step)
        per_step_q.append(read_qprofile(q_path) if q_path.is_file() else None)

    keys: set[ModeKey] = set()
    for records in per_step_four:
        for key in records:
            if (key[1], key[2]) in wanted_modes and (variables is None or key[0] in variables):
                keys.add(key)

    series: dict[ModeKey, np.ndarray] = {}
    for key in keys:
        _, n, m = key
        q_target = m / n
        values = np.full(len(steps), np.nan)
        for i, (records, qprof) in enumerate(zip(per_step_four, per_step_q)):
            record = records.get(key)
            if record is None or qprof is None:
                continue
            psi_n_q, q = qprof
            crossings = find_rational_surfaces(psi_n_q, q, q_target)
            if not crossings:
                continue
            amp_at = np.interp(crossings, record.psi_n, record.abs)
            values[i] = float(np.max(amp_at))
        series[key] = values
    return series


def effective_minor_radius(
    psi_n: np.ndarray, q: np.ndarray, *, delta_psi: float, f0: float, r_axis: float
) -> tuple[np.ndarray, np.ndarray]:
    """(psi_n, r) with r the radius of the circle that holds each surface's
    toroidal flux: pi r^2 B0 = 2 pi int q dpsi, B0 = f0 / r_axis.

    psi is per radian (JOREK's), delta_psi = |psi_bnd - psi_axis|. For a
    shaped surface r is sqrt(area / pi) to leading order in aspect ratio,
    e.g. a sqrt(kappa) for an ellipse. The q-profile starts just off the
    axis; q is held flat from there to psi_n = 0, where r = 0.
    """
    psi_n = np.asarray(psi_n, dtype=float)
    q = np.abs(np.asarray(q, dtype=float))
    if psi_n[0] > 0:
        psi_n = np.concatenate(([0.0], psi_n))
        q = np.concatenate(([q[0]], q))
    flux = np.concatenate(([0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(psi_n))))
    return psi_n, np.sqrt(2.0 * r_axis * abs(delta_psi) * flux / abs(f0))


def minor_radius_profile(
    paths: RunPaths, step: int, *, f0: float, r_axis: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """effective_minor_radius for one step, from its q-profile and zeroD
    caches. None if either is missing or unreadable."""
    q_path = paths.qprofile(step)
    zero_d = paths.zero_d(step)
    if not q_path.is_file() or not zero_d_is_usable(zero_d):
        return None
    values = read_zeroD(zero_d)
    if "psi_axis" not in values or "psi_bnd" not in values:
        return None
    psi_n, q = read_qprofile(q_path)
    if psi_n.size < 2:
        return None
    return effective_minor_radius(
        psi_n, q, delta_psi=values["psi_bnd"] - values["psi_axis"], f0=f0, r_axis=r_axis
    )


def delta_b_series(
    paths: RunPaths,
    steps: Sequence[int],
    *,
    r_axis: float,
    f0: float,
    modes: Sequence[tuple[int, int]] | None = None,
    b_ref: float | None = None,
    rational: bool = False,
    edge: bool = False,
) -> dict[ModeKey, np.ndarray]:
    """Perturbed radial field of each Psi mode, one value per `steps` entry.

    A flux perturbation Psi_mn exp(i m theta) has a field normal to the
    surface of (1/R) dPsi/dl_pol, so

        delta_b(psi_n) = |m| |Psi_mn(psi_n)| / (r_axis * r(psi_n))   [T]

    with r the surface's effective minor radius (effective_minor_radius).
    The value kept per step is the largest over the radial grid, or, with
    `rational`, the largest over the mode's q = m/n surfaces (as
    rational_surface_series). With `edge` it is the value at the outermost
    point of the radial grid instead: the edge of JOREK's domain, which with
    an extended boundary is in the vacuum outside the plasma, where a probe
    would sit. Keys are (DELTA_B, n, m); with `b_ref` the
    values are divided by it and keyed (DELTA_B_OVER_B, n, m).

    Approximations: r_axis stands for R everywhere on the surface, and the
    poloidal angle advances evenly along the surface (a circle of radius r).

    m = 0 modes carry no radial field in this form and are left out, as are
    n = 0 modes when `rational`. A step without a four cache, or without the
    q-profile and zeroD caches r needs, is nan.
    """
    target = DELTA_B if b_ref is None else DELTA_B_OVER_B
    per_step = [fc.read_cache(paths.four_cache(step)) for step in steps]
    radii = [minor_radius_profile(paths, step, f0=f0, r_axis=r_axis) for step in steps]
    keys = _select_keys(per_step, ["Psi"], modes)

    series: dict[ModeKey, np.ndarray] = {}
    for _, n, m in keys:
        if m == 0 or (rational and n == 0):
            continue
        values = np.full(len(steps), np.nan)
        for i, (records, radius, step) in enumerate(zip(per_step, radii, steps)):
            record = records.get(("Psi", n, m))
            if record is None or radius is None or not record.abs.size:
                continue
            if edge:
                at, amp = record.psi_n[-1:], record.abs[-1:]
            elif rational:
                psi_n_q, q = read_qprofile(paths.qprofile(step))
                at = np.asarray(find_rational_surfaces(psi_n_q, q, m / n), dtype=float)
                amp = np.interp(at, record.psi_n, record.abs)
            else:
                at, amp = record.psi_n, record.abs
            # r^2 is the toroidal flux, smooth in psi_n; r itself is a
            # square root at the axis and interpolates badly there.
            r = np.sqrt(np.interp(at, radius[0], radius[1] ** 2))
            ok = r > 0
            if not np.any(ok):
                continue
            field = abs(m) * amp[ok] / (r_axis * r[ok])
            values[i] = float(np.max(field)) / (b_ref if b_ref is not None else 1.0)
        series[(target, n, m)] = values
    return series


def edge_minor_radius(
    paths: RunPaths, steps: Sequence[int], *, f0: float, r_axis: float
) -> float | None:
    """The minor radius [m] delta_b_series(edge=True) evaluates at: the
    outermost radial point of the four cache, at the first of `steps` that
    has what it needs. None if no step does."""
    for step in steps:
        radius = minor_radius_profile(paths, step, f0=f0, r_axis=r_axis)
        records = fc.read_cache(paths.four_cache(step))
        psi_n = next((r.psi_n for k, r in records.items() if k[0] == "Psi" and r.psi_n.size), None)
        if radius is None or psi_n is None:
            continue
        return float(np.sqrt(np.interp(psi_n[-1], radius[0], radius[1] ** 2)))
    return None


@dataclass(frozen=True)
class GrowthFit:
    """Least-squares exponential-growth fit: |amplitude| ~
    exp(intercept)*exp(gamma*t), i.e. ln|amplitude| = gamma*t + intercept,
    fit against real time in seconds.

    gamma is always physical (1/s), independent of a plot's x-axis units
    (step index vs. microseconds) -- computed once here, reused unchanged
    everywhere shown.
    """

    gamma: float
    intercept: float
    n_points: int


def fit_growth_rate(t: Sequence[float], y: Sequence[float]) -> GrowthFit | None:
    """Least-squares fit of ln(y) vs t. None if fewer than 2 finite,
    positive-y points survive (not enough to fit a line; ln of a
    non-positive amplitude is undefined)."""
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(t_arr) & np.isfinite(y_arr) & (y_arr > 0)
    n_points = int(mask.sum())
    if n_points < 2:
        return None
    gamma, intercept = np.polyfit(t_arr[mask], np.log(y_arr[mask]), 1)
    return GrowthFit(gamma=float(gamma), intercept=float(intercept), n_points=n_points)


def growth_rate_series(
    series: Mapping[ModeKey, np.ndarray],
    true_times: Sequence[float],
    steps: Sequence[int],
    *,
    step_range: tuple[int, int] | None = None,
) -> dict[ModeKey, GrowthFit]:
    """One GrowthFit per mode in `series`, fit against true_times (seconds,
    one per `steps` entry, same alignment as max_amplitude_series's output).

    step_range, if given, restricts the fit to [start, end] inclusive --
    picks the visually-linear region, since noise-floor or post-saturation
    points bias a whole-range fit. None (default) uses every step.

    A mode with fewer than 2 valid points in the window is omitted, not
    given a meaningless fit.
    """
    steps_arr = np.asarray(steps)
    t = np.asarray(true_times, dtype=float)
    if step_range is not None:
        lo, hi = step_range
        mask = (steps_arr >= lo) & (steps_arr <= hi)
    else:
        mask = np.ones(len(steps_arr), dtype=bool)

    out: dict[ModeKey, GrowthFit] = {}
    for key, y in series.items():
        fit = fit_growth_rate(t[mask], np.asarray(y)[mask])
        if fit is not None:
            out[key] = fit
    return out


def format_growth_rates(fits: Mapping[ModeKey, GrowthFit]) -> str:
    """Human-readable table, sorted by (variable, m, n) -- m before n to
    match cases.toml's modes [m, n] convention. Written by
    `plot --diag four` to four_dir/growth_rates.txt."""
    header = f"{'variable':<12}{'m':>4}{'n':>4}{'gamma [1/s]':>18}{'n_points':>10}"
    lines = [header]
    for var, n, m in sorted(fits, key=lambda k: (k[0], k[2], k[1])):
        fit = fits[(var, n, m)]
        lines.append(f"{var:<12}{m:>4}{n:>4}{fit.gamma:>18.6e}{fit.n_points:>10}")
    return "\n".join(lines) + "\n"
