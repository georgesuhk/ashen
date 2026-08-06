"""Time evolution of jorek2_four mode amplitudes across restart steps.

Extracts, from the per-step caches ``analyse --diag four`` already wrote
(:mod:`ashen.diagnostics.four_cache`), the peak ``|amplitude|`` over the
radial (``psi_n``) grid for each requested ``(variable, n, m)``, one value per
restart step -- the data :mod:`ashen.plotting.four_modes` draws as a time
series. Pure/no matplotlib import, like :mod:`ashen.diagnostics.
connection_length`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ashen.diagnostics import four_cache as fc
from ashen.diagnostics.qprofile import find_rational_surfaces, read_qprofile
from ashen.paths import RunPaths

__all__ = [
    "ModeKey", "max_amplitude_series", "rational_surface_series",
    "GrowthFit", "fit_growth_rate", "growth_rate_series", "format_growth_rates",
]

#: (variable, toroidal mode n, poloidal mode m).
ModeKey = tuple[str, int, int]


def max_amplitude_series(
    paths: RunPaths,
    steps: Sequence[int],
    *,
    variables: Sequence[str] | None = None,
    modes: Sequence[tuple[int, int]] | None = None,
) -> dict[ModeKey, np.ndarray]:
    """``{(variable, n, m): amplitudes}``, one value per ``steps`` entry.

    ``amplitudes[i]`` is ``max(record.abs)`` over the radial grid for that
    key at ``steps[i]`` -- ``nan`` if that step has no cache at all, or the
    cache doesn't carry that particular key (e.g. an ``n`` the run's model
    doesn't produce). Visibly missing rather than silently dropped, same
    convention as :func:`ashen.diagnostics.connection_length.
    harmonic_connection_length`.

    ``variables``/``modes`` filter which keys come back; ``None`` for either
    means "every one found across the requested steps' caches" -- the union,
    since different steps can have carried different keys.
    """
    per_step: list[dict[ModeKey, fc.FourRecord]] = [
        fc.read_cache(paths.four_cache(step)) for step in steps
    ]

    keys: set[ModeKey] = set()
    for records in per_step:
        keys.update(records)
    if variables is not None:
        wanted_vars = set(variables)
        keys = {k for k in keys if k[0] in wanted_vars}
    if modes is not None:
        wanted_modes = {(int(n), int(m)) for n, m in modes}
        keys = {k for k in keys if (k[1], k[2]) in wanted_modes}

    series: dict[ModeKey, np.ndarray] = {}
    for key in keys:
        values = np.full(len(steps), np.nan)
        for i, records in enumerate(per_step):
            record = records.get(key)
            if record is not None and record.abs.size:
                values[i] = float(np.max(record.abs))
        series[key] = values
    return series


def rational_surface_series(
    paths: RunPaths,
    steps: Sequence[int],
    modes: Sequence[tuple[int, int]],
    *,
    variables: Sequence[str] | None = None,
) -> dict[ModeKey, np.ndarray]:
    """``{(variable, n, m): amplitudes}`` pinned to the ``q = m/n`` rational
    surface, instead of :func:`max_amplitude_series`'s whole-domain max.

    For each step, the cached q-profile (:func:`ashen.diagnostics.qprofile.
    run_qprofile_step`, gathered alongside ``--diag four``) is searched for
    every ``psi_n`` where ``q`` crosses ``m/n``; the four cache's amplitude
    is linearly interpolated onto each crossing and the largest kept. A
    reversed-shear profile can cross a given q more than once -- each is a
    distinct physical rational surface, and only the strongest is relevant
    to "how hard is this helicity driven".

    Uses ``abs``, not the signed real part, even though the physically
    literal quantity is Re(psi) at that surface: a Fourier component's phase
    is an arbitrary toroidal-angle offset with no fixed sign convention
    across steps, so the signed real part flips sign as the mode rotates and
    is not a meaningful growth trace. ``abs`` is also what the log-scale
    default axis in :mod:`ashen.plotting.four_modes` requires.

    ``n=0`` modes have no rational surface (``m/0`` is undefined) and are
    silently skipped. A step missing either cache, or whose q-profile never
    crosses the target, is ``nan`` there -- same convention as
    :func:`max_amplitude_series`.
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


@dataclass(frozen=True)
class GrowthFit:
    """A least-squares exponential-growth fit, ``|amplitude| ~
    exp(intercept) * exp(gamma * t)`` -- i.e. ``ln|amplitude| = gamma * t +
    intercept``, fit against real time in seconds.

    ``gamma`` is always physical (1/s), independent of whatever units a
    plot's x-axis happens to display (step index, or time in microseconds)
    -- growth rate is only meaningful against real time, so it is computed
    once here and reused unchanged everywhere it's shown.
    """

    gamma: float
    intercept: float
    n_points: int


def fit_growth_rate(t: Sequence[float], y: Sequence[float]) -> GrowthFit | None:
    """Least-squares fit of ``ln(y)`` vs ``t``. ``None`` if fewer than 2
    finite, positive-``y`` points survive -- not enough to fit a line
    through, and ``ln`` of a non-positive amplitude is undefined.
    """
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
    """One :class:`GrowthFit` per mode in ``series``, fit against
    ``true_times`` (seconds, one per ``steps`` entry, same alignment as
    :func:`max_amplitude_series`'s output).

    ``step_range``, if given, restricts the fit to steps within
    ``[start, end]`` inclusive -- picking the visually-linear region of a
    growth curve, since points near the noise floor (pre-growth) or past
    saturation bias a whole-range least-squares fit. ``None`` (default)
    uses every step in ``steps``.

    A mode with fewer than 2 valid points inside the window is silently
    omitted from the result rather than given a meaningless fit.
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
    """A human-readable table, sorted by ``(variable, m, n)`` -- ``m``
    before ``n`` to match the ``[m, n]`` convention ``cases.toml``'s
    ``four_modes`` field uses. What ``plot --diag four`` writes to
    ``four_dir/growth_rates.txt``.
    """
    header = f"{'variable':<12}{'m':>4}{'n':>4}{'gamma [1/s]':>18}{'n_points':>10}"
    lines = [header]
    for var, n, m in sorted(fits, key=lambda k: (k[0], k[2], k[1])):
        fit = fits[(var, n, m)]
        lines.append(f"{var:<12}{m:>4}{n:>4}{fit.gamma:>18.6e}{fit.n_points:>10}")
    return "\n".join(lines) + "\n"
