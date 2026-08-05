"""Time evolution of jorek2_four mode amplitudes across restart steps.

Extracts, from the per-step caches ``analyse --diag four`` already wrote
(:mod:`ashen.diagnostics.four_cache`), the peak ``|amplitude|`` over the
radial (``psi_n``) grid for each requested ``(variable, n, m)``, one value per
restart step -- the data :mod:`ashen.plotting.four_modes` draws as a time
series. Pure/no matplotlib import, like :mod:`ashen.diagnostics.
connection_length`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ashen.diagnostics import four_cache as fc
from ashen.paths import RunPaths

__all__ = ["ModeKey", "max_amplitude_series"]

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
