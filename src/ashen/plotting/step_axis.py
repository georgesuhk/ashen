"""A second x axis along the top of a time-series panel: the time step when
the panel is against time, the time when it is against the step.

The mapping is the run's own (diagnostics.live_data.read_step_times), so a
run whose time step changed part way is labelled correctly: the two axes
are not a fixed multiple of each other.
"""

from __future__ import annotations

import numpy as np

from ashen.diagnostics.live_data import StepTimes

__all__ = ["add_step_time_axis"]


def add_step_time_axis(ax, step_times: StepTimes, *, x_is: str):
    """Add the other of step / time along the top of ``ax``, whose x axis is
    ``x_is`` ("time", in step_times' unit, or "step"). Returns the new axis,
    or None when the run has fewer than two steps to map between."""
    if x_is not in ("time", "step"):
        raise ValueError(f"x_is must be 'time' or 'step', not {x_is!r}")
    steps = np.asarray(step_times.steps, dtype=float)
    time = np.asarray(step_times.time, dtype=float)
    # time only ever increases with the step; a repeat would make the map
    # two-valued, so keep each time's first step
    time, first = np.unique(time, return_index=True)
    steps = steps[first]
    if len(steps) < 2:
        return None
    if x_is == "time":
        forward, inverse = _line(time, steps), _line(steps, time)
        label = "time step"
    else:
        forward, inverse = _line(steps, time), _line(time, steps)
        label = step_times.time_label
    top = ax.secondary_xaxis("top", functions=(forward, inverse))
    top.set_xlabel(label, fontsize=9)
    top.tick_params(labelsize=8)
    return top


def _line(x, y):
    """Piecewise-linear x -> y through the points, carried on straight past
    both ends. np.interp alone would clamp there, and the axis margins lie
    beyond the data: their ticks would all read the end value."""
    def f(values):
        values = np.asarray(values, dtype=float)
        out = np.interp(values, x, y)
        low, high = values < x[0], values > x[-1]
        out[low] = y[0] + (values[low] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        out[high] = y[-1] + (values[high] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
        return out
    return f
