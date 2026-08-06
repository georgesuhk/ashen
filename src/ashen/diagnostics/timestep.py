"""Simulation time at a restart step, in both SI seconds and JOREK's own
code units.

A one-off inspection tool, not a batch diagnostic to keep warm: it always
re-runs ``jorek2_postproc``'s ``zeroD_quantities`` (in each unit system)
rather than trusting whatever ``zeroD_quantities_s<step>.dat`` cache
``analyse --diag zerod`` may have already left behind, since that cache is
SI-only and may be stale relative to a `--force` re-run here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ashen.jorek2 import Jorek2Run, run_zero_d
from ashen.paths import RunPaths
from ashen.postproc import read_zeroD

__all__ = ["StepTime", "step_time"]


@dataclass(frozen=True)
class StepTime:
    """One restart step's simulation time, in both unit systems."""

    step: int
    time_si: float
    time_jorek: float


def step_time(run: Jorek2Run, paths: RunPaths, step: int) -> StepTime:
    """Run ``zeroD_quantities`` twice for ``step`` -- once ``si-units``, once
    ``jorek-units`` -- and read back the ``Time`` column each time.
    """
    si_path = run_zero_d(run, step, paths, si_units=True)
    jorek_path = run_zero_d(run, step, paths, si_units=False)
    time_si = read_zeroD(si_path)["Time"]
    time_jorek = read_zeroD(jorek_path)["Time"]
    return StepTime(step=step, time_si=time_si, time_jorek=time_jorek)
