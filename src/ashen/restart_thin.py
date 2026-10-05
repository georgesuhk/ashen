"""Thinning a run's restart files (`util --func downsample_restarts`):
keep every N-th step, delete the rest, to give back disk space -- e.g. a
run that saved every 20 steps kept at every 40.

A deleted restart is gone for good, so this keeps, whatever N says:

- every step the case uses in cases.toml: its steps, each diag's own
  steps, and its ptrace_start_step, ptrace_end_step and ptrace_pdf_step --
  `analyse`, `plot` and `ptrace` would otherwise find them missing;
- the run's first restart (usually the equilibrium) and its last (the one
  to continue the run from);
- anything not named exactly ``jorek<5 or 6 digits>.h5`` -- jorek_restart.h5
  and every other file are not restarts in this sense.

A step is kept when step % every == 0, so the steps kept line up across
runs whatever step each started at. :func:`plan_thin` decides, touching
nothing; :func:`apply_thin` deletes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ThinPlan", "apply_thin", "case_steps", "plan_thin"]

#: A numbered restart, at either width JOREK names them.
_RESTART = re.compile(r"^jorek(\d{5,6})\.h5$")


@dataclass
class ThinPlan:
    """What downsample_restarts does to one run folder."""

    run_dir: Path
    every: int
    #: Files to delete, with their steps and sizes.
    delete: list[tuple[int, Path, int]] = field(default_factory=list)
    #: Steps that are not multiples of `every` but are kept, and why.
    kept_anyway: dict[int, str] = field(default_factory=dict)
    #: How many restart steps there are, and how many stay.
    n_steps: int = 0
    n_kept: int = 0

    @property
    def freed(self) -> int:
        return sum(size for _, _, size in self.delete)


def case_steps(case) -> set[int]:
    """Every restart step a case names in cases.toml."""
    steps = set(case.steps)
    for diag_steps in case.diag_steps.values():
        steps.update(diag_steps)
    for step in (case.ptrace_start_step, case.ptrace_end_step, case.ptrace_pdf_step):
        if step is not None:
            steps.add(step)
    return steps


def plan_thin(run_dir: Path | str, every: int, *, keep: set[int] = frozenset()) -> ThinPlan:
    """Decide which restarts of run_dir go: those whose step is not a
    multiple of `every`, unless in `keep` (the case's own steps) or the
    first or last. Touches nothing."""
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    run_dir = Path(run_dir)
    plan = ThinPlan(run_dir=run_dir, every=every)
    files: dict[int, list[Path]] = {}
    for path in sorted(run_dir.iterdir()) if run_dir.is_dir() else []:
        match = _RESTART.match(path.name)
        if match and (path.is_file() or path.is_symlink()):
            files.setdefault(int(match.group(1)), []).append(path)
    steps = sorted(files)
    plan.n_steps = len(steps)
    if not steps:
        return plan
    for step in steps:
        if step % every == 0:
            continue
        if step == steps[0]:
            plan.kept_anyway[step] = "the run's first restart"
        elif step == steps[-1]:
            plan.kept_anyway[step] = "the run's last restart"
        elif step in keep:
            plan.kept_anyway[step] = "cases.toml uses it"
        else:
            for path in files[step]:
                size = 0 if path.is_symlink() else path.stat().st_size
                plan.delete.append((step, path, size))
    plan.n_kept = len(steps) - len({step for step, _, _ in plan.delete})
    return plan


def apply_thin(plan: ThinPlan) -> None:
    """Delete the plan's files. Irreversible."""
    for _, path, _ in plan.delete:
        path.unlink()
