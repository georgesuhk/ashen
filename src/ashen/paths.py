"""Run-folder conventions: restart-step padding and derived filenames.

Ports basics.py:120-163 (get_jorek_padding_length, restart_filename, pad_t_step).

Bug this module prevents: pad width is per-run, not constant, hence
get_jorek_padding_length sniffs it -- but old code mixed the sniffed width
with the width=6 default across call sites (poinc_diag.py read flux
surfaces with jorek_pad_width at :100/:157/:186 but wrote .npz with the
default at :211; analysis.py used sniffed at :122, default at :155), so a
non-6-padded run silently produced caches whose names didn't match what
the reader looked for. Here the width is resolved once per run into
RunPaths, and every filename comes from that object.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_PAD_WIDTH",
    "PaddingError",
    "RunPaths",
    "read_float",
    "write_float",
    "detect_pad_width",
    "step_str",
]

#: Only a fallback for synthetic cases. Real runs must sniff the width.
DEFAULT_PAD_WIDTH = 6

_RESTART_RE = re.compile(r"^jorek.*?(\d+)\.h5$")


class PaddingError(RuntimeError):
    """Raised when the restart padding width cannot be determined."""


def step_str(step: int | float, width: int = DEFAULT_PAD_WIDTH) -> str:
    """Zero-pad a step index. Prefer :meth:`RunPaths.step_str`."""
    return f"{int(step):0{width}d}"


def write_float(path: Path | str, value: float) -> None:
    """Write a single float, full double precision. Ports ``basics.py:17``."""
    Path(path).write_text(f"{value:.16e}\n", encoding="utf-8")


def read_float(path: Path | str) -> float:
    """Read a single float written by :func:`write_float`. Ports ``basics.py:21``."""
    return float(Path(path).read_text(encoding="utf-8").strip())


def detect_pad_width(directory: Path | str = ".") -> int:
    """Infer the zero-pad width from ``jorek*.h5`` files in ``directory``.

    Takes the majority width, so a stray differently-named file does not
    change the answer.
    """
    widths = [
        len(match.group(1))
        for path in Path(directory).glob("jorek*.h5")
        if (match := _RESTART_RE.match(path.name))
    ]
    if not widths:
        raise PaddingError(
            f"{directory}: no JOREK restart files (jorek*.h5) to infer the "
            "step padding width from."
        )
    return Counter(widths).most_common(1)[0][0]


@dataclass(frozen=True)
class RunPaths:
    """Filename conventions for one run folder, with the pad width fixed once.

    Construct with :meth:`detect` so the width comes from the run itself; pass
    it around rather than re-deriving, which is how the read/write mismatch
    arose in the first place.
    """

    run_dir: Path
    pad_width: int = DEFAULT_PAD_WIDTH

    @classmethod
    def detect(cls, run_dir: Path | str) -> RunPaths:
        run_dir = Path(run_dir)
        return cls(run_dir=run_dir, pad_width=detect_pad_width(run_dir))

    def step_str(self, step: int | float) -> str:
        return step_str(step, self.pad_width)

    # --- JOREK outputs ---

    def restart(self, step: int | float, prefix: str = "jorek", ext: str = ".h5") -> Path:
        return self.run_dir / f"{prefix}{self.step_str(step)}{ext}"

    @property
    def live_restart(self) -> Path:
        """The file ``jorek2_*`` tools read: always ``jorek_restart.h5``."""
        return self.run_dir / "jorek_restart.h5"

    # --- postprocessing artefacts ---

    @property
    def postproc_dir(self) -> Path:
        return self.run_dir / "postproc"

    def zero_d(self, step: int | float, *, si_units: bool = True) -> Path:
        if si_units:
            return self.postproc_dir / f"zeroD_quantities_s{self.step_str(step)}.dat"
        return self.postproc_dir / f"zeroD_quantities_jorek_s{self.step_str(step)}.dat"

    def flux_surface(self, psi_n: float, step: int | float) -> Path:
        return (
            self.postproc_dir
            / f"fluxsurface_at_psi_{psi_n:.3f}_s{self.step_str(step)}.dat"
        )

    def qprofile(self, step: int | float) -> Path:
        """One step's ``Psi_n``/``q`` table -- ``exec_commands.f90::qprofile``'s
        own naming for a single-step ``for step`` loop (``loop_min_step ==
        loop_max_step``): ``qprofile_s<step>.dat``, no range suffix."""
        return self.postproc_dir / f"qprofile_s{self.step_str(step)}.dat"

    def profile_cache(
        self,
        coords_var: str,
        var: str,
        step: int | float,
        tor_mode: str = "midplane",
    ) -> Path:
        """One (coords_var, var, tor_mode) radial profile for one step.

        "midplane" keeps the legacy name exactly -- <coords_var>_<var>_
        <step>.npz, the format/path legacy gather_profiles.
        plot_postproc_profs still reads (KNOWN_ISSUES.md #5 promises this
        stays unaffected). Every other mode gets a slug suffix, so
        gathering the same var under two modes (e.g. average alongside
        midplane outer) doesn't overwrite the first.
        """
        stem = f"{coords_var}_{var}"
        if tor_mode != "midplane":
            stem += "_" + tor_mode.strip().replace(" ", "-")
        return self.postproc_dir / f"{stem}_{self.step_str(step)}.npz"

    # --- Poincare artefacts ---

    @property
    def poinc_dir(self) -> Path:
        return self.run_dir / "poinc_dir"

    def poincare_cache(self, step: int | float) -> Path:
        """One step's Poincare cache -- see poincare_cache module for layout.

        One HDF5 file per step, one group per traced field line, so a scan
        can be widened/extended in place. Replaces the four
        poinc_t*_{psi_n,theta,R,Z}.npz files, whose dense
        (n_psi, ang_sample_freq) shape made both impossible.
        """
        return self.poinc_dir / f"poinc_s{self.step_str(step)}.h5"

    def poincare_cache_legacy(self, step: int | float, kind: str) -> Path:
        """A pre-Phase-4b .npz, for reading only. `kind`: psi_n/theta/R/Z.
        Written at poinc_diag.py:211 with the default pad width but read
        back with the sniffed one -- the mismatch this class removes;
        old-code caches at a width other than 6 are simply not findable.
        """
        return self.poinc_dir / f"poinc_t{self.step_str(step)}_{kind}.npz"

    # --- jorek2_four artefacts ---

    @property
    def four_dir(self) -> Path:
        return self.run_dir / "four_dir"

    def four_cache(self, step: int | float) -> Path:
        """One step's Fourier-decomposition cache -- see four_cache module
        for layout. One HDF5 file per step, written whole: no incremental
        "extend" like Poincare's, since a Fourier decomposition of a
        single restart isn't resumable."""
        return self.four_dir / f"four_s{self.step_str(step)}.h5"

    # --- inputs written by the runner ---

    @property
    def real_psi_edge(self) -> Path:
        return self.run_dir / "real_psi_edge.dat"

    @property
    def log(self) -> Path:
        """The main run's log, as opposed to ``log_eq`` (the equilibrium
        stage's -- see ``runner.py``'s module docstring). ``R_axis`` and other
        scalars extracted via :mod:`ashen.logfile` are read from here."""
        return self.run_dir / "log"

    @property
    def figures_dir(self) -> Path:
        """Where plotting output lands. Mirrors the legacy convention of
        saving Poincare and connection-length figures alongside the traces
        that produced them, in ``poinc_dir``, rather than inventing a new
        top-level output folder."""
        return self.poinc_dir

    @property
    def in_eq(self) -> Path:
        return self.run_dir / "in_eq"

    @property
    def in_main(self) -> Path:
        return self.run_dir / "in_main"

    @property
    def in_main_r(self) -> Path:
        return self.run_dir / "in_main_r"

    @property
    def in_bnd(self) -> Path:
        return self.run_dir / "in_bnd"

    @property
    def namelists(self) -> list[Path]:
        """The three namelists the runner edits together."""
        return [self.in_eq, self.in_main, self.in_main_r]

    @property
    def profile_files(self) -> list[str]:
        """Profile files staged into a jorek2_* scratch directory."""
        return ["T_prof.dat", "rho_prof.dat", "ffprime_prof.dat"]
