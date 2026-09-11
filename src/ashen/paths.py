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
    "JOREK_PAD_WIDTHS",
    "PaddingError",
    "RunPaths",
    "read_float",
    "write_float",
    "detect_pad_width",
    "step_name_variants",
    "step_str",
]

#: Only a fallback for synthetic cases. Real runs must sniff the width.
DEFAULT_PAD_WIDTH = 6

#: Step-index widths a JOREK build may use in a filename, preferred first.
#: Mirrors ``rst_file_ind_fmt = (/'(a,i6.6)', '(a,i5.5)'/)``
#: (``communication/mod_import_restart.f90:5``).
#:
#: The two halves of JOREK are not symmetric about this, which is what makes
#: a mismatch so quiet. *Importing* a restart loops over both entries
#: (``mod_import_restart.f90:2686``), so ``jorek08002.h5`` and
#: ``jorek008002.h5`` are equally readable. *Naming an output* takes
#: ``rst_file_ind_fmt(1)`` alone (``step_range_string``,
#: ``exec_commands.f90:1068``) -- one fixed width, whatever the restarts
#: happen to use. So a build reads your run happily and then writes
#: ``..._s008002.dat`` next to ``jorek08002.h5``, and a reader that derived
#: its width from the restart filenames looks for a file that is right there
#: under a name one character longer.
#:
#: Different postproc builds differ in which entry is first, so this is a
#: property of the binary in use, not of the run.
JOREK_PAD_WIDTHS = (6, 5)

_STEP_DIGITS_RE = re.compile(r"\d+")


def step_name_variants(
    name: str, step: int | float, widths: tuple[int, ...] = JOREK_PAD_WIDTHS
) -> list[str]:
    """``name`` with the step index re-padded to each width in ``widths``.

    Only digit runs that *are* the step are touched, so
    ``fluxsurface_at_psi_0.200_s08002.dat`` re-pads the ``08002`` and leaves
    the ``0`` and ``200`` of the psi value alone. A psi value that happened
    to read as the step number would be rewritten too; with psi formatted to
    three decimals and steps in the thousands that cannot arise.

    The original spelling is never included -- these are the *alternatives*
    to whatever the caller already tried.
    """
    target = int(step)
    variants: list[str] = []
    for width in widths:
        padded = step_str(target, width)
        variant = _STEP_DIGITS_RE.sub(
            lambda m: padded if int(m.group()) == target else m.group(), name
        )
        if variant != name and variant not in variants:
            variants.append(variant)
    return variants

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

    def _resolve(self, step: int | float, build) -> Path:
        """The existing file among this step's padding variants, else the
        canonical one.

        Used only for paths **JOREK writes**, whose width comes from the
        postproc binary rather than from this run (see JOREK_PAD_WIDTHS).
        Paths *ashen* writes -- the Poincare, profile and jorek2_four caches
        -- keep one spelling at ``self.pad_width`` and must not go through
        here: they are only ever written and read by this package, so a
        second accepted spelling would be a way to end up with two caches
        for one step rather than a way to find the one that exists.

        Falling back to the canonical name rather than raising keeps this
        usable as a write target and keeps "expected <path>" messages
        predictable when nothing exists yet.
        """
        canonical = build(self.step_str(step))
        if canonical.exists():
            return canonical
        for name in step_name_variants(canonical.name, step):
            alt = canonical.with_name(name)
            if alt.exists():
                return alt
        return canonical

    # --- JOREK outputs ---

    def restart(self, step: int | float, prefix: str = "jorek", ext: str = ".h5") -> Path:
        """One step's restart file.

        Width-tolerant like the rest of the JOREK-written paths, even though
        ``detect_pad_width`` took the majority width from these very files:
        a folder that holds both spellings (a run continued with a different
        build) would otherwise have half its steps unreachable, and JOREK's
        own importer accepts either.
        """
        return self._resolve(
            step, lambda s: self.run_dir / f"{prefix}{s}{ext}"
        )

    @property
    def live_restart(self) -> Path:
        """The file ``jorek2_*`` tools read: always ``jorek_restart.h5``."""
        return self.run_dir / "jorek_restart.h5"

    # --- postprocessing artefacts ---

    @property
    def postproc_dir(self) -> Path:
        return self.run_dir / "postproc"

    def zero_d(self, step: int | float, *, si_units: bool = True) -> Path:
        stem = "zeroD_quantities" if si_units else "zeroD_quantities_jorek"
        return self._resolve(step, lambda s: self.postproc_dir / f"{stem}_s{s}.dat")

    def flux_surface(self, psi_n: float, step: int | float) -> Path:
        return self._resolve(
            step,
            lambda s: self.postproc_dir / f"fluxsurface_at_psi_{psi_n:.3f}_s{s}.dat",
        )

    def qprofile(self, step: int | float) -> Path:
        """One step's ``Psi_n``/``q`` table -- ``exec_commands.f90::qprofile``'s
        own naming for a single-step ``for step`` loop (``loop_min_step ==
        loop_max_step``): ``qprofile_s<step>.dat``, no range suffix."""
        return self._resolve(step, lambda s: self.postproc_dir / f"qprofile_s{s}.dat")

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
        """Where Poincare/connection_length/theta_hist plotting output
        lands. Mirrors the legacy convention of saving Poincare and
        connection-length figures alongside the traces that produced
        them, in ``poinc_dir``, rather than inventing a new top-level
        output folder. Radial profile figures use ``profile_figures_dir``
        instead -- they aren't Poincare-derived, so they don't belong
        under ``poinc_dir``."""
        return self.poinc_dir

    @property
    def profile_figures_dir(self) -> Path:
        """Where radial-profile figures (``plot --diag profiles``) land --
        a top-level ``profiles/`` folder, separate from ``figures_dir``
        since these figures come from ``postproc_dir``'s caches, not a
        Poincare trace."""
        return self.run_dir / "profiles"

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
    def input_starwall(self) -> Path:
        return self.run_dir / "input_starwall"

    @property
    def namelists(self) -> list[Path]:
        """The three namelists the runner edits together."""
        return [self.in_eq, self.in_main, self.in_main_r]

    @property
    def profile_files(self) -> list[str]:
        """Profile files staged into a jorek2_* scratch directory."""
        return ["T_prof.dat", "rho_prof.dat", "ffprime_prof.dat"]
