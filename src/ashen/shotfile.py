"""Loading and validating a shotfile.

A shotfile stays exactly what it always was: a plain Python module of module
-level globals, executed and read back as attributes -- this is what lets a
shotfile compute values inline (``rho_const = n0``), which is why it was kept
in this form rather than moved to a declarative format (see the refactor
plan's scope decisions).

What changes is *loading*. The old ``run_jorek.py`` read each optional
attribute inside a bare ``try/except: pass`` (four of them, `run_jorek.py:76
-108`), which silently absorbed genuine typos along with missing-but-optional
fields. Loading here goes through :class:`ShotParams`: required fields raise
by name if missing (which already catches a typo of a *required* field --
mistype ``eta`` and it simply shows up as missing), and defaults apply where
the old code had them.

A typo of an *optional* field is different: since it has a default, a
misspelled ``freebondary = False`` would otherwise be silently ignored and
``freeboundary`` would quietly stay at its default. The real shotfile
`qa2.1_g2.3/eta1e-3_RE/shotfile.py` also defines `n0` purely as a local
stepping stone for `rho_const = n0` -- exactly the "compute inline" pattern
that's the whole reason shotfiles stayed plain Python rather than becoming
declarative config. So an unrecognised attribute is **not** an error by
default (that would break `n0`); it is only flagged when it is a close
spelling match to a real field name (`difflib`), which is precisely the
"did you mean...?" signal a typo produces without rejecting legitimate scratch
variables.
"""

from __future__ import annotations

import dataclasses
import difflib
import importlib.util
import types
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["ShotParams", "ShotfileError", "load_shotfile"]

#: Fields the old shotfile.py sometimes set that are now supplied elsewhere:
#: shot_folder is always the cwd, template_folder/castor_master_folder come
#: from site.toml. Accepted-and-ignored, with a warning, so old shotfiles
#: keep loading rather than breaking outright.
_DEPRECATED_FIELDS = {"shot_folder", "template_folder", "castor_master_folder"}

_RE_FIELDS = (
    "re_initialize",
    "initial_re_current_fraction",
    "vpar_re_sign",
    "re_adv_fact",
    "Dre_num",
    "Dre_par",
)


class ShotfileError(RuntimeError):
    """Raised for a missing/unknown field, or an unmet method requirement."""


@dataclass
class ShotParams:
    """Validated shotfile parameters. See the module docstring for loading."""

    # --- required -------------------------------------------------------
    qa: float
    g: float
    eta: float
    tstep_n: list
    nstep_n: list
    nout: int
    exe: str
    jobscript: str
    ffprime_method: str
    T_method: str
    rho_method: str
    bnd_method: str

    # --- defaults ---------------------------------------------------------
    extend_bnd: bool = True
    extend_ratio: float = 1.2
    extend_reso: int = 20
    freeboundary: bool = True
    with_refluid: bool = False
    Dre_iso: float = 0.0
    allow_other_starwall: bool = False
    namelist_options: dict = field(default_factory=dict)

    #: Machine suffix for CASTOR3D filenames (`xn_fpol_stor0_<suffix>`, etc).
    #: The old code hardcoded "DIIID" into five filenames across
    #: run_jorek_util.py; every shotfile so far has implicitly meant DIIID,
    #: so that stays the default -- but it is now a real, overridable field
    #: rather than a silent assumption. See boundary.py / profiles.py.
    castor_suffix: str = "DIIID"

    # --- required only for certain methods ---------------------------------
    rho_const: float | None = None
    bnd_file: str | None = None
    castor_params: dict | None = None

    # --- required only if with_refluid ---------------------------------
    re_initialize: int | None = None
    initial_re_current_fraction: float | None = None
    vpar_re_sign: int | None = None
    re_adv_fact: float | None = None
    Dre_num: float | None = None
    Dre_par: float | None = None

    def __post_init__(self) -> None:
        uses_castor = "castor" in (
            self.ffprime_method, self.T_method, self.rho_method, self.bnd_method
        )
        if uses_castor and self.castor_params is None:
            raise ShotfileError(
                "castor_params is required when any of ffprime_method/T_method/"
                "rho_method/bnd_method is 'castor'"
            )
        if uses_castor and not ({"machine", "machine_folder"} & self.castor_params.keys()):
            raise ShotfileError(
                "castor_params needs 'machine' (a subfolder name under site.toml's "
                "castor_root, e.g. 'DIIID_low_pres') or, for backward compatibility, "
                "an explicit absolute 'machine_folder'"
            )
        if self.rho_method == "const" and self.rho_const is None:
            raise ShotfileError("rho_const is required when rho_method='const'")
        if self.bnd_method == "file" and self.bnd_file is None:
            raise ShotfileError("bnd_file is required when bnd_method='file'")

        if self.with_refluid:
            missing = [f for f in _RE_FIELDS if getattr(self, f) is None]
            if missing:
                raise ShotfileError(
                    "with_refluid=True requires: " + ", ".join(missing)
                )
            if "RE" not in self.exe:
                warnings.warn(
                    f"with_refluid=True but 'RE' not in exe ({self.exe!r})",
                    stacklevel=2,
                )


def load_shotfile(path: Path | str) -> ShotParams:
    """Execute a shotfile module and validate it into a :class:`ShotParams`.

    Raises :class:`ShotfileError` naming the offending field for a missing
    required field, an unknown field (the typo-catcher the old code lacked),
    or an unmet cross-field requirement (see :meth:`ShotParams.__post_init__`).
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location("shotfile", path)
    if spec is None or spec.loader is None:
        raise ShotfileError(f"{path}: could not be loaded as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return _from_module(module, source=path)


def _from_module(module: types.ModuleType, source: Path) -> ShotParams:
    known = {f.name for f in fields(ShotParams)}
    provided: dict[str, Any] = {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
    }

    for deprecated in _DEPRECATED_FIELDS & provided.keys():
        warnings.warn(
            f"{source}: {deprecated!r} is no longer used (shot_folder is "
            "always the cwd; template/castor paths come from site.toml) -- "
            "ignoring it",
            stacklevel=2,
        )
        del provided[deprecated]

    # Extra names are allowed (scratch/intermediate variables, e.g. `n0` used
    # only to compute `rho_const = n0`), UNLESS one is a close spelling match
    # to a real field -- that is the typo signal, not "any extra name".
    scratch_candidates = {
        name
        for name in provided
        if name not in known
        and not isinstance(provided[name], (types.ModuleType, types.FunctionType, type))
    }
    likely_typos = {
        name: match[0]
        for name in scratch_candidates
        if (match := difflib.get_close_matches(name, known, n=1, cutoff=0.75))
    }
    if likely_typos:
        detail = ", ".join(f"{k!r} (did you mean {v!r}?)" for k, v in sorted(likely_typos.items()))
        raise ShotfileError(f"{source}: possible typo in shotfile field(s): {detail}")

    args = {name: provided[name] for name in known if name in provided}
    missing = [
        f.name
        for f in fields(ShotParams)
        if f.name not in args
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ShotfileError(f"{source}: missing required field(s): {', '.join(missing)}")

    try:
        return ShotParams(**args)
    except ShotfileError:
        raise
    except TypeError as exc:
        raise ShotfileError(f"{source}: {exc}") from exc
