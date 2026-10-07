"""Loading and validating a shotfile.

A shotfile stays exactly what it always was: a plain Python module of
module-level globals, executed and read back as attributes -- this is what
lets a shotfile compute values inline (rho_const = n0), the reason it was
kept in this form rather than moved to declarative config (refactor plan's
scope decisions).

What changes is loading. Old run_jorek.py read each optional attribute
inside a bare try/except: pass (4x, run_jorek.py:76-108), silently
absorbing genuine typos along with missing-but-optional fields. Loading
here goes through ShotParams: required fields raise by name if missing
(already catches a typo of a required field -- mistype `eta` and it shows
up as missing), defaults apply where old code had them.

A typo of an optional field is different: since it has a default, a
misspelled `freebondary = False` would otherwise silently leave
`freeboundary` at its default. The real shotfile
qa2.1_g2.3/eta1e-3_RE/shotfile.py also defines `n0` purely as a local
stepping stone for `rho_const = n0` -- exactly the "compute inline"
pattern that's the whole reason shotfiles stayed plain Python. So an
unrecognised attribute is NOT an error by default (that would break `n0`);
it's flagged only when it's a close spelling match to a real field name
(difflib) -- the "did you mean...?" signal a typo produces without
rejecting legitimate scratch variables.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import importlib.util
import types
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["ShotParams", "ShotfileError", "load_shotfile", "set_shotfile_values",
           "shotfile_text_with"]

#: Fields the old shotfile.py sometimes set that are now supplied elsewhere:
#: shot_folder is always the cwd, template_folder/castor_master_folder come
#: from site.toml. Accepted-and-ignored, with a warning, so old shotfiles
#: keep loading rather than breaking outright.
_DEPRECATED_FIELDS = {"shot_folder", "template_folder", "castor_master_folder"}

#: Fields that were renamed: old name -> new name. An old one is an error
#: that names the new one, not a silent alias.
_RENAMED_FIELDS = {"current_q_edge": "current_qa"}

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

    # --- required unless bnd_method = "template" ---------------------------
    #: CASTOR3D's edge q and peaking parameter. They name the STARWALL
    #: response and the batch job. A run on a campaign boundary
    #: (bnd_method="template") is named by that boundary instead
    #: (runner.starwall_response_name) and needs neither.
    qa: float | None = None
    g: float | None = None

    # --- defaults ---------------------------------------------------------
    extend_bnd: bool = True
    extend_ratio: float = 1.2
    extend_reso: int = 20
    freeboundary: bool = True
    with_refluid: bool = False
    Dre_iso: float = 0.0
    allow_other_starwall: bool = False
    namelist_options: dict = field(default_factory=dict)
    #: Fields to set in input_starwall's namelist /params/ or /params_wall/.
    #: Validated against STARWALL's own source -- see runner._validate_starwall_options.
    starwall_options: dict = field(default_factory=dict)

    #: Machine suffix for CASTOR3D filenames (xn_fpol_stor0_<suffix>, etc).
    #: Old code hardcoded "DIIID" into 5 filenames across run_jorek_util.py;
    #: every shotfile so far has implicitly meant DIIID, so that stays the
    #: default -- now a real, overridable field, not a silent assumption.
    #: See boundary.py / profiles.py.
    castor_suffix: str = "DIIID"

    #: Filename under the run's exe/ that `analyse --diag four` runs in place
    #: of the default jorek2_four -- e.g. one built against a different
    #: model. Read only by analyse; prepare_run ignores it. None = default.
    four_exe: str | None = None

    # --- required only for certain methods ---------------------------------
    rho_const: float | None = None
    bnd_file: str | None = None
    castor_params: dict | None = None

    #: ffprime_method="current": a two-column file in the run folder, x and
    #: j_phi at R = current_R0 [A/m^2]. current_coord says what x is:
    #: "psi_n", or "rho" (r/a, mapped as a circular cylinder). See
    #: ashen.current_profile.
    current_file: str | None = None
    current_coord: str = "psi_n"
    #: Major radius the current is referred to [m]. None: the centre of the
    #: plasma boundary.
    current_R0: float | None = None

    #: ffprime_method="q_li": the current profile is built from these three
    #: (ashen.current_profile.current_from_q_li) and written to j_prof.dat.
    #: current_qa is q at the plasma edge (called q_edge in the model).
    current_q0: float | None = None
    current_li: float | None = None
    current_qa: float | None = None

    #: T_method="const": Te + Ti [eV], flat.
    T_const: float | None = None
    #: T_method="file": a two-column file in the run folder, psi_N of the
    #: plasma (0 at the axis, 1 at its edge) and Te + Ti [eV].
    T_file: str | None = None

    #: bnd_method="template": bnd_file names a plasma boundary shared by the
    #: campaign, template/symlink/boundary/<bnd_file>. The run folder gets a
    #: symlink to it, and the STARWALL response is named after it.
    #: bnd_method="file": True means bnd_file is the plasma boundary, to be
    #: expanded by extend_ratio like a CASTOR3D one when extend_bnd is on.
    #: False (as before) uses bnd_file as the domain boundary as it stands.
    bnd_file_is_plasma: bool = False
    #: psi on the boundary when no CASTOR3D psi supplies it.
    psi_bnd: float = 0.0

    # --- required only if with_refluid ---------------------------------
    re_initialize: int | None = None
    initial_re_current_fraction: float | None = None
    vpar_re_sign: int | None = None
    re_adv_fact: float | None = None
    Dre_num: float | None = None
    Dre_par: float | None = None

    def __post_init__(self) -> None:
        if self.bnd_method != "template":
            missing = [name for name in ("qa", "g") if getattr(self, name) is None]
            if missing:
                raise ShotfileError("missing required field(s): " + ", ".join(missing))
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
        if self.bnd_method in ("file", "template") and self.bnd_file is None:
            raise ShotfileError(f"bnd_file is required when bnd_method={self.bnd_method!r}")
        if self.bnd_method == "template" and Path(self.bnd_file).name != self.bnd_file:
            raise ShotfileError(
                "bnd_method='template' takes a file name in the template's "
                f"symlink/boundary/ folder, not a path: {self.bnd_file!r}"
            )
        if self.ffprime_method == "q_li":
            missing = [
                f for f in ("current_q0", "current_li", "current_qa")
                if getattr(self, f) is None
            ]
            if missing:
                raise ShotfileError("ffprime_method='q_li' requires: " + ", ".join(missing))
        if self.ffprime_method == "current":
            if self.current_file is None:
                raise ShotfileError("ffprime_method='current' requires: current_file")
            if self.current_coord not in ("psi_n", "rho"):
                raise ShotfileError(
                    f"current_coord={self.current_coord!r}; expected 'psi_n' or 'rho'"
                )
        if self.T_method == "const" and self.T_const is None:
            raise ShotfileError("T_const is required when T_method='const'")
        if self.T_method == "file" and self.T_file is None:
            raise ShotfileError("T_file is required when T_method='file'")

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
    """Execute a shotfile module and validate it into a ShotParams.

    Raises ShotfileError naming the offending field for a missing required
    field, an unknown field (the typo-catcher old code lacked), or an
    unmet cross-field requirement (ShotParams.__post_init__).
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

    renamed = sorted(_RENAMED_FIELDS.keys() & provided.keys())
    if renamed:
        detail = ", ".join(f"{old} is now {_RENAMED_FIELDS[old]}" for old in renamed)
        raise ShotfileError(f"{source}: renamed shotfile field(s): {detail}")

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


def _is_literal(source: str) -> bool:
    try:
        ast.literal_eval(source)
    except (ValueError, SyntaxError):
        return False
    return True


def shotfile_text_with(text: str, values: dict[str, Any], source: Path | str = "shotfile") -> str:
    """``text`` (a shotfile's source) with module-level ``name = value`` lines set.

    A name already assigned on one line to a plain literal has that line
    replaced (a trailing comment is kept); a name not assigned at all is
    appended. Anything else -- a computed right-hand side such as
    ``rho_const = n0``, a name assigned twice, a statement spanning lines --
    raises ShotfileError, so a hand-written expression is never overwritten.
    Values are written with ``repr``. ``source`` only names the file in errors.
    """
    path = source
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ShotfileError(f"{path}: not valid Python ({exc})") from exc
    lines = text.splitlines()

    assigned: dict[str, list[ast.Assign]] = {name: [] for name in values}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in assigned:
                    assigned[target.id].append(node)

    appended: list[str] = []
    for name, value in values.items():
        nodes = assigned[name]
        if not nodes:
            appended.append(f"{name} = {value!r}")
            continue
        node = nodes[0]
        rhs = ast.get_source_segment(text, node.value) or ""
        if len(nodes) > 1 or len(node.targets) != 1 or node.lineno != node.end_lineno:
            raise ShotfileError(
                f"{path}: {name} is not a single one-line assignment; edit it by hand"
            )
        if not _is_literal(rhs):
            raise ShotfileError(
                f"{path}: {name} = {rhs} is computed, not a plain value; edit it by hand"
            )
        line = lines[node.lineno - 1]
        tail = line[node.end_col_offset:]            # spaces and any comment
        lines[node.lineno - 1] = f"{line[:node.value.col_offset]}{value!r}{tail}"

    if appended:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(appended)
    return "\n".join(lines) + "\n"


def set_shotfile_values(path: Path | str, values: dict[str, Any]) -> None:
    """Set module-level ``name = value`` lines in a shotfile, in place.

    See shotfile_text_with for what is replaced, appended or refused. On a
    refusal the file is left untouched.
    """
    path = Path(path)
    new_text = shotfile_text_with(path.read_text(encoding="utf-8"), values, source=path)
    path.write_text(new_text, encoding="utf-8")
