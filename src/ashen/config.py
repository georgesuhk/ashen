"""Site configuration discovery.

Every machine-specific path lives in a single ``site.toml``.  Nothing else in
Ashen may contain an absolute path, and no module may touch ``sys.path``.

The file is located by, in order:

1. ``$ASHEN_SITE``
2. the nearest ``site.toml`` walking up from the current directory
3. ``~/.config/ashen/site.toml``

Relative entries resolve against **the directory containing site.toml**, not
against the current directory.  That is what makes a campaign self-describing:
drop ``site.toml`` at the campaign root with ``exe = "./exe"`` and the tree can
be renamed, moved or cloned elsewhere without editing anything.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

SITE_FILENAME = "site.toml"
ENV_VAR = "ASHEN_SITE"

#: Path keys every site.toml must define.
REQUIRED_PATHS = ("exe", "template", "jobscripts", "jorek", "jorek_re", "castor_root")


class SiteConfigError(RuntimeError):
    """Raised for a missing, malformed or incomplete site.toml."""


@dataclass(frozen=True)
class Launch:
    """How to invoke MPI and the environment modules on this machine.

    Lifts the shell preludes out of the driver, where they were previously
    embedded in five near-identical heredocs.
    """

    interactive_prelude: str = ""
    batch_prelude: str = ""
    mpirun: str = "mpirun -n {n}"
    n_jorek: int = 4
    n_starwall: int = 32

    def mpirun_cmd(self, n: int) -> str:
        """Render the MPI launch prefix for ``n`` ranks."""
        return self.mpirun.format(n=n)


@dataclass(frozen=True)
class Site:
    """A resolved site configuration.

    ``paths`` values are absolute but are *not* required to exist -- a laptop
    clone legitimately lacks most of them.  Use :meth:`missing` to report on
    that explicitly rather than failing at load time.
    """

    source: Path
    root: Path
    paths: dict[str, Path]
    launch: Launch

    def path(self, key: str) -> Path:
        try:
            return self.paths[key]
        except KeyError:
            known = ", ".join(sorted(self.paths)) or "(none)"
            raise SiteConfigError(
                f"{self.source}: no path named {key!r}. Defined: {known}"
            ) from None

    @property
    def exe(self) -> Path:
        return self.path("exe")

    @property
    def template(self) -> Path:
        return self.path("template")

    @property
    def jobscripts(self) -> Path:
        return self.path("jobscripts")

    @property
    def jorek(self) -> Path:
        return self.path("jorek")

    @property
    def jorek_re(self) -> Path:
        return self.path("jorek_re")

    @property
    def castor_root(self) -> Path:
        return self.path("castor_root")

    def jorek_util(self, with_refluid: bool) -> Path:
        """The JOREK ``util/`` tree to link into a run folder.

        Mirrors the RE/standard branch at ``run_jorek.py:154-159``.
        """
        base = self.jorek_re if with_refluid else self.jorek
        return base / "util"

    def missing(self) -> list[str]:
        """Path keys whose targets do not exist on this machine."""
        return sorted(k for k, v in self.paths.items() if not v.exists())

    def describe(self) -> str:
        """Human-readable dump backing ``--show-config``."""
        width = max((len(k) for k in self.paths), default=0)
        lines = [
            f"site.toml : {self.source}",
            f"root      : {self.root}",
            "",
            "[paths]",
        ]
        for key in sorted(self.paths):
            target = self.paths[key]
            mark = "  " if target.exists() else "  (missing)"
            lines.append(f"  {key:<{width}} = {target}{mark}")
        lines += [
            "",
            "[launch]",
            f"  mpirun              = {self.launch.mpirun}",
            f"  n_jorek             = {self.launch.n_jorek}",
            f"  n_starwall          = {self.launch.n_starwall}",
            f"  interactive_prelude = {self.launch.interactive_prelude!r}",
            f"  batch_prelude       = {self.launch.batch_prelude!r}",
        ]
        return "\n".join(lines)


def find_site_file(start: Path | str | None = None) -> Path:
    """Locate ``site.toml``. See the module docstring for the search order."""
    env = os.environ.get(ENV_VAR)
    if env:
        candidate = Path(env).expanduser()
        if not candidate.is_file():
            raise SiteConfigError(
                f"${ENV_VAR} points at {candidate}, which is not a file."
            )
        return candidate.resolve()

    origin = Path(start) if start is not None else Path.cwd()
    origin = origin.resolve()
    searched: list[Path] = []
    for directory in (origin, *origin.parents):
        candidate = directory / SITE_FILENAME
        searched.append(candidate)
        if candidate.is_file():
            return candidate.resolve()

    fallback = Path.home() / ".config" / "ashen" / SITE_FILENAME
    searched.append(fallback)
    if fallback.is_file():
        return fallback.resolve()

    trail = "\n  ".join(str(p) for p in searched)
    raise SiteConfigError(
        f"No {SITE_FILENAME} found. Set ${ENV_VAR}, or create one at the campaign "
        f"root. Searched:\n  {trail}"
    )


def _resolve_path(root: Path, value: object, key: str, source: Path) -> Path:
    if not isinstance(value, str):
        raise SiteConfigError(
            f"{source}: paths.{key} must be a string, got {type(value).__name__}."
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    # Do not require existence: a laptop clone legitimately lacks most targets.
    return Path(os.path.normpath(candidate))


def load_site(
    path: Path | str | None = None,
    *,
    start: Path | str | None = None,
) -> Site:
    """Load and resolve a site configuration.

    Args:
        path: an explicit ``site.toml``. When omitted it is discovered.
        start: directory to begin the upward search from. Defaults to the cwd.
    """
    source = Path(path).resolve() if path is not None else find_site_file(start)

    try:
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SiteConfigError(f"{source}: malformed TOML -- {exc}") from exc
    except OSError as exc:
        raise SiteConfigError(f"{source}: cannot be read -- {exc}") from exc

    unknown_sections = sorted(set(data) - {"paths", "launch"})
    if unknown_sections:
        raise SiteConfigError(
            f"{source}: unknown section(s) {', '.join(unknown_sections)}. "
            "Expected [paths] and [launch]."
        )

    raw_paths = data.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise SiteConfigError(f"{source}: [paths] must be a table.")

    missing = [key for key in REQUIRED_PATHS if key not in raw_paths]
    if missing:
        raise SiteConfigError(
            f"{source}: missing required path key(s): {', '.join(missing)}. "
            f"Required: {', '.join(REQUIRED_PATHS)}."
        )

    root = source.parent
    paths = {
        key: _resolve_path(root, value, key, source)
        for key, value in raw_paths.items()
    }

    raw_launch = data.get("launch", {})
    if not isinstance(raw_launch, dict):
        raise SiteConfigError(f"{source}: [launch] must be a table.")
    allowed = set(Launch.__dataclass_fields__)
    unknown = sorted(set(raw_launch) - allowed)
    if unknown:
        raise SiteConfigError(
            f"{source}: unknown launch key(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(allowed))}."
        )
    launch = Launch(**raw_launch)

    return Site(source=source, root=root, paths=paths, launch=launch)
