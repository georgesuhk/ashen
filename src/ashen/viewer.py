"""Looking at a case from a notebook: profiles, boundaries, equilibrium, diagnostics.

Every view is a ``*_figure`` function that returns a matplotlib Figure from
files already in the run folder, plus a thin ipywidgets wrapper around it
(``profile_tuner``, ``boundary_view``, ``equilibrium_view``, ``four_view``,
``profiles_view``). The figure functions need neither a notebook nor
ipywidgets; the wrappers import ipywidgets when called.

Nothing here runs a ``jorek2_*`` tool. A missing cache is reported with the
command that makes it.

The equilibrium view contours psi on the grid nodes, not on JOREK's Bezier
elements: good for looking, not for measuring.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ashen import current_profile as cur
from ashen.castor_io import load_two_col_data
from ashen.diagnostics import four_cache as fc
from ashen.diagnostics.equilibrium import AchievedQLi, achieved_q_li
from ashen.diagnostics.four_modes import max_amplitude_series, radial_amplitude_series
from ashen.diagnostics.profiles import read_profile_series
from ashen.diagnostics.qprofile import read_qprofile
from ashen.namelist import NamelistError, read_boundary_points, read_field
from ashen.padding import PaddingError, restart_steps
from ashen.paths import RunPaths, read_float
from ashen.physics import MU_0
from ashen.shotfile import ShotfileError, load_shotfile, set_shotfile_values

__all__ = [
    "boundary_figure",
    "boundary_view",
    "case_boundaries",
    "equilibrium_figure",
    "equilibrium_view",
    "folder_name_mismatches",
    "four_figure",
    "four_steps",
    "four_view",
    "grid_boundary",
    "plasma_geometry",
    "profile_tuner",
    "profiles_available",
    "profiles_figure",
    "profiles_view",
    "restart_nodes",
    "starwall_wall",
    "tuner_figure",
    "tuner_status",
]

_BLUE, _GREY, _RED, _INK, _ORANGE = "#2a78d6", "#898781", "#c8442f", "#52514e", "#d98a1f"


# --- reading what is in a run folder -------------------------------------------


def _starwall_array(text: str, name: str) -> list[float]:
    match = re.search(rf"^\s*{name}\s*=\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return []
    values = match.group(1).split("!")[0]
    return [float(v.lower().replace("d", "e")) for v in values.replace(",", " ").split()]


def starwall_wall(path: Path | str, n_points: int = 240) -> np.ndarray | None:
    """(R, Z) of the axisymmetric part of STARWALL's Fourier wall
    (``iwall = 1``: rc_w/rs_w/zc_w/zs_w against m_w, for n_w = 0), or None if
    the file does not describe one."""
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m_w, n_w = _starwall_array(text, "m_w"), _starwall_array(text, "n_w")
    terms = {k: _starwall_array(text, k) for k in ("rc_w", "rs_w", "zc_w", "zs_w")}
    if not m_w or not terms["rc_w"]:
        return None
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    R, Z = np.zeros_like(theta), np.zeros_like(theta)
    for i, m in enumerate(m_w):
        if i < len(n_w) and n_w[i] != 0:
            continue

        def term(key: str) -> float:
            return terms[key][i] if i < len(terms[key]) else 0.0

        R += term("rc_w") * np.cos(m * theta) + term("rs_w") * np.sin(m * theta)
        Z += term("zc_w") * np.cos(m * theta) + term("zs_w") * np.sin(m * theta)
    return np.column_stack((R, Z))


def grid_boundary(path: Path | str) -> np.ndarray | None:
    """(R, Z) of the boundary nodes of JOREK's grid, from ``boundary.txt``."""
    path = Path(path)
    if not path.is_file():
        return None
    points = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5 and "." in fields[3]:
            points.append((float(fields[3]), float(fields[4])))
    return np.array(points) if points else None


def case_boundaries(run_dir: Path | str) -> dict[str, np.ndarray]:
    """Every boundary the run folder holds, by name. Missing ones are left out."""
    run_dir = Path(run_dir)
    found: dict[str, np.ndarray] = {}
    if (run_dir / "original_bnd.dat").is_file():
        found["plasma"] = load_two_col_data(run_dir / "original_bnd.dat")
    if (run_dir / "in_bnd").is_file():
        points = read_boundary_points(run_dir / "in_bnd")
        if points:
            found["domain (in_bnd)"] = np.array(points)
    grid = grid_boundary(run_dir / "boundary.txt")
    if grid is not None:
        found["JOREK grid"] = grid
    wall = starwall_wall(run_dir / "input_starwall")
    if wall is not None:
        found["STARWALL wall"] = wall
    return found


def restart_nodes(path: Path | str) -> dict[str, np.ndarray]:
    """R, Z and the axisymmetric node values of psi and zj from a restart."""
    import h5py  # lazy, as elsewhere: the package imports without it

    with h5py.File(path, "r") as f:
        values, x = f["values"][()], f["x"][()]
    return {
        "R": x[0, 0, 0, :], "Z": x[1, 0, 0, :],
        "psi": values[0, 0, 0, :], "zj": values[2, 0, 0, :],
    }


def _paths(run_dir: Path | str) -> RunPaths:
    """RunPaths for a folder that may not have a restart yet (a new run)."""
    try:
        return RunPaths.detect(Path(run_dir))
    except PaddingError:
        return RunPaths(Path(run_dir))


def _real_psi_edge(paths: RunPaths) -> float:
    try:
        return read_float(paths.real_psi_edge)
    except (OSError, ValueError):
        return 1.0


def _f0(run_dir: Path, site=None) -> float:
    """F0 from the run's own in_eq, else the campaign template's."""
    candidates = [run_dir / "in_eq", run_dir / "in_main"]
    if site is not None:
        candidates.append(site.template / "copy" / "in_eq")
    for path in candidates:
        try:
            return float(read_field(path, "F0", float))
        except (NamelistError, OSError, ValueError, TypeError):
            continue
    raise ShotfileError(
        f"F0 not found in {' or '.join(str(p) for p in candidates)}"
    )


def _find_site(run_dir: Path):
    """The campaign's site: the nearest site.toml above the run folder (a
    notebook's own folder is rarely inside the campaign), else whatever
    load_site finds from here. None if there is none."""
    from ashen.config import load_site

    for folder in (run_dir, *run_dir.parents):
        if (folder / "site.toml").is_file():
            try:
                return load_site(folder / "site.toml")
            except Exception:
                return None
    try:
        return load_site(None)
    except Exception:
        return None


def plasma_geometry(run_dir: Path | str, site=None, params=None) -> cur.PlasmaGeometry:
    """R0, a, kappa, B0 of the case's plasma: from ``original_bnd.dat``, or
    the shotfile's plasma boundary file before the run folder is prepared."""
    run_dir = Path(run_dir)
    if (run_dir / "original_bnd.dat").is_file():
        bnd = load_two_col_data(run_dir / "original_bnd.dat")
    elif params is not None and params.bnd_method == "file" and params.bnd_file_is_plasma:
        bnd = load_two_col_data(run_dir / params.bnd_file)
    elif params is not None and params.bnd_method == "template" and (
        (run_dir / params.bnd_file).is_file() or site is not None
    ):
        linked = run_dir / params.bnd_file          # the symlink, once prepared
        if linked.is_file():
            bnd = load_two_col_data(linked)
        else:
            from ashen.runner import boundary_template

            bnd = load_two_col_data(boundary_template(site, params.bnd_file))
    else:
        raise ShotfileError(
            f"{run_dir}: no plasma boundary to take R0, a and kappa from "
            "(no original_bnd.dat, and the shotfile names no plasma boundary file); "
            "run run_jorek on the shotfile once first"
        )
    return cur.boundary_geometry(bnd, _f0(run_dir, site))


_REQUESTED = re.compile(
    r"requested:\s*q0\s*=\s*([-\d.eE+]+),\s*l_i\s*=\s*([-\d.eE+]+),\s*(?:qa|q_edge)\s*=\s*([-\d.eE+]+)"
)


def _same(a, b) -> bool:
    return all(abs(x - y) <= 5e-5 * max(1.0, abs(y)) for x, y in zip(a, b))


#: A value written into a folder name: qa2.1, li1.25, q01.0, between "_" or
#: at either end of a name ("qa2.1_li1.0_q01.0").
_NAME_TOKEN = re.compile(r"(?:^|[_\-])(qa|li|q0)(\d+(?:\.\d+)?)(?=$|[_\-])")
_NAME_LABELS = {"qa": "qa", "li": "l_i", "q0": "q0"}


def folder_name_mismatches(run_dir: Path | str, q0: float, li: float, qa: float) -> list[str]:
    """Where the run folder's name says one thing and the profile another.

    Looks for qa<x>, li<x> and q0<x> in the run folder's name and its
    parent's (``qa2.1_li1.0_q01.0/eta1e-3``). A value agrees with the name
    if it rounds to what is written, to the digits written: li1.0 agrees
    with 1.04, not with 1.06. A name with none of these says nothing.

    The names are only ever checked here, never used: nothing in ashen
    takes a run's settings from its folder name.
    """
    run_dir = Path(run_dir).resolve()
    values = {"qa": qa, "li": li, "q0": q0}
    notes = []
    for folder in (run_dir.parent.name, run_dir.name):
        for key, written in _NAME_TOKEN.findall(folder):
            decimals = len(written.split(".")[1]) if "." in written else 0
            if abs(values[key] - float(written)) > 0.5 * 10.0**-decimals + 1e-9:
                notes.append(
                    f"Folder name says {_NAME_LABELS[key]} = {written} ('{folder}'), "
                    f"but the profile has {_NAME_LABELS[key]} = {values[key]:g}."
                )
    return notes


def tuner_status(run_dir: Path | str, q0: float, li: float, q_edge: float) -> list[str]:
    """What is out of step between the sliders, the shotfile and the run
    folder's input files. Empty when all three agree.

    The input files are judged by the ``requested:`` line run_jorek writes at
    the top of ``j_prof.dat``.
    """
    run_dir = Path(run_dir)
    warnings: list[str] = []
    try:
        params = load_shotfile(run_dir / "shotfile.py")
    except (ShotfileError, OSError) as exc:
        return [f"shotfile.py could not be read: {exc}"]
    saved = (params.current_q0, params.current_li, params.current_qa)
    is_q_li = params.ffprime_method == "q_li" and None not in saved

    if not is_q_li:
        warnings.append(
            f"shotfile.py has ffprime_method = {params.ffprime_method!r}, so these "
            "values are not what the run uses. Save to shotfile switches it to \"q_li\"."
        )
    elif not _same((q0, li, q_edge), saved):
        warnings.append(
            "Sliders differ from shotfile.py "
            f"(q0 = {saved[0]:g}, l_i = {saved[1]:g}, qa = {saved[2]:g}): not saved."
        )

    if is_q_li:
        j_prof = run_dir / "j_prof.dat"
        match = None
        if j_prof.is_file():
            with open(j_prof, encoding="utf-8", errors="replace") as f:
                match = _REQUESTED.search("".join(f.readline() for _ in range(8)))
        if match is None:
            warnings.append(
                "The input files have not been made from this shotfile yet "
                "(no j_prof.dat from \"q_li\"). Regenerate inputs."
            )
        else:
            generated = tuple(float(v) for v in match.groups())
            if not _same(generated, saved):
                warnings.append(
                    "Input files are out of date: j_prof.dat and ffprime_prof.dat were "
                    f"made for q0 = {generated[0]:g}, l_i = {generated[1]:g}, "
                    f"qa = {generated[2]:g}, but shotfile.py now has "
                    f"q0 = {saved[0]:g}, l_i = {saved[1]:g}, qa = {saved[2]:g}. "
                    "Regenerate inputs before running JOREK."
                )
    return warnings


# --- figures -------------------------------------------------------------------


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")


def tuner_figure(
    q0: float,
    li: float,
    q_edge: float,
    geometry: cur.PlasmaGeometry,
    *,
    achieved: AchievedQLi | None = None,
):
    """j, q and FF' of the q0/l_i/q_edge profile; JOREK's q overlaid if given.

    Outside the family's q0 window nothing is drawn but the message.
    """
    plt = _plt()
    fig, (ax_j, ax_q, ax_f) = plt.subplots(1, 3, figsize=(14, 4.2))
    try:
        made = cur.current_from_q_li(q0, li, q_edge, geometry)
        message = (
            f"q0 = {made.q[0]:.3f}   l_i = {made.li:.3f}   qa = {made.q[-1]:.3f}   |   "
            f"j0 = {made.j0 / 1e6:.2f} MA/m²   Ip = {made.Ip / 1e3:.0f} kA   "
            f"alpha = {made.alpha:.2f}   nu = {made.nu:.3g}"
        )
    except ValueError as exc:
        made, message = None, str(exc)

    if made is not None:
        ax_j.plot(made.rho, made.j / 1e6, color=_BLUE, lw=2)
        ax_q.plot(made.rho, made.q, color=_BLUE, lw=2, label="requested (cylinder)")
        ax_f.plot(made.psi_n, cur.ffprime_from_current(made.j, geometry.R0), color=_BLUE, lw=2)
    if achieved is not None:
        inside = achieved.r <= achieved.a * 1.0001
        ax_q.plot(
            achieved.r[inside] / achieved.a, achieved.q[inside], color=_RED, lw=1.6, ls="--",
            label="JOREK equilibrium",
        )
        message += (
            f"\nJOREK:  q0 = {achieved.q0:.3f}   l_i = {achieved.li:.3f}   "
            f"qa = {achieved.q_edge:.3f}   (plasma only; a = {achieved.a:.3f} m)"
        )
        whole = [f"{k} = {achieved.zero_d[k]:.3f}" for k in ("li3", "q95") if k in achieved.zero_d]
        if whole:
            message += "   |   whole domain: " + ", ".join(whole)
    if made is not None or achieved is not None:
        ax_q.legend(frameon=False, labelcolor=_INK, loc="upper left")

    _style(ax_j, "r / a", "j [MA/m²]", "Current density at R0")
    _style(ax_q, "r / a", "q", "Safety factor")
    _style(ax_f, r"$\psi_N$ (plasma)", "FF' [T]", "FF' as JOREK reads it")
    for ax in (ax_j, ax_q):
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
    ax_f.set_xlim(0, 1)
    fig.suptitle(message, x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if achieved is not None else 0.94))
    return fig


_BOUNDARY_STYLE = {
    "plasma": dict(color=_BLUE, lw=2),
    "domain (in_bnd)": dict(color=_RED, lw=1.2, marker="o", ms=3),
    "JOREK grid": dict(color=_INK, lw=1, ls="--"),
    "STARWALL wall": dict(color=_GREY, lw=2),
}


def _draw_boundaries(ax, boundaries: dict[str, np.ndarray], only=None) -> None:
    for name, points in boundaries.items():
        if only is not None and name not in only:
            continue
        closed = np.vstack([points, points[:1]])
        ax.plot(closed[:, 0], closed[:, 1], label=name, **_BOUNDARY_STYLE.get(name, {}))


def boundary_figure(run_dir: Path | str):
    """The run's boundaries on one R-Z plot."""
    plt = _plt()
    boundaries = case_boundaries(run_dir)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _draw_boundaries(ax, boundaries)
    ax.set_aspect("equal")
    _style(ax, "R [m]", "Z [m]", "Boundaries")
    if boundaries:
        ax.legend(frameon=False, labelcolor=_INK, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no boundary files in this folder", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def equilibrium_figure(run_dir: Path | str, step: int = 0, *, levels: int = 14):
    """psi_N contours of one restart with the plasma edge marked, the
    q-profile, and the toroidal current density on the nodes."""
    plt = _plt()
    paths = _paths(run_dir)
    nodes = restart_nodes(paths.restart(step))
    R, Z, psi, zj = nodes["R"], nodes["Z"], nodes["psi"], nodes["zj"]
    axis = int(np.argmax(np.abs(zj)))
    bnd_psi = psi[np.argmax(np.abs(psi - psi[axis]))]
    psi_n = (psi - psi[axis]) / (bnd_psi - psi[axis])
    edge = _real_psi_edge(paths)

    fig = plt.figure(figsize=(14, 6))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.15))
    ax_map = fig.add_subplot(grid[:, 0])
    ax_q = fig.add_subplot(grid[0, 1])
    ax_j = fig.add_subplot(grid[1, 1])

    contours = ax_map.tricontour(R, Z, psi_n, levels=np.linspace(0.0, 1.0, levels + 1)[1:-1],
                                 colors=_GREY, linewidths=0.7)
    del contours
    if edge < 1.0:
        ax_map.tricontour(R, Z, psi_n, levels=[edge], colors=_ORANGE, linewidths=2)
        ax_map.plot([], [], color=_ORANGE, lw=2, label=rf"$\psi_N$ = {edge:.3f} (JOREK's plasma edge)")
    ax_map.plot(R[axis], Z[axis], "+", color=_RED, ms=10, mew=2, label="axis (node)")
    _draw_boundaries(ax_map, case_boundaries(run_dir), only=("plasma", "JOREK grid"))
    ax_map.set_aspect("equal")
    _style(ax_map, "R [m]", "Z [m]", rf"$\psi_N$, step {step}")
    ax_map.legend(frameon=False, labelcolor=_INK, loc="upper right", fontsize=8)

    q_path = paths.qprofile(step)
    if q_path.is_file():
        psi_n_q, q = read_qprofile(q_path)
        ax_q.plot(psi_n_q, np.abs(q), color=_BLUE, lw=2)
    else:
        ax_q.text(0.5, 0.5, f"no q-profile cache for step {step}\n"
                  "(analyse --diag four gathers it)", ha="center", va="center",
                  transform=ax_q.transAxes, color=_INK)
    _style(ax_q, r"$\psi_N$ (JOREK grid)", "q", "Safety factor")

    j_phi = -zj / (MU_0 * R) / 1e6
    ax_j.scatter(psi_n, j_phi, s=3, color=_BLUE, alpha=0.5, linewidths=0)
    _style(ax_j, r"$\psi_N$ (JOREK grid)", r"$j_\phi$ [MA/m²]", "Toroidal current density on the nodes")
    for ax in (ax_q, ax_j):
        ax.set_xlim(0, 1)
        if edge < 1.0:
            ax.axvline(edge, color=_GREY, lw=1, ls="--")
    fig.tight_layout()
    return fig


def four_steps(run_dir: Path | str) -> list[int]:
    """Steps that have a jorek2_four cache."""
    paths = _paths(run_dir)
    found = []
    for path in paths.four_dir.glob("four_s*.h5"):
        match = re.fullmatch(r"four_s(\d+)\.h5", path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def four_figure(
    run_dir: Path | str,
    variable: str = "Psi",
    *,
    step: int | None = None,
    modes: list[tuple[int, int]] | None = None,
    n_modes: int = 6,
    log: bool = True,
):
    """Mode amplitudes against step, and the radial eigenfunctions at one step.

    ``modes`` is a list of (n, m); None draws the ``n_modes`` largest.
    """
    from ashen.plotting.four_modes import draw_mode_amplitudes

    plt = _plt()
    paths = _paths(run_dir)
    steps = four_steps(run_dir)
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(14, 4.6))
    if not steps:
        ax_t.text(0.5, 0.5, "no jorek2_four cache (analyse --diag four)", ha="center",
                  transform=ax_t.transAxes)
        return fig
    step = steps[-1] if step is None else step

    series = max_amplitude_series(paths, steps, variables=[variable], modes=modes)
    if modes is None:
        biggest = sorted(series, key=lambda k: -np.nanmax(series[k]))[:n_modes]
        series = {k: series[k] for k in biggest}
    draw_mode_amplitudes(ax_t, steps, series, variable=variable, log=log, xlabel="Time step")
    ax_t.axvline(step, color=_GREY, lw=1, ls="--")
    ax_t.set_title(f"max |{variable}| per mode", loc="left", fontsize=11, fontweight="bold")

    radial = radial_amplitude_series(
        paths, [step], variables=[variable], modes=[(n, m) for _, n, m in series]
    )
    # same colour per mode as the left panel
    left = {line.get_label(): line.get_color() for line in ax_t.get_lines()}
    for (_, n, m), curves in sorted(radial.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        psi_n, values = curves[step]
        label = f"n={n}, m={m}"
        ax_r.plot(psi_n, values, lw=1.6, label=label, color=left.get(label))
    if log:
        ax_r.set_yscale("log")
    _style(ax_r, r"$\psi_N$ (JOREK grid)", f"|{variable}|", f"Radial structure, step {step}")
    if radial:
        ax_r.legend(frameon=False, labelcolor=_INK, fontsize=8)
    edge = _real_psi_edge(paths)
    if edge < 1.0:
        ax_r.axvline(edge, color=_GREY, lw=1, ls="--")
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class ProfileKey:
    coords_var: str
    var: str
    tor_mode: str


def profiles_available(run_dir: Path | str) -> dict[ProfileKey, list[int]]:
    """The cached radial profiles in ``postproc/``, with their steps."""
    paths = _paths(run_dir)
    modes = {"midplane-outer": "midplane outer", "midplane-inner": "midplane inner",
             "average": "average"}
    found: dict[ProfileKey, list[int]] = {}
    for path in sorted(paths.postproc_dir.glob("*.npz")):
        match = re.fullmatch(r"(.+)_(\d+)", path.stem)
        if not match:
            continue
        stem, step = match.group(1), int(match.group(2))
        tor_mode = "midplane"
        for slug, name in modes.items():
            if stem.endswith("_" + slug):
                stem, tor_mode = stem[: -len(slug) - 1], name
                break
        for coords_var in ("Psi_N", "R"):
            if stem.startswith(coords_var + "_"):
                key = ProfileKey(coords_var, stem[len(coords_var) + 1:], tor_mode)
                found.setdefault(key, []).append(step)
                break
    return {key: sorted(steps) for key, steps in found.items()}


def profiles_figure(run_dir: Path | str, key: ProfileKey, *, step: int | None = None):
    """One cached profile variable at every gathered step, coloured by step;
    ``step`` picks one to draw in bold."""
    from ashen.plotting.profiles import draw_profile_family

    plt = _plt()
    paths = _paths(run_dir)
    steps = profiles_available(run_dir).get(key, [])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    series = read_profile_series(paths, steps, key.coords_var, key.var, key.tor_mode)
    if not series:
        ax.text(0.5, 0.5, "no cached profile (analyse --diag profiles)", ha="center",
                transform=ax.transAxes)
        return fig
    colourer = draw_profile_family(
        ax, series, xlabel=key.coords_var, ylabel=key.var, title=f"{key.var} ({key.tor_mode})"
    )
    del colourer
    if step in series:
        x, y = series[step]
        ax.plot(x, y, color="black", lw=2.2, label=f"step {step}")
        ax.legend(frameon=False, labelcolor=_INK)
    fig.tight_layout()
    return fig


# --- notebook wrappers ----------------------------------------------------------


def _widgets():
    try:
        import ipywidgets
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the viewer's interactive views need ipywidgets "
            "(pip install --user ipywidgets ipykernel)"
        ) from exc
    return ipywidgets


def _show(out, make_figure) -> None:
    """Draw ``make_figure()`` into an Output widget, replacing what was there."""
    from IPython.display import display

    plt = _plt()
    with out:
        out.clear_output(wait=True)
        try:
            fig = make_figure()
        except Exception as exc:  # a view should say what is missing, not die
            print(f"{type(exc).__name__}: {exc}")
            return
        display(fig)
        plt.close(fig)


def _slider(w, value, lo, hi, step, name):
    # One per row at a comfortable width: three abreast leaves each too short
    # to set a value to two decimals with the mouse.
    return w.FloatSlider(value=value, min=lo, max=hi, step=step, description=name,
                         continuous_update=False, readout_format=".2f",
                         layout=w.Layout(width="min(720px, 95%)", height="36px"),
                         style={"description_width": "70px"})


def profile_tuner(run_dir: Path | str = ".", *, step: int = 0, site=None):
    """Sliders for q0, l_i and qa (edge q) of the case's shotfile.

    **Save to shotfile** writes the three values (and ``ffprime_method =
    "q_li"``) into ``shotfile.py``. **Reset to shotfile** puts the sliders back
    to the values saved there. **Regenerate inputs** then prepares the run
    folder from the shotfile, as ``run_jorek shotfile.py`` does, without
    submitting anything. A yellow band says when the sliders are not saved,
    or the input files were made for other values (tuner_status); a red one
    when the folder's name gives other values (folder_name_mismatches). If the equilibrium at ``step`` has q-profile and
    zeroD caches, what JOREK achieved is drawn next to what is requested.
    """
    w = _widgets()
    run_dir = Path(run_dir).resolve()
    shotfile = run_dir / "shotfile.py"
    params = load_shotfile(shotfile)
    if site is None:
        site = _find_site(run_dir)
    geometry = plasma_geometry(run_dir, site, params)
    paths = _paths(run_dir)
    try:
        achieved = achieved_q_li(paths, step, f0=geometry.B0 * geometry.R0)
    except Exception:
        achieved = None

    start = (
        params.current_q0 or (achieved.q0 if achieved else 1.0),
        params.current_li or (achieved.li if achieved else 1.2),
        params.current_qa or (achieved.q_edge if achieved else 3.0),
    )
    q0 = _slider(w, start[0], 0.3, 4.0, 0.01, "q0")
    li = _slider(w, start[1], 0.55, 2.5, 0.01, "l_i")
    q_edge = _slider(w, start[2], 1.5, 10.0, 0.05, "qa")
    save = w.Button(description="Save to shotfile", button_style="primary")
    regenerate = w.Button(description="Regenerate inputs")
    reset = w.Button(description="Reset to shotfile", tooltip="Put the sliders back to shotfile.py's values")
    figure_out, log_out = w.Output(), w.Output()
    status = w.HTML()

    def refresh_status():
        # red: the folder is named for another profile; yellow: things not yet in step
        wrong_name = folder_name_mismatches(run_dir, q0.value, li.value, q_edge.value)
        notes = tuner_status(run_dir, q0.value, li.value, q_edge.value)
        status.value = "".join(
            '<div style="background:#f8d7da;border-left:4px solid #c8442f;color:#7a1f14;'
            f'padding:6px 10px;margin:3px 0;font-weight:600">&#9888; {html.escape(note)}</div>'
            for note in wrong_name
        ) + "".join(
            '<div style="background:#fff3cd;border-left:4px solid #d98a1f;color:#52514e;'
            f'padding:6px 10px;margin:3px 0">&#9888; {html.escape(note)}</div>'
            for note in notes
        )

    def redraw(*_):
        refresh_status()
        _show(figure_out, lambda: tuner_figure(
            q0.value, li.value, q_edge.value, geometry, achieved=achieved))

    def on_save(_):
        with log_out:
            log_out.clear_output()
            try:
                cur.solve_shape(q0.value, li.value, q_edge.value)
                set_shotfile_values(shotfile, {
                    "ffprime_method": "q_li",
                    "current_q0": round(q0.value, 4),
                    "current_li": round(li.value, 4),
                    "current_qa": round(q_edge.value, 4),
                })
            except (ShotfileError, ValueError) as exc:
                print(f"not saved: {exc}")
                return
            print(f"saved to {shotfile}: q0 = {q0.value:.4g}, l_i = {li.value:.4g}, "
                  f"qa = {q_edge.value:.4g}.")
        refresh_status()

    def on_regenerate(_):
        with log_out:
            log_out.clear_output()
            if site is None:
                print("no site.toml found from here; run `run_jorek shotfile.py` in the run folder")
                return
            try:
                from ashen.runner import prepare_run

                result = prepare_run(load_shotfile(shotfile), site, run_dir)
            except Exception as exc:
                print(f"not regenerated: {type(exc).__name__}: {exc}")
                return
            print(f"run folder prepared ({len(result.actions)} steps): j_prof.dat, "
                  "ffprime_prof.dat, T/rho profiles, boundary and namelists rewritten.\n"
                  "Next, in the run folder:  run_jorek shotfile.py --run_eq")
        refresh_status()

    def on_reset(_):
        with log_out:
            log_out.clear_output()
            try:
                now = load_shotfile(shotfile)
            except ShotfileError as exc:
                print(f"not reset: {exc}")
                return
            saved = (now.current_q0, now.current_li, now.current_qa)
            if None in saved:
                print("not reset: shotfile.py has no current_q0, current_li and current_qa yet")
                return
            for slider, value in zip((q0, li, q_edge), saved):
                # a value outside the slider's travel would be clipped silently
                slider.min, slider.max = min(slider.min, value), max(slider.max, value)
                slider.value = value
            print(f"sliders back to shotfile.py: q0 = {saved[0]:g}, l_i = {saved[1]:g}, "
                  f"qa = {saved[2]:g}")
        refresh_status()

    for slider in (q0, li, q_edge):
        slider.observe(redraw, names="value")
    save.on_click(on_save)
    regenerate.on_click(on_regenerate)
    reset.on_click(on_reset)
    redraw()
    return w.VBox([w.VBox([q0, li, q_edge]), w.HBox([save, regenerate, reset]), status, log_out,
                   figure_out])


def boundary_view(run_dir: Path | str = "."):
    """The run's boundaries (no controls)."""
    w = _widgets()
    out = w.Output()
    _show(out, lambda: boundary_figure(run_dir))
    return out


def equilibrium_view(run_dir: Path | str = "."):
    """psi_N map, q and j of a restart, with a step selector."""
    w = _widgets()
    steps = restart_steps(Path(run_dir))
    if not steps:
        return w.HTML(f"no restart files in {Path(run_dir).resolve()}")
    step = w.SelectionSlider(options=steps, value=steps[0], description="step",
                             continuous_update=False)
    out = w.Output()
    step.observe(lambda _: _show(out, lambda: equilibrium_figure(run_dir, step.value)),
                 names="value")
    _show(out, lambda: equilibrium_figure(run_dir, step.value))
    return w.VBox([step, out])


def four_view(run_dir: Path | str = "."):
    """Mode amplitudes and radial structure, with variable and step selectors."""
    w = _widgets()
    steps = four_steps(run_dir)
    if not steps:
        return w.HTML("no jorek2_four cache (run analyse --diag four)")
    paths = _paths(run_dir)
    variables = sorted({key[0] for key in fc.read_cache(paths.four_cache(steps[-1]))})
    variable = w.Dropdown(options=variables,
                          value="Psi" if "Psi" in variables else variables[0],
                          description="variable")
    step = w.SelectionSlider(options=steps, value=steps[-1], description="step",
                             continuous_update=False)
    n_modes = w.IntSlider(value=6, min=1, max=12, description="modes", continuous_update=False)
    log = w.Checkbox(value=True, description="log scale")
    out = w.Output()

    def redraw(*_):
        _show(out, lambda: four_figure(run_dir, variable.value, step=step.value,
                                       n_modes=n_modes.value, log=log.value))

    for control in (variable, step, n_modes, log):
        control.observe(redraw, names="value")
    redraw()
    return w.VBox([w.HBox([variable, step, n_modes, log]), out])


def profiles_view(run_dir: Path | str = "."):
    """Cached radial profiles, with variable and step selectors."""
    w = _widgets()
    available = profiles_available(run_dir)
    if not available:
        return w.HTML("no cached profiles (run analyse --diag profiles)")
    labels = {f"{k.var} vs {k.coords_var} ({k.tor_mode})": k for k in available}
    which = w.Dropdown(options=list(labels), description="profile",
                       layout=w.Layout(width="420px"))
    first = available[labels[which.value]]
    step = w.SelectionSlider(options=first, value=first[-1], description="step",
                             continuous_update=False)
    out = w.Output()

    def redraw(*_):
        _show(out, lambda: profiles_figure(run_dir, labels[which.value], step=step.value))

    def on_which(_):
        steps = available[labels[which.value]]
        step.options = steps
        step.value = steps[-1]
        redraw()

    which.observe(on_which, names="value")
    step.observe(redraw, names="value")
    redraw()
    return w.VBox([w.HBox([which, step]), out])
