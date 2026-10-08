"""Looking at a case from a notebook: profiles, boundaries, equilibrium, diagnostics.

Every view is a ``*_figure`` function that returns a matplotlib Figure from
files already in the run folder, plus a thin ipywidgets wrapper around it
(``profile_tuner``, ``boundary_view``, ``equilibrium_view``, ``four_view``,
``profiles_view``). The figure functions need neither a notebook nor
ipywidgets; the wrappers import ipywidgets when called. ``case_viewer`` is
all the views on one page, and is the one call a run folder's notebook makes:
a view added to it reaches every notebook already written.

Drawing never runs a ``jorek2_*`` tool. Gathering is done only by the
views' own buttons: ``run_analyse`` (analyse for this run) and
``gather_step_caches`` (one step's q-profile and zeroD).

The equilibrium view contours psi on the grid nodes, not on JOREK's Bezier
elements: good for looking, not for measuring.
"""

from __future__ import annotations

import html
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ashen import current_profile as cur
from ashen.castor_io import load_two_col_data
from ashen.diagnostics import four_cache as fc
from ashen.diagnostics.equilibrium import AchievedQLi, achieved_q_li
from ashen.diagnostics.four_modes import max_amplitude_series, radial_amplitude_series
from ashen.diagnostics.profiles import read_profile_series
from ashen.diagnostics.qprofile import find_rational_surfaces, read_qprofile
from ashen.namelist import NamelistError, read_boundary_points, read_field
from ashen.padding import PaddingError, restart_steps
from ashen.paths import RunPaths, read_float
from ashen.physics import MU_0
from ashen.shotfile import ShotfileError, load_shotfile, set_shotfile_values

__all__ = [
    "analyse_command",
    "boundary_figure",
    "boundary_view",
    "case_boundaries",
    "case_viewer",
    "default_four_modes",
    "equilibrium_figure",
    "equilibrium_view",
    "folder_name_mismatches",
    "four_figure",
    "four_steps",
    "four_view",
    "four_view_modes",
    "gather_step_caches",
    "grid_boundary",
    "input_profile",
    "plasma_geometry",
    "profile_tuner",
    "profiles_available",
    "profiles_figure",
    "profiles_view",
    "restart_nodes",
    "run_analyse",
    "shotfile_qa",
    "starwall_wall",
    "tuner_figure",
    "tuner_status",
    "FourData",
    "FourPlot",
    "load_four_data",
]

_FIGURE_DPI = 144   # 1.5 pixels per CSS pixel: sharp on a laptop screen
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


def input_profile(run_dir: Path | str, geometry: cur.PlasmaGeometry) -> cur.QLiProfile | None:
    """The cylinder picture (j, q, l_i) of the FF' profile the run folder
    holds now, ``ffprime_prof.dat``, whatever made it: CASTOR3D, a current
    file or q0/l_i/qa. None if there is no such file, or it carries no
    current. See current_profile.profile_from_ffprime for what the picture
    leaves out."""
    run_dir = Path(run_dir)
    path = run_dir / "ffprime_prof.dat"
    if not path.is_file():
        return None
    try:
        data = np.loadtxt(path)
        edge = _real_psi_edge(_paths(run_dir))
        plasma = data[:, 0] <= edge * (1.0 + 1e-9)
        return cur.profile_from_ffprime(data[plasma, 0] / edge, data[plasma, 1], geometry)
    except (ValueError, IndexError, OSError):
        return None


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


# --- gathering what a view needs -------------------------------------------------


def analyse_command(run_dir: Path | str, diags: list[str]) -> tuple[list[str], Path]:
    """(command, folder to run it in) for ``analyse --case <this run> --diag ...``.

    The campaign's cases.toml is the nearest one above the run folder, and
    the case is the run folder's path below it. Raises ShotfileError, with
    the lines to add, if there is no cases.toml or the run is not in it.
    """
    run_dir = Path(run_dir).resolve()
    root = next((f for f in run_dir.parents if (f / "cases.toml").is_file()), None)
    if root is None:
        raise ShotfileError(
            f"no cases.toml above {run_dir}: analyse needs one in the campaign folder"
        )
    name = run_dir.relative_to(root).as_posix()
    import tomllib

    try:
        listed = tomllib.loads((root / "cases.toml").read_text(encoding="utf-8")).get("cases", {})
    except tomllib.TOMLDecodeError as exc:
        raise ShotfileError(f"{root / 'cases.toml'}: not valid TOML ({exc})") from exc
    if name not in listed:
        raise ShotfileError(
            f"{root / 'cases.toml'} has no entry for this run. Add:\n\n"
            f'[cases."{name}"]\nsteps = {{ first_last = true }}\n'
        )
    command = [
        sys.executable, "-c",
        "import sys; from ashen.cli.analyse import main; sys.exit(main())",
        "--cases", str(root / "cases.toml"), "--case", name,
    ]
    for diag in diags:
        command += ["--diag", diag]
    return command, root


def run_analyse(run_dir: Path | str, diags: list[str]) -> int:
    """Run ``analyse`` for this run and print its output as it comes.

    This is the one place the viewer starts JOREK's tools: it runs in the
    notebook's process until analyse is done, which for ``four`` or
    ``poincare`` over many steps is long. Returns analyse's exit status.
    """
    import os
    import subprocess

    command, root = analyse_command(run_dir, diags)
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    print(f"in {root}:  analyse --case {command[command.index('--case') + 1]} "
          + " ".join(f"--diag {d}" for d in diags), flush=True)
    process = subprocess.Popen(
        command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def gather_step_caches(run_dir: Path | str, step: int) -> list[str]:
    """Gather the q-profile and zeroD of one step with jorek2_postproc, if
    missing: what the tuner's "JOREK:" line and the equilibrium view's q
    need. Two short calls; no cases.toml entry needed. Returns notes on what
    was done."""
    from ashen.diagnostics.qprofile import run_qprofile_step
    from ashen.jorek2 import Jorek2Run, run_zero_d
    from ashen.postproc import zero_d_is_usable

    paths = _paths(run_dir)
    run = Jorek2Run(run_dir=paths.run_dir, exe_dir=paths.run_dir,
                    namelist=paths.run_dir / "in_main", pad_width=paths.pad_width)
    notes = []
    for label, have, gather in (
        ("q-profile", paths.qprofile(step).is_file(), lambda: run_qprofile_step(run, step, paths)),
        ("zeroD", zero_d_is_usable(paths.zero_d(step)), lambda: run_zero_d(run, step, paths)),
    ):
        if have:
            notes.append(f"{label} of step {step}: already there")
            continue
        try:
            gather()
            notes.append(f"{label} of step {step}: gathered")
        except Exception as exc:  # report, and still try the other
            notes.append(f"{label} of step {step}: failed ({type(exc).__name__}: {exc})")
    return notes


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
    inputs: cur.QLiProfile | None = None,
    inputs_label: str = "ffprime_prof.dat",
):
    """j, q and FF' of the q0/l_i/q_edge profile; JOREK's q overlaid if given.

    ``inputs`` is the profile the run folder's ffprime_prof.dat holds now
    (input_profile), drawn in dark grey on all three panels: with
    ffprime_method="castor" that is the CASTOR3D profile, and it is what the
    sliders' profile can be laid over. Outside the family's q0 window the
    sliders' profile is not drawn, only the message.
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
    if inputs is not None:
        # a broad grey line underneath, so the sliders' profile shows on top of it
        under = dict(color=_GREY, lw=4.5, alpha=0.75, zorder=1, solid_capstyle="round")
        ax_j.plot(inputs.rho, inputs.j / 1e6, label=inputs_label, **under)
        ax_q.plot(inputs.rho, inputs.q, label=f"{inputs_label} (cylinder)", **under)
        ax_f.plot(inputs.psi_n, cur.ffprime_from_current(inputs.j, geometry.R0),
                  label=inputs_label, **under)
        ax_j.legend(frameon=False, labelcolor=_INK, loc="upper right")
        message += (
            f"\n{inputs_label}:  q0 = {inputs.q[0]:.3f}   l_i = {inputs.li:.3f}   "
            f"qa = {inputs.q[-1]:.3f}   |   j0 = {inputs.j0 / 1e6:.2f} MA/m²   "
            f"Ip = {inputs.Ip / 1e3:.0f} kA   (cylinder estimate)"
        )
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
    if made is not None or achieved is not None or inputs is not None:
        ax_q.legend(frameon=False, labelcolor=_INK, loc="upper left")

    _style(ax_j, "r / a", "j [MA/m²]", "Current density at R0")
    _style(ax_q, "r / a", "q", "Safety factor")
    _style(ax_f, r"$\psi_N$ (plasma)", "FF' [T]", "FF' as JOREK reads it")
    for ax in (ax_j, ax_q):
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
    ax_f.set_xlim(0, 1)
    fig.suptitle(message, x=0.01, ha="left", fontsize=10.5)
    lines = message.count("\n") + 1
    fig.tight_layout(rect=(0, 0, 1, 0.99 - 0.05 * lines))
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


def default_four_modes(qa: float, n_values=(1, 2)) -> list[tuple[int, int]]:
    """The (n, m) modes worth looking at for an edge safety factor ``qa``.

    For each n, m runs up to one past the highest rational surface m/n
    inside the plasma: ``floor(n qa) + 1``. It starts at 1/1 for n = 1 and at
    m = n + 1 for higher n, whose m = n mode sits on the q = 1 surface that
    1/1 already stands for. qa = 2.8 gives 1/1 2/1 3/1 and 3/2 4/2 5/2 6/2.
    """
    modes = []
    for n in n_values:
        first = 1 if n == 1 else n + 1
        last = math.floor(n * qa + 1e-9) + 1
        modes += [(n, m) for m in range(first, last + 1)]
    return modes


def shotfile_qa(run_dir: Path | str) -> float | None:
    """The edge safety factor the shotfile asks for: ``current_qa`` for a
    q0/l_i/qa profile, else ``qa``. None if the shotfile gives neither or
    cannot be read."""
    try:
        params = load_shotfile(Path(run_dir) / "shotfile.py")
    except Exception:
        return None
    if params.ffprime_method == "q_li" and params.current_qa is not None:
        return float(params.current_qa)
    return None if params.qa is None else float(params.qa)


def four_view_modes(run_dir: Path | str, variable: str, n_fallback: int = 6) -> list[tuple[int, int]]:
    """The modes four_view offers for a variable: default_four_modes for the
    shotfile's qa, those of them the cache holds. Without a qa, or if the
    cache holds none of them, the ``n_fallback`` largest."""
    paths, steps = _paths(run_dir), four_steps(run_dir)
    if not steps:
        return []
    cached = {(n, m) for var, n, m in fc.list_keys(paths.four_cache(steps[-1])) if var == variable}
    qa = shotfile_qa(run_dir)
    wanted = [mode for mode in default_four_modes(qa) if mode in cached] if qa else []
    if wanted:
        return wanted
    series = max_amplitude_series(paths, steps, variables=[variable])
    biggest = sorted(series, key=lambda k: -np.nanmax(series[k]))[:n_fallback]
    return sorted((n, m) for _, n, m in biggest)


@dataclass(frozen=True)
class FourData:
    """One variable's chosen modes at every cached step, in memory, so a
    view can change step or modes without going back to the files."""

    variable: str
    steps: list[int]
    #: {(n, m): max |c| over the radius, one value per step (nan if missing)}
    amplitude: dict[tuple[int, int], np.ndarray]
    #: {(n, m): {step: (psi_n, |c|)}}
    radial: dict[tuple[int, int], dict[int, tuple[np.ndarray, np.ndarray]]]
    #: {(n, m): the largest |c| on a q = m/n surface, per step (nan if the
    #: step has no q-profile cache or q never reaches m/n)}: the numbers
    #: four_modes.rational_surface_series gives
    rational: dict[tuple[int, int], np.ndarray]
    #: {(n, m): {step: (psi_n of its q = m/n surfaces, |c| there)}}
    surfaces: dict[tuple[int, int], dict[int, tuple[np.ndarray, np.ndarray]]]


def load_four_data(run_dir: Path | str, variable: str, modes) -> FourData:
    """Read ``modes`` ((n, m) pairs) of ``variable`` from every four cache.
    Only those records are read, not the whole of each file."""
    paths, steps = _paths(run_dir), four_steps(run_dir)
    modes = [(int(n), int(m)) for n, m in modes]
    amplitude = {mode: np.full(len(steps), np.nan) for mode in modes}
    rational = {mode: np.full(len(steps), np.nan) for mode in modes}
    radial: dict = {mode: {} for mode in modes}
    surfaces: dict = {mode: {} for mode in modes}
    for i, step in enumerate(steps):
        q_path = paths.qprofile(step)
        q_profile = read_qprofile(q_path) if q_path.is_file() else None
        for (_, n, m), record in fc.read_records(paths.four_cache(step), variable, modes).items():
            values = record.abs
            if not values.size:
                continue
            amplitude[(n, m)][i] = float(np.max(values))
            radial[(n, m)][step] = (record.psi_n, values)
            if q_profile is None or n == 0:
                continue
            crossings = find_rational_surfaces(*q_profile, m / n)
            if crossings:
                there = np.interp(crossings, record.psi_n, values)
                rational[(n, m)][i] = float(np.max(there))
                surfaces[(n, m)][step] = (np.asarray(crossings), there)
    return FourData(variable, steps, amplitude, radial, rational, surfaces)


def _mode_label(mode: tuple[int, int]) -> str:
    return f"n={mode[0]}, m={mode[1]}"


class FourPlot:
    """The two four panels drawn once on ``fig``; after that a change of
    step, visible modes or scale only updates the lines already there."""

    def __init__(self, fig, data: FourData, *, step: int | None = None, colors=None,
                 log: bool = True, log_radial: bool = False, real_psi_edge: float = 1.0,
                 rational: bool = False):
        from ashen.plotting.four_modes import draw_mode_amplitudes

        self.fig, self.data = fig, data
        self.ax_t, self.ax_r = fig.subplots(1, 2)
        variable = data.variable
        series = {(variable, n, m): values for (n, m), values in data.amplitude.items()}
        draw_mode_amplitudes(self.ax_t, data.steps, series, variable=variable, log=log,
                             xlabel="Time step", colors=colors)
        by_label = {line.get_label(): line for line in self.ax_t.get_lines()}
        self.amplitude_lines = {mode: by_label[_mode_label(mode)] for mode in data.amplitude}
        self.marker = self.ax_t.axvline(data.steps[-1] if data.steps else 0, color=_GREY, lw=1, ls="--")
        self.ax_t.set_title(f"max |{variable}| per mode", loc="left", fontsize=11, fontweight="bold")

        # same colour per mode as the left panel
        self.radial_lines = {
            mode: self.ax_r.plot([], [], lw=1.6, label=_mode_label(mode),
                                 color=self.amplitude_lines[mode].get_color())[0]
            for mode in sorted(data.amplitude)
        }
        # where each mode's q = m/n surfaces are, shown with the rational amplitude
        self.surface_marks = {
            mode: self.ax_r.plot([], [], ls="", marker="o", ms=6, mec="white",
                                 color=line.get_color(), label="_nolegend_")[0]
            for mode, line in self.radial_lines.items()
        }
        self.rational = False
        _style(self.ax_r, r"$\psi_N$ (JOREK grid)", f"|{variable}|", "")
        if real_psi_edge < 1.0:
            self.ax_r.axvline(real_psi_edge, color=_GREY, lw=1, ls="--")
        if log_radial:
            self.ax_r.set_yscale("log")
        self.step = None
        self.set_step(data.steps[-1] if step is None and data.steps else step)
        if rational:
            self.set_rational(True)
        self._legends()

    def _rescale(self, ax) -> None:
        # Follows the data unless the axis was zoomed or panned by hand:
        # matplotlib turns autoscaling off for an axis whose limits were set.
        ax.relim(visible_only=True)
        ax.autoscale_view()

    def _legends(self) -> None:
        for ax, lines in ((self.ax_t, self.amplitude_lines), (self.ax_r, self.radial_lines)):
            shown = [lines[mode] for mode in sorted(lines) if lines[mode].get_visible()]
            if shown:
                ax.legend(handles=shown, frameon=False, labelcolor=_INK, fontsize=8)
            elif ax.get_legend() is not None:
                ax.get_legend().remove()

    def set_step(self, step: int | None) -> None:
        self.step = step
        for mode, line in self.radial_lines.items():
            psi_n, values = self.data.radial[mode].get(step, ((), ()))
            line.set_data(psi_n, values)
        self._mark_surfaces()
        if step is not None:
            self.marker.set_xdata([step, step])
        self.ax_r.set_title(f"Radial structure, step {step}", loc="left", fontsize=11,
                            fontweight="bold")
        self._rescale(self.ax_r)

    def _mark_surfaces(self) -> None:
        for mode, marks in self.surface_marks.items():
            at = self.data.surfaces[mode].get(self.step) if self.rational else None
            marks.set_data(*(at if at is not None else ((), ())))

    def set_rational(self, rational: bool) -> None:
        """The left panel's amplitude: each mode's largest value on its own
        q = m/n surface (True), or its maximum over the radius (False)."""
        self.rational = bool(rational)
        source = self.data.rational if self.rational else self.data.amplitude
        for mode, line in self.amplitude_lines.items():
            line.set_ydata(source[mode])
        variable = self.data.variable
        what = f"|{variable}| at q = m/n" if self.rational else f"max |{variable}|"
        self.ax_t.set_ylabel(what)
        self.ax_t.set_title(f"{what} per mode", loc="left", fontsize=11, fontweight="bold")
        self._mark_surfaces()
        self._rescale(self.ax_t)

    def set_visible(self, modes) -> None:
        modes = {(int(n), int(m)) for n, m in modes}
        for lines in (self.amplitude_lines, self.radial_lines, self.surface_marks):
            for mode, line in lines.items():
                line.set_visible(mode in modes)
        self._legends()
        self._rescale(self.ax_t)
        self._rescale(self.ax_r)

    def set_log(self, log: bool, log_radial: bool) -> None:
        self.ax_t.set_yscale("log" if log else "linear")
        self.ax_r.set_yscale("log" if log_radial else "linear")
        self._rescale(self.ax_t)
        self._rescale(self.ax_r)

    def reset_view(self) -> None:
        """Back to limits that follow the data, after zooming or panning."""
        for ax in (self.ax_t, self.ax_r):
            ax.set_autoscale_on(True)
            self._rescale(ax)


_FOUR_FIGSIZE = (14, 4.6)


def four_figure(
    run_dir: Path | str,
    variable: str = "Psi",
    *,
    step: int | None = None,
    modes: list[tuple[int, int]] | None = None,
    n_modes: int = 6,
    log: bool = True,
    log_radial: bool = False,
    colors: dict[tuple[int, int], str] | None = None,
    rational: bool = False,
):
    """Mode amplitudes against step, and the radial eigenfunctions at one step.

    ``modes`` is a list of (n, m); None draws the ``n_modes`` largest.
    ``rational`` draws each mode's amplitude on its q = m/n surface instead
    of its maximum over the radius, and marks those surfaces on the right.
    ``log`` is the amplitude axis against step, ``log_radial`` the radial
    one. ``colors`` maps (n, m) to a colour, so a mode keeps its colour
    when others are left out.
    """
    from matplotlib.figure import Figure

    paths = _paths(run_dir)
    steps = four_steps(run_dir)
    fig = Figure(figsize=_FOUR_FIGSIZE, layout="constrained")
    if not steps:
        ax = fig.subplots(1, 2)[0]
        ax.text(0.5, 0.5, "no jorek2_four cache (analyse --diag four)", ha="center",
                transform=ax.transAxes)
        return fig
    if modes is None:
        series = max_amplitude_series(paths, steps, variables=[variable])
        biggest = sorted(series, key=lambda k: -np.nanmax(series[k]))[:n_modes]
        modes = [(n, m) for _, n, m in biggest]
    FourPlot(fig, load_four_data(run_dir, variable, modes), step=step, colors=colors, log=log,
             log_radial=log_radial, real_psi_edge=_real_psi_edge(paths), rational=rational)
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


def _canvas(w):
    """Where _show puts a figure."""
    return w.VBox([])


def _show(canvas, make_figure) -> None:
    """Draw ``make_figure()`` into a _canvas, replacing what was there.

    The figure goes in as a PNG Image widget, not through an Output widget:
    VS Code shows what an Output widget captures while its cell runs a second
    time, below the widgets.
    """
    import io

    w = _widgets()
    plt = _plt()
    try:
        fig = make_figure()
    except Exception as exc:  # a view should say what is missing, not die
        canvas.children = (w.HTML(
            f"<pre>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</pre>"
        ),)
        return
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=_FIGURE_DPI)
    plt.close(fig)
    width = f"{fig.get_figwidth() * 96:.0f}px"   # the size an inline figure has
    canvas.children = (w.Image(value=buffer.getvalue(), format="png",
                               layout=w.Layout(width=width, max_width="100%")),)


class _Log:
    """Where a button's messages go: ``with log:`` empties it and then shows
    what is printed, line by line as it comes.

    An HTML widget whose text is set, not an Output widget: in VS Code an
    Output widget can keep what it should have cleared and show a message
    several times.
    """

    def __init__(self, w):
        self.widget = w.HTML()
        self.text = ""

    def write(self, text: str) -> int:
        self.text += text
        self.widget.value = (
            f"<pre style='margin:2px 0;white-space:pre-wrap'>{html.escape(self.text)}</pre>"
        )
        return len(text)

    def flush(self) -> None:
        pass

    def __enter__(self):
        import contextlib

        self.text, self.widget.value = "", ""
        self._redirect = contextlib.redirect_stdout(self)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc):
        return self._redirect.__exit__(*exc)


def _live_figure(w, figsize):
    """(figure, widget showing it, draw()) for a figure that is updated in
    place. With ipympl installed the widget holds its canvas: zoom and pan
    with the mouse from the toolbar, and a draw sends only the new image.
    Without it the widget holds a PNG that draw() renders again, with a note
    saying so. Either way the figure is the widget's first child.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, layout="constrained")
    try:
        from ipympl.backend_nbagg import Canvas, FigureManager
    except ImportError:
        image = w.Image(format="png", layout=w.Layout(width=f"{figsize[0] * 96:.0f}px",
                                                      max_width="100%"))
        note = w.HTML("<small>For zoom and pan: <code>pip install --user ipympl</code>, "
                      "then restart the kernel.</small>")

        def draw():
            import io

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=_FIGURE_DPI)
            image.value = buffer.getvalue()

        return fig, w.VBox([image, note]), draw

    canvas = Canvas(fig)
    manager = FigureManager(canvas, 0)
    canvas.header_visible = False
    canvas.footer_visible = True       # the cursor's x, y
    canvas.toolbar_position = "top"

    # A failure while handling the mouse or the toolbar is otherwise silent:
    # the figure just does not respond. Say what failed, under the figure.
    problem, handle = w.HTML(), manager.handle_json

    def handle_json(content):
        try:
            return handle(content)
        except Exception as exc:
            import ipympl
            import matplotlib

            problem.value = (
                "<small style='color:#c8442f'>The figure could not handle "
                f"<code>{html.escape(str(content.get('type')))}</code>: "
                f"{html.escape(type(exc).__name__)}: {html.escape(str(exc))} "
                f"(matplotlib {matplotlib.__version__}, ipympl {ipympl.__version__}). If the two "
                "do not go together: <code>pip install --user -U ipympl matplotlib</code>, "
                "then restart the kernel.</small>")
            raise

    manager.handle_json = handle_json
    return fig, w.VBox([canvas, problem]), canvas.draw_idle


def _gathering(w, label: str, action, build):
    """``build()``'s view under a button that runs ``action()`` (printing
    into a log) and then builds the view again from what is now on disk."""
    button = w.Button(description=label, layout=w.Layout(width="auto"),
                      tooltip="Runs in this notebook until it is done")
    log, holder = _Log(w), w.VBox([build()])

    def on_click(_):
        button.disabled = True
        try:
            with log:
                try:
                    action()
                except Exception as exc:
                    print(f"{type(exc).__name__}: {exc}")
            holder.children = (build(),)
        finally:
            button.disabled = False

    button.on_click(on_click)
    return w.VBox([button, log.widget, holder])


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

    def current_inputs():
        made = input_profile(run_dir, geometry)
        try:
            method = load_shotfile(shotfile).ffprime_method
        except ShotfileError:
            method = params.ffprime_method
        return made, f"ffprime_prof.dat ({method})"

    # Without saved values the sliders start on the profile the folder has,
    # so a "castor" run opens with the model lying near its own profile.
    have, _ = current_inputs()
    start = (
        params.current_q0 or (have.q[0] if have else achieved.q0 if achieved else 1.0),
        params.current_li or (have.li if have else achieved.li if achieved else 1.2),
        params.current_qa or (have.q[-1] if have else achieved.q_edge if achieved else 3.0),
    )
    q0 = _slider(w, start[0], 0.3, 4.0, 0.01, "q0")
    li = _slider(w, start[1], 0.55, 2.5, 0.01, "l_i")
    q_edge = _slider(w, start[2], 1.5, 10.0, 0.05, "qa")
    save = w.Button(description="Save to shotfile", button_style="primary")
    regenerate = w.Button(description="Regenerate inputs")
    reset = w.Button(description="Reset to shotfile", tooltip="Put the sliders back to shotfile.py's values")
    gather = w.Button(description=f"Gather JOREK's q-profile (step {step})",
                      layout=w.Layout(width="auto"),
                      tooltip="q-profile and zeroD of that step, for the JOREK: line")
    state = {"achieved": achieved}
    figure_out, log_out = _canvas(w), _Log(w)
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
        inputs, label = current_inputs()
        _show(figure_out, lambda: tuner_figure(
            q0.value, li.value, q_edge.value, geometry, achieved=state["achieved"],
            inputs=inputs, inputs_label=label))

    def on_save(_):
        with log_out:
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
        redraw()

    def on_reset(_):
        with log_out:
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

    def on_gather(_):
        with log_out:
            print("\n".join(gather_step_caches(run_dir, step)))
            try:
                state["achieved"] = achieved_q_li(_paths(run_dir), step, f0=geometry.B0 * geometry.R0)
            except Exception as exc:
                print(f"could not read them back: {type(exc).__name__}: {exc}")
        redraw()

    for slider in (q0, li, q_edge):
        slider.observe(redraw, names="value")
    save.on_click(on_save)
    regenerate.on_click(on_regenerate)
    reset.on_click(on_reset)
    gather.on_click(on_gather)
    redraw()
    return w.VBox([w.VBox([q0, li, q_edge]), w.HBox([save, regenerate, reset, gather]), status, log_out.widget,
                   figure_out])


def boundary_view(run_dir: Path | str = "."):
    """The run's boundaries (no controls)."""
    w = _widgets()
    out = _canvas(w)
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
    gather = w.Button(description="Gather q-profile + zeroD for this step",
                      layout=w.Layout(width="auto"))
    out, log = _canvas(w), _Log(w)

    def redraw(*_):
        _show(out, lambda: equilibrium_figure(run_dir, step.value))

    def on_gather(_):
        with log:
            print("\n".join(gather_step_caches(run_dir, step.value)))
        redraw()

    step.observe(redraw, names="value")
    gather.on_click(on_gather)
    redraw()
    return w.VBox([w.HBox([step, gather]), log.widget, out])


def four_view(run_dir: Path | str = ".", *, variable: str = "Psi"):
    """Mode amplitudes and radial structure of one jorek2_four ``variable``
    (Psi unless another is named), with a step selector
    and a checkbox per mode (four_view_modes: by the shotfile's qa), under a
    button that runs ``analyse --diag four`` for this run.

    The modes are read once (load_four_data); a control then
    updates the one figure (FourPlot) instead of drawing a new one. With
    ipympl installed the figure can be zoomed and panned with the mouse.
    """
    w = _widgets()

    def build():
        steps = four_steps(run_dir)
        if not steps:
            return w.HTML("no jorek2_four cache yet")
        paths = _paths(run_dir)
        variables = sorted({key[0] for key in fc.list_keys(paths.four_cache(steps[-1]))})
        if variable not in variables:
            return w.HTML(f"no {html.escape(variable)} in the jorek2_four cache; it holds "
                          f"{html.escape(', '.join(variables))}")
        step = w.SelectionSlider(options=steps, value=steps[-1], description="step",
                                 continuous_update=False)
        log = w.Checkbox(value=True, description="log amplitudes", indent=False,
                         layout=w.Layout(width="auto"))
        log_radial = w.Checkbox(value=False, description="log radial structure", indent=False,
                                layout=w.Layout(width="auto"))
        amplitude = w.ToggleButtons(
            options=[("max over radius", False), ("at q = m/n surface", True)], value=False,
            description="amplitude", style={"button_width": "auto"},
            tooltips=["each mode's largest |c| anywhere on the radius",
                      "each mode's |c| on its own rational surface (needs q-profile caches)"])
        note = w.HTML()
        boxes = w.HBox([], layout=w.Layout(flex_flow="row wrap"))
        reset = w.Button(description="Reset view", layout=w.Layout(width="auto"),
                         tooltip="Back to limits that follow the data, after zooming")
        holder = w.VBox([])
        state = {}
        edge = _real_psi_edge(paths)

        def visible():
            return [box.mode for box in boxes.children if box.value]

        def load(*_):
            """The modes into memory and a new figure: the slow step, done
            once. Everything else updates that figure."""
            modes = four_view_modes(run_dir, variable)
            palette = _plt().get_cmap("tab20").colors
            colors = {
                mode: "#%02x%02x%02x" % tuple(round(255 * c) for c in palette[i % len(palette)])
                for i, mode in enumerate(modes)
            }
            made = []
            for n, m in modes:
                box = w.Checkbox(value=True, description=f"{m}/{n}", indent=False,
                                 layout=w.Layout(width="70px"))
                box.mode = (n, m)
                box.observe(lambda _: update(lambda plot: plot.set_visible(visible())), names="value")
                made.append(box)
            boxes.children = made
            try:
                data = load_four_data(run_dir, variable, modes)
                fig, widget, state["draw"] = _live_figure(w, _FOUR_FIGSIZE)
                state["plot"] = FourPlot(fig, data, step=step.value, colors=colors, log=log.value,
                                         log_radial=log_radial.value, real_psi_edge=edge,
                                         rational=amplitude.value)
            except Exception as exc:  # say what is missing, do not die
                state.pop("plot", None)
                holder.children = (w.HTML(
                    f"<pre>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</pre>"),)
                return
            holder.children = (widget,)
            state["draw"]()

        def update(change):
            if "plot" in state:
                change(state["plot"])
                state["draw"]()
            explain()

        def explain():
            """Say so when the rational amplitude has nothing to show."""
            note.value = ""
            if amplitude.value and "plot" in state:
                data = state["plot"].data
                if not any(np.isfinite(values).any() for values in data.rational.values()):
                    note.value = (
                        "<small>No amplitude on a rational surface: these steps have no "
                        "q-profile cache (<code>analyse --diag four</code> gathers them), or q "
                        "never reaches m/n for these modes.</small>")

        step.observe(lambda _: update(lambda plot: plot.set_step(step.value)), names="value")
        for control in (log, log_radial):
            control.observe(
                lambda _: update(lambda plot: plot.set_log(log.value, log_radial.value)), names="value")
        reset.on_click(lambda _: update(lambda plot: plot.reset_view()))
        amplitude.observe(
            lambda _: update(lambda plot: plot.set_rational(amplitude.value)), names="value")
        load()
        explain()
        return w.VBox([w.HBox([step, log, log_radial, reset]),
                       w.HBox([w.HTML("modes m/n:&nbsp;"), boxes]), w.HBox([amplitude, note]),
                       holder])

    return _gathering(w, "Run analyse --diag four", lambda: run_analyse(run_dir, ["four"]), build)


def profiles_view(run_dir: Path | str = "."):
    """Cached radial profiles, with variable and step selectors, under a
    button that runs ``analyse --diag profiles`` for this run."""
    w = _widgets()

    def build():
        available = profiles_available(run_dir)
        if not available:
            return w.HTML("no cached profiles yet")
        labels = {f"{k.var} vs {k.coords_var} ({k.tor_mode})": k for k in available}
        which = w.Dropdown(options=list(labels), description="profile",
                           layout=w.Layout(width="420px"))
        first = available[labels[which.value]]
        step = w.SelectionSlider(options=first, value=first[-1], description="step",
                                 continuous_update=False)
        out = _canvas(w)

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

    return _gathering(
        w, "Run analyse --diag profiles", lambda: run_analyse(run_dir, ["profiles"]), build
    )


#: The sections of case_viewer, in order: (heading, note under it, the view).
#: A new view is added here, and so appears in every run folder's notebook.
_SECTIONS = [
    (
        "Current profile from q0, l_i, qa",
        "<b>Save to shotfile</b> writes the three values and <code>ffprime_method = \"q_li\"</code> "
        "into <code>shotfile.py</code>; <b>Regenerate inputs</b> then rewrites this folder's input "
        "files from the shotfile. A yellow band says when the two are out of step. The grey "
        "profile is the one in this folder's <code>ffprime_prof.dat</code>.",
        lambda run_dir, step: profile_tuner(run_dir, step=step),
    ),
    ("Boundaries", "", lambda run_dir, step: boundary_view(run_dir)),
    (
        "Equilibrium",
        "Contours are drawn on the grid nodes: good for looking, not for measuring.",
        lambda run_dir, step: equilibrium_view(run_dir),
    ),
    ("Fourier modes (analyse --diag four)", "", lambda run_dir, step: four_view(run_dir)),
    ("Radial profiles (analyse --diag profiles)", "", lambda run_dir, step: profiles_view(run_dir)),
]


def case_viewer(run_dir: Path | str = ".", *, step: int = 0):
    """Every view of a run folder on one page, each under its heading.

    ``step`` is the restart whose q-profile the tuner shows as JOREK's
    result. A view that fails says so in its place and the others still draw.
    """
    w = _widgets()
    run_dir = Path(run_dir).resolve()
    children = [w.HTML(f"<h2>Case viewer</h2><code>{html.escape(str(run_dir))}</code>")]
    for title, note, build in _SECTIONS:
        children.append(w.HTML(f"<h3>{html.escape(title)}</h3>{note}"))
        try:
            children.append(build(run_dir, step))
        except Exception as exc:  # one broken view must not hide the rest
            children.append(w.HTML(
                f"<pre style='color:{_RED}'>{html.escape(type(exc).__name__)}: "
                f"{html.escape(str(exc))}</pre>"
            ))
    return w.VBox(children)
