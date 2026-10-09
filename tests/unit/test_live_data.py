"""ashen.diagnostics.live_data -- one quantity of macroscopic_vars.dat."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ashen.diagnostics.live_data import LIVE_DATA_FILE, read_live_data  # noqa: E402

HEADER = """@times_xlabel: time step
@magnetic_energies_xlabel: normalized time
@magnetic_energies_xlabel_si: time [ms]
@magnetic_energies_ylabel: normalized magnetic energy
@magnetic_energies_ylabel_si: normalized magnetic energy
@magnetic_energies_x2si:   2.000000000E-03
@magnetic_energies_y2si:   1.000000000E+00
@magnetic_energies_logy: 1
@magnetic_energies: %"time"           "E_{mag,00}" "E_{mag,01}"
@kinetic_energies_xlabel: normalized time
@kinetic_energies: %"time"           "E_{kin,00}" "E_{kin,01}"
"""


def _file(tmp_path, times, *, extra=""):
    lines = [HEADER]
    for t in times:
        lines.append(f"@times:  1  {t}\n")
        lines.append(f"@magnetic_energies:  {t:.9E}  5.0D-02  {1e-12 * (1 + t):.9E}\n")
        lines.append(f"@kinetic_energies:  {t:.9E}  9.0E-09  0.0E+00\n")
    path = tmp_path / LIVE_DATA_FILE
    path.write_text("".join(lines) + extra)
    return path


def test_reads_one_quantity_with_the_files_own_labels_and_units(tmp_path):
    live = read_live_data(_file(tmp_path, [1.0, 2.0, 3.0]))
    assert live.labels == ["E_{mag,00}", "E_{mag,01}"]
    np.testing.assert_allclose(live.x, [2e-3, 4e-3, 6e-3])            # x2si applied
    np.testing.assert_allclose(live.values[:, 0], 0.05)               # Fortran D exponent
    np.testing.assert_allclose(live.values[:, 1], [2e-12, 3e-12, 4e-12])
    assert (live.xlabel, live.ylabel, live.logy) == (
        "time [ms]", "normalized magnetic energy", True)

    kinetic = read_live_data(tmp_path / LIVE_DATA_FILE, "kinetic_energies")
    assert kinetic.labels == ["E_{kin,00}", "E_{kin,01}"] and not kinetic.logy
    np.testing.assert_allclose(kinetic.x, [1.0, 2.0, 3.0])            # no factor given: as written


def test_a_restart_from_an_earlier_step_replaces_the_rows_it_overwrites(tmp_path):
    # ran to t = 4, restarted from t = 2 and ran to 5
    live = read_live_data(_file(tmp_path, [1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 5.0]))
    np.testing.assert_allclose(live.x / 2e-3, [1.0, 2.0, 2.5, 3.5, 5.0])
    assert np.all(np.diff(live.x) > 0)


def test_missing_file_and_missing_quantity_say_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_live_data(tmp_path / LIVE_DATA_FILE)
    path = _file(tmp_path, [1.0])
    with pytest.raises(ValueError, match="no 'nope'; it has kinetic_energies, magnetic_energies"):
        read_live_data(path, "nope")
    (tmp_path / "empty.dat").write_text(HEADER)
    with pytest.raises(ValueError, match="no rows yet"):
        read_live_data(tmp_path / "empty.dat")


def test_viewer_draws_the_magnetic_energies_and_reloads(tmp_path, monkeypatch):
    w = pytest.importorskip("ipywidgets")
    pytest.importorskip("h5py")
    from ashen import viewer

    _file(tmp_path, [1.0, 2.0, 3.0])
    fig = viewer.energies_figure(tmp_path)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log" and len(ax.get_lines()) == 2
    assert [l.get_label() for l in ax.get_lines()] == ["$E_{mag,00}$", "$E_{mag,01}$"]
    assert viewer.energies_figure(tmp_path, log=False).axes[0].get_yscale() == "linear"
    assert "macroscopic_vars.dat" in viewer.energies_figure(tmp_path / "nope").axes[0].texts[0].get_text()

    monkeypatch.setitem(sys.modules, "ipympl.backend_nbagg", None)    # the PNG path
    controls, holder = viewer.energies_view(tmp_path).children
    reload, log, note = controls.children
    assert log.value is True and "3 time steps" in note.value
    before = bytes(holder.children[0].children[0].value)
    _file(tmp_path, [1.0, 2.0, 3.0, 4.0, 5.0])                        # the run went on
    reload.click()
    assert "5 time steps" in note.value
    assert bytes(holder.children[0].children[0].value) != before

    empty = tmp_path / "none"
    empty.mkdir()
    assert isinstance(viewer.energies_view(empty).children[1].children[0], w.HTML)
    assert any("Magnetic energies" in title for title, _, _ in viewer._SECTIONS)
