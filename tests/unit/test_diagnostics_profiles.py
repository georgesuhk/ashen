"""ashen.diagnostics.profiles -- compound-var expansion, cache-gated gathering
across several tor_modes, per-mode resilience, and the .npz reader.

No jorek2_postproc involved: gather_profiles is exercised with n_workers=1
(or a single task), which keeps it off the process pool entirely, so
extract_profile can be monkeypatched -- a pooled call is pickled by
reference and would re-import the real function fresh in the child process.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashen.diagnostics import profiles as profiles_mod
from ashen.diagnostics.profiles import (
    TOR_MODES,
    edge_toroidal_field,
    expand_compound_vars,
    gather_profiles,
    read_profile_series,
)
from ashen.jorek2 import Jorek2Error, Jorek2Run, MissingRestartError
from ashen.paths import RunPaths


# --- expand_compound_vars ----------------------------------------------------------


def test_expand_compound_vars_leaves_plain_vars_alone():
    assert expand_compound_vars(["currdens", "T"]) == ["currdens", "T"]


def test_expand_compound_vars_expands_q():
    out = expand_compound_vars(["q"])
    assert set(out) == {"r_minor", "Btheta", "Btor"}


def test_expand_compound_vars_expands_jgrad():
    out = expand_compound_vars(["Jgrad"])
    assert set(out) == {"currdens", "Btheta", "Btor", "r_minor"}


def test_expand_compound_vars_deduplicates_shared_components():
    out = expand_compound_vars(["q", "Jgrad"])
    # Btheta/Btor are shared between q and Jgrad -- must appear once each.
    assert out.count("Btheta") == 1
    assert out.count("Btor") == 1


def test_tor_modes_lists_all_four_variants():
    assert set(TOR_MODES) == {"average", "midplane", "midplane outer", "midplane inner"}


# --- read_profile_series -------------------------------------------------------------


@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path / "run", pad_width=6)


def _write_cache(paths, coords_var, var, step, tor_mode, x, y):
    cache = paths.profile_cache(coords_var, var, step, tor_mode)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=np.asarray(x), y=np.asarray(y))


def test_read_profile_series_reads_every_cached_step(paths):
    _write_cache(paths, "Psi_N", "currdens", 100, "midplane", [0.1, 0.5], [1.0, 2.0])
    _write_cache(paths, "Psi_N", "currdens", 200, "midplane", [0.1, 0.5], [1.5, 2.5])

    series = read_profile_series(paths, [100, 200], "Psi_N", "currdens", "midplane")

    assert set(series) == {100, 200}
    np.testing.assert_allclose(series[100][1], [1.0, 2.0])
    np.testing.assert_allclose(series[200][1], [1.5, 2.5])


def test_read_profile_series_omits_missing_steps(paths):
    _write_cache(paths, "Psi_N", "currdens", 100, "midplane", [0.1], [1.0])
    # step 200 never gathered.
    series = read_profile_series(paths, [100, 200], "Psi_N", "currdens", "midplane")
    assert set(series) == {100}


def test_read_profile_series_distinguishes_tor_modes(paths):
    _write_cache(paths, "Psi_N", "currdens", 100, "midplane", [0.1], [1.0])
    _write_cache(paths, "Psi_N", "currdens", 100, "average", [0.1], [9.0])

    midplane = read_profile_series(paths, [100], "Psi_N", "currdens", "midplane")
    average = read_profile_series(paths, [100], "Psi_N", "currdens", "average")

    assert midplane[100][1][0] == pytest.approx(1.0)
    assert average[100][1][0] == pytest.approx(9.0)


def test_read_profile_series_no_caches_returns_empty(paths):
    assert read_profile_series(paths, [100, 200], "Psi_N", "currdens", "midplane") == {}


# --- edge_toroidal_field -------------------------------------------------------------


def test_edge_toroidal_field_interpolates_to_psi_n(paths):
    _write_cache(
        paths, "Psi_N", "Btor", 0, "midplane outer", [0.0, 0.5, 1.0], [4.0, 3.0, 2.0]
    )
    assert edge_toroidal_field(paths) == pytest.approx(2.0)


def test_edge_toroidal_field_reads_step_zero_by_default(paths):
    _write_cache(paths, "Psi_N", "Btor", 100, "midplane outer", [0.0, 1.0], [4.0, 3.0])
    # No profile at step 0, so this must not fall back to step 100.
    assert edge_toroidal_field(paths) is None


def test_edge_toroidal_field_custom_step_and_psi_n(paths):
    _write_cache(paths, "Psi_N", "Btor", 100, "midplane outer", [0.0, 1.0], [4.0, 2.0])
    assert edge_toroidal_field(paths, step=100, psi_n=0.0) == pytest.approx(4.0)


def test_edge_toroidal_field_missing_cache_is_none(paths):
    assert edge_toroidal_field(paths) is None


def test_edge_toroidal_field_ignores_bare_midplane_cache(paths):
    # Bare "midplane" is double-valued in Psi_N and must not be picked up as
    # a stand-in for "midplane outer".
    _write_cache(paths, "Psi_N", "Btor", 0, "midplane", [0.0, 1.0], [4.0, 2.0])
    assert edge_toroidal_field(paths) is None


def test_edge_toroidal_field_unsorted_x_still_interpolates_correctly(paths):
    _write_cache(
        paths, "Psi_N", "Btor", 0, "midplane outer", [1.0, 0.0, 0.5], [2.0, 4.0, 3.0]
    )
    assert edge_toroidal_field(paths) == pytest.approx(2.0)


# --- gather_profiles: cache-gating, multi-mode, and per-mode resilience -------------


@pytest.fixture
def jrun(tmp_path) -> Jorek2Run:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)


def test_gather_profiles_writes_a_cache_per_step_var_mode(jrun, paths, monkeypatch):
    def fake_extract(run, step, var, coords_var, **kwargs):
        return np.array([0.1, 0.5]), np.array([float(step)])

    monkeypatch.setattr(profiles_mod, "extract_profile", fake_extract)

    succeeded = gather_profiles(
        jrun, paths, [100, 200], ["currdens"],
        coords_var="Psi_N", tor_modes=["midplane", "average"], n_workers=1,
    )

    assert succeeded == {"midplane": 2, "average": 2}
    assert paths.profile_cache("Psi_N", "currdens", 100, "midplane").is_file()
    assert paths.profile_cache("Psi_N", "currdens", 100, "average").is_file()
    assert paths.profile_cache("Psi_N", "currdens", 200, "average").is_file()


def test_gather_profiles_accepts_a_bare_string_tor_mode(jrun, paths, monkeypatch):
    monkeypatch.setattr(
        profiles_mod, "extract_profile",
        lambda run, step, var, coords_var, **kwargs: (np.array([0.1]), np.array([1.0])),
    )
    succeeded = gather_profiles(
        jrun, paths, [100], ["currdens"], tor_modes="midplane", n_workers=1,
    )
    assert succeeded == {"midplane": 1}


def test_gather_profiles_skips_cached_steps_unless_forced(jrun, paths, monkeypatch):
    _write_cache(paths, "Psi_N", "currdens", 100, "midplane", [0.1], [1.0])

    calls = []

    def fake_extract(run, step, var, coords_var, **kwargs):
        calls.append(step)
        return np.array([0.1]), np.array([2.0])

    monkeypatch.setattr(profiles_mod, "extract_profile", fake_extract)

    succeeded = gather_profiles(
        jrun, paths, [100, 200], ["currdens"],
        coords_var="Psi_N", tor_modes=["midplane"], n_workers=1,
    )

    assert calls == [200]
    assert succeeded == {"midplane": 2}  # 1 cached + 1 newly gathered


def test_gather_profiles_force_recomputes_cached_steps(jrun, paths, monkeypatch):
    _write_cache(paths, "Psi_N", "currdens", 100, "midplane", [0.1], [1.0])

    calls = []
    monkeypatch.setattr(
        profiles_mod, "extract_profile",
        lambda run, step, var, coords_var, **kwargs: (calls.append(step), (np.array([0.1]), np.array([9.0])))[1],
    )

    gather_profiles(
        jrun, paths, [100], ["currdens"],
        coords_var="Psi_N", tor_modes=["midplane"], n_workers=1, force=True,
    )

    assert calls == [100]


def test_gather_profiles_missing_restart_is_warned_and_skipped(jrun, paths, monkeypatch):
    def fake_extract(run, step, var, coords_var, **kwargs):
        if step == 200:
            raise MissingRestartError(f"restart file not found: step {step}")
        return np.array([0.1]), np.array([1.0])

    monkeypatch.setattr(profiles_mod, "extract_profile", fake_extract)

    with pytest.warns(UserWarning, match="skipping profile step 200"):
        succeeded = gather_profiles(
            jrun, paths, [100, 200], ["currdens"],
            coords_var="Psi_N", tor_modes=["midplane"], n_workers=1,
        )

    assert succeeded == {"midplane": 1}
    assert paths.profile_cache("Psi_N", "currdens", 100, "midplane").is_file()
    assert not paths.profile_cache("Psi_N", "currdens", 200, "midplane").is_file()


def test_gather_profiles_average_failure_does_not_abort_other_modes(jrun, paths, monkeypatch):
    """The key resilience property: `average` dying (Jorek2Error, from the
    hard Fortran `stop` inside trace_fieldlines once flux surfaces are gone)
    must not prevent `midplane` from being gathered, and must not raise out
    of gather_profiles at all."""
    def fake_extract(run, step, var, coords_var, *, tor_mode, **kwargs):
        if tor_mode == "average":
            raise Jorek2Error("jorek2_postproc exited 1: incomplete poloidal turn")
        return np.array([0.1]), np.array([1.0])

    monkeypatch.setattr(profiles_mod, "extract_profile", fake_extract)

    with pytest.warns(UserWarning, match="skipping profile.*mode 'average'"):
        succeeded = gather_profiles(
            jrun, paths, [100], ["currdens"],
            coords_var="Psi_N", tor_modes=["midplane", "average"], n_workers=1,
        )

    assert succeeded == {"midplane": 1, "average": 0}
    assert paths.profile_cache("Psi_N", "currdens", 100, "midplane").is_file()
    assert not paths.profile_cache("Psi_N", "currdens", 100, "average").is_file()


def test_gather_profiles_all_cached_returns_without_calling_extract(jrun, paths, monkeypatch):
    _write_cache(paths, "Psi_N", "currdens", 100, "midplane", [0.1], [1.0])

    calls = []
    monkeypatch.setattr(
        profiles_mod, "extract_profile",
        lambda *a, **k: calls.append(1) or (np.array([0.1]), np.array([1.0])),
    )

    succeeded = gather_profiles(
        jrun, paths, [100], ["currdens"], tor_modes=["midplane"], n_workers=1,
    )

    assert calls == []
    assert succeeded == {"midplane": 1}


def test_gather_profiles_on_progress_fires_per_success(jrun, paths, monkeypatch):
    monkeypatch.setattr(
        profiles_mod, "extract_profile",
        lambda run, step, var, coords_var, **kwargs: (np.array([0.1]), np.array([1.0])),
    )

    progress = []
    gather_profiles(
        jrun, paths, [100, 200], ["currdens"],
        coords_var="Psi_N", tor_modes=["midplane"], n_workers=1,
        on_progress=lambda done, total, step, var, mode: progress.append((done, total, step, var, mode)),
    )

    assert len(progress) == 2
    assert all(total == 2 for _, total, *_ in progress)
