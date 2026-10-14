"""Which steps each diag has cached (diagnostics.cached_steps), what
`plot --step all` draws."""

from __future__ import annotations

import pytest

from ashen.cases import Case
from ashen.diagnostics.cached_steps import CACHED_STEP_DIAGS, cached_steps
from ashen.paths import RunPaths


@pytest.fixture
def paths(tmp_path):
    run = tmp_path / "run"
    for sub in ("poinc_dir", "four_dir", "postproc"):
        (run / sub).mkdir(parents=True)
    for step in (0, 200, 400):
        (run / f"jorek{step:06d}.h5").write_bytes(b"")
    return RunPaths(run, pad_width=6)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_each_diag_finds_its_own_caches_at_any_padding(paths):
    case = Case(name="run", steps=[0], vars=["T"], coords_var="Psi_N",
                tor_mode=["midplane", "midplane outer"])
    run = paths.run_dir
    _touch(paths.poincare_cache(200))
    _touch(run / "poinc_dir" / "poinc_s00600.h5")              # another width
    _touch(paths.poincare_cache_legacy(800, "theta"))          # the old format
    _touch(paths.four_cache(400))
    _touch(paths.profile_cache("Psi_N", "T", 200, "midplane"))
    _touch(paths.profile_cache("Psi_N", "T", 400, "midplane outer"))
    _touch(paths.profile_cache("Psi_N", "rho", 600, "midplane"))   # not one of the case's vars
    _touch(paths.profile_cache("Psi_N", "currdens", 200, "midplane outer"))
    _touch(paths.profile_cache("Psi_N", "currdens", 400, "midplane outer"))
    _touch(paths.profile_cache("Psi_N", "recurrdens", 400, "midplane outer"))
    _touch(paths.zero_d(0))
    _touch(run / "postproc" / "zeroD_quantities_s01000.dat")
    _touch(run / "poinc_dir" / "200_poincare.png")              # figures are not caches

    for diag in ("poincare", "connection_length", "theta_hist", "wetted_fraction"):
        assert cached_steps(diag, case, paths) == [200, 600, 800]
    assert cached_steps("four", case, paths) == [400]
    assert cached_steps("profiles", case, paths) == [200, 400]
    assert cached_steps("re_current", case, paths) == [400]      # both densities needed
    assert cached_steps("zerod", case, paths) == [0, 1000]


def test_nothing_cached_is_an_empty_list_and_other_diags_are_refused(tmp_path):
    paths = RunPaths(tmp_path, pad_width=6)                     # not even the subfolders
    case = Case(name="run", steps=[0])
    for diag in CACHED_STEP_DIAGS:
        assert cached_steps(diag, case, paths) == []
    with pytest.raises(ValueError, match="not drawn over restart steps"):
        cached_steps("particles", case, paths)
