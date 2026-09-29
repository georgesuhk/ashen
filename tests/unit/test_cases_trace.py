"""ashen.cases -- a case's trace_* fields, for `bin/trace`."""

from __future__ import annotations

import textwrap

import pytest

from ashen.cases import CasesError, load_cases
from ashen.particle_programs import PROGRAMS

RE_GC = "re_gc_current_density_initialisation"

_TRACED = f"""
[cases.a]
steps            = [1]
trace_program    = "{RE_GC}"
trace_start_step = 3000
"""


def _load(tmp_path, body: str):
    path = tmp_path / "cases.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_cases(path)


def test_a_case_without_trace_fields_does_not_trace(tmp_path):
    case = _load(tmp_path, "[cases.a]\nsteps = [1]\n")["a"]
    assert case.trace_program is None
    assert case.trace_start_step is None


def test_minimal_trace_takes_the_defaults(tmp_path):
    case = _load(tmp_path, _TRACED)["a"]
    assert case.trace_program == RE_GC
    assert case.trace_start_step == 3000
    assert case.trace_end_step is None
    assert case.trace_particles is None
    assert case.trace_n_mpi == 1
    assert case.trace_omp_threads == 0
    assert case.trace_exe is None


@pytest.mark.parametrize("program", sorted(PROGRAMS))
def test_every_known_program_is_accepted(tmp_path, program):
    assert _load(tmp_path, _TRACED.replace(RE_GC, program))["a"].trace_program == program


def test_defaults_seed_trace_fields(tmp_path):
    """[defaults] works for trace_* like any other case key."""
    cases = _load(tmp_path, f"""
        [defaults]
        trace_program = "{RE_GC}"
        trace_n_mpi   = 4

        [cases.a]
        steps            = [1]
        trace_start_step = 3000

        [cases.b]
        steps            = [1]
        trace_start_step = 5000
        trace_n_mpi      = 8
    """)
    assert (cases["a"].trace_program, cases["a"].trace_n_mpi) == (RE_GC, 4)
    assert (cases["b"].trace_start_step, cases["b"].trace_n_mpi) == (5000, 8)


def test_trace_particles_resolve_against_cases_toml(tmp_path):
    case = _load(tmp_path, _TRACED + 'trace_particles = "seeds/part.h5"\n')["a"]
    assert case.trace_particles == tmp_path / "seeds" / "part.h5"


def test_trace_particles_rejected_for_a_program_that_makes_its_own(tmp_path):
    body = _TRACED.replace(RE_GC, "ex7_jorek") + 'trace_particles = "part.h5"\n'
    with pytest.raises(CasesError, match="always makes its own particles"):
        _load(tmp_path, body)


def test_trace_fields_without_a_program(tmp_path):
    with pytest.raises(CasesError, match=r"sets \['trace_start_step'\] but no trace_program"):
        _load(tmp_path, "[cases.a]\nsteps = [1]\ntrace_start_step = 3000\n")


def test_program_without_a_start_step(tmp_path):
    body = "\n".join(l for l in _TRACED.splitlines() if not l.startswith("trace_start_step"))
    with pytest.raises(CasesError, match="no trace_start_step"):
        _load(tmp_path, body)


@pytest.mark.parametrize("extra, message", [
    ("trace_end_step = 2000", "before trace_start_step"),
    ("trace_n_mpi = 0", "trace_n_mpi must be >= 1"),
    ("trace_omp_threads = -1", "trace_omp_threads >= 0"),
    ("trace_t_span = 1e-5", "unknown key"),
])
def test_invalid_trace_settings(tmp_path, extra, message):
    with pytest.raises(CasesError, match=message):
        _load(tmp_path, _TRACED + extra + "\n")


def test_unknown_program(tmp_path):
    with pytest.raises(CasesError, match="trace_program must be one of"):
        _load(tmp_path, _TRACED.replace(RE_GC, "ex9_jorek"))


def test_negative_start_step(tmp_path):
    with pytest.raises(CasesError, match="trace_start_step must be >= 0"):
        _load(tmp_path, _TRACED.replace("3000", "-1"))
