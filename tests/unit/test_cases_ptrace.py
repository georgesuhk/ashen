"""ashen.cases -- a case's ptrace_* fields, for `bin/ptrace`."""

from __future__ import annotations

import textwrap

import pytest

from ashen.cases import CasesError, load_cases

RE_GC = "./exe/re_gc_current_density_initialisation"

_TRACED = f"""
[cases.a]
steps            = [1]
ptrace_exe        = "{RE_GC}"
ptrace_start_step = 3000
"""


def _load(tmp_path, body: str):
    path = tmp_path / "cases.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_cases(path)


def test_a_case_without_trace_fields_does_not_trace(tmp_path):
    case = _load(tmp_path, "[cases.a]\nsteps = [1]\n")["a"]
    assert case.ptrace_exe is None
    assert case.ptrace_start_step is None


def test_minimal_trace_takes_the_defaults(tmp_path):
    case = _load(tmp_path, _TRACED)["a"]
    assert case.ptrace_exe == RE_GC
    assert case.ptrace_start_step == 3000
    assert case.ptrace_end_step is None
    assert case.ptrace_particles is None
    assert case.ptrace_n_mpi == 1
    assert case.ptrace_omp_threads == 0


def test_any_executable_is_accepted(tmp_path):
    """Not only the programs ashen recognises -- anything runs as-is."""
    case = _load(tmp_path, _TRACED.replace(RE_GC, "../bin/my_tracer"))["a"]
    assert case.ptrace_exe == "../bin/my_tracer"


def test_defaults_seed_trace_fields(tmp_path):
    """[defaults] works for ptrace_* like any other case key."""
    cases = _load(tmp_path, f"""
        [defaults]
        ptrace_exe   = "{RE_GC}"
        ptrace_n_mpi = 4

        [cases.a]
        steps            = [1]
        ptrace_start_step = 3000

        [cases.b]
        steps            = [1]
        ptrace_start_step = 5000
        ptrace_n_mpi      = 8
    """)
    assert (cases["a"].ptrace_exe, cases["a"].ptrace_n_mpi) == (RE_GC, 4)
    assert (cases["b"].ptrace_start_step, cases["b"].ptrace_n_mpi) == (5000, 8)


def test_trace_particles_kept_as_written(tmp_path):
    """Resolved against the run folder at trace time (tracing.run_path),
    like ptrace_exe -- not against cases.toml."""
    case = _load(tmp_path, _TRACED + 'ptrace_particles = "seeds/part.h5"\n')["a"]
    assert case.ptrace_particles == "seeds/part.h5"


def test_trace_particles_allowed_for_any_executable(tmp_path):
    """ashen doesn't judge by name which programs read a particle file."""
    for exe in ("./exe/ex7_jorek", "./exe/my_tracer"):
        body = _TRACED.replace(RE_GC, exe) + 'ptrace_particles = "part.h5"\n'
        assert _load(tmp_path, body)["a"].ptrace_particles == "part.h5"


def test_trace_fields_without_an_exe(tmp_path):
    with pytest.raises(CasesError, match=r"sets \['ptrace_start_step'\] but no ptrace_exe"):
        _load(tmp_path, "[cases.a]\nsteps = [1]\nptrace_start_step = 3000\n")


def test_exe_without_a_start_step(tmp_path):
    body = "\n".join(l for l in _TRACED.splitlines() if not l.startswith("ptrace_start_step"))
    with pytest.raises(CasesError, match="no ptrace_start_step"):
        _load(tmp_path, body)


@pytest.mark.parametrize("extra, message", [
    ("ptrace_end_step = 2000", "before ptrace_start_step"),
    ("ptrace_n_mpi = 0", "ptrace_n_mpi must be >= 1"),
    ("ptrace_omp_threads = -1", "ptrace_omp_threads >= 0"),
    ("trace_program = \"ex7_jorek\"", "unknown key"),
])
def test_invalid_trace_settings(tmp_path, extra, message):
    with pytest.raises(CasesError, match=message):
        _load(tmp_path, _TRACED + extra + "\n")


def test_empty_exe(tmp_path):
    with pytest.raises(CasesError, match="ptrace_exe must be a path"):
        _load(tmp_path, _TRACED.replace(RE_GC, " "))


def test_negative_start_step(tmp_path):
    with pytest.raises(CasesError, match="ptrace_start_step must be >= 0"):
        _load(tmp_path, _TRACED.replace("3000", "-1"))


def test_trace_inputs_kept_as_written(tmp_path):
    case = _load(tmp_path, _TRACED + 'ptrace_inputs = ["ptrace_params.nml", "/abs/x.txt"]\n')["a"]
    assert case.ptrace_inputs == ["ptrace_params.nml", "/abs/x.txt"]


def test_trace_inputs_accepts_a_single_path(tmp_path):
    case = _load(tmp_path, _TRACED + 'ptrace_inputs = "ptrace_params.nml"\n')["a"]
    assert case.ptrace_inputs == ["ptrace_params.nml"]


@pytest.mark.parametrize("value, message", [
    ('["a/p.nml", "b/p.nml"]', r"\['p.nml'\] appear more than once"),
    ('["part_restart.h5"]', "go in ptrace_particles"),
    ("[1]", "must be a list of paths"),
])
def test_invalid_trace_inputs(tmp_path, value, message):
    with pytest.raises(CasesError, match=message):
        _load(tmp_path, _TRACED + f"ptrace_inputs = {value}\n")


# --- plot --diag particles keys ---------------------------------------------------


def test_particle_plot_keys_default_off(tmp_path):
    case = _load(tmp_path, _TRACED)["a"]
    assert case.ptrace_poincare is False
    assert case.ptrace_poincare_psi_n is None
    assert case.ptrace_poincare_n_turns is None
    assert case.ptrace_original_boundary is False


def test_choosing_poincare_lines_or_turns_turns_the_overlay_on(tmp_path):
    case = _load(tmp_path, _TRACED + "ptrace_poincare_n_turns = 200\n")["a"]
    assert case.ptrace_poincare and case.ptrace_poincare_n_turns == 200
    case = _load(tmp_path, _TRACED + "ptrace_poincare_psi_n = [0.5, 0.9]\n")["a"]
    assert case.ptrace_poincare and case.ptrace_poincare_psi_n == [0.5, 0.9]


def test_poincare_psi_n_takes_a_range_like_psi_n_in(tmp_path):
    case = _load(
        tmp_path, _TRACED + "ptrace_poincare_psi_n = { start = 0.2, stop = 0.8, n = 4 }\n"
    )["a"]
    assert case.ptrace_poincare_psi_n == pytest.approx([0.2, 0.4, 0.6, 0.8])


@pytest.mark.parametrize("extra, message", [
    ("ptrace_poincare = 1", "ptrace_poincare must be true or false"),
    ('ptrace_original_boundary = "yes"', "ptrace_original_boundary must be true or false"),
    ("ptrace_poincare_n_turns = 0", "ptrace_poincare_n_turns must be a whole number >= 1"),
    ("ptrace_poincare_n_turns = 2.5", "ptrace_poincare_n_turns must be a whole number >= 1"),
])
def test_invalid_particle_plot_keys(tmp_path, extra, message):
    with pytest.raises(CasesError, match=message):
        _load(tmp_path, _TRACED + extra + "\n")


def test_exit_keys(tmp_path):
    case = _load(tmp_path, _TRACED)["a"]
    assert (case.ptrace_exit_psi_n, case.ptrace_exit_bins) == (1.0, 72)
    case = _load(tmp_path, _TRACED + "ptrace_exit_psi_n = 1\nptrace_exit_bins = 36\n")["a"]
    assert (case.ptrace_exit_psi_n, case.ptrace_exit_bins) == (1.0, 36)
    assert isinstance(case.ptrace_exit_psi_n, float)


@pytest.mark.parametrize("extra, message", [
    ("ptrace_exit_psi_n = 0", "ptrace_exit_psi_n must be a number > 0"),
    ('ptrace_exit_psi_n = "1"', "ptrace_exit_psi_n must be a number > 0"),
    ("ptrace_exit_bins = 0", "ptrace_exit_bins must be a whole number >= 1"),
])
def test_invalid_exit_keys(tmp_path, extra, message):
    with pytest.raises(CasesError, match=message):
        _load(tmp_path, _TRACED + extra + "\n")


def test_particle_color(tmp_path):
    assert _load(tmp_path, _TRACED)["a"].ptrace_particle_color == "red"
    case = _load(tmp_path, _TRACED + 'ptrace_particle_color = "#ff8800"\n')["a"]
    assert case.ptrace_particle_color == "#ff8800"


@pytest.mark.parametrize("value", ['"notacolour"', "3"])
def test_invalid_particle_color(tmp_path, value):
    with pytest.raises(CasesError, match="ptrace_particle_color must be a matplotlib colour"):
        _load(tmp_path, _TRACED + f"ptrace_particle_color = {value}\n")


# --- ptrace_<setting>: ptrace_gc's &ptrace settings ------------------------------


def test_ptrace_settings_are_collected_and_normalised(tmp_path):
    case = _load(tmp_path, _TRACED + textwrap.dedent("""
        ptrace_dt          = 1e-10
        ptrace_initialiser = "current_pdf_simple"
        ptrace_n_markers   = 1000
        ptrace_E_kin_eV    = 10000000
        ptrace_R0          = [1.5, 1.6]
        ptrace_charge      = -1
        ptrace_hold_last_field = true
    """))["a"]
    assert case.ptrace_settings == {
        "dt": 1e-10, "initialiser": "current_pdf_simple", "n_markers": 1000,
        "E_kin_eV": [1e7], "R0": [1.5, 1.6], "charge": [-1], "hold_last_field": True,
    }
    assert isinstance(case.ptrace_settings["E_kin_eV"][0], float)


def test_ptrace_setting_names_ignore_case(tmp_path):
    case = _load(tmp_path, _TRACED + "ptrace_e_kin_ev = 1e7\n")["a"]
    assert case.ptrace_settings == {"E_kin_eV": [1e7]}
    with pytest.raises(CasesError, match="sets ptrace_E_kin_eV twice"):
        _load(tmp_path, _TRACED + "ptrace_e_kin_ev = 1e7\nptrace_E_kin_eV = 2e7\n")


def test_no_ptrace_settings_by_default(tmp_path):
    assert _load(tmp_path, _TRACED)["a"].ptrace_settings == {}


def test_ptrace_settings_from_defaults(tmp_path):
    cases = _load(tmp_path, f"""
        [defaults]
        ptrace_dt = 1e-10

        [cases.a]
        steps             = [1]
        ptrace_exe        = "{RE_GC}"
        ptrace_start_step = 3000
        ptrace_dt         = 2e-10
    """)
    assert cases["a"].ptrace_settings == {"dt": 2e-10}


@pytest.mark.parametrize("extra, message", [
    ('ptrace_initialiser = "uniform"', r"ptrace_initialiser must be one of \['markers', 'current_pdf_simple'\]"),
    ('ptrace_field_mode = "frozen"', "ptrace_field_mode must be one of"),
    ('ptrace_dt = "1e-10"', "ptrace_dt must be a number"),
    ("ptrace_n_markers = 2.5", "ptrace_n_markers must be a whole number"),
    ("ptrace_hold_last_field = 1", "ptrace_hold_last_field must be true or false"),
    ('ptrace_R0 = [1.5, "x"]', "ptrace_R0 must be a number or a list of numbers"),
    ("ptrace_charge = [-1.5]", "ptrace_charge must be a whole number or a list"),
    ("ptrace_restart_index = 1", "unknown key"),
    ("ptrace_settings = { dt = 1e-10 }", "one key each"),
])
def test_invalid_ptrace_settings(tmp_path, extra, message):
    with pytest.raises(CasesError, match=message):
        _load(tmp_path, _TRACED + extra + "\n")
