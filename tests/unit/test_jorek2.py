"""ashen.jorek2.run_tool, exercised against a stub "jorek2_*" executable.

No real JOREK involved -- the stub just echoes stdin and its own cwd listing
to two files, which is enough to verify staging (restart/namelist/profiles/
extra_files all land in the scratch dir under the right names), stdin
plumbing (both stdin_text and stdin_is_namelist modes), and error handling
(non-zero exit, a missing declared output) without needing jorek2_poincare or
jorek2_postproc to exist.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from ashen.jorek2 import Jorek2Error, Jorek2Run, MissingRestartError, run_tool, run_zero_d
from ashen.paths import RunPaths

TOOL_NAME = "stub_tool.cmd" if os.name == "nt" else "stub_tool"

_STUB_PAYLOAD = """
import os
import sys

data = sys.stdin.read()
with open("stdin_echo.txt", "w") as f:
    f.write(data)
with open("cwd_listing.txt", "w") as f:
    f.write("\\n".join(sorted(os.listdir("."))))
with open("env_echo.txt", "w") as f:
    f.write(os.environ.get("OMP_NUM_THREADS", "<unset>"))
for name in os.environ.get("STUB_GLOB_FILES", "").split(os.pathsep):
    if name:
        with open(name, "w") as f:
            f.write("x")
sys.stdout.write(os.environ.get("STUB_STDOUT", ""))
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
"""


def _make_stub_tool(directory: Path, name: str) -> Path:
    """Writes a stub executable named exactly ``name`` into ``directory``.

    Wraps the same Python payload in a ``.cmd`` (Windows) or a ``#!/bin/sh``
    shim (POSIX) rather than relying on a Python shebang directly -- Windows
    has no reliable way to exec a shebang script without shell=True, and this
    keeps run_tool's own invocation (``subprocess.run([str(exe)], ...)``,
    no shell) exercised as written.
    """
    payload = directory / "_stub_payload.py"
    payload.write_text(_STUB_PAYLOAD, encoding="utf-8")

    exe = directory / name
    if os.name == "nt":
        exe.write_text(f'@echo off\r\n"{sys.executable}" "{payload}"\r\n', encoding="utf-8")
    else:
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{payload}"\n', encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return exe


@pytest.fixture
def stub_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_stub_tool(run_dir, TOOL_NAME)

    namelist = run_dir / "in_main"
    namelist.write_text("&in1\n eta = 1.d-6\n&end\n", encoding="utf-8")

    (run_dir / "jorek000100.h5").write_bytes(b"fake-h5-data")
    (run_dir / "T_prof.dat").write_text("1.0 2.0\n", encoding="utf-8")
    (run_dir / "ffprime_prof.dat").write_text("1.0 3.0\n", encoding="utf-8")
    # rho_prof.dat is deliberately absent: run_tool must skip a profile that
    # doesn't exist for this run rather than failing to stage it.

    return Jorek2Run(
        run_dir=run_dir,
        exe_dir=run_dir,
        namelist=namelist,
        pad_width=6,
        profiles=("T_prof.dat", "rho_prof.dat", "ffprime_prof.dat"),
    )


def test_stages_restart_namelist_and_present_profiles(stub_run, tmp_path):
    collected = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["cwd_listing.txt", "stdin_echo.txt"],
        stdin_text="hello", restart_name="jorek_restart.h5",
    )
    listing = collected["cwd_listing.txt"].read_text(encoding="utf-8").splitlines()
    assert "jorek_restart.h5" in listing
    assert "in_main" in listing
    assert "T_prof.dat" in listing
    assert "ffprime_prof.dat" in listing
    assert "rho_prof.dat" not in listing
    assert collected["stdin_echo.txt"].read_text(encoding="utf-8") == "hello"


def test_stdin_is_namelist_pipes_the_copied_namelist(stub_run, tmp_path):
    collected = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["stdin_echo.txt"], stdin_is_namelist=True,
    )
    assert collected["stdin_echo.txt"].read_text(encoding="utf-8") == (
        stub_run.namelist.read_text(encoding="utf-8")
    )


def test_extra_files_are_staged(stub_run, tmp_path):
    collected = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["cwd_listing.txt"], stdin_text="x",
        extra_files={"stpts": "# n_lines\n   1\n"},
    )
    assert "stpts" in collected["cwd_listing.txt"].read_text(encoding="utf-8").splitlines()


def test_copy_exe_stages_a_copy_of_the_tool_itself(stub_run, tmp_path):
    collected = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["cwd_listing.txt"], stdin_text="x", copy_exe=True,
    )
    assert TOOL_NAME in collected["cwd_listing.txt"].read_text(encoding="utf-8").splitlines()


def test_raises_on_nonzero_exit(stub_run, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_EXIT", "3")
    with pytest.raises(Jorek2Error, match="exited 3"):
        run_tool(
            stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
            outputs=["stdin_echo.txt"], stdin_text="x",
        )


def test_raises_on_missing_declared_output(stub_run, tmp_path):
    with pytest.raises(Jorek2Error, match="did not produce"):
        run_tool(
            stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
            outputs=["never_written.dat"], stdin_text="x",
        )


def test_requires_exactly_one_stdin_mode(stub_run, tmp_path):
    with pytest.raises(ValueError):
        run_tool(stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest", outputs=[])
    with pytest.raises(ValueError):
        run_tool(
            stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest", outputs=[],
            stdin_text="x", stdin_is_namelist=True,
        )


def test_missing_exe_raises(stub_run, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_tool(
            stub_run, "does_not_exist", step=100, dest_dir=tmp_path / "dest",
            outputs=[], stdin_text="x",
        )


def test_missing_restart_raises(stub_run, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_tool(
            stub_run, TOOL_NAME, step=999999, dest_dir=tmp_path / "dest",
            outputs=[], stdin_text="x",
        )


def test_missing_restart_raises_missing_restart_error(stub_run, tmp_path):
    """MissingRestartError is a FileNotFoundError subclass so old catches
    still work, but callers that want to distinguish "no restart for this
    step" from any other missing-file failure can catch it specifically."""
    with pytest.raises(MissingRestartError):
        run_tool(
            stub_run, TOOL_NAME, step=999999, dest_dir=tmp_path / "dest",
            outputs=[], stdin_text="x",
        )


def test_run_zero_d_missing_restart_raises(tmp_path):
    """run_zero_d never stages anything (it runs in place), so it must check
    for the restart file itself rather than relying on jorek2_postproc to
    fail -- this is what lets analyse's zerod loop distinguish "no restart
    yet for this step" from a genuine tool crash and skip just that step."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "jorek2_postproc").write_text("x", encoding="utf-8")
    namelist = run_dir / "in_main"
    namelist.write_text("&in1\n&end\n", encoding="utf-8")
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=namelist, pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)

    with pytest.raises(MissingRestartError):
        run_zero_d(run, 100, paths)


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr


def _fake_zero_d_subprocess_run(run_dir: Path, *, value_si: float = 1.5, value_jorek: float = 9.0):
    """A ``subprocess.run`` stand-in for ``run_zero_d`` -- like the real
    ``jorek2_postproc``, always writes to the one fixed filename
    ``postproc/zeroD_quantities_s<step>.dat``, with a value that depends on
    whether ``jorek-units`` was in the piped script, so a test can tell
    which run produced it. Lets ``run_zero_d``'s exe-exists check pass with
    a placeholder file (real execution is never reached), matching how
    ``test_run_zero_d_missing_restart_raises`` already does this.
    """
    def fake_run(args, *, stdin, cwd, stdout, stderr, env=None):
        content = stdin.read()
        value = value_jorek if "jorek-units" in content else value_si
        out_dir = Path(cwd) / "postproc"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "zeroD_quantities_s000100.dat").write_text(
            f"Time\n{value}\n", encoding="utf-8"
        )
        return _FakeCompleted()

    return fake_run


def test_run_zero_d_si_units_writes_the_default_cache_path(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "jorek2_postproc").write_text("x", encoding="utf-8")
    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000100.h5").write_bytes(b"x")
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)
    monkeypatch.setattr(
        "ashen.jorek2.subprocess.run", _fake_zero_d_subprocess_run(run_dir)
    )

    out = run_zero_d(run, 100, paths, si_units=True)

    assert out == paths.zero_d(100, si_units=True)
    assert out.read_text(encoding="utf-8").strip().endswith("1.5")


def test_run_zero_d_jorek_units_is_renamed_to_its_own_path(tmp_path, monkeypatch):
    """JOREK always writes zeroD_quantities_s<step>.dat regardless of units
    mode -- the jorek-units result must be moved to its own path afterwards,
    not left where a subsequent si-units run would silently overwrite it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "jorek2_postproc").write_text("x", encoding="utf-8")
    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000100.h5").write_bytes(b"x")
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)
    monkeypatch.setattr(
        "ashen.jorek2.subprocess.run", _fake_zero_d_subprocess_run(run_dir)
    )

    out = run_zero_d(run, 100, paths, si_units=False)

    assert out == paths.zero_d(100, si_units=False)
    assert out.read_text(encoding="utf-8").strip().endswith("9.0")
    assert not paths.zero_d(100, si_units=True).is_file()


def test_run_zero_d_both_unit_systems_coexist(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "jorek2_postproc").write_text("x", encoding="utf-8")
    (run_dir / "in_main").write_text("&in1\n&end\n", encoding="utf-8")
    (run_dir / "jorek000100.h5").write_bytes(b"x")
    run = Jorek2Run(run_dir=run_dir, exe_dir=run_dir, namelist=run_dir / "in_main", pad_width=6)
    paths = RunPaths(run_dir, pad_width=6)
    monkeypatch.setattr(
        "ashen.jorek2.subprocess.run", _fake_zero_d_subprocess_run(run_dir)
    )

    si_out = run_zero_d(run, 100, paths, si_units=True)
    jorek_out = run_zero_d(run, 100, paths, si_units=False)

    assert si_out != jorek_out
    assert si_out.read_text(encoding="utf-8").strip().endswith("1.5")
    assert jorek_out.read_text(encoding="utf-8").strip().endswith("9.0")


# --- env / stdout, added for the batched Poincare tracer -------------------------


def test_env_reaches_the_child(stub_run, tmp_path):
    """OMP_NUM_THREADS must be set per invocation rather than inherited --
    site.toml's interactive_prelude exports OMP_NUM_THREADS=10, which used to
    leak into every diagnostic worker at once."""
    collected = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["env_echo.txt"], stdin_text="x", env={"OMP_NUM_THREADS": "3"},
    )
    assert collected["env_echo.txt"].read_text(encoding="utf-8") == "3"


def test_env_is_merged_not_replaced(stub_run, tmp_path, monkeypatch):
    """A child given env= must still see the rest of the parent environment."""
    monkeypatch.setenv("STUB_STDOUT", "from-parent")
    result = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["env_echo.txt"], stdin_text="x",
        env={"OMP_NUM_THREADS": "2"}, capture_stdout=True,
    )
    assert result.stdout == "from-parent"


def test_stdout_is_discarded_unless_requested(stub_run, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_STDOUT", "chatter")
    result = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["stdin_echo.txt"], stdin_text="x",
    )
    assert result.stdout == ""


# --- output_glob, added for jorek2_four's dynamically-named outputs --------------


def test_output_glob_collects_matching_files(stub_run, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_GLOB_FILES", "Psi_modes_n000" + os.pathsep + "u_modes_n000")
    result = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        output_glob="*_modes_n???", stdin_text="x",
    )
    assert set(result.outputs) == {"Psi_modes_n000", "u_modes_n000"}
    assert result["Psi_modes_n000"].read_text(encoding="utf-8") == "x"


def test_output_glob_and_outputs_can_combine(stub_run, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_GLOB_FILES", "Psi_modes_n000")
    result = run_tool(
        stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
        outputs=["stdin_echo.txt"], output_glob="*_modes_n???", stdin_text="x",
    )
    assert set(result.outputs) == {"stdin_echo.txt", "Psi_modes_n000"}


def test_output_glob_raises_when_nothing_matches(stub_run, tmp_path):
    with pytest.raises(Jorek2Error, match="produced no output matching"):
        run_tool(
            stub_run, TOOL_NAME, step=100, dest_dir=tmp_path / "dest",
            output_glob="*_modes_n???", stdin_text="x",
        )
