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

from ashen.jorek2 import Jorek2Error, Jorek2Run, run_tool

TOOL_NAME = "stub_tool.cmd" if os.name == "nt" else "stub_tool"

_STUB_PAYLOAD = """
import os
import sys

data = sys.stdin.read()
with open("stdin_echo.txt", "w") as f:
    f.write(data)
with open("cwd_listing.txt", "w") as f:
    f.write("\\n".join(sorted(os.listdir("."))))
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
