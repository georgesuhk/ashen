"""Test helpers shared between tests/unit and tests/golden.

Not itself a test module -- imported by both suites' conftest.py so the
symlink-privilege workaround (see below) isn't duplicated.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def symlinks_supported(tmp_path: Path) -> bool:
    """Probe whether this machine can create symlinks without elevation.

    False on Windows without Developer Mode enabled -- confirmed on the
    primary dev machine this project was built on.
    """
    probe_target = tmp_path / "_probe_target"
    probe_target.mkdir()
    probe_link = tmp_path / "_probe_link"
    try:
        probe_link.symlink_to(probe_target, target_is_directory=True)
        return True
    except OSError:
        return False
    finally:
        if probe_link.exists() or probe_link.is_symlink():
            probe_link.unlink()


def install_symlink_bypass(monkeypatch) -> None:
    """Replace ashen.fs's symlink functions with plain-copy equivalents.

    Used where a real machine's `fs.symlink_dir` would create a *dangling*
    symlink if the source doesn't exist yet (real symlinks don't require
    their target to exist) -- e.g. this dev clone has no `Columbia/jobscripts`
    or `Columbia/jorek` directory. A copy can't dangle, so a missing source
    becomes an empty directory instead, which is close enough for verifying
    prepare_run's namelist/profile/boundary-writing logic (the actual point
    of these tests) without needing every referenced directory to exist.
    """
    from ashen import runner as runner_module

    def fake_symlink_dir(src, dst, link_name=None):
        src, dst = Path(src), Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / (link_name or src.name)
        if target.exists():
            shutil.rmtree(target)
        if src.exists():
            shutil.copytree(src, target)
        else:
            target.mkdir()
        return target

    def fake_symlink_files_in(src, dst):
        src, dst = Path(src), Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        created = []
        if not src.exists():
            return created
        for entry in src.iterdir():
            if entry.is_file():
                target = dst / entry.name
                shutil.copy2(entry, target)
                created.append(target)
        return created

    def fake_symlink_file(src, dst):
        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.write_text("")
        return dst

    monkeypatch.setattr(runner_module.fs, "symlink_dir", fake_symlink_dir)
    monkeypatch.setattr(runner_module.fs, "symlink_files_in", fake_symlink_files_in)
    monkeypatch.setattr(runner_module.fs, "symlink_file", fake_symlink_file)
