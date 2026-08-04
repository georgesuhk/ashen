# Ashen

A wrapper around [JOREK](https://www.jorek.eu/) for preparing, running and
analysing shots. Extracted from `castor3d/util/`, where the JOREK tooling had
accumulated inside an older CASTOR3D-only project.

Two rules keep this package portable, and both are load-bearing:

1. **No module contains an absolute machine path.** They all come from
   `site.toml` (see below).
2. **No library module touches `sys.path`.** Only the shims in `bin/` do, and
   they derive it from their own location.

The previous arrangement violated both -- `sys.path.append("/tokp/work/geosu/castor3d/")`
was copied into six files, two of them libraries -- which is what made the code
unusable anywhere but one directory on one cluster.

## Getting it importable

In preference order. None of these involves an absolute path.

**1. Just run it (no setup).** The shims put `src/` on `sys.path` relative to
themselves, so a bare clone works:

```bash
git clone <url> ~/ashen
python ~/ashen/bin/run_jorek --show-config
```

**2. For notebooks and interactive use.** One line in `~/.bashrc`:

```bash
export PYTHONPATH=$HOME/ashen/src:$PYTHONPATH
```

**3. If `pip` is available** (usually is, into user site-packages -- not required):

```bash
pip install --user -e ~/ashen
```

Requires Python **3.11+** (stdlib `tomllib`). The HPC runs 3.13; keep syntax
within 3.11 so the package imports in both places.

## Configuring a campaign

Copy `site.example.toml` to your campaign root as `site.toml`:

```
Columbia/NL_kinks/site.toml
```

Running from any folder underneath finds it by walking up. **Relative entries
resolve against the config file's directory**, never the current directory:

```toml
[paths]
exe        = "./exe"
template   = "./template"
jorek_re   = "../jorek_RE"
```

So the campaign describes itself. Rename `NL_kinks`, move the whole tree, or
clone it to another machine, and nothing needs editing -- the property the old
code lacked, where the campaign name was baked into `run_jorek.py` *and* into
every `shotfile.py`.

Check what was picked up:

```bash
python ~/ashen/bin/run_jorek --show-config
```

It prints where `site.toml` was found, what each key resolved to, and flags
targets that do not exist on this machine.

Resolution order: `$ASHEN_SITE` → nearest `site.toml` walking up from the
current directory → `~/.config/ashen/site.toml`.

`site.toml` is gitignored; only `site.example.toml` is tracked.

## Layout

```
bin/            entry-point shims; the only place sys.path is touched
src/ashen/
  config.py     site.toml discovery and path resolution
  namelist.py   Fortran namelist reading and editing
  paths.py      run-folder conventions, restart-step padding
  physics.py    constants used on the JOREK path
  castor_io.py  shared CASTOR3D two-column file parser
  boundary.py   plasma boundary geometry, psi-grid extension
  profiles.py   CASTOR3D -> JOREK profile translation
  cli/          argument handling, importable for testing
tests/
  unit/         run anywhere, no JOREK needed
  golden/       reference run-folder outputs (Phase 3)
  fixtures/     vendored CASTOR3D inputs
```

**`KNOWN_ISSUES.md`** tracks physics-affecting behaviour found during the port
and deliberately left unfixed pending George's judgement -- read it before
touching `profiles.py`.

## Status

Phase 2 of the refactor: the leaf modules above are complete and unit-tested,
several verified numerically against real CASTOR3D fixture data. `shotfile.py`
and `runner.py` (Phase 3, what actually makes `run_jorek` runnable) do not
exist yet -- the CLI's run stages still print "not implemented". Use the
existing `Columbia/run_jorek.py` until Phase 3 lands.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/unit -q    # Windows
python -m pytest tests/unit -q                      # HPC
```

Unit tests must pass on both Windows and the HPC, so they never follow symlinks
or require JOREK.

## Related code

`Columbia/jorek_RE/` is **vendored upstream** from the ITER JOREK repository
(`ssh://git@git.iter.org/stab/jorek.git`). Never modify it. It ships tooling
worth reaching for rather than reimplementing -- `util/setinput.sh`,
`util/continue_run.sh`, `util/select_restart_files.py`, `util/convert2vtk.sh`.
