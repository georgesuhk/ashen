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

## Gathering analysis data for a case

Copy `cases.example.toml` to a campaign folder as `cases.toml` and define a
named entry per investigation -- see the file for the format. Then, from
anywhere:

```bash
python ~/ashen/bin/analyse --list                          # show defined cases
python ~/ashen/bin/analyse --case "qa2.1_g2.3/eta1e-3_RE" --diag zerod --diag poincare --diag profiles
```

`--force` re-runs even where a cached `.npz`/`postproc/*.dat` already exists
(default: skip anything already cached). This gathers and caches data only --
plotting is not ported yet, see Status below.

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
  shotfile.py   ShotParams dataclass + validating loader
  fs.py         copy/symlink helpers used when populating a run folder
  runner.py     prepare_run() + submit_*() -- what bin/run_jorek drives
  postproc.py   jorek2_postproc control scripts + output parsers
  jorek2.py     shared stage/run/collect runner for jorek2_* tools
  diagnostics/  poincare.py, profiles.py -- thin wrappers over jorek2.py
  cases.py      cases.toml loader for bin/analyse
  cli/          argument handling, importable for testing
tests/
  unit/         run anywhere, no JOREK needed
  golden/       reference run-folder outputs, captured from the real HPC script
  fixtures/     vendored CASTOR3D inputs
```

**`KNOWN_ISSUES.md`** tracks physics-affecting behaviour found during the port
and deliberately left unfixed pending George's judgement -- read it before
touching `profiles.py`.

## Status

Phases 1-4 (data-gathering half) are built. `run_jorek` prepares and submits
runs end-to-end; `analyse` gathers and caches zeroD/Poincare/profile data.
**Not yet ported:** the matplotlib plotting layer on top of that cached data
(`plot_poincare`, `plot_field_line_diffusion`, etc. from
`castor3d/util/data_jorek.py`) -- see `KNOWN_ISSUES.md` #4. Use the existing
`Columbia/NL_kinks/analysis.py` for plotting against Ashen-gathered caches in
the meantime.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/unit tests/golden -q    # Windows
python -m pytest tests/unit tests/golden -q                      # HPC
```

Unit tests must pass on both Windows and the HPC, so they never follow
symlinks or require JOREK. The golden suite compares `prepare_run`'s output
against a real HPC capture (`tests/golden/reference/`) and skips itself if
that reference hasn't been captured.

## Related code

`Columbia/jorek_RE/` is **vendored upstream** from the ITER JOREK repository
(`ssh://git@git.iter.org/stab/jorek.git`). Never modify it. It ships tooling
worth reaching for rather than reimplementing -- `util/setinput.sh`,
`util/continue_run.sh`, `util/select_restart_files.py`, `util/convert2vtk.sh`.
