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
python ~/ashen/bin/analyse --case "qa2.1_g2.3/eta1e-3_RE" --diag zerod --diag poincare --diag profiles --diag four
```

`--force` re-runs even where cached output already exists (default: reuse it).
This gathers and caches data only -- plotting is not ported yet, see Status
below.

`--diag poincare` also runs `zerod` even if not requested explicitly: `plot`'s
LCTT figure reads each step's true time from the zeroD cache, so a
poincare-only gather would otherwise leave it with nothing to read. Cache-gated
per step, so this costs nothing once zerod has already run.

### Poincaré scans are incremental

The Poincaré cache stores **one field line per starting point**
(`poinc_dir/poinc_s<step>.h5`), so a scan can be grown rather than repeated:

- adding a `psi_n` to `psi_n_in` traces only the new starting positions
- raising `n_turns` resumes each line from its last puncture and traces only
  the shortfall
- re-running an unchanged request traces nothing at all

Raising `ang_sample_freq` re-samples the flux surface, so only the positions
that coincide with the previous sampling are reused.

`psi_n_in` accepts an explicit list, or a generated range --
`{ start, stop, step }` (fixed spacing) or `{ start, stop, n }` (fixed point
count, like the legacy `np.linspace(min, max, 20)`).

A resumed trace is stitched from more than one integration (`n_segments > 1` on
the record). For stochastic lines it will not match an uninterrupted trace of
the same length point-for-point -- it samples the same field and the same
invariant set, so island widths, diffusion and Poincaré plots are unaffected,
but it is not bit-reproducible. Use `--force` when a result must be.

### Fourier decomposition (jorek2_four)

`--diag four` gathers a toroidal Fourier decomposition of each restart step,
one `jorek2_four` process per step. Unlike Poincare tracing this isn't
incremental -- a decomposition of a single restart isn't resumable, so a step
with an existing cache (`four_dir/four_s<step>.h5`) is skipped whole unless
`--force` is given.

`jorek2_four`'s own `nstpts`/`nTht`/`nmaxsteps`/`deltaphi`/`nsmallsteps`/
`rad_range` knobs (normally read from a hand-written `four_params.nml`) come
from the case's `[defaults]`/`[cases.*]` entries instead -- `analyse` generates
that file itself per step. An unconfigured case reproduces `jorek2_four`'s own
built-in defaults exactly.

`--diag four` also gathers each step's q-profile via `jorek2_postproc`'s
`qprofile` command (cached to `postproc/qprofile_s<step>.dat`, same
cache-gating/`--force` rules as above) -- `plot --diag four`'s rational-surface
overlay needs it; see below.

### Cores

One `jorek2_poincare` or `jorek2_four` process per restart step, each
OpenMP-threaded internally. Set the split in `site.toml`:

```toml
[diagnostics]
n_workers   = 0   # restart steps at once; 0 = derive from cpu_count
omp_threads = 0   # OpenMP threads per process; 0 = min(8, cpu_count)
```

`--n-workers` / `--omp-threads` override per invocation. None of JOREK's
`jorek2_*` tools use MPI, so these two are the only axes that exist.

## Plotting

`bin/plot` draws figures from data `analyse` already gathered -- it never runs
a `jorek2_*` tool itself, and reads the same `cases.toml`:

```bash
python ~/ashen/bin/plot --list
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag poincare --diag connection_length --diag four
```

Kept as a separate command from `analyse` on purpose: gathering is slow and
batch, plotting is fast and iterative, and re-plotting should never risk
touching the gathering path.

- `--step N` (repeatable) restricts Poincare plots to specific steps (default:
  every step in the case).
- `--linear` / `--smooth` control the connection-length colour maps.
- `--psi-range MIN MAX` further bounds-filters whichever psi_n_in list is
  already in effect for connection-length -- plot-time only, no re-gather
  needed.
- `--four-linear` draws four's mode amplitudes on a linear scale instead of
  the default log.
- `--dpi N` overrides the figure resolution.

### Plotting a different psi_n selection than you gathered

`lc_psi_n_in` is a separate, plot-time-only case setting for
`connection_length` -- independent of `psi_n_in`, which controls what
`analyse` actually traces. It takes the same list/range formats as
`psi_n_in`, plus a `{ min, max }` bounds filter over whatever `psi_n_in`
resolved to:

```toml
lc_psi_n_in = [0.3, 0.5, 0.7]                          # explicit subset
lc_psi_n_in = { start = 0.1, stop = 0.9, step = 0.2 }  # generated range
lc_psi_n_in = { min = 0.2, max = 0.8 }                 # bounds filter
```

Omit it to plot every gathered `psi_n_in`, as before. **No interpolation**:
`connection_lengths_for_step` matches a plot-time psi_n against the cache
exactly (same quantisation `LineKey` uses), so a value that wasn't actually
traced during gathering renders as a visible black cell rather than an
error -- widening `lc_psi_n_in` beyond what was gathered doesn't add real
data, only `analyse --diag poincare` with a wider/denser `psi_n_in` does.

Covered this pass: Poincare puncture plots, the LC/LCTT connection-length
maps, and jorek2_four mode-amplitude time series -- see below. Everything else
the legacy `data_jorek.py` plotted (macroscopic-variable traces, field-line
diffusion, radial profiles, stochastic factor) is not ported -- see
`KNOWN_ISSUES.md` #8. Colour is by each line's `psi_n` through a colormap by
default (`ashen.plotting.colors`); a discrete palette is also available.

### jorek2_four mode-amplitude time series

`--diag four` draws one figure per variable, one coloured line per `(n, m)`
mode -- the peak `|amplitude|` over the radial (psi_n) grid at each restart
step, from the caches `analyse --diag four` already wrote
(`four_dir/four_s<step>.h5`). Each restart step is drawn as a marker, joined
by a line, so a sparse or irregular step selection stays legible rather than
implying data between steps that weren't actually gathered.

Two x-axis variants are always written, mirroring connection_length's
LC/LCTT split: `four_dir/<variable>_modes_step.png` (raw step index) and
`four_dir/<variable>_modes_time.png` (true time in microseconds, from the
zeroD cache). The time variant is skipped -- with a printed note, not an
error -- if the zeroD cache doesn't cover every requested step.

`four_vars` and `four_modes` (case fields, plot-time only) restrict which
variables/`(n, m)` pairs get drawn; empty (default) draws everything found in
the cache:

```toml
four_vars  = ["Psi", "T"]
four_modes = [[0, 1], [1, 1], [2, 1]]   # [n, m] pairs
```

A step or `(variable, n, m)` combination missing from the cache shows as a
gap (`nan`) in that line rather than an error. `--four-linear` switches the
default log amplitude scale to linear.

**Rational-surface overlay.** `analyse --diag four` also gathers each step's
q-profile (`jorek2_postproc`'s `qprofile` command, cached to
`postproc/qprofile_s<step>.dat`) alongside the Fourier decomposition -- no
separate `--diag` needed. For every `(n, m)` mode with `n != 0`, `plot --diag
four` uses that cache to locate the mode's resonant surface (`q = m/n`,
solved by linearly interpolating the q-profile's crossings, same as JOREK's
own `find_q_surface` postproc command) and overlays a dashed line pinning the
mode's `|amplitude|` to that surface, in the same colour as its solid
whole-domain-max line. This is the useful comparison: whether a mode's
growth is actually concentrated at the radius it resonates on, or the
domain-max is being driven by something else (numerical noise near the axis,
a different structure entirely).

A reversed-shear q-profile can cross a given `q` more than once; the
strongest of the crossings is kept. `n = 0` modes have no rational surface
(`m/0`) and are drawn without an overlay. Cases gathered before this feature
existed (no `qprofile_s*.dat` cache) simply draw without the overlay -- no
error, no re-gather required for the base plot.

Connection lengths use `R0` extracted from the run's log
(`ashen.logfile.r_axis`) rather than the legacy hardcoded `R0 = 1.36` -- see
`KNOWN_ISSUES.md` #6 and #7 for what changed and what's still an open question.

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
  diagnostics/  poincare.py + poincare_cache.py, profiles.py,
                connection_length.py -- pure math, no matplotlib
  logfile.py    scalar extraction from a JOREK log (R_axis, etc.)
  plotting/     poincare.py, connection_length.py, colors.py, style
  cases.py      cases.toml loader for bin/analyse and bin/plot
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

Phases 1-4 are built. `run_jorek` prepares and submits runs end-to-end;
`analyse` gathers and caches zeroD/Poincare/profile data; `plot` draws Poincare
and connection-length figures from it. **Not yet ported:** the rest of the
matplotlib plotting layer (`plot_field_line_diffusion`, radial profiles,
macroscopic-variable traces, stochastic factor -- see `KNOWN_ISSUES.md` #8).
Legacy `Columbia/NL_kinks/analysis.py` still works for those against
Ashen-gathered caches, except the Poincare cache itself, which moved to a new
per-line HDF5 format legacy plotting cannot read (`KNOWN_ISSUES.md` #5).

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
