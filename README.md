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
named entry per investigation -- see the file for the format. A case's name
*is* the run folder it points at (`[cases."qa2.1_g2.3/eta1e-3_RE"]` reads
that exact path) -- there is no separate `folder` key to override it. Then,
from anywhere:

```bash
python ~/ashen/bin/analyse --list                          # show defined cases
python ~/ashen/bin/analyse --case "qa2.1_g2.3/eta1e-3_RE" --diag zerod --diag poincare --diag profiles --diag four
```

`--force` re-runs even where cached output already exists (default: reuse it).
This gathers and caches data only; `bin/plot` (below) draws figures from it.

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

### Radial profiles (jorek2_postproc)

`--diag profiles` gathers `case.vars` against `case.coords_var` (default
`"R"`; use `"Psi_N"` for a psi-normalised profile) at every requested step,
cached to `postproc/<coords_var>_<var>_<step>.npz` (or `..._<mode>_<step>.npz`
for anything other than plain `midplane` -- see below). `q` and `Jgrad` are
compound vars expanded into their components (`r_minor`/`Btheta`/`Btor`, and
`currdens`/`Btheta`/`Btor`/`r_minor` respectively) before gathering.

`case.tor_mode` is a list -- a bare string is accepted and normalised to a
one-element list -- of any of `"midplane"`, `"midplane outer"`,
`"midplane inner"`, `"average"`. Listing more than one gathers the same
variables multiple ways for comparison, e.g. `tor_mode = ["midplane outer",
"average"]` for **toroidal current density vs `Psi_N`**:

```toml
coords_var = "Psi_N"
vars       = ["currdens"]
tor_mode   = ["midplane outer", "average"]
```

**Use `"midplane outer"`, not bare `"midplane"`, when `coords_var = "Psi_N"`.**
Bare `midplane` cuts through the magnetic axis, so `Psi_N` runs `1 -> 0 -> 1`
along it and the profile is double-valued; `midplane outer` is single-sided
and monotone while surfaces stay nested. Neither does any flux-surface
finding or field-line tracing -- `Psi_N` is just an ordinary pointwise
expression, evaluated the same way `currdens` is -- so it always produces a
profile, at every step, regardless of how disrupted the field is. It is a
**cut**, not a flux average, though: the value at a point is the local field
at that point on that line, not `<J>` over the whole surface.

**`average` is a genuine flux-surface average, and it can fail.** It traces
field lines to build each flux surface, which means it depends on those
surfaces actually existing and closing -- something a nonlinear kink/tearing
run can violate at exactly the timesteps of physics interest. When it fails
the underlying JOREK process exits non-zero (occasionally via a hard abort
inside the tracer); `gather_profiles` catches this **per step**, warns, and
moves on rather than losing the rest of the gather -- see `KNOWN_ISSUES.md`
#9 for the full mechanism. `Case.profile_rad_range` (default `[0.001,
0.999]`, matching `jorek2_postproc`'s own default) is the main lever for
keeping it alive longer: pulling the upper bound in off the separatrix keeps
tracing inside the still-nested core. `profile_surfaces`/`profile_nmaxsteps`/
`profile_deltaphi` are the remaining tracing knobs, all `average`-only and
separate from the identically-named `jorek2_four` fields (`nstpts` etc.) --
same physical parameters, different consumer.

After gathering, `analyse` prints a warning naming any `tor_mode` that
produced **zero** profiles across every step -- distinct from a mode that
partly worked (expected for `average` on a disrupting run) versus one that
never worked at all (more likely a configuration problem, e.g.
`profile_rad_range` set too wide for this case from the start).

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
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag poincare --diag connection_length --diag four --diag profiles
```

Kept as a separate command from `analyse` on purpose: gathering is slow and
batch, plotting is fast and iterative, and re-plotting should never risk
touching the gathering path.

- `--step N` (repeatable) restricts every diag drawn this run to specific
  steps, overriding any configured per-diag steps (default: each diag's own
  steps -- see below).
- `--linear` / `--smooth` control the connection-length colour maps.
- `--psi-range MIN MAX` further bounds-filters whichever psi_n_in list is
  already in effect for connection-length -- plot-time only, no re-gather
  needed.
- `--four-linear` draws four's mode amplitudes on a linear scale instead of
  the default log.
- `--dpi N` overrides the figure resolution.

### Different step ranges for different diags

Steps resolve through a three-tier `default -> case -> case+diag` tree, most
specific wins. `[defaults]`/a case's own `steps` (both already existed) are
the first two tiers; a nested `[cases.NAME.<diag>]` table with its own
`steps` -- `<diag>` being one of `zerod`, `poincare`, `profiles`, `four`,
`connection_length` -- is the third, e.g. a long range for `four`'s growth
curve alongside a handful of `poincare` snapshots for the same run:

```toml
[cases."qa2.1_g2.3/eta1e-3_RE"]
steps = { start = 200, stop = 5800, step = 200 }

[cases."qa2.1_g2.3/eta1e-3_RE".four]
steps = { start = 200, stop = 12000, step = 200 }   # four gets a longer range

[cases."qa2.1_g2.3/eta1e-3_RE".poincare]
steps = [200, 3000, 5800]                            # poincare only these
```

Both `analyse` and `plot` respect it -- `analyse --diag four` gathers exactly
the `four`-overridden steps, not the case's plain ones. `--step` on the
`plot` command line still outranks all three config tiers when given.
`connection_length`'s own override only *selects which already-gathered
poincare steps to plot* (same "no interpolation" rule `lc_psi_n_in` has
below) -- it never gathers new data on its own, so its steps must be a
subset of whatever `poincare` (or its own override) actually traced.
`[defaults.<diag>]` works too, seeding every case, but is replaced wholesale
-- not merged key-by-key -- by a case's own `[cases.NAME.<diag>]` table.

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

### Highlighting rational surfaces in Poincare plots

Setting `poincare_highlight = true` (plus `poincare_highlight_modes`,
`[m, n]` pairs like `four_modes`, and a parallel `poincare_highlight_colors`)
colours the field lines nearest each mode's `q = m/n` resonant surface,
dimming everything else to grey. It needs the q-profile cache -- `analyse
--diag poincare` gathers it automatically once this is set, the same way it
already force-includes `zerod`. Because Poincare only traces the discrete
`psi_n_in` grid requested, a computed rational surface is snapped to the
*nearest actually-traced* line rather than requiring an exact match; which
physical line that is can shift step to step as the q-profile evolves --
that tracks the real resonance moving, not a bug.

Covered this pass: Poincare puncture plots, the LC/LCTT connection-length
maps, jorek2_four mode-amplitude time series, and radial profiles -- see
below. Everything else the legacy `data_jorek.py` plotted (macroscopic-variable
traces, field-line diffusion, the derived q-profile/`dJ/dr`, stochastic
factor) is not ported -- see `KNOWN_ISSUES.md` #8. Colour is by each line's
`psi_n` through a colormap by default (`ashen.plotting.colors`); a discrete
palette is also available.

### Radial profiles

`--diag profiles` draws one figure per `(coords_var, var)` gathered by
`analyse --diag profiles` -- one **panel per `tor_mode`** (sharing a y-axis),
one **line per restart step** within each panel, coloured by true time (from
the zeroD cache) or step index if that cache is incomplete. Saved to
`<coords_var>_<var>_profile.png`.

A `tor_mode` with no cached data still gets its (empty, labelled) panel rather
than being silently dropped -- when comparing e.g. `["midplane outer",
"average"]`, that `average`'s panel is empty (or thins out partway through
the step sequence) *is* the result: it shows directly where the flux-surface
average stopped being computable, rather than requiring a run through the
warnings `analyse` printed at gather time. See `KNOWN_ISSUES.md` #9.

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
variables/modes get drawn; empty (default) draws everything found in the
cache. `four_modes` entries are `[m, n]` pairs (poloidal, toroidal) -- `[3,
2]` is `m=3, n=2`, matching how a mode is normally written (`m/n`):

```toml
four_vars  = ["Psi", "T"]
four_modes = [[2, 1], [3, 2], [1, 1]]   # [m, n] pairs
```

A step or `(variable, n, m)` combination missing from the cache shows as a
gap (`nan`) in that line rather than an error. `--four-linear` switches the
default log amplitude scale to linear.

**Rational-surface overlay / `four_quantities`.** `analyse --diag four` also
gathers each step's q-profile (`jorek2_postproc`'s `qprofile` command, cached
to `postproc/qprofile_s<step>.dat`) alongside the Fourier decomposition -- no
separate `--diag` needed. For every `(n, m)` mode with `n != 0`, that cache
locates the mode's resonant surface (`q = m/n`, solved by linearly
interpolating the q-profile's crossings, same as JOREK's own `find_q_surface`
postproc command) and pins the mode's `|amplitude|` to that surface.

`four_quantities` (case field, plot-time only, default `["max"]`) chooses
what actually gets drawn:

```toml
four_quantities = ["max"]                        # default: domain-wide max only
four_quantities = ["rational_surface"]            # only the q=m/n-pinned value
four_quantities = ["max", "rational_surface"]     # both: max solid, rational dashed
```

With both selected, the rational-surface value overlays as a dashed line in
the same colour as its mode's solid max line -- the useful comparison,
whether a mode's growth is actually concentrated at the radius it resonates
on, or the domain-max is being driven by something else (numerical noise
near the axis, a different structure entirely). With `rational_surface`
alone, that value becomes the primary (solid) line instead, and the y-axis
label changes from `max |var|` to `|var| @ rational surface` accordingly.

A reversed-shear q-profile can cross a given `q` more than once; every
crossing is kept and drawn (not just the strongest). `n = 0` modes have no
rational surface (`m/0`) and are never part of the rational-surface series.
A case gathered before this feature existed (no `qprofile_s*.dat` cache), or
one with no `n != 0` modes at all, prints a note and skips the
rational-surface figure rather than erroring -- `four_quantities = ["max"]`
(the default) is unaffected either way.

**Growth rate.** `four_growth_rate = true` (case field, plot-time only) fits
each drawn mode's exponential growth rate -- `gamma` [1/s], the slope of
`ln|amplitude|` vs real time -- and shows it two ways: appended to that
mode's legend label (`n=1, m=2 (γ=1.23e+05 /s)`, on both the step and
time figures, since `gamma` is a single physical number independent of
which x-axis it's shown against) and written to
`four_dir/growth_rates.txt`, one row per `(variable, m, n)`. Needs the
zeroD cache for real time -- skipped with a printed note, not an error, if
it's incomplete, same as the time-axis variant.

`four_growth_steps = [start_step, end_step]` restricts the fit to that
inclusive step range instead of every requested step -- useful for picking
the visually-linear region of a growth curve, since points near the noise
floor (pre-growth) or past saturation bias a whole-range least-squares fit.
A mode with fewer than 2 valid (finite, positive-amplitude) points in the
window is silently omitted from the fit rather than given a meaningless
line:

```toml
four_growth_rate  = true
four_growth_steps = [1000, 3000]   # inclusive; omit to fit every step
```

**`delta_b_over_b`.** A pseudo-variable for `four_vars` -- not a raw
`jorek2_four` output (there is no `B` primitive in JOREK's restart file, only
`Psi`, the poloidal flux), so it's derived from `Psi`'s amplitude using the
standard tearing-mode shorthand:

```
delta_b_over_b(m,n) = (m / R_axis^2) * |Psi_mn| / B_axis
```

`R_axis` and `B_axis = F0/R_axis` (`phys_module.f90`'s fixed-toroidal-field
constant, `B_phi = F0/R`) are both read from the run's `log`
(`ashen.logfile.r_axis`/`b_axis`). This is an approximation: the exact
relation uses the true local minor radius and `|grad Psi|`, not the
(constant) major radius at the magnetic axis, but the four cache only carries
`|Psi_mn|` on a `psi_n` grid, not real-space geometry, so `R_axis` stands in
for it everywhere. `m = 0` modes are dropped (no helical radial-field content
in this shorthand) rather than drawn as a flat zero line.

Only computed when explicitly requested -- an empty/unset `four_vars` never
picks it up, since it isn't "everything found in the cache". Works with
either `four_quantities` selection (whole-domain max or rational-surface
value), since it's a post-conversion of whichever `Psi` series was computed:

```toml
four_vars = ["delta_b_over_b"]              # only the derived quantity
four_vars = ["Psi", "delta_b_over_b"]       # raw flux amplitude alongside it
```

**`delta_b`** is the same quantity un-normalised -- `(m / R_axis^2) *
|Psi_mn|`, in Tesla, with no division by `B_axis`. It only needs `R_axis`
from the log, not `F0`, so it still works on a run whose log doesn't carry
`F0`; `delta_b_over_b` does not. The two can be requested together
(`four_vars = ["delta_b", "delta_b_over_b"]`) and are computed independently,
so a missing `F0` skips only `delta_b_over_b`.

A run whose log is missing `R_axis` prints one `skipping delta_b,
delta_b_over_b: ...` message and drops both (neither can be computed without
it); missing only `F0` prints `skipping delta_b_over_b: ...` and `delta_b`
is still drawn. Either way the rest of the requested `--diag four` plot is
unaffected, not an error.

Either figure also gets a small boxed caption in the lower-right corner
giving the peak value actually drawn -- `max δB = 1.2 T` or `max δB/B =
0.03` -- the largest finite value across every mode and step in that
figure, so a reader doesn't have to eyeball the plot to answer "how big does
this get." No other `four_vars` variable gets a caption.

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
`analyse` gathers and caches zeroD/Poincare/profile/jorek2_four/q-profile data;
`plot` draws Poincare, connection-length, jorek2_four mode-amplitude, and
radial-profile figures from it. **Not yet ported:** the rest of the matplotlib
plotting layer (`plot_field_line_diffusion`, the derived q-profile/`dJ/dr`,
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
