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

**Choosing cases.** `--case` works the same in `analyse`, `plot` and
`ptrace`. It takes:

- a case's name: `--case "qa2.1_g2.3/eta1e-3_RE"`;
- a pattern, quoted: `--case 'qa2.1*'` is every case whose name starts
  `qa2.1`. `*`, `?` and `[...]` work as in the shell, except that `*` also
  crosses the `/` between folder and run;
- a folder holding cases: `--case qa2.1_g2.3` is every case under it.

Several may follow one `--case`, and `--case` may be repeated. The cases run
in `cases.toml` order, each once. A value that selects nothing is an error,
before anything runs.

Quote patterns. Unquoted, the shell expands `qa2.1*` itself, to the run
folders in the current directory. That usually still works, because each
folder selects the cases under it -- but zsh refuses a pattern that matches
no file, and a stray matching file (`qa2.1_notes.txt`) becomes an unknown
case.

```bash
python ~/ashen/bin/analyse --diag poincare --case 'qa2.1*'
python ~/ashen/bin/plot --diag four --case qa2.1_g2.3 qa2.1_g2.5
```

**Gather, then plot, in one command.** `-plot DIAG` (or `--plot`, repeatable)
runs `plot --diag DIAG` afterwards, for the cases that gathered:

```bash
python ~/ashen/bin/analyse --diag four -plot four
python ~/ashen/bin/analyse --diag poincare -plot connection_length -plot poincare
python ~/ashen/bin/analyse --diag four -plot four --four-linear --dpi 200
```

- `DIAG` is any of plot's diags, spelled as plot spells them
  (`connection_length`, not `connection length`).
- Options `analyse` doesn't know are passed on to `plot`. They are checked
  before the gather starts, so a typo fails at once rather than after it.
- `--cases`, `--case`, `--site` and `--n-workers` apply to both.
- A case whose gather failed is not plotted. The exit status is non-zero
  if either stage failed.
- `-plot` doesn't choose what is gathered: `--diag` does. Plotting a diag
  whose data was never gathered reports that, as `plot` alone would.

`--diag poincare` also runs `zerod` even if not requested explicitly: `plot`'s
LCTT figure reads each step's true time from the zeroD cache, so a
poincare-only gather would otherwise leave it with nothing to read. Cache-gated
per step, so this costs nothing once zerod has already run.

`bin/plot` itself also backfills a missing zeroD step on demand, for any
figure that needs true time (LCTT, `four`'s time-axis/growth-rate, `profiles`'
colour-by-time) -- it's cheap (one `jorek2_postproc` call, no tracing), so a
separate `analyse --diag zerod` pass isn't required first. Reported
concisely, one line per step:

```
  zerod: missing for step(s) [3000, 3200], gathering
  zerod: step 3000 done
  zerod: step 3200 done
```

A step whose restart genuinely doesn't exist (or whose `jorek2_postproc`
call fails for another reason) is reported and skipped the same way, and the
figure falls back to its existing "no zeroD cache" behaviour for that step
rather than crashing the whole `plot` invocation.

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

### Step padding differs between postproc builds

JOREK's two halves are not symmetric about how wide a step index is, which
is what makes a mismatch quiet. *Importing* a restart tries both widths
(`mod_import_restart.f90:2686`), so `jorek08002.h5` and `jorek008002.h5`
read equally well. *Naming an output* uses `rst_file_ind_fmt(1)` alone
(`step_range_string`, `exec_commands.f90:1068`) -- one fixed width,
whatever the restarts happen to use. Builds differ in which entry comes
first, so the width is a property of the binary you ran, not of the run.

The visible symptom was a tool that "succeeded" and a reader that then
failed on a file sitting right there under a name one character longer.

Ashen resolves this on read. Anything **JOREK** writes -- restarts,
`zeroD_quantities_*`, `fluxsurface_at_psi_*`, `qprofile_*`, and the
profile tables collected out of a scratch dir -- is looked up at this run's
own width first and then at the other, so a folder may hold both spellings,
including a run continued under a different build. What gets *collected*
into a cache is filed under one spelling, so nothing downstream needs to
know which build produced it.

Anything **ashen** writes -- the Poincare, profile and jorek2_four caches --
keeps exactly one spelling. A second accepted name there would be a way to
end up with two caches for one step rather than a way to find the one that
exists.

Nothing is renamed on disk: JOREK's output keeps the name JOREK gave it, so
legacy tooling reading the same run folder is unaffected.

All of this is decided in one module, `ashen.padding`; nothing else formats
a step width. It also covers a third case: a JOREK binary that looks a file
up by a name *it* builds, trying only its own width (the particle tracer's
field reader does this). Staging for such a binary uses the width that
build writes, read from `communication/mod_import_restart.f90` in the
checkout it was compiled from. `--show-config` prints it for both `jorek`
and `jorek_re`, e.g.:

```
[restart step width]
  jorek    = 6 digits
  jorek_re = 6 digits
```

### Seeing what the jorek2_* tools are doing

By default a tool's stdout is discarded and its stderr kept back to be quoted
if it exits non-zero -- so a successful gather is quiet, and a failing one
reports the tool's own complaint. Neither helps with a tool that *hangs*,
which prints nothing either way.

`--tool-output` (on `analyse`, `plot` and `timestep`) hands the tool this
terminal instead of capturing it, so its output arrives **live**:

```bash
python ~/ashen/bin/analyse --case NAME --diag zerod --tool-output
```

```
==== qa2.1_g2.3/eta1e-3_RE ====
--- jorek2_postproc step 3000
    exe /path/to/run/jorek2_postproc
    cwd /path/to/run
 reading namelist in_main
 ...
```

The header is printed *before* the launch, so it appears even if the tool
then hangs -- which distinguishes "the exe never started" from "it started
and stopped partway". Because the streams are inherited, a non-zero exit no
longer quotes stderr into the error message (it is already on screen) and
`ToolResult.stdout`/`.stderr` come back empty.

The flag drops `analyse` to one step at a time, since concurrent steps
sharing one terminal interleave unreadably; pass `--n-workers` explicitly to
override that.

`jorek2_poincare` is a special case, but not an exception: its progress
messages are *also* parsed, to tell which output block belongs to which field
line. So rather than choosing, its streams are teed -- drained line by line
and echoed as they arrive, while still being kept for the caller. A failing
tee therefore does quote stderr in the error message, unlike the inherited
case. Note that tracing prints one line per field line, so a wide `psi_n_in`
is loud.

Both echoed streams go to **stderr**, whichever they came from, so `analyse`'s
own progress on stdout stays separable:

```bash
python ~/ashen/bin/analyse --case NAME --diag poincare --tool-output 2> ptrace.log
```

`ASHEN_TOOL_OUTPUT=1` does the same thing without a flag -- for a jobscript,
or a notebook calling into `ashen` directly.

## Plotting

`bin/plot` draws figures from data `analyse` already gathered, and reads the
same `cases.toml`. It never *traces* -- no `jorek2_poincare`, no `four`, none
of the slow work, which is what keeps it fast enough to iterate on. It will,
though, top up three cheap `jorek2_postproc` caches on demand if a figure
needs one and it is missing or unreadable: zeroD (every true-time x-axis),
the q-profile (rational-surface lines), and the edge `Btor` profile
(`delta_b_over_b`). Each is one call per step; each is reported as it runs
and skipped rather than fatal if it fails.

```bash
python ~/ashen/bin/plot --list
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag poincare --diag connection_length --diag four --diag profiles --diag theta_hist
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
- `--four-linear` draws four's mode-amplitude time series on a linear scale
  instead of the default log (and keeps the radial figures linear too).
  The radial eigenfunction figures are linear by default; `--four-radial-log`
  (or the case field `four_radial_log = true`) puts them on a log axis.
- `--four-radial-quantity {abs,real,phase}` -- `four`: draw the radial
  figures as `|amp|`, the phase-aligned signed part, or the phase; overrides
  the case's `four_radial_quantity`.
- `--theta_target_psi`, `--theta_bins`, `--theta_psi_n_range MIN MAX` override
  `theta_hist`'s case config; `--n-cols` sets its grid width (see below).
- `--theta_wetted_threshold FLOAT` overrides `wetted_fraction`'s bin-count
  threshold (comparison-only, see below).
- `--compare NAME` (repeatable) draws a `[comparisons.*]` figure instead of
  per-case figures; `--list-comparisons` shows what's defined (see below).
- `--dpi N` overrides the figure resolution.
- `scan_map` (comparison-only, see below): `--x-quantity`/`--y-quantity`/
  `--c-quantity NAME` pick the per-run scalars; `--map-encoding
  {color,size}` picks how the third is shown; `--map-log-x`/`--map-log-y`/
  `--map-log-c {auto,on,off}` override each axis/encoding's scale;
  `--edge-q-psi-n PSI_N` and `--equilibrium-step STEP` configure the
  `edge_q`/`q95`/`li` family; `--annotate-points` labels each point.
  `--list-quantities` lists the per-run scalars these accept and exits --
  works with no `cases.toml` present, for use while writing one.

### Steps without knowing the run's length

A `{start, stop, step}` table may leave out `start`, `stop` or both. The
run's own restarts (`jorek<step>.h5` in the case's folder) fill them in:

```toml
steps = { step = 400 }                 # every 400 steps, first restart to last
steps = {}                             # every restart
steps = { start = 1000, step = 400 }   # from step 1000 to the last restart
```

- `start` defaults to the run's first restart, `stop` to its last, which is
  included. A `stop` you give stays exclusive, as it is with both ends given.
- The steps are picked from the restarts that exist: those at `start`,
  `start + step`, `start + 2 step`, and so on. With restarts every 200
  steps, `{ step = 300 }` gives every 600th.
- They are counted from the first restart, not from 0. A run whose first
  restart is 200 gives 200, 600, 1000, ... for `{ step = 400 }`.
- The folder is read each time `cases.toml` is loaded, so a run still going
  gets further each time. In `[defaults]`, each case gets its own run's range.
- With both `start` and `stop` given, nothing changes: the range is taken as
  written, whether or not those restarts exist yet.

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

A `steps` list can also mix explicit values with one or more `{start, stop,
step}` tables as entries -- e.g. a dense early stretch plus a coarser range
after it:

```toml
steps = [200, 400, 600, 800, { start = 1000, stop = 5800, step = 200 }]
```

Entries are unioned and returned sorted, so a value covered by both an
explicit entry and a range (like a range's own `start`) collapses to one
rather than being processed twice. `cases.example.toml` has a fuller example.

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

Setting `poincare_highlight = true` (plus `modes`, the same `[m, n]` pairs
used by `four`) colours the field lines nearest each mode's `q = m/n`
resonant surface, dimming everything else to grey. Colour is auto-assigned
per mode (sorted `(n, m)` order into `plotting.colors.DISCRETE_PALETTE`) --
the same mode always gets the same colour on every figure that draws it.
Override individual modes with `mode_colors`, a table keyed `"m,n"` matching
a `modes` entry; any mode left out keeps its auto-assigned colour. The
same colours are used for `plot --diag four`'s mode-amplitude lines:

```toml
modes       = [[3, 2], [2, 1]]
mode_colors = { "3,2" = "red", "2,1" = "blue" }
```

It needs the q-profile cache -- `analyse --diag poincare` gathers it
automatically once this is set, the same way it already force-includes
`zerod`. Because Poincare only traces the discrete `psi_n_in` grid requested,
a computed rational surface is snapped to the *nearest actually-traced* line
rather than requiring an exact match; which physical line that is can shift
step to step as the q-profile evolves -- that tracks the real resonance
moving, not a bug.

**Marking rational surfaces on radial profiles.** `mark_rational = true`
(case field, or `--mark_rational` for one invocation) draws the same
`modes`' `q = m/n` crossings on `plot --diag profiles` figures, with a
legend labelling each mode (`mode_colors` overrides apply here too) --
needs `coords_var = "Psi_N"`. Auto-gathers the q-profile cache for any
requested step that's missing one, in parallel under `--n-workers`.

The surfaces are resolved **per step**, because q evolves through a run and
a surface pinned to one step misrepresents every other. Since the static
figure overlays all steps at once, each surface is drawn as a shaded band
spanning where it travelled, with a dashed line at its position at the
**last** step -- so the figure shows both where the resonance ended up and
how far it moved to get there. A surface that barely moves gets no visible
band, just its line; one that merges away before the last step keeps its
band but has no line to sit on. A step whose q-profile can't be gathered is
named and left out rather than filled in from a neighbour.

Surfaces are followed between steps by position, not by their order within
a step, so a reversed-shear pair that merges partway through doesn't
re-label the survivor and smear its band across the domain.

**Puncture size.** `poincare_point_size` (case field, plot-time only, default
`0.1`) sets each puncture's marker area -- matplotlib's scatter `s`, in
points². The default is tuned for a dense scan; a short run, or a plot zoomed
into a small region, usually wants it larger. `plot --point-size N` overrides
it for one invocation without editing `cases.toml`.

Covered this pass: Poincare puncture plots, the LC/LCTT connection-length
maps, jorek2_four mode-amplitude time series, radial profiles, and the
field-line theta-crossing histogram -- see below. Everything else the legacy
`data_jorek.py` plotted (macroscopic-variable traces, field-line diffusion,
the derived q-profile/`dJ/dr`, stochastic factor) is not ported -- see
`KNOWN_ISSUES.md` #8. Colour is by each line's `psi_n` through a colormap by
default (`ashen.plotting.colors`); a discrete palette is also available.

### Field-line theta-crossing histogram

`--diag theta_hist` answers "where poloidally do field lines leave?" -- for
each traced line it finds the first puncture past `theta_target_psi` (a
user-facing plasma-fraction `psi_n`, default `1.05`) and histograms the
poloidal angle `theta` there over `theta_bins` bins (default `500`) spanning
`(-pi, pi]`. A line that never crosses (confined) contributes nothing. One
figure per case, one panel per step, written to `theta_hist.png`; the CLI
also prints how many lines crossed out of how many were considered per step,
so a `target_psi` set too high to produce anything shows up immediately
rather than as a flat, silent histogram.

```toml
[cases."qa2.1_g2.3/eta1e-3_RE"]
theta_target_psi  = 1.05
theta_bins        = 1000
theta_psi_n_range = [0.2, 0.9]   # optional: only count lines starting in this psi_n_in range
```

`--theta_target_psi`, `--theta_bins`, `--theta_psi_n_range MIN MAX`, and
`--n-cols` override the case config per invocation. `theta_psi_n_range`
filters by each line's **starting** flux surface (its `psi_n_in`), unlike
`lc_psi_n_in`'s bounds filter above which selects *which already-gathered
surfaces to plot* -- here every traced surface within range is pooled into
one histogram, not drawn as separate panels.

Ported from a notebook (`Columbia/NL_kinks/prod_plots_draft0.ipynb`,
`plot_theta_histogram_matrix`) -- see the docstring of
`ashen.diagnostics.theta_histogram` for how `real_psi_edge` is applied
(exactly once, scaling the threshold, never dividing the traced data).

### Comparing across runs

Most figures are about one run's own evolution. `[comparisons.*]` in
`cases.toml` groups **already-defined cases** into a single cross-run figure
instead -- e.g. a resistivity scan's theta-crossing histograms, one panel per
run rather than per step:

```toml
[comparisons.eta_scan]
note          = "resistivity scan, theta dispersion at the q=1 crossing"
cases         = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE", "qa2.1_g2.3/eta1e-5_RE"]
x_tick_labels = ["$\\eta = 10^{-3}$", "$\\eta = 10^{-4}$", "$\\eta = 10^{-5}$"]
n_cols        = 5
```

```bash
python ~/ashen/bin/plot --list-comparisons
python ~/ashen/bin/plot --compare eta_scan --diag theta_hist --theta_target_psi 1.05
```

Each member's own steps still supply that panel's pooled time window --
`[cases.NAME.theta_hist] steps = [...]` picks which steps one run's panel
draws from, the same per-diag override mechanism as everywhere else in
`cases.toml`. Figures land in `./figures/`, named
`<comparison-name>_<diag>.png`, not in any one member's own run folder.
Asking for a diag without a comparison renderer (e.g. `--compare eta_scan
--diag profiles`) is reported and skipped, not silently ignored.

#### Wetted fraction vs. a scan parameter

`--diag wetted_fraction` also has a single-run figure under plain `--case`
mode -- that case's own wetted-fraction evolution across its own steps, one
point per step rather than one pooled point per case. The `--compare`
renderer described here is the cross-run one: for each member case it pools
the same theta-crossing histogram `theta_hist` would,
then reduces it to one scalar: the fraction of bins whose count exceeds a
threshold, so "wetted" means "above what uniform spreading over theta would
give". That one number per case is plotted against `x_values`, an explicit
numeric value per member -- **not** inferred from the run folder's own name
(e.g. parsing `"eta1e-3"` out of `"eta1e-3_RE"`): `CLAUDE.md` flags
CASTOR3D's directory-name parsing as exactly the kind of hazard this project
exists to not repeat, so a folder rename must never silently change what
gets plotted.

**Set the analysis parameters on the comparison itself**, not per case: a
scan is only a fair comparison if every point was computed with the same
`theta_target_psi`/`theta_bins`/`theta_psi_n_range`/threshold, so
`[comparisons.*]` accepts all four and applies them to every member
uniformly, instead of each `[cases.*]` entry needing a matching copy (or
every case in the file sharing one `[defaults]`, which would also affect
anything else those cases are used for). The threshold itself is
`theta_wetted_threshold`; unset, it falls back to `1/theta_bins` -- the value
a perfectly uniform distribution would put in every bin.

```toml
[comparisons.eta_scan]
cases                  = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE", "qa2.1_g2.3/eta1e-5_RE"]
x_values               = [1e-3, 1e-4, 1e-5]
x_label                = "$\\eta$ [$\\Omega \\cdot$ m]"
theta_target_psi       = 1.05
theta_bins             = 500
theta_wetted_threshold = 0.002
```

```bash
python ~/ashen/bin/plot --compare eta_scan --diag wetted_fraction
```

Precedence, most specific wins: a CLI flag (`--theta_target_psi`,
`--theta_bins`, `--theta_psi_n_range`, `--theta_wetted_threshold`) overrides this
comparison's own setting, which overrides the member case's own setting,
which falls back to the diagnostic's built-in default. A member case can
still set these individually for when it's plotted **outside** this
comparison (e.g. its own `theta_hist` figure via `--case`) -- the comparison
tier only wins while `--compare eta_scan` is the one being drawn:

```bash
python ~/ashen/bin/plot --compare eta_scan --diag wetted_fraction --theta_wetted_threshold 0.01
```

`theta_hist` under `--compare` respects the same comparison-level
`theta_target_psi`/`theta_bins`/`theta_psi_n_range` (just not
`theta_wetted_threshold`, which only `wetted_fraction` uses) -- so one
`[comparisons.eta_scan]` table keeps both figures computed identically.

If a member case has its *own* non-default setting for one of these fields
and the comparison also sets it, the comparison wins silently for the
resolved value -- but the CLI prints a warning naming which case and field
got shadowed, so a leftover per-case override doesn't quietly do nothing:

```
qa2.1_g2.3/eta1e-4_RE: comparison 'eta_scan' sets theta_target_psi=1.0, overriding this case's own theta_target_psi=1.02
```

No warning when a case simply never set the field (nothing to shadow), and
none when a CLI flag is given for that field (a CLI flag legitimately
outranks both tiers, so its override isn't surprising).

Written to `figures/<comparison-name>_wetted_fraction.png`; the CLI also
prints each case's fraction. `x_values` is required for this diag --
`theta_hist` on the same comparison works fine without it, since its panels
are keyed by case label, not a numeric axis. Ported from the core of the
notebook's `eta_plot` (`Columbia/NL_kinks/prod_plots_draft0.ipynb`, cell 0),
generalised over what the plotted scalar is -- `ashen.plotting.
wetted_fraction` takes any `(x, y)` pair, not only wetted fraction, for
whatever the next "scalar vs. scan parameter" plot turns out to need.

#### Overlaying several related scans: `datasets`

A comparison names its members one of two ways, never both: flat `cases`
(above), or nested `[comparisons.NAME.datasets.DATASET]` tables -- for more
than one *related* scan sharing the same x-axis, e.g. a resistivity scan
repeated under two different profile assumptions. `wetted_fraction`, `four`
and `scan_map` (below) all draw `datasets`; `theta_hist` needs flat `cases`
and reports/skips a `datasets`-only comparison rather than silently drawing
nothing.

```toml
[comparisons.wetted_vs_eta]
note     = "wetted fraction vs. eta, normal profile vs. rho19"
x_values = [1e-3, 1e-4, 1e-5]
x_label  = "$\\eta$ [$\\Omega \\cdot$ m]"

[comparisons.wetted_vs_eta.datasets.normal]
cases = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE", "qa2.1_g2.3/eta1e-5_RE"]

[comparisons.wetted_vs_eta.datasets.rho19]
cases = ["qa2.1_g2.3_rho19/eta1e-3_RE", "qa2.1_g2.3_rho19/eta1e-4_RE", "qa2.1_g2.3_rho19/eta1e-5_RE"]
color = "tab:red"
```

```bash
python ~/ashen/bin/plot --compare wetted_vs_eta --diag wetted_fraction
python ~/ashen/bin/plot --compare wetted_vs_eta --diag wetted_fraction --dataset rho19
```

Each dataset is drawn as its own coloured, legend-labelled series on one axes
(`ashen.plotting.wetted_fraction.plot_wetted_fraction_datasets`). A dataset's
own `x_values`/`x_tick_labels` fall back to the comparison's -- set them on a
dataset only when that one scan's points genuinely differ. `color` is
optional; omitted, each dataset gets one from
`ashen.plotting.colors.DISCRETE_PALETTE` by position. The legend text itself
is `dataset_label` if set, else the dataset's TOML key (`rho19` above).
`--dataset NAME` (repeatable) restricts a draw to only the named dataset(s).
`theta_target_psi`/`theta_bins`/`theta_psi_n_range`/`theta_wetted_threshold`
apply uniformly across every dataset's cases, same as flat-mode comparisons.

#### Delta-B vs. a scan parameter

`--compare NAME --diag four` draws the same kind of "scalar vs. `x_values`"
figure as `wetted_fraction`, but for delta-B: one delta-B (or delta-B/B)
number per case, converted from that case's `Psi` jorek2_four cache exactly
as the single-run `four_deconfinement_step`/`four_max_delta_b` caption does
(see [Fourier decomposition](#fourier-decomposition-jorek2_four) below), then
plotted against `x_values`. `--delta-b-quantity` picks which scalar:

- `max` (default) -- the domain-wide peak over every mode and every
  requested step, same value `four_max_delta_b`'s caption line shows.
- `mode` -- the peak of one `(m, n)` mode's own series, needs
  `--delta-b-mode M,N` (same `[m, n]` convention as `cases.toml`'s `modes`).
- `deconfinement` -- the domain-wide value at the case's own
  `four_deconfinement_step`; a case that hasn't set one is skipped (reported,
  not silently zero).

```bash
python ~/ashen/bin/plot --compare eta_scan --diag four
python ~/ashen/bin/plot --compare eta_scan --diag four --delta-b-quantity mode --delta-b-mode 3,2
python ~/ashen/bin/plot --compare eta_scan --diag four --delta-b-quantity deconfinement
python ~/ashen/bin/plot --compare eta_scan --diag four --delta-b-over-b
```

`--delta-b-over-b` plots `delta_b_over_b` instead of `delta_b` -- it needs
the same step-0 Btor profile as the single-run figure (auto-gathered on
demand, same as there). Each member case's own `steps`/`modes` still select
what's fetched, the same `steps_for("four")`/`--step` override as everywhere
else; `x_values` is required, same as `wetted_fraction`. Written to
`figures/<comparison-name>_<delta_b|delta_b_over_b>_<quantity>.png`. Also
draws `datasets`-style comparisons (one legend-labelled series per dataset,
`--dataset NAME` to restrict), the same split as `wetted_fraction` above.

#### 2D scan map: one point per run

`--compare NAME --diag scan_map` is the most general cross-run figure: one
point per member run, **x and y each a named per-run scalar** (not
`x_values` -- see below), and optionally a third scalar encoded as point
colour or ring size. `ashen.quantities` is the registry of what a "named
scalar" can be -- `plot --list-quantities` is the authority on the current
list (do not duplicate it here; it will only drift):

```
$ python ~/ashen/bin/plot --list-quantities
eta (log) -- resistivity, read from the run's own namelist
edge_q (linear) -- qprofile interpolated at edge_q_psi_n (default 1.0)
q95 (linear) -- zeroD column, at the equilibrium step
li (linear) -- zeroD li3, at the equilibrium step
wetted_fraction (linear) -- fraction of theta_hist bins above threshold, pooled over steps
delta_b_max (log) -- domain-wide peak delta-B over every mode and step
delta_b_over_b_max (log) -- domain-wide peak delta-B/B over every mode and step
...
zerod:<COLUMN> (linear) -- any zeroD column, at the equilibrium step
```

Each name is **one unambiguous scalar** -- the over-time reduction is part
of the name (`delta_b_over_b_max` vs. `delta_b_over_b_at_deconfinement`),
not a separate knob, so a figure axis labelled "delta_b/B" cannot silently
mean "max" for one run and "final value" for another. `edge_q` and `q95`
are two different definitions on purpose: `edge_q` interpolates the cached
q-profile at a configurable `psi_n` (default 1.0, the separatrix), while
`q95` is JOREK's own zeroD column computed its own way -- registering both
separately keeps a figure honest about which one it plotted.

```toml
[comparisons.eta_q_map]
note       = "max delta-B/B across the (eta, q95) plane"
cases      = ["qa2.1_g2.3/eta1e-3_RE", "qa2.1_g2.3/eta1e-4_RE", "qa2.1_g2.3/eta1e-5_RE"]
x_quantity = "eta"
y_quantity = "q95"
c_quantity = "delta_b_over_b_max"
```

```bash
python ~/ashen/bin/plot --compare eta_q_map --diag scan_map
python ~/ashen/bin/plot --compare eta_q_map --diag scan_map --map-encoding size
```

**No `x_values` needed.** `eta` is read from the run's own namelist, `q95`
and `delta_b_over_b_max` from its own caches -- there is no hand-maintained
parallel array to keep in sync with what the run actually solved, and
nothing parses the run folder's name (the CASTOR3D hazard `CLAUDE.md` warns
about). A comparison may still carry `x_values` for its `wetted_fraction`/
`four` figures alongside `x_quantity`/`y_quantity` for its `scan_map` --
the two coexist without conflict.

**The two encodings** (`map_encoding`, default `"color"`):

| encoding | third quantity shown as | datasets told apart by |
|---|---|---|
| `color` | point colour + colourbar | marker shape (`plotting.MARKER_CYCLE`) |
| `size`  | ring radius + a size legend | colour (`DISCRETE_PALETTE`) |

Whichever encoding is *not* active is free for datasets to use instead --
under `size` encoding a dataset's `color` distinguishes it; under `color`
encoding its `marker` does. Both fields can be set on the same
`[comparisons.X.datasets.Y]` table regardless of which figure is drawn from
it; `plot` notes when a figure ignores the one it doesn't need. `c_quantity`
is optional -- omitted, the figure is a plain 2D scatter of runs with no
third quantity at all.

**`edge_q`'s clamping.** `jorek2_postproc` writes the q-profile over the
case's `rad_range`, whose default outer bound is `0.999` -- so the default
`edge_q_psi_n = 1.0` lands just off the end of the grid on essentially every
run. Rather than silently extrapolating, `edge_q` **clamps** to the nearest
grid point within a small tolerance and reports that it did so; a target
further off the grid than that returns no value at all rather than a number
the cached data doesn't actually contain. See `KNOWN_ISSUES.md` for the
open physics question this raises about what `edge_q`'s default should mean.

Axis/colourbar labels default to each chosen quantity's own label
(`ashen.quantities.Quantity.label`); a comparison's `x_label`/`y_label`/
`c_label` override that default the same way `x_label` already does for
`wetted_fraction`. `equilibrium_step` (comparison-level, default: each
case's own first plotted step) is what `edge_q`/`q95`/`li` are read at --
deliberately not per-case, since a scan mixing "q95 at step 200 for this run
and step 3000 for that one" isn't a scan. Written to
`figures/<comparison-name>_scan_map_<y>_vs_<x>[_<c>].png`.

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

**`currdens`'s radial gradient.** Whenever `currdens` is among the drawn
`vars`, `|d(currdens)/d(coords_var)|` (absolute value, `np.gradient` over
the already-cached radial grid -- no extra gather) is drawn automatically
alongside it, saved to `<coords_var>_currdens_grad_profile.png`. Absolute
value because a sharpening *magnitude* is the tearing-mode-onset signal
(same convention as the legacy `gather_profiles.py::plot_postproc_profs`
Jgrad panel) -- the sign flips depending on which side of the peak you're
on and isn't the interesting part. Not a separate `vars` entry -- always on
when `currdens` is plotted, off otherwise; a step with fewer than two
radial points is dropped from that mode's derivative line (can't
differentiate a point).

`profile_cmap` (case field, plot-time only, default `"turbo"`) sets the
colourmap for the time/step colourbar -- any matplotlib colormap name.
Defaults to a rainbow-style map rather than the package's usual `viridis`,
since a profile figure is read by eye for "which step is this line", and a
rainbow's wider hue range makes that easier to track than viridis's narrower
one. `--profile-cmap NAME` overrides it for one invocation without editing
`cases.toml`.

`profile_ylim` (case field, plot-time only) pins a variable's y-axis to
`[min, max]` instead of matplotlib's auto-scaling, keyed by variable name --
same convention as `four_ylim`. `currdens`'s auto-drawn gradient figure is a
separate quantity/figure and needs its own entry, keyed `"<var>_grad"`:

```toml
profile_ylim = { currdens = [0, 5e5], currdens_grad = [0, 1e7] }
```

Applies to both the static PNG and (if `animate`) every GIF frame, in place
of the animation's usual fixed-to-the-data-range y-limits.

**Animating the time evolution.** `animate = true` (case field, or
`--animate` for one invocation) additionally writes
`<coords_var>_<var>_profile.gif` alongside the PNG -- one frame per restart
step, each panel showing only that step's curve (not the whole family at
once), coloured the same way as the static figure, with fixed axis limits
so panels don't rescale frame to frame. `mark_rational`'s surfaces, if on,
move with the frame -- each frame draws its own step's crossings, so a
resonance visibly tracks the profile it belongs to. The legend is built
once from every step's modes, so it doesn't flicker as surfaces come and
go. Skipped, with a
printed note, for a figure with fewer than two steps.

Every frame's title states the restart step *and* the true time (from the
zeroD cache, if available) as text -- regardless of which of the two the
colourbar itself is keyed on. So even when the zeroD cache is incomplete
and the colourbar falls back to colouring by step index, a frame's real
time is still readable; and even when the colourbar is by time, the step
number stays visible too.

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

`four_vars` and `modes` (case fields, plot-time only) restrict which
variables/modes get drawn; empty (default) draws everything found in the
cache. `modes` is shared with `poincare_highlight` and `mark_rational`
(below) -- one list, one convention. Entries are `[m, n]` pairs (poloidal,
toroidal) -- `[3, 2]` is `m=3, n=2`, matching how a mode is normally written
(`m/n`):

```toml
four_vars = ["Psi", "T"]
modes     = [[2, 1], [3, 2], [1, 1]]   # [m, n] pairs
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

**Radial eigenfunctions / `four_quantities = ["radial"]`.** `max` and
`rational_surface` are both scalar-per-step time series -- they reduce each
mode's radial profile to one number. `radial` draws the profile itself:
`|amplitude|` against `psi_n`, one figure per variable
(`four_dir/<var>_eigenfunction_psin.png`), one panel per `(n, m)`, one
colour-graded line per step.

```toml
four_quantities = ["radial"]            # eigenfunctions only
four_quantities = ["max", "radial"]     # both: time series and eigenfunctions
```

No extra gathering is needed -- `jorek2_four` already writes the whole radial
table and `analyse --diag four` already caches it; the scalar quantities were
simply discarding it. A case that has been analysed can be plotted this way
with no further `analyse` run.

Unlike the time-series quantities, this figure's x-axis is `psi_n`, so it
needs **no zeroD cache**: there is no step-to-time conversion to make, and a
`radial`-only case neither requires nor warns about one. Steps colour the
lines instead, on the shared colourbar.

When `modes` is set, each mode's `q = m/n` surface is resolved **per step**,
since q evolves through a run and every step's curve is on the figure. The
q-profile cache is gathered on demand for any step missing one (in parallel
under `--n-workers`). Each surface is shaded over the range it swept across
the plotted steps, with a dashed vertical line where it sat at the last step
-- the same treatment as `plot --diag profiles --mark_rational`. One shared
set of markers is drawn on every panel (each labelled and coloured by mode),
so a panel shows its neighbours' surfaces too. A step whose q-profile can't
be gathered is reported and left out of the band; if none can, the curves
are drawn without markers.

The y-axis is shared across panels and **linear by default**, which shows
each eigenfunction's shape. For modes spanning orders of magnitude, set
`four_radial_log = true` on the case, or pass `--four-radial-log` for one
invocation; the time-series figures keep their own log default either way.
On a log axis a mode whose profile decays into a very small tail can stretch
the shared range enough to flatten everything else -- `four_ylim` (keyed by
variable, as for the time-series figures) pins it. `--four-linear` keeps
both figure families linear, overriding a radial log request.

```toml
four_quantities = ["max", "radial"]
four_radial_log = true               # radial on a log axis (default: linear)
```

**Signed eigenfunctions / `four_radial_quantity`.** `jorek2_four` writes the
full complex coefficient `c(psi_n)` and the cache keeps its real and imaginary
parts, so the radial figures can draw more than `|c|` with no regathering:

```toml
four_radial_quantity = "abs"     # |amplitude| (default)
four_radial_quantity = "real"    # signed, phase-aligned
four_radial_quantity = "phase"   # radians, relative to the peak
```

`--four-radial-quantity {abs,real,phase}` overrides it for one invocation.
The raw `Re(c)` is **not** drawn: its phase is measured from `theta* = 0`,
`phi = 0`, which drifts as the mode rotates, so the sign and shape would
change between steps for no physical reason. Instead each step's profile is
rotated by the phase at its `|c|` peak -- `Re(c e^{-i phi0})` is positive at
the peak and comparable across steps, while a real sign change (e.g. across
a tearing mode's rational surface) survives. `phase` is noise wherever `|c|`
is near zero. Both are always drawn linear (a log request is reported and
ignored), `phase` on fixed `[-pi, pi]` bounds instead of `four_ylim`, and
each goes to its own `<var>_eigenfunction_real_psin.png` /
`<var>_eigenfunction_phase_psin.png` so the `|c|` figure is never
overwritten.

`delta_b`/`delta_b_over_b` have no radial form here: `b_r ~ (m/R_axis^2)
|Psi_mn|` scales by a **constant**, so a radial `delta_b` curve would be the
`Psi` eigenfunction with a relabelled y-axis. Requesting one under `radial`
falls back to that `Psi` eigenfunction rather than drawing a rescaling that
carries no extra information.

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
delta_b_over_b(m,n) = (m / R_axis^2) * |Psi_mn| / B_ref
```

`R_axis` is read from the run's `log` (`ashen.logfile.r_axis`). `B_ref` is
`Btor` interpolated to the plasma edge (`psi_n = 1`) from the cached
step-0 (initial-equilibrium) midplane profile
(`ashen.diagnostics.profiles.edge_toroidal_field`) -- a fixed reference
field, not the perturbed run's own evolving field. This is an approximation:
the exact relation uses the true local minor radius and `|grad Psi|`, not
the (constant) major radius at the magnetic axis, but the four cache only
carries `|Psi_mn|` on a `psi_n` grid, not real-space geometry, so `R_axis`
stands in for it everywhere. `m = 0` modes are dropped (no helical
radial-field content in this shorthand) rather than drawn as a flat zero line.

`B_ref` needs `Btor` at step 0 (`"midplane outer"`, not bare `"midplane"`,
which is double-valued in `Psi_N`) -- `plot` gathers this one profile itself
on demand if it isn't already cached
(`ashen.diagnostics.profiles.ensure_edge_toroidal_field`), the one deliberate
exception to `bin/plot` otherwise never running a `jorek2_*` tool (see the
module docstring): a single-valued lookup like this is cheap and one-off,
unlike a full profiles gather, so doing it inline doesn't blur the
`analyse`/`plot` slow-batch/fast-iterative split the rest of this file
describes. A run whose step-0 restart is missing, or whose `jorek2_postproc`
fails, prints `skipping delta_b_over_b: ...` and moves on -- same as any
other missing input this section describes -- rather than aborting the whole
`--diag four` plot.

Only computed when explicitly requested -- an empty/unset `four_vars` never
picks it up, since it isn't "everything found in the cache". Works with
either `four_quantities` selection (whole-domain max or rational-surface
value), since it's a post-conversion of whichever `Psi` series was computed:

```toml
four_vars = ["delta_b_over_b"]              # only the derived quantity
four_vars = ["Psi", "delta_b_over_b"]       # raw flux amplitude alongside it
```

**`delta_b`** is the same quantity un-normalised -- `(m / R_axis^2) *
|Psi_mn|`, in Tesla, with no division by `B_ref`. It only needs `R_axis`
from the log, not the `Btor` profile, so it still works on a run that hasn't
gathered step-0 profiles; `delta_b_over_b` does not. The two can be
requested together (`four_vars = ["delta_b", "delta_b_over_b"]`) and are
computed independently, so a missing `Btor` profile skips only
`delta_b_over_b`.

A run whose log is missing `R_axis` prints one `skipping delta_b,
delta_b_over_b: ...` message and drops both (neither can be computed without
it); missing only the step-0 `Btor` profile prints `skipping
delta_b_over_b: ...` and `delta_b` is still drawn. Either way the rest of
the requested `--diag four` plot is unaffected, not an error.

`four_max_delta_b = true` (case field, plot-time only, default off) adds a
small boxed caption in the lower-right corner of either figure giving the
peak value actually drawn -- `max δB = 1.2 T` or `max δB/B = 0.03` -- the
largest finite value across every mode and step in that figure, so a reader
doesn't have to eyeball the plot to answer "how big does this get." No other
`four_vars` variable gets a caption, with or without the flag.

```toml
four_vars        = ["delta_b_over_b"]
four_max_delta_b = true
```

**Y-axis bounds.** `four_ylim` (case field, plot-time only) pins a
variable's y-axis to a fixed `[min, max]` range instead of matplotlib
auto-scaling each figure independently -- useful for comparing the same
variable's amplitude across several runs/cases at a glance. Keyed on the
same variable name used in `four_vars` / the output filename (including
`delta_b`/`delta_b_over_b`); a variable not listed keeps auto-scaling:

```toml
four_ylim = { Psi = [1e-6, 1e-1], T = [1e-4, 1e1] }
```

**Deconfinement step.** `four_deconfinement_step` (case field, plot-time
only, a time step -- e.g. from a separate diagnostic, not computed here)
draws a vertical dashed line marking it on every four-mode figure. On the
step-axis (`*_modes_step.png`) figures it's drawn directly at that step; on
the time-axis (`*_modes_time.png`) figures it's drawn at that step's real
time, read from the zeroD cache -- gathered on demand if missing, the same
precedent as `delta_b_over_b`'s `Btor` profile above. The step need not be
one of the case's own requested `steps`. If its zeroD cache can't be
gathered (e.g. the restart doesn't exist), the step-axis line still draws
but the time-axis one is skipped with a printed note. Unset (default) draws
nothing:

```toml
four_deconfinement_step = 1200
```

Whenever `four_deconfinement_step` is set, the `delta_b`/`delta_b_over_b`
figures' boxed caption also gets a `δB at deconfinement = ...% of max` /
`δB/B at deconfinement = ...% of max` line by default -- the domain-wide
value at that step (max across modes), expressed as a percentage of that
same figure's own peak value (also the domain-wide max across modes, but
over every requested step, not just the deconfinement one). No
interpolation, same rule as `connection_length`'s `psi_n` matching: a step
not among the case's own requested `steps` produces no caption line,
silently, rather than a wrong number -- if the line doesn't appear, check
`four_deconfinement_step` matches one of `steps` (or this diag's own
`[cases.NAME.four] steps` override) exactly. Independent of
`four_max_delta_b` -- with both set, the caption gets two lines,
`max δB/B = ...` then `δB/B at deconfinement = ...% of max`. Set
`four_deconfinement_caption = false` to keep the vline but drop just this
caption line:

```toml
four_deconfinement_step    = 1200
four_deconfinement_caption = false   # vline only, no caption
```

Connection lengths use `R0` extracted from the run's log
(`ashen.logfile.r_axis`) rather than the legacy hardcoded `R0 = 1.36` -- see
`KNOWN_ISSUES.md` #6 and #7 for what changed and what's still an open question.

## Tracing particles

`bin/ptrace` runs a particle-tracing executable -- normally one of JOREK's
own `particles/examples` programs -- unmodified, against the restarts of a
run that already exists. ashen wraps it; it does not change what it
computes. A case traces when it sets `ptrace_exe`, alongside its other
keys in `cases.toml` (and `[defaults]` can seed any
`ptrace_*` key, like the rest). `ptrace_exe` is a path **relative to the run
folder**, so a binary in the run's `exe/` is `./exe/<name>`:

```toml
[cases."qa2.1_g2.3/eta1e-3_RE"]
steps            = { start = 200, stop = 5800, step = 200 }
ptrace_exe        = "./exe/re_gc_current_density_initialisation"
ptrace_start_step = 3000
```

```bash
python ~/ashen/bin/ptrace --list                                      # tracing cases, and the programs
python ~/ashen/bin/ptrace --case "qa2.1_g2.3/eta1e-3_RE" --dry-run    # what it would link and run
python ~/ashen/bin/ptrace --case "qa2.1_g2.3/eta1e-3_RE" --run_i      # run it here
python ~/ashen/bin/ptrace --case "qa2.1_g2.3/eta1e-3_RE" --run        # queue it (jobscripts/2h)
python ~/ashen/bin/ptrace --case "qa2.1_g2.3/eta1e-3_RE" --run -job 23h
python ~/ashen/bin/ptrace                                             # where every trace stands
```

**Launching**, as `run_jorek` launches a main run:

- `--run_i` runs the program here: `site.toml`'s `interactive_prelude`, then
  `mpirun` with the case's `ptrace_n_mpi` ranks and `ptrace_omp_threads`
  threads.
- `--run` queues it with `sbatch` and a jobscript from `site.toml`'s
  `jobscripts` folder: `2h` unless `-job` names another (`-job 23h`, or any
  file there). The job runs in the ptrace folder. The jobscript's own
  `#SBATCH` lines and `srun` set the ranks and threads, so `ptrace_n_mpi` and
  `ptrace_omp_threads` don't apply. The jobscript is part of the trace's
  cache key: re_gc's particle count depends on the rank count.
- With neither, `ptrace` runs nothing. It reports each trace's state:
  not traced, `[cached]`, queued or running, out of date, or failed.

A queued job is concluded the next time `ptrace` runs for its case (with
or without a flag). At that point its restart links are removed and its
result is recorded. The jobscripts pipe the program through `tee`, which
hides its exit status. So a queued trace counts as done only if SLURM
doesn't report it failed (`sacct`, e.g. `TIMEOUT`) *and* it wrote
`part_restart.h5` or stopped at a lost particle. While its job is queued
or running, a trace's folder is left alone, even with `--force`.

**Any executable, under any name.** ashen never goes by the
executable's filename: a zero exit is success, and what it does beyond
that it reads from what the program writes -- a `ptrace_gc: NOTE:` line in
the log is repeated as a `note:`, ex7's `PARTICLE IS LOST, STOPPING` is
reported as a lost particle rather than a failure, a run that exits 0
without writing `part_restart.h5` gets a note, and the plots use whichever
particle files are in the folder. So a program built per model under its
own name (`ptrace_gc_refluid_fixed_T_rho`) works the same. JOREK's own
programs, for reference:

| program | what it traces (all fixed in its source) | writes |
|---|---|---|
| `re_gc_current_density_initialisation` | 32 relativistic guiding-centre electrons **per MPI rank**, sampled from the current density (20 MeV, pitch near pi), 1e-5 s from `ptrace_start_step`'s own time -- or the particles in `ptrace_particles`, if given | `part_diag.h5`, `part_restart.h5` |
| `ex6_jorek` | one relativistic full-orbit electron, **starting at t = 2.5 ms**, for 1e-5 s | `diag.h5`, `part_restart.h5` |
| `ex7_jorek` | one relativistic guiding-centre electron, **starting at t = 2.5 ms**, for 1e-6 s; stops at the first lost particle | `diag.h5`, `part_restart.h5` |

Anything else -- particle positions, energies, time step, duration -- means
editing the program in JOREK and rebuilding it; ashen only chooses which
restarts it sees and how it is launched. Keys: `ptrace_start_step`
(first restart it sees; default the run's first), `ptrace_end_step` (last
restart it sees; default the run's last), `ptrace_particles` (a JOREK particle file to start from,
*copied* in as `part_restart.h5` because the program overwrites that file
at the end; ex6/ex7 and `ptrace_gc` ignore it), `ptrace_inputs` (files
copied into the ptrace folder under their own names before the run, for a
program that reads some), `ptrace_n_mpi`, `ptrace_omp_threads`
(default: `site.toml`'s `[diagnostics]`). Like `ptrace_exe`, the paths in
`ptrace_particles` and `ptrace_inputs` are **relative to the run folder**, so
a bare `"seeds.dat"` is the file in the run folder.

Each trace runs in `<run>/ptrace/<executable filename>/` (for `ptrace_gc`, a
folder under that per energy and marker count, see below), with its output in
`ptrace.log`. A completed trace is `[cached]` until its settings, restarts,
particle file or executable change; `--force` reruns it. `--tool-output`
echoes the program live. ex7 stopping at a lost particle is reported as
such, not as a failure.

**ex6/ex7 start at a hard-coded 2.5 ms**, whatever `ptrace_start_step` is: they
pick the last linked restart before that time, so choose restarts that span
it -- otherwise the particle starts in fields from the wrong time, and the
program only prints a warning. They also abort when they find no next
restart, so give them at least two.

**How the restarts reach the program.** JOREK's particle field reader looks
at most 20 file numbers ahead for the next restart, opens each at one
width only (`rst_file_ind_fmt(1)`), and -- for ex6/ex7 -- chooses its first
file with `last_file_before_time`, which only understands 5-digit names. So
the ptrace folder links the chosen restarts in as a consecutive sequence
under **both** widths (`jorek00001.h5` and `jorek000001.h5 -> ../../jorek03200.h5`,
...). Each file carries its own time, so nothing is lost by the
renumbering. Between restarts the fields are interpolated linearly in time,
so the restarts traced through must share one grid. The restart links are
removed again once the program exits, so the folder holds only what the
run produced (`--dry-run` marks them).

**Building.** In the JOREK checkout the run was built from, `make <program>`
with the **same `MODEL`** as the run (variable indices differ between
models); the binary lands in the checkout's top folder. Copy it where
`ptrace_exe` points -- e.g. the campaign's shared `exe/`, which a prepared
run folder links as `./exe`. Keep its name to keep ashen's checks for it.

### ashen's own tracer: `ptrace_gc`

JOREK's programs hard-code their particles (and re_gc's current-density
sampling can hang on some equilibria). `fortran/ptrace_gc.f90` is a
configurable ex7: guiding-centre electrons started at the (R, Z, phi),
energy and pitch you list -- or drawn from the current profile -- pushed
with RK4 through static or evolving fields, with nothing hard-coded.

**It traces from `ptrace_start_step` to `ptrace_end_step`** -- from the
first linked restart's time to the last one's -- static or evolving. No
`ptrace_end_step` means every later restart of the run. (`t_span` > 0 in
the settings traces for that long instead; static fields with a single
restart need it.)

**Settings** are keys of the case in `cases.toml`: each one as
`ptrace_<name>`, with a default for most. There is no settings file of your
own to keep (`ptrace_params.nml` is gone, and listing one in `ptrace_inputs`
is an error that says so):

```toml
[cases."qa2.1_g2.3/eta1e-3_RE"]
ptrace_exe        = "./exe/ptrace_gc"
ptrace_start_step = 3000
ptrace_end_step   = 3400
ptrace_initialiser = "current_pdf_simple"
ptrace_n_markers   = 1000
ptrace_E_kin_eV    = 1e7
ptrace_cos_pitch   = 0.9
```

| key | default | |
|---|---|---|
| `ptrace_n_markers` | none: required | how many markers, up to 100000 |
| `ptrace_E_kin_eV` | none: required | kinetic energy [eV] |
| `ptrace_initialiser` | `"markers"` | how the markers are placed, see below |
| `ptrace_R0`, `ptrace_Z0` | none: required for `"markers"` | where each marker starts [m] |
| `ptrace_phi0` | 0 | its toroidal angle [rad] |
| `ptrace_cos_pitch` | none: required | v_par / v, in -1..1 |
| `ptrace_charge` | -1 | [e] |
| `ptrace_mass` | 5.48579909065e-4 | [amu]: an electron |
| `ptrace_dt` | 1e-10 | RK4 step [s] |
| `ptrace_diag_step` | 1e-8 | between `ptrace_diag.h5` writes [s] |
| `ptrace_n_snapshots` | 100 | about how many `part_restart_s<step>_t<time>.h5` snapshots over the trace, however long it is; 0 = only the final `part_restart.h5` |
| `ptrace_snapshot_step` | not set | a fixed time between snapshots [s] instead; when set it is used and `n_snapshots` is not (0 = only the final one) |
| `ptrace_t_span` | 0 | 0: trace from `ptrace_start_step` to `ptrace_end_step`; > 0: for this long [s] instead |
| `ptrace_field_mode` | `"evolving"` | or `"static"`: the start step's field, frozen |
| `ptrace_hold_last_field` | false | with `t_span` past the last restart: keep its field (true) or stop there (false) |
| `ptrace_stop_when_stalled` | false | stop the trace early once the markers have all but stopped leaving, see below |
| `ptrace_stall_rate_fraction` | 0.05 | ...when the loss rate has fallen to this fraction of its peak |
| `ptrace_stall_min_lost` | 0.10 | ...but not before this fraction of the markers has left, |
| `ptrace_stall_min_time` | 0.5 | ...or this fraction of the trace has passed, whichever is first |
| `ptrace_stall_window` | 0 | the time the loss rate is measured over [s]; 0 = a tenth of the trace |
| `ptrace_pdf_n_sub` | 4 | `current_pdf_simple`: cells per grid element side |
| `ptrace_seed` | 1 | `current_pdf_simple`: same seed, same markers |

- **Names** are the tracer's `&ptrace` names, in any case. `[defaults]`
  seeds them like any key: `[defaults] ptrace_dt = 1e-10` is every case's.
- **Checked when `cases.toml` is loaded**, so a trace that cannot run is
  refused before it is staged or queued: a required key missing, a list
  whose length is not `n_markers`, a negative step.
- **Per-marker keys** (`R0`, `Z0`, `phi0`, `E_kin_eV`, `cos_pitch`,
  `charge`): under `"markers"`, one value is every marker's and a list is one
  value each. Under `current_pdf_simple` every marker shares one energy,
  pitch and charge, and `R0`/`Z0`/`phi0` are refused -- it places the
  markers itself.
- **Snapshots scale with the trace.** By default the tracer writes about
  `n_snapshots` of them: one every (traced time) / `n_snapshots`, rounded up
  to 1, 2 or 5 x 10^k s. A 123 us trace gets one every 2 us, 62 in all; a
  1 ms trace one every 10 us. So between 40 and 100, at round times, and
  the animation has the same length whatever the trace's. `ptrace.log` says
  which spacing it chose. Set `ptrace_snapshot_step` for the same physical
  spacing across cases, e.g. to compare frames at matching times.
- **Each trace records what it ran with.** ashen writes *every* setting,
  defaults included, to `ptrace_settings.nml` in the trace folder
  (below). That is the only settings file `ptrace_gc`
  reads, so it is exactly what the trace used, and it stays there with the
  outputs.
- **Traces at another energy or marker count are kept side by side.** A
  trace's folder is `<run>/ptrace/<executable>/E<energy in eV>eV_n<markers>/`,
  e.g. `ptrace/ptrace_gc/E10000000eV_n1000/`, so changing `ptrace_E_kin_eV`
  or `ptrace_n_markers` starts a new folder instead of overwriting, and
  changing back finds the old trace still cached. The energy is always in
  whole eV; markers of several energies give the range
  (`E1000000-10000000eV`). The plots read the folder that matches the case's
  current two keys, and `ptrace` with no flag lists the others. Any other
  setting (`cos_pitch`, the steps, ...) still retraces in the same folder.
- **Printed when a trace runs or is queued** (`ptrace --run_i`, `--run`),
  with the defaults marked and energies in MeV or keV (the file keeps eV);
  `--dry-run` shows the same.
- A changed setting, or a changed default, makes the trace out of date.
- Only `ptrace_gc` (under any filename) reads them. A case that sets none
  gets no file, as for JOREK's own programs.

```
==== qa2.1_g2.3/eta1e-3_RE (./exe/ptrace_gc) ====
  steps 3000..3400 (3 restart(s))
  settings (ptrace_settings.nml in the trace folder):
    field_mode      = evolving   (default)
    dt              = 1e-10   (default)
    snapshot_step   = from n_snapshots   (default)
    n_snapshots     = about 100, at round times   (default)
    initialiser     = current_pdf_simple
    n_markers       = 1000
    E_kin_eV        = 10 MeV
    ...
```

**Diagnostics files are repacked when a trace ends.** JOREK writes
`ptrace_diag.h5` about 50000/n_markers times bigger than its data (see
`util --func compress_traces`). Once the program has exited -- after
`--run_i`, or when `ptrace` concludes a queued job -- ashen repacks it to
the size of its data, losslessly, and says so in a `note:`. The file still
grows to its full size *while* the trace runs, so the quota needs that
room until then.

**Stopping a trace early.** Once the field perturbation has saturated and
the markers have stopped leaving, tracing on costs time and disk for
nothing. Two rules end a trace before its end step, with every output
written as usual and a `note:` from `ptrace` saying why:

- **Every marker is off the grid.** Nothing is left to push. Always on.
- **`ptrace_stop_when_stalled = true`**: the loss rate -- markers leaving
  per second, over the last `ptrace_stall_window` -- has fallen to
  `ptrace_stall_rate_fraction` (5 %) of the highest it has been.

The stall rule is only looked at once `ptrace_stall_min_lost` (10 %) of
the markers has left **or** `ptrace_stall_min_time` (50 %) of the trace
has passed, whichever comes first, and never while no marker has left at
all.

- **What "left" means.** Off the grid -- or, in a run prepared with
  `extend_bnd`, outside the plasma boundary (`original_bnd.dat`, which
  ashen writes into the trace folder as `ptrace_boundary.dat`). That is
  what the plots call escaped. Once out, a marker counts as gone for good.
- **It cannot see the field.** It cannot tell losses that are over from
  losses that have paused. A little early loss, a quiet gap, and the main
  loss after half-time: at half-time it finds the rate stalled and stops,
  missing the main loss. If a case's losses come late, raise
  `ptrace_stall_min_time` past where they start (1 = no time gate), or
  leave the rule off for that case.
- **A slow tail keeps it going.** A loss that carries on at a tenth of its
  peak is above the 5 % mark; the trace runs on until it drops below.
- **Off by default.** The rule changes how much of start..end a trace
  covers, so it is per case, like any key (`[defaults]` to share it).
- **It changes no marker's path.** The rules are looked at only where the
  trace stops anyway, every `diag_step` and snapshot.
- **The plots** label such a trace `stopped early, losses stalled: 62 % of
  the way to ptrace_end_step 3400` rather than unfinished.

A binary built before this change reads `ptrace_params.nml` and
`ptrace_overrides.nml` instead and stops at once, finding neither: rebuild
it from `fortran/ptrace_gc.f90`.

**Initialisers.** `initialiser` in `&ptrace` chooses how the markers are
placed:

| `initialiser` | markers |
|---|---|
| `'markers'` (default) | one at each listed `R0`, `Z0`, `phi0`, each with its own `E_kin_eV`, `cos_pitch`, `charge` |
| `'current_pdf_simple'` | `n_markers` drawn with the toroidal current density of step `ptrace_pdf_step` (default `ptrace_start_step`) as their pdf, all at `E_kin_eV(1)`, `cos_pitch(1)`, `charge(1)` |

`current_pdf_simple` samples the n = 0 (axisymmetric) current profile of
one restart: JOREK's `zj`, which is R·j_phi, so particles per unit R-Z area
follow the current in each volume. Which restart is `ptrace_pdf_step` in
`cases.toml` -- any step of the run, e.g. the current just before a
disruption while tracing from a later step; ashen links it into the ptrace
folder as `jorek_pdf.h5`, and `ptrace.log` names the step it used. Only the
current along the net plasma current counts; `ptrace.log` says what
fraction runs against it and gets no markers. Each grid element is split
into `pdf_n_sub` × `pdf_n_sub` cells (default 4), markers are drawn from that
table -- uniform in the element's (s, t) within a cell, uniform in phi -- so
it always finishes, unlike re_gc's rejection sampling. `seed` (default 1)
makes it repeatable. Up to 100000 markers.

It writes `ptrace_diag.h5` (energy, mu, psi_N, p_phi, lost, R, Z, phi every
`diag_step`), `part_restart_s<step>_t<time>.h5` every `snapshot_step` (the
JOREK step of the closest restart, and the time in seconds, e.g.
`part_restart_s003200_t2.500000E-03.h5`) and
`part_restart.h5` at the end -- so `plot --diag particles` shows the
distribution evolve. A marker that leaves the grid is flagged lost and the
rest carry on. With `hold_last_field = .false.`, a `t_span` > 0 that runs
past the last restart it sees (`ptrace_end_step`) stops at that restart's
time with every output written -- JOREK's reader on its own would abort
with nothing written -- and `bin/ptrace` prints a `note:` saying how much
of `t_span` was traced. Keep `restart_index = 0`: ashen links the restarts from
index 0, and `jorek_restart.h5` to the start restart for
`field_mode = 'static'`.

**It has never been compiled** -- this repository is developed without a
Fortran compiler -- so expect the first build to need small fixes. Build it
like the others: copy `fortran/ptrace_gc.f90` into a JOREK checkout's
`particles/examples/`, `make ptrace_gc` with the run's `MODEL`, and copy the
binary to where `ptrace_exe` points. JOREK itself is never modified by ashen.

**Where its time goes.** Almost all of it is spent in JOREK's RK4
pusher: four field evaluations and element searches per marker per `dt`.
Markers are shared round-robin between MPI ranks, and each rank's markers
are spread dynamically over its OpenMP threads, so lost markers leave no
thread idle. Neither split changes any marker's path. The next biggest
cost is every `diag_step`: `ptrace_diag.h5` gathers every marker to rank 0,
and JOREK's diagnostics writer searches the whole grid for the magnetic
axis and X-point, each time. The two settings that make a trace cheaper
also change its results, so ashen leaves them to you:
- a longer `diag_step` gives coarser exit times and positions;
- a longer `dt` gives less accurate orbits. Check convergence by halving it.

### Plotting particle positions

`plot --diag particles` draws the particle files in a case's ptrace folder
(`part_restart*.h5`, sorted by the time stored in each) on the R-Z plane.
`particles.png` is the last snapshot: where the particles ended up.
`--animate` adds `particles.gif`, with every snapshot as a frame, all on the
same R and Z limits:

```bash
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag particles            # particles.png: the last snapshot
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag particles --animate  # + particles.gif: all of them
```

Each panel or frame shows the particles at that time in red (`ptrace_particle_color`,
any matplotlib colour) over the first
snapshot's in light grey, so drift away from the start reads in any single
panel; lost particles (grid element <= 0) are black crosses where they left
the grid. Each panel's title gives the time and how many of all the
particles have escaped so far, as `XX/XX escaped` -- lost from the grid, or (with
`ptrace_original_boundary`) out of the plasma boundary. Particles at every
toroidal angle are projected onto the one R-Z plane. The PNG has a caption saying what the colours mean; the GIF
doesn't, and plays in about ten seconds however many frames it has (2 to
20 a second). Figures are written into the ptrace folder, next to the files
they draw.

**The view is framed on the plasma boundary**, not on the particles, so
a few particles far away don't shrink the plasma to a dot; anything outside
the view is simply not shown. The limits are the boundary's bounding box,
grown about its centre:

- by 25 % for a run prepared with `extend_bnd`. The boundary is then
  `original_bnd.dat`, and the grid reaches beyond it.
- by 10 % otherwise. The boundary is then `in_bnd`'s, or the namelist's if
  there is no `in_bnd`.

Without any of those files the plot fits everything drawn, and says so.

**How many frames you get is up to the program.**
`re_gc_current_density_initialisation` writes `part_restart<time>.h5`
every `write_step` (1e-4 s) and `part_restart.h5` at the end -- but its
run is only 1e-5 s, so out of the box that is two snapshots, start and
end. More needs a smaller `write_step` in the program itself.

**Poincare overlay.** With `ptrace_poincare = true`, each panel also shows
the Poincare punctures `analyse --diag poincare` cached, drawn a little
faint underneath the particles, from the traced restart closest in time to
the snapshot -- the step in a `ptrace_gc` snapshot's name when that step has
a cache, otherwise the nearest cached step by zeroD time. The step is in
each panel's title. So for evolving fields, gather Poincare at the steps you
traced through: put them in the case's `steps` (or a `[cases.NAME.poincare]`
`steps`) and run `analyse --case X --diag poincare`.
Two keys choose what to draw, and either turns the overlay on by itself:

```toml
ptrace_poincare_psi_n   = [0.5, 0.9]   # these lines only (psi_n_in units, list or {start, stop, n})
ptrace_poincare_n_turns = 200          # at most 200 punctures per line
```

The lines have to be in the cache already: a `ptrace_poincare_psi_n` value
that isn't is reported with the ones that are. `--point-size` (or
`poincare_point_size`) sets the puncture size. The punctures are one
toroidal plane (the case's `phi_start`); the particles are every phi.

**Original boundary.** For a run prepared with `extend_bnd`,
`ptrace_original_boundary = true` draws the plasma boundary from before the
extension (`original_bnd.dat`) as a magenta dashed line and treats it as
where the plasma ends: a particle that leaves it is **no longer tracked**.
From then on it is drawn where it first left, as a magenta cross, counted in
the panel title's `XX/XX escaped` -- even if it later wanders back in or
leaves the grid. When it left is taken from the program's diagnostics file
(`ptrace_diag.h5`, every `diag_step`) when there is one for the same
particles, else from the snapshots themselves, which is only as fine as
their spacing.

**Clipped to the traced steps.** Both particle plots (`particles` and
`particle_exits`) show only what happened between the times of
`ptrace_start_step` and `ptrace_end_step` -- a trace can run past the end
step (`t_span`, `hold_last_field`), and anything after it is left out, with
a note: later snapshots aren't drawn, and a particle that only exits after
the end step doesn't count. The times come from `ptrace_gc`'s restart table
in `ptrace.log`, else from an existing zeroD cache; without either, the
plots aren't clipped and say so.

A case without `ptrace_exe` is skipped silently in a default (no `--diag`)
run, and with a note under an explicit `--diag particles`.

### Where particles leave the plasma

`plot --diag particle_exits` is the particle counterpart of `theta_hist`:
histograms of the poloidal angle theta and the toroidal angle phi at which
each traced particle first goes past a chosen psi_n, side by side, as a
fraction of the particles that exit. Read from the program's diagnostics
file (`ptrace_diag.h5` for `ptrace_gc`), written to `particle_exits.png` in
the ptrace folder:

```toml
ptrace_exit_psi_n = 1.0    # the psi_n that counts as leaving (default 1)
ptrace_exit_bins  = 72     # bins over each of theta and phi
```

**Only the markers that started in a psi_n range.**
`ptrace_initial_psi_n_range = [min, max]` restricts both this plot and
`particle_wetted` to the markers whose psi_n at the first diagnostics time
was inside the range (ends included) -- where they started, whatever they
did afterwards. For example, to see where the markers from the edge go:

```toml
ptrace_initial_psi_n_range = [0.8, 1.0]
```

- psi_n is the tracer's own, as for `ptrace_exit_psi_n`: JOREK's
  (psi - psi_axis)/(psi_limit - psi_axis), not scaled by `real_psi_edge`.
  In a run with an extended boundary, psi_n = 1 is the edge of the extended
  grid, not the original plasma boundary.
- The caption, the printed line and the wetted JSON say how many markers
  the range kept (`markers starting at psi_n 0.8 to 1: 312 of 999`).
- A range's results are written beside the all-marker ones, under names
  that carry it: `particle_exits_psi0.8-1.png`, `particle_wetted_psi0.8-1.png`
  and `.json`. So several ranges can be kept and compared.
- A range no marker started in is skipped, with the psi_n the markers do
  start at.
- Unset, every marker counts, as before.

```bash
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag particle_exits
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag particle_exits --exit-psi-n 0.95
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag particle_exits --animate  # + particle_exits.gif
```

`--animate` (or the case's `animate = true`) also writes
`particle_exits.gif`: the histograms filling in over the trace, 40 frames
from its start to its end, each counting the exits up to that time. Bars
are fractions of all the exits, on the final histogram's scale -- drawn as
a grey outline -- so they grow into it.

- **Units.** The threshold is psi_n as JOREK's particle diagnostics compute it,
  `(psi - psi_axis)/(psi_limit - psi_axis)`, where `psi_limit` is the X-point's
  psi. It is **not** rescaled by `real_psi_edge` the way `theta_target_psi` is.
  With no X-point, `psi_limit` is 0 and JOREK logs a warning in `ptrace.log`.
- **With `ptrace_original_boundary = true`**, leaving that boundary is an exit
  too, and a particle exits at whichever of the boundary or the ψ_N
  threshold it reaches first. The caption counts how many exited each way;
  a large `--exit-psi-n` makes the boundary the only criterion.
- **Leaving the grid first.** A particle that leaves the grid before any
  diagnostics time shows it past the threshold still counts as an exit. Its
  angles are taken from its last position on the grid, and the caption says
  how many exits did this.
- **Resolution.** Exits are only as fine as `diag_step`.
- **Excluded particles.** A particle that is off the grid from the start is
  not counted.
- **Theta.** A rebuilt `ptrace_gc` writes theta about the moving magnetic
  axis. Older `ptrace_diag.h5` files have no theta, so it is computed from R
  and Z about the axis the run's log gives first.
- **Which file.** Whichever particle diagnostics file is in the ptrace folder
  -- `ptrace_diag.h5`, `part_diag.h5` (re_gc) or `diag.h5` (ex6/ex7) -- is
  read, whatever `ptrace_exe` is called, so a `ptrace_gc` built as
  `ptrace_gc_refluid_fixed_T_rho` plots the same. It must hold psi_n, R, Z,
  phi and lost.

### Plotting a trace that has not finished

`particle_exits`, `particle_wetted` and `particle_loss` plot whatever a
trace has written, so one that is still running, ran out of time or was
killed can be looked at as far as it got. Nothing extra is needed: run the
plot as usual.

- The plot says how far the trace got, on the terminal and in the caption:
  `trace unfinished: 45 % of the way to ptrace_end_step 3400` (or to the
  last restart it was given, without a `ptrace_end_step`). The wetted JSON
  records it as `trace_fraction`, `null` for a trace that reached its end.
- A trace stopped *while writing* a diagnostics row leaves that row in
  some datasets only. It is left out, with a note; every earlier row is
  used. A file cut off so badly that it cannot be opened at all cannot be
  recovered: retrace.
- Numbers from an unfinished trace are not comparable with a finished
  one's: fewer particles have reached the wall. The duration in the
  caption says what they cover.

### Where the lost particles started

`plot --diag particle_loss` is the connection-length map with particle loss
as its colour: time along x, the psi_n each particle *started* at along y,
and the colour the fraction of the particles from that psi_n that have left
by that time. Written into the ptrace folder as `particle_loss.png`.

```bash
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag particle_loss
```

- **Lost** means what `particle_exits` counts as exited: past
  `ptrace_exit_psi_n` (or `--exit-psi-n`), outside the original boundary
  under `ptrace_original_boundary`, or off the grid. A particle off the
  grid from the start is not counted. The fraction is cumulative, so each
  row only ever rises.
- **Made to be read beside the connection-length map** (`LCTT_*.png`). The
  time axis is the same simulation time in microseconds, and the colours
  mean the same: green where field lines are short and particles leave, red
  where field lines are long and particles stay. Black is no data -- no
  particle started at that psi_n.
- **The y axes are not the same psi_n.** This one is the tracer's psi_n,
  as for `ptrace_exit_psi_n`; the connection-length map's is `psi_n_in`,
  scaled by `real_psi_edge`. In a run with an extended boundary the two
  differ by that factor.
- **The panel on the right** is how many particles started in each bin. A
  bin with a handful moves in big steps: 4 particles can only show 0, 25,
  50, 75 or 100 %.
- `ptrace_loss_bins` (default 40) sets the psi_n bins, from 0 to the
  outermost start. With `ptrace_initial_psi_n_range` they span that range
  instead, and the file is `particle_loss_psi<min>-<max>.png`.
- Long traces are thinned to 400 times across, evenly.

### How much wall the particles wet

`plot --diag particle_wetted` measures how widely the escaping particles
spread over the wall -- the plasma boundary before `extend_bnd`
(`original_bnd.dat`), so it needs a run prepared with it:

```bash
python ~/ashen/bin/plot --case "qa2.1_g2.3/eta1e-3_RE" --diag particle_wetted
```

Each hit is placed on the wall by its arc length l (from the outboard
midplane, counter-clockwise) and phi -- not by geometric theta, whose equal
steps cover unequal lengths of wall. The crossing is interpolated between
the last diagnostics row inside and the first outside, so it is finer than
`diag_step`; a particle that leaves the grid first is put at the nearest
wall point. Four numbers, each with a bootstrap error bar:

| | what it is |
|---|---|
| `f_pol` | fraction of the wall's poloidal extent wetted, over all phi |
| `f_tor` | fraction of the torus wetted, over all l |
| `f_tot` | fraction of the wall's area wetted, over (l, phi) cells -- also in m² |
| `s` | `f_tot / (f_pol * f_tor)`: 1 for a separable footprint (same poloidal pattern at every phi), well below 1 for a helical stripe, which can reach every angle while wetting little |

Each fraction is a participation ratio, `(Σn)² / Σ(n²/A)` over the cells
(area A, hits n) divided by the wall's area: the area the hits would cover
spread evenly at their mean density -- total load over peak load, but
using every cell rather than the single peak, so steadier with few markers.
Areas are true wall areas, `R dl dphi`, so an outboard cell counts for more
than an inboard one.

The map's colour is a hit density in 1/m^2: the share of all hits in a
cell, divided by the cell's area. It integrates to 1 over the wall, so it
is not bounded by 1 -- a cell of 0.016 m^2 holding 2 % of the hits shows
1.3 -- and even wetting would be 1 / (wall area) everywhere.
`ptrace_wetted_density_range = [min, max]` fixes the colour scale (default:
the data's own range), e.g. to compare cases; cells above `max` take the
top colour.

**Counts instead of fractions.** `--wetted-counts` on the command line, or
`ptrace_wetted_counts = true` in the case, draws numbers of particles: the
map shows how many hit each cell, the profiles how many hit each bin. It is
written as `particle_wetted_counts.png`, beside the fraction version, and
the numbers in the JSON are the same either way. Counts per cell are not
divided by the cell's area, so for the same density a smaller inboard cell
shows fewer -- but they show directly how many hits each cell rests on.
Its colour scale is set apart from the density map's, by
`ptrace_wetted_count_range = [min, max]` in particles per cell (default:
the data's own range); `ptrace_wetted_density_range` doesn't touch it.

**Replotting is cached.** Where each particle hit the wall is kept in
`particle_wetted_cache.npz` in the ptrace folder, so a second run of
`--diag particle_wetted` neither reads the diagnostics file nor finds the
hits again: changing the bins, colour ranges, counts or
`ptrace_initial_psi_n_range` just redraws. The cache is used only while
the diagnostics file (its name, size and modification time),
`original_bnd.dat` and the `ptrace_start_step..ptrace_end_step` times are
what it was made from; otherwise -- a trace still running, retraced,
repacked -- it is recomputed and overwritten. Delete it to force that.

`ptrace_wetted_bins = [n_l, n_phi]` (default `[36, 36]`, or one number for
both) sets the cells. The 2D `f_tot` needs many more hits than cells to be
trusted -- use thousands of markers (`ptrace_initialiser = "current_pdf_simple"`)
-- while `f_pol` and `f_tor` settle with far fewer; halving the bins and
seeing the numbers barely move is a quick check. The figure,
`particle_wetted.png`, is the hit density on (phi, l) with the toroidal and
poloidal profiles alongside; the numbers also go to `particle_wetted.json`
in the ptrace folder. Like the other particle plots it is clipped to
`ptrace_start_step..ptrace_end_step`.

The caption, the printed line and the JSON also say how much of the trace
the hits were collected over (`over 123 µs of trace`; in the JSON
`duration_microseconds`, with `t_start`, `t_end` and `duration` in
seconds). That is the span of the diagnostics times used, after the
clipping -- so for a trace still running, or cut short, it is how far the
data got, not how far it was meant to go. More particles reach the wall
the longer the trace, so compare numbers taken over like spans.

With `diag_step` much above the RK4
`dt`, the exit positions are still interpolated rather than exact -- a
10 MeV electron covers about 3 m per 10 ns, mostly toroidally.

## Housekeeping: `util`

`bin/util` runs one housekeeping function over the runs in `cases.toml`,
chosen with `--func`. Like `analyse` and `plot`, it runs from the folder
holding `cases.toml` and takes `--case` (names, patterns, folders).

```bash
python ~/ashen/bin/util --func trace_organize --dry-run   # what it would do
python ~/ashen/bin/util --func trace_organize             # do it
python ~/ashen/bin/util --func trace_organize --case 'qa2.1*'
```

**`trace_organize`** tidies every trace folder under each run:

- **Removes leftover restart links** (`jorek*.h5`) from trace folders.
  ashen links the restarts in while a trace runs and removes them after,
  but an interrupted trace, a failed staging or an unconcluded queued job
  leaves them behind. They are the program's input, never a result. A link
  frees no space; a *copy* frees its size, and the summary says how much.
  The run's own restarts are never touched.
- **Moves traces from older layouts** to where the plots look now,
  `<run>/ptrace/<exe>/E<eV>eV_n<markers>/`:
  - `<run>/trace/<exe>/`, from before "trace" became "ptrace". Its files
    take today's names: `trace.log` -> `ptrace.log`, `trace_meta.json` ->
    `ptrace_meta.json`, `trace_diag.h5` -> `ptrace_diag.h5`.
  - a trace loose in `<run>/ptrace/<exe>/`, from before traces got a folder
    per energy and marker count.

  The folder name comes from the trace's **own** settings files
  (`ptrace_settings.nml`; or `ptrace_overrides.nml` over
  `ptrace_params.nml`; or `trace_params.nml`), so it names what that trace
  ran with, whatever `cases.toml` says now. A program without settings
  (re_gc, ex6/ex7) goes to `<run>/ptrace/<exe>/` itself.

It leaves alone, saying why:

- a folder whose queued job may still be running (it asks SLURM, as
  `ptrace` does);
- a `ptrace_gc` trace whose settings files do not give its energy and
  marker count -- move that one by hand;
- a trace whose new place already holds another trace. Nothing is
  overwritten.

**`downsample_restarts`** deletes restart files to give back disk space:
it keeps the restarts whose step is a multiple of `--every`, so a run that
saved every 20 steps can be cut to every 40.

```bash
python ~/ashen/bin/util --func downsample_restarts --every 40           # what it would delete
python ~/ashen/bin/util --func downsample_restarts --every 40 --apply   # delete
```

- **Deleted restarts cannot be recovered**, so without `--apply` it only
  says what it would delete and how much space that frees.
- **Always kept**, multiple or not: every step the case uses in
  `cases.toml` (its `steps`, each diag's own steps, `ptrace_start_step`,
  `ptrace_end_step`, `ptrace_pdf_step`), the run's first restart and its
  last (the one to continue from). It lists what it kept and why.
- Only files named exactly `jorek<5 or 6 digits>.h5` are touched;
  `jorek_restart.h5` and everything else stay.
- A case whose `steps` asks for every restart (`steps = {}`, or `{ step =
  20 }` on a run saved every 20) keeps them all. Change `steps` first.
- Gathered caches (zeroD, Poincare, four) for deleted steps stay. A trace
  over the run's restarts changes when its restarts do, so it shows as
  out of date; its existing outputs still plot.

**`delete_figures`** deletes the figures `plot` drew for each run, to free
disk space. `plot` draws them again from the gathered data whenever you
want them back (`plot --case X`, or with `--diag` for just some).

```bash
python ~/ashen/bin/util --func delete_figures                  # what it would delete, and the space
python ~/ashen/bin/util --func delete_figures --apply          # delete
python ~/ashen/bin/util --func delete_figures --case 'qa2.1*' --apply
```

- **Deleted:** `.png` and `.gif` files where `plot` writes them --
  `poinc_dir/`, `four_dir/`, `profiles/` -- and in each trace folder the
  `particles`, `particle_exits*`, `particle_wetted*` and `particle_loss*`
  figures.
- **Kept:** every cache and trace output (`.h5`, `.dat`, `.json`, ...), any
  other image in a trace folder, and anything linked rather than a file.
- **Not touched:** comparison figures in the campaign's `figures/`, which
  belong to no one run.
- Like `downsample_restarts`, it only says what it would delete and how
  much space that frees until given `--apply`.

**`compress_traces`** repacks the particle diagnostics files of traces
made before ashen did this itself (below), to the size of their data:

```bash
python ~/ashen/bin/util --func compress_traces            # how much it would free
python ~/ashen/bin/util --func compress_traces --apply    # repack
```

JOREK's diagnostics writer chunks every dataset 50000 particles x 1 time,
uncompressed, and HDF5 stores each chunk at full size: every diagnostics
row takes the room of 50000 particles however many are traced. A
`ptrace_gc` trace writes 2.8 MB a row, so 120 us at `diag_step = 1e-8` is
about 34 GB -- with 1000 markers, ~50 times its data. Repacked, that file
is well under 1 GB.

- **Lossless.** The new file is written beside the old one, read back and
  checked against what was read from the original (a hash of every block
  of values), and only then put in its place; on any failure the original
  stays. The plots read the repacked file exactly as before.
- **Every layout:** `ptrace_diag.h5`, `part_diag.h5`, `diag.h5`, and the
  oldest `trace_diag.h5`.
- **Left alone:** a file already compressed, one whose queued job may be
  running, and one written to in the last ten minutes (its trace may still
  be running and appending to it).
- It reads the whole file to repack it -- once, padding included -- and
  compresses on up to 8 cores, so a 34 GB file takes a minute or two.

## Simulation time at a restart step

`bin/timestep` is a one-off lookup, not a `cases.toml`-driven gather: run it
from inside a prepared run folder to see one or two restart steps' simulation
time, in both SI seconds and JOREK's own code units:

```bash
cd Columbia/NL_kinks/qa2.1_g2.3/eta1e-3_RE
python ~/ashen/bin/timestep 3000
# step 3000: t = 1.234500e-04 s (SI), t = 5.678900e+02 (JOREK units)

python ~/ashen/bin/timestep 3000 3200
# step 3000: t = 1.234500e-04 s (SI), t = 5.678900e+02 (JOREK units)
# step 3200: t = 1.334500e-04 s (SI), t = 6.123400e+02 (JOREK units)
# Δt (step 3000 -> 3200): 1.000000e-05 s (SI), 4.445000e+01 (JOREK units)
#   = 5.000000e-08 s/step (SI), 2.222500e-01 /step (JOREK units), over 200 steps
```

Always re-runs `jorek2_postproc`'s `zeroD_quantities` (once per unit system,
via `si-units`/`jorek-units`) rather than trusting an existing zeroD cache,
since that cache (from `analyse --diag zerod`) is SI-only and this tool's
whole point is the JOREK-unit side. `--namelist` picks which namelist to read
(default `in_main`).

## Layout

```
bin/            entry-point shims; the only place sys.path is touched
src/ashen/
  config.py     site.toml discovery and path resolution
  namelist.py   Fortran namelist reading and editing
  paths.py      run-folder conventions (derived filenames)
  padding.py    restart-step padding: every 5- vs 6-digit decision
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
                connection_length.py, timestep.py -- pure math, no matplotlib
  logfile.py    scalar extraction from a JOREK log (R_axis, etc.)
  plotting/     poincare.py, connection_length.py, colors.py, style
  cases.py      cases.toml loader for bin/analyse, bin/plot and bin/ptrace
  particle_programs.py  the JOREK particle programs bin/ptrace wraps
  tracing.py    stage and run a case's ptrace
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
