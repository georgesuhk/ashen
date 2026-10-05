# Changelog

What changed in ashen, newest first. Each entry gives the commit, what it
does, and how to use it. The README has the full reference for every key
and flag named here.

## 2026-10-05: `util --func downsample_restarts`

- Keeps only the restarts whose step is a multiple of `--every` (e.g. 40
  to go from every 20 steps to every 40), deleting the rest.
- Never deletes a step `cases.toml` uses, or a run's first and last
  restart.
- Shows what it would delete until given `--apply`; deleting cannot be
  undone.

## 2026-10-05: `util --func trace_organize`

New entry point `bin/util`, one housekeeping function per `--func`. The
first is `trace_organize`, for a full disk quota and the trace layouts
left by earlier versions:

- Removes leftover restart links and copies (`jorek*.h5`) from every trace
  folder. A copy frees its size; the summary says how much in total.
- Moves traces from older layouts (`<run>/trace/<exe>/`, or loose in
  `<run>/ptrace/<exe>/`) into `<run>/ptrace/<exe>/E<eV>eV_n<markers>/`,
  named from each trace's own settings files, renaming old file names to
  today's.
- Leaves alone a folder whose queued job may be running, a `ptrace_gc`
  trace whose energy cannot be read, and any place already taken.
- `--dry-run` shows everything first. Run it before the real thing.

## 2026-10-01 and 2026-10-02: particle tracing and plotting session

Fifteen changes, all merged into `main` (`45fb7d8` to `ac00300`). Nothing
has been pushed.

### Before using any of this

**Rebuild the tracer.** Copy `fortran/ptrace_gc.f90` into the JOREK
checkout's `particles/examples/` and run `make ptrace_gc`. A binary built
before `1037832` stops at once, because it looks for settings files that
no longer exist.

**Update `cases.toml`.** For each case that traces with `ptrace_gc`:

- Remove `ptrace_inputs = ["ptrace_params.nml"]`. It is now an error.
- Set every tracer setting as a `ptrace_<name>` key.
- `ptrace_n_markers`, `ptrace_E_kin_eV` and `ptrace_cos_pitch` are
  required.

**Existing traces.** They sit loose in `<run>/ptrace/<exe>/`. The plots
now look in `<run>/ptrace/<exe>/E<eV>eV_n<markers>/`. Move the files into
that folder to keep plotting them; the plot tells you which folder.

### Not yet verified

These have only been tested on a laptop, with stand-ins for JOREK and
SLURM.

| What | Status |
|---|---|
| The Fortran changes in `ptrace_gc.f90` | Never compiled. Expect small build fixes. |
| `ptrace --run` (sbatch, squeue, sacct) | Tested against fake SLURM commands only. |
| The names of SLURM's `.out` / `.err` files | Expected to be `2h.<jobid>.*`; not confirmed. |
| The new figures | Checked on synthetic data, not on a real trace. |
| The early-stop rule | A Python copy of the rule is tested on made-up loss curves. The Fortran itself has not run. |

### Running traces

**`ptrace --run_i`, `ptrace --run`, default steps** (`c0fd87b`)

- `ptrace --run_i` runs here, with `ptrace_n_mpi` and `ptrace_omp_threads`.
- `ptrace --run` queues with `sbatch` and `jobscripts/2h`. `-job 23h`
  picks another jobscript. The jobscript sets ranks and threads.
- `ptrace` with no flag runs nothing. It reports each trace's state, and
  finishes off any queued job that has ended.
- `ptrace_start_step` and `ptrace_end_step` default to the run's first and
  last restart.
- The tracer spreads markers over threads dynamically. Results are
  unchanged.

**Settings live in `cases.toml`** (`1037832`)

- `ptrace_params.nml` and `ptrace_overrides.nml` are gone.
- ashen writes every setting, defaults included, to `ptrace_settings.nml`
  in the trace folder. That file is the record of what the trace ran with.
- The settings are printed when a trace runs or is queued.
- Settings are checked when `cases.toml` loads. Under `markers`, one value
  now applies to every marker; before, a short list left the other markers
  at 0 eV.
- Snapshots scale with the trace: `ptrace_n_snapshots` (default 100),
  spaced at a round interval. `ptrace_snapshot_step`, if set, is used
  instead.

**A folder per energy and marker count** (`3c09072`)

- A trace goes in `<run>/ptrace/<exe>/E10000000eV_n1000/`, so changing
  `ptrace_E_kin_eV` or `ptrace_n_markers` no longer overwrites.
- Changing any other setting still retraces in the same folder.
- Printed energies read `10 MeV`, `500 keV`. Files keep eV.

**Stopping a trace early** (`a7be83a`)

- Always on: the trace ends when every marker is off the grid.
- `ptrace_stop_when_stalled = true` ends it when the loss rate falls below
  5 % of its peak (`ptrace_stall_rate_fraction`).
- The rule is only checked once 10 % of markers have left
  (`ptrace_stall_min_lost`) or half the trace has passed
  (`ptrace_stall_min_time`).
- "Left" means outside the original plasma boundary in an
  extended-boundary run, otherwise off the grid.
- Known weakness: a small early loss, a quiet gap, then the main loss
  after half-time. The rule stops at half-time and misses the main loss.
  Raise `ptrace_stall_min_time` for such cases.

### Plots

| Change | Commit | What it does |
|---|---|---|
| Particles view | `343be71` | `--diag particles` is framed on the plasma boundary, plus 25 % with `extend_bnd` and 10 % without. Far-off particles are not drawn. |
| Last snapshot | `459fa76` | `particles.png` is one panel, the last snapshot. `--animate` adds the GIF of all of them. |
| Animation length | `1037832` | The GIF plays in about ten seconds whatever the frame count. |
| Wetted labels | `d4e8f8a`, `c4a3610` | Colourbar reads "(fraction of total hits) / m²". Y axis reads "poloidal arc length". |
| Wetted colour range | `d4e8f8a` | `ptrace_wetted_density_range = [min, max]` fixes the colour scale. |
| Wetted duration | `8716bf2` | Caption and JSON say how many µs of trace the hits cover. |
| Starting psi_n range | `5c21d1c` | `ptrace_initial_psi_n_range = [min, max]` counts only markers that started in that range. Output files carry the range in their name. |
| Loss map | `a8180a1` | New `--diag particle_loss`: time against starting psi_n, coloured by fraction lost. `ptrace_loss_bins` sets the bins. |
| Unfinished traces | `13f03c9` | A trace that stopped early is plotted as far as it got, and the caption says so. A half-written last row is skipped. |

### Working with cases

| Change | Commit | What it does |
|---|---|---|
| Chained plots | `1e25296` | `analyse --diag four -plot four` gathers, then plots. Options `analyse` does not know are passed to `plot`. |
| Case patterns | `3ddf9cd` | `--case 'qa2.1*'` selects by pattern; `--case qa2.1_g2.3` selects every case in that folder. Works in `analyse`, `plot` and `ptrace`. |
| Open-ended steps | `f564d83` | `steps = { step = 400 }` takes its start and end from the run's restarts. `steps = {}` is every restart. |

### Choices made that you may want to revisit

These were decided during the session without an explicit answer from you.

1. **Loss-map colours.** Green means lost, to match the connection-length
   map where green is short field lines. Read alone, that may look
   backwards.
2. **Loss-map y axis.** It uses the tracer's psi_n, not `psi_n_in`. In an
   extended-boundary run the two differ by `real_psi_edge`, so the loss map
   and the connection-length map do not line up vertically. Converting
   needs confirmation that the tracer's psi_n and the Poincaré tool's are
   the same quantity.
3. **`ptrace_initial_psi_n_range`** uses the tracer's psi_n as well, for
   the same reason.
4. **Trace folders** are keyed on energy and marker count only. `cos_pitch`
   is not part of the name.
5. **`-plot` does not choose what is gathered.** `analyse -plot poincare`
   alone gathers nothing new.
6. **Early stopping is off by default**, and never fires while no marker
   has left at all.
7. **Particles view margin.** "25 %" means 25 % wider and taller overall,
   so 12.5 % on each side.
8. **Step ranges** count from the run's first restart, not from zero.

### Worth knowing

- **Sparse wetted maps read low.** With fewer hits than cells, `f_tot` and
  `s` are underestimates. 665 hits in 36 × 36 cells gave `f_tot` = 0.16; a
  rough correction puts it near 0.23. Use more markers or fewer bins. A
  bias-corrected estimator was offered and not yet built.
- **A queued job is only confirmed finished** when `ptrace` is run again
  for that case. The jobscripts pipe through `tee`, which hides the exit
  status, so success is judged from SLURM's state and the final
  `part_restart.h5`.
- **`diag_step` and `dt`** are the two settings that make a trace cheaper.
  Both change the results, so neither was touched.

### Deferred

- A better current-profile initialiser, closer to JOREK's (3D current,
  energy and pitch ranges). The design was being discussed when the
  session moved to the settings change.
- Extending a finished trace instead of rerunning it.
- Comparing wetted area across cases in the `--compare` machinery.
