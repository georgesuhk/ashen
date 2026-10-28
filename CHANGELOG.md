# Changelog

What changed in ashen, newest first. Each entry gives the commit, what it
does, and how to use it. The README has the full reference for every key
and flag named here.

## 2026-10-28: bigger axis labels on the wetted-area figure

- Every axis label on `particle_wetted.png` (map, both profiles, colourbar)
  is now 18 pt, up from the style's 12. Tick labels and the caption keep
  their size. The size is `LABEL_SIZE` in `ashen/plotting/particle_wetted.py`.

## 2026-10-27: job names from qa, g, li and eta

- SLURM jobs are named after the run's qa, g, li and eta, written as in
  scan folder names: `qa2.1_g2.3_li1.2_eta1e-3` for `run_jorek` and
  `pt_qa2.1_g2.3_li1.2_eta1e-3` for `ptrace --run` (was `2.3_0.001` and
  `pt_<last part of the case name>`). li appears only for
  `ffprime_method = "q_li"`; a run on a campaign boundary has its boundary
  in place of qa and g (`boundary1_li1.2_eta1e-3`). `squeue` shows 8
  characters of a name by default: `squeue -o "%.10i %.30j %.8T %R"`
  shows it whole. ptrace reads these from the
  run folder's `shotfile.py`, and falls back to the old name without one.
- `Columbia/jobscripts` has 1-node versions, `1node_2h` and `1node_23h`
  (`p.tok.openmp`, QOS `p.tok.openmp.2h` / `.24h`, still 8 tasks x 32
  threads). Use them with `jobscript = "1node_2h"` or `--job 1node_2h`.

## 2026-10-27: Poincaré steps of their own from the case viewer

- The case viewer's "Steps to analyse (cases.toml)" section has a second
  row for the Poincaré steps alone. It has its own start step and step size,
  a **Copy from case steps** button that fills them from the case's row,
  and **Save Poincaré steps**, which writes
  `[cases."<run>".poincare] steps = { start = ..., step = ... }`. It
  previews them the way the case's row does. `analyse --diag poincare`
  reads that table already (`Case.steps_for`), so tracing can take fewer
  steps than the rest. Saving refuses a case not yet in cases.toml.
- `cases_edit.set_case_steps(..., diag="poincare")` makes that edit: it
  replaces the table's `steps`, or adds the table after the case and its
  other tables. As before, it writes nothing unless the result parses to
  the old file with only that value changed.
- Unverified: not yet clicked through in a live notebook; tested through
  the widgets in the unit tests.

## 2026-10-23: time-step suggestions from the runs' GMRES iterations; logs kept

- `run_jorek shotfile.py --suggest-tstep` suggests the time step for the
  folder's next job. It prints every limit and which one sets the step.
  `--apply` writes `tstep_n` / `nstep_n`, keeping the job's length, with
  the old values in a comment. The suggestion comes from the GMRES
  iteration count of every step on record: a step fails when GMRES reaches
  `gmres_max_iter`. N ~ dt^q is measured where a crash and its restart
  stepped the same state, and taken on the safe side. Further limits are
  2 x the largest step run, half a NaN crash's step, and
  `tstep_accuracy / gamma`. A run with no history takes the hardest step of
  the most similar runs. New shotfile keys `tstep_safety` (0.5) and
  `tstep_accuracy` (0.1). README, "Choosing the time step".
- Each launch (`--run`, `--run_i`, `--run_r`, also with `--keep-inputs` and
  `--scan`) first copies `log` into `logs/log_s<first>-s<last>[_<ending>].gz`
  and updates `tstep_history.json.gz`. If the log cannot be copied, nothing
  is launched.
- `util --func archive_logs` files the logs already there, `log` and
  SLURM's `.out` copies. It is preview first; `--apply` deletes each `.out`
  once kept. On a copy of the two NL_kinks runs, 654 MB of `.out` files
  became 47 MB. `util --func tstep_harvest` rebuilds the records.
- A log two jobs wrote at once is read only up to where the second job
  starts, and named `_mixed`. The local `qa2.1_g1.8/eta1e-5_adv0.1/log` is
  one: the job restarted from step 3400 while the job from step 960 was
  still writing.
- Unverified: not yet run on the HPC; the q range comes from one crash and
  restart pair; scaling limits between runs of different eta or density is
  not attempted.

## 2026-10-23: `diag_exe_folder` picks the run folder's jorek2_* tools

- New shotfile key `diag_exe_folder = "<name>"`: `run_jorek` links the
  diagnostic tools (`jorek2_postproc`, `jorek2_poincare`, ...) from
  `template/symlink/<name>/` in place of `RE` or `standard`. For runs
  whose model or number of variables differs from both, where the usual
  postproc stops with "The number of variables in the restart file and the
  compiled JOREK binary does not agree" and every zeroD, qprofile and
  profile gather comes back "not produced".
- Unset, the choice is as before (`RE` if `with_refluid`, else
  `standard`). `util/` still follows `with_refluid`. A folder that does not
  exist stops `run_jorek` before anything is written.
- Applies when a folder is prepared. For a folder prepared already, relink
  `jorek2_postproc` by hand (README, "Which jorek2_* tools a run folder
  gets").

## 2026-10-23: case viewer shows step and time together

- The case viewer's time-series panels carry a second x axis along the top:
  the time step over a time axis (magnetic energies, runaway current
  totals and ratio map), the time [ms] over a step axis (four view
  amplitudes). Taken from the run's `macroscopic_vars.dat` `times` lines,
  so it stays right when `tstep` changed during the run. A folder without
  that file shows the panels as before.
- Fix: reading `times` from `macroscopic_vars.dat` failed, because its
  column-name line has no leading `%`.

## 2026-10-23: restarts staged under both step widths

- Every `jorek2_*` call (postproc, four, ...) now finds its restart in the
  scratch folder as both `jorek08002.h5` and `jorek008002.h5` (hard links,
  no extra disk). An older RE-fluid `jorek2_postproc`, which only opens
  5-digit names, therefore works on a run written by a 6-digit build:
  put that postproc in the run folder and `analyse --diag re_current` as
  usual. Nothing on disk is renamed. Steps from 100000 up have no 5-digit
  name, so an old postproc still cannot read those.

## 2026-10-20: `area` labelled "wetted area"

- The scan-map quantity `area` is now labelled "wetted area [m²]" (was
  "effective wetted area [m²]"), on its axis and in its description. Its
  values, name and logarithmic axis are unchanged.

## 2026-10-20: `peak_hits`; `peak_fraction` renamed

- New `particle_wetted` measure `peak_hits`, in the JSON and as a
  scan-map quantity (no error bar): the hits in the busiest (l, phi) cell,
  or with `peak_fraction` the mean hits per cell over the busiest cells
  filling that share of the wall's area. A check on how many hits the peak
  rests on; it grows with the number of markers.
- `ptrace_wetted_peak_fraction` is now `peak_fraction` in cases.toml, and
  `--ptrace-wetted-peak-fraction` is now `--peak-fraction`. The old names
  are gone (cases.toml reports the old key as unknown). It is now checked
  on every case, with or without `ptrace_exe`.

## 2026-10-20: the wetted peak averaged over the densest part of the wall

- `ptrace_wetted_peak_fraction = x` (case key) and
  `--ptrace-wetted-peak-fraction X` (flag, overrides it, also on scan
  maps) make `peak_density` and `peaking` the mean hit density over the
  densest `x` of the wall's area -- densest cells first, the last in part
  -- instead of the single densest cell. Much less noisy and less
  bin-dependent. Default 0: the densest cell, as before, so existing
  numbers are unchanged. The JSON records it as `peak_fraction`.

## 2026-10-20: the peak of the particle wetted map

- New `particle_wetted` measures, in `particle_wetted.json` and as
  scan-map quantities with bootstrap error bars:
  - `peak_density`: the map's maximum, the densest (l, phi) cell's share
    of all hits per m² [1/m²], exactly the top of the drawn map;
  - `peaking`: `peak_density × area`, the peak over the hit-weighted mean
    density. 1 for even wetting, larger for a peaked footprint.
- Both are biased high with few hits, and `peak_density` depends on
  `ptrace_wetted_bins` (README). The existing measures are unchanged.

## 2026-10-20: `ptrace` says what a finished trace covered

- For a finished `ptrace_gc` trace -- `[cached]` in the status or on a
  rerun, just done, or a queued job found ended -- `ptrace` prints the
  markers and how many were lost, the time traced against the time it
  was set to trace (µs), the JOREK steps that spans (`3000 -> ~4882 /
  5000`, interpolated between restarts), and why it stopped early if it
  did (losses stalled, every marker off the grid). Read from `ptrace.log`
  (`ptracing.trace_summary`); nothing is printed for other programs.
- Tested on logs in `ptrace_gc.f90`'s formats; not yet on the HPC.

## 2026-10-19: `--color` for the `particle_exits` comparison

- `plot --compare ... --diag particle_exits --color 'red, green'` sets each
  case's colour, comma-separated, in the comparison's case order (also
  `--colors`). A skipped case takes its colour with it. Too few or too
  many colours: reported, figure skipped. An invalid colour, or `--color`
  without `--compare`, is an error. Unset: tab10, as before.

## 2026-10-19: part of a colour map for `particle_wetted`

- `ptrace_wetted_cmap_range = [lo, hi]` (case key) and
  `--ptrace-wetted-cmap-range LO HI` (flag, overrides it) use only that
  part of `ptrace_wetted_cmap`, fractions 0..1 of it, stretched over the
  colourbar. `hot` with `[0, 0.9]` ends in light yellow rather than
  white; matplotlib has no such map of its own. Colours only: the
  colourbar's values and `ptrace_wetted_density_range` are unchanged.
  Ranges outside 0 <= lo < hi <= 1 are an error in both places.

## 2026-10-19: `particle_wetted` without the side profiles; colour of empty cells

- `ptrace_wetted_profiles = false` (case key) draws the `particle_wetted`
  map alone, without the toroidal and poloidal profiles; the colourbar
  sits beside the map. `--no-ptrace-wetted-profiles` and
  `--ptrace-wetted-profiles` override the key for one run. Default on, as
  before. Applies to the counts version too; the numbers are unchanged.
- `ptrace_wetted_empty_color` (case key) and `--ptrace-wetted-empty-color`
  (flag, overrides it) colour the map's cells that no particle hit; any
  matplotlib colour. Unset, they stay white, as before. An invalid colour
  is an error in both places.

## 2026-10-19: `particle_wetted` colour map; shorter arc length label

- `ptrace_wetted_cmap` (case key) and `--ptrace-wetted-cmap NAME` (flag,
  overrides it) set the map's colour map, for the density and the counts
  version alike. Any matplotlib name; default `"viridis"`, as before. An
  unknown name is an error in both places.
- The map's y axis now reads "poloidal arc length l [m]", without
  "(0: outboard midplane, ccw)". Where l starts and which way it runs is
  in the README's particle_wetted section.

## 2026-10-19: particle_exits compared across cases; `--compare` lists; `--ptrace-exit-bins`

- `plot --compare ... --diag particle_exits` draws every selected case's
  theta and phi exit histograms on one pair of axes, a colour each, as
  `figures/<name>_particle_exits.png`. Each case is computed exactly as its
  own `particle_exits` figure (shared code, `_particle_exits_result`), and
  each is a fraction of its own exits. A case without a trace is skipped
  with the reason.
- `--compare` takes a comma-separated list of cases, patterns or folders,
  as one comparison, in the order given:
  `plot --compare 'qa2.1_g2.8/eta1e-3, qa2.1_g2.8/eta1e-4' --diag particle_exits`.
  This works for every diag `--compare` draws. If any item selects nothing,
  the whole value is reported as unknown.
- `--ptrace-exit-bins N` (or `--ptrace_exit_bins`) overrides each case's
  `ptrace_exit_bins` for `particle_exits`, per case and under `--compare`.
  It must be a whole number >= 1.
- Flags that override a cases.toml key now carry the key's name:
  `--ptrace-exit-psi-n` and `--ptrace-wetted-counts` are new names for
  `--exit-psi-n` and `--wetted-counts`. The old names still work.

## 2026-10-19: `run_jorek --legacy-density`

- `--legacy-density` writes `rho_prof.dat` the old, incorrect way, as
  before 2026-10-07: `rho_const / 1e20` in units of `central_density`, so
  `rho_const = 1e18` runs at 1e16 m⁻³. For reproducing or continuing runs
  prepared then. Only `rho_prof.dat` differs; it prints a note when used.
  Works with a shotfile and with `--scan --apply`.
- `--run_r` now also refuses, writing nothing, when preparing the folder
  again would switch its `rho_prof.dat` between the two formats, and gives
  the command that keeps it (`--legacy-density`, or `--keep-inputs`).
  (`runner.restart_rho_prof_change`)
- `--legacy-density` with `--keep-inputs` is an error: that writes no
  profiles.
- Tested on synthetic folders with the submit command intercepted. Not
  run on the HPC.

## 2026-10-19: scan maps draw f_tot on a linear axis

- `f_tot` is now drawn on a linear scale on scan maps, like `f_pol`,
  `f_tor` and `s`. Its values are unchanged.
- `area` is now labelled "effective wetted area [m²]", which is what it
  is: the participation-ratio area (Σn)² / Σ(n²/A) over (l, phi) cells,
  equal to `f_tot` times the wall's area. Its axis stays logarithmic.

## 2026-10-15: re_current against R as well as psi_N

- New case key `re_current_coords`: the radial coordinate(s) the
  re_current current-density profiles are gathered and drawn against,
  `"Psi_N"` (the default, as before) and/or `"R"` (major radius, m).
  `re_current_coords = ["Psi_N", "R"]` gives both.
- `analyse --diag re_current` gathers each listed coordinate (two more
  `jorek2_postproc` calls per step for R). `plot` draws each:
  `re_current_R.png` and `re_current_ratio_R.png` beside the psi_N ones,
  whose names are unchanged. The R figures have no plasma-edge line, and
  the ratio map's R axis runs over the R the profiles cover.
- Case viewer: the "Runaway and thermal current" section has a **ψ_N** /
  **R** switch before the step slider. Switching to a coordinate that has
  no profiles yet says to add it to `re_current_coords` and press **Run
  analyse --diag re_current**.
- `plot --step all` counts a step as cached for re_current when it has
  both profiles against any of the case's coordinates.
- Default output unchanged: with psi_N only, `re_current.png` and
  `re_current_ratio.png` are byte-identical to before. In code,
  `CurrentDensity.psi_n` and `RatioMap.psi_n` are now `.x`, with a
  `coords_var` beside them.

## 2026-10-15: delta-B and mode amplitudes about 8x faster

- Reading the jorek2_four caches was nearly all the time of delta-B
  (`plot --diag four`, `delta_b_*` on scan maps, the `four` comparison)
  and of the mode-amplitude series: every step's whole cache was read,
  all of its variables (Psi, T, n_RE, omega, rho), when these use one.
  Each now reads only the variables it uses (Psi for delta-B).
- On the local qa3.3_g3.2/eta1e-3_adv0.1 run (43 steps): delta-B series
  6.1 s -> 0.7 s; max-amplitude and rational-surface series 4.9 s ->
  0.6 s each.
- Results unchanged: on that run every delta-B, delta-B/B (edge),
  max-amplitude and rational-surface value, and the edge minor radius,
  are bit-identical to before. `four_cache.read_cache` takes an optional
  list of variables.
- Still read once per figure or quantity rather than once per run, so a
  `--diag four` plot reads each cache a few times; at ~15 ms a step now,
  not worth the extra machinery yet. Not timed on the HPC.

## 2026-10-15: scan-map `li` is the plasma's own, from the q-profile

- **Changed meaning:** the `li` quantity is now the plasma's own l_i,
  from the step's q-profile inside JOREK's plasma edge (`real_psi_edge`),
  cylinder definition (`diagnostics.equilibrium.achieved_q_li`). This is
  the "plasma only" curve under the re_current ratio map, and the l_i the
  q0/l_i/qa model is built with.
- JOREK's zeroD value is still there as **`li3`**: everything inside the
  last closed surface, which with an extended boundary includes the
  vacuum. A comparison with `x_quantity = "li"` (or y/c) now plots the
  plasma value; write `li3` for the old one.
- Read at the equilibrium step, as before. It needs that step's q-profile
  and zeroD caches (with `psi_axis`, `psi_bnd`, `R_axis`), topped up on
  demand like `edge_q`'s, and F0 from the namelist. A case missing them
  drops out with a note.
- On the local qa3.3_g3.2/eta1e-3_adv0.1 run: li = 1.392 against
  li3 = 1.644 at step 5000, and 1.356 against 1.599 at step 26000.

## 2026-10-15: `plot --si`: eta in ohm metres on scan maps

- `--si` (or `si = true` on a comparison) plots `eta` in Ω·m: the
  namelist's eta times `sqrt(mu0/rho0)` from the NORMALIZATION FACTORS
  block of each run's own `log` (`logfile.sqrt_mu0_over_rho0`), the
  factor JOREK's own SI output uses. Per run, so each point uses the
  value its build used.
- A run whose log lacks the line drops out of an SI map with a note; no
  factor is guessed.
- **Label fix:** without `--si`, eta's axis said "[Ω·m]" while plotting
  JOREK units. It now says "(JOREK units)". The values are unchanged.
- Only eta has a JOREK-units value among the quantities, so `--si`
  changes nothing else.

## 2026-10-15: the particle wetted area on scan maps, with error bars

- New scan-map quantities `f_tot`, `f_pol`, `f_tor`, `s` and `area`
  (m²): the `particle_wetted` measures, computed by the same code as the
  per-case figure, so the numbers are identical to its caption and
  `particle_wetted.json`. Example, with a case pattern:
  `plot --compare 'qa2.1_g2.8/eta*0.1' --diag scan_map --x-quantity eta --y-quantity f_tot`.
- Scan maps now draw error bars for quantities that have an uncertainty:
  these five, from their bootstrap errors. Other quantities are unchanged.
  The printed summary gives `value ± error`.
- Each case says once how many hits its numbers rest on and the bins,
  e.g. `particle_wetted from 400 hits of 400 particles, 36 x 36 bins`,
  since f_tot and s are biased low when hits are few for the bins (see
  the README). A case without a trace or without hits drops out of the
  map with a note, as for other missing quantities.
- `--list-quantities` lists them. The computation is shared through
  `cli/plot.py::_particle_wetted_numbers`, which `--diag particle_wetted`
  now also uses; its figure and JSON are unchanged.

## 2026-10-15: `plot --compare` takes a case pattern

- `--compare` accepts the same selectors as `--case` when no
  `[comparisons.NAME]` has that name: a pattern or a folder. The cases it
  selects are compared as one, with no table in cases.toml:
  `plot --compare 'qa2.1_g2.8/eta*0.1' --diag theta_hist`.
- It prints which cases it took and what the figures are called. Labels
  are each case's path below the folder they share; file names are the
  pattern with unsafe characters made `_` (`qa2.1_g2.8_eta_0.1_*.png`).
- A defined comparison of the same name always wins.
- Works for `theta_hist`, and for `scan_map` given `--x-quantity` and
  `--y-quantity`. `wetted_fraction` and the `four` delta-B comparison need
  `x_values`, which a pattern has none of: they say so and skip.
- A value that names no comparison and selects no case is an error, as an
  unknown comparison was.

## 2026-10-14: Poincaré plots in the case viewer

- New section "Poincaré plots (analyse --diag poincare)", before the
  Fourier modes. It shows one step's cached punctures at a time, drawn as
  `plot --diag poincare` draws them (coloured by each line's starting
  psi_N), with the plasma and JOREK grid boundaries over them.
- **◀** / **▶** step to the previous / next cached step; the slider jumps
  to any. **point size** sets the marker size (log scale, default 0.1, the
  cases.toml default).
- Steps are those with a Poincaré cache (the same lookup as
  `plot --step all`). The title gives the step's time when its zeroD cache
  has one, and a note gives the number of lines and punctures.
- **Run analyse --diag poincare** above it gathers for this run's case,
  then redraws. Tracing can take long; it runs in the notebook until done.
- From a cell of your own: `viewer.poincare_figure(".", step, point_size=0.3)`.
  Tested on synthetic caches only.

## 2026-10-14: a start step for the case viewer's steps section

- "Steps to analyse (cases.toml)" has a **start step** box beside **step
  size**. It writes `steps = { start = <start>, step = <size> }`: the
  restarts at start, start + size, ... up to the last restart, picked up
  as the run goes on, as before.
- Both boxes open at the values already saved for the case, else at 0 and
  the restart spacing.
- The preview says when the start is after the last restart, when no
  restart is at start plus a multiple of the size, and when the last
  restart is left out.

## 2026-10-14: `plot --step all`: every cached step

- `plot --step all` draws, for each diag, every step that diag already has
  a cache for, per case, instead of the steps cases.toml lists. Example:
  `plot --case NAME --diag four --step all`.
- What counts as cached, per diag:
  - poincare, connection_length, theta_hist, wetted_fraction: the Poincare
    caches (`poinc_dir/poinc_s<step>.h5`, or the legacy `.npz` set);
  - four: `four_dir/four_s<step>.h5`;
  - profiles: a cache for any of the case's `vars` and `tor_mode`s;
  - re_current: steps with both `currdens` and `recurrdens`.
  Any step padding is recognised. Figures in the same folders are not
  mistaken for caches.
- It is a value of the existing `--step` flag, so it takes precedence over
  per-diag `[cases.NAME.<diag>]` steps, as `--step N` does. `--step all`
  with step numbers next to it is refused.
- Works with `--compare` too, per member case. A scan_map quantity that
  reduces over no diag (eta, q95, ...) uses the steps with a zeroD cache.
- Each diag prints how many steps it found, e.g.
  `poincare: --step all, 3 cached step(s) (100 to 300)`. A diag with none
  is skipped with a message rather than drawn empty.
- Nothing is gathered; the particle diags take no steps and are unchanged.
  (`diagnostics/cached_steps.py`)

## 2026-10-12: `analyse --diag re_current` gathers the q-profile too

- `--diag re_current` now also gathers each of its steps' q-profile
  (`postproc/qprofile_s<step>.dat`), as `--diag four` does. That is what
  the plasma-only l_i under the thermal / runaway ratio map is computed
  from, so `re_current` alone now gives both l_i curves; before, the
  plasma's own appeared only for steps `four` had happened to gather.
- The **Run analyse --diag re_current** button in the case viewer does the
  same, since it runs this command.
- Cached per step: steps that already have a q-profile are not rerun
  (`--force` reruns them). A step whose restart is missing is warned
  about and skipped, as for the other gathers.
- Cost: one more short `jorek2_postproc` call per new step. Not timed.

## 2026-10-12: runaway density in the case viewer

- New section "Runaway density (from the restart files)", above
  "Runaway and thermal current". It reads n_RE of the RE fluid straight
  from the restart file, every toroidal harmonic, so nothing is gathered
  and any step with a restart can be looked at.
- Controls: **step**, toroidal angle **φ**, and **n > 0 part only**.
- Left: n_RE in the poloidal plane at that angle, over the axisymmetric
  psi_N contours (plasma edge darker). The colour scale is the first
  restart's at every step, so lost runaways show as a dimmer map. With
  **n > 0 part only** it shows the non-axisymmetric part on its own scale.
- Right, top: the axisymmetric density on the nodes against psi_N, with
  the first restart in grey. Flattening or loss shows here.
- Right, bottom: the largest amplitude of each toroidal mode relative to
  the n = 0 peak.
- Title line: e c ∫ n_RE dA for this step, its change against the first
  restart in percent, and JOREK's own `Ipre_tot` for the step. A falling
  percentage means runaways have left the domain.
- Units and conventions are JOREK's own: m^-3 as `jorek2_postproc`'s `nre`
  expression, harmonics as cos/sin of n·φ. n_RE is taken as the restart's
  last variable (model600 with the RE fluid); the view refuses a run whose
  in_main sets none of `vpar_re_sign`, `re_initialize`,
  `initial_re_current_fraction`.
- Checked on the local qa3.3_g3.2/eta1e-3_adv0.1 run: the integral gives
  437.19 kA against JOREK's 437.46 kA (0.06% apart: linear between nodes,
  and without B_phi/|B|), and the restart's time equals the zeroD file's.
  That run's local restarts stop before the instability, so the view has
  not yet been seen on a crash.
- Drawn on the grid nodes with linear interpolation between them: good
  for looking, not for measuring. From a cell of your own:
  `viewer.re_density_figure(".", step, phi=90, perturbation=True)`.
  (`diagnostics/re_density.py`, `viewer.re_density_view`)

## 2026-10-12: set a case's `steps` from the case viewer

- New section "Steps to analyse (cases.toml)", after Equilibrium and
  before the gathered views. Choose a **step size** and press **Save to
  cases.toml**: the run's case gets `steps = { start = 0, step = <size> }`,
  which is every restart at a multiple of the size, from 0 to the run's
  last restart. Restarts written later are picked up without saving again.
- Before saving, the section shows the entry as it is now, how many
  restarts the folder has, and the steps the chosen size would give. It
  says so when the last restart is not a multiple of the size (it is then
  left out), or when no restart matches at all.
- The box starts at the size already saved, else at the spacing of the
  restarts.
- The rest of cases.toml stays as written, comments included. Only the
  case's own `steps` value is replaced (a list over several lines too; a
  comment on that value goes with it). A case that took its steps from
  `[defaults]` gets its own `steps` line; a run not in the file is added
  at the end. Per-diag overrides (`[cases.NAME.four]` ...) are not
  changed, and the section lists them.
- Safety: the edited text must parse to the old file with that one value
  changed, or nothing is written and the section says to edit by hand
  (for example a case written as an inline table under `[cases]`).
- The cases.toml used is the nearest one above the run folder, as for the
  **Run analyse** buttons. (`cases_edit.set_case_steps`, `viewer.steps_view`)
- Tested on files made for the tests; not yet tried on your real
  cases.toml or in a live notebook.

## 2026-10-12: choose which currents the case viewer's re_current plot draws

- The "Runaway and thermal current" section has three checkboxes next to
  the step slider: **total**, **runaway**, **thermal**. Unticking one
  removes that curve from both panels (current against time, current
  density against psi_N). All three start ticked.
- With a selection, the y axis of each panel fits the curves still drawn:
  the line at zero no longer holds zero in view, so runaway alone at
  700-790 kA fills the panel instead of sitting at the top of 0-800. With
  all three ticked the range is as before (zero included).
- The thermal / runaway ratio map underneath is not affected.
- From a cell of your own: `viewer.re_current_figure(".", step, show=["re", "thermal"])`.
  `plotting.re_current`'s `draw_current_totals`, `draw_current_density` and
  `re_current_figure` take the same `show`.
- `plot --diag re_current` is unchanged: `profiles/re_current.png` still
  draws all three.

## 2026-10-09: magnetic energies in the case viewer

- New section "Magnetic energies (live data)": the magnetic energy of
  each toroidal harmonic against time, read from the run's
  `macroscopic_vars.dat`, as `plot_live_data.sh -q magnetic_energies`
  draws it. Time in ms and the labels (`E_mag,00`, ...) are the file's
  own; the energies stay normalised, as in plot_live_data.
- **Reload** reads the file again for a run still going. **log scale**
  starts as the file says (on for the energies). With ipympl the figure
  zooms and pans.
- Nothing is gathered and no cases.toml entry is needed: JOREK writes
  this file itself at every step.
- Where a run was restarted from an earlier step, the rows it overwrote
  are dropped, so each time appears once.
- Other quantities of the file from a cell of your own:
  `viewer.energies_view(".", quantity="kinetic_energies")`.
- Read on the local qa3.3_g3.2/eta1e-3_adv0.1 run: 26569 steps from a
  79 MB file in 0.25 s. (`diagnostics/live_data.py`)

## 2026-10-09: `T_method = "parametric"`: a temperature profile from four numbers

- In the shotfile:

  ```python
  T_method = "parametric"
  T_core   = 20.0     # Te + Ti on the axis [eV]
  T_edge   = 2.0      # at the plasma edge, and in the vacuum beyond [eV]
  T_alpha  = 1.0      # optional: larger = flatter core
  T_beta   = 2.0      # optional: how it meets the edge; >= 2 arrives with no gradient
  ```

  T(psi_N) = T_edge + (T_core - T_edge) (1 - psi_N^T_alpha)^T_beta, with
  psi_N the plasma's own (0 on the axis, 1 at the plasma edge).
- The defaults give the parabola (1 - psi_N)^2. `T_alpha = 4` or more
  gives a flat core with the fall near the edge.
- With the density held constant this is the pressure profile's shape
  too: p = n (Te + Ti).
- Converted to JOREK units with `rho_const`, as `"const"` and `"file"`
  are. `T_core = T_edge` is the same as `T_method = "const"`.
- It is a guess function, not a fit to anything.
  (`current_profile.parametric_temperature`)
- **In the case viewer**: a "Temperature profile" section under the
  current-profile one (`viewer.temperature_tuner`). Boxes for `T_core`
  and `T_edge` in eV, sliders for `T_alpha` and `T_beta`; the profile and
  the pressure it gives at `rho_const` are drawn as you change them, over
  the folder's own `T_prof.dat` in grey (turned back into eV, whatever
  made it). **Save to shotfile** writes the four values and
  `T_method = "parametric"`; **Reset to shotfile** and **Regenerate
  inputs** work as in the current-profile section; a yellow band says
  when sliders, shotfile and `T_prof.dat` are out of step.
- For a folder with no saved values the controls start on the axis and
  edge temperatures of its `T_prof.dat`.
- Tested through `prepare_run` on the synthetic campaign, and the grey
  line drawn from the local qa3.3_g3.2 run's CASTOR profile (623 eV on
  axis; close to the default parabola). No JOREK run.

## 2026-10-09: `re_current` ratio map: l_i against time underneath

- The ratio map (`profiles/re_current_ratio.png`, and in the viewer) has
  a panel below it, on the same time axis, with the internal inductance:
  - `l_i(3)`, JOREK's own zeroD `li3`. With an extended boundary it is
    taken over the whole domain, vacuum included.
  - `l_i` of the plasma only, to the real plasma edge, from each step's
    q-profile in the cylinder definition (as the tuner's "JOREK:" line).
    Drawn for the steps that have a q-profile cache
    (`analyse --diag four` gathers them) when F0 is in the namelist.
- The two differ by definition and by region; neither is rescaled. On
  the local qa3.3_g3.2/eta1e-3_adv0.1 run: li3 1.64 falling to 1.60,
  plasma l_i 1.39 falling to 1.36, both at about 0.025 ms.
- Nothing extra is gathered: it reads the zeroD caches `re_current`
  already makes, and q-profiles if they are there.
  (`diagnostics.equilibrium.li_series`, `plotting.re_current.draw_li`)

## 2026-10-09: fixed: two profile variables of one step overwrote each other

- Seen on the HPC with `analyse --diag re_current`: warnings such as
  "output for step 10200 has no ['currdens'] column(s) (got ['Psi_N',
  'recurrdens'])", then a `StopIteration` that ended the whole gather.
- Cause: jorek2_postproc names its output file by step and cut only, not
  by expression. Each call ran in its own folder but its result was then
  copied into one shared folder, so with several workers the two
  variables of a step landed on the same file. A call read the other's
  table, or an empty file caught mid-copy.
- Now each call collects into a folder of its own
  (`profiles.extract_profile`). An empty table is a per-step warning, not
  a crash.
- `analyse --diag profiles` with several `vars` and more than one worker
  had the same fault.
- **Profiles already gathered in parallel**: a mixed-up table was nearly
  always refused, because the column names are checked. A table swapped
  between reading its header and its rows would not have been caught.
  That window is very small, but to be sure, gather again with `--force`:
  `analyse --case X --diag re_current --force` (and `--diag profiles`
  where several vars were gathered at once).

## 2026-10-09: `re_current`: thermal / runaway against time and psi_N

- `plot --diag re_current` also writes `profiles/re_current_ratio.png`,
  and the viewer's section shows it under the profiles: a colour map of
  `|j_thermal / j_RE|` with time across and psi_N (outer midplane) up, in
  the manner of the connection-length map.
- **Red: more runaway current. Green: more thermal. Yellow: equal.** The
  colour scale is logarithmic, from 100 times more runaway to 100 times
  more thermal, and saturates beyond (`plotting.re_current.RATIO_RANGE`).
- **Hatched**: the thermal current runs against the runaway current there
  (the ratio is negative; the colour is its size).
- **Grey**: no ratio. Either that step's profile does not reach that
  psi_N, or the total current density there is below 1/1000 of that
  step's largest, where the ratio is noise (the vacuum of an extended
  boundary). No runaway current at all counts as all thermal (deep green).
- Needs profiles at two steps or more.
  (`diagnostics.re_current.current_ratio_map`)
- Tested on synthetic profiles only, like the profile panel.

## 2026-10-09: `run_jorek -job NAME`

- `run_jorek shotfile.py --run -job 2h` (or `--run_r -job 2h`) queues
  with that jobscript from site.toml's jobscripts folder instead of the
  shotfile's `jobscript`, as `ptrace -job` does. `--job` is the same.
- The shotfile is not changed; the next run without `-job` uses its own
  `jobscript` again.
- Works with `--keep-inputs` and with `--scan ... --apply --run`.
- A name not in the jobscripts folder is refused, listing those there.
  `-job` without `--run` or `--run_r` is refused: the interactive stages
  use no jobscript.
- Tested with the submit command intercepted; not run on the HPC.

## 2026-10-08: `re_current`: runaway and thermal current

- New diagnostic for runs with the RE fluid:
  `analyse --case X --diag re_current`, then
  `plot --case X --diag re_current` (`profiles/re_current.png`), and a
  section in the case viewer with its own gather button.
- **Per radial position**: total, runaway and thermal toroidal current
  density [A/m^2] against psi_N on the outer midplane, from
  jorek2_postproc's `currdens` and `recurrdens` (two calls per step,
  cached as ordinary profiles). Thermal = total - runaway: the current
  JOREK's resistive term acts on, `eta (zj - zj_RE)`.
- **In total**: `Ip_tot` and `Ipre_tot` from the zeroD caches against
  time. JOREK gives these two opposite sign conventions (`Ipre_tot`
  integrates `|Vlight|`), so the runaway current is taken as
  `-sign(vpar_re_sign) * Ipre_tot`, with `vpar_re_sign` read from the
  run's `in_main`. Without it the runaway total is not drawn.
- Steps: the case's `steps`, or `[cases.NAME.re_current]`; points along
  the midplane: `n_points`.
- `recurrdens` exists only in a jorek2_postproc built with the RE fluid;
  `analyse` says so when it gets no such profile.
- Not included: the current enclosed by each flux surface. A midplane cut
  cannot give it; it needs the poloidal plane (jorek2_postproc's
  `rectangle`), which is not built yet.
- Checked: the sign rule against the zeroD files of the local
  qa3.3_g3.2/eta1e-3_adv0.1 run (total 439.83 kA, runaway 437.50 kA,
  thermal 2.33 kA at step 26000). The profile half is tested on synthetic
  caches only: no `recurrdens` profile has been gathered from a real run.
  (`diagnostics/re_current.py`, `plotting/re_current.py`)

## 2026-10-08: `--run_r --keep-inputs`; a guard on `--run_r`; stale shotfile reads

- `run_jorek shotfile.py --run_r --keep-inputs` restarts a run without
  preparing its folder again. Only `tstep_n`, `nstep_n` and `nout` of
  `in_main_r` are set from the shotfile; profiles, boundary, `in_eq`,
  `in_main` and every other field of `in_main_r` (`central_density`,
  `eta`, RE settings, `namelist_options`) stay as the run was started
  with. This is the way to restart a run prepared before the density fix
  (`central_density` = `rho_const`/1e20) and the exact boundary. To change
  anything else for the restart, edit `in_main_r` by hand.
  `--dry-run` shows what it would do. (`runner.prepare_restart`)
- Plain `--run_r` now refuses, writing nothing, when the folder's
  `in_main_r` has a `central_density` other than the one it would write
  (`runner.restart_density_change`): before, it silently re-prepared such
  a folder and the restart went on at another density normalisation. The
  message gives both ways on: `--keep-inputs`, or `run_jorek shotfile.py`
  first to prepare it again on purpose.
- Fixed: a shotfile rewritten within the same second at the same length
  was read back as it was before (Python's bytecode cache; `"3000"` to
  `"7000"`, or the viewer's Save followed at once by Regenerate).
  `load_shotfile` now compiles the text itself, and no longer leaves
  `__pycache__` in run folders.
- Tested on synthetic folders with the submit command intercepted. Not
  run on the HPC.

## 2026-10-08: four view: Psi only; a failing zoom says why

- The four view shows Psi; the variable menu is gone.
  `viewer.four_view(folder, variable="T")` in a cell of your own shows
  another one.
- Zoom and pan did not respond in VS Code on the HPC although the toolbar
  was there. Here the kernel side works (a test zooms with the mouse
  events a front end sends), so the cause is not found. A failure while
  handling the mouse or the toolbar was silent; it is now written in red
  under the figure with the matplotlib and ipympl versions. An ipympl
  older than its matplotlib is the known way to get exactly this.

## 2026-10-08: four view: amplitude at the rational surface; logs that do not repeat

- **amplitude** switch in the four view: "max over radius" (as before) or
  "at q = m/n surface", each mode's |c| on its own rational surface. The
  second is the same number as `four_modes.rational_surface_series`
  (`four_rational` in cases.toml): the largest value over the surfaces
  where that step's q-profile crosses m/n. In that mode a dot on each
  radial curve marks the surface.
- It needs the q-profile cache of each step (`analyse --diag four`
  gathers them). A step without one, or a mode whose m/n the q-profile
  never reaches (1/1 when q0 > 1), has no point; a note says so when
  nothing can be drawn.
- `FourData.rational`, `FourData.surfaces`, `FourPlot.set_rational`,
  `four_figure(..., rational=True)`.
- The messages under the viewer's buttons (Save, Regenerate, Reset, the
  gather buttons) are now an HTML widget whose text is replaced on each
  press (`viewer._Log`), not an Output widget. In VS Code the gather
  message was seen four times; the cause was not reproduced here, the
  widget that could repeat is gone. To confirm in VS Code.

## 2026-10-08: four view: fast controls, zoom and pan

- Every control used to read all of every four cache again (all
  variables, all modes) and draw a new figure: about 5 s per click on the
  local qa3.3_g3.2 run (43 steps). Now the chosen modes of one variable
  are read once (`viewer.load_four_data`, using the new
  `four_cache.read_records`, which opens only the records asked for), and
  a control updates the lines of one figure (`viewer.FourPlot`). Measured
  on the same run: 0.5 s to open, 0.06 to 0.2 s per step change or mode
  tick, 0.1 s to change variable.
- **Zoom and pan** need `ipympl` in the notebook's Python:
  `pip install --user ipympl`, then restart the kernel. The figure then
  has matplotlib's toolbar (pan, zoom to rectangle, home, save) and shows
  the cursor's x, y. Without ipympl the view works as a still image and
  says how to get the toolbar.
- A range you zoomed to is kept when the step or the modes change.
  **Reset view** makes the axes follow the data again.
- The numbers drawn are the same as before (max |c| and |c|(psi_N) from
  the same records).
- Only the four view is changed; the other views still draw a new image
  per change.
- Tested with and without ipympl, and by executing the notebook on the
  demo campaign. The ipympl canvas has not been tried in VS Code, locally
  or over Remote SSH.

## 2026-10-08: four view: modes chosen by qa, a checkbox per mode

- The modes drawn are now `viewer.default_four_modes(qa)`: for n = 1 and
  n = 2, m up to one past the highest rational surface inside the plasma
  (`floor(n qa) + 1`), starting at 1/1 and at 3/2. qa = 2.8 gives
  1/1 2/1 3/1 and 3/2 4/2 5/2 6/2. Modes not in the cache are left out.
- qa is read from the shotfile (`viewer.shotfile_qa`): `current_qa` for
  `ffprime_method = "q_li"`, else `qa`. It is the requested value, not
  what JOREK's equilibrium reached. Without one, the six largest modes
  are offered, as before.
- A row of checkboxes, one per mode (`m/n`), turns modes off. A mode
  keeps its colour when others are unticked. The "modes" slider is gone.
- "log scale" is now two boxes: "log amplitudes" (on) and "log radial
  structure" (off by default). `four_figure` takes `log_radial` and
  `colors`.
- Tested on synthetic caches and drawn on the local qa3.3_g3.2 run.

## 2026-10-08: viewer figures no longer appear twice in VS Code

- Every view drew its figure into an ipywidgets Output widget. VS Code
  shows what an Output widget captures while its cell runs a second time,
  as a plain copy below the widgets.
- Figures now go in as PNG Image widgets (`viewer._show`, `_canvas`), at
  the same on-screen size, 144 dpi. Nothing to change in a notebook: run
  the cell again after updating.
- The duplicate was not reproduced here (plain Jupyter shows one copy
  before and after); the fix removes the mechanism. To confirm in VS Code.

## 2026-10-08: the case notebook is one cell; `bin/case_viewer`

- `case_viewer.ipynb` is now a single cell calling the new
  `viewer.case_viewer(folder)`, which draws every view under its heading.
  Views are listed in `_SECTIONS` in `viewer.py`: a view added there
  shows up in every run folder's notebook, with no change to the file.
- The cell drops any imported `ashen` modules before importing, so after
  `git pull` you run the cell again: no kernel restart.
- New command `python ~/ashen/bin/case_viewer [folders]` (default: the
  current folder; `--existing */*/` for every folder that has a
  notebook). Writes a missing notebook, replaces one with other cells
  (its saved figures and edits are lost), leaves a current one alone.
  Run it once for notebooks written before this change.
- One view failing prints its error in place; the others still draw.
- Tested with synthetic folders and by executing the generated notebook
  on the demo campaign. Not tried in VS Code over Remote SSH.

## 2026-10-07: gather buttons in the viewer

- **Run analyse --diag four** / **--diag profiles** above those views:
  runs `analyse --case <this run>` from the campaign folder
  (`viewer.run_analyse`), prints its output, then redraws. Needs the run
  in `cases.toml`; otherwise it prints the entry to add.
- **Gather q-profile + zeroD for this step** in the equilibrium view and
  **Gather JOREK's q-profile** in the tuner: two `jorek2_postproc` calls
  for one step (`viewer.gather_step_caches`), no `cases.toml` needed. The
  tuner then shows what JOREK achieved.
- A gather runs in the notebook and holds it until done.
- Drawing a view still runs nothing.
- Tested with a real `analyse` subprocess on a folder with no restarts,
  and with stand-ins for the JOREK tools. Not run against real JOREK
  executables.

## 2026-10-07: the tuner shows the folder's own profile, for any ffprime_method

- With an `ffprime_prof.dat` in the run folder, the tuner draws its
  current density, safety factor and FF' as a grey line under the
  sliders' profile, and gives its q0, l_i and qa. For a `"castor"` run
  that is the CASTOR3D profile.
- Without saved q0/l_i/qa, the sliders start on that profile's values.
- `current_profile.profile_from_ffprime`: the cylinder picture (j, q, l_i)
  of any FF' profile. It returns exactly what a family member was made
  from. `viewer.input_profile` reads it from a run folder.
- On `qa3.3_g3.2/eta1e-3_adv0.1` the CASTOR3D profile comes out as
  q0 = 0.95, l_i = 1.26, qa = 2.97, and the family fits it almost exactly
  (alpha = 2.09, nu = 2.25). JOREK's equilibrium from the same FF' has
  0.93, 1.39, 3.23: the first measure of the cylinder estimate's error,
  about +9% in qa and +10% in l_i.

## 2026-10-07: the tuner warns in red when the folder name gives other values

- A run folder named for one profile (`qa2.1_li1.0_q01.0/eta1e-3`) holding
  another gets a red band in the tuner: "Folder name says l_i = 1.0 ...,
  but the profile has l_i = 1.2." (`viewer.folder_name_mismatches`.)
- It reads `qa<x>`, `li<x>`, `q0<x>` from the run folder's name and its
  parent's, compares to the digits written, and follows the sliders.
- The name is only checked. Nothing takes a run's settings from it.
- It does not check `eta`, `g` or anything else in the name.
- A third button, **Reset to shotfile**, puts the sliders back to the
  values saved in `shotfile.py` (read afresh). It writes nothing.

## 2026-10-07: the viewer notebook is written before the shotfile is checked

- `run_jorek shotfile.py` (and `--scan --apply`) now puts
  `case_viewer.ipynb` in the run folder first, so it is there even when
  the shotfile is then refused. Before, a refused shotfile left no
  notebook. `--dry-run` still writes nothing.
- The message for an unreachable q0 says `qa`, not `q_edge`, and reads
  "... allow for this profile family: q0 from 0.757 to 1.480".

## 2026-10-07: `T_method = "file"`

- `T_file` names a two-column file in the run folder: psi_N of the plasma
  (0 to 1) and Te + Ti in eV. Converted to JOREK's units, held at its edge
  value through the vacuum extension, written to `T_prof.dat`.
- A file that does not run from 0 to 1, does not increase, or has T <= 0
  stops `run_jorek` with the reason.
- `T_method` now takes `"castor"`, `"const"` or `"file"`.
- Unverified: no JOREK run with a T profile from a file.

## 2026-10-07: `current_q_edge` renamed `current_qa`

- The shotfile field for the edge q of a `"q_li"` profile is now
  `current_qa`. A shotfile still using `current_q_edge` stops with
  "current_q_edge is now current_qa".
- The viewer's slider, warnings and figure title say `qa`, and so does the
  `requested:` line of a new `j_prof.dat`. An older `j_prof.dat` (saying
  `q_edge`) is still read.
- Earlier entries below use the old name.
- A demo campaign made before this has the old name in its shotfile:
  delete `notebooks/demo_campaign/` and run the demo's first cell again.

## 2026-10-07: a campaign boundary in the template; STARWALL response named after it

- `bnd_method = "template"`, `bnd_file = "boundary1.dat"`: the plasma
  boundary is `template/symlink/boundary/boundary1.dat`, symlinked into
  each run folder. Expanded by `extend_ratio` with `extend_bnd`, like a
  CASTOR3D boundary.
- Such a run's STARWALL response is
  `starwall-response_boundary1_ext1.2.dat`, or `..._noext.dat` without
  `extend_bnd`, both for the lookup and for the archive after `--run_sw`
  (`runner.starwall_response_name`). One response serves every current
  profile on that boundary: the response depends on the domain boundary
  and the wall, not on the plasma.
- `qa` and `g` are not required in such a shotfile; the batch job is
  `<boundary>_<eta>` (`runner.job_name`).
- Other runs keep `starwall-response_qa<qa>_g<g>.dat` and `<g>_<eta>`.
- Found while checking: JOREK verifies only the number of boundary
  elements when reading a response (`vacuum_response.f90`), so a response
  for another boundary shape is accepted silently. That includes the old
  responses made with the 1 cm-rounded boundary.
- The demo campaign uses a campaign boundary.
- Unverified: no `--run_sw` has been run under the new name.

## 2026-10-07: scan files: several runs from one base shotfile

- A scan file (TOML, in the campaign) names a base shotfile, a parent
  folder, a name pattern and the shotfile fields to vary. `run_jorek
  --scan FILE` lists the run folders it would create; `--apply` creates
  and prepares them; a stage flag (`--run`, `--run_eq`, ...) launches each.
- `[vary]` gives every combination; `[[runs]]` gives hand-picked ones.
  `copy` lists files to put in each run folder.
- The campaign is the one the scan file sits in (nearest `site.toml`
  above it), so the command works from anywhere.
- Re-applying acts only on runs it creates: existing runs are not
  prepared or launched again. A folder with a different `shotfile.py` is
  left alone unless `--force`.
- Applying appends the runs and a comparison to `cases.toml` (`cases =
  false` turns that off). It never edits what is there.
- New steps form: `steps = { first_last = true }`, the first and last
  restart of the run, which the added cases use.
- `ashen.shotfile.shotfile_text_with`: the text form of
  `set_shotfile_values`.
- Tried on a made-up campaign (three resistivities). No JOREK job has
  been submitted through it.

## 2026-10-07: a viewer notebook in every run folder

- `run_jorek` now writes `case_viewer.ipynb` into the run folder it
  prepares, if there is none. An existing one is never overwritten
  (not by `run_jorek`, not by the tuner's Regenerate), so its figures and
  edits are kept.
- The notebook shows the folder it is in (`RUN = Path(".")`) and needs no
  editing. If its kernel cannot import ashen it falls back to the `src/`
  of the checkout that wrote it.
- For older run folders: run `run_jorek shotfile.py` there again, or
  `ashen.case_notebook.write_case_notebook(folder)`.
- The notebook is generated by `ashen.case_notebook`, not copied from a
  file. `notebooks/case_viewer.ipynb` is the same notebook, kept in step
  by a test.
- The tuner's three sliders are one per row and wider.
- Checked by executing the notebook headless inside a demo run folder
  with ashen off the path. Not opened in VS Code; not tried on the HPC.

## 2026-10-07: the tuner warns when values, shotfile and inputs disagree

- A yellow band under the tuner's buttons, from `viewer.tuner_status`:
  sliders not saved to `shotfile.py`; shotfile saved but `j_prof.dat` /
  `ffprime_prof.dat` made for other values ("Input files are out of
  date ... Regenerate inputs before running JOREK"); inputs never made;
  shotfile on another `ffprime_method`.
- It updates on every slider move, Save and Regenerate, and is empty when
  all three agree.
- It reads the `requested:` line `run_jorek` writes at the top of
  `j_prof.dat`. It does not check that the equilibrium on screen was run
  from the current inputs.

## 2026-10-07: a demo notebook for the viewer, on made-up files

- `notebooks/case_viewer_demo.ipynb` runs every view of `ashen.viewer`
  with no JOREK and no HPC. Its first cell calls
  `ashen.demo.make_demo_campaign("demo_campaign")`, which writes a small
  campaign (site.toml, template) and one run folder.
- The run's inputs are prepared by the real `prepare_run` from a `"q_li"`
  shotfile, so **Save to shotfile** and **Regenerate inputs** do what they
  do on a real case.
- The "JOREK output" is invented, not physics: an equilibrium that is the
  cylinder model with q0, l_i, q_edge off the request by +0.05, +0.06,
  -0.12, and five modes that grow and saturate.
- The tuner now finds the campaign's `site.toml` by walking up from the
  run folder, not from the notebook's folder.
- `notebooks/demo_campaign/` is gitignored.
- Checked by executing the notebook headless; not opened in VS Code.

## 2026-10-07: a notebook viewer for a case

- `notebooks/case_viewer.ipynb`, backed by `ashen.viewer`. Five views of a
  run folder: the q0/l_i/q_edge profile tuner, boundaries, equilibrium,
  Fourier modes, radial profiles. Needs `ipywidgets` and `h5py`
  (`pip install --user ipywidgets ipykernel h5py`; extra `viewer` in
  `pyproject.toml`).
- The tuner's **Save to shotfile** writes the three values and
  `ffprime_method = "q_li"`; **Regenerate inputs** prepares the run folder
  from the shotfile, as `run_jorek shotfile.py` does, and submits nothing.
- It shows what JOREK's equilibrium achieved beside what was asked, when
  the step's q-profile and zeroD caches exist.
- Each view is also a function returning a matplotlib Figure.
- Tried on `qa3.3_g3.2/eta1e-3_adv0.1`: all five views render. The model
  given that run's own q0 = 0.93, l_i = 1.39, q_edge = 3.23 lies on
  JOREK's q-profile, with Ip = 424 kA against JOREK's 438 kA.
- Unverified: not opened in VS Code by hand, and not run over Remote SSH
  on the HPC. The notebook was executed headless here.

## 2026-10-07: current profile from q0, l_i and q_edge

- `ffprime_method = "q_li"` with `current_q0`, `current_li`,
  `current_q_edge` in the shotfile. `run_jorek` builds
  `j = j0 (1 - rho^alpha)^nu`, writes it to `j_prof.dat` and converts it
  to `ffprime_prof.dat`. The solver is the one from
  `q0_li_playground.ipynb`, now in `ashen.current_profile`.
- Size from q0 on the axis of an ellipse; `R0`, `a`, `kappa` from the
  plasma boundary, `F0` from `namelist_options` or the template's `in_eq`.
- A q0 the family cannot reach at that l_i and q_edge stops `run_jorek`
  with the reachable window.
- `current_R0` is now optional (centre of the plasma boundary).
- `ashen.diagnostics.equilibrium.achieved_q_li`: q0, plasma-edge q and l_i
  of a JOREK equilibrium, from its q-profile and zeroD caches. On
  `qa3.3_g3.2/eta1e-3_adv0.1`: q0 = 0.93, edge q = 3.23, l_i = 1.39
  (zeroD `li3` = 1.64 and `q95` = 4.46 include the vacuum region).
- `ashen.shotfile.set_shotfile_values` rewrites plain `name = value` lines
  in a shotfile and refuses computed ones.
- In `prepare_run` the boundary is now worked out before ffprime. Output
  for existing methods is unchanged.
- Unverified: no JOREK equilibrium has been run from a `"q_li"` profile,
  so how far JOREK lands from the three targets is not yet known.

## 2026-10-07: `four_delta_b_at = "edge"`: delta_b at the domain edge

- A case key. `"max"` (default) keeps the largest `delta_b` over the
  radial grid. `"edge"` takes the value at the outermost radial point, the
  edge of JOREK's domain: with `extend_bnd`, in the vacuum outside the
  plasma, as a probe there would see it.
- Figures: `four_dir/delta_b_edge_modes_step.png` and `_time.png`, labelled
  with the radius. Also used by `delta_b_over_b`, `--compare ... --diag
  four` and the `delta_b_*` scan-map quantities for that case.
- On `qa3.3_g3.2/eta1e-3_adv0.1`, step 26000, at r = 0.447 m: 2/1 is
  0.35 mT (8.2 mT at its peak inside), 3/1 is 0.80 mT.
- The radius cannot be chosen. A fall-off law to reach further out was
  tried and dropped: the fitted exponents in the run's vacuum region were
  -3.8, -3.2 and -1.6 for m = 2, 3, 4, against -2, -3, -4 expected.
- Caveat: the modes are straight-field-line harmonics. At the domain edge
  one perturbation spreads over several `m` (here n = 1 is largest in
  m = 3 to 7), which is not what a probe array would resolve.
- Unverified: not compared with `jorek2_fields_xyz`, which is not built.

## 2026-10-07: `delta_b` divides by the surface's minor radius

- `delta_b` was `(m / R_axis^2) |Psi_mn|`. The field of a flux
  perturbation is `m |Psi_mn| / (R r)`, with `r` the minor radius of the
  surface, so the old value was too small by about `R_axis / r`.
- Now `delta_b = |m| |Psi_mn| / (R_axis r)`, with `r` from the toroidal
  flux inside each surface (`pi r^2 B0 = 2 pi int q dpsi`). The largest
  value over the radial grid is kept. `delta_b_over_b`, the rational-surface
  values, the `--compare ... --diag four` figures and the `delta_b_*` and
  `energy_32_over_21_*` scan-map quantities all use it.
- It needs each step's q-profile and zeroD caches and `F0` from the
  namelist. `plot` gathers a missing q-profile or zeroD on demand.
- On `qa3.3_g3.2/eta1e-3_adv0.1` (43 steps): the 2/1 peak goes from 1.2 mT
  to 8.2 mT, 3/1 from 0.76 to 3.4 mT. `r` at the plasma edge comes out
  0.3737 m; the CASTOR3D boundary's `sqrt(area / pi)` is 0.3733 m.
- `m = 1` now peaks at the axis (2.0 mT, was 0.06 mT): `Psi` goes as `r`
  there, so `Psi / r` stays finite.
- **Every delta_b figure and number made before this is too small** by a
  factor that differs per mode. Replot to update; nothing needs
  re-gathering beyond the q-profile and zeroD.
- `energy_32_over_21` changes too: each mode is now weighted by the radius
  of the surface it peaks on.
- Tests: most plot and quantity tests pin `r = r_axis` (an autouse fixture)
  to keep their arithmetic; `test_four_modes.py` and the tests marked
  `real_minor_radius` use the real `r`.
## 2026-10-07: current profile and boundary from files, without CASTOR3D

- `ffprime_method = "current"` reads `current_file` (x, j in A/m^2) and
  writes JOREK's `FFprime = -mu_0 * current_R0 * j`. `current_coord` is
  `"psi_n"` or `"rho"` (r/a, mapped as a circular cylinder).
- `T_method = "const"` with `T_const` (Te + Ti in eV) gives a flat T.
- `bnd_method = "file"` no longer needs a CASTOR3D psi. `bnd_file_is_plasma
  = True` expands the file's boundary by `extend_ratio` and resamples it to
  50 points. `psi_bnd` sets the boundary psi (default 0).
- Without CASTOR3D the plasma edge is at `1/extend_ratio` of the extended
  psi_N, and `FFprime` is 0 beyond it.
- Runs that use CASTOR3D are prepared exactly as before.
- The README section "Profiles and boundary without CASTOR3D" has the
  conventions and the limits (j at `R = current_R0`; pressure-gradient
  current left out).
- Unverified: no JOREK equilibrium has been run from these files yet. The
  sign and size were checked against JOREK's source, the wiki's
  normalization page and an existing run's `ffprime_prof.dat`.

## 2026-10-07: `rho_prof.dat` now gives the density the shotfile asks for

- `run_jorek` wrote `rho_const / 1e20` into both `central_density` and
  `rho_prof.dat`. JOREK's rho is in units of `central_density * 1e20`, so
  the density was normalised twice: `rho_const = 1e18` ran at 1e16 m^-3,
  with 1% of the CASTOR3D pressure.
- `rho_prof.dat` now holds 1. `central_density` and `T_prof.dat` are
  unchanged. Density is `rho_const` and pressure is CASTOR3D's.
- Seen in `qa3.3_g3.2/eta1e-3_adv0.1`: rho = 0.01 in the restarts, and
  `Part_tot / volume` = 1.000e16 in the zeroD output.
- **Runs prepared before this are 1e16-type plasmas** (for `rho_const =
  1e18`): Alfven speed 10 times the intended one. Their time axis in
  seconds is right for the plasma that was simulated.
- A new run differs from an old one with the same shotfile. To reproduce
  an old run, edit its `rho_prof.dat` back to `rho_const / 1e20`.
- The golden `rho_prof.dat` was rewritten by hand to 1 (the golden test
  needs the real CASTOR3D tree, which this machine lacks).
- Unverified: no JOREK run with the corrected file yet.

## 2026-10-07: boundary R and Z written to 6 decimals, not 2

- `run_jorek` wrote the boundary's R and Z with `.2f`, rounding each of
  the 50 points to 1 cm. In `qa3.3_g3.2/eta1e-3_adv0.1` that put the
  points up to 6 mm off the smooth boundary, with jumps of up to 10 mm
  between neighbours and 8 concave kinks. JOREK's grid boundary
  (`boundary.txt`) carried the same wobble.
- R and Z now go to `in_eq`, `in_main`, `in_main_r` and `in_bnd` with 6
  decimals. `psi_boundary` is written as before.
- A new run's domain boundary differs from an old one's by up to 5 mm per
  point. Existing run folders are untouched until `run_jorek` is run in
  them again.
- The golden `in_bnd` was regenerated from the golden `original_bnd.dat`.
  Rounded to 2 decimals it reproduces the old file byte for byte. The
  golden `in_eq` is still the legacy capture; its test now accepts R and Z
  that round to the legacy values.
- Unverified: no JOREK grid built from the new boundary yet; the golden
  tests need the real CASTOR3D tree.

## 2026-10-06: repacking a trace's diagnostics faster

Measured on a 10000-marker file in JOREK's layout (3.3 GB on disk, 660 MB
of data): 4.3 s, down from 10.1 s. The repacked file is the same size and
holds the same values.

- **The original is read once, not twice.** Each block of values is hashed
  as it is copied. The finished copy is then read back through HDF5 and
  its hashes compared. A copy that differs is still thrown away and the
  original kept. Before, the check read the original a second time,
  padding and all.
- **Compression runs on up to 8 cores.** Each chunk is shuffled and
  gzipped on a thread and handed to HDF5 ready-made. The file is still a
  standard gzip HDF5 file, read as before.
- Nothing to set: `ptrace` and `util --func compress_traces` use it.
- Unverified: not yet run on the HPC, where the read of the original is
  likely the larger share of the time.

## 2026-10-06: particle plots faster and leaner

These were measured on a 2000-marker, 12000-row trace with 60 snapshots
and a 256-point wall. Every figure, GIF and JSON is byte-identical to
before.

- `particles`, `particle_exits`, `particle_loss` and `particle_wetted`
  together take 5.0 s, down from 18.6 s. With `--animate` they take 7.9 s,
  down from 17.4 s. Peak memory is 2.1 GB, down from 3.1 GB.
- **The inside-the-wall test is about 10x faster.** It looks up each
  point's band of Z between the wall's vertex heights and tests only that
  band's edges. It runs in chunks, so its memory stays bounded. The
  arithmetic is unchanged and is tested against the old version.
- **One read of `ptrace_diag.h5` per case.** The particle diags drawn in
  one `plot` run share the read; before, each diag read the file again.
  The shared arrays are read-only. A file that changes on disk is read
  again.
- **Less memory per read.** The float32 data now becomes float64 in one
  step instead of two full copies.
- **`particle_exits` without theta in the file** computes it only at each
  particle's exit row, not at every row.
- **Not changed:** GIF writing (matplotlib drawing, Pillow's palette per
  frame) is most of what is left under `--animate`. Making it faster would
  change the images.
- Unverified: not yet run on the HPC's real traces.

## 2026-10-06: `particle_wetted` faster, cached, own counts range

- The test of which points are inside the wall now sorts the points by Z
  once and gives each edge only the points that can cross it. Before, it
  checked every (row, particle) point against every edge. The answer is
  bit-for-bit the same (tested against the old code). This speeds up
  `particle_exits`, `particle_loss` and the `particles` exits too. On a
  1000-marker, 12000-row trace with a 256-point wall, `particle_wetted`
  went from 7.3 s to 2.7 s, with identical `particle_wetted.json`.
- The first wall hit of each particle is cached in
  `particle_wetted_cache.npz` in the ptrace folder. A replot of an
  unchanged trace takes 0.4 s, whether it changes the bins, either colour
  range, counts or `ptrace_initial_psi_n_range`. A changed diagnostics file
  (name, size or mtime), `original_bnd.dat` or start..end window makes a
  fresh cache. Delete the cache to force one.
- `ptrace_wetted_count_range = [min, max]` sets the colour scale of the
  counts map, in particles per cell. `ptrace_wetted_density_range` now
  sets only the density map.
- Unverified: not yet run on the HPC's real traces.

## 2026-10-05: trace diagnostics repacked to their data's size

JOREK's diagnostics writer chunks every dataset 50000 particles x 1 time,
uncompressed, so each row takes the room of 50000 particles: about 34 GB
for a 1000-marker, 120 us trace at `diag_step = 1e-8`, ~50 times its data.

- `ptrace` now repacks the file losslessly (fitted chunks, gzip) as soon as
  the program ends, after `--run_i` or when a queued job is concluded. It
  still reaches full size while the trace runs.
- `util --func compress_traces` repacks the files of earlier traces. It says
  what it would free until given `--apply`.
- Every value is compared with the original before it is replaced.

## 2026-10-05: `util --func delete_figures`

- Deletes each run's figures (`.png`, `.gif`) where `plot` writes them,
  keeping every cache and trace output. `plot` draws them again.
- Comparison figures in the campaign's `figures/` are not touched.
- Shows what it would delete and the space it frees until given `--apply`.

## 2026-10-05: `particle_wetted` in counts

- `--wetted-counts`, or `ptrace_wetted_counts = true`, draws how many
  particles hit each cell and each bin instead of fractions, as
  `particle_wetted_counts.png` beside the fraction figure.

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
