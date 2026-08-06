# Known issues

Physics-affecting behaviour that Ashen has **found but deliberately not
changed**, because fixing it would alter numerical output for runs that may
already be published. Code-quality bugs that get fixed opportunistically
during the refactor (string-truthiness, `--replace` inversion, etc.) are not
here -- they're tracked in the refactor plan's "Behaviour changes to confirm"
list. This file is only for things that need George's physics judgement
before anyone touches them.

Each entry stays until it is either fixed (own commit, own golden-file update)
or confirmed intentional and closed out.

---

## 1. T profile: computed on the wrong grid when the boundary is extended

**Where:** `ashen/src/ashen/profiles.py`, `get_t_profile_from_castor` (ported
from `castor3d/util/run_jorek_util.py:257 get_T_prof_from_castor`)

**What happens:** the function takes a `psi_n` grid as an argument but
immediately discards it and recomputes its own from
`xn_fpol_stor0_<suffix>` on disk:

```python
def get_T_prof_from_castor(psi_n, rho_prof, cotrans_dir, sigma=5):
  ...
  psi = np.abs(load_two_col_data(f"{cotrans_dir}/xn_fpol_stor0_DIIID")[:,1])
  psi_edge = psi[-1]
  psi_n = psi/psi_edge   # <- the caller's psi_n is thrown away here
```

**Why it matters:** when a shotfile sets `extend_bnd=True`, `run_jorek.py`
builds an extended psi grid (`boundary.extend_psi`) and resamples ffprime onto
it. T should be resampled onto that same extended grid to describe the same
plasma -- but because this function silently recomputes its own unextended
grid, **T and ffprime end up defined on different domains** for every
extended-boundary run.

**Status:** confirmed a real bug by George, 2026-08-04 ("a bit of a bug").
**Not yet fixed** -- fixing it changes numerical output for every
extended-boundary run already published, so it needs to land as its own
isolated commit with the golden reference deliberately updated, not bundled
into the rest of Phase 3. `profiles.py` currently replicates the bug exactly.

---

## 2. T profile: output does not depend on density at all

**Where:** same function as #1.

**What happens:** the last two lines are

```python
T_eV    = pres / (rho_prof * e)
T_jorek = T_eV * (rho_prof * e * u0)
```

Substituting the first into the second: `T_jorek = pres * u0`. `rho_prof` and
`e` appear in the same combination in both numerator and denominator and
cancel out completely, algebraically, for any value of `rho_prof`.

**Verified, not just derived:** run against the real `qa2.1_g2.300` fixture
with `rho_prof` = 0.01, 1.0, 1,000,000, and a full 1000-point non-constant
array -- `T_jorek` was bit-for-bit identical in every case. Pinned as a
regression test:
`ashen/tests/unit/test_profiles.py::test_t_profile_output_is_independent_of_rho`.

**Consequence:** whatever `rho_const` a shotfile sets has never affected the T
profile JOREK actually receives.

**Status:** open, unconfirmed whether bug or intentional. Two readings:

- **Bug** -- an earlier version of the formula may have used density
  meaningfully, and it was refactored into this cancelling form by accident.
- **Intentional** -- JOREK's `T_file` may expect a normalised quantity
  (effectively `p * mu_0`) rather than a literal temperature, in which case
  the cancellation is by design and `rho_prof` is a vestigial parameter.

Settling this needs checking what `T_file` is physically expected to contain
in `Columbia/jorek_RE/models/model600/initialise_parameters.f90`, not
something inferable from this wrapper alone. `profiles.py` preserves the
formula exactly pending an answer.

---

## 3. ffprime: half-mesh/full-mesh alignment, unconfirmed

**Where:** `ashen/src/ashen/profiles.py`, `get_ffprime_profile_from_castor`
(ported from `run_jorek_util.py:238 get_ffprime_prof_from_castor`)

**What happens:** `jpol = load_two_col_data(f"{cotrans_dir}/xn_hjpol_stor0_{suffix}")[:, 1][1:]`
drops the first point of the `hjpol` file before pairing it against `psi`
(from `xn_fpol_stor0_{suffix}`).

**Status:** lower priority than #1/#2, included for completeness. An earlier
pass flagged this as a length-mismatch crash; that was checked against the
real fixtures and is **false** -- both arrays are length 1000 after the
slice, matching by construction. The `xn_h*` (half-mesh) vs `xn_f*`
(full-mesh) naming is the standard VMEC/NEMEC convention, where the half-mesh
grid has exactly one fewer point than the full-mesh grid -- which plausibly
explains the `[1:]` slice as intentional mesh alignment, not a bug. Whether
the alignment direction is physically correct (`[1:]` vs. some other offset)
is still open but not urgent. `profiles.py` preserves it as-is.

---

## 4. Diagnostics plotting layer not yet ported (Phase 4 scope note, not a bug)

**Where:** `castor3d/util/data_jorek.py`'s plotting functions -- `plot_poincare`,
`plot_field_line_diffusion`, `plot_connection_length`, `get_island_width`,
`postproc_get_q` / `plot_postproc_profs`'s plotting half -- and
`analysis.py`'s `max_fieldline_pos` diag, which calls `plot_max_fieldline_pos`,
a function **that does not exist anywhere in the tree** and would raise
`NameError` if selected.

**Status:** Phase 4 (`ashen/jorek2.py`, `ashen/diagnostics/{poincare,profiles}.py`,
`ashen/cases.py`, `ashen/cli/analyse.py` + `bin/analyse`) ports the *data
gathering* these functions consume -- zeroD caching, Poincare tracing, radial
profile extraction, all writing the same `.npz` files the legacy plotting code
already reads -- but not the matplotlib code itself. That code carries its own
bugs independent of anything above (`R0 = 1.36` hardcoded in four places in
`data_jorek.py` instead of using the log-extracted value `postproc_get_q`
already computes correctly; `get_island_width` defined twice with identical
bodies where the second silently wins, breaking any caller passing `mode=`)
and needs its own confirmation pass before porting, not a silent carry-over.
Legacy `analysis.py` remains usable for plotting against Ashen-gathered data
in the meantime.

---

## 5. Legacy plotting can no longer read the Poincaré cache

**Where:** `ashen/src/ashen/diagnostics/poincare_cache.py` (new format) vs
`castor3d/util/data_jorek.py:254`, `:354`, `:447` (`plot_field_line_diffusion`,
`plot_poincare`, `get_island_width`), which load
`poinc_t<step>_{psi_n,theta,R,Z}.npz`.

**What changed:** the Poincaré cache moved from four dense `.npz` files per
step, holding `(n_psi, ang_sample_freq)` pickled object arrays, to one HDF5
file per step holding one group per field line keyed by its starting point.
This is what makes a scan extendable — the dense shape is *why* adding a
`psi_n` or raising `n_turns` used to force a full retrace, since both change
the array's shape. It cannot be expressed in the old format.

**Consequence, accepted deliberately (George, 2026-08-05):** legacy
`analysis.py` plotting works against caches produced *before* this change but
not against new ones. `read_legacy_cache` reads the old files so nothing
already computed is lost, but those records carry no start position (the old
format never stored one), so they are marked unextendable: the first time such
a step is extended it is retraced from scratch, once.

Unlike #1 and #2 this is not a physics question and needs no confirmation — it
is here because it is a **capability regression with a date attached**, and it
closes when the plotting layer in #4 is ported. Profile caches are unaffected;
they keep the `.npz` format.

---

## 6. Connection-length `R0` now comes from the log, not a hardcoded constant

**Where:** `ashen/src/ashen/diagnostics/connection_length.py` vs
`castor3d/util/data_jorek.py:497,531` (`R0 = 1.36`, used) and `:409,452`
(same constant, dead), and `gather_profiles.py:140` (`R0=1.369`, a different
value, also dead — never referenced in that function's body).

**What changed:** `line_connection_length` now takes `R0` from
`ashen.logfile.r_axis(paths.log)`, which extracts the JOREK-computed
`R_axis` the same way the one already-correct legacy call site did
(`postproc_get_q`, `gather_profiles.py:130-138`). This is a **numeric**
change: every connection length is `L = n_inside * 2*pi*R0`, so the result
scales linearly with whatever `R0` was wrong by. For `qa2.1_g2.3/eta1e-3_RE`
the log-extracted value is ~1.363, vs the hardcoded 1.36 — under 0.3%, but it
is a real change to published-shape numbers, not a no-op.

**Decision (George, 2026-08-05):** extract from the log and fail loudly
(`LogfileError`) rather than falling back to the old constant — a
missing/unreadable log should stop the calculation, not silently reintroduce
the guessed value. `ashen/cli/plot.py` catches `LogfileError` per case and
reports it rather than crashing the whole run.

---

## 7. Connection-length `psi_n` may be double-normalised by `real_psi_edge`

**Where:** `ashen/src/ashen/diagnostics/connection_length.py:line_connection_length`
vs `castor3d/util/data_jorek.py:546` (`psi_n_out[n_t][i] =
np.array(psi_n_out[n_t][i])/psi_edge`).

**What's suspicious:** `jorek2_poincare`'s own output is already normalised
to the boundary — the file header in `poinc_rho-theta.dat` states
`psi_n=(psi - psi_axis)/(psi_bnd - psi_axis)` — so `record.psi_n` (derived
from that file) should already be in `[0, ~1+]` before any division. Legacy
`plot_connection_length` divides by `real_psi_edge` a second time anyway. If
`real_psi_edge` (a CASTOR3D-derived quantity written to `real_psi_edge.dat`
by the runner) is not numerically equal to JOREK's own internal `psi_bnd`,
this second division shifts the `psi_n < 1` inside/outside threshold and
therefore every connection length.

**What `real_psi_edge` actually is (settled).** `boundary.py::extend_psi`
defines it as `psi[-1] / new_psi[-1]` -- the true plasma edge as a fraction
of the *extended* (vacuum-padded) grid's edge, so it is in `(0, 1]` and is
`1.0` exactly when `extend_bnd` is off. It converts a **plasma-fraction**
`psi_n` into **JOREK-grid** `psi_n`. It is *not* related to JOREK's
`psi_bnd`: JOREK's own `get_psi_n` already normalises to `psi_bnd`, so
anything JOREK reports or accepts is in JOREK-grid `psi_n` with no further
conversion needed. Confirmed from `exec_commands.f90:3063-3068`, where
`fluxsurface` rejects an argument outside `[0, 1]` and expands it as
`psi_axis + psi_n*(psi_bnd - psi_axis)` -- the exact inverse of the
`get_psi_n` that `qprofile` writes.

This means the rule is: apply `real_psi_edge` **exactly once**, when turning
a user-facing `case.psi_n_in` into a JOREK-grid position (which
`cli/analyse.py` does before tracing). Anything read back out of JOREK or a
cache keyed on those positions is already JOREK-grid and must not be divided
again. The Poincare rational-surface highlight had exactly that second
division and was fixed accordingly (`cli/plot.py::
_rational_highlight_for_step`), with a regression test pinning
`real_psi_edge != 1`.

**Status:** the *semantics* above are settled; the connection-length
behaviour is still **not resolved**, and `ashen` still preserves the legacy
second division in `line_connection_length`. By the rule above that division
is very likely wrong -- `record.psi_n` derives from `jorek2_poincare`'s own
`poinc_rho-theta.dat`, already JOREK-normalised. But dropping it moves the
`psi_n < 1` confined/lost threshold and therefore *every* connection length
ever plotted, so it stays George's call rather than a silent change. Nothing
blocks it technically now that the units question is answered.

---

## 8. Plotting layer: scope of this pass, and what's still legacy-only

**Status:** Phase 4c ports two of the legacy plots against the Phase 4b
per-line cache: Poincare puncture plots (`ashen/plotting/poincare.py`, ports
`data_jorek.py:354 plot_poincare`) and the LC/LCTT connection-length colour
maps (`ashen/plotting/connection_length.py`, ports
`data_jorek.py:597 color_con_length_plot`), driven by a new `bin/plot` /
`ashen/cli/plot.py` that reads the same `cases.toml` as `analyse`.

**2026-08-06 addition:** a further pass ports `plot_theta_histogram_matrix`
(`Columbia/NL_kinks/prod_plots_draft0.ipynb`, cell 5 -- a notebook function,
not `data_jorek.py`; not previously tracked here) as
`ashen/diagnostics/theta_histogram.py` + `ashen/plotting/theta_histogram.py`,
`--diag theta_hist`. Its `i_lim` (a positional index into scan order) is
replaced by `theta_psi_n_range`, a `psi_n_in`-based filter -- consistent with
why the Phase 4b cache is keyed by starting position rather than index in the
first place. The notebook's `show_threshold`/`threshold_percentile` line
overlay and `counts_compare` companion are dropped, not ported (George,
2026-08-06). Alongside it, `ashen/comparisons.py` adds a `[comparisons.*]`
section to `cases.toml` and `--compare`/`--list-comparisons` to `bin/plot` --
grouping already-defined cases into one cross-run figure (e.g. an eta scan),
which nothing in `ashen` had a mechanism for before this pass. `theta_hist` is
the only diag with a comparison renderer so far; the mechanism itself is
general.

**Same-day follow-on:** `wetted_fraction` (`--diag wetted_fraction`,
comparison-only) ports the core of the notebook's `eta_plot` plus its
`wetted_A/total_bins` computation (same notebook, cell 0 and cell 8): the
fraction of a case's `theta_hist` bins whose count exceeds a threshold
(`ashen.diagnostics.theta_histogram.wetted_fraction`), plotted against an
explicit `x_values` on a `[comparisons.*]` entry
(`ashen.plotting.wetted_fraction`). George flagged this "scalar vs. scan
parameter" shape as one he'll reuse often -- `x_values` is deliberately not
inferred from run-folder names (same directory-name-parsing hazard `CLAUDE.md`
already warns about for CASTOR3D), and `ashen.plotting.wetted_fraction` is
written generic over the y-quantity, not specific to wetted fraction, so the
next such request can reuse it.

**Deliberately not ported this pass** (recorded here as future work, per
George, 2026-08-05):
- `plot_field_line_diffusion` (`data_jorek.py:180`) — the scatter-by-time
  diffusion-extent plot. Its inline island-width computation duplicates
  `get_island_width` exactly (see #4's note on that duplication).
- `connection_length_line_plot` (`data_jorek.py:570`, the `L2_`/`L2TT_` line
  plots) — the *other* half of `plot_connection_length`; only the colour-map
  half (LC/LCTT) was in scope this pass.
- `plot_macro_var` (`data_jorek.py:137`) — macroscopic-variable time series
  from `macroscopic_vars.dat`. Leaks figures (no `plt.close()`).
- `postproc_get_q` (`gather_profiles.py:130`) — the derived q-profile
  computed from `Btor`/`Btheta`/`r_minor` (as opposed to `qprofile`'s own
  q output, already available via `ashen.diagnostics.qprofile`) and the
  `dJ/dr` at the q=2 surface it feeds, which exists only as unsaved scatter
  points inside `gather_profiles.py:163-175`. **The plain radial profile
  plots themselves (`plot_postproc_profs`'s main loop) are ported** -- see
  #9 -- this bullet is only the q-derivation/`dJ/dr` half.
- `calc_stochast_factor` (`data_jorek.py:702`) — computed, never plotted;
  the `"plot_stochastic_factor"` diag name in legacy `analysis.py:171`
  is a misnomer that actually calls `plot_connection_length`.
- `plot_max_fieldline_pos` — still does not exist anywhere in the legacy
  tree (see #4); implementing it is new work, not a port.
- The CASTOR3D-scan plots in `castor3d/util/data.py` (`create_li_gr_plot`
  and friends) are a separate lineage entirely, out of scope for the JOREK
  wrapper migration.

---

## 9. `jorek2_postproc`'s `average` command dies once flux surfaces are destroyed

**Where:** vendored JOREK, not ashen -- `Columbia/jorek_RE/diagnostics/
postproc/exec_commands.f90:1747` (`average`) via `new_diag/mod_position.f90:167`
(`pol_pos`) via `new_diag/mod_straight_field_line.f90:342` (`trace_fieldlines`).
Recorded here, not fixed, because the fix is in code this tree does not edit
(see `CLAUDE.md`: `Columbia/jorek_RE` is vendored upstream).

**What happens:** `average` computes a genuine flux-surface average by real
field-line tracing (unlike `qprofile`/`fluxsurface`/`find_q_surface`, which use
2D contour finding and have none of this failure mode). Once a run's magnetic
axis approaches the grid boundary closely enough,
`is_LCFS_lost` (`models/equil_info.f90:1265`) sets `ES%LCFS_is_lost = .true.`,
and `get_psi_n` (`equil_info.f90:982`) then returns the **constant** `1.01`
for every point in the domain:

```fortran
if (ES%LCFS_is_lost .or. abs(ES%psi_bnd - ES%psi_axis) < 1d-6 ) then
  get_psi_n = 1.01d0
  return
endif
```

The tracer's starting-point bisection therefore converges to the magnetic axis
for every requested surface, no field line ever completes a poloidal turn, and
the routine hits a **hard Fortran `stop`**
(`mod_straight_field_line.f90:518`) rather than an error return --
`jorek2_postproc` exits the whole process, mid-`for`-loop, rather than
skipping the one bad step. The same `stop` is also reachable earlier and more
generally whenever a traced line simply leaves the mesh (e.g. near a
stochastic/ergodic edge, well before `LCFS_is_lost` itself trips).

**Consequence:** `average` cannot produce a flux-surface-averaged profile
(e.g. `<currdens>(Psi_N)`) past the point in a nonlinear run where the
original flux surfaces stop being nested and closed -- which, for the kink/
tearing-mode runs this tree is built around, is often the timestep of actual
physics interest.

**What ashen does about it:** `ashen.diagnostics.profiles.gather_profiles`
catches the resulting `Jorek2Error` **per `(step, var, tor_mode)` task**, warns,
and continues -- one failing `average` step no longer aborts gathering every
other step, variable, or mode. `Case.profile_rad_range` (default `[0.001,
0.999]`, matching `jorek2_postproc`'s own default) is the main lever for
avoiding the failure in the first place: pulling the upper bound in off the
separatrix keeps tracing inside surfaces that are still actually nested. The
companion `midplane`/`midplane outer`/`midplane inner` commands do no tracing
at all (`Psi_N` is an ordinary pointwise expression, `mod_expression.f90:120`,
evaluated the same way `currdens` is) and always produce *something*, even
once `average` cannot -- gathering both under one case's `tor_mode` list and
plotting them side by side (`ashen.plotting.profiles.plot_profile_comparison`)
makes "where does the flux average stop being computable" a visible result
rather than a crashed run.

**Not fixed, and cannot be from here:** `pol_pos` (`mod_position.f90:167`)
also declares its own `ierr` as a local variable it never propagates to its
caller, and `average` never checks `ierr` after `eval_expr`
(`exec_commands.f90:1790`) -- so some failure modes end in an unallocated-array
write rather than even reaching the `stop` above. Both are vendored-source
bugs.

---

## 10. `Jgrad`/`q` gathering (`r_minor`) not confirmed to work on every model/tor_mode

**Status:** open, 2026-08 (George's report).

`Case.vars`'s `"Jgrad"`/`"q"` compound entries expand
(`ashen.diagnostics.profiles.expand_compound_vars`, ported unchanged from
`castor3d/util/diagnostics/gather_profiles.py:94-106`) into `currdens`/
`r_minor`/`Btheta`/`Btor` (`Jgrad`) or `r_minor`/`Btheta`/`Btor` (`q`) --
requested from `jorek2_postproc` the same way any other variable is. In
practice, gathering `r_minor` against a `"midplane outer"` cut on a
`model600`-family run failed: `jorek2_postproc`'s output for that
`(step, tor_mode)` simply had no `r_minor` column. Whether that's because
`r_minor` needs `tor_mode = "average"` instead (a flux-surface label, which
only a real flux-surface trace can define -- unlike `Btor`/`Btheta`/
`currdens`, ordinary pointwise expressions any midplane cut can evaluate),
or isn't a valid `jorek2_postproc` expression for this model at all, wasn't
established -- the vendored source this session could read didn't turn up
either an `r_minor` definition or the `mod_expression.f90` file #9 cites for
the general expression registry, so neither could be confirmed against the
actual executable in play.

**What ashen does about it:** `extract_profile` used to let a missing
column surface as a raw `ValueError` (`headers.index(var)`), which
`gather_profiles`'s per-task handling doesn't catch (only
`MissingRestartError`/`Jorek2Error`), so one bad `(step, var, tor_mode)`
combination crashed the entire `analyse --diag profiles` run -- including
every other variable/step/mode in the same batch. It now raises
`Jorek2Error` instead, naming the missing column(s) and the tor_mode/step,
which the existing per-task warn-and-continue handling (see #9) already
catches -- a broken `r_minor` request now costs only that one task, same as
an `average` tracing failure does.

**Not fixed:** whether `Jgrad`/`q` gathering is usable at all, and if so
under which `tor_mode`, needs confirming against a real run before
recommending it -- `postproc_get_q`, the actual q/`dJ/dr` consumer of these
components, was never ported to `ashen` in the first place (#8), so this
path had no prior end-to-end exercise.
