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
