# Golden reference: qa2.1_g2.3/eta1e-3_RE

Reference output from running the **unmodified legacy** `Columbia/run_jorek.py`
against the real `qa2.1_g2.3/eta1e-3_RE/shotfile.py`. `test_prepare_run.py`
compares Ashen's `prepare_run` output against these files.

## How this was captured

On the HPC, in a **fresh empty directory** (not the live run folder):

```bash
mkdir -p /tokp/work/geosu/Columbia/NL_kinks/qa2.1_g2.3/_golden_capture
cp /tokp/work/geosu/Columbia/NL_kinks/qa2.1_g2.3/eta1e-3_RE/shotfile.py \
   /tokp/work/geosu/Columbia/NL_kinks/qa2.1_g2.3/_golden_capture/
cd /tokp/work/geosu/Columbia/NL_kinks/qa2.1_g2.3/_golden_capture
python /tokp/work/geosu/Columbia/run_jorek.py shotfile.py
```

No `--run*` flags -- only the folder-population step is being compared, not
an actual JOREK execution.

## Files here

The regular files from that populated directory, unmodified:

```
in_eq  in_main  in_main_r  in_bnd
ffprime_prof.dat  T_prof.dat  rho_prof.dat
real_psi_edge.dat  original_bnd.dat  input_starwall
```

Plus `symlinks.txt`: the raw `ls -la` output from that directory, so the
symlink targets (`exe`, `util`, `jobscripts`, `jorek2_poincare`,
`jorek2_postproc`, `submit_jorek.sh`, `stpts`, `starwall-response.dat`) are
recorded without needing the symlinks themselves to survive a file transfer
off the HPC.

## A note on validity

`shotfile.py`'s `castor_params["machine_folder"]` points at
`/tokp/work/geosu/castor3d/DIIID_low_pres` on the HPC -- the same source this
checkout's `castor3d/DIIID_low_pres/` fixtures were copied from. If that data
is regenerated on the HPC between the fixture copy and this capture, the
comparison is no longer meaningful; re-copy both together if in doubt.
