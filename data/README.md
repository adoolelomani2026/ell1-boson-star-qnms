# Generated data

Large or regenerable numerical outputs are ignored by Git. Recreate the current
artifacts with:

```powershell
python -m background.ell_boson_star solve --ell 1 --a0 0.08 --output data/background_a010_0p08.npz
python -m background.ell_boson_star scan --ell 1 --a0-min 0.02 --a0-max 0.12 --count 21 --output data/ell1_sequence.csv
python -m background.radius_audit --a0 0.10 --output data/background_radius_audit.json
```

The profile archive contains conventions and solver diagnostics alongside the
radial arrays. The sequence CSV is generated directly from solved profiles.
The compact JSON radius audit is versioned because it documents a literature
discrepancy central to the provisional Gate A decision.

