# Numerical reports

This directory is the machine-readable evidence base for the manuscript. It
plays the same role as `reports/` in the Schwarzschild--AdS reference repository,
while retaining NumPy profile archives required by the coupled solvers. Records
are split into `background/`, `radial/`, `axial/`, `controls/`, and
`extensions/`.

## Artifact groups

Prefix | Contents
--- | ---
`background_*`, `ell1_sequence.csv` | Equilibrium profiles, branch sequence, radius audit, sensitivities, and IVP checks
`radial_*`, `report_radial_stability.csv`, `refined_radial_*` | BVP/spectral benchmarks, stability scans, and deterministic uncertainty budgets
`axial_projection_*` | Exact and numerical angular-reduction controls
`axial_qnm_*` | Declared search, refined pole, far-boundary campaign, and seven-star continuation
`axial_quadtree_census_pilot.json` | Explicit-sheet adaptive Evans census in one declared cut-free validation window
`axial_constraint_*` | Independent unused-Einstein-equation monitor
`axial_scattering_*`, `axial_resonant_*`, `axial_static_*` | Open-channel, driven, and zero-frequency responses
`axial_time_domain_*` | Time-evolution checkpoint and current resolution/Courant matrix
`schwarzschild_qnm_benchmark.json` | Independent Regge--Wheeler/Leaver control
`ordinary_ell0/*` | Ordinary mini-boson-star turning-point scan and normalization benchmark
`charged_*`, `semiclassical_*`, `physics_extension_*` | Supporting studies not applied to the neutral pole

JSON files store conventions, parameters, residual definitions, acceptance
tests, and provenance alongside results. CSV files store sequences and
resolution campaigns. NPZ files store the radial arrays needed to regenerate
the nonradial calculations.

## Regeneration

Install the project in editable mode, then run experiments from the repository
root:

```powershell
python -m experiments.axial_projection_checkpoint
python -m experiments.schwarzschild_qnm_benchmark
python -m experiments.ordinary_ell0_background_benchmark
python -m experiments.axial_qnm_discovery
python -m experiments.axial_quadtree_census
python -m experiments.axial_qnm_checkpoint
python -m experiments.axial_qnm_far_boundary
python -m experiments.axial_qnm_branch
python -m experiments.axial_constraint_checkpoint
python -m experiments.axial_time_domain_checkpoint
python -m experiments.axial_scattering_checkpoint
python -m experiments.axial_resonant_response
python -m experiments.axial_static_response
```

The radial certification tables can be regenerated with:

```powershell
python -m radial.diagnostics --output reports/radial/radial_benchmarks.csv --uncertainty-output reports/radial/radial_uncertainty.json --overtone-uncertainty-output reports/radial/radial_overtone_uncertainty.json
```

Large computations overwrite only their declared report targets. The tests
check the numerical contracts used by the manuscript.
