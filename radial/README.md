# Radial pulsation solver

This package implements the Appendix equations of arXiv:2103.15012 in the
variables `delta_varphi_11` and
`delta_L = delta_lambda/(2 kappa psi_1^2)`. The numerical eigenvalue is
`sigma2`, so stable and unstable modes use the same code path.

The background interface supplies `alpha`, `gamma`, `u_1`, `u_1'`, `psi_1`,
and `psi_1'`. Metric logarithmic derivatives and `psi_1''` are evaluated from
the field equations rather than differentiated splines.

The two published outer conditions become nearly collinear when the growing
rescaled solution dominates. Because the ODE is linear, the solver integrates
two center-coefficient basis solutions, eliminates `center_c` analytically with
the first condition, and solves the remaining scalar residual with Brent's
method. The final physical perturbations are reconstructed from that basis.

Run the focused benchmark with:

```powershell
python -m pytest tests/test_radial_a008.py -q
```

Regenerate the complete, slower validation table with:

```powershell
python -m radial.diagnostics --output data/radial_benchmarks.csv
```

The center expansion uses the actual coefficient in
`psi_1 = a0 r + O(r^3)`. For the validated background normalization this is
the tabulated amplitude divided by `kappa`, so the `a_1^0=0.08` model uses
`a0=0.08/3`. Substitution into the ODE operator verifies this choice directly.

