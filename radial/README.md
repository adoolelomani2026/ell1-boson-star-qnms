# Radial pulsation solver

This package implements the Appendix equations of arXiv:2103.15012 in the
variables `delta_varphi_11` and
`delta_L = delta_lambda/(2 kappa psi_1^2)`. The numerical eigenvalue is
`sigma2`, so stable and unstable modes use the same code path.

The background interface supplies `alpha`, `gamma`, `u_1`, `u_1'`, `psi_1`,
and `psi_1'`. Metric logarithmic derivatives and `psi_1''` are evaluated from
the field equations rather than differentiated splines.

The official validator is `solve_radial_bvp`, a global nonlinear collocation
problem that treats `sigma2` as a free parameter and imposes both published
physical outer conditions directly. It reports the unscaled physical boundary
residuals and the maximum collocation residual.

The earlier affine outward-basis implementation is retained in
`radial/shooting.py` only for historical comparison and emits a deprecation
warning. Cancellation between exponentially amplified basis solutions makes
that formulation unsuitable for certification or for extension to nonradial
QNMs.

Run the focused benchmark with:

```powershell
python -m pytest tests/test_radial_a008.py -q
```

Regenerate the complete, slower validation table with:

```powershell
python -m radial.diagnostics --output data/radial_benchmarks.csv
```

The background interface interpolates the regular field `u=psi/r^ell` once and
derives both `psi` and `psi'` from that same interpolant. The center expansion uses the actual coefficient in
`psi_1 = a0 r + O(r^3)`. For the validated background normalization this is
the tabulated amplitude divided by `kappa`, so the `a_1^0=0.08` model uses
`a0=0.08/3`. Substitution into the ODE operator verifies this choice directly.
