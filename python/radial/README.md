# Radial pulsation solver

This package implements the Appendix equations of arXiv:2103.15012 in the
variables `delta_varphi_11` and
`delta_L = delta_lambda/(2 kappa psi_1^2)`. The numerical eigenvalue is
`sigma2`, so stable and unstable modes use the same code path.

The background interface supplies `alpha`, `gamma`, `u_1`, `u_1'`, `psi_1`,
and `psi_1'`. Metric logarithmic derivatives and `psi_1''` are evaluated from
the field equations rather than differentiated splines.

The primary validator is `solve_radial_bvp`, a global nonlinear collocation
problem that treats `sigma2` as a free parameter and imposes both published
physical outer conditions directly. It reports the unscaled physical boundary
residuals, SciPy's maximum interval RMS relative residual, and a separately
sampled maximum pointwise relative ODE residual.

`solve_radial_spectrum` is an independent Chebyshev-Lobatto discretization of
`A x = sigma2 B x`. Its coefficient matrix is transcribed separately from the
BVP right-hand side and checked against direct operator evaluations. The pencil
is equilibrated on both sides before solution and reports scaled and unscaled
residuals plus a left/right eigenvalue condition number. It has no nonlinear
eigenvalue guess and returns the ground mode and overtones simultaneously. At
`a_1^0=0.08`, both methods reproduce
`sigma0^2 ~= 2.40043e-4`; the spectral first overtone has one node and
`sigma1^2 ~= 8.227e-3`.

The spectral solver deliberately rejects PCHIP backgrounds: a small algebraic
matrix residual does not certify a pencil built from nonsmooth coefficient
interpolation. PCHIP is retained only for local-BVP representation uncertainty.
Nodes are located by interpolation and bracketing, and modes can additionally
be continued by normalized physical-eigenfunction overlap.

The earlier affine outward-basis implementation is retained in
`python/radial/shooting.py` only for historical comparison and emits a deprecation
warning. Cancellation between exponentially amplified basis solutions makes
that formulation unsuitable for certification or for extension to nonradial
QNMs.

Run the focused benchmark with:

```powershell
python -m pytest tests/test_radial_a008.py -q
```

Regenerate the complete, slower validation table with:

```powershell
python -m radial.diagnostics --output reports/radial/radial_benchmarks.csv --uncertainty-output reports/radial/radial_uncertainty.json --overtone-uncertainty-output reports/radial/radial_overtone_uncertainty.json
```

The default background interface reconstructs the regular field `u=psi/r^ell`
with a cubic Hermite spline built from the stored `psi` and `psi'`, then derives
both quantities consistently. PCHIP remains available only for representation
uncertainty checks. The center expansion uses the actual coefficient in
`psi_1 = a0 r + O(r^3)`. For the validated background normalization this is
the tabulated amplitude divided by `kappa`, so the `a_1^0=0.08` model uses
`a0=0.08/3`. Substitution into the ODE operator verifies this choice directly.
