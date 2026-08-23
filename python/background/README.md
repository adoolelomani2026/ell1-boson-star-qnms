# Background equations and solver

For `mu=1`, `kappa=2 ell+1`, and `gamma^2=(1-2M/r)^(-1)`, the static equations
implemented here are

```text
M' = kappa r^2/2 [psi'^2/gamma^2
                  + (omega^2/alpha^2 + 1 + ell(ell+1)/r^2) psi^2]

alpha'/alpha = gamma^2 {M/r^2 + kappa r/2 [psi'^2/gamma^2
                  + (omega^2/alpha^2 - 1 - ell(ell+1)/r^2) psi^2]}

psi'' + (2/r + alpha'/alpha - gamma'/gamma) psi'
      + gamma^2 [omega^2/alpha^2 - 1 - ell(ell+1)/r^2] psi = 0.
```

They follow directly from Eqs. (4)-(5) of arXiv:2103.15012. In the rescaled
radial-field convention used by the numerical sequence, at the center
`psi = (a_ell^0/kappa) r^ell + O(r^(ell+2))`, `M=O(r^(2ell+1))`, and the metric
is regular. At the finite outer boundary the code applies the
Schwarzschild lapse and the decaying Robin condition through order `1/r`, using
the power in Eq. (45) of arXiv:2103.15012.

`scipy.integrate.solve_bvp` solves the nonlinear eigenvalue problem, using
continuation in `a_ell^0` when needed. Diagnostics include the independent
first-order-equation residuals and the scalar-tail logarithmic derivative.

This amplitude conversion reproduces the paper's Table II value
`omega=0.8519` for `ell=1`, `a_1^0=0.08`. The finite-domain boundary condition
is controlled by repeating calculations at
larger `r_max`; it is not a substitute for an asymptotic error study.
