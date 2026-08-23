# Radial v0.3 spectral certification

## Result

The relativistic radial `ell=1` boson-star calculation is now independently
certified by two well-conditioned formulations:

1. a nonlinear global boundary-value problem that directly imposes both
   physical outer conditions; and
2. a Chebyshev-Lobatto generalized eigenproblem `A x = sigma2 B x` that requires
   no nonlinear eigenvalue initial guess.

For `a_1^0=0.08`, `epsilon=1e-3`, and `r_max=40`, the Hermite-background BVP
gives

```text
sigma0^2 = 2.4004311443394838e-4
```

The 80-point spectral result differs by `1.64e-9` in absolute `sigma0^2`. The
independent solver also obtains the one-node first overtone,

```text
sigma1^2 = 8.227667674704427e-3
```

Both formulations give a positive ground eigenvalue at `a_1^0=0.10` and a
negative eigenvalue at `a_1^0=0.105`, independently reproducing the stability
crossing.

## Numerical evidence

- Hermite reconstruction uses the stored `psi` and `psi'` consistently.
- BVP diagnostics explicitly separate SciPy's interval RMS relative residual,
  a dense pointwise relative ODE residual, and imposed endpoint residuals.
- Spectral convergence is recorded at 60, 80, and 100 points.
- The uncertainty budget includes domain, cross-method, spectral-resolution,
  and Hermite-versus-PCHIP representation changes.
- Combined quadrature uncertainty: `4.00e-9` in absolute `sigma0^2`.
- Conservative sum of components: `6.80e-9` in absolute `sigma0^2`.
- The complete automated suite passes: `13 passed`.

See `reports/radial/radial_benchmarks.csv` for all 14 certification rows and
`reports/radial/radial_uncertainty.json` for the machine-readable uncertainty and runtime
provenance.

## Gate status

The radial part of Gate B has independent numerical certification. Gate A
remains in progress pending an independent background shooting solution and
resolution of the published radius discrepancy. Full Gate B still requires the
ordinary `ell=0` QNM benchmark; production nonradial work should follow that
benchmark.
