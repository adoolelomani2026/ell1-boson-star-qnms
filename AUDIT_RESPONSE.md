# Response to the radial-v0.1 numerical audit

The audit's central finding is accepted: the affine outward-basis method can
approximate the interior eigenmode but cannot certify the physical outer
conditions because it subtracts exponentially amplified numerical solutions.

## Corrected

- Added `radial/bvp.py`, a nonlinear global collocation eigenvalue problem that
  imposes both published physical outer conditions directly.
- Replaced independent PCHIP representations of `psi` and `psi'` with one
  interpolant of `u=psi/r^ell`; both quantities now come from that interpolant.
- Deprecated the affine solver and removed its normalized cancellation metric
  from the official benchmark table.
- Regenerated the table with formulation, direct `F1` and `F2`, maximum
  collocation residual, adaptive node count, environment lock, and provenance.
- Replaced CSV-only convergence and stability tests with tests that execute the
  nonlinear BVP.
- Added an exact dependency lock and CI for the locked Windows/Python 3.14
  environment plus an alternate Ubuntu/Python 3.13 environment.

## Corrected benchmark

For `ell=1`, `a_1^0=0.08`, `r_max=25`, the global BVP gives

```text
sigma2   = 2.4009102283608806e-4
center_c = -2.783599461663854e-2
F1       = 7.789363997171208e-22
F2       = 1.1241438868329525e-21
```

The `r_max=25,30,40` sequence forms a plateau and the BVP reproduces the sign
change between `a_1^0=0.100` and `0.105`.

## Still open

- A second well-conditioned formulation (QR-stabilized multiple shooting,
  Riccati/Evans propagation, or spectral collocation) is still required.
- Gate A still needs an independent background shooting implementation.
- Therefore the radial numerical milestone remains provisional and work does
  not proceed to nonradial QNMs on the basis of this BVP alone.

