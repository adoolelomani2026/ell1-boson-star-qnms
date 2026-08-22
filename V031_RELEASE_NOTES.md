# Radial v0.3.1 hardening checkpoint

The v0.3 audit certifies the radial `ell=1` milestone. This checkpoint preserves
that result and implements the audit's pre-production hardening actions.

## Changes

- Row-and-column equilibration of the generalized eigenvalue pencil.
- Scaled and unscaled algebraic residuals and a left/right eigenvalue condition
  number for every spectral mode.
- An independently transcribed spectral coefficient matrix and center
  condition, tested against the BVP differential operator on arbitrary fields.
- Explicit rejection of nonsmooth PCHIP backgrounds by the spectral solver.
- Interpolation-and-bracketing node location without the former `1e-4` cutoff.
- Eigenfunction-overlap mode tracking across spectral resolutions.
- A broad `N=50,60,80,100,120,160` ground/overtone stress envelope.
- BVP first-overtone certification at `r_max=40,50,60`, plus an `N=160`,
  `r_max=60` spectral comparison.
- Separate conservative deterministic-systematics records for the ground mode
  and first overtone.
- Automated matrix-assembly, center/outer-row, higher-node, overlap, and
  BVP-versus-spectral eigenfunction checks.

The raw single-domain pencil becomes increasingly ill-conditioned for higher
modes and larger domains. This release measures that behavior; it does not
claim that equilibration turns one dense domain into a production nonradial
architecture. A smooth-background mapped multidomain method remains required
before publishing new complex nonradial QNMs.

## Gates

- Gate A: in progress (independent background shooting and radius discrepancy).
- Radial portion of Gate B: pass.
- Full Gate B: in progress; ordinary `ell=0` QNM is next.
- Production nonradial solver: not yet certified.
