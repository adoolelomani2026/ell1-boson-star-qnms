# Conventions

These choices are frozen for the project. Any comparison to another source must
include an explicit conversion into this normalization.

## Geometry and units

- Metric signature: `(-,+,+,+)`.
- Units: `G = c = hbar = mu = 1` unless a dimensional factor is displayed.
- Einstein equations: `G_(mu nu) = 8 pi T_(mu nu)`.
- Areal-radius metric:
  `ds^2 = -alpha(r)^2 dt^2 + gamma(r)^2 dr^2 + r^2 dOmega^2`,
  with `gamma^2 = (1-2M(r)/r)^(-1)`.
- ADM mass: `M_T = lim_(r->infinity) M(r)`.
- Radius: `M(R99) = 0.99 M_T`; compactness `C99 = M_T/R99`.

## Matter normalization

- There are `2 ell+1` equal-mass complex fields
  `Phi_(ell m) = exp(+i omega t) psi_ell(r) Y_(ell m)`.
- Spherical harmonics obey
  `sum_m |Y_(ell m)|^2 = (2 ell+1)/(4 pi)`.
- The stress tensor is Eq. (3) of arXiv:2103.15012, including its factor `1/2`.
- `kappa_ell = 2 ell+1`.
- The CLI family parameter `a_ell^0` is the tabulated regular-center amplitude.
  With the rescaled radial field used for the background equations,
  `lim_(r->0) psi_ell/r^ell = a_ell^0/kappa_ell`. This is the convention that
  reproduces Table II of arXiv:2103.15012 and follows the regular variable in
  Eq. (12c) of arXiv:1805.11488. Any unrescaled physical-field amplitude must
  be converted explicitly.
- The total Noether charge (boson number) is
  `Q = kappa_ell omega integral psi_ell^2 gamma r^2/alpha dr`.

## Harmonics and perturbations

- Scalar harmonics use the Condon-Shortley phase and unit integral norm.
- Clebsch-Gordan coefficients use the standard Condon-Shortley/Wigner convention.
- Perturbations use `exp(-i sigma t)`.
- Therefore `Im(sigma) < 0` is damped and `Im(sigma) > 0` is unstable.
- Gravitational waves are outgoing as `exp(+i sigma r_*)`.
- Massive-scalar sidebands are `omega_+ = omega + sigma` and
  `omega_- = omega - sigma`; their square-root sheets are to be fixed by
  analytic continuation from outgoing propagating or decaying physical data.

## Reproducibility policy

- Persist raw profiles and metadata; derive plots and tables from saved data.
- Report uncertainty from actual resolution/domain/method changes.
- Do not claim a gate passed solely because a nonlinear solver reports success.
