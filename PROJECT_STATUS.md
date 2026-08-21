# Project status and publication gates

## Completed in the initial implementation

- Repository layout, frozen conventions, environment specification, and tests.
- Nonlinear collocation solver for static, nodeless ell-boson-star backgrounds.
- Physical bound-state restriction `0 < omega/mu < 1` and Schwarzschild plus
  massive-tail outer boundary conditions.
- Reproduction of the `ell=1`, `a_1^0=0.08` frequency from Table II of
  arXiv:2103.15012: computed `0.8518974`, tabulated `0.8519`.
- A 21-model `ell=1` sequence from `a_1^0=0.02` through `0.12`. Its grid maximum
  is at `a_1^0=0.10`, with `omega=0.8356722` and `M=1.1763866`, consistent with
  the published maximum model (`omega about 0.836`, `M about 1.18`).
- Outer-domain comparison at `r_max=60, 80, 100`: the benchmark frequency and
  ADM mass agree beyond ten and nine decimal places, respectively.

## Reproducibility record

- Background implementation commit:
  `600c3677290bb05e8ac0901b945f7f98806af98c`.
- Annotated checkpoint tag: `background-v0.1-provisional` (placed on the
  metadata-only commit that records the implementation hash).
- Runtime: Python 3.14.3; NumPy 2.4.3; SciPy 1.17.1;
  Matplotlib 3.10.8; pytest 9.1.0.
- Sequence command:
  `python -m background.ell_boson_star scan --ell 1 --a0-min 0.02 --a0-max 0.12 --count 21 --output data/ell1_sequence.csv`.
- Radius-audit command:
  `python -m background.radius_audit --a0 0.10 --output data/background_radius_audit.json`.
- BVP settings: `r_max=80`, 800 initial mesh points, relative collocation
  tolerance `1e-7`, and a maximum of 40,000 adaptive nodes.
- Outer-domain comparison: `r_max=60, 80, 100`, with 800 initial points and
  tolerance `1e-7` in each run.
- Large profiles (`*.npz`) and sequences (`*.csv`) are regenerated and ignored
  by Git. The small JSON radius-audit record is committed.

## Unresolved published-radius discrepancy

For the maximum-mass model, the literal radius satisfying `M(r)=0.99 M_T` is
about `10.18`, whereas the background paper explicitly defines that quantity
but quotes `R(99%)=12.75`. The latter is close to this implementation's
`M(r)=0.999 M_T` radius. This is classified as an unresolved numerical or
labeling discrepancy, not an alternative published convention and not yet a
published error. Both `R99` and `R999` are stored, and the committed audit also
records proper-energy, Noether-charge, finite-mass, and extrapolated-mass radii.

Gate A remains **in progress**, not passed, until an independent shooting solver
reproduces representative profiles and this radius discrepancy is resolved.

## Not yet implemented

- Gate B remainder: ordinary ell=0 QNM benchmark. The radial ell=1 solver is
  now implemented separately from the tagged background checkpoint.
- Gate C: fully relativistic nonradial harmonic reduction.
- Gates D-H: new QNMs, parameter campaign, physical conclusions, and paper.

The next hard milestone is an independent ordinary-boson-star QNM benchmark.

## Radial pulsation progress after the background checkpoint

- The Appendix system for `(delta varphi_11, delta L)` is implemented with
  analytic background derivatives and `sigma2` as the eigenvalue.
- The official nonlinear global BVP directly imposes both physical outer
  conditions. At `a_1^0=0.08`, it gives `sigma2=2.40091e-4` and
  `center_c=-2.78360e-2` at `r_max=25`, with zero resolved interior nodes.
- The BVP domain plateau at `r_max=25,30,40` spans about `0.02%`.
- Direct physical boundary residuals are at or below `O(1e-20)` and the stored
  collocation residual is about `3e-6`.
- The computed ground eigenvalue is positive at `a_1^0=0.050`, approximately
  zero at `0.100`, and negative at `0.105`.
- Detailed solver settings and results are versioned in
  `data/radial_benchmarks.csv`.
- The affine outward-basis solver is deprecated: subtracting exponentially
  amplified basis states gives misleading normalized boundary residuals.
- The background perturbation interface now derives `psi` and `psi'` from one
  interpolant of the regular field `u=psi/r^ell`.

The center-series coefficient called `a0` in the Appendix must equal the
actual leading coefficient of the background variable used in the ODE. For the
validated repository normalization this is `a_1^0/kappa = 0.08/3`, not `0.24`.
Using `0.24` leaves a non-vanishing `O(1)` center-operator residual; the
difference is exactly accounted for by the quadratic `6 kappa a0^2` term.

The radial physics is reproduced, but the numerical milestone remains
**provisional** until a second well-conditioned formulation (for example,
multiple shooting with QR stabilization or global spectral collocation)
independently reproduces the BVP result. Do not use the deprecated affine
formulation for nonradial QNMs.
