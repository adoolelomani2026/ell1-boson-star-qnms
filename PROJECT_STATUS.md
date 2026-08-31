# Project status and publication gates

## Headline result

For the neutral `ell=1`, `J=L=2` odd sector, the internal-state sum has zero
density projection but nonzero axial vector and tensor projections,

```text
<X^A,S_A>   = -i/(2 sqrt(pi))
<X^AB,Q_AB> =  i/sqrt(pi).
```

The resulting six-state relativistic system has a counted stable branch across
seven backgrounds. At the central model, two refined matching domains give
`sigma = 0.0493977307850 - 5.9651933e-7 i`, with real- and imaginary-part
domain spreads of `6.3e-13` and `2.7e-14`. A 41-point response scan selected
without reading the pole result recovers the center to `1.1e-9` fractionally
and the half-width to `4.5e-5`.

## Passed gates

- Regular `ell=1` equilibrium background and maximum-mass sequence.
- Published radial ground mode, first overtone, eigenfunctions, and stability
  crossing, independently reproduced by a global BVP and an equilibrated
  Chebyshev pencil.
- Exact axial angular projection, all five `M` values, and the ordinary
  `ell=0` null control.
- Closed axial Einstein--Klein--Gordon first-order system and flat/exterior
  limits.
- Declared root discovery, exterior-algebra Evans count, two-domain
  refinement, and massive-channel sheet handling.
- Third-order gravitational and Coulomb-corrected scalar asymptotics; the
  `r_far=600` to `900` shift is below `6e-13` per frequency component.
- Schwarzschild Regge--Wheeler/Leaver benchmark within `6.2e-9` in complex
  frequency and with normalized continued-fraction residual below `8e-15`.
- Predeclared targeted driven-resonance center and width recovery without loading the stored pole during fitting.
- Seven-background axial continuation with one locally counted pole per star,
  two matching domains, and adjacent profile overlaps above `0.9991`.
- Nine center-start refinements spanning a factor of 40, five ODE-tolerance
  refinements spanning a factor of 20, and local Cauchy--Riemann checks of both
  holomorphic matching determinants.
- Two-sided stationary axial-response match across six domains.

## v0.6 research-program prerequisites

- An explicit analytic `physical-lower` sideband sheet replaces the legacy
  pointwise square-root choice in the new census path.  The finite cuts and all
  four branch points are represented as data.
- A cached adaptive-quadtree contour with a strict `pi/4` phase-increment bound
  counts and independently assigns one pole in the declared cut-free pilot
  window.  This validates the machinery locally; it is not a global-spectrum
  result.
- The ordinary `ell=0` mini-boson-star family now reaches its nodeless
  maximum-mass turning point.  In the published normalization it gives
  `phi_c=0.191686` and `M=0.6330009`, providing the background prerequisite for
  the polar benchmark.

## Disclosed open boundaries

- The Richardson-extrapolated unused-Einstein-equation monitor has relative
  `L2=2.8e-8`--`3.6e-8` and a `1.5e-7`--`2.2e-7` pointwise plateau across
  250--1000 samples. It is controlled numerically but is not called an exact
  continuum identity until an analytic matrix derivative is implemented.
- The positive diagonal weight that balances the sampled scattering matrix is
  an algebraic reciprocity metric. A canonical symplectic or stress-energy
  flux derivation remains open, so the manuscript does not claim a physical
  conversion probability.
- The short `1+1` evolution reproduces the oscillation frequency to about
  `2.6e-7` fractionally, but cannot independently determine the damping rate:
  the predicted lifetime is about `1.68e6` in code units.
- `B/(A M^5)=-122.47418` is an explicit exterior-basis normalization, not a
  gauge-invariant magnetic Love number.
- The literal `R99` discrepancy with one published table remains documented;
  both `R99` and `R999` are retained.
- All seven claims in `Hidden_Axial_Matter_v1_RESEARCH_PROGRAM.md` remain open.
  In particular, the quadtree pilot covers only a narrow lower-half-plane
  window, and the ordinary-star result currently certifies only the equilibrium
  background, not any polar QNM.

## Model boundary

The paper solves the neutral, free massive-scalar theory. The charged EMKG
equilibrium and coherent-state occupation calculations are supporting studies,
not corrections applied to the neutral pole. A renormalized semiclassical
backreaction calculation would require a specified state, field content, and
renormalization prescription and is therefore outside the stated theory.
