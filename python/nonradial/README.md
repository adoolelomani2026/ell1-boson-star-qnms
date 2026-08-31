# Relativistic odd-parity calculation

This package implements the closed neutral `ell=1`, `J=L=2` axial
Einstein--Klein--Gordon system. The exact internal-state sum has zero density
projection but nonzero axial current and odd anisotropic stress,

```text
<X^A,S_A>   = -i/(2 sqrt(pi))
<X^AB,Q_AB> =  i/sqrt(pi).
```

The frequency-domain solver evolves three regular center solutions and three
physical exterior channels, matches their subspaces, refines zeros of a
holomorphic determinant, and counts them using the exterior product of two
three-planes in `Lambda^3(C^6)`. The production branch contains seven stable
backgrounds from `a1_0=0.065` through `0.095`; every local contour has winding
one and every adjacent six-state profile overlap exceeds `0.9991`.

The central result is

```text
sigma = 0.0493977307850 - 5.9651933e-7 i.
```

Current limitations are explicit: the Richardson-extrapolated unused Einstein
monitor has a `1.5e-7`--`2.2e-7` pointwise finite-difference plateau, the ordinary `ell=0` matter-filled stellar QNM has
not been reproduced with this same pipeline, the scattering reciprocity weight
is not a canonical energy flux, and the short time evolution does not measure
the damping rate independently.

Reproduce the core records with:

```powershell
python -m experiments.axial_qnm_discovery
python -m experiments.axial_qnm_checkpoint
python -m experiments.axial_qnm_branch
python -m experiments.axial_constraint_checkpoint
python -m experiments.axial_numerical_robustness
python -m experiments.axial_time_domain_checkpoint
pytest -q
```
