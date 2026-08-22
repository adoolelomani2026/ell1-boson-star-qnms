# Relativistic ell=1 boson-star QNMs

Reproducible research code for fully relativistic, coupled gravito-scalar
perturbations of ground-state ell-boson stars.  The repository is intentionally
gate-driven: new nonradial calculations are added only after the background and
published perturbation benchmarks pass.

## Current status

- Gate A (background): in progress. The collocation solver reproduces the
  `ell=1`, `a10=0.08` frequency and the maximum-mass model; independent shooting
  and the published effective-radius convention still need resolution.
- Gate B (perturbation benchmarks): the radial `ell=1` milestone passes. The
  ground mode, first overtone, eigenfunctions, and stability crossing are
  independently reproduced by a nonlinear global BVP and an equilibrated
  Chebyshev generalized-eigenvalue solver. The ordinary `ell=0` QNM benchmark
  is the next hard milestone.
- Gates C-H: not yet implemented.

The conventions are fixed in [CONVENTIONS.md](CONVENTIONS.md). The equations and
numerical method are documented in [background/README.md](background/README.md).

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r environment/requirements-pins.txt
python -m background.ell_boson_star solve --ell 1 --a0 0.08 --output data/background_a010_0p08.npz
python -m background.ell_boson_star scan --ell 1 --a0-min 0.01 --a0-max 0.12 --count 24 --output data/ell1_sequence.csv
pytest
```

The exact top-level pins reproduce the checkpoint environment; the looser
`requirements.txt` is retained for development. The pins are not a complete
hash-locked transitive environment.

Here `a0` is the tabulated regular-center amplitude used by the cited papers.
In their rescaled field convention the raw leading coefficient is
`psi_ell/r^ell = a0/(2 ell+1)`.

## Output

The solver writes compressed NumPy profiles containing `r`, `mass`, `alpha`,
`gamma`, `psi`, and `dpsi`, plus scalar metadata. A scan writes a CSV containing
frequency, ADM mass, Noether charge, `R99`, `R999`, compactnesses, residual diagnostics,
and solver status for every model.

## Primary benchmark source

M. Alcubierre et al., *On the linear stability of ell-boson stars with respect
to radial perturbations*, Phys. Rev. D 103, 124046 (2021), arXiv:2103.15012.
