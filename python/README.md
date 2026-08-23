# Python code

The scientific implementation is divided by physical layer:

- `background/`: equilibrium and charged-background solvers;
- `radial/`: relativistic radial pulsations and certification;
- `nonradial/`: axial perturbations, spectra, response, and evolution;
- `experiments/`: executable production and validation runs.

Install from the repository root with `python -m pip install -e .`. Production
runs should then use module form, for example
`python -m experiments.axial_qnm_checkpoint`.
