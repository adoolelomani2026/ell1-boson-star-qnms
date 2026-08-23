# Publication artifacts

`build_preprint.py` regenerates the numerical figures and compiles the
REVTeX 4.2 PRD author manuscript.

```powershell
pip install -r environment/requirements-report.txt
python paper/build_preprint.py
```

The final PDF is written to
`output/pdf/axial_matter_channel_ell1_preprint.pdf`. Temporary LaTeX and visual
QA assets are placed under `tmp/pdfs/`.

The manuscript reports the exact odd-parity axial matter-source projection,
the certified radial substrate, the counted coupled axial pole, a predeclared targeted driven
response, a Schwarzschild QNM control, far-boundary tests, a short time-domain
frequency check, scattering amplitudes, and the stationary axial response.
The remaining pointwise constraint floor and the absence of a canonical
conversion-flux current are stated explicitly.
