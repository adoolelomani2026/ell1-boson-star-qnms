# Publication artifacts

`build_preprint.py` is the primary manuscript builder. It regenerates the
numerical figures and compiles the line-numbered LaTeX author manuscript.

```powershell
pip install -r environment/requirements-report.txt
python paper/build_preprint.py
```

The final PDF is written to
`output/pdf/boson_star_background_radial_preprint_v031.pdf`. High-resolution
PNG previews and vector figure PDFs are temporary QA/build assets under
`tmp/pdfs/certification_report/`. Rendered report pages used for layout review
are also temporary.

The manuscript is a certification preprint, not the final PRD article. It
explicitly leaves Gate A, the ordinary `ell=0` QNM benchmark, and production
nonradial calculations open.
