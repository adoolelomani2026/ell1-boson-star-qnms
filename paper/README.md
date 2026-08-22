# Publication artifacts

`build_certification_report.py` regenerates the background-and-radial
publication checkpoint, including six figures, eight interpretive tables, and
the 13-model radial stability source table.

```powershell
pip install -r environment/requirements-report.txt
python paper/build_certification_report.py
```

The final PDF is written to
`output/pdf/boson_star_background_radial_certification_v031.pdf`. High-resolution
PNG previews and vector figure PDFs are temporary QA/build assets under
`tmp/pdfs/certification_report/`. Rendered report pages used for layout review
are also temporary.

The report is a certification milestone, not the final PRD manuscript. It
explicitly leaves Gate A, the ordinary `ell=0` QNM benchmark, and production
nonradial calculations open.
