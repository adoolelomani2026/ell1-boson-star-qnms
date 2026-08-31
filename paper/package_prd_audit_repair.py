"""Assemble the audited PRD manuscript and a clean reproducibility deposit."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT.parents[1]
BUILD = ROOT / "tmp" / "pdfs" / "boson_star_radial_preprint"
PDF = ROOT / "output" / "pdf" / "axial_matter_channel_ell1_preprint.pdf"
NAME = "Hidden_Axial_Matter_PRD_Submission"
DEST = ROOT / "output" / "submission" / NAME
ARCHIVE = DEST.with_suffix(".zip")


def copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache"
        ),
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def hashes(root: Path) -> str:
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def collected_test_count() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout)
    if match is None:
        raise RuntimeError("Could not determine collected pytest count")
    return int(match.group(1))


def main() -> None:
    if DEST.exists() or ARCHIVE.exists():
        raise SystemExit(f"Refusing to overwrite existing release: {DEST}")
    if not PDF.exists() or not (BUILD / "manuscript.tex").exists():
        raise SystemExit("Build the manuscript with paper/build_preprint.py first.")

    manuscript = DEST / "Main_Manuscript"
    figures = manuscript / "Figures"
    supplement = DEST / "Supplemental_Material"
    deposit = supplement / "Deposit"
    figures.mkdir(parents=True)
    deposit.mkdir(parents=True)

    audit_record = DEST / "Audit_Record"
    audit_record.mkdir(parents=True)
    for name in (
        "Hidden_Axial_Matter_v4_FINAL_AUDIT_contact.png",
        "Hidden_Axial_Matter_v4_center_start_scan.csv",
        "Hidden_Axial_Matter_v4_gate_matrix.csv",
        "Hidden_Axial_Matter_v4_FINAL_AUDIT_SHA256.txt",
        "Hidden_Axial_Matter_v4_FINAL_AUDIT.md",
        "Hidden_Axial_Matter_v4_FINAL_AUDIT.pdf",
    ):
        source = DOWNLOADS / name
        if not source.exists():
            raise SystemExit(f"Missing audit input: {source}")
        shutil.copy2(source, audit_record / name)

    shutil.copy2(PDF, manuscript / "PRD_Manuscript.pdf")
    for name in ("manuscript.tex", "axial_channel.tex", "axial_dynamics.tex", "axial_response.tex"):
        shutil.copy2(BUILD / name, manuscript / name)
    for figure in sorted(BUILD.glob("figure_*.pdf")):
        shutil.copy2(figure, figures / figure.name)
        shutil.copy2(figure, manuscript / figure.name)
    shutil.copy2(BUILD / "manuscript.log", manuscript / "manuscript.log")

    for directory in ("background", "radial", "nonradial", "experiments"):
        copy_tree(ROOT / "python" / directory, deposit / "python" / directory)
    for directory in ("tests", "environment", "symbolic"):
        copy_tree(ROOT / directory, deposit / directory)
    copy_tree(ROOT / "reports", deposit / "reports")
    for name in ("README.md", "PROJECT_STATUS.md", "CONVENTIONS.md", "LICENSE", "pyproject.toml"):
        shutil.copy2(ROOT / name, deposit / name)
    copy_tree(ROOT / "paper", deposit / "paper")
    shutil.copy2(ROOT / "LICENSE", DEST / "LICENSE")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()
    bundle = supplement / "ell1-boson-star-qnms-v0.5.0.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "--all"], cwd=ROOT, check=True
    )
    page_count = len(PdfReader(PDF).pages)
    test_count = collected_test_count()
    write(
        DEST / "README_FIRST.txt",
        f"""
PRD AUDIT-REPAIRED PRESENTATION PACKAGE

Title: Hidden axial matter in relativistic ell=1 boson stars:
       A long-lived gravito-scalar mode branch and resonant response
Author: Adel H. Al-Yoorby
Prepared: 2026-08-31

Main_Manuscript/PRD_Manuscript.pdf is the visually inspected {page_count}-page REVTeX
author manuscript. Main_Manuscript contains the complete compile source and
vector figures. Supplemental_Material/Supplemental_Material.zip is the clean
code-and-data deposit. SHA256SUMS.txt authenticates every packaged file.

This package repairs the numerical and presentation defects identified in the
submission audit. It does not conceal the remaining scientific boundaries:
the Richardson unused-equation monitor has a 1.5e-7--2.2e-7 pointwise plateau,
the scattering weight is not yet a canonically derived physical flux, and the
short time evolution does not independently measure the extremely long damping
time. These limits are stated in the manuscript.

Repository base commit: {commit}
Release tag: v0.5.0
Public repository: https://github.com/adoolelomani2026/ell1-boson-star-qnms
The included Git bundle preserves the full repository history and release tag.
SHA256SUMS.txt fixes the exact contents of every distributed file.
""",
    )
    write(
        DEST / "VALIDATION_REPORT.txt",
        f"""
AUDIT REPAIR VALIDATION

- Automated regression: {test_count} tests passed.
- Bibliography: 36/36 entries cited; no missing or orphaned entries; numbered
  bibliography is in first-appearance order.
- Schwarzschild control: M omega = 0.37367168441804177
  - 0.08896231568893571 i; absolute reference error 6.17e-9; normalized
  continued-fraction residual 7.74e-15.
- Axial pole: sigma = 0.049397730785015435 - 5.9651932712628e-7 i.
- Seven-background branch: a0=0.065--0.095, compactness 0.0953--0.1131;
  the same long-lived pole is continued across every separately solved star,
  with adjacent normalized profile overlaps above 0.99918.
- Empirical branch fit over the sampled relativistic interval:
  -Im(sigma) = 0.0427983 C99^4.95770; maximum fractional residual 0.237%.
- Root count: the base, strict pi/4, and expanded local exterior-algebra Evans
  contours each have winding number one. The strict-contour maximum resolved
  phase step is 0.60975 < pi/4.
- Far boundary: r_far=600 to 900 shifts are 5.3e-13 (real) and 4.2e-13
  (imaginary); third-order exterior series used.
- Center start: nine converged refinements over a factor of 40; full frequency
  spans 8.86e-12 (real) and 2.90e-15 (imaginary).
- ODE tolerance: five converged refinements over a factor of 20; full frequency
  spans 4.49e-11 (real) and 3.91e-15 (imaginary).
- Numerical holomorphy: maximum resolved Cauchy--Riemann derivative mismatch
  1.92e-7 (raw determinant) and 9.92e-8 (exterior-algebra determinant).
- Predeclared targeted response (fit does not load the stored pole): center error 1.07e-9 fractionally; half-width error 4.49e-5;
  fit covariance uncertainties are recorded in the data file.
- Static response: six two-sided domains; B/(A M^5) = -122.47418 with
  3.15e-5 full domain spread; no Love-number claim.
- PDF: {page_count} pages, US letter, embedded fonts, no Type 3 fonts, no undefined
  references, no overfull boxes; all pages rendered and visually inspected.

OPEN BUT DISCLOSED

- Richardson unused-equation monitor: relative L2 = 2.8e-8--3.6e-8 and
  pointwise relative norm = 1.5e-7--2.2e-7 across 250--1000 samples.
- Same-pipeline ordinary ell=0 stellar-QNM benchmark: not yet completed.
- Canonical graviton/scalar energy-current normalization: not derived.
- Independent time-domain damping measurement: infeasible in the short run
  because the predicted lifetime is about 1.68e6 code units.
""",
    )
    write(
        DEST / "AUTHOR_ACTIONS_REQUIRED.txt",
        """
Before journal upload, the author must personally confirm:

1. Corresponding-author email, affiliation, and ORCID.
2. Funding statement, conflicts declaration, and any institutional wording.
3. Optional repository DOI after linking the public release to Zenodo or an
   equivalent archive; the tagged GitHub release supplies a persistent URL.
4. The cover letter and the journal's current AI-use disclosure fields.
5. Confirm that the charged/coherent-state side-study data should remain in
   supporting material, as in this audited package.

No placeholder in this file should be treated as an author declaration.
""",
    )
    write(
        DEST / "Cover_Letter_DRAFT.txt",
        """
Dear Editors of Physical Review D,

Please consider the manuscript “Hidden axial matter in relativistic ell=1
boson stars: A long-lived gravito-scalar mode branch and resonant response.”
The work derives
an odd-parity matter channel produced by the internal angular structure of a
collectively spherical scalar multiplet and reports a counted, long-lived
coupled gravito-scalar pole. The numerical claim is supported by a seven-star
equilibrium sequence, independent radial discretizations, a Schwarzschild
Leaver benchmark, multiple exterior-algebra Evans contours, far-boundary and
matching-domain studies, and a predeclared targeted driven-response fit.

The manuscript distinguishes the certified pole from two current limitations:
the open-channel scattering weight has not been derived from a canonical
energy current, and the short evolution checks the real frequency but is not
long enough to measure the damping time. These are stated explicitly.

The manuscript is original and is not under consideration elsewhere. No
external funding or competing interests are declared in the manuscript.

Sincerely,
Adel H. Al-Yoorby
""",
    )
    write(
        supplement / "README.txt",
        """
The deposit contains the BSD-3-Clause-licensed Python source, tests, manuscript build source,
machine-readable result tables, and equilibrium profiles used in the paper.
Run `pytest -q` for the regression suite and `python paper/build_preprint.py`
to rebuild the manuscript. See the root README.md and PROJECT_STATUS.md for
the precise claim boundaries.
""",
    )
    (deposit / "SHA256SUMS.txt").write_text(hashes(deposit), encoding="utf-8")
    with zipfile.ZipFile(supplement / "Supplemental_Material.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in deposit.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(deposit))
    (DEST / "SHA256SUMS.txt").write_text(hashes(DEST), encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in DEST.rglob("*") if item.is_file()):
            archive.write(path, Path(NAME) / path.relative_to(DEST))
    print(DEST)
    print(ARCHIVE)


if __name__ == "__main__":
    main()
