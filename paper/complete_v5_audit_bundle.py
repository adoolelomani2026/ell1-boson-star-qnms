"""Complete and integrity-check the v5 release-audit artifact bundle."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path


DOWNLOADS = Path(__file__).resolve().parents[3]
PREFIX = "Hidden_Axial_Matter_v5"
PDF = DOWNLOADS / f"{PREFIX}_RELEASE_AUDIT.pdf"
MARKDOWN = DOWNLOADS / f"{PREFIX}_RELEASE_AUDIT.md"
CONTACT = DOWNLOADS / f"{PREFIX}_RELEASE_AUDIT_contact.png"
FRESH = DOWNLOADS / f"{PREFIX}_fresh_checks.json"
GATE = DOWNLOADS / f"{PREFIX}_gate_matrix.csv"
ARTIFACTS = DOWNLOADS / f"{PREFIX}_RELEASE_AUDIT_ARTIFACTS.zip"
MANIFEST = DOWNLOADS / f"{PREFIX}_RELEASE_AUDIT_SHA256.txt"


GATE_ROWS = (
    ("Package integrity", "PASS", "203/203 top-level and 149/149 deposit checksums"),
    ("Public repository/release", "PASS", "Public v0.5.0; exact ZIP/PDF digests match"),
    ("License", "PASS", "BSD-3-Clause"),
    ("Git provenance", "PASS", "Complete bundle, clean annotated tag"),
    ("Regression tests", "PASS", "78 passed in alternate environment"),
    ("Manuscript build", "PASS", "65 pages, clean references and fonts"),
    ("Axial angular source", "PASS", "Exact density null and nonzero axial stress"),
    ("Radial foundation", "PASS", "BVP and independently assembled spectral pencil"),
    ("Schwarzschild benchmark", "PASS", "6.17e-9 absolute error"),
    ("Central axial pole", "PASS, local", "Two-domain root, grid search, winding-one count"),
    ("Seven-star branch", "PASS, sampled interval", "Counts, thresholds, overlaps"),
    ("Center/tolerance/holomorphy", "PASS", "Included and reproducible"),
    ("Unused Einstein equation", "PASS at stated accuracy", "Explicit 1e-8--1e-7 monitor"),
    ("Time-domain experiment", "PASS with limited claim", "Real-frequency check only"),
    ("Canonical channel flux", "OPEN, disclosed", "Not claimed as completed"),
    ("Ordinary ell=0 stellar QNM", "OPEN, disclosed", "Recommended, not required for local claim"),
    ("Global nonradial stability", "OUT OF SCOPE", "Not claimed"),
    ("Author declarations", "SIGN-OFF REQUIRED", "Personal confirmation before upload"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path in (PDF, MARKDOWN, CONTACT, FRESH):
        if not path.is_file():
            raise SystemExit(f"Missing required v5 audit artifact: {path}")

    with GATE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("component", "status", "evidence_or_remaining_action"))
        writer.writerows(GATE_ROWS)

    with zipfile.ZipFile(ARTIFACTS, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (MARKDOWN, CONTACT, GATE, FRESH):
            archive.write(path, path.name)

    manifest = (
        f"{sha256(PDF)}  /mnt/data/{PDF.name}\n"
        f"{sha256(ARTIFACTS)}  /mnt/data/{ARTIFACTS.name}\n"
    )
    MANIFEST.write_text(manifest, encoding="utf-8", newline="\n")

    with zipfile.ZipFile(ARTIFACTS) as archive:
        expected = {MARKDOWN.name, CONTACT.name, GATE.name, FRESH.name}
        if set(archive.namelist()) != expected:
            raise SystemExit("Audit artifact ZIP member mismatch")

    print(GATE)
    print(ARTIFACTS)
    print(MANIFEST)


if __name__ == "__main__":
    main()
