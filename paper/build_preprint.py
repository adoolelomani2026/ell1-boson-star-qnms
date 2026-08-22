"""Build the line-numbered LaTeX preprint and regenerate its figure data."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.build_certification_report import ASSETS, _build_numerical_assets, _configure_plotting

SOURCE = ROOT / "paper" / "boson_star_radial_preprint.tex"
BUILD = ROOT / "tmp" / "pdfs" / "boson_star_radial_preprint"
OUTPUT = ROOT / "output" / "pdf" / "boson_star_background_radial_preprint_v031.pdf"


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    _configure_plotting()
    _build_numerical_assets()

    replacements = {
        "@@COMMIT@@": git("rev-parse", "HEAD"),
        "@@TREE@@": git("rev-parse", "HEAD^{tree}"),
        "@@PYTHON@@": platform.python_version(),
        "@@NUMPY@@": np.__version__,
        "@@SCIPY@@": scipy.__version__,
    }
    manuscript = SOURCE.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        manuscript = manuscript.replace(marker, value)
    tex = BUILD / "manuscript.tex"
    tex.write_text(manuscript, encoding="utf-8")

    for figure in ASSETS.glob("figure_*.pdf"):
        shutil.copy2(figure, BUILD / figure.name)

    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"]
    for _ in range(3):
        subprocess.run(command, cwd=BUILD, check=True)
    shutil.copy2(BUILD / "manuscript.pdf", OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
