"""Fit the released axial damping branch to an empirical compactness power law."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BRANCH = ROOT / "reports" / "axial" / "axial_qnm_branch.json"
OUTPUT = ROOT / "reports" / "axial" / "branch_powerlaw_fit.json"


def fit_branch_powerlaw(rows: list[dict[str, object]]) -> dict[str, object]:
    compactness = np.asarray([row["compactness_R99"] for row in rows], dtype=float)
    damping = -np.asarray([row["sigma_imag"] for row in rows], dtype=float)
    exponent, log_amplitude = np.polyfit(np.log(compactness), np.log(damping), 1)
    amplitude = float(np.exp(log_amplitude))
    prediction = amplitude * compactness**exponent
    fractional_residual = prediction / damping - 1.0
    return {
        "model": "-Im(sigma) = A * C99^p",
        "fit_space": "unweighted least squares in log(-Im(sigma)) versus log(C99)",
        "sample_size": int(len(rows)),
        "compactness_interval": [float(compactness.min()), float(compactness.max())],
        "A": amplitude,
        "p": float(exponent),
        "maximum_absolute_fractional_residual": float(
            np.max(np.abs(fractional_residual))
        ),
        "rms_fractional_residual": float(
            np.sqrt(np.mean(fractional_residual**2))
        ),
        "warning": (
            "Empirical fit over a narrow relativistic compactness interval; "
            "not a demonstrated weak-field or post-Newtonian law."
        ),
    }


def main() -> None:
    record = json.loads(BRANCH.read_text(encoding="utf-8"))
    result = fit_branch_powerlaw(record["rows"])
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
