"""Center-start, ODE-tolerance, and numerical-holomorphy audits."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import matching_evans_determinant, matching_raw_determinant
from nonradial.axial_spectrum import refine_analytic_root
from radial.coefficients import RadialBackground


REPORT = ROOT / "reports" / "axial" / "axial_numerical_robustness.json"
CENTER_CSV = ROOT / "reports" / "axial" / "axial_center_start_scan.csv"
TOLERANCE_CSV = ROOT / "reports" / "axial" / "axial_ode_tolerance_scan.csv"
HOLOMORPHY_CSV = ROOT / "reports" / "axial" / "axial_holomorphy_scan.csv"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _refine(
    background: RadialBackground,
    options: dict[str, float | int],
) -> tuple[object, float]:
    started = perf_counter()
    result = refine_analytic_root(
        complex(0.0494, -6.0e-7),
        background,
        matching_options=options,
        determinant=matching_raw_determinant,
    )
    return result, perf_counter() - started


def _span(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def run() -> dict[str, object]:
    solution = BackgroundSolution.load(
        ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz"
    )
    background = RadialBackground(solution, geometry_representation="hermite")
    base: dict[str, float | int] = {
        "r_match": 14.0,
        "r_end": 35.0,
        "r_far": 600.0,
        "asymptotic_order": 3,
        "rtol": 2.0e-8,
        "atol": 2.0e-10,
    }

    center_rows: list[dict[str, object]] = []
    center_starts = (5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 8e-4, 1e-3, 1.5e-3, 2e-3)
    for r_start in center_starts:
        result, elapsed = _refine(background, {**base, "r_start": r_start})
        center_rows.append(
            {
                "r_start": r_start,
                "sigma_real": result.pole.real,
                "sigma_imag": result.pole.imag,
                "quality_factor": result.pole.real / (-2.0 * result.pole.imag),
                "relative_residual": result.relative_residual,
                "converged": result.converged,
                "iterations": result.iterations,
                "evaluations": result.evaluations,
                "elapsed_seconds": elapsed,
            }
        )
    center_reference = next(row for row in center_rows if row["r_start"] == 2e-4)
    for row in center_rows:
        row["delta_real_vs_2e-4"] = float(row["sigma_real"]) - float(center_reference["sigma_real"])
        row["delta_imag_vs_2e-4"] = float(row["sigma_imag"]) - float(center_reference["sigma_imag"])
        row["fractional_damping_shift_vs_2e-4"] = (
            float(row["sigma_imag"]) - float(center_reference["sigma_imag"])
        ) / float(center_reference["sigma_imag"])

    tolerance_rows: list[dict[str, object]] = []
    for rtol in (1e-7, 5e-8, 2e-8, 1e-8, 5e-9):
        result, elapsed = _refine(
            background, {**base, "rtol": rtol, "atol": 0.01 * rtol}
        )
        tolerance_rows.append(
            {
                "rtol": rtol,
                "atol": 0.01 * rtol,
                "sigma_real": result.pole.real,
                "sigma_imag": result.pole.imag,
                "relative_residual": result.relative_residual,
                "converged": result.converged,
                "iterations": result.iterations,
                "evaluations": result.evaluations,
                "elapsed_seconds": elapsed,
            }
        )
    tolerance_reference = next(row for row in tolerance_rows if row["rtol"] == 2e-8)
    for row in tolerance_rows:
        row["delta_real_vs_2e-8"] = float(row["sigma_real"]) - float(tolerance_reference["sigma_real"])
        row["delta_imag_vs_2e-8"] = float(row["sigma_imag"]) - float(tolerance_reference["sigma_imag"])

    sigma = complex(
        float(center_reference["sigma_real"]), float(center_reference["sigma_imag"])
    )
    holomorphy_rows: list[dict[str, object]] = []
    for step in (2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8):
        row: dict[str, object] = {"step": step}
        for name, determinant in (
            ("raw", matching_raw_determinant),
            ("exterior_algebra", matching_evans_determinant),
        ):
            derivative_real = (
                determinant(sigma + step, background, **base)
                - determinant(sigma - step, background, **base)
            ) / (2.0 * step)
            derivative_imag = (
                determinant(sigma + 1j * step, background, **base)
                - determinant(sigma - 1j * step, background, **base)
            ) / (2.0j * step)
            mismatch = abs(derivative_real - derivative_imag) / max(
                abs(derivative_real), abs(derivative_imag), 1e-300
            )
            row[f"{name}_relative_derivative_mismatch"] = float(mismatch)
        holomorphy_rows.append(row)

    result: dict[str, object] = {
        "calculation": "axial QNM center-start, ODE-tolerance, and local numerical-holomorphy audit",
        "background_a1_0": solution.a0,
        "matching_options": base,
        "center_start_rows": center_rows,
        "center_start_summary": {
            "sample_count": len(center_rows),
            "all_converged": all(bool(row["converged"]) for row in center_rows),
            "sigma_real_span": _span(center_rows, "sigma_real"),
            "sigma_imag_span": _span(center_rows, "sigma_imag"),
            "fractional_damping_span": _span(center_rows, "sigma_imag")
            / abs(float(center_reference["sigma_imag"])),
        },
        "ode_tolerance_rows": tolerance_rows,
        "ode_tolerance_summary": {
            "sample_count": len(tolerance_rows),
            "all_converged": all(bool(row["converged"]) for row in tolerance_rows),
            "sigma_real_span": _span(tolerance_rows, "sigma_real"),
            "sigma_imag_span": _span(tolerance_rows, "sigma_imag"),
        },
        "holomorphy_rows": holomorphy_rows,
        "holomorphy_summary": {
            "resolved_step_minimum": 5e-7,
            "resolved_step_maximum": 2e-6,
            "raw_maximum_resolved_mismatch": max(
                float(row["raw_relative_derivative_mismatch"])
                for row in holomorphy_rows
                if 5e-7 <= float(row["step"]) <= 2e-6
            ),
            "exterior_algebra_maximum_resolved_mismatch": max(
                float(row["exterior_algebra_relative_derivative_mismatch"])
                for row in holomorphy_rows
                if 5e-7 <= float(row["step"]) <= 2e-6
            ),
            "interpretation": "Local finite-difference Cauchy--Riemann consistency on a fixed physical sheet; smaller steps expose ODE-solver noise.",
        },
    }
    _write_csv(CENTER_CSV, center_rows)
    _write_csv(TOLERANCE_CSV, tolerance_rows)
    _write_csv(HOLOMORPHY_CSV, holomorphy_rows)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run()
    print(json.dumps({
        "center_start_summary": result["center_start_summary"],
        "ode_tolerance_summary": result["ode_tolerance_summary"],
        "holomorphy_summary": result["holomorphy_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
