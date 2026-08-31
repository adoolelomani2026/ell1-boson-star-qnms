"""Benchmark the ordinary mini-boson-star background and turning point."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import scan_sequence


def main() -> None:
    amplitudes = np.linspace(0.25, 0.29, 21)
    solutions = scan_sequence(
        0, amplitudes, r_max=80.0, points=800, tolerance=1.0e-7
    )
    rows = [
        {
            "a0_repository": solution.a0,
            # Macedo et al. omit the repository stress tensor's explicit 1/2,
            # so equal physical stress energy maps phi_c=a0/sqrt(2).
            "phi_c_macedo_normalization": solution.a0 / np.sqrt(2.0),
            "omega": solution.omega,
            "adm_mass": solution.adm_mass,
            "r99": solution.r99,
            "compactness99": solution.compactness99,
            "nodeless": bool(np.count_nonzero(np.diff(np.signbit(solution.psi))) == 0),
            "solver_status": solution.solver_status,
        }
        for solution in solutions
    ]
    peak_index = int(np.argmax([row["adm_mass"] for row in rows]))
    fit_indices = np.arange(max(0, peak_index - 3), min(len(rows), peak_index + 3))
    fit_x = np.asarray([rows[index]["a0_repository"] for index in fit_indices])
    fit_y = np.asarray([rows[index]["adm_mass"] for index in fit_indices])
    coefficients = np.polyfit(fit_x, fit_y, 2)
    vertex_a0 = float(-coefficients[1] / (2.0 * coefficients[0]))
    vertex_mass = float(np.polyval(coefficients, vertex_a0))
    vertex_phi = float(vertex_a0 / np.sqrt(2.0))
    published = {
        "kaup_maximum_mass": 0.633,
        "macedo_critical_phi_c": 0.1916,
    }
    result = {
        "calculation": "ordinary ell=0 mini-boson-star background benchmark",
        "scope": "background and turning-point normalization; not a QNM result",
        "normalization_conversion": "phi_c(Macedo)=a0(repository)/sqrt(2)",
        "quadratic_fit_indices": fit_indices.tolist(),
        "turning_point": {
            "a0_repository": vertex_a0,
            "phi_c_macedo_normalization": vertex_phi,
            "adm_mass": vertex_mass,
        },
        "published_targets": published,
        "absolute_errors": {
            "maximum_mass": abs(vertex_mass - published["kaup_maximum_mass"]),
            "critical_phi_c": abs(vertex_phi - published["macedo_critical_phi_c"]),
        },
        "acceptance": {
            "all_nodeless": all(row["nodeless"] for row in rows),
            "all_solvers_converged": all(row["solver_status"] == 0 for row in rows),
            "maximum_mass_within_5e-4": abs(vertex_mass - published["kaup_maximum_mass"]) < 5.0e-4,
            "critical_phi_c_within_5e-4": abs(vertex_phi - published["macedo_critical_phi_c"]) < 5.0e-4,
        },
        "sources": {
            "polar_qnm_and_critical_density": "https://arxiv.org/abs/1603.02095",
            "background_and_polar_equations": "https://arxiv.org/abs/1307.4812",
        },
    }
    report_dir = ROOT / "reports" / "ordinary_ell0"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "background_turning_point_scan.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "background_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["turning_point"] | result["absolute_errors"], indent=2))


if __name__ == "__main__":
    main()
