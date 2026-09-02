"""Certify the isolated axial pole with complex scaling and stabilized planes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import (
    matching_evans_determinant,
    matching_raw_determinant,
    matching_singular_value,
)
from nonradial.axial_spectrum import rectangular_contour, refine_scaled_root, winding_number
from nonradial.riemann_sheet import SidebandSheet
from radial.coefficients import RadialBackground


_COUNT_BACKGROUND = None


def initialize_count_worker() -> None:
    global _COUNT_BACKGROUND
    solution = BackgroundSolution.load(
        ROOT / "reports/axial/background_ell1_a008_axial_refined.npz"
    )
    _COUNT_BACKGROUND = RadialBackground(
        solution, geometry_representation="hermite"
    )


def evaluate_count_point(point: complex) -> tuple[complex, complex]:
    value = matching_evans_determinant(
        point,
        _COUNT_BACKGROUND,
        sheet=SidebandSheet.physical_lower_half_plane(),
        r_match=14.0,
        r_end=35.0,
        r_far=300.0,
        asymptotic_order=3,
        exterior_method="complex_scaled",
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    return point, value


def parallel_local_count() -> dict[str, object]:
    """Resolve the local contour while caching samples across six workers."""

    cache: dict[complex, complex] = {}
    uniform_history = []
    bounds = ((0.0493, 0.0495), (-1.5e-6, -1.0e-7))
    with ProcessPoolExecutor(
        max_workers=min(6, os.cpu_count() or 1), initializer=initialize_count_worker
    ) as pool:
        for points_per_edge in (16, 32, 64, 128):
            contour = [
                complex(point)
                for point in rectangular_contour(*bounds, points_per_edge)
            ]
            missing = [point for point in contour if point not in cache]
            for point, value in pool.map(evaluate_count_point, missing, chunksize=1):
                cache[point] = value
            values = np.asarray([cache[point] for point in contour])
            winding, maximum = winding_number(values)
            uniform_history.append(
                {
                    "points_per_edge": points_per_edge,
                    "winding_number": winding,
                    "maximum_phase_increment": maximum,
                }
            )

        points = contour
        adaptive_history = []
        for level in range(11):
            values = np.asarray([cache[point] for point in points])
            increments = np.angle(np.roll(values, -1) / values)
            violating = np.flatnonzero(np.abs(increments) >= np.pi / 16.0)
            winding = int(np.rint(np.sum(increments) / (2.0 * np.pi)))
            adaptive_history.append(
                {
                    "level": level,
                    "contour_points": len(points),
                    "winding_number": winding,
                    "maximum_phase_increment": float(np.max(np.abs(increments))),
                }
            )
            if not len(violating):
                break
            flagged = set(map(int, violating))
            refined = []
            for index, point in enumerate(points):
                refined.append(point)
                if index in flagged:
                    refined.append(0.5 * (point + points[(index + 1) % len(points)]))
            points = refined
            missing = [point for point in points if point not in cache]
            for point, value in pool.map(evaluate_count_point, missing, chunksize=1):
                cache[point] = value

    final = adaptive_history[-1]
    return {
        "real_bounds": bounds[0],
        "imaginary_bounds": bounds[1],
        "uniform_history": uniform_history,
        "adaptive_history": adaptive_history,
        "winding_number": final["winding_number"],
        "final_contour_points": final["contour_points"],
        "maximum_phase_increment": final["maximum_phase_increment"],
        "minimum_boundary_determinant": float(
            min(abs(cache[point]) for point in points)
        ),
        "phase_resolution_pass": final["maximum_phase_increment"] < np.pi / 16.0,
        "count_stability_pass": len(
            {
                row["winding_number"]
                for row in uniform_history[-2:] + adaptive_history[-2:]
            }
        )
        == 1,
        "determinant_evaluations": len(cache),
    }


def refinement_record(result) -> dict[str, object]:
    return {
        "pole_real": result.pole.real,
        "pole_imag": result.pole.imag,
        "relative_residual": result.relative_residual,
        "converged": result.converged,
        "iterations": result.iterations,
        "evaluations": result.evaluations,
    }


def main() -> None:
    solution = BackgroundSolution.load(
        ROOT / "reports/axial/background_ell1_a008_axial_refined.npz"
    )
    background = RadialBackground(solution, geometry_representation="hermite")
    sheet = SidebandSheet.physical_lower_half_plane()
    seed = 0.0493977304 - 5.96519e-7j
    rows = []
    for r_match in (12.0, 14.0):
        for r_far in (300.0, 400.0):
            options = {
                "sheet": sheet,
                "r_match": r_match,
                "r_end": 35.0,
                "r_far": r_far,
                "asymptotic_order": 3,
                "exterior_method": "complex_scaled",
                "rtol": 2.0e-8,
                "atol": 2.0e-10,
            }
            evans = refine_scaled_root(
                seed,
                background,
                matching_options=options,
                determinant=matching_evans_determinant,
                derivative_step=2.0e-8,
                step_tolerance=2.0e-13,
                residual_tolerance=1.0e-5,
                maximum_iterations=4,
            )
            raw = refine_scaled_root(
                evans.pole,
                background,
                matching_options=options,
                determinant=matching_raw_determinant,
                derivative_step=2.0e-8,
                step_tolerance=2.0e-13,
                residual_tolerance=1.0e-5,
                maximum_iterations=4,
            )
            singular_value = matching_singular_value(
                evans.pole, background, **options
            )
            rows.append(
                {
                    "r_match": r_match,
                    "r_end": 35.0,
                    "ray_endpoint_control": r_far,
                    "evans_refinement": refinement_record(evans),
                    "direct_determinant_refinement": refinement_record(raw),
                    "cross_determinant_pole_difference": abs(raw.pole - evans.pole),
                    "orthonormal_subspace_singular_value": singular_value,
                }
            )
            print(
                f"completed r_match={r_match:g}, ray_control={r_far:g}",
                flush=True,
            )
            seed = evans.pole

    poles = np.asarray(
        [
            complex(
                row["evans_refinement"]["pole_real"],
                row["evans_refinement"]["pole_imag"],
            )
            for row in rows
        ]
    )
    print("counting the local contour", flush=True)
    local_count = parallel_local_count()
    acceptance = {
        "all_refinements_converged": all(
            row["evans_refinement"]["converged"]
            and row["direct_determinant_refinement"]["converged"]
            for row in rows
        ),
        "orthonormal_rank_loss_confirmed": max(
            row["orthonormal_subspace_singular_value"] for row in rows
        )
        < 1.0e-7,
        "matching_and_ray_controls_stable": bool(
            np.ptp(poles.real) < 1.0e-7 and np.ptp(poles.imag) < 1.0e-8
        ),
        "cross_determinant_agreement": max(
            row["cross_determinant_pole_difference"] for row in rows
        )
        < 1.0e-7,
        "local_count_is_one": local_count["winding_number"] == 1,
        "local_contour_phase_resolved": local_count["phase_resolution_pass"],
        "local_count_stable": local_count["count_stability_pass"],
    }
    report = {
        "calculation": "v0.7 complex-scaled isolated-pole checkpoint",
        "background": {
            "a1_0": solution.a0,
            "omega": solution.omega,
            "adm_mass": solution.adm_mass,
        },
        "sheet": {
            "name": sheet.name,
            "signs": [sheet.plus_sign, sheet.minus_sign],
            "branch_points": sheet.branch_points(solution.omega),
        },
        "normalization_note": (
            "Positive real segment scalings preserve Evans phase and zeros; "
            "root refinement solves real and imaginary equations without a "
            "holomorphy assumption."
        ),
        "configuration_rows": rows,
        "evans_pole_spread": {
            "real": float(np.ptp(poles.real)),
            "imaginary": float(np.ptp(poles.imag)),
        },
        "local_contour": local_count,
        "acceptance": acceptance,
        "checkpoint_pass": all(acceptance.values()),
    }
    target = ROOT / "reports/axial/axial_complex_scaled_checkpoint_v07.json"
    target.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
