"""From-scratch grid discovery, raw-determinant refinement, and pole count."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_spectrum import count_modes_adaptive, discover_mode
from radial.coefficients import RadialBackground


def main() -> None:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    search_bounds = {
        "real": (0.0488, 0.0500),
        "imaginary": (-1.2e-6, -1.0e-7),
    }
    matching = {
        "r_match": 14.0,
        "r_end": 35.0,
        "r_far": 300.0,
        "rtol": 2e-7,
        "atol": 2e-9,
    }
    search, scan = discover_mode(
        background,
        real_bounds=search_bounds["real"],
        imaginary_bounds=search_bounds["imaginary"],
        scan_shape=(7, 5),
        matching_options=matching,
    )
    contour_bounds = {
        "real": (0.04936, 0.04944),
        "imaginary": (-1.1e-6, -1.0e-7),
    }
    adaptive_count = count_modes_adaptive(
        background,
        real_bounds=contour_bounds["real"],
        imaginary_bounds=contour_bounds["imaginary"],
        initial_points_per_edge=8,
        matching_options=matching,
    )
    strict_count = count_modes_adaptive(
        background,
        real_bounds=contour_bounds["real"],
        imaginary_bounds=contour_bounds["imaginary"],
        initial_points_per_edge=8,
        maximum_phase_step=0.25 * np.pi,
        matching_options=matching,
    )
    expanded_contour_bounds = {
        "real": (0.04930, 0.04950),
        "imaginary": (-1.5e-6, 1.0e-7),
    }
    expanded_count = count_modes_adaptive(
        background,
        real_bounds=expanded_contour_bounds["real"],
        imaginary_bounds=expanded_contour_bounds["imaginary"],
        initial_points_per_edge=8,
        maximum_phase_step=0.25 * np.pi,
        matching_options=matching,
    )
    result = {
        "calculation": "from-scratch neutral axial QNM discovery and argument-principle count",
        "search_bounds": search_bounds,
        "scan_shape": [7, 5],
        "seed": {"real": search.seed.real, "imaginary": search.seed.imag},
        "pole": {
            "real": search.pole.real,
            "imaginary": search.pole.imag,
            "quality_factor": search.pole.real / (-2.0 * search.pole.imag),
        },
        "refinement": {
            "success": search.solver_success,
            "function_evaluations": search.evaluations,
            "conditioned_matching_determinant_residual_relative_to_local_offset": search.raw_residual_relative,
        },
        "contour_bounds": contour_bounds,
        "adaptive_contour_count": adaptive_count,
        "strict_pi_over_4_contour_count": strict_count,
        "expanded_local_contour_bounds": expanded_contour_bounds,
        "expanded_local_contour_count": expanded_count,
        "contour_acceptance": {
            "criterion": "stable winding and maximum phase increment < pi/2",
            "pass": bool(
                adaptive_count["winding_number"]
                == adaptive_count["initial_winding_number"]
                and adaptive_count["phase_resolution_pass"]
                and strict_count["winding_number"] == 1
                and strict_count["phase_resolution_pass"]
                and expanded_count["winding_number"] == 1
                and expanded_count["phase_resolution_pass"]
            ),
        },
        "matching": matching,
        "scan": scan,
    }
    target = ROOT / "reports" / "axial" / "axial_qnm_discovery.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"pole": result["pole"], "adaptive_contour_count": adaptive_count},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
