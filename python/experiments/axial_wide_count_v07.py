"""Cross-determinant wide-window axial count for v0.7 development."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import matching_evans_determinant, matching_raw_determinant
from nonradial.axial_spectrum import count_modes_adaptive
from nonradial.riemann_sheet import SidebandSheet
from radial.coefficients import RadialBackground


def main() -> None:
    solution = BackgroundSolution.load(
        ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz"
    )
    background = RadialBackground(solution, geometry_representation="hermite")
    sheet = SidebandSheet.physical_lower_half_plane()
    bounds = {"real": (0.005, 0.8), "imaginary": (-0.2, -1.0e-5)}
    matching = {
        "sheet": sheet,
        "r_match": 14.0,
        "r_end": 35.0,
        "r_far": 100.0,
        "asymptotic_order": 2,
        "rtol": 2.0e-6,
        "atol": 2.0e-8,
    }
    common = {
        "real_bounds": bounds["real"],
        "imaginary_bounds": bounds["imaginary"],
        "initial_points_per_edge": 5,
        "maximum_phase_step": 0.25 * np.pi,
        "maximum_refinements": 9,
        "minimum_uniform_refinements": 2,
        "required_stable_refinements": 2,
        "matching_options": matching,
    }
    raw = count_modes_adaptive(
        background, determinant=matching_raw_determinant, **common
    )
    evans = count_modes_adaptive(
        background, determinant=matching_evans_determinant, **common
    )
    passed = bool(
        raw["winding_number"] == evans["winding_number"]
        and raw["phase_resolution_pass"]
        and evans["phase_resolution_pass"]
        and raw["count_stability_pass"]
        and evans["count_stability_pass"]
    )
    result = {
        "calculation": "v0.7 wide-window cross-determinant boundary count",
        "claim_boundary": (
            "finite physical-lower-sheet rectangle only; pole assignment, "
            "threshold keyholes, other sheets, and high-frequency complement open"
        ),
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
        "bounds": bounds,
        "matching": {key: value for key, value in matching.items() if key != "sheet"},
        "raw_determinant": raw,
        "exterior_algebra_evans_determinant": evans,
        "cross_determinant_count_pass": passed,
        "v07_complete_spectrum_gate": False,
    }
    target = ROOT / "reports" / "axial" / "axial_wide_count_v07.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
