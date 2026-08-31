"""Pilot cached-quadtree Evans census on the declared physical sheet.

This is a local validation of the census machinery, not the global spectral
gate.  The finite census window lies strictly below the real-axis sideband
cuts and contains the previously certified long-lived pole.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_spectrum import quadtree_census
from nonradial.riemann_sheet import SidebandSheet
from radial.coefficients import RadialBackground


def _json_cell(value):
    if isinstance(value, complex):
        return {"real": value.real, "imaginary": value.imag}
    if isinstance(value, tuple):
        return [_json_cell(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_cell(item) for key, item in value.items()}
    return value


def main() -> None:
    solution = BackgroundSolution.load(
        ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz"
    )
    background = RadialBackground(solution, geometry_representation="hermite")
    sheet = SidebandSheet.physical_lower_half_plane()
    bounds = {
        "real": (0.04930, 0.04950),
        "imaginary": (-1.5e-6, -1.0e-7),
    }
    matching = {
        "r_match": 14.0,
        "r_end": 35.0,
        "r_far": 300.0,
        "asymptotic_order": 3,
        "rtol": 2.0e-7,
        "atol": 2.0e-9,
    }
    census = quadtree_census(
        background,
        real_bounds=bounds["real"],
        imaginary_bounds=bounds["imaginary"],
        sheet=sheet,
        matching_options=matching,
        initial_points_per_edge=6,
        maximum_phase_step=0.25 * np.pi,
        maximum_depth=8,
    )
    result = {
        "calculation": "pilot cached-quadtree Evans census",
        "gate_scope": "local machinery validation; not a complete-spectrum claim",
        "background": {
            "a1_0": solution.a0,
            "omega": solution.omega,
            "adm_mass": solution.adm_mass,
        },
        "sheet": {
            "name": sheet.name,
            "plus_sign": sheet.plus_sign,
            "minus_sign": sheet.minus_sign,
            "branch_points": sheet.branch_points(solution.omega),
            "cut_convention": "finite real segments joining each branch-point pair",
        },
        "bounds": bounds,
        "matching": matching,
        "counted_zeros": census.counted_zeros,
        "assigned_poles": [_json_cell(pole) for pole in census.assigned_poles],
        "complete_inside_declared_cut_free_window": census.complete,
        "determinant_evaluations": census.determinant_evaluations,
        "excluded_cut_cells": [_json_cell(asdict(cell)) for cell in census.excluded_cut_cells],
        "leaves": [_json_cell(asdict(leaf)) for leaf in census.leaves],
    }
    target = ROOT / "reports" / "axial" / "axial_quadtree_census_pilot.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("counted_zeros", "assigned_poles", "complete_inside_declared_cut_free_window", "determinant_evaluations")}, indent=2))


if __name__ == "__main__":
    main()
