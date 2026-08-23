"""Evaluate the unused tA Einstein equation on the matched axial mode."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import (
    axial_ta_constraint_profile_chain_rule,
    interior_mode_profile,
    matching_mode_coefficients,
)
from radial.coefficients import RadialBackground


def main() -> None:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    checkpoint = json.loads((ROOT / "reports" / "axial" / "axial_qnm_checkpoint.json").read_text())
    sigma = complex(
        checkpoint["matching_setups"][0]["sigma_real"],
        checkpoint["matching_setups"][0]["sigma_imag"],
    )
    coefficients, _, singular = matching_mode_coefficients(
        sigma,
        background,
        r_match=14.0,
        r_end=35.0,
        r_far=600.0,
        asymptotic_order=3,
        rtol=2e-8,
        atol=2e-10,
    )
    rows = []
    production_radii = None
    production_states = None
    production_residual = None
    production_scale = None
    for points in (250, 500, 1000):
        radii = np.linspace(0.02, 14.0, points)
        states = interior_mode_profile(
            radii, sigma, background, coefficients, rtol=2e-9, atol=2e-11
        )
        residual, scale = axial_ta_constraint_profile_chain_rule(
            radii,
            states,
            sigma,
            background,
            derivative_step_multiplier=112.0,
            derivative_method="richardson",
        )
        relative = np.abs(residual[3:-3]) / scale[3:-3]
        rows.append(
            {
                "points": points,
                "spacing": float(radii[1] - radii[0]),
                "relative_l2": float(np.linalg.norm(residual[3:-3]) / np.linalg.norm(scale[3:-3])),
                "relative_median": float(np.median(relative)),
                "relative_p95": float(np.quantile(relative, 0.95)),
                "relative_linf": float(np.max(relative)),
            }
        )
        if points == 500:
            production_radii = radii
            production_states = states
            production_residual = residual
            production_scale = scale
    assert production_radii is not None and production_states is not None
    derivative_step_rows = []
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 256.0):
        residual, scale = axial_ta_constraint_profile_chain_rule(
            production_radii,
            production_states,
            sigma,
            background,
            derivative_step_multiplier=multiplier,
            derivative_method="fourth_order",
        )
        relative = np.abs(residual[3:-3]) / scale[3:-3]
        derivative_step_rows.append(
            {
                "derivative_step_multiplier": multiplier,
                "relative_l2": float(
                    np.linalg.norm(residual[3:-3]) / np.linalg.norm(scale[3:-3])
                ),
                "relative_median": float(np.median(relative)),
                "relative_p95": float(np.quantile(relative, 0.95)),
                "relative_linf": float(np.max(relative)),
            }
        )
    richardson_step_rows = []
    for multiplier in (32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0):
        residual, scale = axial_ta_constraint_profile_chain_rule(
            production_radii,
            production_states,
            sigma,
            background,
            derivative_step_multiplier=multiplier,
            derivative_method="richardson",
        )
        relative = np.abs(residual[3:-3]) / scale[3:-3]
        richardson_step_rows.append(
            {
                "derivative_step_multiplier": multiplier,
                "relative_l2": float(
                    np.linalg.norm(residual[3:-3]) / np.linalg.norm(scale[3:-3])
                ),
                "relative_median": float(np.median(relative)),
                "relative_p95": float(np.quantile(relative, 0.95)),
                "relative_linf": float(np.max(relative)),
            }
        )
    assert production_residual is not None and production_scale is not None
    profile_indices = np.linspace(3, len(production_radii) - 4, 120, dtype=int)
    residual_profile = [
        {
            "radius": float(production_radii[index]),
            "absolute_residual": float(abs(production_residual[index])),
            "normalization_scale": float(production_scale[index]),
            "relative_residual": float(
                abs(production_residual[index]) / production_scale[index]
            ),
        }
        for index in profile_indices
    ]
    result = {
        "calculation": "dependent tA Einstein-constraint evaluation on the axial mode",
        "derivative_representation": "EKG alpha second derivative and independent chain-rule derivative Y''=A'Y+A^2Y; production A' uses sixth-order Richardson extrapolation of a fourth-order centered stencil",
        "production_derivative_method": "richardson",
        "production_derivative_step_multiplier": 112.0,
        "sigma": {"real": sigma.real, "imaginary": sigma.imag},
        "matching_minimum_singular_value": singular,
        "radial_interval": [0.02, 14.0],
        "edge_points_excluded": 3,
        "resolution_rows": rows,
        "derivative_step_rows": derivative_step_rows,
        "richardson_step_rows": richardson_step_rows,
        "residual_profile_500_point_run": residual_profile,
    }
    target = ROOT / "reports" / "axial" / "axial_constraint_checkpoint.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
