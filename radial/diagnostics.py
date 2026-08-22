"""Regenerate cross-method radial certification and uncertainty records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from background.ell_boson_star import solve_by_continuation
from .bvp import solve_radial_bvp
from .spectral import solve_radial_spectrum

PIN_FILE = Path("environment/requirements-pins.txt")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _provenance(solver_version: str) -> dict[str, object]:
    pin_sha = hashlib.sha256(PIN_FILE.read_bytes()).hexdigest()
    return {
        "implementation_commit": _git("rev-parse", "HEAD"),
        "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
        "working_tree_dirty_at_start": bool(_git("status", "--porcelain")),
        "solver_version": solver_version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "version_pin_file": PIN_FILE.as_posix(),
        "version_pin_sha256": pin_sha,
    }


def _signed_frequency(sigma2: float) -> float:
    return float(np.copysign(np.sqrt(abs(sigma2)), sigma2))


def _base_row(provenance, background, *, formulation, representation, r_max, resolution):
    return {
        **provenance,
        "formulation": formulation,
        "background_representation": representation,
        "a0": background.a0,
        "omega": background.omega,
        "alpha_c": background.alpha[0],
        "epsilon": 1e-3,
        "r_max": r_max,
        "resolution": resolution,
    }


def _bvp_row(provenance, background, r_max, *, representation="hermite"):
    guesses = {
        0.05: (3.8e-4, -1.92e-2),
        0.08: (2.4e-4, -2.78e-2),
        0.10: (4.0e-6, -3.33e-2),
        0.105: (-7.1e-5, -3.47e-2),
    }
    sigma_guess, center_guess = guesses[round(background.a0, 3)]
    mode = solve_radial_bvp(
        background,
        sigma2_guess=sigma_guess,
        center_c_guess=center_guess,
        epsilon=1e-3,
        r_max=r_max,
        points=350,
        tolerance=3e-7,
        background_representation=representation,
    )
    row = _base_row(
        provenance,
        background,
        formulation="nonlinear_global_bvp",
        representation=representation,
        r_max=r_max,
        resolution="350 initial / adaptive",
    )
    row.update(
        {
            "mode_index": 0,
            "node_count": mode.node_count,
            "sigma2": mode.sigma2,
            "signed_sigma": _signed_frequency(mode.sigma2),
            "center_c": mode.center_c,
            "physical_F1": mode.physical_boundary_residual[0],
            "physical_F2": mode.physical_boundary_residual[1],
            "bvp_tolerance": 3e-7,
            "max_scipy_interval_rms_relative_residual": mode.max_scipy_interval_rms_relative_residual,
            "max_dense_pointwise_relative_residual": mode.max_dense_pointwise_relative_residual,
            "generalized_eigen_residual": "",
            "nodes_used": mode.nodes_used,
            "solver_status": "converged" if mode.success else "failed",
        }
    )
    return row, mode


def _spectral_rows(provenance, background, points, *, include_overtone=False):
    modes = solve_radial_spectrum(
        background,
        points=points,
        epsilon=1e-3,
        r_max=40.0,
        sigma2_min=-1e-3,
        sigma2_max=0.02,
    )
    selected = []
    for mode_index in range(2 if include_overtone else 1):
        mode = next(mode for mode in modes if mode.node_count == mode_index)
        row = _base_row(
            provenance,
            background,
            formulation="chebyshev_generalized_eigenproblem",
            representation="hermite",
            r_max=40.0,
            resolution=f"{points} Chebyshev-Lobatto",
        )
        row.update(
            {
                "mode_index": mode_index,
                "node_count": mode.node_count,
                "sigma2": mode.sigma2,
                "signed_sigma": _signed_frequency(mode.sigma2),
                "center_c": "",
                "physical_F1": "",
                "physical_F2": "",
                "bvp_tolerance": "",
                "max_scipy_interval_rms_relative_residual": "",
                "max_dense_pointwise_relative_residual": "",
                "generalized_eigen_residual": mode.generalized_residual,
                "nodes_used": points,
                "solver_status": "converged",
            }
        )
        selected.append((row, mode))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uncertainty-output", type=Path, required=True)
    parser.add_argument("--solver-version", default="radial-v0.3-spectral")
    args = parser.parse_args()

    provenance = _provenance(args.solver_version)
    backgrounds = {a0: solve_by_continuation(1, a0) for a0 in (0.05, 0.08, 0.10, 0.105)}
    rows = []
    bvp_modes = {}
    for a0, r_max in ((0.08, 25.0), (0.08, 30.0), (0.08, 40.0), (0.05, 40.0), (0.10, 40.0), (0.105, 40.0)):
        row, mode = _bvp_row(provenance, backgrounds[a0], r_max)
        rows.append(row)
        bvp_modes[a0, r_max] = mode

    spectral_modes = {}
    for points in (60, 80, 100):
        selected = _spectral_rows(
            provenance, backgrounds[0.08], points, include_overtone=(points == 80)
        )
        rows.extend(row for row, _ in selected)
        spectral_modes[points] = selected[0][1]
    for a0 in (0.05, 0.10, 0.105):
        selected = _spectral_rows(provenance, backgrounds[a0], 80)
        rows.extend(row for row, _ in selected)

    pchip_row, pchip_mode = _bvp_row(
        provenance, backgrounds[0.08], 40.0, representation="pchip"
    )
    rows.append(pchip_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reference = bvp_modes[0.08, 40.0].sigma2
    components = {
        "outer_domain_r30_to_r40": abs(bvp_modes[0.08, 30.0].sigma2 - reference),
        "bvp_to_spectral_n80": abs(spectral_modes[80].sigma2 - reference),
        "spectral_resolution_n60_n80_n100": max(
            abs(spectral_modes[60].sigma2 - spectral_modes[80].sigma2),
            abs(spectral_modes[100].sigma2 - spectral_modes[80].sigma2),
        ),
        "hermite_to_pchip": abs(pchip_mode.sigma2 - reference),
    }
    uncertainty = {
        "provenance": provenance,
        "quantity": "ground radial sigma2 at a0=0.08, epsilon=1e-3",
        "reference_method": "Hermite-background nonlinear global BVP",
        "reference_r_max": 40.0,
        "reference_sigma2": reference,
        "components_absolute_sigma2": components,
        "combined_quadrature_absolute_sigma2": float(np.linalg.norm(list(components.values()))),
        "conservative_sum_absolute_sigma2": float(sum(components.values())),
    }
    args.uncertainty_output.parent.mkdir(parents=True, exist_ok=True)
    args.uncertainty_output.write_text(json.dumps(uncertainty, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} cross-method rows to {args.output}")
    print(f"wrote uncertainty record to {args.uncertainty_output}")


if __name__ == "__main__":
    main()
