"""Regenerate cross-method radial certification and uncertainty records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy

from background.ell_boson_star import solve_by_continuation
from .bvp import solve_radial_bvp
from .mode_tracking import track_mode_by_overlap
from .spectral import solve_radial_spectrum

PIN_FILE = Path("environment/requirements-pins.txt")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _provenance(solver_version: str) -> dict[str, object]:
    pin_sha = hashlib.sha256(PIN_FILE.read_bytes()).hexdigest()
    tracked_status = _git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        ".",
        ":!reports/radial/radial_benchmarks.csv",
        ":!reports/radial/radial_uncertainty.json",
        ":!reports/radial/radial_overtone_uncertainty.json",
    )
    return {
        "implementation_commit": _git("rev-parse", "HEAD"),
        "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
        "implementation_worktree_dirty_excluding_outputs": bool(tracked_status),
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


def _bvp_row(
    provenance, background, r_max, *, mode_index=0, representation="hermite"
):
    guesses = {
        0.05: (3.8e-4, -1.92e-2),
        0.08: (2.4e-4, -2.78e-2),
        0.10: (4.0e-6, -3.33e-2),
        0.105: (-7.1e-5, -3.47e-2),
    }
    sigma_guess, center_guess = guesses[round(background.a0, 3)]
    if mode_index == 1:
        if not np.isclose(background.a0, 0.08):
            raise ValueError("the overtone certification is defined for a0=0.08")
        sigma_guess, center_guess = 8.2272e-3, -4.19e-2
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
            "mode_index": mode_index,
            "node_count": mode.node_count,
            "sigma2": mode.sigma2,
            "signed_sigma": _signed_frequency(mode.sigma2),
            "center_c": mode.center_c,
            "physical_F1": mode.physical_boundary_residual[0],
            "physical_F2": mode.physical_boundary_residual[1],
            "bvp_tolerance": 3e-7,
            "max_scipy_interval_rms_relative_residual": mode.max_scipy_interval_rms_relative_residual,
            "max_dense_pointwise_relative_residual": mode.max_dense_pointwise_relative_residual,
            "scaled_generalized_eigen_residual": "",
            "unscaled_generalized_eigen_residual": "",
            "eigenvalue_condition_number": "",
            "tracking_overlap": "",
            "nodes_used": mode.nodes_used,
            "solver_status": "converged" if mode.success else "failed",
        }
    )
    return row, mode


def _spectral_rows(
    provenance,
    background,
    points,
    *,
    r_max=40.0,
    include_overtone=False,
    references=None,
):
    modes = solve_radial_spectrum(
        background,
        points=points,
        epsilon=1e-3,
        r_max=r_max,
        sigma2_min=-1e-3,
        sigma2_max=0.02,
    )
    selected = []
    for mode_index in range(2 if include_overtone else 1):
        overlap = ""
        if references and mode_index in references:
            mode, overlap = track_mode_by_overlap(references[mode_index], modes)
        else:
            mode = next(mode for mode in modes if mode.node_count == mode_index)
        row = _base_row(
            provenance,
            background,
            formulation="chebyshev_generalized_eigenproblem",
            representation="hermite",
            r_max=r_max,
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
                "scaled_generalized_eigen_residual": mode.scaled_generalized_residual,
                "unscaled_generalized_eigen_residual": mode.unscaled_generalized_residual,
                "eigenvalue_condition_number": mode.eigenvalue_condition_number,
                "tracking_overlap": overlap,
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
    parser.add_argument(
        "--overtone-uncertainty-output",
        type=Path,
        default=Path("reports/radial/radial_overtone_uncertainty.json"),
    )
    parser.add_argument("--solver-version", default="radial-v0.3.1-hardened")
    args = parser.parse_args()

    provenance = _provenance(args.solver_version)
    backgrounds = {a0: solve_by_continuation(1, a0) for a0 in (0.05, 0.08, 0.10, 0.105)}
    rows = []
    bvp_modes = {}
    for a0, r_max in ((0.08, 25.0), (0.08, 30.0), (0.08, 40.0), (0.05, 40.0), (0.10, 40.0), (0.105, 40.0)):
        row, mode = _bvp_row(provenance, backgrounds[a0], r_max)
        rows.append(row)
        bvp_modes[a0, r_max] = mode

    overtone_bvp_modes = {}
    for r_max in (40.0, 50.0, 60.0):
        row, mode = _bvp_row(
            provenance, backgrounds[0.08], r_max, mode_index=1
        )
        rows.append(row)
        overtone_bvp_modes[r_max] = mode

    spectral_modes = {}
    spectral_overtones = {}
    tracking_references = None
    for points in (50, 60, 80, 100, 120, 160):
        selected = _spectral_rows(
            provenance,
            backgrounds[0.08],
            points,
            include_overtone=True,
            references=tracking_references,
        )
        rows.extend(row for row, _ in selected)
        spectral_modes[points] = selected[0][1]
        spectral_overtones[points] = selected[1][1]
        if tracking_references is None:
            tracking_references = {0: selected[0][1], 1: selected[1][1]}
    r60_selected = _spectral_rows(
        provenance,
        backgrounds[0.08],
        160,
        r_max=60.0,
        include_overtone=True,
    )
    rows.extend(row for row, _ in r60_selected)
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
        "spectral_resolution_max_displacement_from_bvp_n50_to_n160": max(
            abs(mode.sigma2 - reference) for mode in spectral_modes.values()
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
        "headline_uncertainty": "conservative deterministic-systematics sum",
        "conservative_sum_absolute_sigma2": float(sum(components.values())),
        "quadrature_sum_absolute_sigma2_secondary_only": float(
            np.linalg.norm(list(components.values()))
        ),
    }
    args.uncertainty_output.parent.mkdir(parents=True, exist_ok=True)
    args.uncertainty_output.write_text(json.dumps(uncertainty, indent=2) + "\n", encoding="utf-8")
    overtone_reference = overtone_bvp_modes[60.0].sigma2
    overtone_components = {
        "outer_domain_r40_to_r60": abs(
            overtone_bvp_modes[40.0].sigma2 - overtone_reference
        ),
        "bvp_to_spectral_at_r60_n160": abs(
            r60_selected[1][1].sigma2 - overtone_reference
        ),
        "spectral_resolution_max_displacement_from_r40_bvp_n50_to_n160": max(
            abs(mode.sigma2 - overtone_bvp_modes[40.0].sigma2)
            for mode in spectral_overtones.values()
        ),
    }
    overtone_uncertainty = {
        "provenance": provenance,
        "quantity": "first-overtone radial sigma2 at a0=0.08, epsilon=1e-3",
        "reference_method": "Hermite-background nonlinear global BVP",
        "reference_r_max": 60.0,
        "reference_sigma2": overtone_reference,
        "reference_sigma": float(np.sqrt(overtone_reference)),
        "components_absolute_sigma2": overtone_components,
        "headline_uncertainty": "conservative deterministic-systematics sum",
        "conservative_sum_absolute_sigma2": float(sum(overtone_components.values())),
        "quadrature_sum_absolute_sigma2_secondary_only": float(
            np.linalg.norm(list(overtone_components.values()))
        ),
    }
    args.overtone_uncertainty_output.parent.mkdir(parents=True, exist_ok=True)
    args.overtone_uncertainty_output.write_text(
        json.dumps(overtone_uncertainty, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} cross-method rows to {args.output}")
    print(f"wrote uncertainty record to {args.uncertainty_output}")
    print(f"wrote overtone uncertainty record to {args.overtone_uncertainty_output}")


if __name__ == "__main__":
    main()
