"""Solve and audit the charged ell=1 EMKG equilibrium continuation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.charged_ell_boson_star import solve_charged_background
from background.ell_boson_star import BackgroundSolution


def row(solution) -> dict[str, float]:
    radius = solution.r
    flux = solution.electric_flux
    electromagnetic_mass = float(
        np.trapezoid(2.0 * np.pi * flux**2 / radius**2, radius)
        + 2.0 * np.pi * flux[-1] ** 2 / radius[-1]
    )
    f_outer = (
        1.0
        - 2.0 * solution.adm_mass / radius[-1]
        + 4.0 * np.pi * flux[-1] ** 2 / radius[-1] ** 2
    )
    gauss_expected = solution.gauge_charge * solution.noether_charge
    return {
        "gauge_charge_q": solution.gauge_charge,
        "q_over_sqrt_4pi": solution.gauge_charge / np.sqrt(4.0 * np.pi),
        "omega": solution.omega,
        "adm_mass": solution.adm_mass,
        "noether_charge": solution.noether_charge,
        "total_electric_charge": solution.total_electric_charge,
        "gauss_law_relative_error": float(
            abs(solution.total_electric_charge - gauss_expected)
            / max(abs(gauss_expected), 1e-300)
        ) if solution.gauge_charge else 0.0,
        "electric_potential_center": float(solution.electric_potential[0]),
        "electric_field_energy": electromagnetic_mass,
        "electric_field_energy_fraction": electromagnetic_mass / solution.adm_mass,
        "r99": solution.r99,
        "compactness99": solution.compactness99,
        "minimum_local_frequency": float(
            np.min(solution.omega - solution.gauge_charge * solution.electric_potential)
        ),
        "outer_RN_lapse_residual": float(abs(solution.alpha[-1] ** 2 - f_outer)),
        "outer_Coulomb_potential_residual": float(
            abs(solution.electric_potential[-1] - flux[-1] / radius[-1])
        ),
        "max_ode_residual": solution.max_ode_residual,
        "tail_residual": solution.tail_residual,
    }


def main() -> None:
    neutral = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial.npz")
    charges = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 3.4, 3.5, 3.52)
    rows = []
    current = None
    selected = {0.0, 1.0, 2.0, 3.0, 3.4, 3.5, 3.52}
    for charge in charges:
        current = solve_charged_background(
            1,
            0.08,
            charge,
            neutral_seed=neutral if current is None else None,
            seed=current,
            r_max=100.0,
            points=900,
            tolerance=2e-7,
            max_nodes=80000,
        )
        rows.append(row(current))
        if charge in selected:
            current.save(ROOT / "reports" / "extensions" / f"charged_ell1_a008_q{charge:.2f}.npz")

    # Outer-domain audit at a strong but noncritical coupling.
    seed_q3 = None
    current = None
    for charge in (0.0, 1.0, 2.0, 3.0):
        current = solve_charged_background(
            1,
            0.08,
            charge,
            neutral_seed=neutral if current is None else None,
            seed=current,
            r_max=100.0,
            points=900,
            tolerance=2e-7,
            max_nodes=80000,
        )
    seed_q3 = current
    domains = []
    for outer in (80.0, 100.0, 120.0):
        solved = solve_charged_background(
            1,
            0.08,
            3.0,
            seed=seed_q3,
            r_max=outer,
            points=900,
            tolerance=1e-7,
            max_nodes=80000,
        )
        domains.append(
            {
                "r_max": outer,
                "omega": solved.omega,
                "adm_mass": solved.adm_mass,
                "total_electric_charge": solved.total_electric_charge,
                "r99": solved.r99,
            }
        )
    result = {
        "calculation": "charged ell=1 a1^0=0.08 EMKG equilibrium continuation",
        "critical_coupling_sqrt_4pi": float(np.sqrt(4.0 * np.pi)),
        "sequence": rows,
        "domain_audit_q3": domains,
        "domain_spreads_q3": {
            key: max(item[key] for item in domains) - min(item[key] for item in domains)
            for key in ("omega", "adm_mass", "total_electric_charge", "r99")
        },
    }
    target = ROOT / "reports" / "extensions" / "charged_background_checkpoint.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["domain_spreads_q3"], indent=2))
    print(json.dumps(rows[-1], indent=2))


if __name__ == "__main__":
    main()
