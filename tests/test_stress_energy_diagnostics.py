import csv

import numpy as np

from background.ell_boson_star import solve_by_continuation
from experiments.stress_energy_diagnostics import stress_components


def test_stress_components_reproduce_mass_equation():
    solution = solve_by_continuation(1, 0.08)
    rho, _, _ = stress_components(solution)
    mass_derivative = np.gradient(solution.mass, solution.r, edge_order=2)
    support = rho > np.max(rho) * 1e-8
    relative = np.abs(mass_derivative - solution.r**2 * rho) / (
        1.0 + np.abs(solution.r**2 * rho)
    )
    assert np.max(relative[support]) < 2e-4


def test_sampled_family_satisfies_weak_and_dominant_energy_conditions():
    with open("reports/background/stress_energy_diagnostics.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 21
    for row in rows:
        assert float(row["min_rho_plus_pr_over_peak"]) >= -1e-10
        assert float(row["min_rho_plus_pt_over_peak"]) >= -1e-10
        assert float(row["min_rho_minus_abs_pr_over_peak"]) >= -1e-10
        assert float(row["min_rho_minus_abs_pt_over_peak"]) >= -1e-10
