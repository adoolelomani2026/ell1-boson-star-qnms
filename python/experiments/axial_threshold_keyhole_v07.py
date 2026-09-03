"""Resumable Coulomb-resummed minus-threshold keyhole audit for v0.7."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import os
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import matching_evans_determinant
from nonradial.axial_spectrum import annular_sector_contour
from nonradial.riemann_sheet import SidebandSheet
from radial.coefficients import RadialBackground


BACKGROUND = None
OMEGA = None
CONFIGURATION = None
SHEET = SidebandSheet.physical_lower_half_plane()
RADIAL_BOUNDS = (0.05, 0.4)
ANGULAR_BOUNDS = (float(np.pi / 2 + 0.05), float(np.pi - 0.05))
MAXIMUM_PHASE_STEP = float(np.pi / 16)


def initialize_worker(r_match: float, r_far: float) -> None:
    global BACKGROUND, OMEGA, CONFIGURATION
    solution = BackgroundSolution.load(
        ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz"
    )
    BACKGROUND = RadialBackground(solution, geometry_representation="hermite")
    OMEGA = solution.omega
    CONFIGURATION = (r_match, r_far)


def evaluate(point: complex) -> tuple[float, float, float, float]:
    r_match, r_far = CONFIGURATION
    sigma = SHEET.sigma_from_minus_wavenumber(point, OMEGA)
    value = matching_evans_determinant(
        sigma,
        BACKGROUND,
        sheet=SHEET,
        r_match=r_match,
        r_end=35.0,
        r_far=r_far,
        asymptotic_order=3,
        exterior_method="complex_scaled_coulomb",
        rtol=2e-8,
        atol=2e-10,
    )
    return point.real, point.imag, value.real, value.imag


def contour_statistics(points: list[complex], cache: dict[complex, complex]) -> dict:
    values = np.asarray([cache[point] for point in points])
    increments = np.angle(np.roll(values, -1) / values)
    return {
        "winding_number": int(np.rint(np.sum(increments) / (2 * np.pi))),
        "contour_points": len(points),
        "maximum_phase_increment": float(np.max(np.abs(increments))),
        "minimum_boundary_determinant": float(np.min(np.abs(values))),
        "violating_segments": int(np.count_nonzero(np.abs(increments) >= MAXIMUM_PHASE_STEP)),
    }


def load_cache(connection: sqlite3.Connection, name: str) -> dict[complex, complex]:
    rows = connection.execute(
        "SELECT point_real, point_imag, value_real, value_imag FROM samples WHERE configuration=?",
        (name,),
    )
    return {complex(x, y): complex(u, v) for x, y, u, v in rows}


def store_rows(connection: sqlite3.Connection, name: str, rows) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?)",
        ((name, *row) for row in rows),
    )
    connection.commit()


def audit_configuration(
    connection: sqlite3.Connection, r_match: float, r_far: float
) -> dict:
    name = f"match-{r_match:g}-far-{r_far:g}"
    cache = load_cache(connection, name)
    workers = min(6, os.cpu_count() or 1)
    uniform_history = []
    adaptive_history = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(r_match, r_far),
    ) as pool:
        for count in (16, 32, 64):
            points = list(
                map(
                    complex,
                    annular_sector_contour(RADIAL_BOUNDS, ANGULAR_BOUNDS, count),
                )
            )
            missing = list(dict.fromkeys(point for point in points if point not in cache))
            rows = list(pool.map(evaluate, missing, chunksize=1))
            store_rows(connection, name, rows)
            cache.update({complex(x, y): complex(u, v) for x, y, u, v in rows})
            statistics = contour_statistics(points, cache)
            statistics["points_per_segment"] = count
            uniform_history.append(statistics)
            print(name, "uniform", statistics, flush=True)

        for level in range(15):
            missing = list(dict.fromkeys(point for point in points if point not in cache))
            rows = list(pool.map(evaluate, missing, chunksize=1))
            store_rows(connection, name, rows)
            cache.update({complex(x, y): complex(u, v) for x, y, u, v in rows})
            statistics = contour_statistics(points, cache)
            statistics["level"] = level
            adaptive_history.append(statistics)
            print(name, "adaptive", statistics, flush=True)
            if statistics["violating_segments"] == 0:
                break
            values = np.asarray([cache[point] for point in points])
            increments = np.angle(np.roll(values, -1) / values)
            flagged = set(np.flatnonzero(np.abs(increments) >= MAXIMUM_PHASE_STEP))
            refined = []
            for index, point in enumerate(points):
                refined.append(point)
                if index in flagged:
                    refined.append(0.5 * (point + points[(index + 1) % len(points)]))
            points = refined

    final = adaptive_history[-1]
    return {
        "name": name,
        "r_match": r_match,
        "r_far": r_far,
        "uniform_history": uniform_history,
        "adaptive_history": adaptive_history,
        "winding_number": final["winding_number"],
        "contour_points": final["contour_points"],
        "maximum_phase_increment": final["maximum_phase_increment"],
        "minimum_boundary_determinant": final["minimum_boundary_determinant"],
        "phase_resolution_pass": final["violating_segments"] == 0,
        "determinant_evaluations": len(cache),
    }


def main() -> None:
    cache_path = ROOT / "tmp" / "axial_threshold_keyhole_v07.sqlite3"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS samples ("
            "configuration TEXT, point_real REAL, point_imag REAL, "
            "value_real REAL, value_imag REAL, "
            "PRIMARY KEY(configuration, point_real, point_imag))"
        )
        configurations = [
            audit_configuration(connection, 14.0, 300.0),
            audit_configuration(connection, 14.0, 400.0),
        ]

    counts = [row["winding_number"] for row in configurations]
    phase_pass = all(row["phase_resolution_pass"] for row in configurations)
    ray_count_stable = len(set(counts)) == 1
    report = {
        "calculation": "v0.7 Coulomb-resummed minus-threshold keyhole",
        "sheet": SHEET.name,
        "radial_bounds_k_minus": RADIAL_BOUNDS,
        "angular_bounds_k_minus": ANGULAR_BOUNDS,
        "maximum_phase_step": MAXIMUM_PHASE_STEP,
        "exterior_method": "complex_scaled_coulomb",
        "configurations": configurations,
        "acceptance": {
            "all_contours_phase_resolved": phase_pass,
            "ray_count_stable": ray_count_stable,
            "zero_count_requires_no_root_assignment": counts == [0, 0],
        },
        "checkpoint_pass": bool(phase_pass and ray_count_stable and counts == [0, 0]),
        "classification": (
            "threshold sector contains no QNM zeros"
            if phase_pass and counts == [0, 0]
            else "root assignment remains required"
        ),
    }
    output = ROOT / "reports" / "axial" / "axial_threshold_keyhole_v07.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
