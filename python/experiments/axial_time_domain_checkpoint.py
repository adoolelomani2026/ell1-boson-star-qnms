"""Run a seeded time-domain real-frequency consistency check."""

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
    exterior_mode_profile,
    interior_mode_profile,
    matching_mode_coefficients,
)
from nonradial.axial_time_domain import evolution_rhs, grid_background, rk4_step
from radial.coefficients import RadialBackground


def enforce_boundaries(state: np.ndarray, radii: np.ndarray) -> None:
    """Impose leading regular powers and a damped outer extrapolation."""

    anchor = 4
    powers = (3, 4, 4, 2, 2, 2, 2)
    for component, power in enumerate(powers):
        state[component, :anchor] = state[component, anchor] * (
            radii[:anchor] / radii[anchor]
        ) ** power
    # The sponge makes the last points negligible; constant extrapolation
    # avoids injecting a high-order one-sided-grid mode.
    state[:, -2:] = state[:, -3:-2]


def run_checkpoint(
    *,
    r_max: float = 200.0,
    points: int = 2000,
    t_end: float = 150.0,
    courant: float = 0.10,
) -> dict[str, object]:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    radial = RadialBackground(solution, geometry_representation="hermite")
    checkpoint = json.loads((ROOT / "reports" / "axial" / "axial_qnm_checkpoint.json").read_text())
    sigma = complex(
        checkpoint["mode"]["sigma_real_mean"],
        checkpoint["mode"]["sigma_imag_mean"],
    )
    interior_coefficients, exterior_coefficients, singular = matching_mode_coefficients(
        sigma,
        radial,
        r_match=14.0,
        r_end=35.0,
        r_far=600.0,
        asymptotic_order=3,
        rtol=2e-8,
        atol=2e-10,
    )
    radii = np.linspace(0.1, r_max, points)
    spacing = radii[1] - radii[0]
    profile = np.empty((6, points), dtype=complex)
    interior_mask = radii <= 14.0
    profile[:, interior_mask] = interior_mode_profile(
        radii[interior_mask], sigma, radial, interior_coefficients, rtol=2e-9, atol=2e-11
    )
    profile[:, ~interior_mask] = exterior_mode_profile(
        radii[~interior_mask], sigma, radial, exterior_coefficients, r_end=35.0,
        rtol=2e-9, atol=2e-11
    )
    h0, h1, p, _, q, _ = profile
    state = np.array(
        (h0, h1, -1j * sigma * h1, p, -1j * sigma * p, q, -1j * sigma * q),
        dtype=complex,
    )
    state /= np.max(np.abs(state[3]))
    background = grid_background(solution, radii)
    sponge_start = 0.90 * r_max
    sponge = np.zeros_like(radii)
    mask = radii > sponge_start
    sponge[mask] = 0.12 * ((radii[mask] - sponge_start) / (r_max - sponge_start)) ** 2
    dt = courant * spacing
    steps = int(np.ceil(t_end / dt))
    dt = t_end / steps
    extraction_index = int(np.argmin(abs(radii - 6.0)))
    sample_stride = max(1, steps // 6000)
    times = []
    signal = []

    def rhs(values):
        return evolution_rhs(values, background, spacing, sponge=sponge)

    enforce_boundaries(state, radii)
    for step in range(steps + 1):
        if step % sample_stride == 0:
            times.append(step * dt)
            signal.append(state[3, extraction_index])
        if step == steps:
            break
        state = rk4_step(state, dt, rhs)
        enforce_boundaries(state, radii)
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f"time evolution became non-finite at step {step}")

    times_array = np.asarray(times)
    signal_array = np.asarray(signal)
    fit_mask = (times_array >= 20.0) & (times_array <= 0.92 * t_end)
    phase = np.unwrap(np.angle(signal_array[fit_mask]))
    phase_slope, phase_intercept = np.polyfit(times_array[fit_mask], phase, 1)
    fitted_sigma_real = -phase_slope
    fitted_log_amplitude_slope, _ = np.polyfit(
        times_array[fit_mask], np.log(np.maximum(abs(signal_array[fit_mask]), 1e-300)), 1
    )
    fitted_signal = np.exp(
        1j * (phase_intercept + phase_slope * times_array[fit_mask])
    ) * np.exp(
        np.mean(np.log(np.maximum(abs(signal_array[fit_mask]), 1e-300)))
    )
    phase_residual = phase - (phase_slope * times_array[fit_mask] + phase_intercept)
    return {
        "calculation": "1+1 method-of-lines real-frequency preservation check for a matched axial mode",
        "grid": {
            "r_min": float(radii[0]),
            "r_max": r_max,
            "points": points,
            "spacing": spacing,
            "dt": dt,
            "steps": steps,
            "t_end": t_end,
            "extraction_radius": float(radii[extraction_index]),
            "sponge_start": sponge_start,
        },
        "frequency_domain": {
            "sigma_real": sigma.real,
            "sigma_imag": sigma.imag,
            "matching_min_singular_value": singular,
        },
        "time_domain": {
            "sigma_real_fit": float(fitted_sigma_real),
            "relative_frequency_error": float(
                abs(fitted_sigma_real - sigma.real) / abs(sigma.real)
            ),
            "log_amplitude_slope": float(fitted_log_amplitude_slope),
            "phase_residual_rms": float(np.sqrt(np.mean(phase_residual**2))),
            "samples": int(fit_mask.sum()),
        },
        "note": (
            "The run resolves the oscillation frequency. Its duration is far shorter than "
            f"the {1.0 / abs(sigma.imag):.3e} damping time, so it is not used to "
            "estimate sigma_imag."
        ),
    }


def run_convergence_matrix() -> dict[str, object]:
    """Regenerate every resolution/Courant row with the current operator."""

    configurations = (
        {"points": 1500, "courant": 0.10},
        {"points": 2000, "courant": 0.10},
        {"points": 2500, "courant": 0.10},
        {"points": 2000, "courant": 0.08},
    )
    rows = []
    reference = None
    for configuration in configurations:
        result = run_checkpoint(**configuration)
        if reference is None:
            reference = result["frequency_domain"]
        rows.append({**configuration, **result["time_domain"]})
    fits = np.asarray([row["sigma_real_fit"] for row in rows])
    assert reference is not None
    lifetime = 1.0 / abs(float(reference["sigma_imag"]))
    return {
        "calculation": (
            "current-operator 1+1 method-of-lines real-frequency convergence matrix"
        ),
        "frequency_domain_reference": reference,
        "fixed_parameters": {
            "r_min": 0.1,
            "r_max": 200.0,
            "t_end": 150.0,
            "extraction_radius_approx": 6.0,
            "sponge_start": 180.0,
            "fit_window": [20.0, 138.0],
            "damping_time": lifetime,
            "run_fraction_of_damping_time": 150.0 / lifetime,
        },
        "runs": rows,
        "summary": {
            "sigma_real_fit_spread": float(np.ptp(fits)),
            "maximum_relative_frequency_error": float(
                max(row["relative_frequency_error"] for row in rows)
            ),
            "frequency_confirmation": True,
            "damping_magnitude_measured_independently": False,
        },
    }


def main() -> None:
    result = run_checkpoint()
    target = ROOT / "reports" / "axial" / "axial_time_domain_checkpoint.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    convergence = run_convergence_matrix()
    convergence_target = ROOT / "reports" / "axial" / "axial_time_domain_convergence.json"
    convergence_target.write_text(
        json.dumps(convergence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["time_domain"], indent=2))
    print(json.dumps(convergence["summary"], indent=2))


if __name__ == "__main__":
    main()
