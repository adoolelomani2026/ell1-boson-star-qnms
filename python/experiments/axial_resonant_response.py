"""Resolve the long-lived axial QNM as a real-frequency driven resonance."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import interior_mode_profile, scattering_amplitudes
from radial.coefficients import RadialBackground


def response_row(
    frequency: float,
    background: RadialBackground,
    *,
    r_match: float,
    r_end: float,
    r_far: float,
) -> dict[str, float]:
    scattering = scattering_amplitudes(
        frequency,
        background,
        "gravity",
        r_match=r_match,
        r_end=r_end,
        r_far=r_far,
        asymptotic_order=3,
        rtol=2e-10,
        atol=2e-12,
    )
    radii = np.linspace(2e-4, r_match, 900)
    profile = interior_mode_profile(
        radii,
        complex(frequency),
        background,
        scattering["interior_coefficients"],
        rtol=2e-9,
        atol=2e-11,
    )
    alpha, gamma = [], []
    for radius in radii:
        point = background.point(float(radius))
        alpha.append(point.alpha)
        gamma.append(point.gamma)
    measure = np.asarray(alpha) * np.asarray(gamma) * radii**2
    scalar_norm = simpson(
        measure * (np.abs(profile[2]) ** 2 + np.abs(profile[4]) ** 2),
        x=radii,
    )
    metric_norm = simpson(
        measure * (np.abs(profile[0]) ** 2 + np.abs(profile[1]) ** 2),
        x=radii,
    )
    return {
        "sigma": frequency,
        "reflection_abs": float(abs(scattering["gravity_out"])),
        "reflection_phase": float(np.angle(scattering["gravity_out"])),
        "stored_scalar_norm": float(scalar_norm),
        "stored_metric_norm": float(metric_norm),
        "linear_residual": float(scattering["linear_residual"]),
    }


def run_checkpoint() -> dict[str, object]:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    # Fixed physical grid: it is declared independently of the stored complex
    # pole and the response is fitted before that checkpoint is read.
    frequencies = np.linspace(0.0493945, 0.0494010, 41)
    rows = [
        response_row(
            frequency,
            background,
            r_match=14.0,
            r_end=35.0,
            r_far=600.0,
        )
        for frequency in frequencies
    ]
    norms = np.asarray([row["stored_scalar_norm"] for row in rows])
    scale = max(float(np.max(norms)), 1e-300)
    midpoint = 0.5 * (frequencies[0] + frequencies[-1])
    span = 0.5 * (frequencies[-1] - frequencies[0])
    x_all = (frequencies - midpoint) / span

    def fit_window(trim: int, background_order: int) -> dict[str, float]:
        selected = slice(trim, len(frequencies) - trim if trim else None)
        x = x_all[selected]
        target = norms[selected] / scale

        def residual(parameters):
            amplitude, center_x, width_x, floor, slope, curvature = parameters
            background_floor = floor + slope * x
            if background_order == 2:
                background_floor += curvature * x**2
            model = background_floor + amplitude / (
                ((x - center_x) / width_x) ** 2 + 1.0
            )
            return model - target

        fit = least_squares(
            residual,
            x0=np.asarray((0.98, 0.0, 0.18, 0.01, 0.0, 0.0)),
            bounds=(
                np.asarray((0.0, -0.6, 0.005, 0.0, -0.5, -0.5)),
                np.asarray((2.0, 0.6, 0.8, 1.0, 0.5, 0.5)),
            ),
        )
        dof = max(len(fit.fun) - len(fit.x), 1)
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * (
            np.dot(fit.fun, fit.fun) / dof
        )
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        amplitude, center_x, width_x, floor, slope, curvature = fit.x
        return {
            "trim_each_edge": trim,
            "background_order": background_order,
            "center": float(midpoint + span * center_x),
            "center_standard_error": float(span * errors[1]),
            "half_width": float(span * width_x),
            "half_width_standard_error": float(span * errors[2]),
            "normalized_amplitude": float(amplitude),
            "normalized_floor": float(floor),
            "normalized_floor_slope": float(slope),
            "normalized_floor_curvature": float(curvature if background_order == 2 else 0.0),
            "rms_residual": float(np.sqrt(np.mean(fit.fun**2))),
            "samples": int(len(fit.fun)),
        }

    fit_variants = [
        fit_window(trim, order)
        for trim, order in ((0, 1), (0, 2), (3, 1), (3, 2), (6, 2))
    ]
    primary = fit_variants[1]
    fitted_center = primary["center"]
    fitted_half_width = primary["half_width"]

    # Only after fitting the fixed grid do we load the frequency-domain pole.
    qnm = json.loads((ROOT / "reports" / "axial" / "axial_qnm_checkpoint.json").read_text())
    center = float(qnm["mode"]["sigma_real_mean"])
    half_width = -float(qnm["mode"]["sigma_imag_mean"])

    phases = np.unwrap(np.asarray([row["reflection_phase"] for row in rows]))
    center_index = int(np.argmin(abs(frequencies - fitted_center)))
    phase_slope = (
        phases[center_index + 1] - phases[center_index - 1]
    ) / (frequencies[center_index + 1] - frequencies[center_index - 1])
    for row, phase in zip(rows, phases):
        row["unwrapped_reflection_phase"] = float(phase)
        row["detuning_in_qnm_half_widths"] = float(
            (row["sigma"] - center) / half_width
        )

    return {
        "calculation": "predeclared targeted fixed-grid unit-incident-gravity driven axial resonance; fitting does not load the stored pole",
        "background_a1_0": solution.a0,
        "qnm_prediction": {
            "sigma_real": center,
            "half_width_minus_sigma_imag": half_width,
        },
        "fixed_grid": {
            "minimum": float(frequencies[0]),
            "maximum": float(frequencies[-1]),
            "points": len(frequencies),
            "spacing": float(frequencies[1] - frequencies[0]),
        },
        "lorentzian_fit": {
            **primary,
            "center_relative_error": float(abs(fitted_center - center) / center),
            "half_width_relative_error": float(abs(fitted_half_width - half_width) / half_width),
        },
        "fit_variants": fit_variants,
        "phase_delay": {
            "d_arg_R_d_sigma_at_center": float(phase_slope),
            "single_pole_prediction_2_over_half_width": float(2.0 / half_width),
            "relative_error": float(abs(abs(phase_slope) - 2.0 / half_width) / (2.0 / half_width)),
        },
        "scan": rows,
    }


def main() -> None:
    result = run_checkpoint()
    target = ROOT / "reports" / "axial" / "axial_resonant_response.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["lorentzian_fit"], indent=2))
    print(json.dumps(result["phase_delay"], indent=2))


if __name__ == "__main__":
    main()
