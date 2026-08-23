"""Build the LaTeX preprint and regenerate its numerical figures and tables."""

from __future__ import annotations

import platform
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PYTHON_ROOT))

from paper.build_certification_report import ASSETS, _build_numerical_assets, _configure_plotting

SOURCE = ROOT / "paper" / "boson_star_radial_preprint.tex"
AXIAL_SECTION = ROOT / "paper" / "axial_channel.tex"
AXIAL_DYNAMICS = ROOT / "paper" / "axial_dynamics.tex"
AXIAL_RESPONSE = ROOT / "paper" / "axial_response.tex"
CHARGED_QUANTUM_SECTION = ROOT / "paper" / "charged_quantum.tex"
BUILD = ROOT / "tmp" / "pdfs" / "boson_star_radial_preprint"
OUTPUT = ROOT / "output" / "pdf" / "axial_matter_channel_ell1_preprint.pdf"
REPOSITORY_PDF = ROOT / "pdfs" / "axial_matter_channel_ell1_preprint.pdf"
REPOSITORY_FIGURES = ROOT / "figures" / "manuscript"
REPORTS = ROOT / "reports"
SEQUENCE = REPORTS / "background" / "ell1_sequence.csv"
RADIAL = REPORTS / "radial" / "radial_benchmarks.csv"
GRAVITY = REPORTS / "background" / "gravitational_diagnostics.csv"
REFINED_STABILITY = REPORTS / "radial" / "refined_radial_stability.csv"
PHYSICS_SCALING = REPORTS / "extensions" / "physics_extension_scaling.csv"
BACKGROUND_SENSITIVITY = REPORTS / "background" / "background_sensitivity.csv"
RADIAL_SENSITIVITY = REPORTS / "radial" / "radial_sensitivity.csv"
STRESS = REPORTS / "background" / "stress_energy_diagnostics.csv"
STRESS_PROFILES = REPORTS / "background" / "stress_energy_profiles.csv"
BACKGROUND_IVP = REPORTS / "background" / "background_ivp_consistency.csv"
AXIAL_SCATTERING = REPORTS / "axial" / "axial_scattering_checkpoint.json"
AXIAL_RESONANCE = REPORTS / "axial" / "axial_resonant_response.json"
CHARGED = REPORTS / "extensions" / "charged_background_checkpoint.json"
SEMICLASSICAL = REPORTS / "extensions" / "semiclassical_coherent_qnm.json"
TIME_DOMAIN_CONVERGENCE = REPORTS / "axial" / "axial_time_domain_convergence.json"
AXIAL_BRANCH = REPORTS / "axial" / "axial_qnm_branch.json"


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "archived-source"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _extended_figures() -> None:
    rows = _read_rows(SEQUENCE)
    a0 = np.array([float(row["a0"]) for row in rows])
    mass = np.array([float(row["adm_mass"]) for row in rows])
    charge = np.array([float(row["noether_charge"]) for row in rows])
    ode = np.array([float(row["max_ode_residual"]) for row in rows])
    tail = np.array([float(row["tail_residual"]) for row in rows])
    derivative = np.gradient(mass, a0)

    stability = _read_rows(REPORTS / "radial" / "report_radial_stability.csv")
    sa0 = np.array([float(row["a0"]) for row in stability])
    sigma2 = np.array([float(row["sigma2"]) for row in stability])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    axes[0].plot(a0, mass - charge, "o-", color="#1F6F8B", ms=3)
    axes[0].axhline(0.0, color="#56636B", lw=0.8)
    axes[0].set_xlabel(r"central amplitude $a_1^0$")
    axes[0].set_ylabel(r"binding diagnostic $M_T-Q$")
    axes[0].set_title("Mass-charge binding diagnostic")
    left = axes[1]
    right = left.twinx()
    left.plot(a0, derivative, "o-", color="#15324B", ms=3, label=r"$dM_T/da_1^0$")
    right.plot(sa0, sigma2, "s--", color="#C84C4C", ms=3, label=r"$\sigma_0^2$")
    left.axhline(0.0, color="#56636B", lw=0.8)
    right.axhline(0.0, color="#C84C4C", lw=0.7, alpha=0.55)
    left.set_xlabel(r"central amplitude $a_1^0$")
    left.set_ylabel(r"$dM_T/da_1^0$", color="#15324B")
    right.set_ylabel(r"$\sigma_0^2$", color="#C84C4C")
    left.set_title("Turning point and radial eigenvalue")
    handles = left.get_lines()[:1] + right.get_lines()[:1]
    left.legend(handles, [item.get_label() for item in handles], loc="lower left")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_07_binding_turning_point.pdf", bbox_inches="tight")
    plt.close(fig)


def _new_physics_figures() -> None:
    # Continue the extended validation figures after the separate first-page
    # figure, then add the axial-response and extension panels below.
    sequence_rows = _read_rows(SEQUENCE)
    a0 = np.asarray([float(row["a0"]) for row in sequence_rows])
    ode = np.asarray([float(row["max_ode_residual"]) for row in sequence_rows])
    tail = np.asarray([float(row["tail_residual"]) for row in sequence_rows])
    scattering = json.loads(AXIAL_SCATTERING.read_text(encoding="utf-8"))
    scan = scattering["frequency_scan"]
    frequency = np.asarray([row["sigma"] for row in scan])
    g_to_s = np.asarray(
        [row["gravity_incident"]["scalar_fraction"] for row in scan]
    )
    s_to_g = np.asarray(
        [row["scalar_incident"]["gravity_fraction"] for row in scan]
    )
    flux_error = np.asarray(
        [abs(row["gravity_incident"]["sum"] - 1.0) for row in scan]
    )
    unity = np.asarray([row["unitarity_spectral_defect"] for row in scan])
    resonance = json.loads(AXIAL_RESONANCE.read_text(encoding="utf-8"))
    resonance_scan = resonance["scan"]
    detuning = np.asarray(
        [row["detuning_in_qnm_half_widths"] for row in resonance_scan]
    )
    stored = np.asarray([row["stored_scalar_norm"] for row in resonance_scan])
    stored /= np.max(stored)
    phase = np.asarray(
        [row["unwrapped_reflection_phase"] for row in resonance_scan]
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1))
    axes[0, 0].semilogy(frequency, g_to_s, "o-", label=r"$g\to\phi$", color="#1F6F8B")
    axes[0, 0].semilogy(frequency, s_to_g, "s--", label=r"$\phi\to g$", color="#C84C4C")
    axes[0, 0].set_xlabel(r"driving frequency $\sigma$")
    axes[0, 0].set_ylabel("reciprocity-weighted conversion")
    axes[0, 0].set_title("Inferred-metric channel balance")
    axes[0, 0].legend()
    axes[0, 1].semilogy(frequency, np.maximum(flux_error, 1e-18), "o-", label="weighted sum", color="#6D597A")
    axes[0, 1].semilogy(frequency, unity, "s--", label=r"$W$-isometry", color="#B45309")
    axes[0, 1].set_xlabel(r"driving frequency $\sigma$")
    axes[0, 1].set_ylabel("absolute defect")
    axes[0, 1].set_title("Algebraic consistency tests")
    axes[0, 1].legend()
    axes[1, 0].semilogy(detuning, stored, "o-", color="#2A9D8F")
    axes[1, 0].set_xlabel(r"$(\sigma-\sigma_R)/(-\sigma_I)$")
    axes[1, 0].set_ylabel("normalized stored scalar norm")
    axes[1, 0].set_title("Driven long-lived resonance")
    axes[1, 1].plot(detuning, phase, "o-", color="#15324B")
    axes[1, 1].set_xlabel(r"$(\sigma-\sigma_R)/(-\sigma_I)$")
    axes[1, 1].set_ylabel(r"unwrapped $\arg S_{gg}$")
    axes[1, 1].set_title("Resonant phase rotation")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_15_axial_response.pdf", bbox_inches="tight")
    plt.close(fig)

    branch = json.loads(AXIAL_BRANCH.read_text(encoding="utf-8"))["rows"]
    compactness = np.asarray([row["compactness_R99"] for row in branch])
    sigma_real = np.asarray([row["sigma_real"] for row in branch])
    damping = -np.asarray([row["sigma_imag"] for row in branch])
    quality = np.asarray([row["quality_factor"] for row in branch])
    overlap = np.asarray([
        np.nan if row["overlap_with_previous"] is None else row["overlap_with_previous"]
        for row in branch
    ])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    axes[0, 0].plot(compactness, sigma_real, "o-", color="#15324B")
    axes[0, 0].set_ylabel(r"$\operatorname{Re}\sigma$")
    axes[0, 0].set_title("Oscillation-frequency branch")
    axes[0, 1].plot(compactness, damping * 1e7, "o-", color="#C84C4C")
    axes[0, 1].set_ylabel(r"$-10^7\operatorname{Im}\sigma$")
    axes[0, 1].set_title("Gravitational damping branch")
    axes[1, 0].plot(compactness, quality, "o-", color="#2A9D8F")
    axes[1, 0].set_ylabel(r"quality factor $Q$")
    axes[1, 0].set_title("Mode quality factor")
    axes[1, 1].plot(compactness[1:], overlap[1:], "o-", color="#6D597A")
    axes[1, 1].set_ylim(0.9985, 1.0001)
    axes[1, 1].set_ylabel("adjacent profile overlap")
    axes[1, 1].set_title("Branch identity")
    for axis in axes.flat:
        axis.set_xlabel(r"compactness $C_{99}$")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_17_axial_branch.pdf", bbox_inches="tight")
    plt.close(fig)

    charged = json.loads(CHARGED.read_text(encoding="utf-8"))["sequence"]
    charge_ratio = np.asarray([row["q_over_sqrt_4pi"] for row in charged])
    mass = np.asarray([row["adm_mass"] for row in charged])
    electric_charge = np.asarray([row["total_electric_charge"] for row in charged])
    compactness = np.asarray([row["compactness99"] for row in charged])
    em_fraction = np.asarray([row["electric_field_energy_fraction"] for row in charged])
    quantum = json.loads(SEMICLASSICAL.read_text(encoding="utf-8"))["mass_scaling"]
    boson_mass = np.asarray([row["boson_mass_ev"] for row in quantum])
    dispersion = np.asarray([row["qnm_fractional_standard_deviation"] for row in quantum])
    bias = np.asarray([row["mean_qnm_fractional_shift_magnitude"] for row in quantum])

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1))
    left = axes[0, 0]
    right = left.twinx()
    left.plot(charge_ratio, mass, "o-", color="#1F6F8B")
    right.plot(charge_ratio, electric_charge, "s--", color="#C84C4C")
    left.set_xlabel(r"$q/q_{\rm crit}$")
    left.set_ylabel(r"$M_{\rm ADM}$", color="#1F6F8B")
    right.set_ylabel(r"$Q_{\rm electric}$", color="#C84C4C")
    left.set_title("Backreacting charged continuation")
    left = axes[0, 1]
    right = left.twinx()
    left.plot(charge_ratio, compactness, "o-", color="#6D597A")
    right.plot(charge_ratio, em_fraction, "s--", color="#B45309")
    left.set_xlabel(r"$q/q_{\rm crit}$")
    left.set_ylabel(r"$C_{99}$", color="#6D597A")
    right.set_ylabel(r"$E_{\rm EM}/M$", color="#B45309")
    left.set_title("Compactness and Maxwell energy")
    axes[1, 0].loglog(boson_mass, dispersion, color="#2A9D8F")
    axes[1, 0].set_xlabel(r"boson mass $\mu$ [eV]")
    axes[1, 0].set_ylabel(r"$\Delta\sigma_{\rm rms}/|\sigma|$")
    axes[1, 0].set_title("Coherent occupation dispersion")
    axes[1, 1].loglog(boson_mass, bias, color="#15324B")
    axes[1, 1].set_xlabel(r"boson mass $\mu$ [eV]")
    axes[1, 1].set_ylabel(r"$|\langle\delta\sigma\rangle|/|\sigma|$")
    axes[1, 1].set_title("Mean occupation-noise bias")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_16_charged_quantum.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].semilogy(a0, ode, "o-", color="#1F6F8B", ms=3)
    axes[0].set_xlabel(r"central amplitude $a_1^0$")
    axes[0].set_ylabel("maximum background ODE defect")
    axes[0].set_title("Dense first-order residual")
    axes[1].semilogy(a0, np.maximum(tail, 1e-34), "o-", color="#2A9D8F", ms=3)
    axes[1].set_xlabel(r"central amplitude $a_1^0$")
    axes[1].set_ylabel("normalized outer-tail residual")
    axes[1].set_title("Massive-field Robin condition")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_08_background_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

    gravity = _read_rows(GRAVITY)
    ga0 = np.array([float(row["a0"]) for row in gravity])
    alpha_center = np.array([float(row["alpha_center"]) for row in gravity])
    max_compactness = np.array([float(row["max_two_m_over_r"]) for row in gravity])
    surface_redshift = np.array([float(row["surface_redshift_99"]) for row in gravity])
    kretschmann = np.array([float(row["exterior_kretschmann_99"]) for row in gravity])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    axes[0, 0].plot(ga0, alpha_center, "o-", color="#1F6F8B", ms=3)
    axes[0, 0].set_ylabel(r"central lapse $\alpha_c$")
    axes[0, 0].set_title("Central gravitational time dilation")
    axes[0, 1].plot(ga0, max_compactness, "o-", color="#B45309", ms=3)
    axes[0, 1].axhline(1.0, color="#56636B", lw=0.8, ls="--")
    axes[0, 1].set_ylabel(r"$\max_r(2M/r)$")
    axes[0, 1].set_title("Horizon-avoidance diagnostic")
    axes[1, 0].plot(ga0, surface_redshift, "o-", color="#6D597A", ms=3)
    axes[1, 0].set_ylabel(r"$z_{99}$")
    axes[1, 0].set_title("Redshift at the 99% mass radius")
    axes[1, 1].semilogy(ga0, kretschmann, "o-", color="#2A9D8F", ms=3)
    axes[1, 1].set_ylabel(r"$48M_T^2/R_{99}^6$")
    axes[1, 1].set_title("Exterior curvature proxy")
    for axis in axes[-1]:
        axis.set_xlabel(r"central amplitude $a_1^0$")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_09_gravitational_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

    refined = _read_rows(REFINED_STABILITY)
    ra0 = np.array([float(row["a0"]) for row in refined])
    rsigma = np.array([float(row["sigma2_ground"]) for row in refined])
    rcondition = np.array([float(row["eigenvalue_condition_number"]) for row in refined])
    rresidual = np.array([float(row["unscaled_generalized_residual"]) for row in refined])
    root_index = np.flatnonzero(rsigma[:-1] * rsigma[1:] <= 0.0)[0]
    root = ra0[root_index] - rsigma[root_index] * (
        ra0[root_index + 1] - ra0[root_index]
    ) / (rsigma[root_index + 1] - rsigma[root_index])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    axes[0].plot(ra0, rsigma, "o-", color="#1F6F8B", ms=4)
    axes[0].axhline(0.0, color="#56636B", lw=0.8)
    axes[0].axvline(root, color="#C84C4C", lw=0.9, ls="--")
    axes[0].set_xlabel(r"central amplitude $a_1^0$")
    axes[0].set_ylabel(r"$\sigma_0^2$")
    axes[0].set_title("Fine stability crossing")
    left = axes[1]
    right = left.twinx()
    left.semilogy(ra0, rcondition, "o-", color="#B45309", ms=3)
    right.semilogy(ra0, rresidual, "s--", color="#2A9D8F", ms=3)
    left.set_xlabel(r"central amplitude $a_1^0$")
    left.set_ylabel(r"condition diagnostic $\kappa_\lambda$", color="#B45309")
    right.set_ylabel("unscaled pencil residual", color="#2A9D8F")
    left.set_title("Algebraic quality near zero")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_10_refined_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    background_sensitivity = _read_rows(BACKGROUND_SENSITIVITY)
    baseline = next(
        row
        for row in background_sensitivity
        if row["experiment"] == "outer_domain" and float(row["r_max"]) == 80.0
    )
    radial_sensitivity = _read_rows(RADIAL_SENSITIVITY)
    bvp_epsilon = [
        row for row in radial_sensitivity if row["method"] == "nonlinear_global_bvp"
    ]
    spectral_ground = [
        row for row in radial_sensitivity if row["method"] == "chebyshev_mode_0"
    ]
    spectral_overtone = [
        row for row in radial_sensitivity if row["method"] == "chebyshev_mode_1"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    domain = [row for row in background_sensitivity if row["experiment"] == "outer_domain"]
    axes[0, 0].semilogy(
        [float(row["r_max"]) for row in domain],
        [
            abs(float(row["omega"]) - float(baseline["omega"])) + 1e-18
            for row in domain
        ],
        "o-",
        color="#1F6F8B",
    )
    axes[0, 0].set_xlabel(r"background $r_{\max}$")
    axes[0, 0].set_ylabel(r"$|\Delta\omega|$")
    axes[0, 0].set_title("Background domain sensitivity")
    axes[0, 1].plot(
        [float(row["epsilon"]) for row in bvp_epsilon],
        [float(row["sigma2"]) for row in bvp_epsilon],
        "o-",
        color="#6D597A",
    )
    axes[0, 1].set_xlabel(r"center cutoff $\epsilon$")
    axes[0, 1].set_ylabel(r"BVP $\sigma_0^2$")
    axes[0, 1].set_title("Regular-center cutoff study")
    axes[1, 0].plot(
        [int(row["resolution"]) for row in spectral_ground],
        [float(row["sigma2"]) for row in spectral_ground],
        "o-",
        label="ground",
        color="#1F6F8B",
        ms=3,
    )
    axes[1, 0].plot(
        [int(row["resolution"]) for row in spectral_overtone],
        [float(row["sigma2"]) for row in spectral_overtone],
        "s-",
        label="first overtone",
        color="#C84C4C",
        ms=3,
    )
    axes[1, 0].set_xlabel("Chebyshev points")
    axes[1, 0].set_ylabel(r"$\sigma_j^2$")
    axes[1, 0].set_title("Resolution continuation")
    axes[1, 0].legend()
    axes[1, 1].semilogy(
        [int(row["resolution"]) for row in spectral_ground],
        [float(row["condition_number"]) for row in spectral_ground],
        "o-",
        label="ground",
        color="#1F6F8B",
        ms=3,
    )
    axes[1, 1].semilogy(
        [int(row["resolution"]) for row in spectral_overtone],
        [float(row["condition_number"]) for row in spectral_overtone],
        "s-",
        label="first overtone",
        color="#C84C4C",
        ms=3,
    )
    axes[1, 1].set_xlabel("Chebyshev points")
    axes[1, 1].set_ylabel(r"$\kappa_\lambda$")
    axes[1, 1].set_title("Conditioning under refinement")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_11_solver_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    scaling = _read_rows(PHYSICS_SCALING)
    mu = np.array([float(row["boson_mass_ev"]) for row in scaling])
    physical_mass = np.array([float(row["mass_solar"]) for row in scaling])
    physical_radius = np.array([float(row["r99_km"]) for row in scaling])
    occupation = np.array([float(row["particle_number"]) for row in scaling])
    planck = np.array([float(row["planck_suppression"]) for row in scaling])
    qbalance = np.array([float(row["charge_balance_in_units_of_e"]) for row in scaling])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    axes[0, 0].loglog(mu, physical_mass, color="#1F6F8B")
    axes[0, 0].set_ylabel(r"$M_T/M_\odot$")
    axes[0, 0].set_title("Physical mass scaling")
    axes[0, 1].loglog(mu, physical_radius, color="#6D597A")
    axes[0, 1].set_ylabel(r"$R_{99}$ [km]")
    axes[0, 1].set_title("Physical radius scaling")
    axes[1, 0].loglog(mu, occupation, color="#2A9D8F")
    axes[1, 0].set_ylabel(r"occupation estimate $N$")
    axes[1, 0].set_title("Classical-field occupation")
    left = axes[1, 1]
    right = left.twinx()
    left.loglog(mu, planck, color="#B45309")
    right.loglog(mu, qbalance, color="#C84C4C", ls="--")
    left.set_ylabel(r"$(\mu/M_{\rm Pl})^2$", color="#B45309")
    right.set_ylabel(r"$q_{\rm balance}/e$", color="#C84C4C")
    left.set_title("Quantum-gravity and charge scales")
    for axis in axes[-1]:
        axis.set_xlabel(r"boson mass $\mu$ [eV]")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_12_physics_scaling.pdf", bbox_inches="tight")
    plt.close(fig)

    stress = _read_rows(STRESS)
    profiles = _read_rows(STRESS_PROFILES)
    stress_a0 = np.array([float(row["a0"]) for row in stress])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    benchmark_profile = [
        row for row in profiles if abs(float(row["a0"]) - 0.08) < 1e-12
    ]
    br = np.array([float(row["r"]) for row in benchmark_profile])
    brho = np.array([float(row["rho"]) for row in benchmark_profile])
    axes[0, 0].plot(br, brho / np.max(brho), label=r"$\rho$", color="#15324B")
    axes[0, 0].plot(
        br,
        np.array([float(row["pressure_r"]) for row in benchmark_profile]) / np.max(brho),
        label=r"$p_r$",
        color="#C84C4C",
    )
    axes[0, 0].plot(
        br,
        np.array([float(row["pressure_t"]) for row in benchmark_profile]) / np.max(brho),
        label=r"$p_t$",
        color="#2A9D8F",
    )
    axes[0, 0].set_xlim(0.0, 25.0)
    axes[0, 0].set_xlabel(r"$r\mu$")
    axes[0, 0].set_ylabel(r"component$/\rho_{\rm peak}$")
    axes[0, 0].set_title(r"Benchmark effective stresses")
    axes[0, 0].legend()
    for amplitude, color in zip((0.04, 0.08, 0.12), ("#1F6F8B", "#6D597A", "#B45309")):
        selected = [
            row for row in profiles if abs(float(row["a0"]) - amplitude) < 1e-12
        ]
        radius = np.array([float(row["r"]) for row in selected])
        rho = np.array([float(row["rho"]) for row in selected])
        anisotropy = np.array([float(row["anisotropy"]) for row in selected])
        axes[0, 1].plot(
            radius,
            anisotropy / np.max(rho),
            color=color,
            label=rf"$a_1^0={amplitude:.2f}$",
        )
    axes[0, 1].set_xlim(0.0, 25.0)
    axes[0, 1].set_xlabel(r"$r\mu$")
    axes[0, 1].set_ylabel(r"$(p_t-p_r)/\rho_{\rm peak}$")
    axes[0, 1].set_title("Pressure anisotropy")
    axes[0, 1].legend()
    axes[1, 0].plot(
        stress_a0,
        [float(row["min_rho_plus_pr_over_peak"]) for row in stress],
        "o-",
        label=r"$\min(\rho+p_r)$",
        ms=3,
    )
    axes[1, 0].plot(
        stress_a0,
        [float(row["min_rho_plus_pt_over_peak"]) for row in stress],
        "s-",
        label=r"$\min(\rho+p_t)$",
        ms=3,
    )
    axes[1, 0].axhline(0.0, color="#56636B", lw=0.8)
    axes[1, 0].set_xlabel(r"central amplitude $a_1^0$")
    axes[1, 0].set_ylabel(r"weak-condition margin$/\rho_{\rm peak}$")
    axes[1, 0].set_title("Weak energy-condition margins")
    axes[1, 0].legend()
    left = axes[1, 1]
    right = left.twinx()
    left.semilogy(
        stress_a0,
        [float(row["rho_peak"]) for row in stress],
        "o-",
        color="#15324B",
        ms=3,
    )
    right.plot(
        stress_a0,
        [float(row["max_abs_anisotropy_over_peak"]) for row in stress],
        "s--",
        color="#C84C4C",
        ms=3,
    )
    left.set_xlabel(r"central amplitude $a_1^0$")
    left.set_ylabel(r"$\rho_{\rm peak}$", color="#15324B")
    right.set_ylabel(r"$\max|p_t-p_r|/\rho_{\rm peak}$", color="#C84C4C")
    left.set_title("Density and anisotropy scale")
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_13_stress_energy.pdf", bbox_inches="tight")
    plt.close(fig)

    ivp_rows = _read_rows(BACKGROUND_IVP)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    colors = {0.04: "#1F6F8B", 0.08: "#6D597A", 0.10: "#B45309"}
    for amplitude in (0.04, 0.08, 0.10):
        selected = [
            row for row in ivp_rows if abs(float(row["a0"]) - amplitude) < 1e-12
        ]
        endpoints = [float(row["endpoint"]) for row in selected]
        axes[0].semilogy(
            endpoints,
            [float(row["mass_scaled_max_difference"]) for row in selected],
            "o-",
            color=colors[amplitude],
            label=rf"$a_1^0={amplitude:.2f}$",
        )
        axes[1].semilogy(
            endpoints,
            [float(row["psi_scaled_max_difference"]) for row in selected],
            "o-",
            color=colors[amplitude],
            label=rf"$a_1^0={amplitude:.2f}$",
        )
    axes[0].set_xlabel("IVP comparison endpoint")
    axes[0].set_ylabel("scaled maximum mass difference")
    axes[0].set_title("Independent integration: metric sector")
    axes[1].set_xlabel("IVP comparison endpoint")
    axes[1].set_ylabel("scaled maximum scalar difference")
    axes[1].set_title("Independent integration: scalar sector")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "figure_14_background_ivp.pdf", bbox_inches="tight")
    plt.close(fig)


def _sequence_rows() -> str:
    lines = []
    for row in _read_rows(SEQUENCE):
        lines.append(
            f'{float(row["a0"]):.3f} & {float(row["omega"]):.7f} & '
            f'{float(row["adm_mass"]):.7f} & {float(row["noether_charge"]):.7f} & '
            f'{float(row["r99"]):.4f} & {float(row["r999"]):.4f} & '
            f'{float(row["compactness99"]):.5f} \\\\'
        )
    return "\n".join(lines)


def _spectral_rows() -> str:
    rows = _read_rows(RADIAL)
    selected: dict[int, dict[int, dict[str, str]]] = {}
    for row in rows:
        if (
            row["formulation"] == "chebyshev_generalized_eigenproblem"
            and row["background_representation"] == "hermite"
            and abs(float(row["a0"]) - 0.08) < 1e-12
            and abs(float(row["r_max"]) - 40.0) < 1e-12
        ):
            points = int(row["resolution"].split()[0])
            if points in {50, 60, 80, 100, 120, 160}:
                selected.setdefault(points, {})[int(row["mode_index"])] = row
    reference = 2.4004311443394838e-4
    lines = []
    for points in sorted(selected):
        ground = selected[points][0]
        overtone = selected[points][1]
        gvalue = float(ground["sigma2"])
        lines.append(
            f'{points} & {gvalue:.10e} & {abs(gvalue-reference):.2e} & '
            f'{float(ground["eigenvalue_condition_number"]):.3e} & '
            f'{float(overtone["sigma2"]):.10e} & '
            f'{float(overtone["eigenvalue_condition_number"]):.3e} \\\\'
        )
    return "\n".join(lines)


def _gravity_rows() -> str:
    lines = []
    for row in _read_rows(GRAVITY):
        lines.append(
            f'{float(row["a0"]):.3f} & {float(row["alpha_center"]):.6f} & '
            f'{float(row["central_redshift"]):.5f} & '
            f'{float(row["max_two_m_over_r"]):.5f} & '
            f'{float(row["surface_redshift_99"]):.5f} & '
            f'{float(row["exterior_kretschmann_99"]):.3e} \\\\'
        )
    return "\n".join(lines)


def _refined_stability_rows() -> str:
    lines = []
    for row in _read_rows(REFINED_STABILITY):
        lines.append(
            f'{float(row["a0"]):.5f} & {float(row["omega"]):.8f} & '
            f'{float(row["adm_mass"]):.8f} & '
            f'{float(row["sigma2_ground"]):.9e} & '
            f'{float(row["eigenvalue_condition_number"]):.3e} & '
            f'{float(row["unscaled_generalized_residual"]):.2e} \\\\'
        )
    return "\n".join(lines)


def _physics_scaling_rows() -> str:
    rows = _read_rows(PHYSICS_SCALING)
    lines = []
    for index in (0, 6, 12, 18, 24, 31):
        row = rows[index]
        lines.append(
            f'{float(row["boson_mass_ev"]):.0e} & '
            f'{float(row["mass_solar"]):.3e} & '
            f'{float(row["r99_km"]):.3e} & '
            f'{float(row["particle_number"]):.3e} & '
            f'{float(row["finite_n_fractional_scale"]):.3e} & '
            f'{float(row["planck_suppression"]):.3e} & '
            f'{float(row["charge_balance_in_units_of_e"]):.3e} \\\\'
        )
    return "\n".join(lines)


def _background_sensitivity_rows() -> str:
    lines = []
    for row in _read_rows(BACKGROUND_SENSITIVITY):
        label = row["experiment"].replace("_", " ")
        lines.append(
            f'{label} & {float(row["r_max"]):.0f} & {int(row["initial_points"])} & '
            f'{float(row["tolerance"]):.0e} & {float(row["omega"]):.12f} & '
            f'{float(row["adm_mass"]):.12f} & {float(row["r99"]):.7f} \\\\'
        )
    return "\n".join(lines)


def _radial_sensitivity_rows() -> str:
    lines = []
    for row in _read_rows(RADIAL_SENSITIVITY):
        condition = (
            "---"
            if not row["condition_number"]
            else f'{float(row["condition_number"]):.3e}'
        )
        lines.append(
            f'{row["method"].replace("_", " ")} & {float(row["epsilon"]):.1e} & '
            f'{int(row["resolution"])} & {float(row["sigma2"]):.10e} & '
            f'{float(row["residual"]):.2e} & {condition} \\\\'
        )
    return "\n".join(lines)


def _stress_rows() -> str:
    lines = []
    for row in _read_rows(STRESS):
        lines.append(
            f'{float(row["a0"]):.3f} & {float(row["rho_peak"]):.4e} & '
            f'{float(row["r_at_rho_peak"]):.4f} & '
            f'{float(row["min_rho_plus_pr_over_peak"]):.3e} & '
            f'{float(row["min_rho_plus_pt_over_peak"]):.3e} & '
            f'{float(row["max_abs_anisotropy_over_peak"]):.4f} \\\\'
        )
    return "\n".join(lines)


def _background_ivp_rows() -> str:
    lines = []
    for row in _read_rows(BACKGROUND_IVP):
        lines.append(
            f'{float(row["a0"]):.2f} & {float(row["endpoint"]):.0f} & '
            f'{int(row["ivp_function_evaluations"])} & '
            f'{float(row["mass_scaled_max_difference"]):.3e} & '
            f'{float(row["alpha_scaled_max_difference"]):.3e} & '
            f'{float(row["psi_scaled_max_difference"]):.3e} & '
            f'{float(row["dpsi_scaled_max_difference"]):.3e} \\\\'
        )
    return "\n".join(lines)


def _time_domain_rows() -> str:
    record = json.loads(TIME_DOMAIN_CONVERGENCE.read_text(encoding="utf-8"))
    def latex_scientific(value: float, digits: int) -> str:
        exponent = int(np.floor(np.log10(abs(value)))) if value else 0
        coefficient = value / 10.0**exponent if value else 0.0
        return f"{coefficient:.{digits}f}\\times10^{{{exponent}}}"

    lines = []
    for row in record["runs"]:
        lines.append(
            f'{int(row["points"])} & {float(row["courant"]):.2f} & '
            f'{float(row["sigma_real_fit"]):.10f} & '
            f'${latex_scientific(float(row["log_amplitude_slope"]), 2)}$ & '
            f'${latex_scientific(float(row["phase_residual_rms"]), 3)}$ \\\\'
        )
    return "\n".join(lines)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPOSITORY_PDF.parent.mkdir(parents=True, exist_ok=True)
    REPOSITORY_FIGURES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    _configure_plotting()
    _build_numerical_assets()
    _extended_figures()
    _new_physics_figures()

    replacements = {
        "@@COMMIT@@": git("rev-parse", "HEAD"),
        "@@TREE@@": git("rev-parse", "HEAD^{tree}"),
        "@@PYTHON@@": platform.python_version(),
        "@@NUMPY@@": np.__version__,
        "@@SCIPY@@": scipy.__version__,
        "@@SEQUENCE_ROWS@@": _sequence_rows(),
        "@@SPECTRAL_ROWS@@": _spectral_rows(),
        "@@GRAVITY_ROWS@@": _gravity_rows(),
        "@@REFINED_STABILITY_ROWS@@": _refined_stability_rows(),
        "@@PHYSICS_SCALING_ROWS@@": _physics_scaling_rows(),
        "@@BACKGROUND_SENSITIVITY_ROWS@@": _background_sensitivity_rows(),
        "@@RADIAL_SENSITIVITY_ROWS@@": _radial_sensitivity_rows(),
        "@@STRESS_ROWS@@": _stress_rows(),
        "@@BACKGROUND_IVP_ROWS@@": _background_ivp_rows(),
        "@@TIME_DOMAIN_ROWS@@": _time_domain_rows(),
    }
    manuscript = SOURCE.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        manuscript = manuscript.replace(marker, value)
    tex = BUILD / "manuscript.tex"
    tex.write_text(manuscript, encoding="utf-8")
    shutil.copy2(AXIAL_SECTION, BUILD / AXIAL_SECTION.name)
    dynamics = AXIAL_DYNAMICS.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        dynamics = dynamics.replace(marker, value)
    (BUILD / AXIAL_DYNAMICS.name).write_text(dynamics, encoding="utf-8")
    shutil.copy2(AXIAL_RESPONSE, BUILD / AXIAL_RESPONSE.name)
    shutil.copy2(CHARGED_QUANTUM_SECTION, BUILD / CHARGED_QUANTUM_SECTION.name)

    for figure in ASSETS.glob("figure_*.pdf"):
        shutil.copy2(figure, BUILD / figure.name)
        shutil.copy2(figure, REPOSITORY_FIGURES / figure.name)

    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"]
    for _ in range(3):
        subprocess.run(command, cwd=BUILD, check=True)
    shutil.copy2(BUILD / "manuscript.pdf", OUTPUT)
    shutil.copy2(BUILD / "manuscript.pdf", REPOSITORY_PDF)
    print(OUTPUT)


if __name__ == "__main__":
    main()
