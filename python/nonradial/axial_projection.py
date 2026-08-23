"""Odd-parity angular projections for an ell-boson-star scalar multiplet.

This module performs the first, purely angular checkpoint needed before a
relativistic nonradial solver can be assembled.  It uses the tensor-harmonic
convention

    P_sigma = sum_m <L m, ell sigma | J M> Y_Lm

and the diagonal-rotation singlet background

    B_sigma = (-1)^ell / sqrt(2 ell + 1) conjugate(Y_ell,sigma).

The overall phase of B is conventional.  Projection magnitudes and the
question of whether a source vanishes are phase independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import sph_harm_y
import sympy as sp
from sympy.physics.wigner import clebsch_gordan


@dataclass(frozen=True)
class AxialProjection:
    ell: int
    total_j: int
    orbital_l: int
    magnetic_m: int
    density_norm: float
    vector_coefficient: complex
    vector_source_norm: float
    vector_harmonic_norm: float
    tensor_coefficient: complex
    tensor_source_norm: float
    tensor_harmonic_norm: float
    quadrature_theta: int
    quadrature_phi: int

    def serializable(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("vector_coefficient", "tensor_coefficient"):
            value = result[key]
            assert isinstance(value, complex)
            result[key] = {"real": value.real, "imag": value.imag}
        return result


def symbolic_m0_projection() -> dict[str, sp.Expr]:
    """Return an exact SymPy check for ell=1, J=L=2, M=0.

    Low-order harmonics are written explicitly in the same Condon--Shortley
    convention used by SciPy.  This is deliberately independent of the
    numerical spherical-harmonic and quadrature path below.
    """

    theta, phi = sp.symbols("theta phi", real=True)
    root = sp.sqrt
    y_11 = -root(3 / (8 * sp.pi)) * sp.sin(theta) * sp.exp(sp.I * phi)
    y_1m1 = root(3 / (8 * sp.pi)) * sp.sin(theta) * sp.exp(-sp.I * phi)
    y_21 = (
        -root(15 / (8 * sp.pi))
        * sp.sin(theta)
        * sp.cos(theta)
        * sp.exp(sp.I * phi)
    )
    y_2m1 = (
        root(15 / (8 * sp.pi))
        * sp.sin(theta)
        * sp.cos(theta)
        * sp.exp(-sp.I * phi)
    )
    y_20 = root(5 / (16 * sp.pi)) * (3 * sp.cos(theta) ** 2 - 1)

    background_phase = -1 / root(3)
    b_star = [background_phase * y_11, background_phase * y_1m1]
    perturbation = [-y_2m1 / root(2), y_21 / root(2)]

    density = sp.simplify(sum(b * p for b, p in zip(b_star, perturbation)))
    s_theta = sp.simplify(
        sum(b * sp.diff(p, theta) for b, p in zip(b_star, perturbation))
    )
    s_phi = sp.simplify(
        sum(b * sp.diff(p, phi) for b, p in zip(b_star, perturbation))
    )
    q_tp = sp.simplify(
        sum(
            (
                sp.diff(b, theta) * sp.diff(p, phi)
                + sp.diff(b, phi) * sp.diff(p, theta)
            )
            / 2
            for b, p in zip(b_star, perturbation)
        )
    )

    x_phi = sp.sin(theta) * sp.diff(y_20, theta) / root(6)
    x_tp = sp.simplify(
        sp.diff(x_phi, theta) / 2 - sp.cot(theta) * x_phi
    )
    vector_integrand = sp.conjugate(x_phi) * s_phi / sp.sin(theta)
    tensor_integrand = 2 * sp.conjugate(x_tp) * q_tp / sp.sin(theta)
    vector_coefficient = sp.simplify(
        sp.integrate(vector_integrand, (phi, 0, 2 * sp.pi), (theta, 0, sp.pi))
    )
    tensor_coefficient = sp.simplify(
        sp.integrate(tensor_integrand, (phi, 0, 2 * sp.pi), (theta, 0, sp.pi))
    )

    return {
        "density": density,
        "polar_vector_component": s_theta,
        "vector_coefficient": vector_coefficient,
        "tensor_coefficient": tensor_coefficient,
    }


def _integrate(values: np.ndarray, x_weights: np.ndarray) -> complex:
    """Integrate an array sampled on Gauss-Legendre x and uniform phi."""

    return (2.0 * np.pi / values.shape[1]) * np.sum(x_weights[:, None] * values)


def _harmonic_with_derivatives(
    degree: int,
    order: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value, gradient, hessian = sph_harm_y(
        degree, order, theta, phi, diff_n=2
    )
    return value, gradient, hessian


def axial_projection(
    *,
    ell: int = 1,
    total_j: int = 2,
    orbital_l: int = 2,
    magnetic_m: int = 0,
    n_theta: int = 48,
    n_phi: int = 64,
) -> AxialProjection:
    """Project scalar bilinears onto axial vector and tensor harmonics.

    The vector bilinear is S_A = sum_sigma B_sigma^* D_A P_sigma.
    The trace-free-relevant tensor bilinear is
    Q_AB = sum_sigma D_(A B_sigma^* D_B) P_sigma.

    For a scalar perturbation with radial amplitude u and frequency sigma,
    these enter the matter parts of the relativistic stress tensor as

      delta T_tA = i F (2 omega + sigma) u S_A,
      delta T_rA = (F' u - F u') S_A,
      delta T_AB^odd = 2 F u Q_AB,

    for one sideband, up to the common stress-tensor normalization and the
    chosen overall harmonic phase.  The other sideband has the conjugate
    angular structure and its own radial amplitude.
    """

    if not abs(magnetic_m) <= total_j:
        raise ValueError("magnetic_m must satisfy |M| <= J")
    if not abs(orbital_l - ell) <= total_j <= orbital_l + ell:
        raise ValueError("(L, ell, J) violate angular-momentum addition")
    if n_theta < 8 or n_phi < 8:
        raise ValueError("quadrature resolution is too small")

    x, x_weights = np.polynomial.legendre.leggauss(n_theta)
    theta_1d = np.arccos(x)
    phi_1d = 2.0 * np.pi * np.arange(n_phi) / n_phi
    theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    shape = theta.shape
    density = np.zeros(shape, dtype=complex)
    s_theta = np.zeros(shape, dtype=complex)
    s_phi = np.zeros(shape, dtype=complex)
    q_tt = np.zeros(shape, dtype=complex)
    q_tp = np.zeros(shape, dtype=complex)
    q_pp = np.zeros(shape, dtype=complex)

    background_phase = (-1) ** ell / np.sqrt(2 * ell + 1)
    for spin_order in range(-ell, ell + 1):
        y_b, d_b, _ = _harmonic_with_derivatives(
            ell, spin_order, theta, phi
        )
        b_star = background_phase * y_b
        db_theta = background_phase * d_b[..., 0]
        db_phi = background_phase * d_b[..., 1]

        perturbation = np.zeros(shape, dtype=complex)
        dp_theta = np.zeros(shape, dtype=complex)
        dp_phi = np.zeros(shape, dtype=complex)
        for orbital_order in range(-orbital_l, orbital_l + 1):
            coefficient = complex(
                clebsch_gordan(
                    orbital_l,
                    ell,
                    total_j,
                    orbital_order,
                    spin_order,
                    magnetic_m,
                )
            )
            if coefficient == 0.0:
                continue
            y_p, d_p, _ = _harmonic_with_derivatives(
                orbital_l, orbital_order, theta, phi
            )
            perturbation += coefficient * y_p
            dp_theta += coefficient * d_p[..., 0]
            dp_phi += coefficient * d_p[..., 1]

        density += b_star * perturbation
        s_theta += b_star * dp_theta
        s_phi += b_star * dp_phi
        q_tt += db_theta * dp_theta
        q_tp += 0.5 * (db_theta * dp_phi + db_phi * dp_theta)
        q_pp += db_phi * dp_phi

    y_j, d_j, h_j = _harmonic_with_derivatives(
        total_j, magnetic_m, theta, phi
    )
    angular_eigenvalue = total_j * (total_j + 1)
    vector_scale = np.sqrt(angular_eigenvalue)
    x_theta = -d_j[..., 1] / (sin_theta * vector_scale)
    x_phi = sin_theta * d_j[..., 0] / vector_scale

    dtheta_x_theta = -(
        h_j[..., 0, 1] / sin_theta
        - d_j[..., 1] * cos_theta / sin_theta**2
    ) / vector_scale
    dphi_x_theta = -h_j[..., 1, 1] / (sin_theta * vector_scale)
    dtheta_x_phi = (
        cos_theta * d_j[..., 0] + sin_theta * h_j[..., 0, 0]
    ) / vector_scale
    dphi_x_phi = sin_theta * h_j[..., 1, 0] / vector_scale

    cot_theta = cos_theta / sin_theta
    x_tt = dtheta_x_theta
    x_tp = 0.5 * (
        dtheta_x_phi
        - cot_theta * x_phi
        + dphi_x_theta
        - cot_theta * x_phi
    )
    x_pp = dphi_x_phi + sin_theta * cos_theta * x_theta

    vector_inner = np.conjugate(x_theta) * s_theta
    vector_inner += np.conjugate(x_phi) * s_phi / sin_theta**2
    vector_norm_density = np.abs(s_theta) ** 2
    vector_norm_density += np.abs(s_phi) ** 2 / sin_theta**2
    x_vector_norm_density = np.abs(x_theta) ** 2
    x_vector_norm_density += np.abs(x_phi) ** 2 / sin_theta**2

    tensor_inner = np.conjugate(x_tt) * q_tt
    tensor_inner += 2.0 * np.conjugate(x_tp) * q_tp / sin_theta**2
    tensor_inner += np.conjugate(x_pp) * q_pp / sin_theta**4
    tensor_norm_density = np.abs(q_tt) ** 2
    tensor_norm_density += 2.0 * np.abs(q_tp) ** 2 / sin_theta**2
    tensor_norm_density += np.abs(q_pp) ** 2 / sin_theta**4
    x_tensor_norm_density = np.abs(x_tt) ** 2
    x_tensor_norm_density += 2.0 * np.abs(x_tp) ** 2 / sin_theta**2
    x_tensor_norm_density += np.abs(x_pp) ** 2 / sin_theta**4

    density_norm = float(np.sqrt(max(_integrate(np.abs(density) ** 2, x_weights).real, 0.0)))
    vector_source_norm = float(np.sqrt(max(_integrate(vector_norm_density, x_weights).real, 0.0)))
    vector_harmonic_norm = float(np.sqrt(max(_integrate(x_vector_norm_density, x_weights).real, 0.0)))
    tensor_source_norm = float(np.sqrt(max(_integrate(tensor_norm_density, x_weights).real, 0.0)))
    tensor_harmonic_norm = float(np.sqrt(max(_integrate(x_tensor_norm_density, x_weights).real, 0.0)))

    return AxialProjection(
        ell=ell,
        total_j=total_j,
        orbital_l=orbital_l,
        magnetic_m=magnetic_m,
        density_norm=density_norm,
        vector_coefficient=complex(_integrate(vector_inner, x_weights)),
        vector_source_norm=vector_source_norm,
        vector_harmonic_norm=vector_harmonic_norm,
        tensor_coefficient=complex(_integrate(tensor_inner, x_weights)),
        tensor_source_norm=tensor_source_norm,
        tensor_harmonic_norm=tensor_harmonic_norm,
        quadrature_theta=n_theta,
        quadrature_phi=n_phi,
    )
