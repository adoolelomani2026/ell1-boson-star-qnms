"""Closed neutral J=L=2 axial Einstein--Klein--Gordon radial system.

The state is Y=(h0,h1,u_plus,v_plus,u_minus,v_minus), with v=u'.
Harmonics and sideband phases match ``nonradial.axial_projection`` and the
paper.  The equations are assembled directly from delta G = 8 pi G delta T,
not from a convention-dependent master-equation transcription.
"""

from __future__ import annotations

from itertools import combinations
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from types import SimpleNamespace

from radial.coefficients import RadialBackground
from nonradial.riemann_sheet import SidebandSheet


J = 2
L = 2
MU = 1.0
G_NEWTON = 1.0
KAPPA_V = -1j / (2.0 * np.sqrt(np.pi))
_WEDGE_TRIPLES = tuple(combinations(range(6), 3))
_WEDGE_INDEX = {indices: position for position, indices in enumerate(_WEDGE_TRIPLES)}


def background_axial_scalar(background: RadialBackground, radius: float):
    """Return alpha,gamma,F,F',log derivatives,alpha'', and K0.

    F=sqrt(3) psi is the radial field in the normalized internal singlet
    basis.  K0 is the summed background scalar appearing in the explicit
    metric variation of the stress tensor.
    """

    point = background.point(radius)
    field = np.sqrt(3.0) * point.psi
    dfield = np.sqrt(3.0) * point.dpsi
    alpha2 = point.alpha**2
    gamma2 = point.gamma**2
    k0 = (
        -background.omega**2 * field**2 / alpha2
        + dfield**2 / gamma2
        + background.ell * (background.ell + 1.0) * field**2 / radius**2
        + MU**2 * field**2
    ) / (4.0 * np.pi)
    derivative_method = getattr(
        background,
        "equilibrium_lapse_second_derivative",
        background.lapse_second_derivative,
    )
    alpha_second = float(derivative_method(radius))
    return point, field, dfield, alpha_second, k0


def axial_rhs(
    radius: float,
    state: np.ndarray,
    sigma: complex,
    background: RadialBackground,
) -> np.ndarray:
    """Evaluate the six-component first-order axial EKG system."""

    if abs(sigma) < 1e-12:
        raise ValueError("this constraint-reduced form is singular at sigma=0")
    values = np.asarray(state, dtype=complex)
    if values.ndim not in (1, 2) or values.shape[0] != 6:
        raise ValueError("state must have shape (6,) or (6,n)")
    h0, h1, up, vp, um, vm = values
    point, field, dfield, alpha_second, k0 = background_axial_scalar(
        background, radius
    )
    alpha = point.alpha
    gamma = point.gamma
    alpha2 = alpha**2
    gamma2 = gamma**2
    la = point.log_alpha_prime
    lg = point.log_gamma_prime
    omega = background.omega
    c = 1.0 / (2.0 * np.sqrt(np.pi))

    # Axial AB Einstein equation.  The odd tensor stress has no algebraic
    # metric term in Regge--Wheeler gauge.
    dh1 = (
        -1j * sigma * gamma2 * h0 / alpha2
        + (lg - la) * h1
        + 8.0j * np.pi * G_NEWTON * c * gamma2 * field * (up + um)
    )

    # Axial rA Einstein equation.  ``geometric_h1`` is the coefficient of h1
    # in the directly linearized covariant Einstein tensor.
    geometric_h1 = (
        alpha_second / (alpha * gamma2)
        - la * lg / gamma2
        - lg / (radius * gamma2)
        + la / (radius * gamma2)
        + 2.0 / radius**2
    )
    radial_bilinear = dfield * (up + um) - field * (vp + vm)
    source_twice_alpha2 = (
        -8.0j * np.pi * G_NEWTON * c * alpha2 * radial_bilinear
        - 8.0 * np.pi * G_NEWTON * alpha2 * k0 * h1
    )
    dh0 = 2.0 * h0 / radius + (
        source_twice_alpha2
        + sigma**2 * h1
        - 2.0 * alpha2 * geometric_h1 * h1
    ) / (1j * sigma)

    first_derivative = 2.0 / radius + la - lg
    radial_metric_coupling = field * dh1 + (
        2.0 * dfield + field * (la - lg)
    ) * h1
    potential_plus = (
        (omega - sigma) ** 2 / alpha2 - MU**2 - L * (L + 1.0) / radius**2
    )
    potential_minus = (
        (omega + sigma) ** 2 / alpha2 - MU**2 - L * (L + 1.0) / radius**2
    )
    dvp = (
        -first_derivative * vp
        - gamma2 * potential_plus * up
        - KAPPA_V
        * 1j
        * gamma2
        * field
        * (2.0 * omega - sigma)
        * h0
        / (alpha2 * radius**2)
        + KAPPA_V * radial_metric_coupling / radius**2
    )
    dvm = (
        -first_derivative * vm
        - gamma2 * potential_minus * um
        + KAPPA_V
        * 1j
        * gamma2
        * field
        * (2.0 * omega + sigma)
        * h0
        / (alpha2 * radius**2)
        + KAPPA_V * radial_metric_coupling / radius**2
    )
    return np.array((dh0, dh1, vp, dvp, vm, dvm), dtype=complex)


def axial_ta_constraint_residual(
    radius: float,
    state: np.ndarray,
    state_prime: np.ndarray,
    h0_second: complex,
    sigma: complex,
    background: RadialBackground,
) -> tuple[complex, float]:
    """Evaluate the dependent ``tA`` Einstein equation and a local scale.

    The production evolution uses the ``rA`` and odd-tensor equations.  This
    unused equation is therefore a genuine constraint monitor.  ``h0_second``
    should be obtained by differentiating the evolved first derivative, not
    by substituting this dependent equation.
    """

    h0, h1, up, _, um, _ = np.asarray(state, dtype=complex)
    dh0, dh1 = np.asarray(state_prime, dtype=complex)[:2]
    point, field, _, alpha_second, k0 = background_axial_scalar(background, radius)
    inverse_gamma2 = 1.0 / point.gamma**2
    la = point.log_alpha_prime
    lg = point.log_gamma_prime
    terms = np.array(
        (
            -0.5j * sigma * inverse_gamma2 * dh1,
            0.5j * sigma * inverse_gamma2 * (lg + la) * h1,
            -0.5 * inverse_gamma2 * h0_second,
            0.5 * inverse_gamma2 * (lg + la) * dh0,
            inverse_gamma2 * (alpha_second / point.alpha - la * lg) * h0,
            -1j * sigma * inverse_gamma2 * h1 / radius,
            -2.0 * lg * inverse_gamma2 * h0 / radius,
            (2.0 + inverse_gamma2) * h0 / radius**2,
        ),
        dtype=complex,
    )
    c = 1.0 / (2.0 * np.sqrt(np.pi))
    direct_stress = 0.5 * c * field * (
        (2.0 * background.omega + sigma) * um
        - (2.0 * background.omega - sigma) * up
    )
    matter = 8.0 * np.pi * G_NEWTON * (direct_stress - 0.5 * k0 * h0)
    residual = complex(np.sum(terms) - matter)
    scale = float(np.sum(np.abs(terms)) + abs(matter) + 1e-300)
    return residual, scale


def axial_ta_constraint_profile(
    radii: np.ndarray,
    states: np.ndarray,
    sigma: complex,
    background: RadialBackground,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the dependent constraint on a sampled numerical solution."""

    sample = np.asarray(radii, dtype=float)
    values = np.asarray(states, dtype=complex)
    if values.shape != (6, len(sample)) or len(sample) < 5:
        raise ValueError("states must have shape (6,n) with n at least five")
    derivatives = np.column_stack(
        [axial_rhs(r, values[:, index], sigma, background) for index, r in enumerate(sample)]
    )
    # Differentiate the *evolved* h0' equation rather than substituting the
    # dependent tA equation.  The audit originally used ``np.gradient`` here,
    # whose second-order truncation error dominated this otherwise high-order
    # calculation and produced a non-convergent pointwise residual.  Use the
    # fourth-order centered stencil on the uniform production grids, retaining
    # the one-sided NumPy values only at the two points on either edge (which
    # callers already exclude from acceptance norms).
    spacing = np.diff(sample)
    if np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-14):
        h0_second = np.gradient(derivatives[0], sample, edge_order=2)
        step = spacing[0]
        h0_second[2:-2] = (
            derivatives[0, :-4]
            - 8.0 * derivatives[0, 1:-3]
            + 8.0 * derivatives[0, 3:-1]
            - derivatives[0, 4:]
        ) / (12.0 * step)
    else:
        h0_second = np.gradient(derivatives[0], sample, edge_order=2)
    residuals = np.empty(len(sample), dtype=complex)
    scales = np.empty(len(sample), dtype=float)
    for index, radius in enumerate(sample):
        residuals[index], scales[index] = axial_ta_constraint_residual(
            radius,
            values[:, index],
            derivatives[:, index],
            h0_second[index],
            sigma,
            background,
        )
    return residuals, scales


def axial_ta_constraint_profile_chain_rule(
    radii: np.ndarray,
    states: np.ndarray,
    sigma: complex,
    background: RadialBackground,
    *,
    derivative_step_multiplier: float = 1.0,
    derivative_method: str = "fourth_order",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the dependent constraint using ``Y''=A'Y+A^2Y``.

    The production first-order system is linear, ``Y'=A(r,sigma)Y``.  A
    centered fourth-order derivative of ``A`` therefore supplies ``h0''``
    without differentiating sampled mode data and without substituting the
    dependent Einstein equation being tested.  ``derivative_method='richardson'``
    combines step sizes ``h`` and ``h/2`` as ``(16 D4(h/2)-D4(h))/15`` to
    cancel the leading fourth-order truncation term.  A documented step scan
    is still required because background-interpolation noise grows as ``h``
    becomes too small.
    """

    sample = np.asarray(radii, dtype=float)
    values = np.asarray(states, dtype=complex)
    if derivative_step_multiplier <= 0.0:
        raise ValueError("derivative_step_multiplier must be positive")
    if derivative_method not in {"fourth_order", "richardson"}:
        raise ValueError("derivative_method must be 'fourth_order' or 'richardson'")
    if values.shape != (6, len(sample)):
        raise ValueError("states must have shape (6,n)")
    identity = np.eye(6, dtype=complex)
    residuals = np.empty(len(sample), dtype=complex)
    scales = np.empty(len(sample), dtype=float)
    for index, radius in enumerate(sample):
        generator = axial_rhs(radius, identity, sigma, background)
        step = derivative_step_multiplier * min(
            5.0e-4, max(1.0e-5, 5.0e-5 * radius)
        )
        if radius - 2.0 * step <= background.r_min:
            step = (radius - background.r_min) / 3.0
        def fourth_order_generator_derivative(local_step: float) -> np.ndarray:
            return (
                -axial_rhs(radius + 2.0 * local_step, identity, sigma, background)
                + 8.0 * axial_rhs(radius + local_step, identity, sigma, background)
                - 8.0 * axial_rhs(radius - local_step, identity, sigma, background)
                + axial_rhs(radius - 2.0 * local_step, identity, sigma, background)
            ) / (12.0 * local_step)

        generator_prime = fourth_order_generator_derivative(step)
        if derivative_method == "richardson":
            fine = fourth_order_generator_derivative(0.5 * step)
            generator_prime = (16.0 * fine - generator_prime) / 15.0
        state = values[:, index]
        state_prime = generator @ state
        h0_second = (generator_prime @ state + generator @ state_prime)[0]
        residuals[index], scales[index] = axial_ta_constraint_residual(
            radius,
            state,
            state_prime,
            h0_second,
            sigma,
            background,
        )
    return residuals, scales


def exterior_channel_wavenumbers(
    sigma: complex,
    omega: float,
    sheet: SidebandSheet | None = None,
):
    """Outgoing/decaying asymptotic wavenumbers on the physical sheets."""

    if sheet is not None:
        k_plus, k_minus = sheet.wavenumbers(sigma, omega, MU)
        return sigma, k_plus, k_minus

    def physical_sqrt(value: complex) -> complex:
        root = np.sqrt(complex(value))
        if root.imag < 0.0 or (abs(root.imag) < 1e-14 and root.real < 0.0):
            root = -root
        return root

    return (
        sigma,
        physical_sqrt((omega - sigma) ** 2 - MU**2),
        physical_sqrt((omega + sigma) ** 2 - MU**2),
    )


def center_basis(
    radius: float, sigma: complex, background: RadialBackground
) -> np.ndarray:
    """Coupled leading Frobenius basis at a small positive radius.

    The three free amplitudes are the regular metric coefficient H and the
    two scalar coefficients P,M.  The AB Einstein equation fixes the leading
    h1 coefficient B for each column.
    """

    basis = np.zeros((6, 3), dtype=complex)
    point, field, _, _, _ = background_axial_scalar(background, radius)
    field_slope = field / radius
    c = 1.0 / (2.0 * np.sqrt(np.pi))
    alpha2 = point.alpha**2

    # Metric-led solution: h0=H r^3 and h1=B r^4.
    basis[0, 0] = radius**3
    basis[1, 0] = (-1j * sigma / alpha2) * radius**4 / 4.0
    # Two independent regular scalar sidebands u~r^L.
    basis[2, 1] = radius**L
    basis[3, 1] = L * radius ** (L - 1)
    basis[1, 1] = 2.0j * np.pi * c * field_slope * radius**4
    basis[4, 2] = radius**L
    basis[5, 2] = L * radius ** (L - 1)
    basis[1, 2] = 2.0j * np.pi * c * field_slope * radius**4
    return basis


def integrate_regular_basis(
    sigma: complex,
    background: RadialBackground,
    *,
    r_start: float | None = None,
    r_end: float | None = None,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> np.ndarray:
    """Integrate the three regular basis solutions to the outer boundary."""

    left = max(background.r_min, 2e-4) if r_start is None else r_start
    right = background.r_max if r_end is None else r_end
    initial = center_basis(left, sigma, background).reshape(-1, order="F")

    def rhs(radius, flattened):
        matrix = flattened.reshape((6, 3), order="F")
        derivative = axial_rhs(radius, matrix, sigma, background)
        return derivative.reshape(-1, order="F")

    result = solve_ivp(
        rhs,
        (left, right),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"axial basis integration failed: {result.message}")
    return result.y[:, -1].reshape((6, 3), order="F")


def outgoing_residual_matrix(
    sigma: complex,
    background: RadialBackground,
    endpoint_basis: np.ndarray | None = None,
    *,
    r_end: float | None = None,
) -> np.ndarray:
    """Return the gravitational and two scalar outer residuals by column."""

    radius = background.r_max if r_end is None else r_end
    basis = (
        integrate_regular_basis(sigma, background, r_end=radius)
        if endpoint_basis is None
        else np.asarray(endpoint_basis, dtype=complex)
    )
    point = background.point(radius)
    _, k_plus, k_minus = exterior_channel_wavenumbers(sigma, background.omega)
    residuals = np.empty((3, 3), dtype=complex)
    for column in range(3):
        state = basis[:, column]
        derivative = axial_rhs(radius, state, sigma, background)
        h1 = state[1]
        residuals[0, column] = derivative[1] - (
            2.0 / radius
            + point.log_gamma_prime
            - point.log_alpha_prime
            + 1j * sigma * point.gamma / point.alpha
        ) * h1
        residuals[1, column] = state[3] - (
            1j * k_plus * point.gamma / point.alpha - 1.0 / radius
        ) * state[2]
        residuals[2, column] = state[5] - (
            1j * k_minus * point.gamma / point.alpha - 1.0 / radius
        ) * state[4]
    return residuals


def shooting_determinant(
    sigma: complex, background: RadialBackground, *, r_end: float | None = None
) -> complex:
    """Scale-invariant determinant of the three outgoing channel residuals."""

    matrix = outgoing_residual_matrix(sigma, background, r_end=r_end)
    row_norms = np.linalg.norm(matrix, axis=1)
    if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
        return complex(np.nan, np.nan)
    scaled = matrix / row_norms[:, None]
    column_norms = np.linalg.norm(scaled, axis=0)
    if np.any(column_norms == 0.0) or not np.all(np.isfinite(column_norms)):
        return complex(np.nan, np.nan)
    return complex(np.linalg.det(scaled / column_norms))


def exterior_basis(
    radius: float,
    sigma: complex,
    background: RadialBackground,
    *,
    riccati_radius: float = 300.0,
    asymptotic_order: int = 3,
    exterior_method: str | None = None,
    sheet: SidebandSheet | None = None,
) -> np.ndarray:
    """Outgoing/decaying basis propagated through the vacuum exterior."""

    method = exterior_method
    if method is None:
        method = "complex_scaled" if asymptotic_order > 3 else "real_axis"
    if method not in {"real_axis", "complex_scaled", "complex_scaled_coulomb"}:
        raise ValueError(
            "exterior_method must be 'real_axis', 'complex_scaled', or "
            "'complex_scaled_coulomb'"
        )
    if method in {"complex_scaled", "complex_scaled_coulomb"}:
        return exterior_complex_scaled_basis(
            radius,
            sigma,
            background.omega,
            background.adm_mass,
            order=asymptotic_order,
            sheet=sheet,
            ray_length=max(80.0, riccati_radius - radius),
            coulomb_resummed=method == "complex_scaled_coulomb",
        )
    gravity, plus, minus = exterior_log_derivatives(
        radius,
        sigma,
        background.omega,
        background.adm_mass,
        r_far=riccati_radius,
        asymptotic_order=asymptotic_order,
        sheet=sheet,
        return_states=True,
    )
    basis = np.zeros((6, 3), dtype=complex)
    basis[0:2, 0] = gravity
    basis[2:4, 1] = plus
    basis[4:6, 2] = minus
    return basis


def exterior_complex_scaled_basis(
    radius: float,
    sigma: complex,
    omega: float,
    mass: float,
    *,
    order: int = 12,
    sheet: SidebandSheet | None = None,
    ray_length: float = 286.0,
    coulomb_resummed: bool = False,
) -> np.ndarray:
    """Pole-free outgoing basis from exterior complex scaling.

    Each channel starts from an outgoing Coulomb/tortoise Jost pair on a fixed
    complex ray, which is integrated directly to the real match point.  The
    standard path uses an ``order``-term inverse-radius scalar recurrence; the
    ``complex_scaled_coulomb`` selector instead resums the near-threshold
    minus-channel Coulomb tail with high-precision Coulomb functions.  Both
    paths propagate analytic value/derivative pairs without component
    division.  Convergence must be checked in both ``order`` (where relevant)
    and ``ray_length`` (passed through ``r_far``).
    """

    _, k_plus, k_minus = exterior_channel_wavenumbers(sigma, omega, sheet)
    if order < 1:
        raise ValueError("asymptotic order must be positive")
    if ray_length < 80.0:
        raise ValueError("complex-scaled ray length must be at least 80")

    def complex_ray_state(
        wavenumber: complex,
        exponent: complex,
        initial_pair: tuple[complex, complex],
        rhs,
        *,
        direction: complex,
        length: float,
    ) -> np.ndarray:
        if wavenumber == 0.0:
            raise ValueError("complex scaling cannot start at a channel threshold")
        far_radius = radius + length * direction
        value, derivative = initial_pair(far_radius)
        state = np.asarray((value, derivative), dtype=complex)

        def scaled_rhs(distance: float, channel_state: np.ndarray) -> np.ndarray:
            complex_radius = radius + distance * direction
            return direction * rhs(complex_radius, channel_state)

        right = length
        while right > 0.0:
            left = max(0.0, right - 20.0)
            result = solve_ivp(
                scaled_rhs,
                (right, left),
                state,
                method="DOP853",
                rtol=2.0e-9,
                atol=2.0e-11,
            )
            if not result.success:
                raise RuntimeError(
                    f"complex-scaled exterior integration failed: {result.message}"
                )
            state = result.y[:, -1]
            # Remove the exact, analytic, nonzero plane-wave growth acquired
            # on this inward ray segment.  This is a change of Evans
            # normalization only; it prevents a harmless exponential phase
            # from dominating contour resolution.
            right_radius = radius + right * direction
            left_radius = radius + left * direction
            growth = np.exp(
                1j * wavenumber * (left_radius - right_radius)
                + exponent * (np.log(left_radius) - np.log(right_radius))
            )
            state /= growth
            state /= max(abs(state[0]), abs(state[1]), 1.0e-300)
            right = left
        return state

    def scalar_pair(
        frequency: complex,
        wavenumber: complex,
        *,
        direction: complex,
        use_coulomb: bool = False,
    ) -> np.ndarray:
        exponent = 2j * mass * wavenumber + 1j * mass * MU**2 / wavenumber - 1.0
        def initial_pair(complex_radius: complex) -> tuple[complex, complex]:
            if use_coulomb:
                # A single Whittaker W represents outgoing Coulomb H^(+)
                # without combining the separately branched and exponentially
                # large F and G solutions.  Apart from an irrelevant nonzero
                # scalar normalization,
                # H^(+)_L(eta,rho) = W_{-i eta,L+1/2}(-2 i rho).
                with mp.workdps(50):
                    eta = -(2.0 * mass * wavenumber + mass * MU**2 / wavenumber)
                    rho = wavenumber * complex_radius
                    argument = -2j * rho
                    kappa = -1j * eta
                    order_parameter = L + 0.5
                    outgoing_value = mp.whitw(kappa, order_parameter, argument)
                    outgoing_argument_derivative = mp.diff(
                        lambda value: mp.whitw(
                            kappa, order_parameter, value
                        ),
                        argument,
                    )
                    value = outgoing_value / complex_radius
                    derivative = (
                        -2j
                        * wavenumber
                        * outgoing_argument_derivative
                        / complex_radius
                        - outgoing_value / complex_radius**2
                    )
                    asymptotic_factor = mp.exp(
                        -0.5 * argument + kappa * mp.log(argument)
                    ) / complex_radius
                    value /= asymptotic_factor
                    derivative /= asymptotic_factor
                    scale = max(abs(value), abs(derivative), mp.mpf("1e-300"))
                    return complex(value / scale), complex(derivative / scale)
            # For u=e^(ikr) r^exponent sum_n a_n r^-n, substitute directly
            # into the exact exterior scalar equation.  The recurrence is
            # inexpensive and, unlike a fixed three-term log derivative,
            # remains controllable when inverse powers of k grow close to a
            # massive threshold.
            coefficients = [1.0 + 0.0j]
            p_series = np.zeros(order + 2, dtype=complex)
            q_series = np.zeros(order + 2, dtype=complex)
            for power in range(1, order + 2):
                p_series[power] = (
                    2.0
                    if power == 1
                    else 2.0 * mass * (2.0 * mass) ** (power - 2)
                )
                q_series[power] = (
                    frequency**2 * (power + 1) * (2.0 * mass) ** power
                    - MU**2 * (2.0 * mass) ** power
                    - (
                        L * (L + 1.0) * (2.0 * mass) ** (power - 2)
                        if power >= 2
                        else 0.0
                    )
                )
            for index in range(1, order + 1):
                target = index + 1
                total = 0.0j
                source = target - 2
                total += (
                    (exponent - source)
                    * (exponent - source - 1.0)
                    * coefficients[source]
                )
                for power in range(1, target + 1):
                    source = target - power
                    if source < len(coefficients):
                        total += (
                            p_series[power] * 1j * wavenumber
                            + q_series[power]
                        ) * coefficients[source]
                    source = target - power - 1
                    if 0 <= source < len(coefficients):
                        total += (
                            p_series[power]
                            * (exponent - source)
                            * coefficients[source]
                        )
                coefficients.append(-total / (-2j * wavenumber * index))
            powers = complex_radius ** (-np.arange(order + 1))
            series = np.dot(coefficients, powers)
            derivative_series = np.dot(
                [
                    (exponent - index) * coefficients[index]
                    for index in range(order + 1)
                ],
                powers,
            ) / complex_radius
            # Keep the analytic two-component series pair.  Dividing by
            # ``series`` would reintroduce a meromorphic line chart: a zero
            # of the truncated amplitude would masquerade as a negative
            # Evans winding even though the outgoing one-plane is regular.
            return series, 1j * wavenumber * series + derivative_series

        def scalar_rhs(complex_radius: complex, state: np.ndarray) -> np.ndarray:
            lapse_squared = 1.0 - 2.0 * mass / complex_radius
            first_derivative = (
                2.0 / complex_radius
                + 2.0 * mass / (complex_radius**2 * lapse_squared)
            )
            potential = (
                frequency**2 / lapse_squared**2
                - MU**2 / lapse_squared
                - L * (L + 1.0) / (complex_radius**2 * lapse_squared)
            )
            return np.asarray(
                (state[1], -first_derivative * state[1] - potential * state[0]),
                dtype=complex,
            )

        return complex_ray_state(
            wavenumber,
            exponent,
            initial_pair,
            scalar_rhs,
            direction=direction,
            length=ray_length,
        )

    rw_exponent = 2j * mass * sigma
    def rw_initial(complex_radius: complex) -> tuple[complex, complex]:
        psi_log_star = 1j * sigma
        if order >= 2:
            psi_log_star += (-3j / sigma) / complex_radius**2
        if order >= 3:
            psi_log_star += (
                -3.0 / sigma**2 + 9j * mass / sigma
            ) / complex_radius**3
        lapse_squared = 1.0 - 2.0 * mass / complex_radius
        return 1.0 + 0.0j, psi_log_star / lapse_squared

    def rw_rhs(complex_radius: complex, state: np.ndarray) -> np.ndarray:
        lapse_squared = 1.0 - 2.0 * mass / complex_radius
        lapse_derivative = 2.0 * mass / complex_radius**2
        potential = lapse_squared * (
            L * (L + 1.0) / complex_radius**2
            - 6.0 * mass / complex_radius**3
        )
        return np.asarray(
            (
                state[1],
                -lapse_derivative / lapse_squared * state[1]
                - (sigma**2 - potential) / lapse_squared**2 * state[0],
            ),
            dtype=complex,
        )

    master, master_derivative = complex_ray_state(
        sigma,
        rw_exponent,
        rw_initial,
        rw_rhs,
        direction=np.exp(2j * np.pi / 3.0),
        length=ray_length,
    )
    lapse_squared = 1.0 - 2.0 * mass / radius
    gravity_h1 = master
    gravity_h0 = lapse_squared**2 * (
        master / radius + master_derivative
    ) / (-1j * sigma)
    plus = scalar_pair(omega - sigma, k_plus, direction=1.0 + 0.0j)
    minus = scalar_pair(
        omega + sigma,
        k_minus,
        direction=np.exp(-1j * np.pi / 3.0),
        use_coulomb=coulomb_resummed,
    )
    basis = np.zeros((6, 3), dtype=complex)
    basis[0:2, 0] = (gravity_h0, gravity_h1)
    basis[2:4, 1] = plus
    basis[4:6, 2] = minus
    return basis


def exterior_log_derivatives(
    radius: float,
    sigma: complex,
    omega: float,
    mass: float,
    *,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    sheet: SidebandSheet | None = None,
    return_states: bool = False,
) -> tuple[complex, complex, complex] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate vacuum-channel ratios from large Schwarzschild radius.

    Six decoupled linear variables are integrated in short inward segments
    and renormalized after every segment.  This retains the physical solution
    through zeros of an individual amplitude, where a Riccati ratio would
    develop a pole, while preventing closed scalar channels from overflowing.
    """

    if r_far <= radius:
        raise ValueError("r_far must exceed the exterior matching radius")
    if asymptotic_order not in (1, 2, 3):
        raise ValueError("asymptotic_order must be 1, 2, or 3")
    _, k_plus, k_minus = exterior_channel_wavenumbers(sigma, omega, sheet)

    def geometry(r):
        f = 1.0 - 2.0 * mass / r
        alpha = np.sqrt(f)
        gamma = 1.0 / alpha
        la = mass / (r**2 * f)
        lg = -la
        alpha_second = -2.0 * mass / (r**3 * alpha) - mass**2 / (
            r**4 * alpha**3
        )
        return alpha, gamma, la, lg, alpha_second

    alpha_far, gamma_far, la_far, lg_far, _ = geometry(r_far)
    f_far = alpha_far**2

    def flat_hankel_log_derivative(
        wavenumber: complex, evaluation_radius: float = r_far
    ) -> complex:
        argument = wavenumber * evaluation_radius
        polynomial = 1.0 + 3.0j / argument - 3.0 / argument**2
        polynomial_prime = -3.0j / argument**2 + 6.0 / argument**3
        return wavenumber * (
            1.0j - 1.0 / argument + polynomial_prime / polynomial
        )

    def rw_master_log_derivative() -> complex:
        """Outgoing Regge--Wheeler d ln(Psi)/dr_* through O(r^-3)."""

        if mass == 0.0:
            # The Regge--Wheeler master field is Psi=r h_l in flat space.
            return flat_hankel_log_derivative(sigma) + 1.0 / r_far
        a2 = -3.0j / sigma
        a3 = -3.0 / sigma**2 + 9.0j * mass / sigma
        value = 1.0j * sigma
        if asymptotic_order >= 2:
            value += a2 / r_far**2
        if asymptotic_order >= 3:
            value += a3 / r_far**3
        return value

    def scalar_u_log_derivative(channel_frequency: complex, wavenumber: complex) -> complex:
        """Outgoing/decaying d ln(u)/dr_* including the massive Coulomb tail."""

        if mass == 0.0:
            return flat_hankel_log_derivative(wavenumber)
        a1 = 1.0j * mass * MU**2 / wavenumber
        a2 = (a1 + L * (L + 1.0) - a1**2) / (2.0j * wavenumber)
        a3 = -(
            2.0 * a1 * a2
            + 2.0 * mass * a1
            - 2.0 * a2
            - 2.0 * mass
            + 2.0 * mass * L * (L + 1.0)
        ) / (2.0j * wavenumber)
        psi_log = 1.0j * wavenumber + a1 / r_far
        if asymptotic_order >= 2:
            psi_log += a2 / r_far**2
        if asymptotic_order >= 3:
            psi_log += a3 / r_far**3
        # u=Psi/r and d r/dr_*=f.
        return psi_log - f_far / r_far

    qg_far = (
        1.0 / r_far
        + lg_far
        - la_far
        + rw_master_log_derivative() * gamma_far / alpha_far
    )
    x_far = alpha_far**2 * (qg_far - lg_far + la_far) / (
        -1j * sigma * gamma_far**2
    )
    initial = np.array(
        (
            x_far,
            1.0,
            1.0,
            scalar_u_log_derivative(omega - sigma, k_plus) * gamma_far / alpha_far,
            1.0,
            scalar_u_log_derivative(omega + sigma, k_minus) * gamma_far / alpha_far,
        ),
        dtype=complex,
    )

    def vacuum_rhs(r, state):
        h0, h1, up, vp_state, um, vm_state = state
        alpha, gamma, la, lg, alpha_second = geometry(r)
        alpha2 = alpha**2
        gamma2 = gamma**2
        geometric_h1 = (
            alpha_second / (alpha * gamma2)
            - la * lg / gamma2
            - lg / (r * gamma2)
            + la / (r * gamma2)
            + 2.0 / r**2
        )
        coefficient_h1 = (
            sigma**2 - 2.0 * alpha2 * geometric_h1
        ) / (1j * sigma)
        coefficient_h0_in_h1 = -1j * sigma * gamma2 / alpha2
        coefficient_h1_in_h1 = lg - la
        first_derivative = 2.0 / r + la - lg
        potential_plus = gamma2 * (
            (omega - sigma) ** 2 / alpha2 - MU**2 - L * (L + 1.0) / r**2
        )
        potential_minus = gamma2 * (
            (omega + sigma) ** 2 / alpha2 - MU**2 - L * (L + 1.0) / r**2
        )
        return np.array(
            (
                2.0 * h0 / r + coefficient_h1 * h1,
                coefficient_h0_in_h1 * h0 + coefficient_h1_in_h1 * h1,
                vp_state,
                -first_derivative * vp_state - potential_plus * up,
                vm_state,
                -first_derivative * vm_state - potential_minus * um,
            ),
            dtype=complex,
        )

    state = initial
    right = r_far
    while right > radius:
        left = max(radius, right - 12.0)
        result = solve_ivp(
            vacuum_rhs,
            (right, left),
            state,
            method="DOP853",
            rtol=2e-9,
            atol=2e-11,
        )
        if not result.success:
            raise RuntimeError(f"vacuum exterior integration failed: {result.message}")
        state = result.y[:, -1]
        for start in (0, 2, 4):
            scale = max(abs(state[start]), abs(state[start + 1]), 1e-300)
            state[start : start + 2] /= scale
        right = left
    if return_states:
        # Every segment was divided only by a positive real magnitude.  The
        # returned homogeneous vectors therefore retain the analytic
        # solution's contour phase without introducing poles at zeros of an
        # arbitrarily selected amplitude component.
        return state[0:2].copy(), state[2:4].copy(), state[4:6].copy()
    return complex(state[0] / state[1]), complex(state[3] / state[2]), complex(
        state[5] / state[4]
    )


def integrate_outgoing_basis(
    sigma: complex,
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    exterior_method: str | None = None,
    sheet: SidebandSheet | None = None,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> np.ndarray:
    """Integrate the three physical exterior channel solutions inward."""

    initial_matrix = exterior_basis(
        r_end,
        sigma,
        background,
        riccati_radius=r_far,
        asymptotic_order=asymptotic_order,
        exterior_method=exterior_method,
        sheet=sheet,
    )
    if r_end == r_match:
        return initial_matrix
    initial = initial_matrix.reshape(-1, order="F")

    def rhs(radius, flattened):
        matrix = flattened.reshape((6, 3), order="F")
        derivative = axial_rhs(radius, matrix, sigma, background)
        return derivative.reshape(-1, order="F")

    result = solve_ivp(
        rhs,
        (r_end, r_match),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"outgoing basis integration failed: {result.message}")
    return result.y[:, -1].reshape((6, 3), order="F")


def _integrate_orthonormal_basis(
    initial_basis: np.ndarray,
    interval: tuple[float, float],
    sigma: complex,
    background: RadialBackground,
    *,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Propagate a three-plane while suppressing fundamental-matrix collapse."""

    basis, _ = np.linalg.qr(np.asarray(initial_basis, dtype=complex), mode="reduced")
    left, target = interval
    direction = 1.0 if target > left else -1.0
    while direction * (target - left) > 0.0:
        right = left + direction * min(0.5, abs(target - left))

        def rhs(radius: float, flattened: np.ndarray) -> np.ndarray:
            matrix = flattened.reshape((6, 3), order="F")
            return axial_rhs(radius, matrix, sigma, background).reshape(
                -1, order="F"
            )

        result = solve_ivp(
            rhs,
            (left, right),
            basis.reshape(-1, order="F"),
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(
                f"orthonormal basis integration failed: {result.message}"
            )
        propagated = result.y[:, -1].reshape((6, 3), order="F")
        basis, _ = np.linalg.qr(propagated, mode="reduced")
        left = right
    return basis


def matching_matrix(
    sigma: complex,
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    exterior_method: str | None = None,
    sheet: SidebandSheet | None = None,
    r_start: float | None = None,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> np.ndarray:
    """Return the unnormalized two-sided six-dimensional matching matrix."""

    interior = integrate_regular_basis(
        sigma,
        background,
        r_start=r_start,
        r_end=r_match,
        rtol=rtol,
        atol=atol,
    )
    exterior = integrate_outgoing_basis(
        sigma,
        background,
        r_match=r_match,
        r_end=r_end,
        r_far=r_far,
        asymptotic_order=asymptotic_order,
        exterior_method=exterior_method,
        sheet=sheet,
        rtol=rtol,
        atol=atol,
    )
    return np.column_stack((interior, -exterior))


def matching_raw_determinant(
    sigma: complex,
    background: RadialBackground,
    **kwargs,
) -> complex:
    """Direct matching determinant without row or column conditioning.

    Positive real integration rescalings preserve its zeros and phase but do
    not preserve holomorphy.  It is useful as an independent local root check;
    the exterior-algebra determinant is preferred for contour winding.
    """

    return complex(np.linalg.det(matching_matrix(sigma, background, **kwargs)))


def _plucker_coordinates(basis: np.ndarray) -> np.ndarray:
    """Return the twenty 3-by-3 minors representing a three-plane."""

    matrix = np.asarray(basis, dtype=complex)
    if matrix.shape != (6, 3):
        raise ValueError("basis must have shape (6,3)")
    return np.asarray(
        [np.linalg.det(matrix[np.asarray(indices), :]) for indices in _WEDGE_TRIPLES],
        dtype=complex,
    )


def _third_compound_generator(matrix: np.ndarray) -> np.ndarray:
    """Induced generator on the third exterior power of C^6.

    If ``Y'=A Y`` and ``z`` contains all three-row minors of ``Y``, then
    ``z'=C_3(A) z``.  Evolving ``z`` directly prevents independent basis
    columns from collapsing onto the fastest growing solution and preserves
    holomorphic dependence on the complex frequency.
    """

    a = np.asarray(matrix, dtype=complex)
    if a.shape != (6, 6):
        raise ValueError("matrix must have shape (6,6)")
    compound = np.zeros((20, 20), dtype=complex)
    for row_index, rows in enumerate(_WEDGE_TRIPLES):
        for position, old_row in enumerate(rows):
            for new_row in range(6):
                if new_row in rows and new_row != old_row:
                    continue
                replaced = list(rows)
                replaced[position] = new_row
                inversions = sum(
                    replaced[left] > replaced[right]
                    for left in range(3)
                    for right in range(left + 1, 3)
                )
                column_rows = tuple(sorted(replaced))
                compound[row_index, _WEDGE_INDEX[column_rows]] += (
                    (-1.0 if inversions % 2 else 1.0) * a[old_row, new_row]
                )
    return compound


def _integrate_wedge(
    initial_basis: np.ndarray,
    interval: tuple[float, float],
    sigma: complex,
    background: RadialBackground,
    *,
    rtol: float,
    atol: float,
) -> np.ndarray:
    identity = np.eye(6, dtype=complex)

    def rhs(radius: float, wedge: np.ndarray) -> np.ndarray:
        generator = axial_rhs(radius, identity, sigma, background)
        return _third_compound_generator(generator) @ wedge

    left, target = interval
    wedge = _plucker_coordinates(initial_basis)
    direction = 1.0 if target > left else -1.0
    while direction * (target - left) > 0.0:
        right = left + direction * min(4.0, abs(target - left))
        result = solve_ivp(
            rhs,
            (left, right),
            wedge,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(
                f"exterior-algebra integration failed: {result.message}"
            )
        wedge = result.y[:, -1]
        wedge /= max(float(np.max(np.abs(wedge))), 1.0e-300)
        left = right
    return wedge


def matching_evans_determinant(
    sigma: complex,
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    exterior_method: str | None = None,
    sheet: SidebandSheet | None = None,
    r_start: float | None = None,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> complex:
    """Phase-preserving Evans determinant from third exterior powers.

    This is the production determinant for root refinement and contour counts.
    No singular value, QR phase convention, or stored root is used.  Positive
    real segment rescalings prevent overflow while preserving contour phase
    and zeros.  The normalization is therefore not assumed holomorphic; its
    zeros are intersections of the regular-center and outgoing exterior
    three-planes.
    """

    left = max(background.r_min, 2e-4) if r_start is None else r_start
    if not background.r_min <= left < r_match:
        raise ValueError("r_start must lie inside the interior integration domain")
    interior = _integrate_wedge(
        center_basis(left, sigma, background),
        (left, r_match),
        sigma,
        background,
        rtol=rtol,
        atol=atol,
    )
    exterior_initial = exterior_basis(
        r_end,
        sigma,
        background,
        riccati_radius=r_far,
        asymptotic_order=asymptotic_order,
        exterior_method=exterior_method,
        sheet=sheet,
    )
    if r_end == r_match:
        exterior = _plucker_coordinates(exterior_initial)
    else:
        exterior = _integrate_wedge(
            exterior_initial,
            (r_end, r_match),
            sigma,
            background,
            rtol=rtol,
            atol=atol,
        )
    value = 0.0j
    all_rows = set(range(6))
    for indices, left_minor in zip(_WEDGE_TRIPLES, interior):
        complement = tuple(sorted(all_rows.difference(indices)))
        inversions = sum(i > j for i in indices for j in complement)
        sign = -1.0 if inversions % 2 else 1.0
        value += sign * left_minor * exterior[_WEDGE_INDEX[complement]]
    return complex(value)


def matching_determinant(
    sigma: complex,
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    exterior_method: str | None = None,
    sheet: SidebandSheet | None = None,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> complex:
    """Conditioned determinant used for local complex-root refinement.

    The Euclidean row/column scaling is intentionally not used for argument-
    principle root counts; use :func:`matching_evans_determinant` for those.
    """

    matrix = matching_matrix(
        sigma,
        background,
        r_match=r_match,
        r_end=r_end,
        r_far=r_far,
        asymptotic_order=asymptotic_order,
        exterior_method=exterior_method,
        sheet=sheet,
        rtol=rtol,
        atol=atol,
    )
    row_norms = np.linalg.norm(matrix, axis=1)
    if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
        return complex(np.nan, np.nan)
    scaled = matrix / row_norms[:, None]
    column_norms = np.linalg.norm(scaled, axis=0)
    if np.any(column_norms == 0.0) or not np.all(np.isfinite(column_norms)):
        return complex(np.nan, np.nan)
    return complex(np.linalg.det(scaled / column_norms))


def matching_singular_value(
    sigma: complex,
    background: RadialBackground,
    *,
    r_match: float = 12.0,
    r_end: float = 26.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    exterior_method: str | None = None,
    sheet: SidebandSheet | None = None,
    rtol: float = 2e-6,
    atol: float = 2e-8,
) -> float:
    """Smallest principal-angle singular value of the matched three-planes.

    Each basis is reorthogonalized between short integration segments.  This
    diagnostic therefore remains sensitive to a true subspace intersection
    even where direct fundamental-matrix columns collapse numerically.
    """

    left = max(background.r_min, 2e-4)
    interior = _integrate_orthonormal_basis(
        center_basis(left, sigma, background),
        (left, r_match),
        sigma,
        background,
        rtol=rtol,
        atol=atol,
    )
    exterior = _integrate_orthonormal_basis(
        exterior_basis(
            r_end,
            sigma,
            background,
            riccati_radius=r_far,
            asymptotic_order=asymptotic_order,
            exterior_method=exterior_method,
            sheet=sheet,
        ),
        (r_end, r_match),
        sigma,
        background,
        rtol=rtol,
        atol=atol,
    )
    singular_values = np.linalg.svd(
        np.column_stack((interior, -exterior)), compute_uv=False
    )
    return float(singular_values[-1])


def matching_mode_coefficients(
    sigma: complex,
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    exterior_method: str | None = None,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return interior/exterior coefficients of the least-singular match."""

    interior = integrate_regular_basis(
        sigma, background, r_end=r_match, rtol=rtol, atol=atol
    )
    exterior = integrate_outgoing_basis(
        sigma,
        background,
        r_match=r_match,
        r_end=r_end,
        r_far=r_far,
        asymptotic_order=asymptotic_order,
        exterior_method=exterior_method,
        rtol=rtol,
        atol=atol,
    )
    matrix = np.column_stack((interior, -exterior))
    row_norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-300)
    row_scaled = matrix / row_norms[:, None]
    column_norms = np.maximum(np.linalg.norm(row_scaled, axis=0), 1e-300)
    normalized = row_scaled / column_norms
    _, singular_values, vh = np.linalg.svd(normalized)
    normalized_null = vh.conj().T[:, -1]
    raw_null = normalized_null / column_norms
    scale = max(np.linalg.norm(raw_null[:3]), 1e-300)
    return raw_null[:3] / scale, raw_null[3:] / scale, float(singular_values[-1])


def interior_mode_profile(
    radii: np.ndarray,
    sigma: complex,
    background: RadialBackground,
    interior_coefficients: np.ndarray,
    *,
    r_start: float | None = None,
    rtol: float = 2e-9,
    atol: float = 2e-11,
) -> np.ndarray:
    """Sample the regular matched mode on radii inside the matching point."""

    sample = np.asarray(radii, dtype=float)
    left = max(background.r_min, 2e-4) if r_start is None else r_start
    if np.any(sample < left) or np.any(np.diff(sample) <= 0):
        raise ValueError("mode radii must increase and lie above r_start")
    coefficients = np.asarray(interior_coefficients, dtype=complex)
    if coefficients.shape != (3,):
        raise ValueError("interior_coefficients must have shape (3,)")
    initial = center_basis(left, sigma, background) @ coefficients
    result = solve_ivp(
        lambda radius, state: axial_rhs(radius, state, sigma, background),
        (left, float(sample[-1])),
        initial,
        t_eval=sample,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"mode-profile integration failed: {result.message}")
    return result.y


class _SchwarzschildFrequencyBackground:
    """Minimal background interface for exact exterior frequency evolution."""

    ell = 1

    def __init__(self, mass: float, omega: float):
        self.adm_mass = mass
        self.omega = omega

    def point(self, radius: float):
        f = 1.0 - 2.0 * self.adm_mass / radius
        alpha = np.sqrt(f)
        la = self.adm_mass / (radius**2 * f)
        return SimpleNamespace(
            alpha=alpha,
            gamma=1.0 / alpha,
            psi=0.0,
            dpsi=0.0,
            log_alpha_prime=la,
            log_gamma_prime=-la,
        )

    def lapse_second_derivative(self, radius: float):
        f = 1.0 - 2.0 * self.adm_mass / radius
        alpha = np.sqrt(f)
        return np.asarray(
            -2.0 * self.adm_mass / (radius**3 * alpha)
            - self.adm_mass**2 / (radius**4 * alpha**3)
        )


def exterior_mode_profile(
    radii: np.ndarray,
    sigma: complex,
    background: RadialBackground,
    exterior_coefficients: np.ndarray,
    *,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    rtol: float = 2e-9,
    atol: float = 2e-11,
) -> np.ndarray:
    """Sample the physical exterior combination on either side of r_end."""

    sample = np.asarray(radii, dtype=float)
    if np.any(np.diff(sample) <= 0):
        raise ValueError("mode radii must be strictly increasing")
    coefficients = np.asarray(exterior_coefficients, dtype=complex)
    if coefficients.shape != (3,):
        raise ValueError("exterior_coefficients must have shape (3,)")
    initial = exterior_basis(
        r_end,
        sigma,
        background,
        riccati_radius=r_far,
        asymptotic_order=asymptotic_order,
    ) @ coefficients
    output = np.empty((6, sample.size), dtype=complex)
    lower_mask = sample <= r_end
    if np.any(lower_mask):
        lower = sample[lower_mask]
        result = solve_ivp(
            lambda radius, state: axial_rhs(radius, state, sigma, background),
            (r_end, float(lower[0])),
            initial,
            t_eval=lower[::-1],
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(f"inward exterior-profile integration failed: {result.message}")
        output[:, lower_mask] = result.y[:, ::-1]
    upper_mask = sample > r_end
    if np.any(upper_mask):
        upper = sample[upper_mask]
        vacuum = _SchwarzschildFrequencyBackground(
            background.adm_mass, background.omega
        )

        def metric_rhs(radius, metric_state):
            active = background if radius <= background.r_max else vacuum
            full_state = np.zeros(6, dtype=complex)
            full_state[:2] = metric_state
            return axial_rhs(radius, full_state, sigma, active)[:2]

        result = solve_ivp(
            metric_rhs,
            (r_end, float(upper[-1])),
            initial[:2],
            t_eval=upper,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(f"outward exterior-profile integration failed: {result.message}")
        output[:2, upper_mask] = result.y

        # Closed scalar tails are evaluated analytically.  Outward linear
        # integration would amplify roundoff in the exponentially growing
        # solution even when the physical coefficient is exactly zero.
        _, k_plus, k_minus = exterior_channel_wavenumbers(sigma, background.omega)
        mass = background.adm_mass

        def tortoise(radius_values):
            return radius_values + 2.0 * mass * np.log(radius_values / (2.0 * mass) - 1.0)

        star_end = tortoise(np.asarray(r_end))
        star_upper = tortoise(upper)

        def stable_tail(amplitude, wavenumber):
            z0 = wavenumber * star_end
            z = wavenumber * star_upper
            p0 = 1.0 + 3.0j / z0 - 3.0 / z0**2
            polynomial = 1.0 + 3.0j / z - 3.0 / z**2
            ratio = np.exp(1.0j * (z - z0)) * (z0 / z) * polynomial / p0
            values = amplitude * ratio
            polynomial_prime = -3.0j / z**2 + 6.0 / z**3
            log_derivative_star = wavenumber * (
                1.0j - 1.0 / z + polynomial_prime / polynomial
            )
            radial_log_derivative = log_derivative_star / (
                1.0 - 2.0 * mass / upper
            )
            return values, radial_log_derivative * values

        output[2, upper_mask], output[3, upper_mask] = stable_tail(initial[2], k_plus)
        output[4, upper_mask], output[5, upper_mask] = stable_tail(initial[4], k_minus)
    return output


def integrate_incoming_channel(
    sigma: float,
    background: RadialBackground,
    channel: str,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> np.ndarray:
    """Integrate one real-frequency incoming exterior channel inward."""

    if not np.isreal(sigma):
        raise ValueError("scattering channels require a real frequency")
    outgoing = exterior_basis(
        r_end,
        complex(sigma),
        background,
        riccati_radius=r_far,
        asymptotic_order=asymptotic_order,
    )
    if channel == "gravity":
        # For the e^{-i sigma t} convention the metric first-order pair has
        # h1'=-i sigma (...) h0+... .  Reversing the radial wave therefore
        # conjugates h1 but sends h0 -> -conj(h0).  Plain conjugation is not
        # a solution of the same real-frequency system and spuriously breaks
        # unit flux even in exact flat vacuum.
        initial = np.conjugate(outgoing[:, 0])
        initial[0] *= -1.0
    elif channel == "minus":
        _, _, k_minus = exterior_channel_wavenumbers(complex(sigma), background.omega)
        if abs(k_minus.imag) > 1e-10:
            raise ValueError("minus scalar channel is closed at this frequency")
        initial = np.conjugate(outgoing[:, 2])
    else:
        raise ValueError("channel must be 'gravity' or 'minus'")
    result = solve_ivp(
        lambda radius, state: axial_rhs(radius, state, complex(sigma), background),
        (r_end, r_match),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"incoming channel integration failed: {result.message}")
    return result.y[:, -1]


def scattering_amplitudes(
    sigma: float,
    background: RadialBackground,
    incident_channel: str = "gravity",
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> dict[str, object]:
    """Solve the regular scattering problem for one unit incoming channel.

    Returned amplitudes use the exterior basis normalization h1=1 for the
    gravitational channel and u=1 for each scalar channel at ``r_end``.
    """

    frequency = float(sigma)
    interior = integrate_regular_basis(
        complex(frequency), background, r_end=r_match, rtol=rtol, atol=atol
    )
    outgoing = integrate_outgoing_basis(
        complex(frequency),
        background,
        r_match=r_match,
        r_end=r_end,
        r_far=r_far,
        asymptotic_order=asymptotic_order,
        rtol=rtol,
        atol=atol,
    )
    incoming = integrate_incoming_channel(
        frequency,
        background,
        incident_channel,
        r_match=r_match,
        r_end=r_end,
        r_far=r_far,
        asymptotic_order=asymptotic_order,
        rtol=rtol,
        atol=atol,
    )
    match = np.column_stack((interior, -outgoing))
    row_norms = np.maximum(np.linalg.norm(match, axis=1), 1e-300)
    row_scaled = match / row_norms[:, None]
    column_norms = np.maximum(np.linalg.norm(row_scaled, axis=0), 1e-300)
    scaled_solution = np.linalg.solve(
        row_scaled / column_norms, incoming / row_norms
    ) / column_norms
    outgoing_coefficients = scaled_solution[3:]
    _, k_plus, k_minus = exterior_channel_wavenumbers(
        complex(frequency), background.omega
    )
    return {
        "sigma": frequency,
        "incident_channel": incident_channel,
        "gravity_out": complex(outgoing_coefficients[0]),
        "plus_decay": complex(outgoing_coefficients[1]),
        "minus_out": complex(outgoing_coefficients[2]),
        "k_plus": complex(k_plus),
        "k_minus": complex(k_minus),
        "linear_residual": float(
            np.linalg.norm(match @ scaled_solution - incoming)
            / max(np.linalg.norm(incoming), 1e-300)
        ),
        "interior_coefficients": np.asarray(scaled_solution[:3], dtype=complex),
    }


def scattering_matrix(
    sigma: float,
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    r_far: float = 300.0,
    asymptotic_order: int = 3,
    rtol: float = 2e-8,
    atol: float = 2e-10,
) -> tuple[np.ndarray, float]:
    """Return the raw two-open-channel scattering matrix efficiently.

    Columns correspond to incident gravity and the open minus sideband; rows
    correspond to outgoing gravity and minus-sideband amplitudes.  The common
    regular and outgoing fundamental matrices are evaluated only once.
    """

    frequency = float(sigma)
    _, _, k_minus = exterior_channel_wavenumbers(
        complex(frequency), background.omega
    )
    if abs(k_minus.imag) > 1e-10:
        raise ValueError("the minus scalar channel is closed at this frequency")
    interior = integrate_regular_basis(
        complex(frequency), background, r_end=r_match, rtol=rtol, atol=atol
    )
    outgoing = integrate_outgoing_basis(
        complex(frequency),
        background,
        r_match=r_match,
        r_end=r_end,
        r_far=r_far,
        asymptotic_order=asymptotic_order,
        rtol=rtol,
        atol=atol,
    )
    incoming = np.column_stack(
        tuple(
            integrate_incoming_channel(
                frequency,
                background,
                channel,
                r_match=r_match,
                r_end=r_end,
                r_far=r_far,
                asymptotic_order=asymptotic_order,
                rtol=rtol,
                atol=atol,
            )
            for channel in ("gravity", "minus")
        )
    )
    match = np.column_stack((interior, -outgoing))
    row_norms = np.maximum(np.linalg.norm(match, axis=1), 1e-300)
    row_scaled = match / row_norms[:, None]
    column_norms = np.maximum(np.linalg.norm(row_scaled, axis=0), 1e-300)
    solution = np.linalg.solve(
        row_scaled / column_norms, incoming / row_norms[:, None]
    ) / column_norms[:, None]
    residual = np.linalg.norm(match @ solution - incoming) / max(
        np.linalg.norm(incoming), 1e-300
    )
    return np.asarray(solution[3:][[0, 2], :], dtype=complex), float(residual)
