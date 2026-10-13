"""Equatorial Schwarzschild photon-geodesic ODE solver using u = 1/r and angle phi.

Conserved Quantities & Impact Parameter Definition
--------------------------------------------------
Along an equatorial (theta = pi / 2) null geodesic x^mu(lambda) in static, spherically
symmetric Schwarzschild spacetime with geometric units (G = c = 1), time-translation
invariance (Killing vector partial_t) and axial symmetry (Killing vector partial_phi)
yield two conserved quantities per unit affine parameter lambda:

    E = -g_tt (dt / dlambda) = (1 - 2M / r) (dt / dlambda)    (conserved energy)
    L =  g_phiphi (dphi / dlambda) = r^2 (dphi / dlambda)     (conserved angular momentum)

For a future-directed photon (E > 0) with positive azimuthal motion (L > 0, so phi
increases monotonically along the trajectory), the impact parameter is defined as:

    b = L / E

At spatial infinity (r -> infinity, where 1 - 2M/r -> 1), b equals the perpendicular
Euclidean distance from the optical axis to the asymptotic incoming ray.

Dimensional vs. Dimensionless Quantities (Geometric Units G = c = 1)
--------------------------------------------------------------------
1. Dimensional quantities (length scale [L] or inverse length [L^-1]):
   - Black hole mass parameter: M                     [L]
   - Radial coordinate & reference radii: r, r_s, r_ph [L]
   - Impact parameter: b = L / E, b_crit              [L]
   - Inverse radius: u = 1 / r                        [L^-1]
   - Angular derivative: v = du / dphi                [L^-1]
   - Second derivative: d^2u / dphi^2                 [L^-1]
   - First-integral radicand: R(u; M, b)              [L^-2]

2. Dimensionless quantities [1]:
   - Azimuthal angle & span: phi, phi_0, phi_max, delta_phi [rad]
   - Metric lapse function: f(r) = 1 - 2M / r               [1]
   - Mass-normalized radius: r / M                          [1]
   - Horizon-normalized radius: r / r_s = r / (2M)          [1]
   - Dimensionless compactness: M * u = M / r               [1]
   - Mass-normalized impact parameter: b / M                [1]
   - Criticality ratio: b / b_crit = b / (3 * sqrt(3) * M)  [1]
   - Local orthonormal pitch angle sine:
         sin(psi_0) = (b / r_0) * sqrt(1 - 2M / r_0)        [1]

Assumptions for Finite-Radius Incoming Initial Conditions
---------------------------------------------------------
Substituting E, L, and u = 1 / r into the null condition ds^2 = 0 yields the exact
first integral:
    (du / dphi)^2 = 1 / b^2 - u^2 (1 - 2 M u) = 1 / b^2 - u^2 + 2 M u^3

To initialize a photon at a finite radius r_0 > 2M (u_0 = 1 / r_0) and angle phi_0:
1. Exterior vacuum domain: r_0 must lie strictly outside the configured capture
   boundary r_capture >= r_s = 2M.
2. Classically allowed region: The radicand R(u_0; M, b) = 1/b^2 - u_0^2 + 2Mu_0^3
   must be non-negative, equivalently sin(psi_0) = (b / r_0) sqrt(1 - 2M / r_0) <= 1.
   Otherwise r_0 lies inside the potential barrier for impact parameter b, and a
   ValueError is raised.
3. Exact curved-spacetime initialization: Rather than using a flat-space approximation
   (which drops the 2Mu_0^3 general-relativistic correction), we evaluate the full
   Schwarzschild radicand at u_0 = 1 / r_0.
4. Incoming vs. outgoing branch choice: Because u = 1 / r implies
   du / dphi = -(1 / r^2) (dr / dphi), an incoming photon (dr / dphi < 0 as phi
   increases) takes the positive root v_0 = +sqrt(R(u_0; M, b)), whereas an outgoing
   photon (dr / dphi > 0) takes the negative root v_0 = -sqrt(R(u_0; M, b)).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from blackhole_lab.physics.schwarzschild import (
    _ensure_finite_scalar,
    critical_impact_parameter,
    metric_coefficient_f,
    reference_scales,
    schwarzschild_radius,
    validate_impact_parameter,
    validate_mass,
    validate_radius,
)

RadialDirection = Literal["ingoing", "outgoing"]


class SolverStatus(str, Enum):
    """High-level outcome classification of the numerical geodesic integration."""

    COMPLETED = "completed"
    CAPTURE_BOUNDARY = "capture_boundary"
    INTEGRATION_LIMIT = "integration_limit"
    SOLVER_FAILURE = "solver_failure"


class TerminationReason(str, Enum):
    """Specific physical or numerical reason why integration terminated."""

    OUTER_BOUNDARY_REACHED = "outer_boundary_reached"
    CAPTURE_BOUNDARY_REACHED = "capture_boundary_reached"
    PHI_MAX_REACHED = "phi_max_reached"
    SOLVER_STEP_FAILURE = "solver_step_failure"
    INVALID_DOMAIN_ENTERED = "invalid_domain_entered"


@dataclass(frozen=True)
class PhotonInitialConditions:
    """Validated initial state separating dimensional and dimensionless quantities.

    Dimensional Attributes (Geometric Units G = c = 1)
    --------------------------------------------------
    M : float
        Black hole mass parameter [L].
    r0 : float
        Initial Schwarzschild radial coordinate [L].
    b : float
        Impact parameter b = L / E [L].
    u0 : float
        Initial inverse radius u_0 = 1 / r_0 [L^-1].
    du_dphi0 : float
        Initial angular derivative v_0 = (du / dphi)|_{phi_0} [L^-1].

    Dimensionless Attributes
    ------------------------
    phi0 : float
        Initial azimuthal angle [rad].
    direction : RadialDirection
        Initial radial branch ("ingoing" with v_0 >= 0 or "outgoing" with v_0 <= 0).
    r0_over_M : float
        Mass-normalized initial radius r_0 / M [1].
    r0_over_rs : float
        Horizon-normalized initial radius r_0 / (2M) [1].
    b_over_M : float
        Mass-normalized impact parameter b / M [1].
    b_over_b_crit : float
        Ratio to critical impact parameter b / (3 * sqrt(3) * M) [1].
    u0_M : float
        Dimensionless compactness parameter at start M * u_0 = M / r_0 [1].
    sin_local_pitch_angle : float
        Sine of the angle psi_0 between the photon trajectory and the radial direction
        in a local static orthonormal frame at r_0:
            sin(psi_0) = (b / r_0) * sqrt(1 - 2M / r_0) in [0, 1].
    """

    # Dimensional quantities (G = c = 1)
    M: float
    r0: float
    b: float
    u0: float
    du_dphi0: float
    # Dimensionless quantities
    phi0: float
    direction: RadialDirection
    r0_over_M: float
    r0_over_rs: float
    b_over_M: float
    b_over_b_crit: float
    u0_M: float
    sin_local_pitch_angle: float

    @property
    def state_vector(self) -> NDArray[np.float64]:
        """Initial ODE state vector y_0 = [u_0, v_0]^T with units [L^-1]."""
        return np.array([self.u0, self.du_dphi0], dtype=np.float64)


@dataclass(frozen=True)
class GeodesicConfig:
    """Configuration parameters for the equatorial photon geodesic solver.

    Parameters
    ----------
    M : float
        Black hole mass parameter (M > 0) in geometric units (G = c = 1) [L].
    r0 : float
        Initial radial coordinate (must satisfy r0 > 2M and r0 > capture_radius) [L].
    b : float
        Photon impact parameter b = L / E (b > 0) in geometric units [L].
    direction : RadialDirection, default "ingoing"
        Initial radial motion: "ingoing" (dr/dphi < 0, du/dphi > 0) or
        "outgoing" (dr/dphi > 0, du/dphi < 0).
    phi0 : float, default 0.0
        Initial azimuthal angle in radians [1].
    phi_max : float, default 6 * pi
        Maximum azimuthal angle to integrate up to (must satisfy phi_max > phi0) [1].
    capture_radius_factor : float, default 1.001
        Dimensionless capture boundary factor relative to r_s = 2M (must be >= 1.0) [1].
        Integration stops if r <= capture_radius_factor * r_s.
    outer_radius : float | None, default None
        Outer radial truncation boundary r_max for outgoing trajectories [L].
        If None, defaults to max(1.5 * r0, 50.0 * M).
    rtol : float, default 1e-9
        Dimensionless relative tolerance passed to ``scipy.integrate.solve_ivp`` [1].
    atol : float, default 1e-11
        Absolute tolerance passed to ``scipy.integrate.solve_ivp``.
    method : str, default "DOP853"
        Explicit Runge-Kutta integration method for ``solve_ivp`` ("DOP853" or "RK45").
    max_step : float, default 0.05
        Maximum angular step size dphi in radians [1].
    """

    M: float
    r0: float
    b: float
    direction: RadialDirection = "ingoing"
    phi0: float = 0.0
    phi_max: float = 6.0 * math.pi
    capture_radius_factor: float = 1.001
    outer_radius: float | None = None
    rtol: float = 1e-9
    atol: float = 1e-11
    method: str = "DOP853"
    max_step: float = 0.05

    def __post_init__(self) -> None:
        mass = validate_mass(self.M)
        r0_val = validate_radius(self.r0, mass, require_exterior=True)
        _ = validate_impact_parameter(self.b)
        phi0_val = _ensure_finite_scalar(self.phi0, "Initial angle phi0")
        phi_max_val = _ensure_finite_scalar(self.phi_max, "Maximum angle phi_max")
        if phi_max_val <= phi0_val:
            raise ValueError(
                f"phi_max ({phi_max_val}) must be strictly greater than phi0 ({phi0_val})."
            )

        if self.direction not in ("ingoing", "outgoing"):
            raise ValueError(
                f"direction must be 'ingoing' or 'outgoing'; got {self.direction!r}."
            )

        cap_factor = _ensure_finite_scalar(
            self.capture_radius_factor, "capture_radius_factor"
        )
        if cap_factor < 1.0:
            raise ValueError(
                f"capture_radius_factor must be >= 1.0 (at or outside horizon r_s = 2M); "
                f"got {cap_factor}."
            )

        r_cap = cap_factor * schwarzschild_radius(mass)
        if r0_val <= r_cap:
            raise ValueError(
                f"Initial radius r0 = {r0_val} must be strictly greater than the configured "
                f"capture boundary r_capture = {r_cap}."
            )

        if self.outer_radius is not None:
            outer_r = _ensure_finite_scalar(self.outer_radius, "outer_radius")
            if outer_r <= r0_val:
                raise ValueError(
                    f"outer_radius ({outer_r}) must be strictly greater than r0 ({r0_val})."
                )

        rtol_val = _ensure_finite_scalar(self.rtol, "rtol")
        atol_val = _ensure_finite_scalar(self.atol, "atol")
        if not (1e-14 <= rtol_val <= 1e-2):
            raise ValueError(
                f"rtol must be in the safe range [1e-14, 1e-2]; got {rtol_val}."
            )
        if not (1e-15 <= atol_val <= 1e-2):
            raise ValueError(
                f"atol must be in the safe range [1e-15, 1e-2]; got {atol_val}."
            )

        max_step_val = _ensure_finite_scalar(self.max_step, "max_step")
        if max_step_val <= 0.0:
            raise ValueError(f"max_step must be strictly positive; got {max_step_val}.")

        if self.method not in ("RK45", "DOP853", "RK23", "Radau", "BDF", "LSODA"):
            raise ValueError(f"Unsupported solve_ivp method: {self.method!r}.")

    @property
    def capture_radius(self) -> float:
        """Dimensional capture boundary radius r_capture = capture_radius_factor * 2M [L]."""
        return self.capture_radius_factor * schwarzschild_radius(self.M)

    @property
    def effective_outer_radius(self) -> float:
        """Dimensional outer truncation boundary radius r_max [L]."""
        if self.outer_radius is not None:
            return float(self.outer_radius)
        return max(1.5 * self.r0, 50.0 * self.M)

    @property
    def delta_phi_max(self) -> float:
        """Dimensionless maximum angular integration interval phi_max - phi0 [rad]."""
        return float(self.phi_max - self.phi0)

    @property
    def r0_over_M(self) -> float:
        """Dimensionless starting radius r0 / M [1]."""
        return float(self.r0 / self.M)

    @property
    def b_over_M(self) -> float:
        """Dimensionless impact parameter b / M [1]."""
        return float(self.b / self.M)

    @property
    def b_over_b_crit(self) -> float:
        """Dimensionless criticality ratio b / b_crit [1]."""
        return float(self.b / critical_impact_parameter(self.M))


@dataclass(frozen=True)
class GeodesicResult:
    """Structured output of a Schwarzschild equatorial photon-geodesic integration.

    Attributes
    ----------
    config : GeodesicConfig
        Configuration used for the simulation.
    phi : NDArray[np.float64]
        Dimensionless azimuthal angles (radians) along the trajectory [1].
    u : NDArray[np.float64]
        Dimensional inverse radii u = 1 / r along the trajectory [L^-1].
    du_dphi : NDArray[np.float64]
        Dimensional first derivatives du / dphi along the trajectory [L^-1].
    r : NDArray[np.float64]
        Dimensional radial coordinates r = 1 / u along the trajectory [L].
    x : NDArray[np.float64]
        Dimensional Cartesian coordinate x = r * cos(phi) in the equatorial plane [L].
    y : NDArray[np.float64]
        Dimensional Cartesian coordinate y = r * sin(phi) in the equatorial plane [L].
    status : SolverStatus
        High-level classification of the integration outcome.
    termination_reason : TerminationReason
        Specific termination reason.
    solver_message : str
        Human-readable message describing the solver outcome.
    scipy_status : int
        Raw status code returned by ``scipy.integrate.solve_ivp``.
    r_s : float
        Dimensional Schwarzschild radius 2M [L].
    r_ph : float
        Dimensional photon-sphere radius 3M [L].
    b_crit : float
        Dimensional critical impact parameter 3 * sqrt(3) * M [L].
    b_over_b_crit : float
        Dimensionless impact parameter ratio b / b_crit [1].
    diagnostics : object
        Instance of ``GeodesicDiagnostics`` with first-integral residuals and metadata.
    initial_conditions : PhotonInitialConditions | None, default None
        Validated initial state with dimensional and dimensionless parameters.
    """

    config: GeodesicConfig
    phi: NDArray[np.float64]
    u: NDArray[np.float64]
    du_dphi: NDArray[np.float64]
    r: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    status: SolverStatus
    termination_reason: TerminationReason
    solver_message: str
    scipy_status: int
    r_s: float
    r_ph: float
    b_crit: float
    b_over_b_crit: float
    diagnostics: object
    initial_conditions: PhotonInitialConditions | None = None

    @property
    def is_captured(self) -> bool:
        """True only if the trajectory physically reached the capture boundary.

        Unfinished integrations (PHI_MAX_REACHED or SOLVER_FAILURE) always return False,
        even when b < b_crit.
        """
        return (
            self.status == SolverStatus.CAPTURE_BOUNDARY
            and self.termination_reason == TerminationReason.CAPTURE_BOUNDARY_REACHED
        )

    @property
    def reached_outer_boundary(self) -> bool:
        """True only if the trajectory completed by reaching the outer radial boundary.

        Unfinished integrations (PHI_MAX_REACHED or SOLVER_FAILURE) always return False,
        even when b > b_crit.
        """
        return (
            self.status == SolverStatus.COMPLETED
            and self.termination_reason == TerminationReason.OUTER_BOUNDARY_REACHED
        )

    @property
    def is_unfinished(self) -> bool:
        """True if integration stopped at phi_max or encountered a solver/domain failure."""
        return self.status in (
            SolverStatus.INTEGRATION_LIMIT,
            SolverStatus.SOLVER_FAILURE,
        )

    @property
    def r_over_M(self) -> NDArray[np.float64]:
        """Dimensionless radial coordinate profile r(phi) / M [1]."""
        return self.r / self.config.M

    @property
    def u_M(self) -> NDArray[np.float64]:
        """Dimensionless inverse radial profile M * u(phi) = M / r(phi) [1]."""
        return self.u * self.config.M

    @property
    def delta_phi(self) -> NDArray[np.float64]:
        """Dimensionless angular displacement phi - phi0 [rad]."""
        return self.phi - self.config.phi0


def first_integral_radicand(u: float, M: float, b: float) -> float:
    """Compute the dimensional first-integral radicand R(u; M, b) = 1/b^2 - u^2 + 2Mu^3 [L^-2].

    In the Schwarzschild equatorial plane, (du/dphi)^2 = R(u; M, b).
    A valid real trajectory requires R(u; M, b) >= 0.

    Parameters
    ----------
    u : float
        Dimensional inverse radius u = 1 / r (must be finite and strictly positive) [L^-1].
    M : float
        Dimensional black hole mass parameter (M > 0) [L].
    b : float
        Dimensional impact parameter b = L / E (b > 0) [L].

    Returns
    -------
    float
        Value of 1/b^2 - u^2 + 2*M*u^3 [L^-2].
    """
    mass = validate_mass(M)
    b_val = validate_impact_parameter(b)
    u_val = _ensure_finite_scalar(u, "Inverse radius u")
    if u_val <= 0.0:
        raise ValueError(f"Inverse radius u must be strictly positive; got {u_val}.")

    return (1.0 / (b_val * b_val)) - (u_val * u_val) + (2.0 * mass * (u_val**3))


def initial_du_dphi(
    r0: float,
    M: float,
    b: float,
    direction: RadialDirection = "ingoing",
) -> float:
    """Compute the initial derivative (du/dphi)|_{phi_0} [L^-1] from r0, M, b, and direction.

    Validates that the initial state at finite radius r0 does not lie in a classically
    forbidden region where the first-integral radicand 1/b^2 - u_0^2 + 2Mu_0^3 is negative.

    Parameters
    ----------
    r0 : float
        Starting radial coordinate (r0 > 2M) [L].
    M : float
        Black hole mass parameter (M > 0) [L].
    b : float
        Impact parameter b = L / E (b > 0) [L].
    direction : RadialDirection, default "ingoing"
        "ingoing" returns +sqrt(radicand) (since u = 1/r increases as r decreases);
        "outgoing" returns -sqrt(radicand).

    Returns
    -------
    float
        Initial value of du/dphi [L^-1].

    Raises
    ------
    ValueError
        If the radicand is negative (classically forbidden initial condition) or inputs
        are invalid.
    """
    mass = validate_mass(M)
    radius0 = validate_radius(r0, mass, require_exterior=True)
    b_val = validate_impact_parameter(b)
    if direction not in ("ingoing", "outgoing"):
        raise ValueError(f"direction must be 'ingoing' or 'outgoing'; got {direction!r}.")

    u0 = 1.0 / radius0
    radicand = first_integral_radicand(u0, mass, b_val)

    # Allow tiny negative round-off only within machine precision of an exact turning point
    if -1e-15 <= radicand < 0.0:
        radicand = 0.0
    elif radicand < 0.0:
        raise ValueError(
            f"Impossible initial conditions: at r0 = {radius0} (u0 = {u0:.6g}) and "
            f"b = {b_val}, the first-integral radicand (du/dphi)^2 = {radicand:.6e} < 0. "
            "The starting radius lies inside the classically forbidden potential barrier."
        )

    mag = math.sqrt(radicand)
    return mag if direction == "ingoing" else -mag


def compute_initial_conditions(
    M: float,
    r0: float,
    b: float,
    phi0: float = 0.0,
    direction: RadialDirection = "ingoing",
) -> PhotonInitialConditions:
    """Construct validated incoming or outgoing photon initial conditions at finite radius r0.

    Reuses ``reference_scales``, ``metric_coefficient_f``, and ``initial_du_dphi`` to
    compute both dimensional quantities (in geometric units G = c = 1) and dimensionless
    ratios, including the local static orthonormal pitch angle sin(psi_0).

    Parameters
    ----------
    M : float
        Black hole mass parameter (M > 0) [L].
    r0 : float
        Initial Schwarzschild radius (r0 > 2M) [L].
    b : float
        Impact parameter b = L / E (b > 0) [L].
    phi0 : float, default 0.0
        Initial azimuthal angle in radians [1].
    direction : RadialDirection, default "ingoing"
        Initial radial direction ("ingoing" or "outgoing").

    Returns
    -------
    PhotonInitialConditions
        Structured initial state separating dimensional and dimensionless parameters.
    """
    scales = reference_scales(M)
    radius0 = validate_radius(r0, scales.M, require_exterior=True)
    b_val = validate_impact_parameter(b)
    phi0_val = _ensure_finite_scalar(phi0, "Initial angle phi0")

    v0 = initial_du_dphi(radius0, scales.M, b_val, direction=direction)
    u0 = 1.0 / radius0
    f_r0 = metric_coefficient_f(radius0, scales.M, require_exterior=True)
    sin_psi0 = min(1.0, (b_val / radius0) * math.sqrt(f_r0))

    return PhotonInitialConditions(
        M=scales.M,
        r0=radius0,
        b=b_val,
        u0=u0,
        du_dphi0=v0,
        phi0=phi0_val,
        direction=direction,
        r0_over_M=radius0 / scales.M,
        r0_over_rs=radius0 / scales.r_s,
        b_over_M=b_val / scales.M,
        b_over_b_crit=b_val / scales.b_crit,
        u0_M=scales.M * u0,
        sin_local_pitch_angle=sin_psi0,
    )


def orbit_ode_rhs(
    phi: float,
    state: NDArray[np.float64] | Sequence[float],
    M: float,
) -> tuple[float, float]:
    """Right-hand side of the first-order Schwarzschild equatorial photon orbit ODE.

    Converts d^2u/dphi^2 = -u + 3Mu^2 into:
        du/dphi = v
        dv/dphi = -u + 3 * M * u^2

    Parameters
    ----------
    phi : float
        Independent angular variable (radians) [1]. Validated for finiteness.
    state : sequence of 2 floats
        State vector [u, v] where u = 1/r [L^-1] and v = du/dphi [L^-1].
    M : float
        Black hole mass parameter (M > 0) [L].

    Returns
    -------
    tuple[float, float]
        Derivatives (du/dphi, dv/dphi) with units [L^-1].

    Raises
    ------
    TypeError
        If phi, M, or state elements are not real numeric scalars.
    ValueError
        If phi, M, or state elements are non-finite, M <= 0, or len(state) != 2.
    """
    _ = _ensure_finite_scalar(phi, "Azimuthal angle phi")
    mass = validate_mass(M)

    if isinstance(state, (str, bytes)) or not hasattr(state, "__len__"):
        raise TypeError("state must be a sequence or 1D array of length 2: [u, v].")
    if len(state) != 2:
        raise ValueError(f"state must have length 2 ([u, v]); got length {len(state)}.")

    u = _ensure_finite_scalar(state[0], "State component u")
    v = _ensure_finite_scalar(state[1], "State component v")

    du = v
    dv = -u + 3.0 * mass * (u * u)
    if not (math.isfinite(du) and math.isfinite(dv)):
        raise ValueError(
            f"ODE right-hand side overflowed to non-finite values for u={u}, v={v}, M={mass}."
        )
    return (du, dv)


def solve_photon_geodesic(config: GeodesicConfig) -> GeodesicResult:
    """Integrate an equatorial photon geodesic in Schwarzschild spacetime.

    Uses ``scipy.integrate.solve_ivp`` with terminal event detection for:
    1. Capture boundary: r <= r_capture (u >= 1 / r_capture) while ingoing.
    2. Outer boundary: r >= r_outer (u <= 1 / r_outer) while outgoing.
    3. Invalid domain guard: u <= 0 (r -> infinity / sign flip).

    Critically, solver failures (status == -1, non-finite output, or domain violations)
    and unfinished integrations (reaching phi_max) are never classified as physical
    capture or escape.

    Parameters
    ----------
    config : GeodesicConfig
        Validated solver and physical configuration.

    Returns
    -------
    GeodesicResult
        Complete trajectory, status classification, characteristic scales, initial
        conditions, and first-integral numerical diagnostics.
    """
    if not isinstance(config, GeodesicConfig):
        raise TypeError(
            f"config must be an instance of GeodesicConfig; got {type(config).__name__}."
        )

    from blackhole_lab.validation.diagnostics import evaluate_geodesic_diagnostics

    scales = reference_scales(config.M)
    init_cond = compute_initial_conditions(
        M=config.M,
        r0=config.r0,
        b=config.b,
        phi0=config.phi0,
        direction=config.direction,
    )
    y0 = init_cond.state_vector

    u_capture = 1.0 / config.capture_radius
    u_outer = 1.0 / config.effective_outer_radius
    u_horizon = 1.0 / scales.r_s

    def capture_event(phi: float, y: NDArray[np.float64]) -> float:
        _ = phi
        return float(y[0] - u_capture)

    capture_event.terminal = True  # type: ignore[attr-defined]
    capture_event.direction = 1.0  # type: ignore[attr-defined]

    def outer_boundary_event(phi: float, y: NDArray[np.float64]) -> float:
        _ = phi
        return float(y[0] - u_outer)

    outer_boundary_event.terminal = True  # type: ignore[attr-defined]
    outer_boundary_event.direction = -1.0  # type: ignore[attr-defined]

    def invalid_u_event(phi: float, y: NDArray[np.float64]) -> float:
        _ = phi
        return float(y[0])

    invalid_u_event.terminal = True  # type: ignore[attr-defined]
    invalid_u_event.direction = -1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        fun=lambda phi, y: orbit_ode_rhs(phi, y, scales.M),
        t_span=(config.phi0, config.phi_max),
        y0=y0,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
        events=[capture_event, outer_boundary_event, invalid_u_event],
    )

    phi_arr = np.asarray(sol.t, dtype=np.float64)
    u_arr = np.asarray(sol.y[0], dtype=np.float64)
    v_arr = np.asarray(sol.y[1], dtype=np.float64)

    has_nonfinite = not (
        np.all(np.isfinite(phi_arr))
        and np.all(np.isfinite(u_arr))
        and np.all(np.isfinite(v_arr))
    )

    # Check if trajectory crossed strictly inside the event horizon r < 2M (u > 1/(2M))
    crossed_horizon = bool(
        np.any(np.isfinite(u_arr) & (u_arr > u_horizon * (1.0 + 1e-12)))
    )

    # Determine status and termination reason without conflating failure or phi_max with capture/escape
    if sol.status == -1 or has_nonfinite or phi_arr.size < 2:
        status = SolverStatus.SOLVER_FAILURE
        termination_reason = TerminationReason.SOLVER_STEP_FAILURE
        solver_message = (
            f"Numerical solver failed (solve_ivp status={sol.status}): {sol.message}. "
            "Trajectory cannot be considered physically valid."
        )
    elif (
        (
            sol.status == 1
            and sol.t_events is not None
            and len(sol.t_events) == 3
            and len(sol.t_events[2]) > 0
        )
        or np.any(u_arr <= 0.0)
        or crossed_horizon
    ):
        status = SolverStatus.SOLVER_FAILURE
        termination_reason = TerminationReason.INVALID_DOMAIN_ENTERED
        solver_message = (
            "Integration entered an invalid radial domain (u <= 0 or r < 2M). "
            "Result truncated or rejected."
        )
    elif (
        sol.status == 1
        and sol.t_events is not None
        and len(sol.t_events[0]) > 0
    ):
        status = SolverStatus.CAPTURE_BOUNDARY
        termination_reason = TerminationReason.CAPTURE_BOUNDARY_REACHED
        solver_message = (
            f"Trajectory reached the configured capture boundary "
            f"r_capture = {config.capture_radius:.6g} (capture_radius_factor = "
            f"{config.capture_radius_factor})."
        )
    elif (
        sol.status == 1
        and sol.t_events is not None
        and len(sol.t_events[1]) > 0
    ):
        status = SolverStatus.COMPLETED
        termination_reason = TerminationReason.OUTER_BOUNDARY_REACHED
        solver_message = (
            f"Trajectory scattered and reached the configured finite outer radial boundary "
            f"r_outer = {config.effective_outer_radius:.6g} (note: reaching a finite outer "
            "boundary is a domain truncation, not an exact evaluation at r = infinity)."
        )
    else:
        status = SolverStatus.INTEGRATION_LIMIT
        termination_reason = TerminationReason.PHI_MAX_REACHED
        solver_message = (
            f"Integration reached the configured maximum angle phi_max = "
            f"{config.phi_max:.6g} rad before crossing either the capture or outer "
            "radial boundary. Trajectory is unfinished."
        )

    # Compute r, x, y safely only where u > 0 and finite
    valid_mask = np.isfinite(u_arr) & (u_arr > 0.0)
    r_arr = np.full_like(u_arr, np.nan, dtype=np.float64)
    r_arr[valid_mask] = 1.0 / u_arr[valid_mask]
    x_arr = r_arr * np.cos(phi_arr)
    y_arr = r_arr * np.sin(phi_arr)

    diagnostics = evaluate_geodesic_diagnostics(
        phi=phi_arr,
        u=u_arr,
        du_dphi=v_arr,
        M=scales.M,
        b=config.b,
        rtol=config.rtol,
        atol=config.atol,
        method=config.method,
        status=status,
        termination_reason=termination_reason,
    )

    return GeodesicResult(
        config=config,
        phi=phi_arr,
        u=u_arr,
        du_dphi=v_arr,
        r=r_arr,
        x=x_arr,
        y=y_arr,
        status=status,
        termination_reason=termination_reason,
        solver_message=solver_message,
        scipy_status=int(sol.status),
        r_s=scales.r_s,
        r_ph=scales.r_ph,
        b_crit=scales.b_crit,
        b_over_b_crit=init_cond.b_over_b_crit,
        diagnostics=diagnostics,
        initial_conditions=init_cond,
    )
