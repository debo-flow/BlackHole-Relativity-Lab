"""Equatorial Schwarzschild photon-geodesic ODE solver using u = 1/r and angle phi.

Mathematical Formulation
------------------------
In the equatorial plane (theta = pi / 2) of Schwarzschild spacetime with G = c = 1,
null geodesics parameterized by azimuthal angle phi and inverse radius u = 1 / r
satisfy the exact second-order orbit equation:

    d^2u / dphi^2 = -u + 3 M u^2

and the associated first integral:

    (du / dphi)^2 = 1 / b^2 - u^2 + 2 M u^3

where b = L / E > 0 is the impact parameter.

We convert the second-order ODE into a first-order system for state y = [u, v]^T:
    du / dphi = v
    dv / dphi = -u + 3 M u^2

Initial Condition & Sign Convention
-----------------------------------
At starting angle phi_0 and radius r_0 > 2M (u_0 = 1 / r_0), the first-integral
radicand is:
    R(u_0; M, b) = 1 / b^2 - u_0^2 + 2 M u_0^3

1. If R(u_0; M, b) < 0, the combination (r_0, b) lies in the classically forbidden
   region where V_eff(r_0) > 1 / b^2. An explicit ValueError is raised.
2. Since u = 1 / r implies du/dphi = -(1 / r^2) (dr / dphi):
   - Ingoing photons (dr / dphi < 0 as phi increases) have v_0 = +sqrt(R(u_0; M, b)).
   - Outgoing photons (dr / dphi > 0 as phi increases) have v_0 = -sqrt(R(u_0; M, b)).

Limitations of phi as the Independent Variable
----------------------------------------------
1. Purely radial trajectories (b = 0) have dphi = 0 and cannot be represented with
   phi as the independent variable.
2. Nearly radial trajectories (0 < b << M) have very large initial derivatives
   du / dphi ~ 1 / b, requiring small step sizes as phi changes by very little over
   large radial intervals.
3. Stopping at a finite outer radius r_max (or finite angle phi_max) does NOT by
   itself constitute an analytic proof of escape to spatial infinity (r -> infinity);
   however, in vacuum Schwarzschild spacetime, an outgoing photon (du / dphi < 0) at
   r > 3M faces a monotonically decreasing effective potential V_eff(r) -> 0 and
   therefore has no outer radial turning point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from blackhole_lab.physics.schwarzschild import (
    _ensure_finite_scalar,
    critical_impact_parameter,
    photon_sphere_radius,
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
class GeodesicConfig:
    """Configuration parameters for the equatorial photon geodesic solver.

    Parameters
    ----------
    M : float
        Black hole mass parameter (M > 0) in geometric units (G = c = 1).
    r0 : float
        Initial radial coordinate (must satisfy r0 > 2M and r0 >= capture_radius).
    b : float
        Photon impact parameter (b > 0) in geometric units.
    direction : RadialDirection, default "ingoing"
        Initial radial motion: "ingoing" (dr/dphi < 0, du/dphi > 0) or
        "outgoing" (dr/dphi > 0, du/dphi < 0).
    phi0 : float, default 0.0
        Initial azimuthal angle in radians.
    phi_max : float, default 6 * pi
        Maximum azimuthal angle to integrate up to (must satisfy phi_max > phi0).
    capture_radius_factor : float, default 1.001
        Capture boundary expressed as a multiple of r_s = 2M (must be >= 1.0).
        Integration stops if r <= capture_radius_factor * r_s. Setting >= 1.0
        prevents the trajectory from silently continuing inside the Schwarzschild
        coordinate horizon r = 2M.
    outer_radius : float | None, default None
        Outer radial truncation boundary r_max for outgoing trajectories.
        If None, defaults to max(1.5 * r0, 50.0 * M).
    rtol : float, default 1e-9
        Relative tolerance passed to ``scipy.integrate.solve_ivp``.
    atol : float, default 1e-11
        Absolute tolerance passed to ``scipy.integrate.solve_ivp``.
    method : str, default "DOP853"
        Explicit Runge-Kutta integration method for ``solve_ivp`` ("DOP853" or "RK45").
    max_step : float, default 0.05
        Maximum angular step size dphi in radians.
    dense_points : int | None, default None
        Optional number of minimum output sample points if interpolation is desired;
        by default (None), uses actual solver steps plus refined event endpoints.
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
        b_val = validate_impact_parameter(self.b)
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
        """Configured capture boundary radius r_capture = capture_radius_factor * 2M."""
        return self.capture_radius_factor * schwarzschild_radius(self.M)

    @property
    def effective_outer_radius(self) -> float:
        """Configured or default outer truncation boundary radius r_max."""
        if self.outer_radius is not None:
            return float(self.outer_radius)
        return max(1.5 * self.r0, 50.0 * self.M)


@dataclass(frozen=True)
class GeodesicResult:
    """Structured output of a Schwarzschild equatorial photon-geodesic integration.

    Attributes
    ----------
    config : GeodesicConfig
        Configuration used for the simulation.
    phi : NDArray[np.float64]
        Array of azimuthal angles (radians) along the trajectory.
    u : NDArray[np.float64]
        Array of inverse radii u = 1 / r along the trajectory.
    du_dphi : NDArray[np.float64]
        Array of first derivatives du / dphi along the trajectory.
    r : NDArray[np.float64]
        Array of radial coordinates r = 1 / u along the trajectory.
    x : NDArray[np.float64]
        Cartesian coordinate x = r * cos(phi) in the equatorial plane.
    y : NDArray[np.float64]
        Cartesian coordinate y = r * sin(phi) in the equatorial plane.
    status : SolverStatus
        High-level classification of the integration outcome.
    termination_reason : TerminationReason
        Specific termination reason.
    solver_message : str
        Human-readable message describing the solver outcome.
    scipy_status : int
        Raw status code returned by ``scipy.integrate.solve_ivp``.
    r_s : float
        Schwarzschild radius 2M.
    r_ph : float
        Photon-sphere radius 3M.
    b_crit : float
        Critical impact parameter 3 * sqrt(3) * M.
    b_over_b_crit : float
        Dimensionless impact parameter ratio b / b_crit.
    diagnostics : object
        Instance of ``GeodesicDiagnostics`` with first-integral residuals and metadata.
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


def first_integral_radicand(u: float, M: float, b: float) -> float:
    """Compute the first-integral radicand R(u; M, b) = 1/b^2 - u^2 + 2Mu^3.

    In the Schwarzschild equatorial plane, (du/dphi)^2 = R(u; M, b).
    A valid real trajectory requires R(u; M, b) >= 0.

    Parameters
    ----------
    u : float
        Inverse radius u = 1 / r (must be finite and strictly positive).
    M : float
        Black hole mass parameter (M > 0).
    b : float
        Impact parameter (b > 0).

    Returns
    -------
    float
        Value of 1/b^2 - u^2 + 2*M*u^3.
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
    """Compute the initial derivative (du/dphi)|_{phi_0} from r0, M, b, and direction.

    Validates that the initial state does not lie in a classically forbidden region
    where the first-integral radicand 1/b^2 - u_0^2 + 2Mu_0^3 is negative.

    Parameters
    ----------
    r0 : float
        Starting radial coordinate (r0 > 2M).
    M : float
        Black hole mass parameter (M > 0).
    b : float
        Impact parameter (b > 0).
    direction : RadialDirection, default "ingoing"
        "ingoing" returns +sqrt(radicand) (since u = 1/r increases as r decreases);
        "outgoing" returns -sqrt(radicand).

    Returns
    -------
    float
        Initial value of du/dphi.

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


def orbit_ode_rhs(
    phi: float,
    state: NDArray[np.float64] | list[float] | tuple[float, float],
    M: float,
) -> tuple[float, float]:
    """Right-hand side of the first-order Schwarzschild equatorial photon orbit ODE.

    Converts d^2u/dphi^2 = -u + 3Mu^2 into:
        du/dphi = v
        dv/dphi = -u + 3 * M * u^2

    Parameters
    ----------
    phi : float
        Independent angular variable (radians). Unused explicitly since the system
        is autonomous in phi, but required by ``scipy.integrate.solve_ivp`` signature.
    state : sequence of 2 floats
        State vector [u, v] where u = 1/r and v = du/dphi.
    M : float
        Black hole mass parameter (M > 0).

    Returns
    -------
    tuple[float, float]
        Derivatives (du/dphi, dv/dphi).
    """
    _ = phi
    u, v = float(state[0]), float(state[1])
    du = v
    dv = -u + 3.0 * M * (u * u)
    return (du, dv)


def solve_photon_geodesic(config: GeodesicConfig) -> GeodesicResult:
    """Integrate an equatorial photon geodesic in Schwarzschild spacetime.

    Uses ``scipy.integrate.solve_ivp`` with terminal event detection for:
    1. Capture boundary: r <= r_capture (u >= 1 / r_capture) while ingoing.
    2. Outer boundary: r >= r_outer (u <= 1 / r_outer) while outgoing.
    3. Invalid domain guard: u <= 0 (r -> infinity / sign flip).

    Critically, solver failure (status == -1 or non-finite output) is never labeled
    as physical capture.

    Parameters
    ----------
    config : GeodesicConfig
        Validated solver and physical configuration.

    Returns
    -------
    GeodesicResult
        Complete trajectory, status classification, characteristic scales, and
        first-integral numerical diagnostics.
    """
    from blackhole_lab.validation.diagnostics import evaluate_geodesic_diagnostics

    u0 = 1.0 / config.r0
    v0 = initial_du_dphi(config.r0, config.M, config.b, config.direction)
    y0 = np.array([u0, v0], dtype=np.float64)

    u_capture = 1.0 / config.capture_radius
    u_outer = 1.0 / config.effective_outer_radius

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
        fun=lambda phi, y: orbit_ode_rhs(phi, y, config.M),
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

    # Determine status and termination reason without conflating solver failure with capture
    if sol.status == -1 or has_nonfinite or phi_arr.size == 0:
        status = SolverStatus.SOLVER_FAILURE
        termination_reason = TerminationReason.SOLVER_STEP_FAILURE
        solver_message = (
            f"Numerical solver failed (solve_ivp status={sol.status}): {sol.message}. "
            "Trajectory cannot be considered physically valid."
        )
    elif (
        sol.status == 1
        and sol.t_events is not None
        and len(sol.t_events) == 3
        and len(sol.t_events[2]) > 0
    ) or np.any(u_arr <= 0.0):
        status = SolverStatus.SOLVER_FAILURE
        termination_reason = TerminationReason.INVALID_DOMAIN_ENTERED
        solver_message = (
            "Integration entered an invalid radial domain (u <= 0). "
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
            f"r_capture = {config.capture_radius:.6g} M (capture_radius_factor = "
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
            "radial boundary."
        )

    # Compute r, x, y safely only where u > 0 and finite
    valid_mask = np.isfinite(u_arr) & (u_arr > 0.0)
    r_arr = np.full_like(u_arr, np.nan, dtype=np.float64)
    r_arr[valid_mask] = 1.0 / u_arr[valid_mask]
    x_arr = r_arr * np.cos(phi_arr)
    y_arr = r_arr * np.sin(phi_arr)

    r_s = schwarzschild_radius(config.M)
    r_ph = photon_sphere_radius(config.M)
    b_crit = critical_impact_parameter(config.M)
    b_ratio = config.b / b_crit

    diagnostics = evaluate_geodesic_diagnostics(
        phi=phi_arr,
        u=u_arr,
        du_dphi=v_arr,
        M=config.M,
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
        r_s=r_s,
        r_ph=r_ph,
        b_crit=b_crit,
        b_over_b_crit=b_ratio,
        diagnostics=diagnostics,
    )
