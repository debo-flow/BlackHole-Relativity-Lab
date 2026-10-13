"""First-integral conservation diagnostics and numerical health checks.

First-Integral Residual Definition
----------------------------------
Along an exact equatorial Schwarzschild null geodesic, the first integral
    (du / dphi)^2 = 1 / b^2 - u^2 + 2 M u^3
holds identically at every point. Since our solver integrates the second-order
orbit equation d^2u/dphi^2 = -u + 3Mu^2 directly, the algebraic first-integral
residual:

    residual(phi) = (du / dphi)^2 - (1 / b^2 - u^2 + 2 M u^3)

provides an independent pointwise measure of numerical truncation error accumulated
during the Runge-Kutta integration.

Near-Critical Trajectory Sensitivity
------------------------------------
When b is close to b_crit = 3 * sqrt(3) * M, trajectories approach the unstable
circular photon orbit at r_ph = 3M (u_ph = 1 / (3M)). Perturbations around u_ph
grow exponentially with azimuthal angle (~ exp(+phi)). Consequently, near-critical
orbits amplify numerical errors rapidly, and no fixed universal accuracy bound
should be assumed without inspecting ``max_abs_residual``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blackhole_lab.numerics.geodesics import SolverStatus, TerminationReason
from blackhole_lab.physics.schwarzschild import (
    critical_impact_parameter,
    validate_impact_parameter,
    validate_mass,
)


@dataclass(frozen=True)
class GeodesicDiagnostics:
    """Diagnostic summary of a numerical photon geodesic integration.

    Attributes
    ----------
    rtol : float
        Configured relative tolerance.
    atol : float
        Configured absolute tolerance.
    method : str
        Numerical integration method used.
    num_points : int
        Total number of integration points returned by the solver.
    num_valid_points : int
        Number of finite points with u > 0 where the residual could be evaluated.
    status : SolverStatus
        High-level solver status.
    termination_reason : TerminationReason
        Specific termination reason.
    residuals : NDArray[np.float64]
        Pointwise first-integral residual array (NaN where state is invalid).
    max_abs_residual : float
        Maximum absolute residual across all valid points (NaN if no valid points).
    rms_residual : float
        Root-mean-square residual across all valid points (NaN if no valid points).
    is_near_critical : bool
        True if |b - b_crit| / b_crit < 0.02 (within 2% of the critical threshold).
    warnings : tuple[str, ...]
        Diagnostic warnings regarding near-critical sensitivity, large residuals,
        or solver failures.
    """

    rtol: float
    atol: float
    method: str
    num_points: int
    num_valid_points: int
    status: SolverStatus
    termination_reason: TerminationReason
    residuals: NDArray[np.float64]
    max_abs_residual: float
    rms_residual: float
    is_near_critical: bool
    warnings: tuple[str, ...]


def compute_first_integral_residuals(
    u: NDArray[np.float64] | list[float],
    du_dphi: NDArray[np.float64] | list[float],
    M: float,
    b: float,
) -> NDArray[np.float64]:
    """Compute the pointwise residual of the first-integral equation.

    Evaluates:
        R_i = (du_dphi_i)^2 - (1 / b^2 - u_i^2 + 2 * M * u_i^3)
    wherever u_i > 0 and both u_i and du_dphi_i are finite. Points with non-finite
    values or u_i <= 0 are assigned ``np.nan``.

    Parameters
    ----------
    u : array-like of float
        Inverse radius values u = 1 / r.
    du_dphi : array-like of float
        First derivative values du / dphi.
    M : float
        Black hole mass parameter (M > 0).
    b : float
        Impact parameter (b > 0).

    Returns
    -------
    NDArray[np.float64]
        Array of first-integral residuals matching the shape of ``u``.
    """
    mass = validate_mass(M)
    b_val = validate_impact_parameter(b)

    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(du_dphi, dtype=np.float64)
    if u_arr.shape != v_arr.shape:
        raise ValueError(
            f"Shape mismatch: u has shape {u_arr.shape} but du_dphi has shape {v_arr.shape}."
        )
    if u_arr.size == 0:
        raise ValueError("Cannot compute residuals on empty arrays.")

    residuals = np.full(u_arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(u_arr) & np.isfinite(v_arr) & (u_arr > 0.0)

    if np.any(valid):
        u_v = u_arr[valid]
        v_v = v_arr[valid]
        rhs = (1.0 / (b_val * b_val)) - (u_v * u_v) + (2.0 * mass * (u_v**3))
        residuals[valid] = (v_v * v_v) - rhs

    return residuals


def evaluate_geodesic_diagnostics(
    phi: NDArray[np.float64],
    u: NDArray[np.float64],
    du_dphi: NDArray[np.float64],
    M: float,
    b: float,
    rtol: float,
    atol: float,
    method: str,
    status: SolverStatus,
    termination_reason: TerminationReason,
) -> GeodesicDiagnostics:
    """Evaluate comprehensive numerical diagnostics for a solved photon trajectory.

    Parameters
    ----------
    phi : NDArray[np.float64]
        Angular coordinate samples.
    u : NDArray[np.float64]
        Inverse radial coordinate samples.
    du_dphi : NDArray[np.float64]
        First derivative samples.
    M : float
        Black hole mass parameter (M > 0).
    b : float
        Impact parameter (b > 0).
    rtol : float
        Relative tolerance used.
    atol : float
        Absolute tolerance used.
    method : str
        Solver method name.
    status : SolverStatus
        Integration status.
    termination_reason : TerminationReason
        Integration termination reason.

    Returns
    -------
    GeodesicDiagnostics
        Diagnostic summary object.
    """
    _ = phi
    mass = validate_mass(M)
    b_val = validate_impact_parameter(b)
    b_crit = critical_impact_parameter(mass)

    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(du_dphi, dtype=np.float64)
    num_points = int(u_arr.size)

    if num_points > 0:
        residuals = compute_first_integral_residuals(u_arr, v_arr, mass, b_val)
        finite_res = residuals[np.isfinite(residuals)]
        num_valid = int(finite_res.size)
        if num_valid > 0:
            max_abs = float(np.max(np.abs(finite_res)))
            rms = float(np.sqrt(np.mean(finite_res * finite_res)))
        else:
            max_abs = float("nan")
            rms = float("nan")
    else:
        residuals = np.array([], dtype=np.float64)
        num_valid = 0
        max_abs = float("nan")
        rms = float("nan")

    rel_crit_diff = abs(b_val - b_crit) / b_crit
    is_near_critical = rel_crit_diff < 0.02

    warnings_list: list[str] = []
    if is_near_critical:
        warnings_list.append(
            f"Near-critical impact parameter (|b - b_crit| / b_crit = {rel_crit_diff:.3e} < 2%). "
            "Trajectories near the unstable photon sphere (r_ph = 3M) exhibit exponential "
            "sensitivity to numerical truncation and initial conditions."
        )
    if status == SolverStatus.SOLVER_FAILURE:
        warnings_list.append(
            "Solver reported a numerical failure or invalid domain excursion; "
            "do not interpret this trajectory as physically valid."
        )
    if status == SolverStatus.INTEGRATION_LIMIT:
        warnings_list.append(
            "Integration terminated at phi_max before reaching the capture boundary or "
            "outer truncation radius. The asymptotic fate is not determined by this run alone."
        )
    if math.isfinite(max_abs) and max_abs > max(1000.0 * rtol, 1e-4):
        warnings_list.append(
            f"First-integral maximum residual ({max_abs:.3e}) is large relative to "
            f"configured rtol ({rtol:.1e}). Consider tightening rtol/atol or reducing max_step."
        )

    return GeodesicDiagnostics(
        rtol=float(rtol),
        atol=float(atol),
        method=method,
        num_points=num_points,
        num_valid_points=num_valid,
        status=status,
        termination_reason=termination_reason,
        residuals=residuals,
        max_abs_residual=max_abs,
        rms_residual=rms,
        is_near_critical=is_near_critical,
        warnings=tuple(warnings_list),
    )
