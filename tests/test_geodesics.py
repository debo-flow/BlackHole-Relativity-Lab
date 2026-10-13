"""Tests for the Schwarzschild equatorial photon geodesic ODE solver."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    SolverStatus,
    TerminationReason,
    first_integral_radicand,
    initial_du_dphi,
    orbit_ode_rhs,
    solve_photon_geodesic,
)
from blackhole_lab.physics.schwarzschild import critical_impact_parameter


def test_orbit_ode_rhs_exact_values() -> None:
    """Verify du/dphi = v and dv/dphi = -u + 3Mu^2."""
    M = 1.5
    # At photon sphere r = 3M -> u = 1 / (3M), dv/dphi must vanish identically
    u_ph = 1.0 / (3.0 * M)
    du, dv = orbit_ode_rhs(0.0, [u_ph, 0.25], M)
    assert math.isclose(du, 0.25, rel_tol=1e-15)
    assert math.isclose(dv, 0.0, abs_tol=1e-15)

    # Arbitrary point check
    u_val = 0.1
    v_val = -0.05
    du2, dv2 = orbit_ode_rhs(1.2, [u_val, v_val], M)
    assert math.isclose(du2, v_val, rel_tol=1e-15)
    assert math.isclose(dv2, -u_val + 3.0 * M * (u_val**2), rel_tol=1e-15)


def test_initial_condition_radicand_and_sign_conventions() -> None:
    """Verify sign of initial du/dphi for ingoing vs outgoing photons and forbidden region."""
    M = 1.0
    r0 = 10.0
    b = 6.0
    v_in = initial_du_dphi(r0, M, b, direction="ingoing")
    v_out = initial_du_dphi(r0, M, b, direction="outgoing")

    assert v_in > 0.0
    assert v_out < 0.0
    assert math.isclose(v_in, -v_out, rel_tol=1e-15)

    u0 = 1.0 / r0
    expected_sq = first_integral_radicand(u0, M, b)
    assert math.isclose(v_in * v_in, expected_sq, rel_tol=1e-15)

    # Classically forbidden initial condition: r0 = 4M (between 3M and turning point) with large b
    with pytest.raises(ValueError, match="Impossible initial conditions"):
        initial_du_dphi(r0=4.0, M=1.0, b=10.0, direction="ingoing")


@pytest.mark.parametrize("b_factor", [1.5, 2.0, 3.5])
def test_scattering_trajectories_complete_at_outer_boundary(b_factor: float) -> None:
    """Ingoing trajectories with b > b_crit starting outside r_ph must scatter outward."""
    M = 1.0
    b_crit = critical_impact_parameter(M)
    r0 = 20.0
    outer_r = 30.0
    cfg = GeodesicConfig(
        M=M,
        r0=r0,
        b=b_factor * b_crit,
        direction="ingoing",
        outer_radius=outer_r,
        rtol=1e-10,
        atol=1e-12,
    )
    res = solve_photon_geodesic(cfg)

    assert res.status == SolverStatus.COMPLETED
    assert res.termination_reason == TerminationReason.OUTER_BOUNDARY_REACHED
    assert res.phi.ndim == 1
    assert res.phi.shape == res.u.shape == res.r.shape == res.x.shape == res.y.shape
    assert np.all(np.isfinite(res.phi))
    assert np.all(np.isfinite(res.r))

    # Minimum radius (periapsis) must stay strictly above the photon sphere 3M
    assert float(np.min(res.r)) > 3.0 * M
    # Final radius must match outer_r
    assert math.isclose(float(res.r[-1]), outer_r, rel_tol=1e-6)


@pytest.mark.parametrize("b_factor", [0.3, 0.6, 0.9])
def test_subcritical_trajectories_terminate_at_capture_boundary(b_factor: float) -> None:
    """Ingoing trajectories with b < b_crit must cross 3M and hit the capture boundary."""
    M = 1.0
    b_crit = critical_impact_parameter(M)
    cfg = GeodesicConfig(
        M=M,
        r0=15.0,
        b=b_factor * b_crit,
        direction="ingoing",
        capture_radius_factor=1.005,
        rtol=1e-10,
        atol=1e-12,
    )
    res = solve_photon_geodesic(cfg)

    assert res.status == SolverStatus.CAPTURE_BOUNDARY
    assert res.termination_reason == TerminationReason.CAPTURE_BOUNDARY_REACHED
    assert math.isclose(float(res.r[-1]), cfg.capture_radius, rel_tol=1e-6)
    # Trajectory must remain strictly outside r_s = 2M
    assert np.all(res.r >= 2.0 * M)


def test_integration_limit_termination_when_phi_max_is_small() -> None:
    """Verify that stopping at phi_max is reported as INTEGRATION_LIMIT, not capture or escape."""
    cfg = GeodesicConfig(
        M=1.0,
        r0=20.0,
        b=8.0,
        direction="ingoing",
        phi0=0.0,
        phi_max=0.2,
    )
    res = solve_photon_geodesic(cfg)
    assert res.status == SolverStatus.INTEGRATION_LIMIT
    assert res.termination_reason == TerminationReason.PHI_MAX_REACHED
    assert math.isclose(float(res.phi[-1]), 0.2, rel_tol=1e-12)


def test_solver_failure_is_not_labeled_as_capture() -> None:
    """Verify that a failed solve_ivp integration is explicitly classified as SOLVER_FAILURE."""
    cfg = GeodesicConfig(M=1.0, r0=15.0, b=4.0, direction="ingoing")

    class DummyFailedSolution:
        status = -1
        message = "Mock step size underflow"
        t = np.array([0.0, 0.1])
        y = np.array([[1.0 / 15.0, 1.0 / 14.0], [0.2, 0.21]])
        t_events = [[], [], []]

    with patch("blackhole_lab.numerics.geodesics.solve_ivp", return_value=DummyFailedSolution()):
        res = solve_photon_geodesic(cfg)

    assert res.status == SolverStatus.SOLVER_FAILURE
    assert res.termination_reason == TerminationReason.SOLVER_STEP_FAILURE
    assert res.status != SolverStatus.CAPTURE_BOUNDARY
