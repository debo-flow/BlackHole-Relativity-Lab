"""Tests for Schwarzschild photon initial conditions and numerical ODE integration."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    PhotonInitialConditions,
    SolverStatus,
    TerminationReason,
    compute_initial_conditions,
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


def test_orbit_ode_rhs_invalid_inputs_rejected() -> None:
    """Verify orbit_ode_rhs rejects invalid phi, state length, non-finite state, and invalid M."""
    with pytest.raises(ValueError):
        orbit_ode_rhs(float("nan"), [0.1, 0.0], 1.0)
    with pytest.raises(ValueError):
        orbit_ode_rhs(0.0, [0.1], 1.0)
    with pytest.raises(ValueError):
        orbit_ode_rhs(0.0, [0.1, 0.2, 0.3], 1.0)
    with pytest.raises(ValueError):
        orbit_ode_rhs(0.0, [float("nan"), 0.0], 1.0)
    with pytest.raises(ValueError):
        orbit_ode_rhs(0.0, [0.1, float("inf")], 1.0)
    with pytest.raises(ValueError):
        orbit_ode_rhs(0.0, [0.1, 0.0], -1.0)
    with pytest.raises(TypeError):
        orbit_ode_rhs(0.0, "invalid", 1.0)  # type: ignore[arg-type]


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


def test_compute_initial_conditions_dimensional_and_dimensionless_quantities() -> None:
    """Verify PhotonInitialConditions separates dimensional and dimensionless quantities."""
    M = 2.0
    r0 = 20.0
    b = 12.0
    phi0 = 0.25

    ic = compute_initial_conditions(M=M, r0=r0, b=b, phi0=phi0, direction="ingoing")
    assert isinstance(ic, PhotonInitialConditions)

    # Dimensional quantities [L] and [L^-1]
    assert math.isclose(ic.M, M, rel_tol=1e-15)
    assert math.isclose(ic.r0, r0, rel_tol=1e-15)
    assert math.isclose(ic.b, b, rel_tol=1e-15)
    assert math.isclose(ic.u0, 1.0 / r0, rel_tol=1e-15)
    assert ic.du_dphi0 > 0.0
    assert ic.state_vector.shape == (2,)

    # Dimensionless quantities [1]
    assert math.isclose(ic.r0_over_M, 10.0, rel_tol=1e-15)
    assert math.isclose(ic.r0_over_rs, 5.0, rel_tol=1e-15)
    assert math.isclose(ic.b_over_M, 6.0, rel_tol=1e-15)
    assert math.isclose(ic.b_over_b_crit, b / critical_impact_parameter(M), rel_tol=1e-15)
    assert math.isclose(ic.u0_M, 0.1, rel_tol=1e-15)

    # Local orthonormal pitch angle identity: (du/dphi)_0^2 == (1 / b^2) * (1 - sin^2(psi_0))
    cos_sq_psi0 = 1.0 - ic.sin_local_pitch_angle**2
    assert math.isclose(ic.du_dphi0**2, cos_sq_psi0 / (b * b), rel_tol=1e-14)


def test_geodesic_config_validation_and_dimensionless_properties() -> None:
    """Verify GeodesicConfig validates inputs and exposes dimensionless ratios."""
    cfg = GeodesicConfig(M=2.0, r0=30.0, b=14.0, phi0=0.5, phi_max=4.5)
    assert math.isclose(cfg.delta_phi_max, 4.0, rel_tol=1e-15)
    assert math.isclose(cfg.r0_over_M, 15.0, rel_tol=1e-15)
    assert math.isclose(cfg.b_over_M, 7.0, rel_tol=1e-15)
    assert math.isclose(cfg.b_over_b_crit, 14.0 / critical_impact_parameter(2.0), rel_tol=1e-15)

    with pytest.raises(ValueError, match="phi_max"):
        GeodesicConfig(M=1.0, r0=10.0, b=5.0, phi0=1.0, phi_max=1.0)
    with pytest.raises(ValueError, match="capture_radius_factor"):
        GeodesicConfig(M=1.0, r0=10.0, b=5.0, capture_radius_factor=0.95)
    with pytest.raises(ValueError, match="Initial radius r0"):
        GeodesicConfig(M=1.0, r0=2.001, b=5.0, capture_radius_factor=1.01)
    with pytest.raises(ValueError, match="outer_radius"):
        GeodesicConfig(M=1.0, r0=10.0, b=5.0, outer_radius=9.0)
    with pytest.raises(ValueError, match="rtol"):
        GeodesicConfig(M=1.0, r0=10.0, b=5.0, rtol=1e-1)


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
    assert res.reached_outer_boundary is True
    assert res.is_captured is False
    assert res.is_unfinished is False
    assert res.initial_conditions is not None
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
    assert res.is_captured is True
    assert res.reached_outer_boundary is False
    assert res.is_unfinished is False
    assert math.isclose(float(res.r[-1]), cfg.capture_radius, rel_tol=1e-6)
    # Trajectory must remain strictly outside r_s = 2M
    assert np.all(res.r >= 2.0 * M)


def test_unfinished_trajectories_never_classified_as_captured_or_escaped() -> None:
    """Stopping at phi_max must never be classified as captured or escaped, even for b < b_crit."""
    M = 1.0
    b_crit = critical_impact_parameter(M)

    # Case 1: Supercritical b > b_crit but stopped early at phi_max = 0.2
    cfg_super = GeodesicConfig(
        M=M,
        r0=20.0,
        b=1.5 * b_crit,
        direction="ingoing",
        phi0=0.0,
        phi_max=0.2,
    )
    res_super = solve_photon_geodesic(cfg_super)
    assert res_super.status == SolverStatus.INTEGRATION_LIMIT
    assert res_super.termination_reason == TerminationReason.PHI_MAX_REACHED
    assert res_super.is_unfinished is True
    assert res_super.is_captured is False
    assert res_super.reached_outer_boundary is False
    assert math.isclose(float(res_super.phi[-1]), 0.2, rel_tol=1e-12)

    # Case 2: Subcritical b < b_crit (would eventually be captured) stopped early at phi_max = 0.1
    cfg_sub = GeodesicConfig(
        M=M,
        r0=20.0,
        b=0.7 * b_crit,
        direction="ingoing",
        phi0=0.0,
        phi_max=0.1,
    )
    res_sub = solve_photon_geodesic(cfg_sub)
    assert res_sub.status == SolverStatus.INTEGRATION_LIMIT
    assert res_sub.termination_reason == TerminationReason.PHI_MAX_REACHED
    assert res_sub.is_unfinished is True
    assert res_sub.is_captured is False
    assert res_sub.reached_outer_boundary is False


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
    assert res.is_captured is False
    assert res.reached_outer_boundary is False
    assert res.is_unfinished is True


def test_dimensionless_scale_invariance_across_masses() -> None:
    """Scaling (M, r0, b, outer_radius) by alpha must leave dimensionless profiles invariant."""
    cfg_m1 = GeodesicConfig(
        M=1.0,
        r0=18.0,
        b=6.5,
        outer_radius=25.0,
        rtol=1e-11,
        atol=1e-13,
    )
    scale = 3.5
    cfg_m_scaled = GeodesicConfig(
        M=scale * 1.0,
        r0=scale * 18.0,
        b=scale * 6.5,
        outer_radius=scale * 25.0,
        rtol=1e-11,
        atol=1e-13,
    )

    res1 = solve_photon_geodesic(cfg_m1)
    res2 = solve_photon_geodesic(cfg_m_scaled)

    assert res1.status == res2.status == SolverStatus.COMPLETED
    assert math.isclose(float(res1.delta_phi[-1]), float(res2.delta_phi[-1]), rel_tol=1e-9)
    assert math.isclose(
        float(np.min(res1.r_over_M)),
        float(np.min(res2.r_over_M)),
        rel_tol=1e-9,
    )
