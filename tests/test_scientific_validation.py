"""Scientific validation and numerical reliability suite for BlackHole-Relativity-Lab.

This suite enforces the physical and numerical consistency of the geodesic engine,
ensuring convergence under resolution and tolerance scaling, and protecting against
silent physical regressions.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from blackhole_lab.numerics.geodesics import GeodesicConfig, solve_photon_geodesic
from blackhole_lab.numerics.lensing import PhotonFate, calculate_deflection_angle
from blackhole_lab.numerics.shadow import (
    ShadowGridConfig,
    find_critical_impact_parameter,
    trace_black_hole_shadow,
)
from blackhole_lab.physics.schwarzschild import (
    critical_impact_parameter,
    photon_sphere_radius,
    schwarzschild_radius,
    shadow_angular_radius,
)


def test_reference_scales_and_geometric_unit_consistency() -> None:
    """Validate exact geometric-unit reference scales and asymptotic observer limits."""
    M = 1.0
    r_s = schwarzschild_radius(M)
    r_ph = photon_sphere_radius(M)
    b_crit = critical_impact_parameter(M)

    assert math.isclose(r_s, 2.0, rel_tol=1e-15), "r_s must be exactly 2GM/c^2"
    assert math.isclose(r_ph, 3.0, rel_tol=1e-15), "r_ph must be exactly 3GM/c^2"
    assert math.isclose(b_crit, 3.0 * math.sqrt(3.0), rel_tol=1e-15), "b_crit must be 3*sqrt(3)M"

    # As r_obs -> infinity, the Synge finite-distance formula must converge to b_crit / r_obs
    r_far = 1e6 * M
    psi_far = shadow_angular_radius(r_far, M)
    asymptotic_psi = b_crit / r_far
    # The relative error between the exact relativistic angle and the asymptotic angle
    # at r = 10^6 M should be on the order of M/r, proving consistency between the models.
    assert math.isclose(psi_far, asymptotic_psi, rel_tol=1e-5)


def test_shadow_boundary_convergence_with_resolution() -> None:
    """Verify that estimated shadow boundary converges to b_crit as resolution increases."""
    M = 1.0
    r_obs = 30.0

    # Low resolution grid (11x11)
    cfg_low = ShadowGridConfig(
        M=M, r_obs=r_obs, resolution=11, ray_model="impact_parameter"
    )
    res_low = trace_black_hole_shadow(cfg_low)

    # Higher resolution grid (31x31)
    cfg_high = ShadowGridConfig(
        M=M, r_obs=r_obs, resolution=31, ray_model="impact_parameter"
    )
    res_high = trace_black_hole_shadow(cfg_high)

    assert res_low.estimated_b_crit_from_grid is not None
    assert res_high.estimated_b_crit_from_grid is not None

    err_low = abs(res_low.estimated_b_crit_from_grid - res_low.b_crit_analytical)
    err_high = abs(res_high.estimated_b_crit_from_grid - res_high.b_crit_analytical)

    # We do NOT demand identical masks. We assert that the numerical error of the
    # boundary estimate strictly decreases or remains bounded tightly by pixel size.
    assert err_high < err_low, "Boundary error must decrease with higher resolution."
    
    # The maximum possible error is bounded by the diagonal pixel spacing.
    pixel_width_high = cfg_high.effective_fov_rad * r_obs / 31.0
    assert err_high <= 1.5 * pixel_width_high, "Error exceeds theoretical pixel resolution bound."


def test_bisection_tolerance_refinement() -> None:
    """Verify critical impact parameter bisection converges systematically with tolerance."""
    M = 1.0
    r0 = 25.0

    # Coarse tolerance
    res_coarse = find_critical_impact_parameter(
        M=M, r0=r0, tolerance=1e-2, max_iter=20
    )
    # Tight tolerance
    res_tight = find_critical_impact_parameter(
        M=M, r0=r0, tolerance=1e-5, max_iter=40
    )

    assert res_coarse.converged is True
    assert res_tight.converged is True

    # Bracket width and absolute error must strictly decrease
    assert res_tight.bracket_width < res_coarse.bracket_width
    assert res_tight.absolute_error < res_coarse.absolute_error
    assert res_tight.absolute_error <= 1e-5, "Tight bisection failed requested tolerance."


def test_unresolved_integration_isolation() -> None:
    """Verify numerical integration failures NEVER pollute captured/escaped statistics."""
    # Force a solver failure using a mock
    cfg = ShadowGridConfig(M=1.0, r_obs=20.0, resolution=5)
    
    class DummyFailedSol:
        status = -1
        message = "Mock internal step size underflow"
        t = np.array([0.0])
        y = np.array([[1.0 / 20.0], [0.1]])
        t_events = [[], [], []]

    with patch("blackhole_lab.numerics.geodesics.solve_ivp", return_value=DummyFailedSol()):
        res_fail = trace_black_hole_shadow(cfg)

    # All 25 pixels must be completely isolated as UNRESOLVED.
    assert res_fail.num_unresolved == 25
    assert res_fail.num_captured == 0
    assert res_fail.num_escaped == 0
    assert np.all(res_fail.capture_mask == False)
    assert np.all(res_fail.escaped_mask == False)
    assert np.all(res_fail.unresolved_mask == True)


def test_geodesic_physics_regression() -> None:
    """Regression test against silent breakage of the geodesic conservation laws.
    
    Ensures that for a fixed physical scenario, the first integral residual
    is tightly bounded, proving numerical consistency is preserved.
    """
    M = 1.0
    b = 7.0 * M  # Escaping ray
    cfg = GeodesicConfig(
        M=M, r0=50.0, b=b, outer_radius=100.0, rtol=1e-10, atol=1e-12
    )
    def_res = calculate_deflection_angle(cfg)
    geo_res = def_res.geodesic_result

    assert def_res.fate == PhotonFate.ESCAPED
    
    # 1. Physical Conservation Check: The maximum first-integral residual must be
    # strictly bounded by the solver tolerance parameters, proving energy/momentum
    # are physically conserved throughout the Runge-Kutta steps.
    max_residual = geo_res.diagnostics.max_abs_residual
    assert max_residual < 1e-8, f"First integral conservation broken: residual {max_residual} too high."

    # 2. Deflection Invariance: The deflection angle must be finite and physically sound.
    assert def_res.numerical_deflection is not None
    assert math.isfinite(def_res.numerical_deflection)
    assert def_res.numerical_deflection > 0.0, "Gravity is strictly attractive; deflection must be > 0."
