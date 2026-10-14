"""Tests for Milestone 3 Part 1: Black hole shadow physics, critical boundary & ray-tracing."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from blackhole_lab.numerics.lensing import PhotonFate
from blackhole_lab.numerics.shadow import (
    ShadowGridConfig,
    find_critical_impact_parameter,
    sky_angle_to_impact_parameter,
    trace_black_hole_shadow,
)
from blackhole_lab.physics.schwarzschild import (
    critical_impact_parameter,
    photon_sphere_radius,
    schwarzschild_radius,
    shadow_angular_radius,
)


def test_schwarzschild_reference_scales_and_synge_shadow_angle() -> None:
    """Verify r_s = 2M, r_ph = 3M, b_crit = 3*sqrt(3)*M, and Synge's shadow angular radius."""
    M = 2.0
    assert math.isclose(schwarzschild_radius(M), 4.0, rel_tol=1e-15)
    assert math.isclose(photon_sphere_radius(M), 6.0, rel_tol=1e-15)
    assert math.isclose(critical_impact_parameter(M), 6.0 * math.sqrt(3.0), rel_tol=1e-15)

    # At r_obs = 3M (photon sphere), sin(psi_crit) = 1 ==> psi_crit = pi / 2
    assert math.isclose(shadow_angular_radius(3.0 * M, M), 0.5 * math.pi, rel_tol=1e-15)

    # At large r_obs, psi_crit -> b_crit / r_obs
    r_far = 1e6 * M
    psi_far = shadow_angular_radius(r_far, M)
    assert math.isclose(psi_far, critical_impact_parameter(M) / r_far, rel_tol=1e-5)

    # Inside the photon sphere r_obs < 3M raises ValueError
    with pytest.raises(ValueError, match="photon sphere"):
        shadow_angular_radius(2.5 * M, M)


def test_find_critical_impact_parameter_matches_analytical_value() -> None:
    """Verify numerical bisection recovers b_crit = 3*sqrt(3)*M within requested tolerance."""
    M = 1.0
    tol = 1e-6
    res = find_critical_impact_parameter(
        M=M,
        r0=30.0,
        b_low=3.5,
        b_high=7.0,
        tolerance=tol,
        max_iter=40,
        phi_max=10.0 * math.pi,
    )

    assert res.converged is True
    assert res.unresolved_encountered is False
    assert res.bracket_width <= tol
    assert res.b_low < res.b_crit_analytical < res.b_high
    assert res.absolute_error <= tol
    assert res.relative_error < 1e-6


def test_find_critical_impact_parameter_handles_unresolved_rays_safely() -> None:
    """Verify bisection stops safely if phi_max is too small to resolve near-critical orbits."""
    res = find_critical_impact_parameter(
        M=1.0,
        r0=30.0,
        b_low=3.0,
        b_high=8.0,
        tolerance=1e-8,
        phi_max=1.6 * math.pi,  # Intentionally small so multi-loop near-critical rays hit phi_max
    )
    assert res.unresolved_encountered is True
    assert res.converged is False
    assert any("UNRESOLVED" in w for w in res.warnings)


def test_shadow_ray_tracing_dimensions_classification_and_symmetry() -> None:
    """Verify 2D shadow mask dimensions, ray classification, and 8-fold Schwarzschild symmetry."""
    cfg = ShadowGridConfig(
        M=1.0,
        r_obs=25.0,
        resolution=21,
        ray_model="local_angle",
    )
    res = trace_black_hole_shadow(cfg)

    assert res.alpha_x.shape == (21,)
    assert res.alpha_y.shape == (21,)
    assert res.theta_sky_grid.shape == (21, 21)
    assert res.impact_parameter_grid.shape == (21, 21)
    assert res.capture_mask.shape == (21, 21)
    assert res.escaped_mask.shape == (21, 21)
    assert res.unresolved_mask.shape == (21, 21)

    # Pixel counts must partition the entire grid with zero overlap
    assert res.num_captured + res.num_escaped + res.num_unresolved == 21 * 21
    assert not np.any(res.capture_mask & res.escaped_mask)
    assert not np.any(res.capture_mask & res.unresolved_mask)
    assert not np.any(res.escaped_mask & res.unresolved_mask)

    # Central pixel (10, 10) is on the optical axis (b = 0) and must be CAPTURED
    assert res.impact_parameter_grid[10, 10] == 0.0
    assert res.capture_mask[10, 10] is np.True_
    # Corner pixel (0, 0) is outside the shadow and must be ESCAPED
    assert res.escaped_mask[0, 0] is np.True_

    # Exact 8-fold symmetry on a centered square grid
    assert np.array_equal(res.capture_mask, np.flipud(res.capture_mask))
    assert np.array_equal(res.capture_mask, np.fliplr(res.capture_mask))
    assert np.array_equal(res.capture_mask, res.capture_mask.T)
    assert np.array_equal(res.escaped_mask, res.escaped_mask.T)

    # Grid-estimated b_crit must bracket the analytical value within 1 grid pixel spacing
    assert res.estimated_b_crit_from_grid is not None
    max_pixel_db = float(np.max(np.diff(res.impact_parameter_grid[10, 10:])))
    assert abs(res.estimated_b_crit_from_grid - res.b_crit_analytical) <= max_pixel_db


def test_rectangular_grid_and_deduplication_equivalence() -> None:
    """Verify non-square (ny, nx) resolution and equivalence with full per-pixel integration."""
    cfg_dedup = ShadowGridConfig(
        M=1.0,
        r_obs=20.0,
        resolution=(9, 11),
        ray_model="impact_parameter",
        deduplicate_symmetric_rays=True,
    )
    cfg_full = ShadowGridConfig(
        M=1.0,
        r_obs=20.0,
        resolution=(9, 11),
        ray_model="impact_parameter",
        deduplicate_symmetric_rays=False,
    )

    res_dedup = trace_black_hole_shadow(cfg_dedup)
    res_full = trace_black_hole_shadow(cfg_full)

    assert res_dedup.capture_mask.shape == (9, 11)
    assert res_dedup.unique_rays_traced < res_full.unique_rays_traced
    assert np.array_equal(res_dedup.capture_mask, res_full.capture_mask)
    assert np.array_equal(res_dedup.escaped_mask, res_full.escaped_mask)
    assert np.array_equal(res_dedup.unresolved_mask, res_full.unresolved_mask)


def test_unresolved_pixels_are_never_converted_to_captured() -> None:
    """Verify that unfinished or failed integrations go to unresolved_mask, never capture_mask."""
    # 1. Small phi_max so rays cannot reach capture or outer boundary
    cfg_short = ShadowGridConfig(
        M=1.0,
        r_obs=30.0,
        resolution=6,  # even resolution so no b=0 on-axis pixel
        phi_max=0.05,
    )
    res_short = trace_black_hole_shadow(cfg_short)

    assert res_short.num_unresolved == 36
    assert res_short.num_captured == 0
    assert res_short.num_escaped == 0
    assert not np.any(res_short.capture_mask)
    assert np.all(res_short.unresolved_mask)
    assert any("UNRESOLVED" in w for w in res_short.warnings)

    # 2. Simulated solver failure on every off-axis geodesic
    class DummyFailedSol:
        status = -1
        message = "Mock solver step failure"
        t = np.array([0.0])
        y = np.array([[1.0 / 30.0], [0.1]])
        t_events = [[], [], []]

    with patch("blackhole_lab.numerics.geodesics.solve_ivp", return_value=DummyFailedSol()):
        res_fail = trace_black_hole_shadow(cfg_short)

    assert res_fail.num_unresolved == 36
    assert res_fail.num_captured == 0
    assert not np.any(res_fail.capture_mask)


def test_shadow_invalid_inputs_rejected() -> None:
    """Verify invalid observer radii, resolutions, fields of view, and brackets raise errors."""
    with pytest.raises(ValueError, match="photon sphere"):
        ShadowGridConfig(M=1.0, r_obs=2.8)

    with pytest.raises(ValueError, match="resolution"):
        ShadowGridConfig(M=1.0, r_obs=20.0, resolution=2)

    with pytest.raises(ValueError, match="fov_rad"):
        ShadowGridConfig(M=1.0, r_obs=20.0, fov_rad=-0.5)

    with pytest.raises(ValueError, match="too large"):
        ShadowGridConfig(M=1.0, r_obs=20.0, fov_rad=3.0, ray_model="local_angle")

    with pytest.raises(ValueError, match="did not result in capture"):
        find_critical_impact_parameter(M=1.0, r0=30.0, b_low=6.0, b_high=8.0)

    with pytest.raises(ValueError, match="did not result in escape"):
        find_critical_impact_parameter(M=1.0, r0=30.0, b_low=3.0, b_high=4.5)

    with pytest.raises(ValueError, match="non-negative"):
        sky_angle_to_impact_parameter(-0.1, r_obs=20.0, M=1.0)
