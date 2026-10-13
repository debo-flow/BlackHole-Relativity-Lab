"""Tests for the Milestone 2 Part 3 gravitational light deflection engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blackhole_lab.numerics.geodesics import GeodesicConfig
from blackhole_lab.numerics.lensing import (
    PhotonFate,
    asymptotic_tail_angle,
    calculate_deflection_angle,
    classify_photon_fate,
    unwrap_azimuthal_angle,
)
from blackhole_lab.physics.schwarzschild import (
    critical_impact_parameter,
    second_order_weak_field_deflection,
    weak_field_deflection_angle,
)


def test_weak_field_formula_and_invalid_inputs() -> None:
    """Verify alpha_weak = 4M/b and input validation."""
    assert math.isclose(weak_field_deflection_angle(1.0, 40.0), 0.1, rel_tol=1e-15)
    assert math.isclose(weak_field_deflection_angle(2.5, 50.0), 0.2, rel_tol=1e-15)

    with pytest.raises(ValueError):
        weak_field_deflection_angle(-1.0, 10.0)
    with pytest.raises(ValueError):
        weak_field_deflection_angle(1.0, 0.0)
    with pytest.raises(ValueError):
        weak_field_deflection_angle(float("nan"), 10.0)


@pytest.mark.parametrize("b_over_M", [40.0, 80.0, 160.0])
def test_weak_field_numerical_vs_analytical_agreement(b_over_M: float) -> None:
    """Verify numerical deflection converges to 4M/b with O((M/b)^2) post-Newtonian error."""
    M = 1.0
    b = b_over_M * M
    cfg = GeodesicConfig(
        M=M,
        r0=500.0 * M,
        b=b,
        outer_radius=600.0 * M,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.02,
    )
    res = calculate_deflection_angle(cfg)

    assert res.fate == PhotonFate.ESCAPED
    assert res.is_weak_field_valid is True
    assert res.numerical_deflection is not None
    assert res.absolute_error is not None
    assert res.relative_error is not None
    assert res.convergence_indicator is not None

    # Leading error |alpha_num - 4M/b| is dominated by the 2PN term 15*pi/4 * (M/b)^2
    expected_2pn_term = 0.25 * 15.0 * math.pi * ((M / b) ** 2)
    assert math.isclose(res.absolute_error, expected_2pn_term, rel_tol=0.08)

    # Comparison against 2nd-order post-Newtonian formula should be even tighter: O((M/b)^3)
    err_pn2 = abs(res.numerical_deflection - second_order_weak_field_deflection(M, b))
    assert err_pn2 < 30.0 * ((M / b) ** 3)


def test_finite_radius_invariance_with_asymptotic_tail_correction() -> None:
    """Verify asymptotic-corrected deflection is invariant to finite domain radii r0 and r_end."""
    M = 1.0
    b = 25.0 * M

    cfg_compact = GeodesicConfig(
        M=M,
        r0=60.0 * M,
        b=b,
        outer_radius=80.0 * M,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.02,
    )
    cfg_extended = GeodesicConfig(
        M=M,
        r0=400.0 * M,
        b=b,
        outer_radius=500.0 * M,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.02,
    )

    res_compact = calculate_deflection_angle(cfg_compact, apply_asymptotic_correction=True)
    res_extended = calculate_deflection_angle(cfg_extended, apply_asymptotic_correction=True)

    assert res_compact.numerical_deflection is not None
    assert res_extended.numerical_deflection is not None
    assert res_compact.finite_radius_deflection is not None
    assert res_extended.finite_radius_deflection is not None

    # Raw finite-radius deflections differ substantially because 60M vs 400M omits a large tail
    assert abs(res_compact.finite_radius_deflection - res_extended.finite_radius_deflection) > 0.2

    # Asymptotic-corrected deflections agree to within solver tolerance (~1e-9 rad)
    assert math.isclose(
        res_compact.numerical_deflection,
        res_extended.numerical_deflection,
        rel_tol=1e-8,
        abs_tol=1e-9,
    )


def test_uncorrected_finite_radius_reporting_mode() -> None:
    """Verify apply_asymptotic_correction=False reports raw finite deflection and warns."""
    cfg = GeodesicConfig(M=1.0, r0=50.0, b=15.0, outer_radius=75.0, rtol=1e-10, atol=1e-12)
    res = calculate_deflection_angle(cfg, apply_asymptotic_correction=False)

    assert res.fate == PhotonFate.ESCAPED
    assert res.asymptotic_correction_applied is False
    assert res.numerical_deflection == res.finite_radius_deflection
    assert any("uncorrected finite-radius deflection" in w for w in res.warnings)


def test_angle_unwrapping_and_multi_loop_strong_field_trajectory() -> None:
    """Verify angular wrapping is handled properly even for multi-loop strong-field orbits."""
    M = 1.0
    b_crit = critical_impact_parameter(M)
    # Very close to b_crit so photon makes a full loop (Delta_phi > 3*pi, winding_number >= 1)
    cfg = GeodesicConfig(
        M=M,
        r0=30.0 * M,
        b=1.0015 * b_crit,
        outer_radius=40.0 * M,
        phi_max=8.0 * math.pi,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.02,
    )
    res = calculate_deflection_angle(cfg)

    assert res.fate == PhotonFate.ESCAPED
    assert res.is_weak_field_valid is False
    assert res.winding_number >= 1
    assert res.delta_phi_finite > 3.0 * math.pi
    assert res.numerical_deflection is not None
    assert res.numerical_deflection > 2.0 * math.pi
    assert any("Near-critical strong-field regime" in w for w in res.warnings)

    # Wrap trajectory angles into [-pi, pi) using atan2(sin, cos) and verify unwrapping recovers it
    wrapped_phi = np.arctan2(np.sin(res.geodesic_result.phi), np.cos(res.geodesic_result.phi))
    recovered_phi = unwrap_azimuthal_angle(wrapped_phi)
    np.testing.assert_allclose(recovered_phi, res.geodesic_result.phi, rtol=1e-12, atol=1e-12)


def test_photon_fate_classification_captured_and_unresolved() -> None:
    """Verify captured and unresolved photons never report a bogus numerical deflection angle."""
    M = 1.0
    b_crit = critical_impact_parameter(M)

    # 1. Captured photon (b < b_crit)
    cfg_cap = GeodesicConfig(M=M, r0=20.0, b=0.8 * b_crit)
    res_cap = calculate_deflection_angle(cfg_cap)
    assert res_cap.fate == PhotonFate.CAPTURED
    assert res_cap.numerical_deflection is None
    assert res_cap.absolute_error is None
    assert res_cap.relative_error is None
    assert res_cap.convergence_indicator is None
    assert any("captured" in w.lower() for w in res_cap.warnings)

    # 2. Unresolved photon (stopped early at phi_max before reaching outer boundary)
    cfg_unres = GeodesicConfig(M=M, r0=20.0, b=1.5 * b_crit, phi_max=0.25)
    res_unres = calculate_deflection_angle(cfg_unres)
    assert res_unres.fate == PhotonFate.UNRESOLVED
    assert res_unres.numerical_deflection is None
    assert res_unres.absolute_error is None
    assert res_unres.convergence_indicator is None
    assert any("unresolved" in w.lower() for w in res_unres.warnings)


def test_numerical_convergence_indicator_improves_with_tighter_tolerance() -> None:
    """Verify convergence_indicator decreases systematically as solver tolerances tighten."""
    M = 1.0
    b = 12.0 * M

    cfg_loose = GeodesicConfig(
        M=M,
        r0=40.0,
        b=b,
        outer_radius=60.0,
        rtol=1e-5,
        atol=1e-7,
        max_step=0.2,
    )
    cfg_tight = GeodesicConfig(
        M=M,
        r0=40.0,
        b=b,
        outer_radius=60.0,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.05,
    )

    res_loose = calculate_deflection_angle(cfg_loose, verify_convergence=True)
    res_tight = calculate_deflection_angle(cfg_tight, verify_convergence=True)

    assert res_loose.convergence_indicator is not None
    assert res_tight.convergence_indicator is not None
    assert res_tight.convergence_indicator < 1e-8
    assert res_tight.convergence_indicator < res_loose.convergence_indicator


def test_asymptotic_tail_angle_validation() -> None:
    """Verify asymptotic_tail_angle rejects non-positive u_bound or forbidden barrier points."""
    with pytest.raises(ValueError):
        asymptotic_tail_angle(0.0, M=1.0, b=10.0)
    with pytest.raises(ValueError):
        asymptotic_tail_angle(-0.1, M=1.0, b=10.0)
    # u_bound = 0.25 (r = 4M) is inside the classically forbidden region for b = 10M
    with pytest.raises(ValueError, match="first-integral radicand"):
        asymptotic_tail_angle(0.25, M=1.0, b=10.0)

    # classify_photon_fate rejects non-GeodesicResult
    with pytest.raises(TypeError):
        classify_photon_fate("invalid")  # type: ignore[arg-type]

    # unwrap_azimuthal_angle rejects non-monotonic / reversing angle arrays
    with pytest.raises(ValueError, match="monotonically"):
        unwrap_azimuthal_angle([0.0, 0.5, 0.2])
