"""Unit tests for Schwarzschild geometry, reference scales, and input validation."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from blackhole_lab.numerics.geodesics import orbit_ode_rhs
from blackhole_lab.physics.schwarzschild import (
    SchwarzschildScales,
    critical_impact_parameter,
    effective_potential_u,
    metric_coefficient_f,
    photon_sphere_radius,
    reference_scales,
    schwarzschild_radius,
    validate_impact_parameter,
    validate_mass,
    validate_radius,
)


@pytest.mark.parametrize("mass", [1e-6, 0.5, 1.0, 2.5, 10.0, 1e6, np.float64(4.2)])
def test_characteristic_scales_exact_relations(mass: float) -> None:
    """Verify r_s = 2M, r_ph = 3M, and b_crit = 3 * sqrt(3) * M across mass scales."""
    r_s = schwarzschild_radius(mass)
    r_ph = photon_sphere_radius(mass)
    b_crit = critical_impact_parameter(mass)

    assert isinstance(r_s, float)
    assert isinstance(r_ph, float)
    assert isinstance(b_crit, float)

    assert math.isclose(r_s, 2.0 * float(mass), rel_tol=1e-15)
    assert math.isclose(r_ph, 3.0 * float(mass), rel_tol=1e-15)
    assert math.isclose(b_crit, 3.0 * math.sqrt(3.0) * float(mass), rel_tol=1e-15)

    # Strict ordering of characteristic scales: 0 < r_s < r_ph < b_crit
    assert 0.0 < r_s < r_ph < b_crit

    # Universal dimensionless ratios independent of M
    assert math.isclose(r_ph / r_s, 1.5, rel_tol=1e-15)
    assert math.isclose(b_crit / r_ph, math.sqrt(3.0), rel_tol=1e-15)
    assert math.isclose(b_crit / r_s, 1.5 * math.sqrt(3.0), rel_tol=1e-15)

    # At the photon sphere u_ph = 1 / (3M), V_eff(u_ph) must equal 1 / b_crit^2 = 1 / (27 M^2)
    u_ph = 1.0 / r_ph
    v_eff_max = effective_potential_u(u_ph, mass)
    assert math.isclose(v_eff_max, 1.0 / (b_crit * b_crit), rel_tol=1e-15)

    # Binet relation: b_crit = r_ph / sqrt(f(r_ph))
    f_ph = metric_coefficient_f(r_ph, mass)
    assert math.isclose(f_ph, 1.0 / 3.0, rel_tol=1e-15)
    assert math.isclose(b_crit, r_ph / math.sqrt(f_ph), rel_tol=1e-15)


def test_reference_scales_bundle_consistency() -> None:
    """Verify reference_scales(M) matches the individual scale functions and is immutable."""
    M = 3.25
    scales = reference_scales(M)
    assert isinstance(scales, SchwarzschildScales)
    assert math.isclose(scales.M, M, rel_tol=1e-15)
    assert math.isclose(scales.r_s, schwarzschild_radius(M), rel_tol=1e-15)
    assert math.isclose(scales.r_ph, photon_sphere_radius(M), rel_tol=1e-15)
    assert math.isclose(scales.b_crit, critical_impact_parameter(M), rel_tol=1e-15)


def test_photon_sphere_stationarity_in_geodesic_equation() -> None:
    """Verify that u = 1 / r_ph is an exact equilibrium point of d^2u/dphi^2 = -u + 3Mu^2."""
    for mass in (0.5, 1.0, 5.0):
        r_ph = photon_sphere_radius(mass)
        u_ph = 1.0 / r_ph
        du, dv = orbit_ode_rhs(0.0, [u_ph, 0.0], mass)
        assert du == 0.0
        assert math.isclose(dv, 0.0, abs_tol=1e-16)


def test_metric_coefficient_f_values() -> None:
    """Verify f(r) = 1 - 2M/r in the exterior domain r > 2M."""
    M = 1.5
    assert math.isclose(metric_coefficient_f(3.0 * M, M), 1.0 / 3.0, rel_tol=1e-15)
    assert math.isclose(metric_coefficient_f(4.0 * M, M), 0.5, rel_tol=1e-15)
    assert math.isclose(metric_coefficient_f(1e9 * M, M), 1.0 - 2e-9, rel_tol=1e-15)


@pytest.mark.parametrize(
    "invalid_mass",
    [
        0.0,
        -0.0,
        -1.0,
        -1e-12,
        float("nan"),
        float("inf"),
        float("-inf"),
        np.nan,
        np.inf,
        -np.inf,
        sys.float_info.max,  # finite M whose reference scales 2M, 3M, 3*sqrt(3)*M overflow
    ],
)
def test_invalid_mass_rejected(invalid_mass: float) -> None:
    """Reject non-positive, NaN, infinite, and scale-overflowing masses."""
    with pytest.raises(ValueError):
        validate_mass(invalid_mass)
    with pytest.raises(ValueError):
        schwarzschild_radius(invalid_mass)
    with pytest.raises(ValueError):
        photon_sphere_radius(invalid_mass)
    with pytest.raises(ValueError):
        critical_impact_parameter(invalid_mass)
    with pytest.raises(ValueError):
        reference_scales(invalid_mass)


@pytest.mark.parametrize("bad_type", [True, False, "1.0", None, [1.0], (1.0,), {"M": 1.0}])
def test_non_numeric_inputs_raise_type_error(bad_type: object) -> None:
    """Booleans and non-numeric types must raise TypeError."""
    with pytest.raises(TypeError):
        validate_mass(bad_type)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_radius(bad_type)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_impact_parameter(bad_type)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        schwarzschild_radius(bad_type)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        photon_sphere_radius(bad_type)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        critical_impact_parameter(bad_type)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_r",
    [0.0, -5.0, 2.0, 1.999999, 1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_radius_and_horizon_singularity_rejected(invalid_r: float) -> None:
    """Reject r <= 2M in exterior Schwarzschild metric evaluations and non-finite radii."""
    M = 1.0
    with pytest.raises(ValueError):
        metric_coefficient_f(invalid_r, M)


def test_coordinate_singularity_rejected_even_when_interior_allowed() -> None:
    """The coordinate singularity r = 2M must always be rejected."""
    M = 2.0
    with pytest.raises(ValueError, match="coordinate singularity"):
        metric_coefficient_f(4.0, M, require_exterior=False)


@pytest.mark.parametrize(
    "invalid_b",
    [0.0, -0.1, -10.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_impact_parameter_rejected(invalid_b: float) -> None:
    """Reject b <= 0 (including radial b = 0) and non-finite impact parameters."""
    with pytest.raises(ValueError):
        validate_impact_parameter(invalid_b)
