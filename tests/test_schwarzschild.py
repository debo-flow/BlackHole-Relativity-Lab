"""Unit tests for Schwarzschild geometry, characteristic radii, and input validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blackhole_lab.physics.schwarzschild import (
    critical_impact_parameter,
    effective_potential_u,
    metric_coefficient_f,
    photon_sphere_radius,
    schwarzschild_radius,
    validate_impact_parameter,
    validate_mass,
    validate_radius,
)


@pytest.mark.parametrize("mass", [0.5, 1.0, 2.5, 10.0])
def test_characteristic_scales_exact_relations(mass: float) -> None:
    """Verify r_s = 2M, r_ph = 3M, and b_crit = 3 * sqrt(3) * M."""
    r_s = schwarzschild_radius(mass)
    r_ph = photon_sphere_radius(mass)
    b_crit = critical_impact_parameter(mass)

    assert math.isclose(r_s, 2.0 * mass, rel_tol=1e-15)
    assert math.isclose(r_ph, 3.0 * mass, rel_tol=1e-15)
    assert math.isclose(b_crit, 3.0 * math.sqrt(3.0) * mass, rel_tol=1e-15)

    # At the photon sphere u_ph = 1 / (3M), V_eff(u_ph) must equal 1 / b_crit^2 = 1 / (27 M^2)
    u_ph = 1.0 / r_ph
    v_eff_max = effective_potential_u(u_ph, mass)
    assert math.isclose(v_eff_max, 1.0 / (b_crit * b_crit), rel_tol=1e-15)


def test_metric_coefficient_f_values() -> None:
    """Verify f(r) = 1 - 2M/r in the exterior domain r > 2M."""
    M = 1.5
    assert math.isclose(metric_coefficient_f(3.0 * M, M), 1.0 / 3.0, rel_tol=1e-15)
    assert math.isclose(metric_coefficient_f(4.0 * M, M), 0.5, rel_tol=1e-15)
    assert math.isclose(metric_coefficient_f(1e9 * M, M), 1.0 - 2e-9, rel_tol=1e-15)


@pytest.mark.parametrize(
    "invalid_mass",
    [0.0, -1.0, -1e-12, float("nan"), float("inf"), float("-inf"), np.nan, np.inf],
)
def test_invalid_mass_rejected(invalid_mass: float) -> None:
    """Reject non-positive, NaN, and infinite masses."""
    with pytest.raises(ValueError):
        validate_mass(invalid_mass)
    with pytest.raises(ValueError):
        schwarzschild_radius(invalid_mass)
    with pytest.raises(ValueError):
        photon_sphere_radius(invalid_mass)
    with pytest.raises(ValueError):
        critical_impact_parameter(invalid_mass)


@pytest.mark.parametrize("bad_type", [True, False, "1.0", None, [1.0]])
def test_non_numeric_inputs_raise_type_error(bad_type: object) -> None:
    """Booleans and non-numeric types must raise TypeError."""
    with pytest.raises(TypeError):
        validate_mass(bad_type)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_radius(bad_type)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_impact_parameter(bad_type)  # type: ignore[arg-type]


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
