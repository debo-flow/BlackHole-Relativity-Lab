"""Schwarzschild spacetime geometry and characteristic scales in geometric units (G = c = 1)."""

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

__all__ = [
    "SchwarzschildScales",
    "critical_impact_parameter",
    "effective_potential_u",
    "metric_coefficient_f",
    "photon_sphere_radius",
    "reference_scales",
    "schwarzschild_radius",
    "validate_impact_parameter",
    "validate_mass",
    "validate_radius",
]
