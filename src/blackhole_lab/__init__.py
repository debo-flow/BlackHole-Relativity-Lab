"""BlackHole-Relativity-Lab: Schwarzschild Spacetime & Photon Geodesics Engine.

All core equations use geometric units (G = c = 1) and Schwarzschild coordinates
(t, r, theta, phi) restricted to the equatorial plane (theta = pi / 2).
"""

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    GeodesicResult,
    PhotonInitialConditions,
    RadialDirection,
    SolverStatus,
    TerminationReason,
    compute_initial_conditions,
    first_integral_radicand,
    initial_du_dphi,
    orbit_ode_rhs,
    solve_photon_geodesic,
)
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
from blackhole_lab.validation.diagnostics import (
    GeodesicDiagnostics,
    compute_first_integral_residuals,
    evaluate_geodesic_diagnostics,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GeodesicConfig",
    "GeodesicDiagnostics",
    "GeodesicResult",
    "PhotonInitialConditions",
    "RadialDirection",
    "SchwarzschildScales",
    "SolverStatus",
    "TerminationReason",
    "compute_first_integral_residuals",
    "compute_initial_conditions",
    "critical_impact_parameter",
    "effective_potential_u",
    "evaluate_geodesic_diagnostics",
    "first_integral_radicand",
    "initial_du_dphi",
    "metric_coefficient_f",
    "orbit_ode_rhs",
    "photon_sphere_radius",
    "reference_scales",
    "schwarzschild_radius",
    "solve_photon_geodesic",
    "validate_impact_parameter",
    "validate_mass",
    "validate_radius",
]
