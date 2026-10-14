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
from blackhole_lab.numerics.lensing import (
    DeflectionResult,
    PhotonFate,
    asymptotic_tail_angle,
    calculate_deflection_angle,
    classify_photon_fate,
    unwrap_azimuthal_angle,
)
from blackhole_lab.physics.schwarzschild import (
    SchwarzschildScales,
    critical_impact_parameter,
    effective_potential_u,
    metric_coefficient_f,
    photon_sphere_radius,
    reference_scales,
    schwarzschild_radius,
    second_order_weak_field_deflection,
    validate_impact_parameter,
    validate_mass,
    validate_radius,
    weak_field_deflection_angle,
)
from blackhole_lab.validation.diagnostics import (
    GeodesicDiagnostics,
    compute_first_integral_residuals,
    evaluate_geodesic_diagnostics,
)
from blackhole_lab.visualization.plots import (
    LensingRegime,
    TrajectorySegments,
    classify_lensing_regime,
    plot_cartesian_trajectory,
    plot_deflection_curve,
    plot_lensing_comparison,
    plot_radius_vs_phi,
    simulate_lensing_bundle,
    split_trajectory_segments,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "DeflectionResult",
    "GeodesicConfig",
    "GeodesicDiagnostics",
    "GeodesicResult",
    "LensingRegime",
    "PhotonFate",
    "PhotonInitialConditions",
    "RadialDirection",
    "SchwarzschildScales",
    "SolverStatus",
    "TerminationReason",
    "TrajectorySegments",
    "asymptotic_tail_angle",
    "calculate_deflection_angle",
    "classify_lensing_regime",
    "classify_photon_fate",
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
    "plot_cartesian_trajectory",
    "plot_deflection_curve",
    "plot_lensing_comparison",
    "plot_radius_vs_phi",
    "reference_scales",
    "schwarzschild_radius",
    "second_order_weak_field_deflection",
    "simulate_lensing_bundle",
    "solve_photon_geodesic",
    "split_trajectory_segments",
    "unwrap_azimuthal_angle",
    "validate_impact_parameter",
    "validate_mass",
    "validate_radius",
    "weak_field_deflection_angle",
]
