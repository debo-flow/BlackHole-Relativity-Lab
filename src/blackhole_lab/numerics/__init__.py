"""Numerical ODE solvers, initial conditions, deflection engine, and shadow ray tracer."""

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
from blackhole_lab.numerics.shadow import (
    CriticalBoundaryResult,
    ObserverRayModel,
    ShadowGridConfig,
    ShadowRayTracingResult,
    find_critical_impact_parameter,
    sky_angle_to_impact_parameter,
    trace_black_hole_shadow,
)

__all__ = [
    "CriticalBoundaryResult",
    "DeflectionResult",
    "GeodesicConfig",
    "GeodesicResult",
    "ObserverRayModel",
    "PhotonFate",
    "PhotonInitialConditions",
    "RadialDirection",
    "ShadowGridConfig",
    "ShadowRayTracingResult",
    "SolverStatus",
    "TerminationReason",
    "asymptotic_tail_angle",
    "calculate_deflection_angle",
    "classify_photon_fate",
    "compute_initial_conditions",
    "find_critical_impact_parameter",
    "first_integral_radicand",
    "initial_du_dphi",
    "orbit_ode_rhs",
    "sky_angle_to_impact_parameter",
    "solve_photon_geodesic",
    "trace_black_hole_shadow",
    "unwrap_azimuthal_angle",
]
