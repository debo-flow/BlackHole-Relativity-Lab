"""Numerical ODE solvers, initial conditions, and gravitational light deflection engine."""

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

__all__ = [
    "DeflectionResult",
    "GeodesicConfig",
    "GeodesicResult",
    "PhotonFate",
    "PhotonInitialConditions",
    "RadialDirection",
    "SolverStatus",
    "TerminationReason",
    "asymptotic_tail_angle",
    "calculate_deflection_angle",
    "classify_photon_fate",
    "compute_initial_conditions",
    "first_integral_radicand",
    "initial_du_dphi",
    "orbit_ode_rhs",
    "solve_photon_geodesic",
    "unwrap_azimuthal_angle",
]
