"""Numerical validation and conservation-law diagnostics for photon geodesics."""

from blackhole_lab.validation.diagnostics import (
    GeodesicDiagnostics,
    compute_first_integral_residuals,
    evaluate_geodesic_diagnostics,
)

__all__ = [
    "GeodesicDiagnostics",
    "compute_first_integral_residuals",
    "evaluate_geodesic_diagnostics",
]
