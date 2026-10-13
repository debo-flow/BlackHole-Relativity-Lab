"""Tests for first-integral residuals, convergence under tolerance refinement, and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from blackhole_lab.numerics.geodesics import GeodesicConfig, solve_photon_geodesic
from blackhole_lab.physics.schwarzschild import critical_impact_parameter
from blackhole_lab.validation.diagnostics import compute_first_integral_residuals


def test_first_integral_residual_scales_with_tolerance() -> None:
    """Tighter solver tolerances must yield smaller or comparable first-integral residuals."""
    M = 1.0
    r0 = 15.0
    b = 6.5

    cfg_loose = GeodesicConfig(M=M, r0=r0, b=b, rtol=1e-5, atol=1e-7, max_step=0.2)
    cfg_tight = GeodesicConfig(M=M, r0=r0, b=b, rtol=1e-11, atol=1e-13, max_step=0.05)

    res_loose = solve_photon_geodesic(cfg_loose)
    res_tight = solve_photon_geodesic(cfg_tight)

    diag_loose = res_loose.diagnostics
    diag_tight = res_tight.diagnostics

    assert diag_tight.max_abs_residual < 1e-10
    assert diag_tight.max_abs_residual < diag_loose.max_abs_residual


def test_near_critical_trajectory_flags_diagnostic_warning() -> None:
    """Trajectories within 2% of b_crit must set is_near_critical=True and emit a warning."""
    M = 1.0
    b_crit = critical_impact_parameter(M)

    cfg_near = GeodesicConfig(
        M=M,
        r0=15.0,
        b=1.002 * b_crit,
        rtol=1e-10,
        atol=1e-12,
    )
    res_near = solve_photon_geodesic(cfg_near)
    diag = res_near.diagnostics

    assert diag.is_near_critical is True
    assert any("Near-critical impact parameter" in w for w in diag.warnings)
    assert np.isfinite(diag.max_abs_residual)


def test_residual_handles_invalid_points_cleanly() -> None:
    """Points with non-finite values or u <= 0 must produce NaN residuals without crashing."""
    u = np.array([0.1, -0.05, np.nan])
    v = np.array([0.0, 0.1, 0.1])
    residuals = compute_first_integral_residuals(u, v, M=1.0, b=6.0)

    assert np.isfinite(residuals[0])
    assert np.isnan(residuals[1])
    assert np.isnan(residuals[2])


def test_residual_shape_mismatch_raises_value_error() -> None:
    """Mismatched u and du_dphi array shapes must raise ValueError."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_first_integral_residuals([0.1, 0.2], [0.1], M=1.0, b=5.0)
