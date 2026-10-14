"""Tests for the Streamlit dashboard helpers and parameter-keyed caching in app.py."""

from __future__ import annotations

import math

import pytest

from app import (
    build_sweep_b_over_m_grid,
    resolve_impact_parameter,
    run_cached_lensing_sweep,
    run_cached_single_trajectory,
)
from blackhole_lab.numerics.lensing import PhotonFate


def test_resolve_impact_parameter_modes() -> None:
    """Verify dimensional b and dimensionless b / M input modes resolve consistently."""
    M = 2.5
    b_from_dim = resolve_impact_parameter(M, "Dimensional b", b_value=15.0, b_over_M_value=0.0)
    b_from_ratio = resolve_impact_parameter(
        M, "Dimensionless b / M", b_value=0.0, b_over_M_value=6.0
    )
    assert math.isclose(b_from_dim, 15.0, rel_tol=1e-15)
    assert math.isclose(b_from_ratio, 15.0, rel_tol=1e-15)


def test_cached_single_trajectory_invalidates_on_parameter_change() -> None:
    """Verify parameter changes produce distinct results and never return stale cached data."""
    res_weak = run_cached_single_trajectory(
        M=1.0,
        r0=50.0,
        b=20.0,
        direction="ingoing",
        phi_max=6.0 * math.pi,
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
        apply_asymptotic_correction=True,
    )
    res_cap = run_cached_single_trajectory(
        M=1.0,
        r0=50.0,
        b=4.0,
        direction="ingoing",
        phi_max=6.0 * math.pi,
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
        apply_asymptotic_correction=True,
    )
    res_no_tail = run_cached_single_trajectory(
        M=1.0,
        r0=50.0,
        b=20.0,
        direction="ingoing",
        phi_max=6.0 * math.pi,
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
        apply_asymptotic_correction=False,
    )

    assert res_weak.fate == PhotonFate.ESCAPED
    assert res_cap.fate == PhotonFate.CAPTURED
    assert res_weak.numerical_deflection is not None
    assert res_no_tail.numerical_deflection is not None
    assert res_weak.numerical_deflection != res_no_tail.numerical_deflection


def test_build_sweep_grid_and_cached_lensing_sweep() -> None:
    """Verify sweep grid construction, near-critical clustering, and graceful error handling."""
    grid = build_sweep_b_over_m_grid(
        b_min_over_M=4.0,
        b_max_over_M=15.0,
        num_points=6,
        include_near_critical_cluster=True,
    )
    assert len(grid) > 6
    assert tuple(sorted(grid)) == grid

    results, errors = run_cached_lensing_sweep(
        M=1.0,
        r0=40.0,
        b_over_M_values=grid,
        phi_max=6.0 * math.pi,
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
        apply_asymptotic_correction=True,
    )
    assert len(errors) == 0
    assert len(results) == len(grid)
    assert any(r.fate == PhotonFate.CAPTURED for r in results)
    assert any(r.fate == PhotonFate.ESCAPED for r in results)

    # Include an impossible ray (b = 100 > r0 = 20) to verify graceful per-ray error capture
    partial_results, partial_errors = run_cached_lensing_sweep(
        M=1.0,
        r0=20.0,
        b_over_M_values=(6.0, 100.0),
        phi_max=6.0 * math.pi,
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
        apply_asymptotic_correction=True,
    )
    assert len(partial_results) == 1
    assert len(partial_errors) == 1

    with pytest.raises(ValueError, match="greater than"):
        build_sweep_b_over_m_grid(10.0, 5.0, num_points=6, include_near_critical_cluster=False)
