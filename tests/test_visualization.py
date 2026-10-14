"""Focused tests for strong-field lensing trajectory classification and visualization."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from blackhole_lab.numerics.geodesics import GeodesicConfig, solve_photon_geodesic
from blackhole_lab.numerics.lensing import PhotonFate, calculate_deflection_angle
from blackhole_lab.physics.schwarzschild import critical_impact_parameter
from blackhole_lab.visualization.plots import (
    LensingRegime,
    classify_lensing_regime,
    plot_cartesian_trajectory,
    plot_deflection_curve,
    plot_lensing_comparison,
    plot_radius_vs_phi,
    simulate_lensing_bundle,
    split_trajectory_segments,
)

matplotlib.use("Agg")


def test_classify_lensing_regime_boundaries() -> None:
    """Verify regime classification across subcritical, near-critical, strong, moderate, weak."""
    assert classify_lensing_regime(0.85) == LensingRegime.SUBCRITICAL_CAPTURE
    assert classify_lensing_regime(1.0) == LensingRegime.SUBCRITICAL_CAPTURE
    assert classify_lensing_regime(1.02) == LensingRegime.NEAR_CRITICAL
    assert classify_lensing_regime(1.5) == LensingRegime.STRONG_FIELD
    assert classify_lensing_regime(4.0) == LensingRegime.MODERATE_FIELD
    assert classify_lensing_regime(8.0) == LensingRegime.WEAK_FIELD

    with pytest.raises(ValueError):
        classify_lensing_regime(0.0)
    with pytest.raises(ValueError):
        classify_lensing_regime(-1.0)
    with pytest.raises(ValueError):
        classify_lensing_regime(float("nan"))


def test_split_trajectory_segments_never_draws_captured_or_unresolved_as_escaping() -> None:
    """Verify captured and unresolved photons have empty outgoing segments."""
    M = 1.0
    b_crit = critical_impact_parameter(M)

    # 1. Escaped trajectory (b > b_crit): has both incoming and outgoing segments + periapsis
    cfg_esc = GeodesicConfig(M=M, r0=20.0, b=1.6 * b_crit, outer_radius=30.0)
    res_esc = solve_photon_geodesic(cfg_esc)
    seg_esc = split_trajectory_segments(res_esc)

    assert seg_esc.fate == PhotonFate.ESCAPED
    assert seg_esc.incoming_x.size >= 2
    assert seg_esc.outgoing_x.size >= 2
    assert seg_esc.periapsis_point is not None
    assert seg_esc.periapsis_radius is not None
    assert seg_esc.periapsis_radius > 3.0 * M
    # Incoming end and outgoing start meet at periapsis
    assert np.isclose(seg_esc.incoming_x[-1], seg_esc.outgoing_x[0])
    assert np.isclose(seg_esc.incoming_y[-1], seg_esc.outgoing_y[0])

    # 2. Captured trajectory (b < b_crit): outgoing segment must be strictly empty
    cfg_cap = GeodesicConfig(M=M, r0=20.0, b=0.75 * b_crit)
    res_cap = solve_photon_geodesic(cfg_cap)
    seg_cap = split_trajectory_segments(res_cap)

    assert seg_cap.fate == PhotonFate.CAPTURED
    assert seg_cap.regime == LensingRegime.SUBCRITICAL_CAPTURE
    assert seg_cap.incoming_x.size >= 2
    assert seg_cap.outgoing_x.size == 0
    assert seg_cap.outgoing_y.size == 0
    assert seg_cap.periapsis_point is None

    # 3. Unresolved trajectory (stopped at small phi_max): outgoing segment must be empty
    cfg_unres = GeodesicConfig(M=M, r0=20.0, b=1.5 * b_crit, phi_max=0.2)
    res_unres = solve_photon_geodesic(cfg_unres)
    seg_unres = split_trajectory_segments(res_unres)

    assert seg_unres.fate == PhotonFate.UNRESOLVED
    assert seg_unres.incoming_x.size >= 2
    assert seg_unres.outgoing_x.size == 0
    assert seg_unres.outgoing_y.size == 0
    assert seg_unres.periapsis_point is None


def test_simulate_lensing_bundle_and_comparison_plots() -> None:
    """Verify multi-regime bundle simulation and coordinate-space visualization conventions."""
    bundle = simulate_lensing_bundle(
        M=1.0,
        r0=45.0,
        b_factors=(0.75, 1.01, 1.5, 3.5, 6.5),
    )
    assert len(bundle) == 5

    fates = [item.fate for item in bundle]
    assert fates[0] == PhotonFate.CAPTURED
    assert all(f == PhotonFate.ESCAPED for f in fates[1:])

    # Multi-trajectory comparison plot
    fig_comp, ax_comp = plot_lensing_comparison(bundle)
    assert ax_comp.get_aspect() in ("equal", 1.0)
    assert "Coordinate-Space" in ax_comp.get_title()
    assert "Not a Literal Photograph" in ax_comp.get_title()
    plt.close(fig_comp)

    # Deflection vs impact parameter plot
    fig_def, ax_def = plot_deflection_curve(bundle)
    assert "Deflection" in ax_def.get_title()
    plt.close(fig_def)


def test_single_trajectory_plots_and_near_critical_annotation() -> None:
    """Verify plot_cartesian_trajectory and plot_radius_vs_phi with DeflectionResult input."""
    M = 1.0
    b_crit = critical_impact_parameter(M)
    cfg_near = GeodesicConfig(M=M, r0=25.0, b=1.008 * b_crit, outer_radius=35.0)
    def_near = calculate_deflection_angle(cfg_near, verify_convergence=False)

    fig_cart, ax_cart = plot_cartesian_trajectory(def_near)
    assert ax_cart.get_aspect() in ("equal", 1.0)
    assert "Coordinate-Space" in ax_cart.get_title()
    plt.close(fig_cart)

    fig_rad, ax_rad = plot_radius_vs_phi(def_near)
    assert "Radial" in ax_rad.get_title()
    plt.close(fig_rad)


def test_visualization_invalid_input_handling() -> None:
    """Verify visualization functions reject invalid types, empty sequences, and mismatched M."""
    with pytest.raises(TypeError):
        plot_cartesian_trajectory("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        plot_lensing_comparison([])

    with pytest.raises(ValueError, match="non-empty"):
        plot_deflection_curve([])

    with pytest.raises(TypeError):
        plot_deflection_curve(["invalid"])  # type: ignore[list-item]

    res_m1 = solve_photon_geodesic(GeodesicConfig(M=1.0, r0=20.0, b=8.0))
    res_m2 = solve_photon_geodesic(GeodesicConfig(M=2.0, r0=40.0, b=16.0))
    with pytest.raises(ValueError, match="same black hole mass M"):
        plot_lensing_comparison([res_m1, res_m2])

    with pytest.raises(ValueError, match="view_limit"):
        plot_cartesian_trajectory(res_m1, view_limit=-5.0)
