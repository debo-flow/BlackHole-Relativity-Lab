"""Scientific Matplotlib visualization utilities for Schwarzschild photon geodesics."""

from blackhole_lab.visualization.plots import (
    LensingRegime,
    TrajectorySegments,
    classify_lensing_regime,
    plot_black_hole_shadow,
    plot_cartesian_trajectory,
    plot_deflection_curve,
    plot_deflection_error_curve,
    plot_lensing_comparison,
    plot_radius_vs_phi,
    plot_shadow_radial_profile,
    simulate_lensing_bundle,
    split_trajectory_segments,
)

__all__ = [
    "LensingRegime",
    "TrajectorySegments",
    "classify_lensing_regime",
    "plot_black_hole_shadow",
    "plot_cartesian_trajectory",
    "plot_deflection_curve",
    "plot_deflection_error_curve",
    "plot_lensing_comparison",
    "plot_radius_vs_phi",
    "plot_shadow_radial_profile",
    "simulate_lensing_bundle",
    "split_trajectory_segments",
]
