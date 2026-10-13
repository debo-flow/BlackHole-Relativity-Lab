"""Example script demonstrating weak deflection, strong near-critical lensing, and capture.

Run after editable installation (`pip install -e .`):
    python examples/basic_trajectories.py
"""

from __future__ import annotations

from blackhole_lab.numerics.geodesics import GeodesicConfig, solve_photon_geodesic
from blackhole_lab.physics.schwarzschild import (
    critical_impact_parameter,
    photon_sphere_radius,
    schwarzschild_radius,
)


def run_examples() -> None:
    """Run three representative equatorial photon trajectories around a M = 1 black hole."""
    M = 1.0
    r0 = 20.0
    r_s = schwarzschild_radius(M)
    r_ph = photon_sphere_radius(M)
    b_crit = critical_impact_parameter(M)

    print("=== BlackHole-Relativity-Lab: Milestone 1 Example ===")
    print(f"Mass M = {M:.2f} | r_s = {r_s:.4f} | r_ph = {r_ph:.4f} | b_crit = {b_crit:.6f}\n")

    cases = [
        ("Weak-field scattering (b > b_crit)", 8.0 * M),
        ("Near-critical strong deflection (b ≈ 1.005 * b_crit)", 1.005 * b_crit),
        ("Direct capture (b < b_crit)", 4.2 * M),
    ]

    for label, b_val in cases:
        cfg = GeodesicConfig(M=M, r0=r0, b=b_val, direction="ingoing")
        res = solve_photon_geodesic(cfg)
        diag = res.diagnostics

        print(f"Case: {label}")
        print(f"  Impact parameter b      : {b_val:.6f} (b / b_crit = {res.b_over_b_crit:.5f})")
        print(f"  Solver status           : {res.status.value}")
        print(f"  Termination reason      : {res.termination_reason.value}")
        print(f"  Integration points      : {diag.num_points}")
        print(f"  Final (phi, r)          : ({res.phi[-1]:.4f} rad, {res.r[-1]:.4f} M)")
        print(f"  Max |1st-int residual|  : {diag.max_abs_residual:.3e}")
        if diag.warnings:
            for w in diag.warnings:
                print(f"  Warning                 : {w}")
        print("-" * 72)


if __name__ == "__main__":
    run_examples()
