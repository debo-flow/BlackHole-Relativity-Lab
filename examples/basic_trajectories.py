"""Example script demonstrating Milestone 1 & 2 geodesic integration and light deflection.

Run after editable installation (`pip install -e .`):
    python examples/basic_trajectories.py
"""

from __future__ import annotations

from blackhole_lab.numerics.geodesics import GeodesicConfig
from blackhole_lab.numerics.lensing import PhotonFate, calculate_deflection_angle
from blackhole_lab.physics.schwarzschild import reference_scales


def run_examples() -> None:
    """Run representative photon trajectories across weak, strong, near-critical, and capture."""
    M = 1.0
    r0 = 100.0
    scales = reference_scales(M)

    print("=== BlackHole-Relativity-Lab: Milestones 1 & 2 Example ===")
    print(
        f"Mass M = {scales.M:.2f} | r_s = {scales.r_s:.4f} | "
        f"r_ph = {scales.r_ph:.4f} | b_crit = {scales.b_crit:.6f}\n"
    )

    cases = [
        ("Weak-field scattering (b = 50 M)", 50.0 * M),
        ("Moderate-field lensing (b = 15 M)", 15.0 * M),
        ("Near-critical strong deflection (b = 1.005 * b_crit)", 1.005 * scales.b_crit),
        ("Subcritical direct capture (b = 0.80 * b_crit)", 0.80 * scales.b_crit),
    ]

    for label, b_val in cases:
        cfg = GeodesicConfig(
            M=M,
            r0=r0,
            b=b_val,
            direction="ingoing",
            outer_radius=150.0 * M,
            rtol=1e-10,
            atol=1e-12,
        )
        def_res = calculate_deflection_angle(cfg, apply_asymptotic_correction=True)
        geo = def_res.geodesic_result
        diag = geo.diagnostics

        print(f"Case: {label}")
        print(f"  Impact parameter b      : {b_val:.6f} (b / b_crit = {def_res.b_over_b_crit:.5f})")
        print(f"  Photon fate             : {def_res.fate.value.upper()}")
        print(f"  Solver status / reason  : {geo.status.value} ({geo.termination_reason.value})")
        print(f"  Integration points      : {diag.num_points}")
        print(f"  Max |1st-int residual|  : {diag.max_abs_residual:.3e}")
        print(f"  Weak-field 4M/b         : {def_res.analytical_weak_field:.6f} rad")
        if def_res.fate == PhotonFate.ESCAPED and def_res.numerical_deflection is not None:
            print(f"  Numerical deflection    : {def_res.numerical_deflection:.6f} rad")
            print(f"  Finite-domain sweep - pi: {def_res.finite_radius_deflection:.6f} rad")
            print(f"  Asymptotic tail corr.   : {def_res.asymptotic_tail_correction:.6e} rad")
            print(f"  Absolute error vs 4M/b  : {def_res.absolute_error:.4e} rad")
            print(f"  Convergence indicator   : {def_res.convergence_indicator:.3e} rad")
        for w in def_res.warnings:
            print(f"  Notice                  : {w}")
        print("-" * 76)


if __name__ == "__main__":
    run_examples()
