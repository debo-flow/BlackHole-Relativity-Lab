"""Streamlit interactive laboratory for Milestone 1: Schwarzschild Photon Geodesics.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import streamlit as st

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    RadialDirection,
    SolverStatus,
    solve_photon_geodesic,
)
from blackhole_lab.physics.schwarzschild import (
    critical_impact_parameter,
    photon_sphere_radius,
    schwarzschild_radius,
)
from blackhole_lab.visualization.plots import (
    plot_cartesian_trajectory,
    plot_radius_vs_phi,
)


def main() -> None:
    """Render the Milestone 1 Schwarzschild Photon Geodesics Streamlit interface."""
    st.set_page_config(
        page_title="BlackHole-Relativity-Lab | Milestone 1",
        layout="wide",
    )

    st.title("BlackHole-Relativity-Lab — Milestone 1")
    st.subheader("Schwarzschild Spacetime & Equatorial Photon Geodesics Engine")
    st.markdown(
        """
        This laboratory numerically integrates the **exact Schwarzschild null-geodesic orbit equation**
        in the equatorial plane ($\\theta = \\pi/2$) using geometric units ($G = c = 1$).
        Plots display actual numerical output from `scipy.integrate.solve_ivp` in Schwarzschild coordinates,
        not schematic drawings or synthetic ray-traced accretion images.
        """
    )

    # Sidebar controls
    st.sidebar.header("1. Physical Parameters (G = c = 1)")
    mass_M = st.sidebar.number_input(
        "Black hole mass parameter M",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.3f",
        help="In geometric units (G = c = 1), mass M sets the characteristic length scale.",
    )

    r_s_default = schwarzschild_radius(mass_M)
    r_ph_default = photon_sphere_radius(mass_M)
    b_crit_default = critical_impact_parameter(mass_M)

    r0 = st.sidebar.number_input(
        "Starting radius r₀",
        min_value=float(r_s_default * 1.05),
        max_value=float(500.0 * mass_M),
        value=float(15.0 * mass_M),
        step=float(0.5 * mass_M),
        format="%.3f",
        help="Must lie strictly outside the capture boundary (r₀ > r_capture >= 2M).",
    )

    b = st.sidebar.number_input(
        "Impact parameter b = L / E",
        min_value=0.05,
        max_value=float(200.0 * mass_M),
        value=float(5.25 * mass_M),
        step=0.05,
        format="%.5f",
        help=f"Critical impact parameter for M={mass_M:.2f} is b_crit ≈ {b_crit_default:.5f}.",
    )

    direction: RadialDirection = st.sidebar.selectbox(
        "Initial radial direction",
        options=["ingoing", "outgoing"],
        index=0,
        help="Ingoing means dr/dφ < 0 (du/dφ > 0); outgoing means dr/dφ > 0 (du/dφ < 0).",
    )

    st.sidebar.header("2. Numerical Solver Controls")
    log10_rtol = st.sidebar.slider(
        "log₁₀(rtol) [Relative Tolerance]",
        min_value=-13,
        max_value=-4,
        value=-9,
        step=1,
        help="Safe range restricted to [1e-13, 1e-4].",
    )
    log10_atol = st.sidebar.slider(
        "log₁₀(atol) [Absolute Tolerance]",
        min_value=-14,
        max_value=-5,
        value=-11,
        step=1,
        help="Safe range restricted to [1e-14, 1e-5].",
    )
    phi_max_turns = st.sidebar.slider(
        "Max angular span (φ_max / 2π turns)",
        min_value=0.5,
        max_value=6.0,
        value=3.0,
        step=0.5,
    )
    method = st.sidebar.selectbox(
        "ODE Solver Method",
        options=["DOP853", "RK45"],
        index=0,
    )

    rtol = 10.0**log10_rtol
    atol = 10.0**log10_atol
    phi_max = phi_max_turns * 2.0 * math.pi

    # Characteristic scales panel
    st.markdown("### Characteristic Schwarzschild Scales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Schwarzschild Radius r_s = 2M", f"{r_s_default:.5f}")
    col2.metric("Photon Sphere r_ph = 3M", f"{r_ph_default:.5f}")
    col3.metric("Critical Impact b_crit = 3√3 M", f"{b_crit_default:.5f}")
    col4.metric("Ratio b / b_crit", f"{b / b_crit_default:.5f}")

    # Run solver with explicit validation handling
    try:
        config = GeodesicConfig(
            M=mass_M,
            r0=r0,
            b=b,
            direction=direction,
            phi0=0.0,
            phi_max=phi_max,
            capture_radius_factor=1.001,
            outer_radius=max(1.5 * r0, 50.0 * mass_M),
            rtol=rtol,
            atol=atol,
            method=method,
            max_step=0.05,
        )
        result = solve_photon_geodesic(config)
    except (ValueError, TypeError) as exc:
        st.error(f"**Invalid Physical or Numerical Input:** {exc}")
        st.stop()

    # Status & Diagnostics Panel
    st.markdown("### Integration Status & First-Integral Diagnostics")
    status_labels = {
        SolverStatus.COMPLETED: (
            "COMPLETED INTEGRATION (Reached Outer Radial Boundary)",
            "success",
        ),
        SolverStatus.CAPTURE_BOUNDARY: (
            "CAPTURE-BOUNDARY TERMINATION (Reached Capture Radius near Event Horizon)",
            "warning",
        ),
        SolverStatus.INTEGRATION_LIMIT: (
            "INTEGRATION-LIMIT TERMINATION (Reached Maximum Angle φ_max)",
            "info",
        ),
        SolverStatus.SOLVER_FAILURE: (
            "SOLVER FAILURE (Numerical Error or Invalid Domain)",
            "error",
        ),
    }
    label_text, alert_kind = status_labels[result.status]

    if alert_kind == "success":
        st.success(f"**Status:** {label_text}\n\n{result.solver_message}")
    elif alert_kind == "warning":
        st.warning(f"**Status:** {label_text}\n\n{result.solver_message}")
    elif alert_kind == "info":
        st.info(f"**Status:** {label_text}\n\n{result.solver_message}")
    else:
        st.error(f"**Status:** {label_text}\n\n{result.solver_message}")

    diag = result.diagnostics
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.metric("Integration Points", f"{diag.num_points}")
    dcol2.metric("Termination Reason", f"{diag.termination_reason.value}")
    dcol3.metric("Max |First-Integral Residual|", f"{diag.max_abs_residual:.3e}")
    dcol4.metric("Tolerances (rtol / atol)", f"{diag.rtol:.0e} / {diag.atol:.0e}")

    for warn_msg in diag.warnings:
        st.warning(f"**Diagnostic Notice:** {warn_msg}")

    # Plots
    st.markdown("### Numerical Trajectory Plots")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig_cart, _ = plot_cartesian_trajectory(result)
        st.pyplot(fig_cart)
        plt.close(fig_cart)

    with pcol2:
        fig_rad, _ = plot_radius_vs_phi(result)
        st.pyplot(fig_rad)
        plt.close(fig_rad)

    # Governing equations & conventions
    with st.expander("Governing Equations, Geometric Units & Limitations", expanded=True):
        st.markdown(
            r"""
            #### 1. Schwarzschild Metric & Geometric Units ($G = c = 1$)
            In geometric units where $G = c = 1$, mass $M$, radius $r$, and impact parameter $b$ all carry dimensions of length.
            The exterior Schwarzschild line element ($r > 2M$) in coordinates $(t, r, \theta, \phi)$ is:
            $$ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \left(1 - \frac{2M}{r}\right)^{-1} dr^2 + r^2\left(d\theta^2 + \sin^2\theta\, d\phi^2\right)$$

            #### 2. Equatorial Null-Geodesic Equations ($\theta = \pi/2, u = 1/r$)
            Setting $u(\phi) = 1/r(\phi)$, the exact second-order photon orbit equation and its associated first integral are:
            $$\frac{d^2u}{d\phi^2} = -u + 3Mu^2, \qquad \left(\frac{du}{d\phi}\right)^2 = \frac{1}{b^2} - u^2 + 2Mu^3$$
            The solver integrates the second-order ODE as a first-order system and uses the exact first integral both to initialize
            $\left.\frac{du}{d\phi}\right|_{\phi_0} = \pm\sqrt{1/b^2 - u_0^2 + 2Mu_0^3}$ and to compute the pointwise conservation residual
            $\mathcal{R}(\phi) = (du/d\phi)^2 - (1/b^2 - u^2 + 2Mu^3)$.

            #### 3. Coordinate & Numerical Limitations
            - **Independent variable $\phi$:** Purely radial trajectories ($b = 0$) have $d\phi = 0$ and cannot be integrated with $\phi$ as the independent variable.
            - **Finite outer boundary:** Stopping at a finite outer radius $r_{\mathrm{outer}}$ indicates that the photon scattered back to the edge of the computational domain; it is not an evaluation at $r = \infty$.
            - **Coordinate projection:** The Cartesian plot shows $(x, y) = (r\cos\phi, r\sin\phi)$ in Schwarzschild coordinates.
            """
        )


if __name__ == "__main__":
    main()
