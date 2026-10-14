"""Streamlit interactive laboratory for BlackHole-Relativity-Lab (Milestones 1, 2 & 3).

Preserves all Milestone 1 equatorial photon geodesic features and Milestone 2
gravitational lensing features while implementing the Milestone 3 Interactive
Black Hole Shadow & Photon Sphere Imaging Dashboard.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    RadialDirection,
    SolverStatus,
)
from blackhole_lab.numerics.lensing import (
    DeflectionResult,
    PhotonFate,
    calculate_deflection_angle,
)
from blackhole_lab.numerics.shadow import (
    ObserverRayModel,
    ShadowGridConfig,
    ShadowRayTracingResult,
    trace_black_hole_shadow,
)
from blackhole_lab.physics.schwarzschild import (
    reference_scales,
    shadow_angular_radius,
)
from blackhole_lab.visualization.plots import (
    classify_lensing_regime,
    plot_black_hole_shadow,
    plot_cartesian_trajectory,
    plot_deflection_curve,
    plot_deflection_error_curve,
    plot_lensing_comparison,
    plot_radius_vs_phi,
    plot_shadow_radial_profile,
)


def resolve_impact_parameter(
    mass_M: float,
    input_mode: str,
    b_value: float,
    b_over_M_value: float,
) -> float:
    """Resolve dimensional impact parameter b [L] from either b or dimensionless b / M."""
    if input_mode == "Dimensionless b / M":
        return float(b_over_M_value * mass_M)
    return float(b_value)


@st.cache_data(show_spinner=False, max_entries=128)
def run_cached_single_trajectory(
    M: float,
    r0: float,
    b: float,
    direction: RadialDirection,
    phi_max: float,
    rtol: float,
    atol: float,
    method: str,
    apply_asymptotic_correction: bool,
) -> DeflectionResult:
    """Compute a single photon geodesic and its deflection analysis with parameter-keyed caching."""
    config = GeodesicConfig(
        M=M,
        r0=r0,
        b=b,
        direction=direction,
        phi0=0.0,
        phi_max=phi_max,
        capture_radius_factor=1.001,
        outer_radius=max(1.5 * r0, 50.0 * M),
        rtol=rtol,
        atol=atol,
        method=method,
        max_step=0.05,
    )
    return calculate_deflection_angle(
        config,
        apply_asymptotic_correction=apply_asymptotic_correction,
        verify_convergence=True,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def run_cached_lensing_sweep(
    M: float,
    r0: float,
    b_over_M_values: tuple[float, ...],
    phi_max: float,
    rtol: float,
    atol: float,
    method: str,
    apply_asymptotic_correction: bool,
) -> tuple[list[DeflectionResult], list[str]]:
    """Run a multi-impact-parameter gravitational lensing sweep with parameter-keyed caching."""
    results: list[DeflectionResult] = []
    errors: list[str] = []
    outer_r = max(1.5 * r0, 50.0 * M)

    for b_over_m in b_over_M_values:
        b_val = float(b_over_m * M)
        try:
            cfg = GeodesicConfig(
                M=M,
                r0=r0,
                b=b_val,
                direction="ingoing",
                phi0=0.0,
                phi_max=phi_max,
                capture_radius_factor=1.001,
                outer_radius=outer_r,
                rtol=rtol,
                atol=atol,
                method=method,
                max_step=0.04,
            )
            res = calculate_deflection_angle(
                cfg,
                apply_asymptotic_correction=apply_asymptotic_correction,
                verify_convergence=True,
            )
            results.append(res)
        except (ValueError, TypeError) as exc:
            errors.append(f"b/M = {b_over_m:.4f} (b = {b_val:.4f}): {exc}")

    return results, errors


@st.cache_data(show_spinner=False, max_entries=32)
def run_cached_shadow_image(
    M: float,
    r_obs: float,
    width: int,
    height: int,
    fov_rad: float,
    ray_model: ObserverRayModel,
    phi_max: float,
    rtol: float,
    atol: float,
    method: str,
) -> ShadowRayTracingResult:
    """Compute a 2D ray-traced Schwarzschild black hole shadow with parameter-keyed caching."""
    cfg = ShadowGridConfig(
        M=M,
        r_obs=r_obs,
        width=width,
        height=height,
        fov_rad=fov_rad,
        ray_model=ray_model,
        phi_max=phi_max,
        rtol=rtol,
        atol=atol,
        method=method,
        deduplicate_symmetric_rays=True,
    )
    return trace_black_hole_shadow(cfg)


def build_sweep_b_over_m_grid(
    b_min_over_M: float,
    b_max_over_M: float,
    num_points: int,
    include_near_critical_cluster: bool,
) -> tuple[float, ...]:
    """Construct a sorted tuple of dimensionless b / M values for a lensing sweep."""
    if b_max_over_M <= b_min_over_M:
        raise ValueError(
            f"Sweep maximum b_max/M ({b_max_over_M}) must be greater than "
            f"b_min/M ({b_min_over_M})."
        )
    if not (3 <= num_points <= 40):
        raise ValueError(f"num_points must be between 3 and 40; got {num_points}.")

    base = np.linspace(b_min_over_M, b_max_over_M, num_points, dtype=np.float64)
    if include_near_critical_cluster:
        b_crit_over_m = 3.0 * math.sqrt(3.0)
        cluster = np.array(
            [
                0.85 * b_crit_over_m,
                1.005 * b_crit_over_m,
                1.02 * b_crit_over_m,
                1.08 * b_crit_over_m,
                1.25 * b_crit_over_m,
            ],
            dtype=np.float64,
        )
        in_range = cluster[(cluster >= b_min_over_M) & (cluster <= b_max_over_M)]
        combined = np.unique(np.round(np.concatenate([base, in_range]), decimals=8))
        return tuple(float(val) for val in combined)

    return tuple(float(val) for val in base)


def main() -> None:
    """Render the Schwarzschild Photon Geodesics, Lensing & Shadow Imaging laboratory."""
    st.set_page_config(
        page_title="BlackHole-Relativity-Lab | Geodesics, Lensing & Shadow",
        layout="wide",
    )

    st.title("BlackHole-Relativity-Lab")
    st.subheader(
        "Schwarzschild Spacetime, Photon Geodesics, Gravitational Lensing & Black Hole Shadow"
    )
    st.markdown(
        """
        This laboratory numerically integrates the **exact Schwarzschild null-geodesic orbit equation**
        in the equatorial plane ($\\theta = \\pi/2$) using geometric units ($G = c = 1$, where length unit $[L] = GM/c^2$).
        Plots display actual numerical output from `scipy.integrate.solve_ivp` in Schwarzschild coordinates,
        not schematic drawings or synthetic camera photographs.
        """
    )

    # Sidebar controls: 1. Physical Parameters
    st.sidebar.header("1. Physical Parameters (G = c = 1)")
    mass_M = st.sidebar.number_input(
        "Black hole mass parameter M",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.3f",
        help="In geometric units (G = c = 1), mass M sets the characteristic length scale GM/c².",
    )

    scales = reference_scales(mass_M)

    r0 = st.sidebar.number_input(
        "Starting / Observer radius r₀",
        min_value=float(scales.r_s * 1.05),
        max_value=float(1000.0 * mass_M),
        value=float(50.0 * mass_M),
        step=float(1.0 * mass_M),
        format="%.3f",
        help="Must lie strictly outside the capture boundary (r₀ > r_capture >= 2M).",
    )

    b_input_mode = st.sidebar.radio(
        "Impact parameter input mode",
        options=["Dimensional b", "Dimensionless b / M"],
        index=1,
        help="Specify either dimensional impact parameter b [length] or dimensionless ratio b / M.",
    )

    if b_input_mode == "Dimensionless b / M":
        b_over_M_input = st.sidebar.number_input(
            "Dimensionless impact parameter b / M",
            min_value=0.05,
            max_value=200.0,
            value=6.0,
            step=0.1,
            format="%.5f",
            help="Critical impact parameter ratio is b_crit / M = 3√3 ≈ 5.19615.",
        )
        b = resolve_impact_parameter(mass_M, b_input_mode, 0.0, b_over_M_input)
    else:
        b_dim_input = st.sidebar.number_input(
            "Impact parameter b = L / E",
            min_value=0.05,
            max_value=float(200.0 * mass_M),
            value=float(6.0 * mass_M),
            step=float(0.1 * mass_M),
            format="%.5f",
            help=f"Critical impact parameter for M={mass_M:.2f} is b_crit ≈ {scales.b_crit:.5f}.",
        )
        b = resolve_impact_parameter(mass_M, b_input_mode, b_dim_input, 0.0)

    direction: RadialDirection = st.sidebar.selectbox(
        "Initial radial direction",
        options=["ingoing", "outgoing"],
        index=0,
        help="Ingoing means dr/dφ < 0 (du/dφ > 0); outgoing means dr/dφ > 0 (du/dφ < 0).",
    )

    # Sidebar controls: 2. Numerical Solver Controls
    st.sidebar.header("2. Numerical Solver Controls")
    log10_rtol = st.sidebar.slider(
        "log₁₀(rtol) [Relative Tolerance]",
        min_value=-13,
        max_value=-4,
        value=-10,
        step=1,
        help="Safe range restricted to [1e-13, 1e-4].",
    )
    log10_atol = st.sidebar.slider(
        "log₁₀(atol) [Absolute Tolerance]",
        min_value=-14,
        max_value=-5,
        value=-12,
        step=1,
        help="Safe range restricted to [1e-14, 1e-5].",
    )
    phi_max_turns = st.sidebar.slider(
        "Max angular interval (φ_max / 2π turns)",
        min_value=0.5,
        max_value=8.0,
        value=3.0,
        step=0.5,
        help="Trajectories exceeding this azimuthal limit are marked UNRESOLVED.",
    )
    method = st.sidebar.selectbox(
        "ODE Solver Method",
        options=["DOP853", "RK45"],
        index=0,
    )
    apply_asymptotic = st.sidebar.checkbox(
        "Include asymptotic far-field tail correction",
        value=True,
        help=(
            "When enabled, adds the exact curved-spacetime quadrature tail from finite "
            "(r₀, r_end) to r = ∞ so deflection angles can be compared directly against 4M/b."
        ),
    )

    rtol = 10.0**log10_rtol
    atol = 10.0**log10_atol
    phi_max = phi_max_turns * 2.0 * math.pi

    # Characteristic scales panel
    st.markdown("### Characteristic Schwarzschild Scales")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Schwarzschild Radius r_s = 2GM/c²", f"{scales.r_s:.5f}")
    col2.metric("Photon Sphere r_ph = 3GM/c²", f"{scales.r_ph:.5f}")
    col3.metric("Shadow Radius b_shadow = 3√3 GM/c²", f"{scales.b_crit:.5f}")
    col4.metric("Dimensionless b / M", f"{b / mass_M:.5f}")
    col5.metric("Ratio b / b_crit", f"{b / scales.b_crit:.5f}")

    # Run single-trajectory geodesic & deflection engine
    try:
        deflection_res = run_cached_single_trajectory(
            M=mass_M,
            r0=r0,
            b=b,
            direction=direction,
            phi_max=phi_max,
            rtol=rtol,
            atol=atol,
            method=method,
            apply_asymptotic_correction=apply_asymptotic,
        )
        result = deflection_res.geodesic_result
    except (ValueError, TypeError) as exc:
        st.error(f"**Invalid Physical or Numerical Input:** {exc}")
        st.stop()

    # Milestone 1 Status & Diagnostics Panel
    with st.expander("Single Trajectory Integration Status & Diagnostics", expanded=False):
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
            st.warning(f"**Solver Diagnostic Notice:** {warn_msg}")

        # Numerical Trajectory Plots
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            fig_cart, _ = plot_cartesian_trajectory(deflection_res)
            st.pyplot(fig_cart)
            plt.close(fig_cart)
        with pcol2:
            fig_rad, _ = plot_radius_vs_phi(deflection_res)
            st.pyplot(fig_rad)
            plt.close(fig_rad)

    st.divider()

    # =========================================================================
    # MILESTONE 2: GRAVITATIONAL LENSING & LIGHT DEFLECTION SECTION
    # =========================================================================
    st.markdown("## Gravitational Lensing & Light Deflection Analysis")
    regime = classify_lensing_regime(deflection_res.b_over_b_crit)

    lcol1, lcol2, lcol3 = st.columns(3)
    lcol1.metric("Photon Fate Classification", deflection_res.fate.value.upper())
    lcol2.metric("Physical Lensing Regime", regime.value.replace("_", " ").upper())
    lcol3.metric(
        "Weak-Field Prediction α_weak = 4M/b",
        f"{deflection_res.analytical_weak_field:.6f} rad "
        f"({math.degrees(deflection_res.analytical_weak_field):.3f}°)",
    )

    if deflection_res.fate == PhotonFate.ESCAPED and deflection_res.numerical_deflection is not None:
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric(
            "Numerical Deflection α_num",
            f"{deflection_res.numerical_deflection:.6f} rad "
            f"({math.degrees(deflection_res.numerical_deflection):.3f}°)",
        )
        mcol2.metric("Absolute Error |α_num - 4M/b|", f"{deflection_res.absolute_error:.4e} rad")
        mcol3.metric(
            "Relative Error vs. 4M/b",
            f"{100.0 * float(deflection_res.relative_error):.3f}%"
            if deflection_res.relative_error is not None
            else "N/A",
        )
        mcol4.metric(
            "Convergence Indicator",
            f"{deflection_res.convergence_indicator:.3e} rad"
            if deflection_res.convergence_indicator is not None
            else "N/A",
        )
    else:
        st.info(
            f"**No Scattering Deflection Angle Reported:** Photon fate is "
            f"**{deflection_res.fate.value.upper()}**. Scattering deflection angles are "
            "defined only for photons that scatter at a periapsis $r_p > 3M$ and escape."
        )

    for lens_warn in deflection_res.warnings:
        st.warning(f"**Lensing Regime Notice:** {lens_warn}")

    with st.expander("Multi-Parameter Lensing Sweep (Weak-Field vs. Strong-Field)", expanded=False):
        scol1, scol2, scol3, scol4 = st.columns(4)
        b_min_over_m = scol1.number_input(
            "Sweep min (b_min/M)", min_value=1.0, max_value=50.0, value=4.0, step=0.5
        )
        max_allowed_b_over_m = max(10.0, float(0.85 * (r0 / mass_M)))
        default_b_max = min(25.0, max_allowed_b_over_m)
        b_max_over_m = scol2.number_input(
            "Sweep max (b_max/M)",
            min_value=5.5,
            max_value=float(max_allowed_b_over_m),
            value=float(max(6.0, default_b_max)),
            step=1.0,
        )
        sweep_points = scol3.slider("Sweep trajectories", min_value=5, max_value=25, value=10)
        include_near_crit = scol4.checkbox("Sample near b_crit", value=True)

        try:
            b_grid_tuple = build_sweep_b_over_m_grid(
                b_min_over_m, b_max_over_m, sweep_points, include_near_crit
            )
            sweep_results, sweep_errors = run_cached_lensing_sweep(
                M=mass_M, r0=r0, b_over_M_values=b_grid_tuple, phi_max=phi_max,
                rtol=rtol, atol=atol, method=method, apply_asymptotic_correction=apply_asymptotic,
            )
            for err_msg in sweep_errors:
                st.warning(f"Skipped Sweep Ray: {err_msg}")

            if sweep_results:
                bundle_subset = sweep_results if len(sweep_results) <= 7 else [
                    sweep_results[int(i)] for i in np.linspace(0, len(sweep_results) - 1, 7, dtype=int)
                ]
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    fig_bundle, _ = plot_lensing_comparison(bundle_subset)
                    st.pyplot(fig_bundle)
                    plt.close(fig_bundle)
                with bcol2:
                    fig_def_curve, _ = plot_deflection_curve(sweep_results)
                    st.pyplot(fig_def_curve)
                    plt.close(fig_def_curve)

                fig_err_curve, _ = plot_deflection_error_curve(sweep_results)
                st.pyplot(fig_err_curve)
                plt.close(fig_err_curve)
        except (ValueError, TypeError) as exc:
            st.error(f"Sweep Configuration Error: {exc}")

    st.divider()

    # =========================================================================
    # MILESTONE 3: BLACK HOLE SHADOW & PHOTON SPHERE IMAGING DASHBOARD
    # =========================================================================
    st.markdown("## Interactive Black Hole Shadow Imaging Dashboard")
    st.markdown(
        """
        Simulate the two-dimensional observer image plane for a static observer at distance
        $r_{\\mathrm{obs}}$ (using the configured solver tolerances and limits). The engine traces 
        a grid of rays backward into Schwarzschild spacetime to generate the apparent shadow mask.
        """
    )

    sh_col1, sh_col2, sh_col3, sh_col4 = st.columns(4)
    img_width = sh_col1.slider("Image Width (px)", min_value=15, max_value=255, value=45, step=2)
    img_height = sh_col2.slider("Image Height (px)", min_value=15, max_value=255, value=45, step=2)

    default_obs_r = max(float(r0), float(3.2 * mass_M))
    psi_crit_default = shadow_angular_radius(default_obs_r, mass_M)
    default_fov_deg = min(math.degrees(3.2 * psi_crit_default), 100.0)
    
    fov_deg = sh_col3.number_input(
        "Field of View (degrees)",
        min_value=1.0,
        max_value=120.0,
        value=float(round(default_fov_deg, 2)),
        step=1.0,
        format="%.2f",
    )
    ray_model_choice: ObserverRayModel = sh_col4.selectbox(
        "Observer Camera Model",
        options=["local_angle", "impact_parameter"],
        index=0,
        help="local_angle applies Synge's exact static orthonormal tetrad mapping for finite distances.",
    )

    total_pixels = img_width * img_height
    if total_pixels > 2500:
        st.warning(
            f"**Performance Notice:** You requested a {img_width}x{img_height} grid "
            f"({total_pixels} total rays). Ray-tracing high resolutions dynamically may take "
            "several seconds on a standard laptop. Consider using a smaller grid (e.g., 45x45) "
            "for rapid exploratory tweaks."
        )

    try:
        # Trace the shadow grid using cached background execution
        with st.spinner(f"Tracing {total_pixels} rays through Schwarzschild spacetime..."):
            shadow_res = run_cached_shadow_image(
                M=mass_M,
                r_obs=default_obs_r,
                width=int(img_width),
                height=int(img_height),
                fov_rad=math.radians(fov_deg),
                ray_model=ray_model_choice,
                phi_max=phi_max,
                rtol=max(rtol, 1e-9),  # Keep minimum floor to preserve UI responsiveness
                atol=max(atol, 1e-11),
                method=method,
            )

        # 1. Ray Statistics
        st.markdown("#### Simulation Ray Statistics")
        stat1, stat2, stat3, stat4, stat5 = st.columns(5)
        stat1.metric("Total Image Pixels", f"{total_pixels}")
        stat2.metric("Captured (Shadow)", f"{shadow_res.num_captured}")
        stat3.metric("Escaped", f"{shadow_res.num_escaped}")
        stat4.metric("Unresolved", f"{shadow_res.num_unresolved}")
        stat5.metric(
            "Unique ODE Solves", 
            f"{shadow_res.unique_rays_traced}",
            help="Symmetric pixels with identical impact parameters reuse a single ODE integration.",
        )

        for swarn in shadow_res.warnings:
            st.warning(f"**Shadow Engine Notice:** {swarn}")

        # 2. Boundary Validation
        st.markdown("#### Shadow Boundary Validation")
        if shadow_res.num_captured > 0 and shadow_res.num_escaped > 0:
            max_b_cap = float(np.max(shadow_res.impact_parameter_grid[shadow_res.capture_mask]))
            min_b_esc = float(np.min(shadow_res.impact_parameter_grid[shadow_res.escaped_mask]))
            b_gap = min_b_esc - max_b_cap
            is_adequately_resolved = b_gap < (0.2 * mass_M)
            
            b_val_col1, b_val_col2, b_val_col3 = st.columns(3)
            b_val_col1.metric("Analytical b_shadow", f"{shadow_res.b_shadow:.4f}")
            if is_adequately_resolved:
                b_val_col2.metric("Numerical Grid Estimate", f"{shadow_res.estimated_b_crit_from_grid:.4f}")
            else:
                b_val_col2.metric("Numerical Grid Estimate", "Insufficient Resolution")
            
            b_val_col3.metric("Grid Boundary Uncertainty (Gap)", f"{b_gap:.4f}")
            
            if not is_adequately_resolved:
                st.info(
                    f"**Resolution Note:** The gap between the outermost captured ray ($b = {max_b_cap:.4f}$) "
                    f"and the innermost escaped ray ($b = {min_b_esc:.4f}$) is {b_gap:.4f}. "
                    "This boundary is not adequately resolved to report a precise numerical estimate. "
                    "Increase image resolution or decrease Field of View to tighten the bracket."
                )
        else:
            st.info("Boundary validation requires at least one captured ray and one escaped ray in the FOV.")

        # 3. Plots
        sp1, sp2 = st.columns(2)
        with sp1:
            fig_shadow, _ = plot_black_hole_shadow(shadow_res, show_reference_radii=True)
            st.pyplot(fig_shadow)
            plt.close(fig_shadow)
        with sp2:
            fig_prof, _ = plot_shadow_radial_profile(shadow_res)
            st.pyplot(fig_prof)
            plt.close(fig_prof)

    except (ValueError, TypeError) as exc:
        st.error(f"**Invalid Shadow Imaging Configuration:** {exc}")

    # Physical Explanation Expander
    with st.expander("Physics Assumptions: The Distant-Observer Approximation vs. Finite Distance"):
        st.markdown(
            r"""
            #### Why the Apparent Shadow ($3\sqrt{3}M$) Is Larger Than the Photon Sphere ($3M$)
            A light ray skimming the unstable photon sphere at $r_{\mathrm{ph}} = 3M$ has a conserved impact parameter 
            $b_{\mathrm{shadow}} = r_{\mathrm{ph}} / \sqrt{1 - 2M/r_{\mathrm{ph}}} = 3M / \sqrt{1/3} = 3\sqrt{3}M \approx 5.196M$. 
            Because photons originating near the photon sphere undergo severe gravitational bending as they climb out 
            of the potential well, the apparent cross-section of the shadow on a distant observer's sky is magnified 
            from $3M$ to $3\sqrt{3}M$.

            #### Asymptotic vs. Finite-Distance Viewing Geometry
            The analytical reference $b_{\mathrm{shadow}} = 3\sqrt{3}M$ is strictly a coordinate impact parameter.
            For an idealized observer located at infinity ($r \to \infty$), the angular radius of the shadow 
            on the sky is simply $\theta \approx b_{\mathrm{shadow}} / r_{\mathrm{obs}}$.

            However, for an observer at a **finite distance** $r_{\mathrm{obs}}$, the local gravitational curvature 
            alters the viewing angle. Rather than blindly applying the asymptotic formula, our `local_angle` camera 
            model correctly utilizes **Synge's exact finite-distance formula** for a static orthonormal tetrad:
            $$\sin(\psi_{\mathrm{crit}}) = \frac{3\sqrt{3}M}{r_{\mathrm{obs}}} \sqrt{1 - \frac{2M}{r_{\mathrm{obs}}}}$$
            The numerical engine correctly maps grid pixels back to $b$-space using this exact inverse relationship, 
            ensuring that the finite-distance curvature effects are correctly integrated.
            """
        )

if __name__ == "__main__":
    main()
