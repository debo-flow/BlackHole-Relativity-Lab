"""Scientific visualization routines for Schwarzschild trajectories, lensing, and shadows.

Visualization Conventions & Scientific Honesty
----------------------------------------------
1. Coordinate-Space Projection (Not a Photograph):
   All 2D trajectory plots project the equatorial Schwarzschild chart (r, phi) onto
   Cartesian coordinate-space axes x = r cos(phi), y = r sin(phi). These plots are
   coordinate-space trajectory diagrams, NOT synthetic camera photographs, isometric
   spatial embeddings, or radiative-transfer accretion disk images.
2. Physical Reference Scales:
   Plots explicitly distinguish the Schwarzschild event horizon (r_s = 2GM/c^2 = 2M),
   the photon sphere (r_ph = 3GM/c^2 = 3M), and the apparent shadow impact-parameter
   radius for a distant observer (b_shadow = 3*sqrt(3)*GM/c^2 = 3*sqrt(3)*M).
3. Strict Fate Separation:
   Trajectories and shadow pixels are classified into ``ESCAPED``, ``CAPTURED``,
   and ``UNRESOLVED``. Unresolved trajectories or pixels are always rendered as a
   distinct category and are never merged into captured or escaped regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch
from numpy.typing import NDArray

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    GeodesicResult,
)
from blackhole_lab.numerics.lensing import (
    NEAR_CRITICAL_B_OVER_BCRIT,
    WEAK_FIELD_MIN_B_OVER_BCRIT,
    DeflectionResult,
    PhotonFate,
    calculate_deflection_angle,
    classify_photon_fate,
)
from blackhole_lab.numerics.shadow import (
    ShadowRayTracingResult,
)
from blackhole_lab.physics.schwarzschild import (
    _ensure_finite_scalar,
    reference_scales,
    validate_impact_parameter,
    validate_mass,
    validate_radius,
    weak_field_deflection_angle,
)


class LensingRegime(str, Enum):
    """Physical regime classification based on dimensionless ratio b / b_crit."""

    SUBCRITICAL_CAPTURE = "subcritical_capture"
    NEAR_CRITICAL = "near_critical"
    STRONG_FIELD = "strong_field"
    MODERATE_FIELD = "moderate_field"
    WEAK_FIELD = "weak_field"


@dataclass(frozen=True)
class TrajectorySegments:
    """Coordinate-space trajectory split into incoming and outgoing segments.

    For ``PhotonFate.CAPTURED`` and ``PhotonFate.UNRESOLVED``, ``outgoing_x`` and
    ``outgoing_y`` are guaranteed to be empty 1D arrays so non-escaping photons
    are never rendered with an escaping branch.
    """

    fate: PhotonFate
    regime: LensingRegime
    incoming_x: NDArray[np.float64]
    incoming_y: NDArray[np.float64]
    outgoing_x: NDArray[np.float64]
    outgoing_y: NDArray[np.float64]
    periapsis_point: tuple[float, float] | None
    periapsis_radius: float | None
    has_near_critical_warning: bool


def _extract_geodesic_result(
    item: GeodesicResult | DeflectionResult,
) -> GeodesicResult:
    """Extract the underlying GeodesicResult or raise a clear TypeError."""
    if isinstance(item, GeodesicResult):
        return item
    if isinstance(item, DeflectionResult):
        return item.geodesic_result
    raise TypeError(
        "Expected a GeodesicResult or DeflectionResult instance; "
        f"got {type(item).__name__}."
    )


def classify_lensing_regime(b_over_b_crit: float) -> LensingRegime:
    """Classify the gravitational lensing regime from the dimensionless ratio b / b_crit."""
    ratio = _ensure_finite_scalar(b_over_b_crit, "b_over_b_crit")
    if ratio <= 0.0:
        raise ValueError(f"b_over_b_crit must be strictly positive; got {ratio}.")
    if ratio <= 1.0:
        return LensingRegime.SUBCRITICAL_CAPTURE
    if ratio <= NEAR_CRITICAL_B_OVER_BCRIT:
        return LensingRegime.NEAR_CRITICAL
    if ratio < WEAK_FIELD_MIN_B_OVER_BCRIT:
        return LensingRegime.STRONG_FIELD
    if ratio < 6.0:
        return LensingRegime.MODERATE_FIELD
    return LensingRegime.WEAK_FIELD


def split_trajectory_segments(
    source: GeodesicResult | DeflectionResult,
) -> TrajectorySegments:
    """Split a numerical trajectory into incoming and (if escaped) outgoing segments."""
    result = _extract_geodesic_result(source)
    fate = (
        source.fate
        if isinstance(source, DeflectionResult)
        else classify_photon_fate(result)
    )
    regime = classify_lensing_regime(result.b_over_b_crit)

    valid = np.isfinite(result.x) & np.isfinite(result.y) & np.isfinite(result.r)
    x_v = np.asarray(result.x[valid], dtype=np.float64)
    y_v = np.asarray(result.y[valid], dtype=np.float64)
    r_v = np.asarray(result.r[valid], dtype=np.float64)

    diag_warnings = getattr(result.diagnostics, "warnings", ())
    is_near_crit = bool(getattr(result.diagnostics, "is_near_critical", False))
    has_warning = (
        is_near_crit
        or regime == LensingRegime.NEAR_CRITICAL
        or len(diag_warnings) > 0
    )

    empty = np.array([], dtype=np.float64)

    if x_v.size == 0:
        return TrajectorySegments(
            fate=fate,
            regime=regime,
            incoming_x=empty,
            incoming_y=empty,
            outgoing_x=empty,
            outgoing_y=empty,
            periapsis_point=None,
            periapsis_radius=None,
            has_near_critical_warning=has_warning,
        )

    if fate == PhotonFate.ESCAPED and x_v.size >= 2:
        idx_min = int(np.argmin(r_v))
        inc_x = x_v[: idx_min + 1]
        inc_y = y_v[: idx_min + 1]
        out_x = x_v[idx_min:]
        out_y = y_v[idx_min:]
        peri_pt = (float(x_v[idx_min]), float(y_v[idx_min]))
        peri_r = float(r_v[idx_min])
        return TrajectorySegments(
            fate=fate,
            regime=regime,
            incoming_x=inc_x,
            incoming_y=inc_y,
            outgoing_x=out_x,
            outgoing_y=out_y,
            periapsis_point=peri_pt,
            periapsis_radius=peri_r,
            has_near_critical_warning=has_warning,
        )

    return TrajectorySegments(
        fate=fate,
        regime=regime,
        incoming_x=x_v,
        incoming_y=y_v,
        outgoing_x=empty,
        outgoing_y=empty,
        periapsis_point=None,
        periapsis_radius=None,
        has_near_critical_warning=has_warning,
    )


def simulate_lensing_bundle(
    M: float = 1.0,
    r0: float = 25.0,
    b_factors: Sequence[float] = (0.75, 1.005, 1.15, 1.8, 3.2, 6.5),
    *,
    outer_radius: float | None = None,
    phi_max: float = 8.0 * np.pi,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    apply_asymptotic_correction: bool = True,
) -> list[DeflectionResult]:
    """Simulate a bundle of photon trajectories across lensing regimes using solve_ivp."""
    scales = reference_scales(M)
    r0_val = validate_radius(r0, scales.M, require_exterior=True)
    if isinstance(b_factors, (str, bytes)) or len(b_factors) == 0:
        raise ValueError("b_factors must be a non-empty sequence of positive numbers.")

    results: list[DeflectionResult] = []
    for factor in b_factors:
        f_val = _ensure_finite_scalar(factor, "b_factor")
        b_val = validate_impact_parameter(f_val * scales.b_crit)
        cfg = GeodesicConfig(
            M=scales.M,
            r0=r0_val,
            b=b_val,
            direction="ingoing",
            phi0=0.0,
            phi_max=phi_max,
            outer_radius=outer_radius,
            rtol=rtol,
            atol=atol,
            max_step=0.03,
        )
        def_res = calculate_deflection_angle(
            cfg,
            apply_asymptotic_correction=apply_asymptotic_correction,
            verify_convergence=False,
        )
        results.append(def_res)

    return results


def _draw_black_hole_references(
    ax: Axes,
    r_s: float,
    r_ph: float,
    b_crit: float,
    *,
    show_b_crit: bool = True,
) -> None:
    """Draw the black hole interior, event horizon (2M), photon sphere (3M), and b_crit."""
    horizon_patch = Circle(
        (0.0, 0.0),
        radius=r_s,
        facecolor="#111111",
        edgecolor="#000000",
        linewidth=1.6,
        label=f"Black hole / Event horizon ($r_s = 2M = {r_s:.2f}$)",
        zorder=3,
    )
    ax.add_patch(horizon_patch)

    photon_sphere_patch = Circle(
        (0.0, 0.0),
        radius=r_ph,
        facecolor="none",
        edgecolor="#d95f02",
        linestyle="--",
        linewidth=1.4,
        label=f"Photon sphere ($r_{{\\mathrm{{ph}}}} = 3M = {r_ph:.2f}$)",
        zorder=4,
    )
    ax.add_patch(photon_sphere_patch)

    if show_b_crit:
        b_crit_patch = Circle(
            (0.0, 0.0),
            radius=b_crit,
            facecolor="none",
            edgecolor="#7570b3",
            linestyle=":",
            linewidth=1.3,
            label=(
                f"Critical impact parameter "
                f"($b_{{\\mathrm{{crit}}}} = 3\\sqrt{{3}}M = {b_crit:.2f}$)"
            ),
            zorder=4,
        )
        ax.add_patch(b_crit_patch)


def plot_cartesian_trajectory(
    source: GeodesicResult | DeflectionResult,
    ax: Axes | None = None,
    *,
    view_limit: float | None = None,
    show_impact_parameter: bool = True,
    split_ingoing_outgoing: bool = True,
) -> tuple[Figure, Axes]:
    """Plot a photon trajectory in equatorial coordinate space x = r cos(phi), y = r sin(phi)."""
    result = _extract_geodesic_result(source)
    segments = split_trajectory_segments(source)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    lim = (
        _ensure_finite_scalar(view_limit, "view_limit")
        if view_limit is not None
        else max(1.15 * result.config.r0, 4.5 * result.r_s)
    )
    if lim <= 0.0:
        raise ValueError(f"view_limit must be strictly positive; got {lim}.")

    _draw_black_hole_references(
        ax,
        r_s=result.r_s,
        r_ph=result.r_ph,
        b_crit=result.b_crit,
        show_b_crit=show_impact_parameter,
    )

    if show_impact_parameter and result.config.b <= lim:
        b_ref_patch = Circle(
            (0.0, 0.0),
            radius=result.config.b,
            facecolor="none",
            edgecolor="#2ca02c",
            linestyle="-.",
            linewidth=1.0,
            alpha=0.75,
            label=f"Impact parameter reference ($b = {result.config.b:.2f}$)",
            zorder=4,
        )
        ax.add_patch(b_ref_patch)

    if segments.incoming_x.size > 0:
        if segments.fate == PhotonFate.ESCAPED and split_ingoing_outgoing:
            ax.plot(
                segments.incoming_x,
                segments.incoming_y,
                color="#1f78b4",
                linestyle="-",
                linewidth=2.0,
                label=f"Incoming path ($b/b_{{\\mathrm{{crit}}}} = {result.b_over_b_crit:.4f}$)",
                zorder=5,
            )
            ax.plot(
                segments.outgoing_x,
                segments.outgoing_y,
                color="#00a087",
                linestyle="--",
                linewidth=2.1,
                label="Outgoing path (escaped)",
                zorder=5,
            )
            if segments.periapsis_point is not None and segments.periapsis_radius is not None:
                ax.scatter(
                    [segments.periapsis_point[0]],
                    [segments.periapsis_point[1]],
                    color="#ff7f00",
                    marker="o",
                    s=45,
                    zorder=6,
                    label=f"Periapsis ($r_p = {segments.periapsis_radius:.2f}$)",
                )
            end_x, end_y = segments.outgoing_x[-1], segments.outgoing_y[-1]
            end_color = "#00a087"
        else:
            fate_colors = {
                PhotonFate.ESCAPED: "#1f78b4",
                PhotonFate.CAPTURED: "#e31a1c",
                PhotonFate.UNRESOLVED: "#6a3d9a",
            }
            traj_color = fate_colors[segments.fate]
            fate_label = segments.fate.value.upper()
            ax.plot(
                segments.incoming_x,
                segments.incoming_y,
                color=traj_color,
                linewidth=2.0,
                label=(
                    f"Photon path [{fate_label}] "
                    f"($b/b_{{\\mathrm{{crit}}}} = {result.b_over_b_crit:.4f}$)"
                ),
                zorder=5,
            )
            end_x, end_y = segments.incoming_x[-1], segments.incoming_y[-1]
            end_color = traj_color

        ax.scatter(
            [segments.incoming_x[0]],
            [segments.incoming_y[0]],
            color="#2ca02c",
            s=45,
            zorder=6,
            label=f"Start ($r_0 = {result.config.r0:.2f}, \\phi_0 = {result.config.phi0:.2f}$)",
        )
        ax.scatter(
            [end_x],
            [end_y],
            color=end_color,
            marker="x",
            s=55,
            zorder=6,
            label=f"End ({result.termination_reason.value})",
        )

    if segments.has_near_critical_warning:
        ax.text(
            0.02,
            0.02,
            f"Regime: {segments.regime.value} | Fate: {segments.fate.value.upper()}",
            transform=ax.transAxes,
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff7bc", "alpha": 0.85},
            zorder=7,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel(r"$x = r \cos\phi$ [geometric units, $G=c=1$]")
    ax.set_ylabel(r"$y = r \sin\phi$ [geometric units, $G=c=1$]")
    ax.set_title(
        "Equatorial Schwarzschild Photon Trajectory\n"
        "(Coordinate-Space Visualization — Not a Literal Photograph)"
    )
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    fig.tight_layout()
    return fig, ax


def plot_lensing_comparison(
    trajectories: Sequence[GeodesicResult | DeflectionResult],
    ax: Axes | None = None,
    *,
    view_limit: float | None = None,
    show_b_crit_reference: bool = True,
) -> tuple[Figure, Axes]:
    """Plot multiple photon trajectories comparing weak, moderate, strong, near-critical, and capture."""
    if isinstance(trajectories, (str, bytes)) or len(trajectories) == 0:
        raise ValueError("trajectories must be a non-empty sequence of trajectory results.")

    geodesics = [_extract_geodesic_result(item) for item in trajectories]
    first = geodesics[0]
    mass = first.config.M

    for g in geodesics[1:]:
        if not np.isclose(g.config.M, mass, rtol=1e-12, atol=0.0):
            raise ValueError(
                "All trajectories in plot_lensing_comparison must share the same black hole "
                f"mass M; got {mass} and {g.config.M}."
            )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    max_r0 = max(g.config.r0 for g in geodesics)
    lim = (
        _ensure_finite_scalar(view_limit, "view_limit")
        if view_limit is not None
        else max(1.1 * max_r0, 4.5 * first.r_s)
    )
    if lim <= 0.0:
        raise ValueError(f"view_limit must be strictly positive; got {lim}.")

    _draw_black_hole_references(
        ax,
        r_s=first.r_s,
        r_ph=first.r_ph,
        b_crit=first.b_crit,
        show_b_crit=show_b_crit_reference,
    )

    palette = plt.cm.viridis(np.linspace(0.15, 0.88, len(trajectories)))

    for idx, item in enumerate(trajectories):
        g_res = geodesics[idx]
        seg = split_trajectory_segments(item)
        if seg.incoming_x.size == 0:
            continue

        b_ratio = g_res.b_over_b_crit
        regime_tag = seg.regime.value.replace("_", " ")

        if seg.fate == PhotonFate.ESCAPED:
            color = palette[idx]
            ax.plot(
                seg.incoming_x,
                seg.incoming_y,
                color=color,
                linestyle="-",
                linewidth=1.8,
                label=(
                    f"ESCAPED ({regime_tag}): $b/b_{{\\mathrm{{crit}}}}={b_ratio:.3f}$ "
                    "[solid=in, dashed=out]"
                ),
                zorder=5,
            )
            ax.plot(
                seg.outgoing_x,
                seg.outgoing_y,
                color=color,
                linestyle="--",
                linewidth=1.8,
                zorder=5,
            )
        elif seg.fate == PhotonFate.CAPTURED:
            ax.plot(
                seg.incoming_x,
                seg.incoming_y,
                color="#e31a1c",
                linestyle="-",
                linewidth=1.9,
                label=f"CAPTURED ($b/b_{{\\mathrm{{crit}}}}={b_ratio:.3f}$)",
                zorder=5,
            )
            ax.scatter(
                [seg.incoming_x[-1]],
                [seg.incoming_y[-1]],
                color="#e31a1c",
                marker="x",
                s=50,
                zorder=6,
            )
        else:
            ax.plot(
                seg.incoming_x,
                seg.incoming_y,
                color="#6a3d9a",
                linestyle="-.",
                linewidth=1.8,
                label=f"UNRESOLVED ($b/b_{{\\mathrm{{crit}}}}={b_ratio:.3f}$)",
                zorder=5,
            )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel(r"$x = r \cos\phi$ [geometric units, $G=c=1$]")
    ax.set_ylabel(r"$y = r \sin\phi$ [geometric units, $G=c=1$]")
    ax.set_title(
        "Schwarzschild Strong-Field Lensing Comparison\n"
        "(Coordinate-Space Trajectory Visualization — Not a Literal Photograph)"
    )
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)
    fig.tight_layout()
    return fig, ax


def plot_deflection_curve(
    results: Sequence[DeflectionResult],
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot numerical deflection angle alpha(b) vs. analytical weak-field 4M/b reference."""
    if isinstance(results, (str, bytes)) or len(results) == 0:
        raise ValueError("results must be a non-empty sequence of DeflectionResult objects.")
    for item in results:
        if not isinstance(item, DeflectionResult):
            raise TypeError(
                f"All elements must be DeflectionResult instances; got {type(item).__name__}."
            )

    mass = validate_mass(results[0].M)
    b_crit = results[0].b_crit
    for item in results[1:]:
        if not np.isclose(item.M, mass, rtol=1e-12, atol=0.0):
            raise ValueError("All DeflectionResult items must share the same black hole mass M.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    escaped = [
        r for r in results if r.fate == PhotonFate.ESCAPED and r.numerical_deflection is not None
    ]
    captured = [r for r in results if r.fate == PhotonFate.CAPTURED]
    unresolved = [r for r in results if r.fate == PhotonFate.UNRESOLVED]

    ax.axvline(
        b_crit / mass,
        color="#d95f02",
        linestyle="--",
        linewidth=1.5,
        label=r"Critical threshold $b_{\mathrm{crit}} / M = 3\sqrt{3} \approx 5.196$",
    )

    if escaped:
        escaped_sorted = sorted(escaped, key=lambda r: r.b)
        b_over_m = np.array([r.b / mass for r in escaped_sorted], dtype=np.float64)
        alpha_num = np.array(
            [float(r.numerical_deflection) for r in escaped_sorted], dtype=np.float64
        )

        b_grid = np.linspace(max(min(b_over_m), (b_crit / mass) * 1.001), max(b_over_m), 250)
        alpha_weak_grid = np.array(
            [weak_field_deflection_angle(1.0, float(bm)) for bm in b_grid],
            dtype=np.float64,
        )

        ax.plot(
            b_grid,
            alpha_weak_grid,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.8,
            label=r"Weak-field approximation $\alpha_{\mathrm{weak}} = 4M/b$",
        )
        ax.plot(
            b_over_m,
            alpha_num,
            color="#1f78b4",
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            label=r"Numerical deflection $\alpha_{\mathrm{num}}(b)$ (ESCAPED)",
        )

    if captured:
        b_cap = [r.b / mass for r in captured]
        ax.scatter(
            b_cap,
            np.zeros(len(b_cap)),
            color="#e31a1c",
            marker="x",
            s=55,
            label="CAPTURED (no scattering deflection)",
            zorder=6,
        )

    if unresolved:
        b_unres = [r.b / mass for r in unresolved]
        ax.scatter(
            b_unres,
            np.zeros(len(b_unres)),
            color="#6a3d9a",
            marker="d",
            s=50,
            label="UNRESOLVED",
            zorder=6,
        )

    ax.set_xlabel(r"Dimensionless impact parameter $b / M$ [geometric units]")
    ax.set_ylabel(r"Deflection angle $\alpha$ [rad]")
    ax.set_title("Schwarzschild Gravitational Light Deflection vs. Impact Parameter")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.92)
    fig.tight_layout()
    return fig, ax


def plot_deflection_error_curve(
    results: Sequence[DeflectionResult],
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot weak-field approximation error and numerical convergence indicator vs. b / M."""
    if isinstance(results, (str, bytes)) or len(results) == 0:
        raise ValueError("results must be a non-empty sequence of DeflectionResult objects.")
    for item in results:
        if not isinstance(item, DeflectionResult):
            raise TypeError(
                f"All elements must be DeflectionResult instances; got {type(item).__name__}."
            )

    mass = validate_mass(results[0].M)
    b_crit = results[0].b_crit
    for item in results[1:]:
        if not np.isclose(item.M, mass, rtol=1e-12, atol=0.0):
            raise ValueError("All DeflectionResult items must share the same black hole mass M.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    escaped = [
        r
        for r in results
        if r.fate == PhotonFate.ESCAPED
        and r.numerical_deflection is not None
        and r.absolute_error is not None
    ]

    ax.axvline(
        b_crit / mass,
        color="#d95f02",
        linestyle="--",
        linewidth=1.5,
        label=r"Critical threshold $b_{\mathrm{crit}} / M = 3\sqrt{3}$",
    )

    if escaped:
        escaped_sorted = sorted(escaped, key=lambda r: r.b)
        b_over_m = np.array([r.b / mass for r in escaped_sorted], dtype=np.float64)
        abs_err_1pn = np.array(
            [max(float(r.absolute_error), 1e-16) for r in escaped_sorted],
            dtype=np.float64,
        )
        abs_err_2pn = np.array(
            [
                max(abs(float(r.numerical_deflection) - r.analytical_post_newtonian_2), 1e-16)
                for r in escaped_sorted
            ],
            dtype=np.float64,
        )

        ax.semilogy(
            b_over_m,
            abs_err_1pn,
            color="#e31a1c",
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            label=r"Weak-field error $|\alpha_{\mathrm{num}} - 4M/b|$ [rad]",
        )
        ax.semilogy(
            b_over_m,
            abs_err_2pn,
            color="#1f78b4",
            marker="s",
            markersize=4.5,
            linestyle="--",
            linewidth=1.5,
            label=r"2PN residual $|\alpha_{\mathrm{num}} - \alpha_{\mathrm{2PN}}|$ [rad]",
        )

        conv_pts = [
            (r.b / mass, max(float(r.convergence_indicator), 1e-16))
            for r in escaped_sorted
            if r.convergence_indicator is not None
        ]
        if conv_pts:
            b_conv = np.array([pt[0] for pt in conv_pts], dtype=np.float64)
            c_vals = np.array([pt[1] for pt in conv_pts], dtype=np.float64)
            ax.semilogy(
                b_conv,
                c_vals,
                color="#6a3d9a",
                marker="^",
                markersize=4.5,
                linestyle=":",
                linewidth=1.4,
                label="Numerical solver convergence indicator [rad]",
            )

    ax.set_xlabel(r"Dimensionless impact parameter $b / M$ [geometric units]")
    ax.set_ylabel("Absolute Angular Error / Discrepancy [rad]")
    ax.set_title(r"Analytical Approximation Error & Numerical Convergence vs. $b / M$")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    fig.tight_layout()
    return fig, ax


def plot_radius_vs_phi(
    source: GeodesicResult | DeflectionResult,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the radial coordinate r(phi) versus azimuthal angle phi."""
    result = _extract_geodesic_result(source)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    valid = np.isfinite(result.phi) & np.isfinite(result.r)
    if np.any(valid):
        phi_v = result.phi[valid]
        r_v = result.r[valid]
        ax.plot(phi_v, r_v, color="#1f78b4", linewidth=2.0, label=r"$r(\phi) = 1 / u(\phi)$")

    ax.axhline(
        result.r_s,
        color="#111111",
        linestyle="-",
        linewidth=1.4,
        label=f"Event horizon $r_s = 2M = {result.r_s:.2f}$",
    )
    ax.axhline(
        result.r_ph,
        color="#d95f02",
        linestyle="--",
        linewidth=1.4,
        label=f"Photon sphere $r_{{\\mathrm{{ph}}}} = 3M = {result.r_ph:.2f}$",
    )

    ax.set_xlabel(r"Azimuthal angle $\phi$ [rad]")
    ax.set_ylabel(r"Radial coordinate $r$ [geometric units]")
    ax.set_title(r"Radial Coordinate Profile $r(\phi)$ vs. Azimuthal Angle $\phi$")
    ax.set_ylim(bottom=0.0, top=max(1.1 * result.config.r0, 1.2 * result.r_ph))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_black_hole_shadow(
    shadow_result: ShadowRayTracingResult,
    ax: Axes | None = None,
    *,
    show_reference_radii: bool = True,
) -> tuple[Figure, Axes]:
    """Visualize the 2D Schwarzschild black hole shadow on the observer's image plane.

    Plots the ray-traced classification map in apparent impact-parameter coordinates
    (b_x, b_y) [in units of GM/c^2 = M], explicitly distinguishing:
    - Captured ray region (Black Hole Shadow interior)
    - Escaped ray region (surrounding sky)
    - Unresolved pixels, if present (rendered in distinct amber/gold so they are
      never conflated with captured shadow pixels)
    - Apparent shadow boundary: b_shadow = 3*sqrt(3)*GM/c^2 = 3*sqrt(3)*M
    - Photon sphere coordinate radius reference: r_ph = 3GM/c^2 = 3M
    - Schwarzschild event horizon coordinate radius reference: r_s = 2GM/c^2 = 2M

    Parameters
    ----------
    shadow_result : ShadowRayTracingResult
        Output from ``trace_black_hole_shadow``.
    ax : matplotlib.axes.Axes | None, optional
        Existing axes to draw onto. If None, creates a new Figure and Axes.
    show_reference_radii : bool, default True
        If True, overlays the exact circles for b_shadow = 3*sqrt(3)*M, r_ph = 3M,
        and r_s = 2M to illustrate gravitational lensing magnification of the shadow.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if not isinstance(shadow_result, ShadowRayTracingResult):
        raise TypeError(
            "shadow_result must be a ShadowRayTracingResult instance; "
            f"got {type(shadow_result).__name__}."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Map classification codes {-1: UNRESOLVED, 0: ESCAPED, 1: CAPTURED} to indices {0, 1, 2}
    index_grid = shadow_result.classification_map.astype(np.int32) + 1
    cmap = ListedColormap(["#f39c12", "#dceaf7", "#0d0d0d"])

    bx_1d = shadow_result.bx_grid[shadow_result.config.image_height // 2, :]
    by_1d = shadow_result.by_grid[:, shadow_result.config.image_width // 2]
    b_max_x = float(np.max(np.abs(bx_1d)))
    b_max_y = float(np.max(np.abs(by_1d)))

    ax.imshow(
        index_grid,
        origin="lower",
        extent=(-b_max_x, b_max_x, -b_max_y, b_max_y),
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
        zorder=1,
    )

    legend_handles: list[object] = [
        Patch(
            facecolor="#0d0d0d",
            edgecolor="#000000",
            label=f"Captured rays (Shadow, N={shadow_result.num_captured})",
        ),
        Patch(
            facecolor="#dceaf7",
            edgecolor="#7f9eb2",
            label=f"Escaped rays (N={shadow_result.num_escaped})",
        ),
    ]
    if shadow_result.num_unresolved > 0:
        legend_handles.append(
            Patch(
                facecolor="#f39c12",
                edgecolor="#b9770e",
                label=f"Unresolved pixels (N={shadow_result.num_unresolved})",
            )
        )

    if show_reference_radii:
        shadow_circle = Circle(
            (0.0, 0.0),
            radius=shadow_result.b_shadow,
            facecolor="none",
            edgecolor="#e31a1c",
            linestyle="-",
            linewidth=2.0,
            label=(
                f"Shadow boundary $b_{{\\mathrm{{shadow}}}} = 3\\sqrt{{3}}GM/c^2 = "
                f"{shadow_result.b_shadow:.3f}$"
            ),
            zorder=4,
        )
        rph_circle = Circle(
            (0.0, 0.0),
            radius=shadow_result.r_ph,
            facecolor="none",
            edgecolor="#ff7f00",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Photon sphere radius $r_{{\\mathrm{{ph}}}} = 3GM/c^2 = "
                f"{shadow_result.r_ph:.2f}$"
            ),
            zorder=4,
        )
        rs_circle = Circle(
            (0.0, 0.0),
            radius=shadow_result.r_s,
            facecolor="none",
            edgecolor="#33a02c",
            linestyle=":",
            linewidth=1.6,
            label=(
                f"Schwarzschild radius $r_s = 2GM/c^2 = "
                f"{shadow_result.r_s:.2f}$"
            ),
            zorder=4,
        )
        ax.add_patch(shadow_circle)
        ax.add_patch(rph_circle)
        ax.add_patch(rs_circle)
        legend_handles.extend([shadow_circle, rph_circle, rs_circle])

    ax.set_xlim(-b_max_x, b_max_x)
    ax.set_ylim(-b_max_y, b_max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.45, color="#555555")
    ax.set_xlabel(r"Apparent impact coordinate $b_x$ [$GM/c^2$, geometric units $G=c=1$]")
    ax.set_ylabel(r"Apparent impact coordinate $b_y$ [$GM/c^2$, geometric units $G=c=1$]")
    ax.set_title(
        f"Schwarzschild Black Hole Shadow on Observer Image Plane "
        f"($r_{{\\mathrm{{obs}}}} = {shadow_result.config.r_obs:.1f}M$)\n"
        "(Ray-Traced Classification Mask — Not a Radiative Accretion Photograph)"
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7.8, framealpha=0.92)
    fig.tight_layout()
    return fig, ax


def plot_shadow_radial_profile(
    shadow_result: ShadowRayTracingResult,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the radial classification profile across the observer image plane vs. b / M.

    Clearly distinguishes r_s = 2GM/c^2, r_ph = 3GM/c^2, and b_shadow = 3*sqrt(3)*GM/c^2.
    """
    if not isinstance(shadow_result, ShadowRayTracingResult):
        raise TypeError(
            "shadow_result must be a ShadowRayTracingResult instance; "
            f"got {type(shadow_result).__name__}."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    mass = shadow_result.config.M
    b_flat = shadow_result.impact_parameter_grid.ravel() / mass
    code_flat = shadow_result.classification_map.ravel()

    order = np.argsort(b_flat)
    b_sorted = b_flat[order]
    code_sorted = code_flat[order]

    ax.scatter(
        b_sorted[code_sorted == 1],
        np.ones(int(np.sum(code_sorted == 1))),
        color="#111111",
        s=16,
        label="Captured (1)",
        zorder=4,
    )
    ax.scatter(
        b_sorted[code_sorted == 0],
        np.zeros(int(np.sum(code_sorted == 0))),
        color="#1f78b4",
        s=16,
        label="Escaped (0)",
        zorder=4,
    )
    if np.any(code_sorted == -1):
        ax.scatter(
            b_sorted[code_sorted == -1],
            np.full(int(np.sum(code_sorted == -1)), 0.5),
            color="#f39c12",
            marker="d",
            s=28,
            label="Unresolved",
            zorder=5,
        )

    ax.axvline(
        shadow_result.r_s / mass,
        color="#33a02c",
        linestyle=":",
        linewidth=1.6,
        label=r"$r_s / (GM/c^2) = 2$",
    )
    ax.axvline(
        shadow_result.r_ph / mass,
        color="#ff7f00",
        linestyle="--",
        linewidth=1.6,
        label=r"$r_{\mathrm{ph}} / (GM/c^2) = 3$",
    )
    ax.axvline(
        shadow_result.b_shadow / mass,
        color="#e31a1c",
        linestyle="-",
        linewidth=1.8,
        label=r"$b_{\mathrm{shadow}} / (GM/c^2) = 3\sqrt{3} \approx 5.196$",
    )

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["Escaped", "Unresolved", "Captured"])
    ax.set_ylim(-0.2, 1.25)
    ax.set_xlabel(r"Dimensionless impact parameter $b / (GM/c^2)$")
    ax.set_ylabel("Ray Classification")
    ax.set_title("Radial Ray Classification Profile & Characteristic Scales")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center right", fontsize=8.0, framealpha=0.92)
    fig.tight_layout()
    return fig, ax
