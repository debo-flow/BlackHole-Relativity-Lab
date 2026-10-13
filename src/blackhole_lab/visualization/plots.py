"""Scientific visualization routines for equatorial Schwarzschild photon trajectories.

Visualization Conventions
-------------------------
1. The Cartesian plot projects the equatorial coordinate chart (r, phi) onto
   Euclidean-like display coordinates x = r cos(phi), y = r sin(phi). This is a
   standard coordinate-space representation in Schwarzschild coordinates, NOT an
   isometric embedding of spatial curvature or a synthetic camera image.
2. Both the event horizon r_s = 2M and the unstable photon sphere r_ph = 3M are
   explicitly drawn for physical reference.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from blackhole_lab.numerics.geodesics import GeodesicResult, SolverStatus


def plot_cartesian_trajectory(
    result: GeodesicResult,
    ax: Axes | None = None,
    *,
    view_limit: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot the photon trajectory in equatorial Cartesian coordinates x = r cos(phi), y = r sin(phi).

    Parameters
    ----------
    result : GeodesicResult
        Output from ``solve_photon_geodesic``.
    ax : matplotlib.axes.Axes | None, optional
        Existing axes to draw onto. If None, creates a new Figure and Axes.
    view_limit : float | None, optional
        Half-width of the square plot window in units of length (M). Defaults to
        ``1.15 * result.config.r0``.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Reference circles: Event horizon r_s = 2M and photon sphere r_ph = 3M
    horizon_patch = Circle(
        (0.0, 0.0),
        radius=result.r_s,
        facecolor="#111111",
        edgecolor="#000000",
        linewidth=1.5,
        label=f"Event horizon $r_s = 2M = {result.r_s:.2f}$",
        zorder=3,
    )
    photon_sphere_patch = Circle(
        (0.0, 0.0),
        radius=result.r_ph,
        facecolor="none",
        edgecolor="#d95f02",
        linestyle="--",
        linewidth=1.4,
        label=f"Photon sphere $r_{{\\mathrm{{ph}}}} = 3M = {result.r_ph:.2f}$",
        zorder=4,
    )
    ax.add_patch(horizon_patch)
    ax.add_patch(photon_sphere_patch)

    valid = np.isfinite(result.x) & np.isfinite(result.y)
    if np.any(valid):
        x_v = result.x[valid]
        y_v = result.y[valid]

        color_map = {
            SolverStatus.COMPLETED: "#1f78b4",
            SolverStatus.CAPTURE_BOUNDARY: "#e31a1c",
            SolverStatus.INTEGRATION_LIMIT: "#7570b3",
            SolverStatus.SOLVER_FAILURE: "#666666",
        }
        traj_color = color_map.get(result.status, "#1f78b4")

        ax.plot(
            x_v,
            y_v,
            color=traj_color,
            linewidth=2.0,
            label=f"Photon geodesic ($b/b_{{\\mathrm{{crit}}}} = {result.b_over_b_crit:.4f}$)",
            zorder=5,
        )
        ax.scatter(
            [x_v[0]],
            [y_v[0]],
            color="#2ca02c",
            s=45,
            zorder=6,
            label=f"Start ($r_0 = {result.config.r0:.2f}, \\phi_0 = {result.config.phi0:.2f}$)",
        )
        ax.scatter(
            [x_v[-1]],
            [y_v[-1]],
            color=traj_color,
            marker="x",
            s=55,
            zorder=6,
            label=f"End ({result.termination_reason.value})",
        )

    lim = view_limit if view_limit is not None else max(1.15 * result.config.r0, 4.5 * result.r_s)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel(r"$x = r \cos\phi$ [geometric units, $G=c=1$]")
    ax.set_ylabel(r"$y = r \sin\phi$ [geometric units, $G=c=1$]")
    ax.set_title("Equatorial Schwarzschild Photon Trajectory (Coordinate Space)")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_radius_vs_phi(
    result: GeodesicResult,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the radial coordinate r(phi) versus azimuthal angle phi.

    Parameters
    ----------
    result : GeodesicResult
        Output from ``solve_photon_geodesic``.
    ax : matplotlib.axes.Axes | None, optional
        Existing axes to draw onto. If None, creates a new Figure and Axes.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
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
    ax.set_title(r"Radial Profile $r(\phi)$ vs. Azimuthal Angle $\phi$")
    ax.set_ylim(bottom=0.0, top=max(1.1 * result.config.r0, 1.2 * result.r_ph))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    return fig, ax
