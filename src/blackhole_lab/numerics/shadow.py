"""Schwarzschild black hole shadow physics, critical boundary finder, and ray-tracing engine.

Unit Conventions & Physical Reference Scales
--------------------------------------------
Internally, all calculations use geometric units (G = c = 1) in which mass M, radial
coordinate r, and impact parameter b carry dimensions of length [L]. To convert any
length scale to SI units for a black hole of physical mass M_phys, multiply by GM_phys/c^2:

1. Schwarzschild radius (event horizon):
       r_s = 2M                  (physical: r_s = 2 G M_phys / c^2)
2. Photon sphere radius (unstable circular photon orbit):
       r_ph = 3M                 (physical: r_ph = 3 G M_phys / c^2)
3. Apparent shadow impact-parameter radius for a distant observer:
       b_shadow = b_crit = 3*sqrt(3)*M  (physical: b_shadow = 3*sqrt(3) G M_phys / c^2)

Because outgoing photons from r_ph = 3M experience gravitational redshift and light
bending as they climb out of the potential well, the apparent shadow radius b_shadow =
r_ph / sqrt(1 - 2M / r_ph) = 3*sqrt(3)*M ≈ 5.19615 M is larger than both r_s = 2M and
r_ph = 3M.

Observer Setup & Image-Plane Ray Mapping
----------------------------------------
Consider a static observer at radial coordinate r_obs > r_ph = 3M looking toward the
center of the Schwarzschild black hole. Each pixel (i, j) on the 2D image plane has
angular sky coordinates (alpha_x, alpha_y) in [-FOV/2, +FOV/2]^2 (radians), polar sky
angle theta_sky = sqrt(alpha_x^2 + alpha_y^2), and screen azimuth varphi = atan2(alpha_y, alpha_x).

In the observer's local static orthonormal tetrad, a ray at viewing angle psi_obs = theta_sky
relative to the inward radial axis has conserved impact parameter:

    b(theta_sky) = r_obs * sin(theta_sky) / sqrt(1 - 2M / r_obs)    ("local_angle" model)

or, in the asymptotic small-angle impact-parameter plane convention:

    b(theta_sky) = r_obs * theta_sky                               ("impact_parameter" model)

The 2D apparent impact-parameter coordinates on the observer's sky plane are:

    b_x = b * cos(varphi),    b_y = b * sin(varphi)

Each ray is traced through Schwarzschild spacetime using ``solve_photon_geodesic`` and
classified into exactly one category (`CAPTURED`, `ESCAPED`, or `UNRESOLVED`). Unresolved
rays are never converted into captured shadow pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    solve_photon_geodesic,
)
from blackhole_lab.numerics.lensing import (
    PhotonFate,
    classify_photon_fate,
)
from blackhole_lab.physics.schwarzschild import (
    _ensure_finite_scalar,
    metric_coefficient_f,
    reference_scales,
    shadow_angular_radius,
    validate_impact_parameter,
    validate_mass,
    validate_radius,
)

ObserverRayModel = Literal["local_angle", "impact_parameter"]

# Integer codes for 2D classification maps
CLASSIFICATION_CODE_CAPTURED: int = 1
CLASSIFICATION_CODE_ESCAPED: int = 0
CLASSIFICATION_CODE_UNRESOLVED: int = -1


@dataclass(frozen=True)
class CriticalBoundaryResult:
    """Result of numerically locating the critical impact parameter boundary b_crit.

    Attributes
    ----------
    M : float
        Black hole mass parameter in geometric units (G = c = 1) [L].
    r0 : float
        Initial photon radius used for the trajectory integrations [L].
    b_crit_numerical : float
        Midpoint estimate 0.5 * (b_low + b_high) of the critical boundary [L].
    b_crit_analytical : float
        Exact analytical reference b_crit = 3 * sqrt(3) * M [L].
    b_low : float
        Highest impact parameter verified as ``PhotonFate.CAPTURED`` [L].
    b_high : float
        Lowest impact parameter verified as ``PhotonFate.ESCAPED`` [L].
    bracket_width : float
        Final bracket uncertainty b_high - b_low [L].
    absolute_error : float
        Absolute discrepancy |b_crit_numerical - b_crit_analytical| [L].
    relative_error : float
        Relative discrepancy |b_crit_numerical - b_crit_analytical| / b_crit_analytical [1].
    iterations : int
        Number of bisection iterations executed.
    tolerance : float
        Requested bracket width tolerance [L].
    converged : bool
        True if bracket_width <= tolerance without being halted by unresolved trajectories.
    unresolved_encountered : bool
        True if bisection encountered an ``UNRESOLVED`` trajectory near the photon sphere
        (e.g., reaching phi_max) and safely stopped rather than misclassifying it.
    warnings : tuple[str, ...]
        Diagnostic notices regarding near-critical winding limits or bracket convergence.
    """

    M: float
    r0: float
    b_crit_numerical: float
    b_crit_analytical: float
    b_low: float
    b_high: float
    bracket_width: float
    absolute_error: float
    relative_error: float
    iterations: int
    tolerance: float
    converged: bool
    unresolved_encountered: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ShadowGridConfig:
    """Configuration for the 2D Schwarzschild black hole shadow imaging engine.

    Parameters
    ----------
    M : float, default 1.0
        Black hole mass parameter (M > 0) in geometric units (G = c = 1) [L].
    r_obs : float, default 30.0
        Static observer radial distance (must satisfy r_obs > 3M) [L].
    resolution : int | tuple[int, int], default 41
        Image-plane pixel resolution when ``width`` and ``height`` are not specified.
        If an int N is given, produces an N x N grid; if (ny, nx) is given, produces
        ny rows (height) by nx columns (width).
    width : int | None, default None
        Optional explicit horizontal image width (number of columns nx, in [3, 512]).
        Must be specified together with ``height``.
    height : int | None, default None
        Optional explicit vertical image height (number of rows ny, in [3, 512]).
        Must be specified together with ``width``.
    fov_rad : float | None, default None
        Full angular field of view of the image plane in radians [1].
        If None, defaults to 3.2 * shadow_angular_radius(r_obs, M).
    ray_model : ObserverRayModel, default "local_angle"
        Observer camera mapping convention ("local_angle" or "impact_parameter").
    phi_max : float, default 6 * pi
        Maximum azimuthal integration interval for each ray [rad].
    capture_radius_factor : float, default 1.001
        Capture boundary multiple of r_s = 2M (must be >= 1.0).
    outer_radius : float | None, default None
        Outer radial truncation boundary for escaping rays [L].
    rtol : float, default 1e-8
        Relative ODE solver tolerance.
    atol : float, default 1e-10
        Absolute ODE solver tolerance.
    method : str, default "DOP853"
        ``scipy.integrate.solve_ivp`` integration method.
    max_step : float, default 0.05
        Maximum angular step size dphi [rad].
    deduplicate_symmetric_rays : bool, default True
        When True, exploits the exact spherical symmetry of Schwarzschild spacetime by
        integrating each unique positive impact parameter b on the 2D grid once and
        mapping the exact solver fate to all symmetric pixels sharing that b.
    """

    M: float = 1.0
    r_obs: float = 30.0
    resolution: int | tuple[int, int] = 41
    width: int | None = None
    height: int | None = None
    fov_rad: float | None = None
    ray_model: ObserverRayModel = "local_angle"
    phi_max: float = 6.0 * math.pi
    capture_radius_factor: float = 1.001
    outer_radius: float | None = None
    rtol: float = 1e-8
    atol: float = 1e-10
    method: str = "DOP853"
    max_step: float = 0.05
    deduplicate_symmetric_rays: bool = True

    def __post_init__(self) -> None:
        scales = reference_scales(self.M)
        r_obs_val = validate_radius(self.r_obs, scales.M, require_exterior=True)
        if r_obs_val <= scales.r_ph:
            raise ValueError(
                f"Observer distance r_obs = {r_obs_val} must be strictly outside the "
                f"photon sphere r_ph = 3M = {scales.r_ph}."
            )

        ny, nx = self.grid_shape
        if not (3 <= ny <= 512 and 3 <= nx <= 512):
            raise ValueError(
                f"Grid dimensions (height={ny}, width={nx}) must be in [3, 512]."
            )

        if self.ray_model not in ("local_angle", "impact_parameter"):
            raise ValueError(
                "ray_model must be 'local_angle' or 'impact_parameter'; "
                f"got {self.ray_model!r}."
            )

        fov = self.effective_fov_rad
        if fov <= 0.0:
            raise ValueError(f"fov_rad must be strictly positive; got {fov}.")

        max_corner_theta = 0.5 * math.sqrt(2.0) * fov
        f_obs = metric_coefficient_f(r_obs_val, scales.M, require_exterior=True)
        b_max_allowed = r_obs_val / math.sqrt(f_obs)

        if self.ray_model == "local_angle":
            if max_corner_theta >= 0.5 * math.pi:
                raise ValueError(
                    f"fov_rad = {fov:.4f} rad is too large for 'local_angle' camera model: "
                    f"corner sky angle {max_corner_theta:.4f} rad must be < pi/2 "
                    f"({0.5 * math.pi:.4f} rad) so rays are directed into the forward hemisphere."
                )
        else:
            b_corner = r_obs_val * max_corner_theta
            if b_corner >= b_max_allowed:
                raise ValueError(
                    f"fov_rad = {fov:.4f} rad is too large at r_obs = {r_obs_val}: corner "
                    f"impact parameter b = {b_corner:.4f} exceeds the local turning-point "
                    f"limit b_max(r_obs) = {b_max_allowed:.4f}."
                )

        # Validate numerical parameters by constructing a probe GeodesicConfig at b_crit
        _ = GeodesicConfig(
            M=scales.M,
            r0=r_obs_val,
            b=scales.b_crit,
            direction="ingoing",
            phi0=0.0,
            phi_max=self.phi_max,
            capture_radius_factor=self.capture_radius_factor,
            outer_radius=self.outer_radius,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            max_step=self.max_step,
        )

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return (ny, nx) = (height, width) pixel dimensions of the 2D image plane."""
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "Both width and height must be specified together, or leave both as None "
                "to use resolution."
            )
        if self.width is not None and self.height is not None:
            if (
                isinstance(self.width, bool)
                or isinstance(self.height, bool)
                or not isinstance(self.width, int)
                or not isinstance(self.height, int)
            ):
                raise TypeError("width and height must be integers.")
            return (int(self.height), int(self.width))

        if isinstance(self.resolution, bool):
            raise TypeError("resolution must be an int or tuple[int, int], not bool.")
        if isinstance(self.resolution, int):
            return (self.resolution, self.resolution)
        if (
            isinstance(self.resolution, tuple)
            and len(self.resolution) == 2
            and all(
                isinstance(d, int) and not isinstance(d, bool) for d in self.resolution
            )
        ):
            return (int(self.resolution[0]), int(self.resolution[1]))
        raise TypeError(
            "resolution must be an int or a 2-tuple of ints (ny, nx); "
            f"got {self.resolution!r}."
        )

    @property
    def image_width(self) -> int:
        """Horizontal pixel count nx."""
        return self.grid_shape[1]

    @property
    def image_height(self) -> int:
        """Vertical pixel count ny."""
        return self.grid_shape[0]

    @property
    def effective_fov_rad(self) -> float:
        """Full angular field of view in radians."""
        if self.fov_rad is not None:
            return _ensure_finite_scalar(self.fov_rad, "fov_rad")
        psi_crit = shadow_angular_radius(self.r_obs, self.M)
        return min(3.2 * psi_crit, 0.95 * math.pi / math.sqrt(2.0))


@dataclass(frozen=True)
class ShadowRayTracingResult:
    """Structured output of a 2D Schwarzschild black hole shadow ray-tracing run.

    Attributes
    ----------
    config : ShadowGridConfig
        Configuration used for the ray-tracing simulation.
    alpha_x : NDArray[np.float64]
        1D horizontal angular sky coordinates of shape (nx,) [rad].
    alpha_y : NDArray[np.float64]
        1D vertical angular sky coordinates of shape (ny,) [rad].
    theta_sky_grid : NDArray[np.float64]
        2D angular separation from optical axis sqrt(alpha_x^2 + alpha_y^2) of shape (ny, nx) [rad].
    impact_parameter_grid : NDArray[np.float64]
        2D impact parameter b(i, j) assigned to each pixel of shape (ny, nx) [L].
    bx_grid : NDArray[np.float64]
        2D horizontal apparent impact-parameter coordinate b_x(i, j) of shape (ny, nx) [L].
    by_grid : NDArray[np.float64]
        2D vertical apparent impact-parameter coordinate b_y(i, j) of shape (ny, nx) [L].
    fate_grid : NDArray[np.object_]
        2D array of ``PhotonFate`` classifications of shape (ny, nx).
    classification_map : NDArray[np.int8]
        2D integer classification array of shape (ny, nx) with values strictly in
        ``{1 (CAPTURED), 0 (ESCAPED), -1 (UNRESOLVED)}``.
    capture_mask : NDArray[np.bool_]
        2D boolean shadow mask of shape (ny, nx): True ONLY where fate == CAPTURED.
        Unresolved pixels are strictly False in ``capture_mask``.
    escaped_mask : NDArray[np.bool_]
        2D boolean mask of shape (ny, nx): True ONLY where fate == ESCAPED.
    unresolved_mask : NDArray[np.bool_]
        2D boolean mask of shape (ny, nx): True ONLY where fate == UNRESOLVED.
    num_captured : int
        Total number of captured pixels.
    num_escaped : int
        Total number of escaped pixels.
    num_unresolved : int
        Total number of unresolved pixels.
    unique_rays_traced : int
        Number of ODE trajectory integrations executed.
    r_s : float
        Schwarzschild event-horizon radius r_s = 2M (physical: 2GM/c^2) [L].
    r_ph : float
        Photon-sphere radius r_ph = 3M (physical: 3GM/c^2) [L].
    b_shadow : float
        Apparent shadow impact-parameter radius b_shadow = 3*sqrt(3)*M
        (physical: 3*sqrt(3) GM/c^2) [L].
    b_crit_analytical : float
        Alias for ``b_shadow`` = 3 * sqrt(3) * M [L].
    shadow_angular_radius_analytical : float
        Exact local-observer shadow angular radius psi_crit [rad].
    estimated_b_crit_from_grid : float | None
        Midpoint between maximum captured b and minimum escaped b on the grid [L].
    estimated_shadow_angular_radius : float | None
        Midpoint between maximum captured theta_sky and minimum escaped theta_sky [rad].
    warnings : tuple[str, ...]
        Diagnostic notices if any pixels remained unresolved or failed integration.
    """

    config: ShadowGridConfig
    alpha_x: NDArray[np.float64]
    alpha_y: NDArray[np.float64]
    theta_sky_grid: NDArray[np.float64]
    impact_parameter_grid: NDArray[np.float64]
    bx_grid: NDArray[np.float64]
    by_grid: NDArray[np.float64]
    fate_grid: NDArray[np.object_]
    classification_map: NDArray[np.int8]
    capture_mask: NDArray[np.bool_]
    escaped_mask: NDArray[np.bool_]
    unresolved_mask: NDArray[np.bool_]
    num_captured: int
    num_escaped: int
    num_unresolved: int
    unique_rays_traced: int
    r_s: float
    r_ph: float
    b_shadow: float
    b_crit_analytical: float
    shadow_angular_radius_analytical: float
    estimated_b_crit_from_grid: float | None
    estimated_shadow_angular_radius: float | None
    warnings: tuple[str, ...]


def sky_angle_to_impact_parameter(
    theta_sky: float | NDArray[np.float64],
    r_obs: float,
    M: float,
    ray_model: ObserverRayModel = "local_angle",
) -> NDArray[np.float64]:
    """Map observer sky angle theta_sky [rad] to Schwarzschild impact parameter b [L].

    Parameters
    ----------
    theta_sky : float | NDArray[np.float64]
        Angular separation from the optical axis in radians (>= 0).
    r_obs : float
        Static observer radial coordinate (r_obs > 3M) [L].
    M : float
        Black hole mass parameter (M > 0) [L].
    ray_model : ObserverRayModel, default "local_angle"
        ``"local_angle"`` uses b = r_obs * sin(theta_sky) / sqrt(1 - 2M / r_obs);
        ``"impact_parameter"`` uses b = r_obs * theta_sky.

    Returns
    -------
    NDArray[np.float64]
        Impact parameter array matching the shape of ``theta_sky``.
    """
    mass = validate_mass(M)
    r_val = validate_radius(r_obs, mass, require_exterior=True)
    if r_val <= 3.0 * mass:
        raise ValueError(
            f"r_obs = {r_val} must be strictly greater than r_ph = 3M = {3.0 * mass}."
        )
    if ray_model not in ("local_angle", "impact_parameter"):
        raise ValueError(
            f"ray_model must be 'local_angle' or 'impact_parameter'; got {ray_model!r}."
        )

    theta_arr = np.asarray(theta_sky, dtype=np.float64)
    if not np.all(np.isfinite(theta_arr)) or np.any(theta_arr < 0.0):
        raise ValueError("theta_sky values must be finite and non-negative.")

    f_obs = metric_coefficient_f(r_val, mass, require_exterior=True)
    b_max = r_val / math.sqrt(f_obs)

    if ray_model == "local_angle":
        if np.any(theta_arr >= 0.5 * math.pi):
            raise ValueError(
                "In 'local_angle' mode, theta_sky must be strictly less than pi/2."
            )
        return b_max * np.sin(theta_arr)

    b_arr = r_val * theta_arr
    if np.any(b_arr >= b_max):
        raise ValueError(
            f"Impact parameter exceeds local maximum b_max(r_obs) = {b_max:.6g}."
        )
    return b_arr


def _trace_single_impact_parameter(
    b_val: float,
    config: ShadowGridConfig,
) -> PhotonFate:
    """Trace a single ray with impact parameter b_val >= 0 and classify its fate.

    - For b_val == 0.0 (exact on-axis radial ray), integrates the radial null condition
      dr / dlambda = -E from r_obs to r_capture.
    - For b_val > 0.0, delegates directly to ``solve_photon_geodesic`` and
      ``classify_photon_fate``.
    """
    if b_val == 0.0:
        r_cap = config.capture_radius_factor * (2.0 * config.M)
        if config.r_obs > r_cap:
            return PhotonFate.CAPTURED
        return PhotonFate.UNRESOLVED

    geo_cfg = GeodesicConfig(
        M=config.M,
        r0=config.r_obs,
        b=b_val,
        direction="ingoing",
        phi0=0.0,
        phi_max=config.phi_max,
        capture_radius_factor=config.capture_radius_factor,
        outer_radius=config.outer_radius,
        rtol=config.rtol,
        atol=config.atol,
        method=config.method,
        max_step=config.max_step,
    )
    geo_res = solve_photon_geodesic(geo_cfg)
    return classify_photon_fate(geo_res)


def find_critical_impact_parameter(
    M: float = 1.0,
    r0: float = 30.0,
    *,
    b_low: float | None = None,
    b_high: float | None = None,
    tolerance: float = 1e-6,
    max_iter: int = 50,
    phi_max: float = 8.0 * math.pi,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> CriticalBoundaryResult:
    """Numerically locate the critical impact parameter boundary between capture and escape.

    Uses bracketed bisection with actual ``solve_photon_geodesic`` integrations.
    Never classifies an ``UNRESOLVED`` trajectory (e.g., one that reaches ``phi_max``
    while winding around the photon sphere) as ``CAPTURED`` or ``ESCAPED``.

    Parameters
    ----------
    M : float, default 1.0
        Black hole mass parameter (M > 0) [L].
    r0 : float, default 30.0
        Starting radial coordinate (must satisfy r0 > 3M) [L].
    b_low : float | None, optional
        Lower bracket bound (must be CAPTURED). Defaults to 3.5 * M.
    b_high : float | None, optional
        Upper bracket bound (must be ESCAPED). Defaults to 7.0 * M.
    tolerance : float, default 1e-6
        Target bracket width b_high - b_low [L].
    max_iter : int, default 50
        Maximum number of bisection iterations (>= 1).
    phi_max : float, default 8 * pi
        Maximum azimuthal integration angle [rad].
    rtol : float, default 1e-10
        Relative ODE solver tolerance.
    atol : float, default 1e-12
        Absolute ODE solver tolerance.
    method : str, default "DOP853"
        ``solve_ivp`` integration method.

    Returns
    -------
    CriticalBoundaryResult
        Numerical estimate, analytical reference 3*sqrt(3)*M, error metrics, and convergence info.
    """
    scales = reference_scales(M)
    r0_val = validate_radius(r0, scales.M, require_exterior=True)
    if r0_val <= scales.r_ph:
        raise ValueError(
            f"Starting radius r0 = {r0_val} must be strictly outside the photon sphere "
            f"r_ph = 3M = {scales.r_ph}."
        )

    tol_val = _ensure_finite_scalar(tolerance, "tolerance")
    if tol_val <= 0.0:
        raise ValueError(f"tolerance must be strictly positive; got {tol_val}.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")

    low = validate_impact_parameter(b_low if b_low is not None else 3.5 * scales.M)
    high = validate_impact_parameter(b_high if b_high is not None else 7.0 * scales.M)
    if high <= low:
        raise ValueError(
            f"Upper bound b_high ({high}) must be strictly greater than b_low ({low})."
        )

    def _eval_fate(b_test: float) -> PhotonFate:
        cfg = GeodesicConfig(
            M=scales.M,
            r0=r0_val,
            b=b_test,
            direction="ingoing",
            phi0=0.0,
            phi_max=phi_max,
            rtol=rtol,
            atol=atol,
            method=method,
            max_step=0.04,
        )
        return classify_photon_fate(solve_photon_geodesic(cfg))

    fate_low = _eval_fate(low)
    fate_high = _eval_fate(high)

    if fate_low != PhotonFate.CAPTURED:
        raise ValueError(
            f"Lower bracket b_low = {low} did not result in capture "
            f"(classified as {fate_low.value!r})."
        )
    if fate_high != PhotonFate.ESCAPED:
        raise ValueError(
            f"Upper bracket b_high = {high} did not result in escape "
            f"(classified as {fate_high.value!r})."
        )

    warnings_list: list[str] = []
    unresolved_hit = False
    iters = 0

    for step in range(1, max_iter + 1):
        if (high - low) <= tol_val:
            break
        iters = step
        mid = 0.5 * (low + high)
        fate_mid = _eval_fate(mid)

        if fate_mid == PhotonFate.CAPTURED:
            low = mid
        elif fate_mid == PhotonFate.ESCAPED:
            high = mid
        else:
            unresolved_hit = True
            warnings_list.append(
                f"Encountered an UNRESOLVED trajectory at b = {mid:.10g} during iteration "
                f"{step} (likely reached phi_max = {phi_max:.4g} rad near the photon sphere). "
                "Bisection stopped safely without misclassifying the unresolved ray."
            )
            break

    b_num = 0.5 * (low + high)
    width = high - low
    abs_err = abs(b_num - scales.b_crit)
    rel_err = abs_err / scales.b_crit
    converged = (width <= tol_val) and (not unresolved_hit)

    if not converged and not unresolved_hit:
        warnings_list.append(
            f"Bisection reached max_iter = {max_iter} with bracket width {width:.3e} > "
            f"requested tolerance {tol_val:.3e}."
        )

    return CriticalBoundaryResult(
        M=scales.M,
        r0=r0_val,
        b_crit_numerical=b_num,
        b_crit_analytical=scales.b_crit,
        b_low=low,
        b_high=high,
        bracket_width=width,
        absolute_error=abs_err,
        relative_error=rel_err,
        iterations=iters,
        tolerance=tol_val,
        converged=converged,
        unresolved_encountered=unresolved_hit,
        warnings=tuple(warnings_list),
    )


def trace_black_hole_shadow(config: ShadowGridConfig) -> ShadowRayTracingResult:
    """Trace a 2D grid of observer rays and compute the Schwarzschild black hole shadow mask.

    Every pixel is classified using numerical geodesic integration into ``CAPTURED``,
    ``ESCAPED``, or ``UNRESOLVED``. Unresolved pixels are strictly excluded from
    ``capture_mask`` and tracked separately in ``unresolved_mask``.

    Parameters
    ----------
    config : ShadowGridConfig
        Validated observer and grid configuration.

    Returns
    -------
    ShadowRayTracingResult
        Complete 2D grids, boolean masks, integer classification map, pixel counts,
        and analytical/numerical shadow scales.
    """
    if not isinstance(config, ShadowGridConfig):
        raise TypeError(
            f"config must be a ShadowGridConfig instance; got {type(config).__name__}."
        )

    scales = reference_scales(config.M)
    psi_crit_analytical = shadow_angular_radius(config.r_obs, scales.M)

    ny, nx = config.grid_shape
    half_fov = 0.5 * config.effective_fov_rad

    alpha_x = np.linspace(-half_fov, half_fov, nx, dtype=np.float64)
    alpha_y = np.linspace(-half_fov, half_fov, ny, dtype=np.float64)
    ax_grid, ay_grid = np.meshgrid(alpha_x, alpha_y, indexing="xy")

    theta_sky_grid = np.hypot(ax_grid, ay_grid)
    b_grid = sky_angle_to_impact_parameter(
        theta_sky_grid,
        r_obs=config.r_obs,
        M=scales.M,
        ray_model=config.ray_model,
    )

    # Compute 2D apparent impact-parameter coordinates (b_x, b_y) with safe zero handling
    bx_grid = np.zeros_like(b_grid, dtype=np.float64)
    by_grid = np.zeros_like(b_grid, dtype=np.float64)
    nonzero = theta_sky_grid > 0.0
    bx_grid[nonzero] = b_grid[nonzero] * (ax_grid[nonzero] / theta_sky_grid[nonzero])
    by_grid[nonzero] = b_grid[nonzero] * (ay_grid[nonzero] / theta_sky_grid[nonzero])

    fate_grid = np.empty((ny, nx), dtype=object)

    if config.deduplicate_symmetric_rays:
        b_rounded = np.round(b_grid, decimals=13)
        unique_b_vals, inverse_indices = np.unique(b_rounded, return_inverse=True)
        unique_fates = np.empty(unique_b_vals.size, dtype=object)
        for k, b_val in enumerate(unique_b_vals):
            unique_fates[k] = _trace_single_impact_parameter(float(b_val), config)
        fate_grid[:] = unique_fates[inverse_indices].reshape((ny, nx))
        rays_traced = int(unique_b_vals.size)
    else:
        for i in range(ny):
            for j in range(nx):
                fate_grid[i, j] = _trace_single_impact_parameter(
                    float(b_grid[i, j]), config
                )
        rays_traced = int(ny * nx)

    capture_mask = np.equal(fate_grid, PhotonFate.CAPTURED)
    escaped_mask = np.equal(fate_grid, PhotonFate.ESCAPED)
    unresolved_mask = np.equal(fate_grid, PhotonFate.UNRESOLVED)

    classification_map = np.full(
        (ny, nx), CLASSIFICATION_CODE_UNRESOLVED, dtype=np.int8
    )
    classification_map[escaped_mask] = CLASSIFICATION_CODE_ESCAPED
    classification_map[capture_mask] = CLASSIFICATION_CODE_CAPTURED

    num_cap = int(np.sum(capture_mask))
    num_esc = int(np.sum(escaped_mask))
    num_unres = int(np.sum(unresolved_mask))

    if num_cap > 0 and num_esc > 0:
        max_b_cap = float(np.max(b_grid[capture_mask]))
        min_b_esc = float(np.min(b_grid[escaped_mask]))
        est_b_crit: float | None = 0.5 * (max_b_cap + min_b_esc)

        max_th_cap = float(np.max(theta_sky_grid[capture_mask]))
        min_th_esc = float(np.min(theta_sky_grid[escaped_mask]))
        est_psi_crit: float | None = 0.5 * (max_th_cap + min_th_esc)
    else:
        est_b_crit = None
        est_psi_crit = None

    warnings_list: list[str] = []
    if num_unres > 0:
        warnings_list.append(
            f"{num_unres} pixel(s) remained UNRESOLVED (e.g., reached phi_max or solver "
            "limit) and were kept in unresolved_mask rather than converted into capture_mask."
        )
    if num_cap == 0:
        warnings_list.append(
            "No captured rays were detected on the grid; consider increasing resolution."
        )
    if num_esc == 0:
        warnings_list.append(
            "No escaped rays were detected on the grid; field of view is entirely inside the shadow."
        )

    return ShadowRayTracingResult(
        config=config,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        theta_sky_grid=theta_sky_grid,
        impact_parameter_grid=b_grid,
        bx_grid=bx_grid,
        by_grid=by_grid,
        fate_grid=fate_grid,
        classification_map=classification_map,
        capture_mask=capture_mask,
        escaped_mask=escaped_mask,
        unresolved_mask=unresolved_mask,
        num_captured=num_cap,
        num_escaped=num_esc,
        num_unresolved=num_unres,
        unique_rays_traced=rays_traced,
        r_s=scales.r_s,
        r_ph=scales.r_ph,
        b_shadow=scales.b_crit,
        b_crit_analytical=scales.b_crit,
        shadow_angular_radius_analytical=psi_crit_analytical,
        estimated_b_crit_from_grid=est_b_crit,
        estimated_shadow_angular_radius=est_psi_crit,
        warnings=tuple(warnings_list),
    )
