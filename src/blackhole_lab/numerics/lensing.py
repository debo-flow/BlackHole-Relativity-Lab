"""Gravitational light deflection engine for equatorial Schwarzschild photon trajectories.

Deflection Angle Convention
---------------------------
For an incoming photon that scatters at periapsis r_p > r_ph = 3M and escapes toward
spatial infinity, let Delta_phi be the total cumulative azimuthal angle swept along the
orbit. In flat Minkowski spacetime (M = 0), a straight line extending from r = infinity
through periapsis r = b back to r = infinity sweeps an angle Delta_phi_flat = pi.

We define the net gravitational deflection angle alpha (in radians) as the positive
excess angular sweep relative to a straight line:

    alpha = Delta_phi - pi

For trajectories that loop around the black hole n times (winding_number = n >= 1),
Delta_phi > (2n + 1) * pi and alpha > 2n * pi. Angular unwrapping is strictly enforced
so multiples of 2*pi are never lost.

Accounting for Finite Starting and Ending Radii
-----------------------------------------------
Numerical ODE integrations start at a finite initial radius r_0 (u_0 = 1 / r_0) and
terminate at a finite outer boundary radius r_end (u_end = 1 / r_end), sweeping a finite-
domain angle:

    Delta_phi_finite(r_0, r_end) = phi_end - phi_0

Because the trajectory is truncated at finite radii r_0 and r_end rather than extending
to r = infinity (u = 0), Delta_phi_finite omits the far-field angular sweep along the
incoming leg (u in [0, u_0]) and outgoing leg (u in [0, u_end]). In fact, even in flat
space (M = 0), a straight line between r_0 and r_end sweeps only:

    Delta_phi_flat(r_0, r_end) = pi - arcsin(b / r_0) - arcsin(b / r_end)

To handle finite radii rigorously without lookup tables or hardcoded adjustments, this
module computes and reports three distinct, clearly documented quantities:

1. ``finite_radius_deflection``:
       alpha_finite_raw = Delta_phi_finite(r_0, r_end) - pi
   The uncorrected coordinate angular sweep minus pi.

2. ``euclidean_chord_deflection``:
       alpha_chord = Delta_phi_finite(r_0, r_end) - (pi - arcsin(b * u_0) - arcsin(b * u_end))
   The finite-radius deflection measured relative to a flat-space chord between r_0 and r_end.

3. ``numerical_deflection`` (with ``asymptotic_tail_correction``):
   Along the far-field segments [0, u_0] and [0, u_end], the exact Schwarzschild first
   integral gives:
       delta_phi_tail(u_b; M, b) = integral_{0}^{u_b} du / sqrt(1 / b^2 - u^2 + 2 M u^3)
   Evaluating delta_phi_tail(u_0) and delta_phi_tail(u_end) via controlled adaptive
   Gauss-Kronrod quadrature (``scipy.integrate.quad``) reconstructs the full asymptotic
   deflection angle:
       alpha_asymptotic = Delta_phi_finite + delta_phi_tail(u_0) + delta_phi_tail(u_end) - pi
   When ``apply_asymptotic_correction=True`` (the default), ``numerical_deflection``
   reports ``alpha_asymptotic`` so comparisons against the asymptotic weak-field prediction
   alpha_weak = 4M / b are independent of the finite domain truncation radii r_0 and r_end.
   When ``apply_asymptotic_correction=False``, ``numerical_deflection`` reports
   ``finite_radius_deflection`` directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from blackhole_lab.numerics.geodesics import (
    GeodesicConfig,
    GeodesicResult,
    first_integral_radicand,
    solve_photon_geodesic,
)
from blackhole_lab.physics.schwarzschild import (
    _ensure_finite_scalar,
    reference_scales,
    second_order_weak_field_deflection,
    validate_impact_parameter,
    validate_mass,
    weak_field_deflection_angle,
)

# Thresholds for weak-field suitability warnings
WEAK_FIELD_MIN_B_OVER_BCRIT: float = 3.0
NEAR_CRITICAL_B_OVER_BCRIT: float = 1.15


class PhotonFate(str, Enum):
    """Asymptotic or domain-resolved physical fate of a simulated photon trajectory."""

    ESCAPED = "escaped"
    CAPTURED = "captured"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class DeflectionResult:
    """Structured result of a gravitational light deflection calculation.

    Attributes
    ----------
    fate : PhotonFate
        Classified outcome: ``ESCAPED``, ``CAPTURED``, or ``UNRESOLVED``.
    M : float
        Black hole mass parameter in geometric units (G = c = 1) [L].
    b : float
        Impact parameter b = L / E in geometric units [L].
    b_crit : float
        Critical impact parameter 3 * sqrt(3) * M [L].
    b_over_b_crit : float
        Dimensionless criticality ratio b / b_crit [1].
    r0 : float
        Initial starting radius [L].
    r_end : float
        Final radius reached at termination [L].
    periapsis_radius : float | None
        Minimum radial coordinate r_min reached along the trajectory (for ESCAPED) [L].
    delta_phi_finite : float
        Total unwrapped angular sweep phi_end - phi_0 between r0 and r_end [rad].
    winding_number : int
        Number of complete 2*pi loops around the black hole beyond pi [1].
    finite_radius_deflection : float | None
        Raw finite-domain deflection angle Delta_phi_finite - pi (None if not ESCAPED) [rad].
    euclidean_chord_deflection : float | None
        Finite-domain deflection adjusted for flat-space finite-radius chord geometry [rad].
    asymptotic_tail_correction : float | None
        Curved-spacetime quadrature tail correction delta_phi_tail(u0) + delta_phi_tail(u_end) [rad].
    numerical_deflection : float | None
        Primary numerical deflection angle [rad] (asymptotic-corrected when
        ``asymptotic_correction_applied=True``, or raw finite-radius otherwise;
        None if fate is not ESCAPED).
    asymptotic_correction_applied : bool
        True if ``numerical_deflection`` includes ``asymptotic_tail_correction``.
    analytical_weak_field : float
        Leading-order weak-field deflection prediction alpha_weak = 4M / b [rad].
    analytical_post_newtonian_2 : float
        Second-order weak-field reference 4M / b + 15 * pi * M^2 / (4 * b^2) [rad].
    absolute_error : float | None
        Absolute difference |numerical_deflection - analytical_weak_field| [rad]
        (None if fate is not ESCAPED).
    relative_error : float | None
        Relative difference |numerical_deflection - analytical_weak_field| / analytical_weak_field
        (None if fate is not ESCAPED).
    convergence_indicator : float | None
        Estimated numerical angular discrepancy [rad] from tolerance refinement or
        first-integral conservation residual (None if fate is not ESCAPED).
    is_weak_field_valid : bool
        True if b >= 3 * b_crit where the leading-order 4M/b approximation is meaningful;
        False in the strong-field / near-critical regime.
    warnings : tuple[str, ...]
        Human-readable warnings regarding weak-field breakdown, near-critical orbits,
        finite-radius truncation, or unresolved/captured trajectories.
    geodesic_result : GeodesicResult
        Underlying numerical trajectory result.
    """

    fate: PhotonFate
    M: float
    b: float
    b_crit: float
    b_over_b_crit: float
    r0: float
    r_end: float
    periapsis_radius: float | None
    delta_phi_finite: float
    winding_number: int
    finite_radius_deflection: float | None
    euclidean_chord_deflection: float | None
    asymptotic_tail_correction: float | None
    numerical_deflection: float | None
    asymptotic_correction_applied: bool
    analytical_weak_field: float
    analytical_post_newtonian_2: float
    absolute_error: float | None
    relative_error: float | None
    convergence_indicator: float | None
    is_weak_field_valid: bool
    warnings: tuple[str, ...]
    geodesic_result: GeodesicResult


def unwrap_azimuthal_angle(
    phi: NDArray[np.float64] | Sequence[float],
) -> NDArray[np.float64]:
    """Unwrap an azimuthal angle sequence to preserve cumulative multi-loop rotations.

    Handles angle arrays that may have been wrapped into [-pi, pi) or [0, 2*pi)
    (for example via ``atan2(y, x)``) and verifies finite, monotonically advancing
    angles for positive angular momentum (L > 0) trajectories.

    Parameters
    ----------
    phi : array-like of float
        Sequence of azimuthal angles in radians.

    Returns
    -------
    NDArray[np.float64]
        Continuous, unwrapped 1D array of azimuthal angles in radians.

    Raises
    ------
    ValueError
        If ``phi`` is empty, not 1D, contains non-finite values, or reverses direction.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.ndim != 1 or phi_arr.size < 2:
        raise ValueError(
            f"phi must be a 1D array with at least 2 points; got shape {phi_arr.shape}."
        )
    if not np.all(np.isfinite(phi_arr)):
        raise ValueError("phi array contains non-finite (NaN or infinite) values.")

    unwrapped = np.unwrap(phi_arr)
    diffs = np.diff(unwrapped)
    if np.any(diffs < -1e-12):
        raise ValueError(
            "Unwrapped azimuthal angle is not monotonically non-decreasing; "
            "equatorial geodesics with b > 0 must have dphi > 0 everywhere."
        )
    return unwrapped


def classify_photon_fate(result: GeodesicResult) -> PhotonFate:
    """Classify a solved photon geodesic as ESCAPED, CAPTURED, or UNRESOLVED.

    Rules
    -----
    - ``SOLVER_FAILURE`` or ``INTEGRATION_LIMIT`` (phi_max reached) -> ``UNRESOLVED``.
    - ``CAPTURE_BOUNDARY`` (r <= r_capture reached) -> ``CAPTURED``.
    - ``COMPLETED`` (r >= r_outer reached):
      Verified to have started ingoing, remained strictly outside the photon sphere
      (r_min > 3M), and exited outgoing (du/dphi_end < 0) -> ``ESCAPED``.
      If the trajectory was initialized as purely outgoing (never scattered through
      a periapsis), it is classified as ``UNRESOLVED`` for full scattering deflection.

    Parameters
    ----------
    result : GeodesicResult
        Output from ``solve_photon_geodesic``.

    Returns
    -------
    PhotonFate
        ``PhotonFate.ESCAPED``, ``PhotonFate.CAPTURED``, or ``PhotonFate.UNRESOLVED``.
    """
    if not isinstance(result, GeodesicResult):
        raise TypeError(
            f"result must be a GeodesicResult instance; got {type(result).__name__}."
        )

    if result.is_unfinished:
        return PhotonFate.UNRESOLVED

    if result.is_captured:
        return PhotonFate.CAPTURED

    if result.reached_outer_boundary:
        if (
            result.config.direction == "ingoing"
            and result.du_dphi.size >= 2
            and float(result.du_dphi[-1]) < 0.0
            and np.all(np.isfinite(result.r))
            and float(np.min(result.r)) > result.r_ph
        ):
            return PhotonFate.ESCAPED
        return PhotonFate.UNRESOLVED

    return PhotonFate.UNRESOLVED


def asymptotic_tail_angle(u_bound: float, M: float, b: float) -> float:
    """Compute the exact angular sweep delta_phi_tail from r = infinity (u = 0) to u_bound.

    Evaluates the far-field integral:
        delta_phi_tail(u_bound; M, b) = int_{0}^{u_bound} du / sqrt(1 / b^2 - u^2 + 2 M u^3)
    using adaptive Gauss-Kronrod quadrature (``scipy.integrate.quad``).

    Parameters
    ----------
    u_bound : float
        Upper inverse-radius integration limit u_bound = 1 / r_bound [L^-1].
        Must lie strictly in the classically allowed region 1/b^2 - u^2 + 2Mu^3 > 0.
    M : float
        Black hole mass parameter (M > 0) [L].
    b : float
        Photon impact parameter (b > 0) [L].

    Returns
    -------
    float
        Angular sweep delta_phi_tail in radians [1].

    Raises
    ------
    ValueError
        If u_bound <= 0 or lies at/beyond the radial turning point where radicand <= 0.
    """
    mass = validate_mass(M)
    b_val = validate_impact_parameter(b)
    u_max = _ensure_finite_scalar(u_bound, "Inverse radius bound u_bound")
    if u_max <= 0.0:
        raise ValueError(f"u_bound must be strictly positive; got {u_max}.")

    rad_bound = first_integral_radicand(u_max, mass, b_val)
    if rad_bound <= 0.0:
        raise ValueError(
            f"Cannot compute asymptotic tail angle at u_bound = {u_max}: "
            f"first-integral radicand = {rad_bound:.6e} <= 0 (at or inside turning point)."
        )

    def integrand(u_val: float) -> float:
        rad = (1.0 / (b_val * b_val)) - (u_val * u_val) + (2.0 * mass * (u_val**3))
        return 1.0 / math.sqrt(rad)

    val, _ = quad(integrand, 0.0, u_max, epsabs=1e-13, epsrel=1e-12, limit=100)
    return float(val)


def _estimate_convergence_indicator(
    result: GeodesicResult,
    delta_phi_finite: float,
    *,
    verify_with_coarse_run: bool = True,
) -> float:
    """Compute a numerical convergence indicator (in radians) for an escaped trajectory.

    When ``verify_with_coarse_run=True``, performs a companion integration with 10x
    relaxed tolerances (clamped to safe bounds) and combines the empirical angular shift
    |Delta_phi_fine - Delta_phi_coarse| with the first-integral residual estimate.
    """
    diag = result.diagnostics
    b_val = result.config.b
    max_res = float(getattr(diag, "max_abs_residual", 0.0))
    residual_bound = (
        (b_val * b_val) * max_res * delta_phi_finite if math.isfinite(max_res) else 0.0
    )

    if not verify_with_coarse_run:
        return float(residual_bound)

    coarse_rtol = min(1e-2, max(1e-14, 10.0 * result.config.rtol))
    coarse_atol = min(1e-2, max(1e-15, 10.0 * result.config.atol))
    if math.isclose(coarse_rtol, result.config.rtol, rel_tol=1e-12):
        return float(residual_bound)

    coarse_cfg = replace(
        result.config,
        rtol=coarse_rtol,
        atol=coarse_atol,
    )
    coarse_res = solve_photon_geodesic(coarse_cfg)
    if classify_photon_fate(coarse_res) != PhotonFate.ESCAPED:
        return float(residual_bound)

    phi_coarse = unwrap_azimuthal_angle(coarse_res.phi)
    delta_phi_coarse = float(phi_coarse[-1] - phi_coarse[0])
    step_diff = abs(delta_phi_finite - delta_phi_coarse)
    return float(max(step_diff, residual_bound))


def calculate_deflection_angle(
    source: GeodesicResult | GeodesicConfig,
    *,
    apply_asymptotic_correction: bool = True,
    verify_convergence: bool = True,
) -> DeflectionResult:
    """Calculate gravitational light deflection and weak-field comparison for a trajectory.

    Parameters
    ----------
    source : GeodesicResult | GeodesicConfig
        Either an already-integrated ``GeodesicResult`` or a ``GeodesicConfig`` to
        be integrated via ``solve_photon_geodesic``.
    apply_asymptotic_correction : bool, default True
        If True, ``numerical_deflection`` includes the controlled far-field quadrature
        tail correction ``asymptotic_tail_correction`` from (r_0, r_end) to infinity.
        If False, ``numerical_deflection`` reports ``finite_radius_deflection``
        (Delta_phi_finite - pi) directly.
    verify_convergence : bool, default True
        If True, estimates ``convergence_indicator`` by comparing against a companion
        coarse-tolerance integration in addition to the first-integral residual bound.

    Returns
    -------
    DeflectionResult
        Structured deflection analysis, analytical references, errors, and warnings.
    """
    if isinstance(source, GeodesicConfig):
        result = solve_photon_geodesic(source)
    elif isinstance(source, GeodesicResult):
        result = source
    else:
        raise TypeError(
            "source must be a GeodesicResult or GeodesicConfig; "
            f"got {type(source).__name__}."
        )

    scales = reference_scales(result.config.M)
    b_val = result.config.b
    b_ratio = b_val / scales.b_crit

    alpha_weak = weak_field_deflection_angle(scales.M, b_val)
    alpha_pn2 = second_order_weak_field_deflection(scales.M, b_val)

    fate = classify_photon_fate(result)
    is_weak_valid = b_ratio >= WEAK_FIELD_MIN_B_OVER_BCRIT

    warnings_list: list[str] = []
    if not is_weak_valid:
        if b_ratio <= 1.0:
            warnings_list.append(
                f"Subcritical or critical impact parameter (b / b_crit = {b_ratio:.4f} <= 1). "
                "The weak-field formula alpha = 4M/b is physically inapplicable because "
                "incoming photons are captured rather than scattered."
            )
        elif b_ratio < NEAR_CRITICAL_B_OVER_BCRIT:
            warnings_list.append(
                f"Near-critical strong-field regime (b / b_crit = {b_ratio:.4f}). "
                "The weak-field approximation alpha = 4M/b breaks down severely near the "
                "photon sphere (r_ph = 3M), where true deflection diverges logarithmically."
            )
        else:
            warnings_list.append(
                f"Moderate-to-strong gravitational field (b / b_crit = {b_ratio:.4f} < "
                f"{WEAK_FIELD_MIN_B_OVER_BCRIT}). Higher-order post-Newtonian and strong-field "
                "terms cause significant deviation from the leading-order 4M/b approximation."
            )

    if result.phi.size >= 2 and np.all(np.isfinite(result.phi)):
        phi_unwrapped = unwrap_azimuthal_angle(result.phi)
        delta_phi_finite = float(phi_unwrapped[-1] - phi_unwrapped[0])
    else:
        delta_phi_finite = float("nan")

    winding = (
        int(max(0.0, delta_phi_finite - math.pi) // (2.0 * math.pi))
        if math.isfinite(delta_phi_finite)
        else 0
    )

    r0_val = float(result.config.r0)
    r_end_val = float(result.r[-1]) if result.r.size > 0 else float("nan")

    if fate != PhotonFate.ESCAPED:
        if fate == PhotonFate.CAPTURED:
            warnings_list.append(
                "Trajectory was captured by the black hole; scattering deflection angle "
                "is undefined (reported as None)."
            )
        else:
            warnings_list.append(
                f"Trajectory is unresolved (status={result.status.value}, "
                f"reason={result.termination_reason.value}); deflection angle is not "
                "computed for unfinished or non-scattering trajectories."
            )
        return DeflectionResult(
            fate=fate,
            M=scales.M,
            b=b_val,
            b_crit=scales.b_crit,
            b_over_b_crit=b_ratio,
            r0=r0_val,
            r_end=r_end_val,
            periapsis_radius=None,
            delta_phi_finite=delta_phi_finite,
            winding_number=winding,
            finite_radius_deflection=None,
            euclidean_chord_deflection=None,
            asymptotic_tail_correction=None,
            numerical_deflection=None,
            asymptotic_correction_applied=apply_asymptotic_correction,
            analytical_weak_field=alpha_weak,
            analytical_post_newtonian_2=alpha_pn2,
            absolute_error=None,
            relative_error=None,
            convergence_indicator=None,
            is_weak_field_valid=is_weak_valid,
            warnings=tuple(warnings_list),
            geodesic_result=result,
        )

    periapsis_r = float(np.min(result.r))
    finite_deflection = delta_phi_finite - math.pi

    u0 = float(result.u[0])
    u_end = float(result.u[-1])

    arg0 = min(1.0, max(-1.0, b_val * u0))
    arg_end = min(1.0, max(-1.0, b_val * u_end))
    chord_correction = math.asin(arg0) + math.asin(arg_end)
    euclidean_chord_def = finite_deflection + chord_correction

    tail_0 = asymptotic_tail_angle(u0, scales.M, b_val)
    tail_end = asymptotic_tail_angle(u_end, scales.M, b_val)
    tail_total = tail_0 + tail_end

    if apply_asymptotic_correction:
        num_deflection = finite_deflection + tail_total
    else:
        num_deflection = finite_deflection
        warnings_list.append(
            f"Reporting uncorrected finite-radius deflection between r0={r0_val:.4g} and "
            f"r_end={r_end_val:.4g} (omitted asymptotic tail angle is {tail_total:.4e} rad)."
        )

    abs_err = abs(num_deflection - alpha_weak)
    rel_err = abs_err / alpha_weak if alpha_weak > 0.0 else None

    conv_indicator = _estimate_convergence_indicator(
        result,
        delta_phi_finite,
        verify_with_coarse_run=verify_convergence,
    )

    return DeflectionResult(
        fate=fate,
        M=scales.M,
        b=b_val,
        b_crit=scales.b_crit,
        b_over_b_crit=b_ratio,
        r0=r0_val,
        r_end=r_end_val,
        periapsis_radius=periapsis_r,
        delta_phi_finite=delta_phi_finite,
        winding_number=winding,
        finite_radius_deflection=finite_deflection,
        euclidean_chord_deflection=euclidean_chord_def,
        asymptotic_tail_correction=tail_total,
        numerical_deflection=num_deflection,
        asymptotic_correction_applied=apply_asymptotic_correction,
        analytical_weak_field=alpha_weak,
        analytical_post_newtonian_2=alpha_pn2,
        absolute_error=abs_err,
        relative_error=rel_err,
        convergence_indicator=conv_indicator,
        is_weak_field_valid=is_weak_valid,
        warnings=tuple(warnings_list),
        geodesic_result=result,
    )
