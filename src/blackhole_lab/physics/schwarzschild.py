"""Exact Schwarzschild spacetime relations in geometric units (G = c = 1).

Metric Convention
-----------------
We use Schwarzschild coordinates (t, r, theta, phi) with metric signature (-, +, +, +):

    ds^2 = -f(r) dt^2 + f(r)^(-1) dr^2 + r^2 (dtheta^2 + sin^2(theta) dphi^2)

where the lapse function is:

    f(r) = 1 - 2M / r

In geometric units (G = c = 1), mass M, radial coordinate r, and impact parameter b
all have dimensions of length.

Reference Scales
----------------
- Schwarzschild radius (event horizon): r_s = 2M
- Photon-sphere radius (unstable circular null orbit): r_ph = 3M
- Critical impact parameter: b_crit = 3 * sqrt(3) * M

Coordinate Singularity Note
---------------------------
At r = r_s = 2M, f(r) = 0 and the metric component g_rr = f(r)^(-1) diverges.
This is a coordinate singularity of the Schwarzschild chart (the event horizon),
not a curvature singularity. Functions in this module strictly reject non-finite
inputs, non-positive masses, and unphysical radii, and explicitly guard against
evaluating exterior-chart expressions at or inside the horizon r <= 2M unless
specifically documented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

NumericScalar = Union[int, float, np.number]

_SQRT3: float = math.sqrt(3.0)
_CRIT_FACTOR: float = 3.0 * _SQRT3


@dataclass(frozen=True)
class SchwarzschildScales:
    """Characteristic length scales of a Schwarzschild black hole in geometric units (G = c = 1).

    Attributes
    ----------
    M : float
        Black hole mass parameter (M > 0).
    r_s : float
        Schwarzschild event-horizon radius r_s = 2M.
    r_ph : float
        Photon-sphere radius r_ph = 3M.
    b_crit : float
        Critical photon impact parameter b_crit = 3 * sqrt(3) * M.
    """

    M: float
    r_s: float
    r_ph: float
    b_crit: float


def _ensure_finite_scalar(value: object, name: str) -> float:
    """Validate that an input is a real, finite scalar number (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real numeric scalar; got {type(value).__name__}.")
    float_val = float(value)
    if not math.isfinite(float_val):
        raise ValueError(f"{name} must be finite (not NaN or infinity); got {value}.")
    return float_val


def _ensure_finite_result(value: float, name: str, M: float) -> float:
    """Ensure a derived physical scale did not overflow to infinity."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Computed {name} = {value} is non-finite or underflowed for mass M = {M}. "
            "Choose a mass M within safe IEEE-754 floating-point bounds."
        )
    return value


def validate_mass(M: NumericScalar) -> float:
    """Validate the Schwarzschild black hole mass parameter M in geometric units (G = c = 1).

    Parameters
    ----------
    M : NumericScalar
        Black hole mass parameter (must be finite, strictly positive M > 0, and
        within safe floating-point bounds so 3*sqrt(3)*M does not overflow).

    Returns
    -------
    float
        Validated mass parameter as a standard Python float.

    Raises
    ------
    TypeError
        If M is not a real numeric scalar.
    ValueError
        If M is NaN, infinite, M <= 0, or large enough to overflow reference scales.
    """
    mass = _ensure_finite_scalar(M, "Mass M")
    if mass <= 0.0:
        raise ValueError(f"Mass M must be strictly positive (M > 0); got {mass}.")
    if not math.isfinite(_CRIT_FACTOR * mass):
        raise ValueError(
            f"Mass M = {mass} is too large: reference scale b_crit = 3*sqrt(3)*M overflows "
            "floating-point representation."
        )
    return mass


def validate_radius(
    r: NumericScalar,
    M: NumericScalar | None = None,
    *,
    require_exterior: bool = True,
) -> float:
    """Validate radial coordinate r in Schwarzschild coordinates.

    Parameters
    ----------
    r : NumericScalar
        Schwarzschild radial coordinate (must be finite and strictly positive, r > 0).
    M : NumericScalar | None, optional
        If provided, also checks r against the Schwarzschild horizon r_s = 2M.
    require_exterior : bool, default True
        When M is provided and ``require_exterior=True``, enforces r > 2M so that
        evaluations remain strictly in the static exterior Schwarzschild chart.
        Even when ``require_exterior=False``, the exact coordinate singularity r == 2M
        is rejected.

    Returns
    -------
    float
        Validated radius as a standard Python float.

    Raises
    ------
    TypeError
        If r (or M) is not a real numeric scalar.
    ValueError
        If r is non-finite, r <= 0, at the coordinate singularity r == 2M,
        or inside the horizon r <= 2M when ``require_exterior=True``.
    """
    radius = _ensure_finite_scalar(r, "Radius r")
    if radius <= 0.0:
        raise ValueError(f"Radius r must be strictly positive (r > 0); got {radius}.")

    if M is not None:
        r_s = schwarzschild_radius(M)
        if math.isclose(radius, r_s, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError(
                f"Radius r = {radius} coincides with the Schwarzschild coordinate "
                f"singularity r_s = 2M = {r_s}."
            )
        if require_exterior and radius < r_s:
            raise ValueError(
                f"Radius r = {radius} lies inside the event horizon r_s = 2M = {r_s}. "
                "Exterior Schwarzschild coordinates require r > 2M."
            )
    return radius


def validate_impact_parameter(b: NumericScalar) -> float:
    """Validate the photon impact parameter b = L / E in geometric units.

    Note
    ----
    In the phi-parameterized orbit equation, (du/dphi)^2 = 1/b^2 - u^2 + 2Mu^3,
    b = 0 corresponds to a purely radial geodesic (dphi = 0) for which phi cannot
    serve as the independent variable. Therefore, b must be strictly positive (b > 0).

    Parameters
    ----------
    b : NumericScalar
        Impact parameter in units of length (must be finite and strictly positive).

    Returns
    -------
    float
        Validated impact parameter.

    Raises
    ------
    TypeError
        If b is not a real numeric scalar.
    ValueError
        If b is non-finite or b <= 0.
    """
    impact_b = _ensure_finite_scalar(b, "Impact parameter b")
    if impact_b <= 0.0:
        raise ValueError(
            f"Impact parameter b must be strictly positive (b > 0) for phi-parameterized "
            f"orbits; got {impact_b}. Purely radial trajectories (b = 0) have dphi = 0 "
            "and cannot be parameterized by azimuthal angle phi."
        )
    return impact_b


def schwarzschild_radius(M: NumericScalar) -> float:
    """Compute the Schwarzschild event-horizon radius r_s = 2M in geometric units (G = c = 1).

    At r = r_s = 2M, the Schwarzschild metric function f(r) = 1 - 2M/r vanishes,
    marking the event horizon (coordinate singularity in Schwarzschild coordinates).

    Parameters
    ----------
    M : NumericScalar
        Black hole mass parameter (M > 0) in geometric units (G = c = 1).

    Returns
    -------
    float
        Schwarzschild radius r_s = 2M.
    """
    mass = validate_mass(M)
    return _ensure_finite_result(2.0 * mass, "r_s", mass)


def photon_sphere_radius(M: NumericScalar) -> float:
    """Compute the Schwarzschild photon-sphere radius r_ph = 3M in geometric units (G = c = 1).

    At r_ph = 3M (u_ph = 1 / (3M)), the null effective potential V_eff(r) = (1 - 2M/r) / r^2
    attains its unique unstable maximum V_eff,max = 1 / (27 M^2), permitting unstable
    circular photon orbits (d^2u/dphi^2 = -u + 3Mu^2 = 0).

    Parameters
    ----------
    M : NumericScalar
        Black hole mass parameter (M > 0) in geometric units (G = c = 1).

    Returns
    -------
    float
        Photon-sphere radius r_ph = 3M.
    """
    mass = validate_mass(M)
    return _ensure_finite_result(3.0 * mass, "r_ph", mass)


def critical_impact_parameter(M: NumericScalar) -> float:
    """Compute the critical impact parameter b_crit = 3 * sqrt(3) * M in geometric units (G = c = 1).

    Derived from the peak of the null effective potential at r_ph = 3M:
        b_crit = r_ph / sqrt(f(r_ph)) = 3M / sqrt(1 - 2/3) = 3 * sqrt(3) * M.

    For an ingoing photon originating outside the photon sphere (r_0 > 3M):
    - b < b_crit: the photon crosses the peak of the effective potential and is captured.
    - b = b_crit: the photon asymptotically spirals toward the unstable circular orbit at r = 3M.
    - b > b_crit: the photon encounters a radial turning point (periapsis r_p > 3M) and scatters.

    Parameters
    ----------
    M : NumericScalar
        Black hole mass parameter (M > 0) in geometric units (G = c = 1).

    Returns
    -------
    float
        Critical impact parameter b_crit = 3 * sqrt(3) * M.
    """
    mass = validate_mass(M)
    return _ensure_finite_result(_CRIT_FACTOR * mass, "b_crit", mass)


def reference_scales(M: NumericScalar) -> SchwarzschildScales:
    """Compute all three reference scales (r_s, r_ph, b_crit) for a Schwarzschild black hole.

    Reuses ``schwarzschild_radius``, ``photon_sphere_radius``, and
    ``critical_impact_parameter`` to return a validated, immutable bundle.

    Parameters
    ----------
    M : NumericScalar
        Black hole mass parameter (M > 0) in geometric units (G = c = 1).

    Returns
    -------
    SchwarzschildScales
        Dataclass containing M, r_s = 2M, r_ph = 3M, and b_crit = 3 * sqrt(3) * M.
    """
    mass = validate_mass(M)
    return SchwarzschildScales(
        M=mass,
        r_s=schwarzschild_radius(mass),
        r_ph=photon_sphere_radius(mass),
        b_crit=critical_impact_parameter(mass),
    )


def metric_coefficient_f(
    r: NumericScalar,
    M: NumericScalar,
    *,
    require_exterior: bool = True,
) -> float:
    """Compute the Schwarzschild metric coefficient f(r) = 1 - 2M / r.

    In Schwarzschild coordinates, g_tt = -f(r) and g_rr = f(r)^(-1).
    Evaluation at the coordinate singularity r = 2M is always forbidden; by default,
    evaluation is restricted to the exterior domain r > 2M.

    Parameters
    ----------
    r : NumericScalar
        Radial coordinate (must satisfy r > 2M when ``require_exterior=True``).
    M : NumericScalar
        Black hole mass parameter (M > 0).
    require_exterior : bool, default True
        If True, raises ValueError for r <= 2M. If False, still raises ValueError
        at the coordinate singularity r == 2M and for r <= 0.

    Returns
    -------
    float
        Value of f(r) = 1 - 2M / r.
    """
    mass = validate_mass(M)
    radius = validate_radius(r, mass, require_exterior=require_exterior)
    return 1.0 - (2.0 * mass) / radius


def effective_potential_u(u: NumericScalar, M: NumericScalar) -> float:
    """Compute the null geodesic effective potential V_eff(u) = u^2 * (1 - 2Mu).

    In terms of inverse radius u = 1 / r, the first-integral equation is:
        (du / dphi)^2 = 1 / b^2 - V_eff(u)
    where V_eff(u) = u^2 - 2 M u^3.

    Parameters
    ----------
    u : NumericScalar
        Inverse radial coordinate u = 1 / r (must be finite and strictly positive).
    M : NumericScalar
        Black hole mass parameter (M > 0).

    Returns
    -------
    float
        Effective potential V_eff(u) = u^2 - 2 M u^3.
    """
    mass = validate_mass(M)
    u_val = _ensure_finite_scalar(u, "Inverse radius u")
    if u_val <= 0.0:
        raise ValueError(f"Inverse radius u must be strictly positive (u > 0); got {u_val}.")
    return (u_val * u_val) * (1.0 - 2.0 * mass * u_val)


def weak_field_deflection_angle(M: NumericScalar, b: NumericScalar) -> float:
    """Compute the leading-order weak-field gravitational deflection angle alpha = 4M / b.

    Assumes geometric units (G = c = 1) and the asymptotic weak-field regime b >> b_crit.
    The returned angle is in radians (dimensionless).

    Parameters
    ----------
    M : NumericScalar
        Black hole mass parameter (M > 0) [L].
    b : NumericScalar
        Photon impact parameter b = L / E (b > 0) [L].

    Returns
    -------
    float
        Leading-order Einstein deflection angle 4M / b [rad].
    """
    mass = validate_mass(M)
    b_val = validate_impact_parameter(b)
    alpha = (4.0 * mass) / b_val
    if not math.isfinite(alpha):
        raise ValueError(f"Weak-field deflection angle overflowed for M={mass}, b={b_val}.")
    return alpha


def second_order_weak_field_deflection(M: NumericScalar, b: NumericScalar) -> float:
    """Compute the second-order post-Newtonian deflection angle 4M/b + 15*pi*M^2 / (4*b^2).

    Useful for verifying that the numerical residual (alpha_num - 4M/b) scales as O((M/b)^2)
    in the weak-field regime.

    Parameters
    ----------
    M : NumericScalar
        Black hole mass parameter (M > 0) [L].
    b : NumericScalar
        Photon impact parameter (b > 0) [L].

    Returns
    -------
    float
        Second-order weak-field deflection angle [rad].
    """
    mass = validate_mass(M)
    b_val = validate_impact_parameter(b)
    ratio = mass / b_val
    alpha = 4.0 * ratio + 0.25 * 15.0 * math.pi * (ratio * ratio)
    if not math.isfinite(alpha):
        raise ValueError(f"Second-order deflection angle overflowed for M={mass}, b={b_val}.")
    return alpha
