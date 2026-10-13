# Numerical Methods & Solver Architecture

## 1. Conserved Quantities, Impact Parameter, and Units

Along an equatorial ($\theta = \pi/2$) null geodesic $x^\mu(\lambda)$ in Schwarzschild spacetime with geometric units ($G = c = 1$), stationarity and axisymmetry yield conserved energy $E$ and conserved azimuthal angular momentum $L$:
$$E = \left(1 - \frac{2M}{r}\right)\frac{dt}{d\lambda}, \qquad L = r^2\frac{d\phi}{d\lambda}$$

For a future-directed photon ($E \gt 0$) with $L \gt 0$, the **impact parameter** is:
$$b \equiv \frac{L}{E}$$

### Dimensional vs. Dimensionless Quantities
In geometric units ($G = c = 1$), lengths and masses share dimension $[L]$:
- **Dimensional Quantities:**
  - $M, r, r_0, r_s = 2M, r_{\mathrm{ph}} = 3M, b, b_{\mathrm{crit}} = 3\sqrt{3}M$ have dimension $[L]$.
  - $u = 1/r$, $v = du/d\phi$, and $d^2u/d\phi^2$ have dimension $[L^{-1}]$.
  - $V_{\mathrm{eff}}(u) = u^2(1 - 2Mu)$ and the first-integral radicand $R(u; M, b)$ have dimension $[L^{-2}]$.
- **Dimensionless Quantities ($[1]$):**
  - Azimuthal angles $\phi, \phi_0, \phi_{\max}$, and angular displacement $\Delta\phi = \phi - \phi_0$ ($\mathrm{rad}$).
  - Normalized radii $r / M$, $r / r_s = r / (2M)$, and compactness $M u = M / r$.
  - Normalized impact parameters $b / M$ and $b / b_{\mathrm{crit}} = b / (3\sqrt{3}M)$.
  - Local static orthonormal pitch-angle sine $\sin\psi_0 = \frac{b}{r_0}\sqrt{1 - \frac{2M}{r_0}} \in [0, 1]$.

---

## 2. Reduction to a First-Order ODE System

We solve the second-order equation $\frac{d^2u}{d\phi^2} = -u + 3Mu^2$ as an autonomous first-order initial-value problem for the state vector $\mathbf{y}(\phi) = [u(\phi), v(\phi)]^T$:
$$\frac{du}{d\phi} = v$$
$$\frac{dv}{d\phi} = -u + 3Mu^2$$

Why integrate the second-order system rather than the first-order equation $du/d\phi = \pm\sqrt{1/b^2 - u^2 + 2Mu^3}$ directly?
1. **Smooth Passage Through Radial Turning Points:** At a periapsis $r_p$ where $du/d\phi = 0$, the square-root formulation is non-Lipschitz and requires manual sign switching from $+\sqrt{\cdot}$ to $-\sqrt{\cdot}$. In contrast, the second-order system is completely smooth across turning points ($dv/d\phi = -u_p + 3Mu_p^2 \lt 0$).
2. **Independent Conservation Diagnostic:** Because $b$ enters only through the initial condition $v(\phi_0)$, the conserved quantity $1/b^2 = v^2 + u^2 - 2Mu^3$ serves as a genuine independent check of numerical integration error at all subsequent steps.

---

## 3. Initial Conditions for Finite Starting Radius $r_0$

Given a finite starting radius $r_0 \gt r_{\mathrm{capture}} \ge 2M$ ($u_0 = 1/r_0$) and impact parameter $b \gt 0$:
1. **Exact Curved-Spacetime Radicand:** Instead of using a flat-space parallel-ray approximation, we evaluate the exact Schwarzschild first-integral radicand at $r_0$:
   $$R(u_0; M, b) = \frac{1}{b^2} - u_0^2 + 2Mu_0^3 = \frac{1}{b^2}\left(1 - \sin^2\psi_0\right)$$
2. **Barrier Validation:** If $R(u_0; M, b) \lt 0$ (outside a $10^{-15}$ machine-epsilon tolerance at an exact turning point), $r_0$ lies inside the classically forbidden potential barrier for impact parameter $b$, and a `ValueError` is raised.
3. **Branch Selection:** Because $u = 1/r$ implies $\frac{du}{d\phi} = -\frac{1}{r^2}\frac{dr}{d\phi}$:
   - `"ingoing"` ($dr/d\phi \lt 0$) sets $v_0 = +\sqrt{R(u_0; M, b)}$.
   - `"outgoing"` ($dr/d\phi \gt 0$) sets $v_0 = -\sqrt{R(u_0; M, b)}$.

Both dimensional and dimensionless initial quantities are bundled in `PhotonInitialConditions` via `compute_initial_conditions(M, r0, b, phi0, direction)`.

---

## 4. Event Detection and Safe Unfinished-Trajectory Classification

We integrate from $\phi_0$ to $\phi_{\mathrm{max}}$ using `scipy.integrate.solve_ivp` (default `DOP853`) with three terminal events:
1. **Capture Boundary Event (`CAPTURE_BOUNDARY_REACHED`):** Triggers when $u(\phi) = 1 / r_{\mathrm{capture}}$ while ingoing.
2. **Outer Boundary Event (`OUTER_BOUNDARY_REACHED`):** Triggers when $u(\phi) = 1 / r_{\mathrm{outer}}$ while outgoing.
3. **Invalid Domain Guard (`INVALID_DOMAIN_ENTERED`):** Triggers if $u(\phi) \le 0$ or if any point crosses inside $r \lt 2M$.

Unfinished trajectories (`SolverStatus.INTEGRATION_LIMIT` when $\phi_{\max}$ is reached, or `SolverStatus.SOLVER_FAILURE` when `solve_ivp` fails) are **never** classified as captured or escaped: `result.is_captured` and `result.reached_outer_boundary` both evaluate to `False`, while `result.is_unfinished` evaluates to `True`.
