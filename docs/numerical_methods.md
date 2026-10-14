# Numerical Methods & Solver Architecture

## 1. Dimensional vs. Dimensionless Quantities ($G = c = 1$)

- **Dimensional Quantities:**
  - $M, r, r_0, r_s = 2M, r_{\mathrm{ph}} = 3M, b, b_{\mathrm{crit}} = 3\sqrt{3}M$ carry dimension $[L]$.
  - $u = 1/r$, $v = du/d\phi$, and $d^2u/d\phi^2$ carry dimension $[L^{-1}]$.
  - $V_{\mathrm{eff}}(u) = u^2(1 - 2Mu)$ and the first-integral radicand $R(u; M, b)$ carry dimension $[L^{-2}]$.
- **Dimensionless Quantities ($[1]$):**
  - Azimuthal angles $\phi, \phi_0, \phi_{\max}$, angular sweep $\Delta\phi$, and deflection angle $\alpha$ ($\mathrm{rad}$).
  - Normalized radii $r / M$, $r / r_s$, and compactness $M u = M / r$.
  - Normalized impact parameters $b / M$ and $b / b_{\mathrm{crit}}$.
  - Local static orthonormal pitch-angle sine $\sin\psi_0 = \frac{b}{r_0}\sqrt{1 - \frac{2M}{r_0}} \in [0, 1]$.

---

## 2. First-Order ODE System and Smooth Turning-Point Integration

We solve the second-order Binet equation $\frac{d^2u}{d\phi^2} = -u + 3Mu^2$ as an autonomous first-order initial-value problem for the state vector $\mathbf{y}(\phi) = [u(\phi), v(\phi)]^T$:
$$\frac{du}{d\phi} = v, \qquad \frac{dv}{d\phi} = -u + 3Mu^2$$

Integrating this second-order system rather than $du/d\phi = \pm\sqrt{1/b^2 - u^2 + 2Mu^3}$ directly provides two major numerical advantages:
1. **Regularity at Periapsis:** At a radial turning point $r_p \gt 3M$ where $v = du/d\phi = 0$, $\sqrt{1/b^2 - u^2 + 2Mu^3}$ is non-Lipschitz and requires manual branch switching. In the second-order system, $dv/d\phi = -u_p + 3Mu_p^2 \lt 0$ is smooth, allowing high-order Runge-Kutta solvers (`DOP853`, `RK45`) to pass through periapsis without loss of order.
2. **Independent Conservation Check:** Because $b$ enters only through the initial condition $v(\phi_0)$, the algebraic invariant $1/b^2 = v^2 + u^2 - 2Mu^3$ provides an independent pointwise accuracy check at every step.

---

## 3. Finite-Radius Initial Conditions and Event Detection

Given $r_0 \gt r_{\mathrm{capture}} \ge 2M$ ($u_0 = 1/r_0$) and $b \gt 0$:
1. We evaluate $R(u_0; M, b) = 1/b^2 - u_0^2 + 2Mu_0^3$. If $R(u_0; M, b) \lt 0$, a `ValueError` is raised.
2. Incoming rays ($dr/d\phi \lt 0$) take $v_0 = +\sqrt{R(u_0; M, b)}$; outgoing rays ($dr/d\phi \gt 0$) take $v_0 = -\sqrt{R(u_0; M, b)}$.
3. Terminal events detect:
   - **Capture Boundary (`CAPTURE_BOUNDARY_REACHED`):** $u(\phi) = 1/r_{\mathrm{capture}}$ with $du/d\phi \gt 0$.
   - **Outer Boundary (`OUTER_BOUNDARY_REACHED`):** $u(\phi) = 1/r_{\mathrm{outer}}$ with $du/d\phi \lt 0$.
   - **Invalid Domain (`INVALID_DOMAIN_ENTERED`):** $u(\phi) \le 0$ or $r(\phi) \lt 2M$.

---

## 4. Deflection Angle Calculation, Tail Quadrature, and Convergence Estimation

For trajectories classified as `PhotonFate.ESCAPED` (started incoming, $r_{\min} \gt 3M$, reached $r_{\mathrm{outer}}$ outgoing):
1. **Angle Unwrapping:** `unwrap_azimuthal_angle` applies `numpy.unwrap` and verifies monotonicity ($d\phi \ge 0$) so multi-loop paths ($\Delta\phi \gt 2\pi$) retain all full $2\pi$ turns.
2. **Controlled Asymptotic Tail Quadrature:** `asymptotic_tail_angle` uses `scipy.integrate.quad` (`epsabs=1e-13`, `epsrel=1e-12`) to integrate $(1/b^2 - u^2 + 2Mu^3)^{-1/2}$ over $[0, u_0]$ and $[0, u_{\mathrm{end}}]$, removing finite-radius truncation bias when comparing against asymptotic predictions.
3. **Convergence Indicator:** `_estimate_convergence_indicator` combines the first-integral residual bound $b^2 \max\vert{}\mathcal{R}_i\vert{}\,\Delta\phi_{\mathrm{finite}}$ with the empirical angular shift $\vert{}\Delta\phi_{\mathrm{fine}} - \Delta\phi_{\mathrm{coarse}}\vert{}$ from a companion integration at $10\times$ relaxed tolerances.
