# Physics Formulation: Schwarzschild Spacetime, Light Deflection & Black Hole Shadow

## 1. Units and Metric Conventions

All equations in `blackhole_lab.physics` use **geometric units** ($G = c = 1$):
- Time $t$, radial coordinate $r$, mass parameter $M$, and impact parameter $b$ all have dimensions of **length** $[L]$.
- To convert to SI units for a black hole of physical mass $M_{\mathrm{SI}}$, multiply lengths by $G M_{\mathrm{SI}} / c^2$ and times by $G M_{\mathrm{SI}} / c^3$.

In standard Schwarzschild coordinates $(t, r, \theta, \phi)$ with metric signature $(-, +, +, +)$, the vacuum line element is:
$$ds^2 = -f(r)\,dt^2 + f(r)^{-1}\,dr^2 + r^2\left(d\theta^2 + \sin^2\theta\,d\phi^2\right), \qquad f(r) = 1 - \frac{2M}{r}$$

---

## 2. Conservation Laws, Impact Parameter, and the Equatorial First Integral

By spherical symmetry, any geodesic is confined to an invariant plane passing through the origin. Stationarity ($\partial_t$ Killing vector) and axisymmetry ($\partial_\phi$ Killing vector) yield conserved energy $E = f(r)\,\dot{t}$ and angular momentum $L = r^2\dot{\phi}$. Defining the **impact parameter** $b \equiv L / E$ and inverse radius $u(\phi) = 1/r(\phi)$ gives the exact **first integral**:
$$\left(\frac{du}{d\phi}\right)^2 = \frac{1}{b^2} - u^2 + 2Mu^3 = \frac{1}{b^2} - V_{\mathrm{eff}}(u)$$
and the second-order **Binet orbit equation**:
$$\frac{d^2u}{d\phi^2} = -u + 3Mu^2$$

The three fundamental reference scales in geometric units ($G = c = 1$) are:
1. **Schwarzschild Radius (Event Horizon):** $r_s = 2M$
2. **Photon-Sphere Radius:** $r_{\mathrm{ph}} = 3M = \frac{3}{2}r_s$
3. **Critical Impact Parameter:** $b_{\mathrm{crit}} = 3\sqrt{3}M = \sqrt{3}\,r_{\mathrm{ph}}$

---

## 3. Gravitational Light Deflection

For an escaping photon ($b \gt b_{\mathrm{crit}}$) with unwrapped cumulative azimuthal sweep $\Delta\phi$, the net gravitational deflection angle relative to a flat-space straight line ($\Delta\phi_{\mathrm{flat}} = \pi$) is:
$$\alpha = \Delta\phi - \pi$$
In the weak-field regime ($b \gg b_{\mathrm{crit}}$), the leading-order Einstein approximation and second-order post-Newtonian correction are:
$$\alpha_{\mathrm{weak}}(M, b) = \frac{4M}{b} + \frac{15\pi}{4}\left(\frac{M}{b}\right)^2 + \mathcal{O}\!\left(\left(\frac{M}{b}\right)^3\right)$$

---

## 4. Black Hole Shadow & Observer Ray Geometry

### 4.1 Static Observer Local Orthonormal Tetrad
Consider a static observer at Schwarzschild radial coordinate $r_{\mathrm{obs}} \gt r_{\mathrm{ph}} = 3M$ equipped with the orthonormal frame:
$$e_{(t)} = f(r_{\mathrm{obs}})^{-1/2}\partial_t, \quad e_{(r)} = f(r_{\mathrm{obs}})^{1/2}\partial_r, \quad e_{(\theta)} = \frac{1}{r_{\mathrm{obs}}}\partial_\theta, \quad e_{(\phi)} = \frac{1}{r_{\mathrm{obs}}\sin\theta}\partial_\phi$$
A light ray traced from the observer at local viewing angle $\psi_{\mathrm{obs}} \in [0, \pi/2)$ relative to the inward radial direction $-e_{(r)}$ has conserved impact parameter:
$$b(\psi_{\mathrm{obs}}) = \frac{r_{\mathrm{obs}}\sin\psi_{\mathrm{obs}}}{\sqrt{1 - \frac{2M}{r_{\mathrm{obs}}}}}$$

### 4.2 Critical Boundary & Synge's Shadow Angular Radius Formula
Rays with $b \lt b_{\mathrm{crit}} = 3\sqrt{3}M$ cross the photon sphere $r_{\mathrm{ph}} = 3M$ and are captured by the event horizon (forming the **black hole shadow** on the observer's sky), whereas rays with $b \gt b_{\mathrm{crit}}$ reach a periapsis $r_p \gt 3M$ and escape to infinity. Setting $b(\psi_{\mathrm{crit}}) = b_{\mathrm{crit}}$ yields Synge's exact formula for the angular radius of the Schwarzschild shadow:
$$\sin\psi_{\mathrm{crit}} = \frac{3\sqrt{3}M}{r_{\mathrm{obs}}}\sqrt{1 - \frac{2M}{r_{\mathrm{obs}}}}$$
As $r_{\mathrm{obs}} \to \infty$, $\psi_{\mathrm{crit}} \simeq b_{\mathrm{crit}} / r_{\mathrm{obs}} = 3\sqrt{3}M / r_{\mathrm{obs}}$.
