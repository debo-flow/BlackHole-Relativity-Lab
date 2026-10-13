# Physics Formulation: Schwarzschild Spacetime & Equatorial Null Geodesics

## 1. Units and Metric Conventions

All equations in `blackhole_lab.physics` use **geometric units** ($G = c = 1$):
- Time $t$, radial coordinate $r$, mass parameter $M$, and impact parameter $b$ all have dimensions of **length**.
- To convert to SI units for a black hole of physical mass $M_{\mathrm{SI}}$, multiply lengths by $G M_{\mathrm{SI}} / c^2$ and times by $G M_{\mathrm{SI}} / c^3$.

In standard Schwarzschild coordinates $(t, r, \theta, \phi)$ with metric signature $(-, +, +, +)$, the vacuum line element is:
$$ds^2 = -f(r)\,dt^2 + f(r)^{-1}\,dr^2 + r^2\left(d\theta^2 + \sin^2\theta\,d\phi^2\right)$$
where
$$f(r) = 1 - \frac{2M}{r}$$

### Coordinate Singularity vs. Curvature Singularity
- At $r = r_s = 2M$, $f(r_s) = 0$ and $g_{rr} = f(r)^{-1}$ diverges. This is the **event horizon** and is a coordinate singularity of the Schwarzschild chart.
- At $r = 0$, the Kretschmann scalar $R^{\alpha\beta\gamma\delta}R_{\alpha\beta\gamma\delta} = 48M^2/r^6$ diverges; this is the true physical curvature singularity.
- Because the Schwarzschild coordinate chart $(t, r, \theta, \phi)$ is singular at $r = 2M$ and reverses the timelike/spacelike roles of $t$ and $r$ for $r < 2M$, exterior metric evaluations and initial conditions are restricted to $r > 2M$.

---

## 2. Conservation Laws and the Equatorial First Integral

By spherical symmetry, any geodesic is confined to an invariant plane, which we choose as the equatorial plane $\theta = \pi/2$ ($d\theta = 0, \sin\theta = 1$).

Stationarity ($\partial_t$ Killing vector) and axisymmetry ($\partial_\phi$ Killing vector) yield two conserved quantities along a photon worldline $x^\mu(\lambda)$ parameterized by affine parameter $\lambda$:
$$E = -g_{tt}\frac{dt}{d\lambda} = \left(1 - \frac{2M}{r}\right)\frac{dt}{d\lambda}, \qquad L = g_{\phi\phi}\frac{d\phi}{d\lambda} = r^2\frac{d\phi}{d\lambda}$$

For a null geodesic ($ds^2 = 0$):
$$-\left(1 - \frac{2M}{r}\right)\left(\frac{dt}{d\lambda}\right)^2 + \left(1 - \frac{2M}{r}\right)^{-1}\left(\frac{dr}{d\lambda}\right)^2 + r^2\left(\frac{d\phi}{d\lambda}\right)^2 = 0$$

Assuming $E > 0$ and $L > 0$, we define the **impact parameter**:
$$b \equiv \frac{L}{E}$$

Dividing by $L^2$ and changing the dependent variable to inverse radius $u(\phi) = 1/r(\phi)$, where $\frac{du}{d\phi} = -\frac{1}{r^2}\frac{dr}{d\phi}$, yields the exact **first integral**:
$$\left(\frac{du}{d\phi}\right)^2 = \frac{1}{b^2} - u^2 + 2Mu^3 = \frac{1}{b^2} - V_{\mathrm{eff}}(u)$$
where the effective potential is:
$$V_{\mathrm{eff}}(u) = u^2(1 - 2Mu) = \frac{1}{r^2}\left(1 - \frac{2M}{r}\right)$$

---

## 3. Second-Order Orbit Equation and Reference Scales

Differentiating the first integral with respect to $\phi$ (and factoring out $2\,du/d\phi$ for non-circular segments) yields the second-order **Binet orbit equation**:
$$\frac{d^2u}{d\phi^2} = -u + 3Mu^2$$

The three fundamental reference scales in geometric units ($G = c = 1$) are:
1. **Schwarzschild Radius (Event Horizon):**
   $$r_s = 2M$$
2. **Photon-Sphere Radius:**
   $$\frac{d^2u}{d\phi^2} = 0 \implies u_{\mathrm{ph}} = \frac{1}{3M} \iff r_{\mathrm{ph}} = 3M = \frac{3}{2}r_s$$
3. **Critical Impact Parameter:**
   At $r_{\mathrm{ph}} = 3M$, the effective potential attains its unstable maximum $V_{\mathrm{eff,max}} = \frac{1}{27M^2}$, yielding:
   $$b_{\mathrm{crit}} = \frac{1}{\sqrt{V_{\mathrm{eff,max}}}} = \frac{r_{\mathrm{ph}}}{\sqrt{f(r_{\mathrm{ph}})}} = 3\sqrt{3}M = \sqrt{3}\,r_{\mathrm{ph}} = \frac{3\sqrt{3}}{2}\,r_s$$

All three scales can be queried individually via `schwarzschild_radius(M)`, `photon_sphere_radius(M)`, and `critical_impact_parameter(M)`, or together via `reference_scales(M)`.

---

## 4. What Is and Is Not Physically Modeled

| Feature | Status |
| :--- | :--- |
| Exact equatorial Schwarzschild null geodesic ODE | Modeled |
| Exact first-integral initial condition & conservation check | Modeled |
| Reference scales ($r_s = 2M$, $r_{\mathrm{ph}} = 3M$, $b_{\mathrm{crit}} = 3\sqrt{3}M$) | Modeled |
| Purely radial null geodesics ($b = 0$) | Not modeled in $\phi$-solver (requires affine parameter $\lambda$) |
| Horizon-penetrating coordinate continuation ($r < 2M$) | Not modeled (stopped at capture boundary $r_{\mathrm{capture}} \ge 2M$) |
| Black hole spin (Kerr metric), plasma refraction, accretion emission | Not modeled |
