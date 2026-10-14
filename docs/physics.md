# Physics Formulation: Schwarzschild Spacetime & Gravitational Light Deflection

## 1. Units and Metric Conventions

All equations in `blackhole_lab.physics` use **geometric units** ($G = c = 1$):
- Time $t$, radial coordinate $r$, mass parameter $M$, and impact parameter $b$ all have dimensions of **length** $[L]$.
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

## 2. Conservation Laws, Impact Parameter, and the Equatorial First Integral

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

In a local static orthonormal tetrad at radius $r_0 > 2M$, the angle $\psi_0$ between the photon propagation vector and the radial direction satisfies:
$$\sin\psi_0 = \frac{b}{r_0}\sqrt{1 - \frac{2M}{r_0}}$$

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

## 4. Gravitational Light Deflection: Exact, Finite-Radius, and Weak-Field Formulations

### 4.1 Angle Convention & Angular Unwrapping
For an escaping photon ($b > b_{\mathrm{crit}}$) with unwrapped cumulative azimuthal sweep $\Delta\phi$, a straight-line trajectory in flat Minkowski spacetime ($M = 0$) from $r = \infty$ to $r = \infty$ sweeps $\Delta\phi_{\mathrm{flat}} = \pi$. We define the net gravitational deflection angle as:
$$\alpha = \Delta\phi - \pi$$
When $b$ is only slightly larger than $b_{\mathrm{crit}}$, the photon can execute $n \ge 1$ complete relativistic loops around the photon sphere before escaping, yielding $\Delta\phi > (2n + 1)\pi$ and $\alpha > 2n\pi$.

### 4.2 Finite Starting/Ending Radii and Asymptotic Tail Reconstruction
A numerical ODE integration starting at finite radius $r_0$ ($u_0 = 1/r_0$) and stopping at finite outer radius $r_{\mathrm{end}}$ ($u_{\mathrm{end}} = 1/r_{\mathrm{end}}$) measures the finite-domain angular sweep $\Delta\phi_{\mathrm{finite}}(r_0, r_{\mathrm{end}}) = \phi_{\mathrm{end}} - \phi_0$. The omitted far-field angular sweep between $r_{\mathrm{bound}}$ and $r = \infty$ ($u = 0$) is given by the exact quadrature integral:
$$\delta\phi_{\mathrm{tail}}(u_{\mathrm{bound}}; M, b) = \int_0^{u_{\mathrm{bound}}} \frac{du}{\sqrt{\frac{1}{b^2} - u^2 + 2Mu^3}}$$
Adding $\delta\phi_{\mathrm{tail}}(u_0; M, b) + \delta\phi_{\mathrm{tail}}(u_{\mathrm{end}}; M, b)$ to $\Delta\phi_{\mathrm{finite}}(r_0, r_{\mathrm{end}})$ reconstructs the full asymptotic deflection angle $\alpha_\infty = \Delta\phi_\infty - \pi$ independently of the finite domain truncation radii $r_0$ and $r_{\mathrm{end}}$. Both the raw finite-radius deflection ($\Delta\phi_{\mathrm{finite}} - \pi$), the Euclidean chord-corrected finite deflection, and the asymptotic-corrected deflection are reported in `DeflectionResult`.

### 4.3 Weak-Field Analytical Prediction & Strong-Field Breakdown
In the weak-field regime ($b \gg b_{\mathrm{crit}}$, or $M/b \ll 1$), expanding the exact deflection integral in powers of $M/b$ yields:
$$\alpha_{\mathrm{weak}}(M, b) = \frac{4M}{b} + \frac{15\pi}{4}\left(\frac{M}{b}\right)^2 + \mathcal{O}\!\left(\left(\frac{M}{b}\right)^3\right)$$
The leading-order Einstein prediction $\alpha \approx 4M/b$ is valid only for $b \gg b_{\mathrm{crit}}$. As $b \to b_{\mathrm{crit}}^+$, the true deflection diverges logarithmically while $4M/b$ remains finite ($\frac{4}{3\sqrt{3}} \approx 0.7698\text{ rad}$).

---

## 5. What Is and Is Not Physically Modeled

| Feature | Status |
| :--- | :--- |
| Exact equatorial Schwarzschild null geodesic ODE | Modeled |
| Exact first-integral initial condition & conservation check | Modeled |
| Reference scales ($r_s = 2M$, $r_{\mathrm{ph}} = 3M$, $b_{\mathrm{crit}} = 3\sqrt{3}M$) | Modeled |
| Gravitational light deflection (finite-radius, asymptotic tail, weak-field comparison) | Modeled |
| Purely radial null geodesics ($b = 0$) | Not modeled in $\phi$-solver (requires affine parameter $\lambda$) |
| Horizon-penetrating coordinate continuation ($r < 2M$) | Not modeled (stopped at capture boundary $r_{\mathrm{capture}} \ge 2M$) |
| Black hole spin (Kerr metric), plasma refraction, accretion emission | Not modeled |
