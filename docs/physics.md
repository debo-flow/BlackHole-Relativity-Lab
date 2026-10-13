# Physics Formulation: Schwarzschild Spacetime & Equatorial Null Geodesics

## 1. Units and Metric Conventions

All equations in `blackhole_lab.physics` use **geometric units** ($G = c = 1$):
- Time $t$, radial coordinate $r$, mass parameter $M$, and impact parameter $b$ all have dimensions of **length**.
- To convert to SI units for a black hole of physical mass $M_{\mathrm{SI}}$, multiply lengths by $G M_{\mathrm{SI}} / c^2$ and times by $G M_{\mathrm{SI}} / c^3$.

In standard Schwarzschild coordinates $(t, r, \theta, \phi)$ with metric signature $(-, +, +, +)$, the vacuum line element is:
$$ds^2 = -f(r)\,dt^2 + f(r)^{-1}\,dr^2 + r^2\left(d\theta^2 + \sin^2\theta\,d\phi^2\right)$$
where
$$f(r) = 1 - \frac{2M}{r}$$

---

## 2. Conservation Laws and the Equatorial First Integral

By spherical symmetry, any geodesic is confined to an invariant plane, which we choose as the equatorial plane $\theta = \pi/2$ ($d\theta = 0, \sin\theta = 1$).

Stationarity ($\partial_t$ Killing vector) and axisymmetry ($\partial_\phi$ Killing vector) yield two conserved quantities along a photon worldline $x^\mu(\lambda)$ parameterized by affine parameter $\lambda$:
$$E = -g_{tt}\frac{dt}{d\lambda} = \left(1 - \frac{2M}{r}\right)\frac{dt}{d\lambda}, \qquad L = g_{\phi\phi}\frac{d\phi}{d\lambda} = r^2\frac{d\phi}{d\lambda}$$

Assuming $E \gt 0$ and $L \gt 0$, the **impact parameter** is $b \equiv L / E$, and the exact first integral in terms of $u(\phi) = 1/r(\phi)$ is:
$$\left(\frac{du}{d\phi}\right)^2 = \frac{1}{b^2} - u^2 + 2Mu^3 = \frac{1}{b^2} - V_{\mathrm{eff}}(u)$$

---

## 3. Second-Order Orbit Equation and Reference Scales

Differentiating the first integral with respect to $\phi$ yields the second-order **Binet orbit equation**:
$$\frac{d^2u}{d\phi^2} = -u + 3Mu^2$$

The three fundamental reference scales in geometric units ($G = c = 1$) are:
1. **Schwarzschild Radius (Event Horizon):** $r_s = 2M$
2. **Photon-Sphere Radius:** $r_{\mathrm{ph}} = 3M = \frac{3}{2}r_s$
3. **Critical Impact Parameter:** $b_{\mathrm{crit}} = 3\sqrt{3}M = \sqrt{3}\,r_{\mathrm{ph}}$

---

## 4. Gravitational Light Deflection: Exact, Finite-Radius, and Weak-Field Formulations

### 4.1 Angle Convention & Angular Unwrapping
For an escaping photon ($b \gt b_{\mathrm{crit}}$) with unwrapped cumulative azimuthal sweep $\Delta\phi$, a straight-line trajectory in flat Minkowski spacetime ($M = 0$) from $r = \infty$ to $r = \infty$ sweeps $\Delta\phi_{\mathrm{flat}} = \pi$. We define the net gravitational deflection angle as:
$$\alpha = \Delta\phi - \pi$$
When $b$ is only slightly larger than $b_{\mathrm{crit}}$, the photon can execute $n \ge 1$ complete relativistic loops around the photon sphere before escaping, yielding $\Delta\phi \gt (2n + 1)\pi$ and $\alpha \gt 2n\pi$.

### 4.2 Finite Starting/Ending Radii and Asymptotic Tail Reconstruction
A numerical ODE integration starting at finite radius $r_0$ ($u_0 = 1/r_0$) and stopping at finite outer radius $r_{\mathrm{end}}$ ($u_{\mathrm{end}} = 1/r_{\mathrm{end}}$) measures the finite-domain angular sweep $\Delta\phi_{\mathrm{finite}}(r_0, r_{\mathrm{end}}) = \phi_{\mathrm{end}} - \phi_0$. The omitted far-field angular sweep between $r_{\mathrm{bound}}$ and $r = \infty$ ($u = 0$) is given by the exact quadrature integral:
$$\delta\phi_{\mathrm{tail}}(u_{\mathrm{bound}}; M, b) = \int_0^{u_{\mathrm{bound}}} \frac{du}{\sqrt{\frac{1}{b^2} - u^2 + 2Mu^3}}$$
Adding $\delta\phi_{\mathrm{tail}}(u_0; M, b) + \delta\phi_{\mathrm{tail}}(u_{\mathrm{end}}; M, b)$ to $\Delta\phi_{\mathrm{finite}}(r_0, r_{\mathrm{end}})$ reconstructs the full asymptotic deflection angle $\alpha_\infty = \Delta\phi_\infty - \pi$ independently of the finite domain truncation radii $r_0$ and $r_{\mathrm{end}}$. Both the raw finite-radius deflection ($\Delta\phi_{\mathrm{finite}} - \pi$), the Euclidean chord-corrected finite deflection, and the asymptotic-corrected deflection are reported in `DeflectionResult`.

### 4.3 Weak-Field Analytical Prediction
In the weak-field regime ($b \gg b_{\mathrm{crit}}$, or $M/b \ll 1$), expanding the exact deflection integral in powers of $M/b$ yields:
$$\alpha_{\mathrm{weak}}(M, b) = \frac{4M}{b} + \frac{15\pi}{4}\left(\frac{M}{b}\right)^2 + \mathcal{O}\!\left(\left(\frac{M}{b}\right)^3\right)$$
The leading-order Einstein prediction $\alpha \approx 4M/b$ is valid only for $b \gg b_{\mathrm{crit}}$. As $b \to b_{\mathrm{crit}}^+$, the true deflection diverges logarithmically while $4M/b$ remains finite ($\frac{4}{3\sqrt{3}} \approx 0.7698\text{ rad}$).
