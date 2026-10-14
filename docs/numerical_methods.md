# Numerical Methods & Solver Architecture

## 1. First-Order Geodesic ODE System

We integrate the second-order Binet equation $\frac{d^2u}{d\phi^2} = -u + 3Mu^2$ as a first-order initial-value system for $\mathbf{y}(\phi) = [u(\phi), v(\phi)]^T$:
$$\frac{du}{d\phi} = v, \qquad \frac{dv}{d\phi} = -u + 3Mu^2$$
using `scipy.integrate.solve_ivp` with terminal event detection for capture ($r = r_{\mathrm{capture}} \ge 2M$), outer domain truncation ($r = r_{\mathrm{outer}}$), and invalid domain guards ($u \le 0$).

---

## 2. Numerical Critical Boundary Finder (`find_critical_impact_parameter`)

To locate the critical capture/escape threshold $b_{\mathrm{crit}}$ numerically without relying on the analytical value $3\sqrt{3}M$:
1. **Verified Bracketing:** Starting from $[b_{\mathrm{low}}, b_{\mathrm{high}}]$, we integrate both endpoints using `solve_photon_geodesic` and verify via `classify_photon_fate` that $b_{\mathrm{low}}$ is `CAPTURED` and $b_{\mathrm{high}}$ is `ESCAPED`.
2. **Bisection with Unresolved-Ray Guard:** At each midpoint $b_{\mathrm{mid}} = \frac{1}{2}(b_{\mathrm{low}} + b_{\mathrm{high}})$, we integrate the geodesic and classify its fate:
   - `CAPTURED` $\implies b_{\mathrm{low}} \leftarrow b_{\mathrm{mid}}$
   - `ESCAPED` $\implies b_{\mathrm{high}} \leftarrow b_{\mathrm{mid}}$
   - `UNRESOLVED` $\implies$ bisection stops immediately, sets `unresolved_encountered = True`, and reports the last verified bracket $[b_{\mathrm{low}}, b_{\mathrm{high}}]$ rather than misclassifying an unfinished ray as captured.

---

## 3. Two-Dimensional Shadow Ray-Tracing Engine (`trace_black_hole_shadow`)

1. **2D Image-Plane Grid:** Constructs a pixel grid $(\alpha_x, \alpha_y) \in [-\mathrm{FOV}/2, +\mathrm{FOV}/2]^2$ of shape $(N_y, N_x)$ and computes the polar sky angle $\theta_{\mathrm{sky}}(i, j) = \sqrt{\alpha_x^2 + \alpha_y^2}$ for each pixel.
2. **Ray Initialization Models:**
   - `"local_angle"`: $b(i, j) = r_{\mathrm{obs}}\sin\theta_{\mathrm{sky}}(i, j) / \sqrt{1 - 2M/r_{\mathrm{obs}}}$
   - `"impact_parameter"`: $b(i, j) = r_{\mathrm{obs}}\theta_{\mathrm{sky}}(i, j)$
3. **Geodesic Tracing & Symmetry Deduplication:**
   - Because Schwarzschild spacetime is spherically symmetric, each ray's radial history depends only on $b(i, j)$ and $r_{\mathrm{obs}}$.
   - For $b(i, j) \gt 0$, rays are integrated via `solve_photon_geodesic` and classified via `classify_photon_fate`.
   - For an exact on-axis pixel ($b = 0$, where $d\phi = 0$), the radial null geodesic equation $dr/d\lambda = -E$ is used directly.
   - When `deduplicate_symmetric_rays=True` (default), symmetric pixels sharing the exact same impact parameter $b(i, j)$ reuse a single numerical ODE integration, providing a ~7x speedup on square grids with zero approximation error.
4. **Strict Mask Separation:**
   - `capture_mask` is `True` only where `fate == PhotonFate.CAPTURED`.
   - `escaped_mask` is `True` only where `fate == PhotonFate.ESCAPED`.
   - `unresolved_mask` is `True` only where `fate == PhotonFate.UNRESOLVED`.
   - Unresolved pixels are never silently converted into captured pixels.
