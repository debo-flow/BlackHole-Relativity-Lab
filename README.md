# BlackHole-Relativity-Lab

**Version 3.0.0 — Black Hole Shadow & Photon Sphere Imaging Laboratory**

A modular, research-oriented Python simulation and analysis framework for General Relativity, focusing on equatorial null geodesics, strong-field gravitational light deflection, and 2D black hole shadow ray-tracing in Schwarzschild spacetime.

---

## 1. Project Overview & Implemented Features

This repository provides a scientifically grounded numerical laboratory for studying photon trajectories and static-observer shadow formation around a non-rotating, uncharged Schwarzschild black hole using geometric units ($G = c = 1$).

### Milestone 1: Schwarzschild Spacetime & Photon Geodesics Engine
- **Exact Reference Scales ($G = c = 1$):**
  - Schwarzschild event-horizon radius: $r_s = 2M$ (physical: $2GM/c^2$)
  - Unstable photon-sphere radius: $r_{\mathrm{ph}} = 3M$ (physical: $3GM/c^2$)
  - Critical impact parameter: $b_{\mathrm{crit}} = 3\sqrt{3}M \approx 5.196152\,M$ (physical: $3\sqrt{3}GM/c^2$)
  - Metric lapse function $f(r) = 1 - 2M/r$ and effective potential $V_{\mathrm{eff}}(u) = u^2(1 - 2Mu)$.
- **Controlled Geodesic Integration:** Adaptive Runge-Kutta integration (`scipy.integrate.solve_ivp`, default `DOP853`) of the exact second-order Binet orbit equation $d^2u/d\phi^2 = -u + 3Mu^2$ with terminal event detection for the capture boundary ($r_{\mathrm{capture}} \ge 2M$), finite outer radial boundary ($r_{\mathrm{outer}}$), and invalid domain guards ($u \le 0$).
- **First-Integral Conservation Diagnostics:** Pointwise tracking of the algebraic invariant residual $\mathcal{R}(\phi) = (du/d\phi)^2 - (1/b^2 - u^2 + 2Mu^3)$ and strict status separation (`COMPLETED`, `CAPTURE_BOUNDARY`, `INTEGRATION_LIMIT`, `SOLVER_FAILURE`).

### Milestone 2: Gravitational Light Deflection & Lensing Laboratory
- **Finite-Radius Initial Conditions:** Explicit separation of dimensional quantities ($M, r_0, b, u_0, du/d\phi_0$) and dimensionless quantities ($\phi_0, r_0/M, r_0/r_s, b/M, b/b_{\mathrm{crit}}, M u_0$, and local static orthonormal pitch angle $\sin\psi_0 = \frac{b}{r_0}\sqrt{1 - 2M/r_0}$).
- **Physical Fate Classification (`PhotonFate`):** Every trajectory is classified into `ESCAPED`, `CAPTURED`, or `UNRESOLVED`. Unfinished trajectories ($\phi_{\max}$ reached or solver step failure) are never misclassified as captured or escaped.
- **Gravitational Deflection Engine (`calculate_deflection_angle`):**
  - Unwraps cumulative azimuthal angle $\Delta\phi$ to preserve multi-loop strong-field trajectories ($\Delta\phi > 2\pi$, relativistic winding number $n \ge 1$).
  - Computes raw finite-radius deflection ($\Delta\phi_{\mathrm{finite}} - \pi$), Euclidean chord-corrected finite deflection, and asymptotic deflection via adaptive Gauss-Kronrod quadrature (`scipy.integrate.quad`) of the exact far-field tail integral on $[0, u_0]$ and $[0, u_{\mathrm{end}}]$.
  - Compares numerical deflection against the leading-order Einstein weak-field prediction $\alpha_{\mathrm{weak}} = 4M/b$ and second-order post-Newtonian reference $\frac{4M}{b} + \frac{15\pi M^2}{4b^2}$, reporting absolute error, relative error, and a numerical convergence indicator.
- **Strong-Field Coordinate-Space Visualization:** Splits `ESCAPED` trajectories at periapsis ($r_p > 3M$) into incoming (solid) and outgoing (dashed) paths while guaranteeing `CAPTURED` and `UNRESOLVED` rays are never drawn with an outgoing escaping branch.

### Milestone 3: Black Hole Shadow & Photon Sphere Imaging Laboratory
- **Numerical Critical Boundary Finder (`find_critical_impact_parameter`):** Uses bracketed bisection over actual geodesic integrations to recover $b_{\mathrm{crit}} = 3\sqrt{3}M$, reporting bracket bounds $[b_{\mathrm{low}}, b_{\mathrm{high}}]$, bracket width, absolute/relative errors, and unresolved-ray safeguards.
- **2D Image-Plane Ray-Tracing Engine (`trace_black_hole_shadow`):**
  - Configurable image width, height, field of view, observer distance $r_{\mathrm{obs}} > 3M$, and integration tolerances.
  - Supports both the exact finite-distance static orthonormal tetrad camera model (`"local_angle"`, governed by Synge's formula) and the asymptotic impact-parameter plane convention (`"impact_parameter"`).
  - Exploits exact spherical symmetry (`deduplicate_symmetric_rays=True`) to integrate unique impact parameters once and map exact solver fates across the 2D pixel grid.
  - Generates mutually exclusive `capture_mask`, `escaped_mask`, and `unresolved_mask` arrays (`classification_map` values `1`, `0`, `-1`), never converting unresolved pixels into captured shadow pixels.
- **Interactive Streamlit Dashboard (`app.py`):** Unified desktop browser laboratory covering single-ray geodesics, multi-parameter lensing sweeps, and 2D black hole shadow imaging with parameter-keyed caching and resolution adequacy checks.

---

## 2. Scientific Honesty & Physical Assumptions

1. **Photon Sphere vs. Apparent Shadow Radius:** The coordinate radius of the photon sphere ($r_{\mathrm{ph}} = 3GM/c^2 = 3M$) is **not** equal to the apparent shadow radius ($b_{\mathrm{shadow}} = 3\sqrt{3}GM/c^2 \approx 5.196152\,M$). Gravitational light bending magnifies the photon sphere's capture cross-section for a distant observer by a factor of $\sqrt{3}$.
2. **Distant-Observer vs. Finite-Distance Geometry:** The constant impact-parameter radius $b_{\mathrm{shadow}} = 3\sqrt{3}M$ describes the apparent radius on the impact-parameter plane of an asymptotic observer ($r_{\mathrm{obs}} \to \infty$, where angular radius $\psi_{\mathrm{crit}} \simeq b_{\mathrm{shadow}}/r_{\mathrm{obs}}$). For a static observer at a finite distance $r_{\mathrm{obs}} > 3M$, the true angular radius on the observer's sky is given by Synge's exact formula:
   $$\sin\psi_{\mathrm{crit}} = \frac{3\sqrt{3}M}{r_{\mathrm{obs}}}\sqrt{1 - \frac{2M}{r_{\mathrm{obs}}}}$$
3. **Capture Masks Are Not Accretion Disk Photographs:** The 2D shadow plots generated in this project are ray-traced geometric capture masks showing which observer lines of sight intersect the event horizon versus escaping to infinity. They are **not** synthetic camera photographs of an astrophysical accretion disk; relativistic plasma emission, Doppler beaming, GRMHD, and radiative transfer are not modeled in this release.
4. **Spacetime Symmetry:** The model assumes a static, spherically symmetric vacuum Schwarzschild spacetime ($a = 0, Q = 0$). Rotating Kerr black holes ($a \neq 0$) are reserved for future milestones.
5. **Numerical Dependence:** All numerical outputs depend on grid resolution, ODE integration tolerances (`rtol`, `atol`), capture boundary factor (`capture_radius_factor`), and the maximum azimuthal integration limit (`phi_max`).

---

## 3. Supported Python Versions & Dependencies

- **Python Version:** Python **3.10+**, **3.11**, or **3.12**
- **Runtime Dependencies:**
  - `numpy >= 1.26.0, < 3.0.0`
  - `scipy >= 1.11.0, < 2.0.0`
  - `matplotlib >= 3.8.0, < 4.0.0`
  - `streamlit >= 1.32.0, < 2.0.0`
- **Development & Testing Dependencies:**
  - `pytest >= 8.0.0, < 9.0.0`
  - `pytest-cov >= 5.0.0, < 6.0.0`
  - `ruff >= 0.4.0, < 1.0.0`

---

## 4. Installation Instructions

Clone the repository and create an isolated virtual environment before installing the package in editable mode (`-e`).

### Linux / macOS
```bash
git clone [https://github.com/your-username/BlackHole-Relativity-Lab.git](https://github.com/your-username/BlackHole-Relativity-Lab.git)
cd BlackHole-Relativity-Lab

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows (PowerShell)
```powershell
git clone [https://github.com/your-username/BlackHole-Relativity-Lab.git](https://github.com/your-username/BlackHole-Relativity-Lab.git)
cd BlackHole-Relativity-Lab

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows (Command Prompt)
```cmd
git clone [https://github.com/your-username/BlackHole-Relativity-Lab.git](https://github.com/your-username/BlackHole-Relativity-Lab.git)
cd BlackHole-Relativity-Lab

python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 5. How to Run the Application, Examples & Automated Tests

### Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

### Run the Automated Test Suite
```bash
# Run all unit, integration, and scientific validation tests
pytest -v

# Run specific milestone test suites
pytest tests/test_schwarzschild.py tests/test_geodesics.py tests/test_diagnostics.py -v
pytest tests/test_lensing.py tests/test_visualization.py -v
pytest tests/test_shadow.py tests/test_scientific_validation.py tests/test_app.py -v
```

### Run Code Quality & Linting Checks
```bash
ruff check src tests examples app.py
```

### Run the Command-Line Demonstration Script
```bash
python examples/basic_trajectories.py
```

---

## 6. Mathematical Model & Unit Conventions

All core modules use **geometric units** ($G = c = 1$), in which mass $M$, radial coordinate $r$, and impact parameter $b = L/E$ share dimensions of length $[L]$ (where $1\,M = GM_{\mathrm{SI}}/c^2$).

### Schwarzschild Metric
In Schwarzschild coordinates $(t, r, \theta, \phi)$ with signature $(-, +, +, +)$:
$$ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \left(1 - \frac{2M}{r}\right)^{-1} dr^2 + r^2\left(d\theta^2 + \sin^2\theta\, d\phi^2\right)$$

### Equatorial Null Geodesics ($\theta = \pi/2, u = 1/r$)
With inverse radius $u(\phi) = 1/r(\phi)$ and azimuthal angle $\phi$ as the independent variable, photon trajectories satisfy the second-order Binet equation and conserved first integral:
$$\frac{d^2u}{d\phi^2} = -u + 3Mu^2, \qquad \left(\frac{du}{d\phi}\right)^2 = \frac{1}{b^2} - u^2 + 2Mu^3$$

### Deflection Angle & Weak-Field Expansion
For an escaping photon ($b > b_{\mathrm{crit}}$) with unwrapped cumulative azimuthal sweep $\Delta\phi$, the net gravitational deflection angle is:
$$\alpha = \Delta\phi - \pi$$
In the weak-field limit ($b \gg b_{\mathrm{crit}}$):
$$\alpha_{\mathrm{weak}}(M, b) = \frac{4M}{b} + \frac{15\pi}{4}\left(\frac{M}{b}\right)^2 + \mathcal{O}\!\left(\left(\frac{M}{b}\right)^3\right)$$

---

## 7. Example Simulation Workflow (Python API)

```python
import math
from blackhole_lab import (
    GeodesicConfig,
    ShadowGridConfig,
    calculate_deflection_angle,
    find_critical_impact_parameter,
    reference_scales,
    trace_black_hole_shadow,
)

# 1. Inspect exact Schwarzschild reference scales for M = 1
scales = reference_scales(M=1.0)
print(f"r_s = {scales.r_s:.4f}, r_ph = {scales.r_ph:.4f}, b_crit = {scales.b_crit:.6f}")

# 2. Integrate a single escaping photon and compare with weak-field 4M/b
cfg = GeodesicConfig(M=1.0, r0=100.0, b=40.0, outer_radius=150.0, rtol=1e-10, atol=1e-12)
deflection = calculate_deflection_angle(cfg, apply_asymptotic_correction=True)
print(f"Fate: {deflection.fate.value}")
print(f"Numerical deflection : {deflection.numerical_deflection:.6f} rad")
print(f"Weak-field 4M/b      : {deflection.analytical_weak_field:.6f} rad")

# 3. Locate the critical impact parameter boundary via numerical bisection
crit_res = find_critical_impact_parameter(M=1.0, r0=30.0, tolerance=1e-6, phi_max=10.0 * math.pi)
print(f"Numerical b_crit     : {crit_res.b_crit_numerical:.6f} (abs error = {crit_res.absolute_error:.2e})")

# 4. Ray-trace a 2D black hole shadow mask for an observer at r_obs = 30M
shadow_cfg = ShadowGridConfig(M=1.0, r_obs=30.0, width=51, height=51, ray_model="local_angle")
shadow_res = trace_black_hole_shadow(shadow_cfg)
print(
    f"Shadow grid {shadow_res.capture_mask.shape}: "
    f"captured={shadow_res.num_captured}, escaped={shadow_res.num_escaped}, "
    f"unresolved={shadow_res.num_unresolved}"
)
```

---

## 8. Known Limitations

1. **Azimuthal Angle Parameterization ($\phi$):** Strictly radial trajectories ($b = 0 \implies d\phi = 0$) cannot be integrated by the $\phi$-parameterized Binet ODE. In the 1D solver (`solve_photon_geodesic`), $b \le 0$ raises a `ValueError`; in the 2D shadow engine (`trace_black_hole_shadow`), the exact on-axis pixel ($b = 0$) is handled analytically via the radial null geodesic equation $(dr/d\lambda)^2 = E^2$.
2. **Exterior Schwarzschild Chart ($r > 2M$):** Because the Schwarzschild coordinate chart has a coordinate singularity at $r_s = 2M$, numerical integration stops at a configurable capture boundary $r_{\mathrm{capture}} = \text{capture\_radius\_factor} \times 2M \ge 2M$ (default $1.001\,r_s$).
3. **Exponential Sensitivity Near $b_{\mathrm{crit}}$:** Trajectories skimming the unstable photon sphere ($r_{\mathrm{ph}} = 3M$) have a Lyapunov exponent of $1\text{ rad}^{-1}$ in $\phi$, amplifying numerical truncation errors as $e^{+\phi}$. Rays that exceed `phi_max` before crossing either radial boundary are explicitly classified as `UNRESOLVED`.

---

## 9. Repository Structure

```text
BlackHole-Relativity-Lab/
├── .github/
│   └── workflows/
│       └── tests.yml                     # Multi-OS GitHub Actions CI pipeline
├── .gitignore
├── LICENSE                               # MIT License
├── README.md                             # Project overview and user guide
├── app.py                                # Interactive Streamlit laboratory (Milestones 1–3)
├── pyproject.toml                        # Package build, pytest, and Ruff configuration
├── requirements.txt                      # Pinned runtime and development dependencies
├── docs/
│   ├── numerical_methods.md              # ODE system, tail quadrature & shadow ray-tracing
│   ├── physics.md                        # Schwarzschild metric, lensing & Synge shadow theory
│   └── validation.md                     # Conservation diagnostics & convergence test matrix
├── examples/
│   └── basic_trajectories.py             # CLI demonstration of geodesics and deflection
├── src/
│   └── blackhole_lab/
│       ├── __init__.py                   # Package root exports (v3.0.0)
│       ├── physics/
│       │   ├── __init__.py
│       │   └── schwarzschild.py          # Reference scales, metric functions & Synge formula
│       ├── numerics/
│       │   ├── __init__.py
│       │   ├── geodesics.py              # Initial conditions, Binet ODE & solve_ivp engine
│       │   ├── lensing.py                # Fate classifier, deflection angle & tail quadrature
│       │   └── shadow.py                 # Critical boundary bisection & 2D shadow ray-tracer
│       ├── validation/
│       │   ├── __init__.py
│       │   └── diagnostics.py            # First-integral residuals & instability warnings
│       └── visualization/
│           ├── __init__.py
│           └── plots.py                  # Coordinate-space trajectory, lensing & shadow plots
└── tests/
    ├── __init__.py
    ├── test_app.py                       # Streamlit helper and cache invalidation tests
    ├── test_diagnostics.py               # First-integral residual scaling tests
    ├── test_geodesics.py                 # Initial condition, ODE & termination status tests
    ├── test_lensing.py                   # Weak-field, finite-radius & angle unwrapping tests
    ├── test_schwarzschild.py             # Reference scale & input validation tests
    ├── test_scientific_validation.py     # Resolution/tolerance convergence & regression suite
    ├── test_shadow.py                    # Shadow mask dimensions, symmetry & bisection tests
    └── test_visualization.py             # Trajectory splitting & plot generation tests
```

---

## 10. Roadmap for Future Milestones

- **Milestone 4:** Affine-parameter ($\lambda$) 4D geodesic integration in horizon-penetrating coordinates (ingoing Eddington-Finkelstein / Painlevé-Gullstrand) for smooth continuation across $r = 2M$.
- **Milestone 5:** Thin equatorial accretion disk emission model with gravitational redshift, relativistic Doppler beaming, and higher-order photon ring ($n = 1, 2$) intensity profiles.
- **Milestone 6:** Rotating Kerr spacetime ($0 < |a| < M$) geodesic integrator using Carter's fourth constant of motion $\mathcal{Q}$ and asymmetric D-shaped shadow boundary analysis.
