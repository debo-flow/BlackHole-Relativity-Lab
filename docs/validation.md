# Validation Strategy & Scientific Verification Matrix

## 1. Pointwise First-Integral Residual

At every valid integration node $\phi_i$ where $u_i \gt 0$ and $(u_i, v_i)$ are finite, `blackhole_lab.validation.diagnostics` evaluates:
$$\mathcal{R}_i = v_i^2 - \left(\frac{1}{b^2} - u_i^2 + 2Mu_i^3\right)$$
reporting the maximum absolute residual $\max_i \vert{}\mathcal{R}_i\vert{}$, root-mean-square residual $\mathrm{RMS}(\mathcal{R})$, and valid point count.

---

## 2. Weak-Field Post-Newtonian Error Scaling

In the weak-field regime ($b \gg b_{\mathrm{crit}}$), the exact asymptotic deflection angle satisfies:
$$\alpha_{\mathrm{exact}}(b) - \frac{4M}{b} = \frac{15\pi}{4}\left(\frac{M}{b}\right)^2 + \mathcal{O}\!\left(\left(\frac{M}{b}\right)^3\right)$$
In `tests/test_lensing.py`, trajectories at $b/M \in \{40, 80, 160\}$ are integrated and verified to:
1. Agree with the leading-order Einstein prediction $4M/b$ up to the analytic second-order post-Newtonian term $\frac{15\pi}{4}(M/b)^2$.
2. Agree with the second-order formula $\frac{4M}{b} + \frac{15\pi M^2}{4b^2}$ to within $\mathcal{O}((M/b)^3)$.
3. Remain invariant (to within $\sim 10^{-9}\text{ rad}$) when the finite domain boundaries $(r_0, r_{\mathrm{end}})$ are varied from $(60M, 80M)$ to $(400M, 500M)$ with asymptotic tail correction enabled.

---

## 3. Complete Automated Test Suite (`tests/`)

| Test File | Verified Requirements |
| :--- | :--- |
| `tests/test_schwarzschild.py` | Reference scales $r_s = 2M$, $r_{\mathrm{ph}} = 3M$, $b_{\mathrm{crit}} = 3\sqrt{3}M$, dimensionless ratios, Binet equilibrium at $u = 1/(3M)$, metric lapse $f(r)$, coordinate singularity rejection at $r = 2M$, and invalid/overflow input rejection. |
| `tests/test_geodesics.py` | ODE right-hand side accuracy, incoming/outgoing sign conventions, dimensional vs. dimensionless initial-condition separation, local orthonormal pitch-angle identity, supercritical scattering, subcritical capture, unfinished trajectory safety (`PHI_MAX_REACHED`), solver failure isolation (`SOLVER_FAILURE`), and dimensionless mass-scale invariance. |
| `tests/test_diagnostics.py` | First-integral residual decrease under tolerance refinement, near-critical warning generation ($\vert{}b - b_{\mathrm{crit}}\vert{}/b_{\mathrm{crit}} \lt 0.02$), and `NaN` safety for invalid states. |
| `tests/test_lensing.py` | Weak-field $4M/b$ and 2PN scaling, finite-radius invariance with asymptotic tail quadrature, raw finite-radius reporting mode, $2\pi$ angle unwrapping on multi-loop trajectories ($n \ge 1$), fate classification (`ESCAPED`, `CAPTURED`, `UNRESOLVED`), and convergence indicator refinement. |
| `tests/test_visualization.py` | Lensing regime classification, incoming/outgoing segment splitting at periapsis, guarantee that captured and unresolved trajectories have empty outgoing arrays, equal aspect ratio enforcement, coordinate-space title labeling, and invalid input handling. |
| `tests/test_app.py` | Dimensional $b$ vs. dimensionless $b/M$ input resolution, parameter-keyed cache invalidation on `run_cached_single_trajectory`, sweep grid clustering near $b_{\mathrm{crit}}$, and graceful per-ray error recovery in `run_cached_lensing_sweep`. |
