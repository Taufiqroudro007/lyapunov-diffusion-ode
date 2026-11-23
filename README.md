# Finite-Time Lyapunov Analysis of Probability Flow ODEs in Bimodal Score-Based Diffusion

## 1. Overview
This repository provides the reproducible numerical implementation, validation checks, and figure-generation scripts for the IEEE conference paper:

> **Finite-Time Lyapunov Analysis of Probability Flow ODEs in Bimodal Score-Based Diffusion**  
> M Taufiqul Islam Roudro, Mahmudul Hasan Zihad, and Mohd Ruhul Ameen  
> *2026 International Conference on Recent Progresses in Science, Engineering and Technology (ICRPSET)*

The codebase implements:
- **Analytical & Variational Reverse-Time PF-ODE Formulation:** Integrates the reverse-time probability flow ordinary differential equation (PF-ODE) and its variational flow-map Jacobian for a symmetric bimodal Gaussian mixture with variance schedule $g^2(t) = 0.1 + 19.9t$ over $\tau \in [0, T - \epsilon] = [0, 0.9999]$ ($\sigma_0 = 0.2$, $\mu = 2$, $\epsilon = 10^{-4}$, $T = 1$).
- **Analytical Central-Trajectory Verification:** Evaluates the closed-form central-trajectory finite-time Lyapunov exponent (FTLE), $\Lambda(0) \approx 47.028782$, and verifies numerical integration against the exact analytical expression.
- **Reference-Solution Self-Convergence Audits:** Measures maximum terminal-state differences $\Delta_{\max}^{(N,2N)}$ across RK4 step refinements (`1000` to `16000` steps in 1D; `2000` to `8000` steps in 2D).
- **One- and Two-Dimensional FTLE Resolution Studies:** Computes 1D FTLE profiles across $25$ initial conditions in $[-3, 3]$ (`600` to `4800` steps) and 2D maximal-singular-value FTLE fields over an $11 \times 11$ initial-condition grid in $[-4, 4]^2$ (`600` to `9600` steps) with component means $(\pm 2, 0)$.
- **Variational vs. Centered Finite-Difference Validation:** Compares variational flow-map derivatives $\Phi(L, 0)$ against centered finite-difference approximations $D_h(x_T)$ at $x_T = \pm 2, \pm 1, \pm 0.25$ across perturbations $h \in [10^{-10}, 10^{-3}]$.
- **Matched-NFE Solver Comparisons:** Benchmarks Euler (1 NFE/step), Heun (2 NFE/step), and fourth-order Runge--Kutta (RK4, 4 NFE/step) under matched right-hand-side evaluation budgets (`60` to `960` NFE in 1D across $58$ initial conditions excluding $|x_T| \le 0.05$; `60` to `480` NFE in 2D across $121$ grid points), along with nonfinite trajectory tracking.

---

## 2. Environment Setup
Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
