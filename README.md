# Finite-Time Lyapunov Analysis of Probability Flow ODEs in Bimodal Score-Based Diffusion

## 1. Overview
This repository provides the reproducible implementation, numerical results, and figure-generation code for the IEEE conference paper:

> **Finite-Time Lyapunov Analysis of Probability Flow ODEs in Bimodal Score-Based Diffusion**  
> M Taufiqul Islam Roudro, Mahmudul Hasan Zihad, and Mohd Ruhul Ameen  
> *2026 International Conference on Recent Progresses in Science, Engineering and Technology (ICRPSET)*

The codebase implements:
- **Analytical & Symbolic Verification (`analytical_derivations.py`):** Verifies the closed-form score function, spatial Jacobian, frozen-time pitchfork transition ($t_c \approx 0.62586$), and central-trajectory reverse-time FTLE ($\Lambda(0) \approx 47.028782$) for the variance schedule $g^2(t) = 0.1 + 19.9t$ over $\tau \in [0, T - \epsilon] = [0, 0.9999]$ ($\sigma_0 = 0.2$, $\mu = 2$, $\epsilon = 10^{-4}$, $T = 1$).
- **Numerical Experiments & Plotting (`main_experiments.py`):**
  - Evaluates 1D and 2D RK4 reference-solution self-convergence across step refinements.
  - Computes 1D FTLE resolution profiles across $25$ initial conditions in $[-3, 3]$ (`600` to `4800` steps) and 2D maximal-singular-value FTLE fields over an $11 \times 11$ initial-condition grid in $[-4, 4]^2$ (`600` to `9600` steps) with component means $(\pm 2, 0)$.
  - Validates variational flow-map derivatives $\Phi(L, 0)$ against centered finite differences $D_h(x_T)$ at $x_T = \pm 2, \pm 1, \pm 0.25$ across perturbations $h \in [10^{-10}, 10^{-3}]$.
  - Benchmarks Euler (1 NFE/step), Heun (2 NFE/step), and RK4 (4 NFE/step) under matched right-hand-side evaluation budgets in 1D (`60` to `960` NFE across $58$ initial conditions excluding $|x_T| \le 0.05$) and 2D (`60` to `480` NFE across $121$ grid points).

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

```

---

## 3. Running the Experiments

From the repository root directory:

1. **Verify analytical derivations and closed-form central-trajectory FTLE:**
```bash
python analytical_derivations.py

```


2. **Run all numerical experiments and generate CSV outputs and figures:**
```bash
python main_experiments.py

```



---

## 4. Repository Structure & Output Files

* `analytical_derivations.py`: Closed-form score, Jacobian, critical time $t_c$, and analytical $\Lambda(0)$ evaluation.
* `main_experiments.py`: Full numerical suite generating all CSV tables in `results/` and figures in `plots/`.
* `results/run_metadata.json`: Experiment configuration, parameters, and execution metadata.
* `results/reference_1d_refinement.csv`: 1D RK4 terminal-state self-convergence differences (Table I).
* `results/reference_2d_refinement.csv`: 2D RK4 terminal-state self-convergence differences (Table II).
* `results/ftle_1d_resolution.csv` & `plots/ftle_1d_resolution.png`: 1D reverse-time FTLE resolution profiles (Fig. 1).
* `results/ftle_1d_finite_difference.csv`: Variational vs. centered finite-difference relative error validation.
* `results/ftle_2d_resolution.csv` & `plots/ftle_2d_heatmap.png`: 2D maximal-singular-value FTLE statistics and heatmap (Fig. 2).
* `results/solver_comparison_1d.csv`, `plots/solver_comparison_1d.png`, & `plots/failure_rate_1d.png`: 1D matched-NFE solver comparison and nonfinite trajectory check (Table III, Fig. 3, and Fig. 4).
* `results/solver_comparison_2d.csv` & `plots/solver_comparison_2d.png`: 2D matched-NFE solver comparison (Table IV and Fig. 5).

---

## 5. Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{roudro2026lyapunov,
  author    = {Roudro, M Taufiqul Islam and Zihad, Mahmudul Hasan and Ameen, Mohd Ruhul},
  title     = {Finite-Time Lyapunov Analysis of Probability Flow ODEs in Bimodal Score-Based Diffusion},
  booktitle = {2026 International Conference on Recent Progresses in Science, Engineering and Technology (ICRPSET)},
  year      = {2026}
}

```
