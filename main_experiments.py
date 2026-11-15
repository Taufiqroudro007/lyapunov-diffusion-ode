"""Reproducible experiments for finite-time Lyapunov analysis of PF-ODEs.

Run:
    python main_experiment.py

Outputs are written to ./results and ./plots. The script reports actual
numerical values; no numerical values are hard-coded into the manuscript.

Conventions:
  * t is the forward diffusion time, from 0 to T.
  * tau = T - t is reverse integration time, from 0 to T-epsilon.
  * x_T denotes the state at forward time T (the initial state for reverse
    integration). It is deliberately not called x_0.
  * b(x,tau) = -v(x,T-tau), hence J_reverse = -J_forward.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytical_derivations import (
    variance, diffusion_squared, score_mixture_1d,
    jacobian_forward_1d, jacobian_reverse_1d,
    score_mixture_2d, jacobian_reverse_2d,
)

# ------------------------------- Configuration ----------------------------
MU = 2.0
S0 = 0.2
BETA_MIN = 0.1
BETA_MAX = 20.0
T = 1.0
EPSILON = 1.0e-4
L = T - EPSILON
SEED = 20261115

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)


def b1(x, tau):
    t = T - tau
    return 0.5 * diffusion_squared(t, BETA_MIN, BETA_MAX) * score_mixture_1d(
        x, t, MU, S0, BETA_MIN, BETA_MAX
    )


def j1(x, tau):
    return jacobian_reverse_1d(
        x, tau, T, MU, S0, BETA_MIN, BETA_MAX
    )


def b2(x, tau):
    t = T - tau
    return 0.5 * diffusion_squared(t, BETA_MIN, BETA_MAX) * score_mixture_2d(
        x, t, (MU, 0.0), S0, BETA_MIN, BETA_MAX
    )


def j2(x, tau):
    return jacobian_reverse_2d(
        x, tau, T, (MU, 0.0), S0, BETA_MIN, BETA_MAX
    )


# ------------------------------- ODE solvers -------------------------------
def integrate(rhs, y0, n_steps, method="rk4", t_end=L):
    """Integrate y'=rhs(y,t), t in [0,t_end], using a fixed-step method."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    y = np.array(y0, dtype=float, copy=True)
    h = t_end / n_steps
    t = 0.0
    for _ in range(n_steps):
        if method == "euler":
            y = y + h * rhs(y, t)
        elif method == "heun":
            k1 = rhs(y, t)
            k2 = rhs(y + h * k1, t + h)
            y = y + 0.5 * h * (k1 + k2)
        elif method == "rk4":
            k1 = rhs(y, t)
            k2 = rhs(y + 0.5 * h * k1, t + 0.5 * h)
            k3 = rhs(y + 0.5 * h * k2, t + 0.5 * h)
            k4 = rhs(y + h * k3, t + h)
            y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("method must be euler, heun, or rk4")
        t += h
        if not np.all(np.isfinite(y)):
            return np.full_like(y, np.nan)
    return y


def solve_scalar(xT, n_steps, method="rk4"):
    return float(integrate(lambda x, tau: b1(x, tau), xT, n_steps, method))


def solve_vector(xT, n_steps, method="rk4"):
    return integrate(lambda x, tau: b2(x, tau), xT, n_steps, method)


def reference_1d(x_values, n_steps):
    return np.array([solve_scalar(x, n_steps, "rk4") for x in x_values])


def reference_2d(x_values, n_steps):
    return np.array([solve_vector(x, n_steps, "rk4") for x in x_values])


def error_metrics(approx, reference):
    valid = np.isfinite(approx) & np.isfinite(reference)
    failures = int(np.size(approx) - np.count_nonzero(valid))
    if not np.any(valid):
        return {"mae": np.nan, "rmse": np.nan, "max_abs": np.nan,
                "failure_count": failures, "failure_rate": failures / np.size(approx)}
    err = np.abs(np.asarray(approx)[valid] - np.asarray(reference)[valid])
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(err)),
        "failure_count": failures,
        "failure_rate": float(failures / np.size(approx)),
    }



def write_csv(path, rows):
    if not rows:
        return

    # Collect every column appearing in any row
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)



# ----------------------------- 1D FTLE analysis ---------------------------
def augmented_rhs_1d(y, tau):
    x, ell = y
    return np.array([b1(x, tau), j1(x, tau)])


def compute_ftle_1d(xT, n_steps):
    y_end = integrate(augmented_rhs_1d, np.array([xT, 0.0]), n_steps, "rk4")
    return float(y_end[1] / L), float(y_end[0])


def run_1d_ftle():
    # 25 terminal states, evenly distributed over the stated interval.
    xT_values = np.linspace(-3.0, 3.0, 25)
    resolutions = [600, 1200, 2400, 4800]
    rows = []
    ftle_by_n = {}
    for n in resolutions:
        vals = np.array([compute_ftle_1d(x, n)[0] for x in xT_values])
        ftle_by_n[n] = vals
        rows.append({
            "steps": n,
            "initial_condition_count": len(xT_values),
            "ftle_min": float(np.nanmin(vals)),
            "ftle_max": float(np.nanmax(vals)),
            "ftle_mean": float(np.nanmean(vals)),
        })
    for k in range(1, len(resolutions)):
        n0, n1 = resolutions[k-1], resolutions[k]
        rows[k]["max_abs_change_from_previous"] = float(
            np.nanmax(np.abs(ftle_by_n[n1] - ftle_by_n[n0]))
        )
    write_csv(RESULTS / "ftle_1d_resolution.csv", rows)

    # Independent finite-difference check of the derivative of the flow map.
    n_ref = resolutions[-1]
    deltas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
    check_x = [-2.0, -1.0, -0.25, 0.25, 1.0, 2.0]
    fd_rows = []
    for xT in check_x:
        ftle, _ = compute_ftle_1d(xT, n_ref)
        variational_derivative = np.exp(ftle * L)
        for delta in deltas:
            xp = solve_scalar(xT + delta, n_ref, "rk4")
            xm = solve_scalar(xT - delta, n_ref, "rk4")
            fd = (xp - xm) / (2.0 * delta)
            fd_rows.append({
                "x_T": xT, "delta": delta,
                "variational_d_flow": variational_derivative,
                "finite_difference_d_flow": fd,
                "absolute_difference": abs(fd - variational_derivative),
                "relative_difference": abs(fd - variational_derivative) /
                    max(abs(variational_derivative), 1e-14),
            })
    write_csv(RESULTS / "ftle_1d_finite_difference.csv", fd_rows)

    plt.figure(figsize=(7, 4.5))
    for n in resolutions:
        plt.plot(xT_values, ftle_by_n[n], marker="o", ms=3, label=f"N={n}")
    plt.xlabel(r"Reverse-time initial state $x_T$")
    plt.ylabel("Finite-time Lyapunov exponent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "ftle_1d_resolution.png", dpi=220)
    plt.close()


# ----------------------- Reference refinement / 1D solver test -------------
def run_1d_solver_comparison():
    # Same set of reverse-time initial states for all solvers.
    xT_all = np.linspace(-2.5, 2.5, 60)
    xT_values = xT_all[np.abs(xT_all) > 0.05]

    ref_steps = [1000, 2000, 4000, 8000, 16000]
    refs = [reference_1d(xT_values, n) for n in ref_steps]
    refinement_rows = []
    for i, n in enumerate(ref_steps):
        row = {"rk4_steps": n}
        if i > 0:
            d = np.abs(refs[i] - refs[i-1])
            row["max_abs_change_from_previous"] = float(np.nanmax(d))
            row["max_scaled_change"] = float(np.nanmax(
                d / (1.0 + np.abs(refs[i]))
            ))
        refinement_rows.append(row)
    write_csv(RESULTS / "reference_1d_refinement.csv", refinement_rows)
    reference = refs[-1]
    achieved = float(np.nanmax(
        np.abs(refs[-1] - refs[-2]) / (1.0 + np.abs(refs[-1]))
    ))
    print(f"1D reference refinement: max scaled change (8000 -> 16000) = {achieved:.3e}")
    if achieved > 1e-8:
        print("WARNING: the 1D reference did not meet the 1e-8 refinement target; "
              "increase the final RK4 reference resolution before reporting it as converged.")

    # Matched RHS-evaluation budgets: Euler=1, Heun=2, RK4=4 evaluations/step.
    budgets = [60, 120, 240, 480, 960]
    rows = []
    for budget in budgets:
        for method, cost in [("euler", 1), ("heun", 2), ("rk4", 4)]:
            n_steps = budget // cost
            approx = np.array([solve_scalar(x, n_steps, method) for x in xT_values])
            metrics = error_metrics(approx, reference)
            rows.append({
                "nfe_budget": budget, "method": method, "steps": n_steps,
                "initial_condition_count": len(xT_values),
                "error_population": "finite trajectories only; failures reported separately",
                **metrics,
            })
    write_csv(RESULTS / "solver_comparison_1d.csv", rows)

    plt.figure(figsize=(7, 4.5))
    for method in ("euler", "heun", "rk4"):
        subset = [r for r in rows if r["method"] == method]
        plt.loglog([r["nfe_budget"] for r in subset],
                   [max(r["rmse"], 1e-16) for r in subset],
                   marker="o", label=method.capitalize())
    plt.xlabel("RHS function-evaluation budget")
    plt.ylabel("RMSE against refined RK4 reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "solver_comparison_1d.png", dpi=220)
    plt.close()

    # Plot failure rate separately; a large error is not itself a numerical failure.
    plt.figure(figsize=(7, 4.5))
    for method in ("euler", "heun", "rk4"):
        subset = [r for r in rows if r["method"] == method]
        plt.plot([r["nfe_budget"] for r in subset],
                 [r["failure_rate"] for r in subset],
                 marker="o", label=method.capitalize())
    plt.xlabel("RHS function-evaluation budget")
    plt.ylabel("Non-finite trajectory fraction")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "failure_rate_1d.png", dpi=220)
    plt.close()


# ------------------------------ 2D experiment ------------------------------
def augmented_rhs_2d(y, tau):
    x = y[:2]
    A = y[2:].reshape(2, 2)
    dx = b2(x, tau)
    dA = j2(x, tau) @ A
    return np.concatenate([dx, dA.reshape(-1)])


def compute_ftle_2d(xT, n_steps):
    y0 = np.concatenate([np.asarray(xT, dtype=float), np.eye(2).reshape(-1)])
    y_end = integrate(augmented_rhs_2d, y0, n_steps, "rk4")
    A = y_end[2:].reshape(2, 2)
    singular_values = np.linalg.svd(A, compute_uv=False)
    return float(np.log(max(singular_values[0], 1e-300)) / L)


def run_2d_experiment():
    # A deterministic 11x11 grid. This is a synthetic 2D extension, not a
    # learned score model or an empirical high-dimensional benchmark.
    axis = np.linspace(-4.0, 4.0, 11)
    xT_values = np.array([[x, y] for x in axis for y in axis])
    ftle_rows = []
    ftle_values = {}
    for n in [600, 1200, 2400, 4800, 9600]:
        vals = np.array([compute_ftle_2d(x, n) for x in xT_values])
        ftle_values[n] = vals
        ftle_rows.append({
            "rk4_steps": n, "initial_condition_count": len(xT_values),
            "ftle_max_minimum_singular_direction_min": float(np.min(vals)),
            "max_ftle": float(np.max(vals)),
            "mean_max_ftle": float(np.mean(vals)),
        })
    ftle_rows[-1]["max_abs_change_from_previous"] = float(
        np.max(np.abs(ftle_values[2400] - ftle_values[1200]))
    )
    write_csv(RESULTS / "ftle_2d_resolution.csv", ftle_rows)

    # Reference refinement and matched-NFE Euler/Heun/RK4 comparison in 2D.
    ref_steps = [2000, 4000, 8000]
    refs = [reference_2d(xT_values, n) for n in ref_steps]
    ref_rows = []
    for i, n in enumerate(ref_steps):
        row = {"rk4_steps": n}
        if i:
            d = np.linalg.norm(refs[i] - refs[i-1], axis=1)
            row["max_abs_change_from_previous"] = float(np.max(d))
            row["max_scaled_change"] = float(np.max(
                d / (1.0 + np.linalg.norm(refs[i], axis=1))
            ))
        ref_rows.append(row)
    write_csv(RESULTS / "reference_2d_refinement.csv", ref_rows)
    reference = refs[-1]
    achieved = float(np.max(
        np.linalg.norm(refs[-1] - refs[-2], axis=1) /
        (1.0 + np.linalg.norm(refs[-1], axis=1))
    ))
    print(f"2D reference refinement: max scaled change (4000 -> 8000) = {achieved:.3e}")
    if achieved > 1e-8:
        print("NOTE: the 2D reference did not meet 1e-8; report the achieved "
              "refinement value and increase reference resolution if needed.")

    budgets = [60, 120, 240, 480]
    rows = []
    for budget in budgets:
        for method, cost in [("euler", 1), ("heun", 2), ("rk4", 4)]:
            n_steps = budget // cost
            approx = np.array([solve_vector(x, n_steps, method) for x in xT_values])
            finite = np.all(np.isfinite(approx), axis=1) & np.all(np.isfinite(reference), axis=1)
            failures = int(len(xT_values) - np.count_nonzero(finite))
            if np.any(finite):
                err = np.linalg.norm(approx[finite] - reference[finite], axis=1)
                mae, rmse, maxerr = float(np.mean(err)), float(np.sqrt(np.mean(err**2))), float(np.max(err))
            else:
                mae = rmse = maxerr = float("nan")
            rows.append({
                "nfe_budget": budget, "method": method, "steps": n_steps,
                "initial_condition_count": len(xT_values),
                "mae_terminal_euclidean_error": mae,
                "rmse_terminal_euclidean_error": rmse,
                "max_terminal_euclidean_error": maxerr,
                "failure_count": failures,
                "failure_rate": failures / len(xT_values),
                "error_population": "finite trajectories only; failures reported separately",
            })
    write_csv(RESULTS / "solver_comparison_2d.csv", rows)

    # Plot maximum FTLE over the grid as a spatial field.
    vals = ftle_values[2400].reshape(len(axis), len(axis))
    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(vals, origin="lower", extent=[-4, 4, -4, 4],
                    aspect="equal")
    plt.colorbar(im, label="Maximum finite-time Lyapunov exponent")
    plt.xlabel(r"$x_{T,1}$")
    plt.ylabel(r"$x_{T,2}$")
    plt.tight_layout()
    plt.savefig(PLOTS / "ftle_2d_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for method in ("euler", "heun", "rk4"):
        subset = [r for r in rows if r["method"] == method]
        plt.loglog([r["nfe_budget"] for r in subset],
                   [max(r["rmse_terminal_euclidean_error"], 1e-16) for r in subset],
                   marker="o", label=method.capitalize())
    plt.xlabel("RHS function-evaluation budget")
    plt.ylabel("RMSE of terminal Euclidean error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "solver_comparison_2d.png", dpi=220)
    plt.close()


def write_run_metadata():
    metadata = {
        "model": "equal-weight symmetric Gaussian mixture",
        "mu_1d": MU, "mu_2d": [MU, 0.0],
        "initial_component_standard_deviation": S0,
        "beta_min": BETA_MIN, "beta_max": BETA_MAX,
        "T": T, "epsilon": EPSILON, "reverse_integration_interval": [0.0, L],
        "reverse_time_definition": "tau = T - t",
        "reverse_vector_field": "b(x,tau) = -v(x,T-tau)",
        "jacobian_sign": "J_reverse = -J_forward",
        "random_seed": SEED,
        "1d_ftle_initial_conditions": "25 equally spaced x_T values in [-3,3]",
        "1d_solver_initial_conditions": "60 equally spaced values in [-2.5,2.5], excluding |x_T|<=0.05",
        "2d_initial_conditions": "11x11 Cartesian grid over [-4,4]^2",
        "failure_definition": "a trajectory is failed only if a computed state becomes non-finite",
        "error_definition": "errors are computed over finite trajectories; failure count/rate reported separately",
        "solver_cost": "matched RHS evaluations: Euler 1, Heun 2, classical RK4 4 per step",
        "reference_note": "refinement differences are reported; convergence must be checked from generated CSV files",
    }
    with open(RESULTS / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    np.random.seed(SEED)
    write_run_metadata()
    run_1d_ftle()
    run_1d_solver_comparison()
    run_2d_experiment()
    print("\nCompleted. Inspect CSV files in results/ and figures in plots/.")
    print("Do not describe a reference as converged unless its refinement values "
          "meet the tolerance you state in the manuscript.")


if __name__ == "__main__":
    main()
