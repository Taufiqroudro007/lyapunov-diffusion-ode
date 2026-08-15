import numpy as np
import matplotlib.pyplot as plt
import os
from analytical_derivations import score_mixture_1d, jacobian_mixture_1d

os.makedirs("plots", exist_ok=True)

mu = 2.0
s0 = 0.2
beta_min = 0.1
beta_max = 20.0
T = 1.0

def g_sq(t):
    return beta_min + (beta_max - beta_min) * t

def reverse_drift(x, t):
    score = score_mixture_1d(x, t, mu=mu, s0=s0, beta_min=beta_min, beta_max=beta_max)
    return -0.5 * g_sq(t) * score

def rk4_step(x, t, dt):
    k1 = reverse_drift(x, t)
    k2 = reverse_drift(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = reverse_drift(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = reverse_drift(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# 1. Phase Portrait & FTLE
steps = 600
time_grid = np.linspace(T, 1e-4, steps)
x0_vals = np.linspace(-3.0, 3.0, 25)
trajectories = np.zeros((len(x0_vals), steps))
ftle_values = []

for idx, x_init in enumerate(x0_vals):
    x = x_init
    trajectories[idx, 0] = x
    j_sum = 0.0
    for s in range(steps - 1):
        t = time_grid[s]
        dt = time_grid[s+1] - time_grid[s]
        x = rk4_step(x, t, dt)
        trajectories[idx, s+1] = x
        j_sum += jacobian_mixture_1d(x, t, mu=mu, s0=s0) * abs(dt)
    ftle_values.append(j_sum / T)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(len(x0_vals)):
    plt.plot(time_grid, trajectories[i], color='royalblue', alpha=0.6, lw=1.2)
plt.axhline(mu, color='red', linestyle='--', label=r'Target Modes $\pm\mu$')
plt.axhline(-mu, color='red', linestyle='--')
plt.axhline(0, color='black', linestyle=':', label='Separatrix ($x=0$)')
plt.gca().invert_xaxis()
plt.title("Trajectory Phase Portrait (Reverse PF-ODE)")
plt.xlabel("Diffusion Time $t$ ($T \\to 0$)")
plt.ylabel("State $x(t)$")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x0_vals, ftle_values, 'o-', color='darkgreen', lw=1.8, markersize=4)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Finite-Time Lyapunov Exponent $\lambda(x_T)$")
plt.xlabel("Initial Condition $x_T$")
plt.ylabel("Maximal FTLE $\lambda$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/fig1_phase_and_ftle.png", dpi=300)
plt.close()

# 2. Jacobian Heatmap
t_grid = np.linspace(1e-4, 1.0, 300)
x_grid = np.linspace(-3.0, 3.0, 300)
T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)
J_mesh = jacobian_mixture_1d(X_mesh, T_mesh, mu=mu, s0=s0)

plt.figure(figsize=(6.5, 4.5))
contour = plt.contourf(T_mesh, X_mesh, J_mesh, levels=50, cmap='coolwarm', vmin=-15, vmax=15)
cbar = plt.colorbar(contour)
cbar.set_label(r'Jacobian $J(x, t) = \nabla_x v(x, t)$')
plt.axhline(0, color='black', linestyle=':', label='Separatrix ($x=0$)')
plt.axhline(mu, color='black', linestyle='--', alpha=0.6, label=r'Target Modes $\pm\mu$')
plt.axhline(-mu, color='black', linestyle='--', alpha=0.6)
plt.gca().invert_xaxis()
plt.title("Spatial Jacobian Field $J(x, t)$")
plt.xlabel("Diffusion Time $t$ ($T \\to 0$)")
plt.ylabel("State $x$")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plots/fig2_jacobian_heatmap.png", dpi=300)
plt.close()

# 3. Step-Size Instability
x_inits = np.linspace(-2.5, 2.5, 60)
x_inits = x_inits[np.abs(x_inits) > 0.05]
ref_final = []
time_ref = np.linspace(T, 1e-4, 2000)
for x0 in x_inits:
    x = x0
    for s in range(len(time_ref) - 1):
        t = time_ref[s]
        dt = time_ref[s+1] - time_ref[s]
        x = rk4_step(x, t, dt)
    ref_final.append(x)
ref_final = np.array(ref_final)

step_counts = [15, 25, 35, 50, 75, 100, 150, 250, 500]
mean_errors = []
for steps_k in step_counts:
    time_k = np.linspace(T, 1e-4, steps_k)
    errors = []
    for idx, x0 in enumerate(x_inits):
        x = x0
        for s in range(steps_k - 1):
            t = time_k[s]
            dt = time_k[s+1] - time_k[s]
            x = x + dt * reverse_drift(x, t)
            if np.isnan(x) or abs(x) > 20.0:
                x = 20.0
                break
        errors.append(abs(x - ref_final[idx]))
    mean_errors.append(np.mean(errors))

plt.figure(figsize=(6.5, 4.5))
plt.plot(step_counts, mean_errors, 's-', color='#D62728', lw=2.2, markersize=7)
plt.axvline(35, color='black', linestyle='--', label=r'Stability Threshold ($N_{crit} \approx 35$)')
plt.yscale('log')
plt.title("Euler Solver Discretization Error vs. Steps ($N$)")
plt.xlabel("Number of Solver Steps ($N$)")
plt.ylabel("Mean Trajectory Error")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/fig3_stepsize_instability.png", dpi=300)
plt.close()
