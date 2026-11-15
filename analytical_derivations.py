"""Analytical score and Jacobian functions for the Gaussian-mixture PF-ODE.

The mixture is p_t(x) = 0.5 N(x; -mu, q(t)) + 0.5 N(x; +mu, q(t))
in 1D, with the isotropic 2D analogue used in the extension experiment.
"""
import numpy as np


def variance(t, s0=0.2, beta_min=0.1, beta_max=20.0):
    """Marginal variance q(t) for the variance-exploding schedule."""
    return s0**2 + beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def diffusion_squared(t, beta_min=0.1, beta_max=20.0):
    """g(t)^2."""
    return beta_min + (beta_max - beta_min) * t


def _sech2(z):
    """Numerically stable sech(z)^2."""
    z = np.asarray(z)
    a = np.exp(-2.0 * np.abs(z))
    return 4.0 * a / (1.0 + a) ** 2


def score_mixture_1d(x, t, mu=2.0, s0=0.2,
                     beta_min=0.1, beta_max=20.0):
    q = variance(t, s0, beta_min, beta_max)
    z = np.clip(mu * np.asarray(x) / q, -40.0, 40.0)
    return -np.asarray(x) / q + (mu / q) * np.tanh(z)


def jacobian_forward_1d(x, t, mu=2.0, s0=0.2,
                        beta_min=0.1, beta_max=20.0):
    """d v(x,t)/dx for the forward probability-flow ODE."""
    q = variance(t, s0, beta_min, beta_max)
    g2 = diffusion_squared(t, beta_min, beta_max)
    z = np.clip(mu * np.asarray(x) / q, -40.0, 40.0)
    ds_dx = -1.0 / q + (mu**2 / q**2) * _sech2(z)
    return -0.5 * g2 * ds_dx


def jacobian_reverse_1d(x, tau, T=1.0, mu=2.0, s0=0.2,
                        beta_min=0.1, beta_max=20.0):
    """d b(x,tau)/dx, where b(x,tau)=-v(x,T-tau)."""
    return -jacobian_forward_1d(
        x, T - tau, mu, s0, beta_min, beta_max
    )


def score_mixture_2d(x, t, mu_vec=(2.0, 0.0), s0=0.2,
                     beta_min=0.1, beta_max=20.0):
    """Score of an equal-weight, isotropic two-component 2D mixture."""
    x = np.asarray(x, dtype=float)
    m = np.asarray(mu_vec, dtype=float)
    q = variance(t, s0, beta_min, beta_max)
    z = np.clip(np.dot(m, x) / q, -40.0, 40.0)
    return -x / q + (m / q) * np.tanh(z)


def jacobian_forward_2d(x, t, mu_vec=(2.0, 0.0), s0=0.2,
                        beta_min=0.1, beta_max=20.0):
    """2x2 Jacobian of the forward probability-flow vector field."""
    x = np.asarray(x, dtype=float)
    m = np.asarray(mu_vec, dtype=float)
    q = variance(t, s0, beta_min, beta_max)
    g2 = diffusion_squared(t, beta_min, beta_max)
    z = np.clip(np.dot(m, x) / q, -40.0, 40.0)
    ds = -np.eye(2) / q + np.outer(m, m) * (_sech2(z) / q**2)
    return -0.5 * g2 * ds


def jacobian_reverse_2d(x, tau, T=1.0, mu_vec=(2.0, 0.0),
                        s0=0.2, beta_min=0.1, beta_max=20.0):
    return -jacobian_forward_2d(
        x, T - tau, mu_vec, s0, beta_min, beta_max
    )
