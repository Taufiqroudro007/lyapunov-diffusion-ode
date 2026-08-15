import numpy as np

def score_mixture_1d(x, t, mu=2.0, s0=0.2, beta_min=0.1, beta_max=20.0):
    s2 = s0**2 + beta_min * t + 0.5 * (beta_max - beta_min) * (t**2)
    arg = np.clip(mu * x / s2, -40.0, 40.0)
    return -x / s2 + (mu / s2) * np.tanh(arg)

def jacobian_mixture_1d(x, t, mu=2.0, s0=0.2, beta_min=0.1, beta_max=20.0):
    s2 = s0**2 + beta_min * t + 0.5 * (beta_max - beta_min) * (t**2)
    g2 = beta_min + (beta_max - beta_min) * t
    arg = np.clip(mu * x / s2, -40.0, 40.0)
    sech2 = 1.0 / (np.cosh(arg)**2)
    return 0.5 * g2 * (-1.0 / s2 + (mu**2 / (s2**2)) * sech2)
