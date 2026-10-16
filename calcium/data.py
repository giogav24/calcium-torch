"""Synthetic calcium imaging data from the Li-Rinzel model.

Ground truth is known, so every fitted parameter and every learned flux
can be checked against the value that generated the data.
"""

import numpy as np

from .model import LI_RINZEL_PARAMS, simulate

DT = 0.1                    # s; RK4 step, also the sampling interval (10 Hz)
T_END = 40.0                # s; three to four oscillations per trace
IP3_LEVELS = (0.4, 0.5, 0.6)  # uM; one trace per IP3 stimulation level
C_INIT, H_INIT = 0.1, 0.7   # resting state, assumed known
NOISE_SD = 0.03             # uM; additive Gaussian measurement noise


def n_steps(t_window):
    return int(round(t_window / DT))


def make_dataset(seed=0):
    """Return time, IP3 levels, clean and noisy traces of shape (n_obs, n_traces)."""
    ip3 = np.array(IP3_LEVELS)
    batch = len(ip3)
    clean = simulate(
        LI_RINZEL_PARAMS, ip3, np.full(batch, C_INIT), np.full(batch, H_INIT),
        DT, n_steps(T_END), np.stack,
    )
    rng = np.random.default_rng(seed)
    noisy = clean + NOISE_SD * rng.standard_normal(clean.shape)
    t = np.arange(clean.shape[0]) * DT
    return {"t": t, "ip3": ip3, "clean": clean, "noisy": noisy}
