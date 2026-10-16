"""Li-Rinzel model of IP3-induced Ca2+ oscillations in a closed cell.

Every function here uses only arithmetic operators (+, -, *, /, **).
That means the same code runs on NumPy arrays and on PyTorch tensors,
and autograd flows through the whole RK4 solver when tensors are used.

Reference: Li & Rinzel (1994), J. Theor. Biol. 166:461-473.
"""

# Units: concentrations in uM, time in s.
LI_RINZEL_PARAMS = {
    "c0": 2.0,      # total free Ca2+ (cytosol + ER, cytosolic units), uM
    "c1": 0.185,    # ER / cytosol volume ratio
    "v1": 6.0,      # max IP3R channel flux rate, 1/s
    "v2": 0.11,     # ER leak rate, 1/s
    "v3": 0.9,      # max SERCA pump rate, uM/s
    "k3": 0.1,      # SERCA half-activation constant, uM
    "d1": 0.13,     # IP3 dissociation constant, uM
    "d2": 1.049,    # Ca2+ inactivation dissociation constant, uM
    "d3": 0.9434,   # IP3 dissociation constant, uM
    "d5": 0.08234,  # Ca2+ activation dissociation constant, uM
    "a2": 0.2,      # IP3R inactivation binding rate, 1/(uM s)
}


def serca_hill(c, p):
    """Mechanistic SERCA flux: Hill function with coefficient 2."""
    return p["v3"] * c**2 / (p["k3"] ** 2 + c**2)


def rhs(c, h, ip3, p, serca=serca_hill):
    """Right-hand side of the Li-Rinzel ODEs.

    c   : cytosolic Ca2+ (uM)
    h   : fraction of IP3 receptors not inactivated by Ca2+
    ip3 : IP3 concentration (uM), held constant
    serca(c, p) : SERCA flux; swap in a neural network for the hybrid model
    """
    m_inf = ip3 / (ip3 + p["d1"])
    n_inf = c / (c + p["d5"])
    c_er = (p["c0"] - c) / p["c1"]

    j_ipr = p["c1"] * p["v1"] * m_inf**3 * n_inf**3 * h**3 * (c_er - c)
    j_leak = p["c1"] * p["v2"] * (c_er - c)
    j_serca = serca(c, p)
    dc = j_ipr + j_leak - j_serca

    alpha_h = p["a2"] * p["d2"] * (ip3 + p["d1"]) / (ip3 + p["d3"])
    beta_h = p["a2"] * c
    dh = alpha_h * (1 - h) - beta_h * h
    return dc, dh


def simulate(p, ip3, c_init, h_init, dt, n_steps, stack, serca=serca_hill):
    """Integrate with classical RK4 and return c(t), shape (n_steps + 1, batch).

    `stack` is np.stack or torch.stack. ip3, c_init and h_init can be
    arrays/tensors of shape (batch,), so several IP3 levels run at once.
    """
    c, h = c_init, h_init
    trajectory = [c]
    for _ in range(n_steps):
        k1c, k1h = rhs(c, h, ip3, p, serca)
        k2c, k2h = rhs(c + 0.5 * dt * k1c, h + 0.5 * dt * k1h, ip3, p, serca)
        k3c, k3h = rhs(c + 0.5 * dt * k2c, h + 0.5 * dt * k2h, ip3, p, serca)
        k4c, k4h = rhs(c + dt * k3c, h + dt * k3h, ip3, p, serca)
        c = c + dt / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)
        h = h + dt / 6 * (k1h + 2 * k2h + 2 * k3h + k4h)
        trajectory.append(c)
    return stack(trajectory)
