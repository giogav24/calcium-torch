"""Part 1: estimate Li-Rinzel parameters by backpropagating through the ODE solver.

Synthetic noisy traces at three IP3 levels are generated with known
parameters. Starting from a deliberately wrong guess, three kinetic
parameters (v1, v3, a2) are fitted with Adam, using gradients that PyTorch
autograd computes through every RK4 step of the simulation.

Run:  python fit_parameters.py          (about 2-5 minutes on a laptop CPU)
      python fit_parameters.py --quick  (short smoke test)
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from calcium.data import C_INIT, DT, H_INIT, NOISE_SD, make_dataset, n_steps
from calcium.model import LI_RINZEL_PARAMS, simulate

torch.set_default_dtype(torch.float64)

FITTED = ["v1", "v3", "a2"]
INITIAL_GUESS = {"v1": 3.6, "v3": 0.54, "a2": 0.32}   # 40-60% away from the truth
# (time window in s, Adam iterations): the window grows so the fit locks on
# to the first oscillations before it has to match the later ones.
SCHEDULE = [(10, 60), (20, 60), (30, 60), (40, 80)]
QUICK_SCHEDULE = [(10, 15), (40, 15)]
FIG_DIR = Path(__file__).resolve().parent / "figures"


def to_numpy(x):
    """Tensor -> array without torch's NumPy bridge (works with any NumPy version)."""
    return np.array(x.tolist())


def run_model(log_theta, ip3, window):
    """Simulate with the fitted parameters; positivity via log-parameterisation."""
    p = dict(LI_RINZEL_PARAMS)
    for name, value in zip(FITTED, torch.exp(log_theta)):
        p[name] = value
    batch = ip3.shape[0]
    c_init = torch.full((batch,), C_INIT)
    h_init = torch.full((batch,), H_INIT)
    return simulate(p, ip3, c_init, h_init, DT, n_steps(window), torch.stack)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short smoke test")
    args = parser.parse_args()

    ds = make_dataset(seed=0)
    t = ds["t"]
    ip3 = torch.tensor(ds["ip3"].tolist())
    data = torch.tensor(ds["noisy"].tolist())

    log_theta = torch.log(torch.tensor([INITIAL_GUESS[n] for n in FITTED]))
    log_theta.requires_grad_(True)
    optimizer = torch.optim.Adam([log_theta], lr=0.05)

    history = {"loss": [], "params": []}
    start = time.time()
    for window, iterations in QUICK_SCHEDULE if args.quick else SCHEDULE:
        target = data[: n_steps(window) + 1]
        for _ in range(iterations):
            optimizer.zero_grad()
            c = run_model(log_theta, ip3, window)
            loss = ((c - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            history["loss"].append(loss.item())
            history["params"].append(torch.exp(log_theta).tolist())
        print(f"window {window:>2} s  loss {loss.item():.6f}  "
              f"({time.time() - start:.0f} s elapsed)")

    fitted = torch.exp(log_theta).detach()
    print(f"\nnoise floor (sigma^2) = {NOISE_SD**2:.6f}")
    print(f"{'param':<6}{'initial':>10}{'fitted':>10}{'true':>10}{'error':>10}")
    for name, value in zip(FITTED, fitted.tolist()):
        true = LI_RINZEL_PARAMS[name]
        print(f"{name:<6}{INITIAL_GUESS[name]:>10.3f}{value:>10.4f}{true:>10.3f}"
              f"{100 * (value / true - 1):>+9.2f}%")

    with torch.no_grad():
        c_fit = to_numpy(run_model(log_theta, ip3, t[-1]))
        c_init_guess = to_numpy(run_model(
            torch.log(torch.tensor([INITIAL_GUESS[n] for n in FITTED])), ip3, t[-1]
        ))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    trace_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    for k, ax in enumerate(trace_axes):
        ax.plot(t, ds["noisy"][:, k], ".", ms=2.5, color="0.6", label="noisy data")
        ax.plot(t, c_init_guess[:, k], ":", color="tab:orange", label="initial guess")
        ax.plot(t, c_fit[:, k], color="tab:blue", lw=1.6, label="fitted model")
        ax.set_title(f"IP$_3$ = {ds['ip3'][k]:.1f} µM")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("cytosolic Ca$^{2+}$ (µM)")
    trace_axes[0].legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    params = np.array(history["params"])
    for j, name in enumerate(FITTED):
        ax.plot(params[:, j] / LI_RINZEL_PARAMS[name], label=name)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("fitted / true value")
    ax.set_title("Parameter convergence")
    ax.legend(fontsize=8)

    fig.suptitle("Li-Rinzel parameter estimation by backpropagation through RK4")
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "part1_parameter_fit.png", dpi=150)
    print("\nsaved figures/part1_parameter_fit.png")


if __name__ == "__main__":
    main()
