"""Part 2: a hybrid mechanistic / neural model (universal differential equation).

The SERCA pump term of the Li-Rinzel model is removed and replaced by a
small neural network. Everything else stays mechanistic. The network is
trained end-to-end through the RK4 solver on traces at IP3 = 0.4 and
0.6 uM, then tested on IP3 = 0.5 uM, which it never sees in training.

Afterwards the learned flux is compared with the true Hill function that
generated the data: the question is whether the network recovers the
mechanism, not only whether it fits the traces.

Run:  python fit_hybrid.py          (about 5-15 minutes on a laptop CPU)
      python fit_hybrid.py --quick  (short smoke test, will not converge)
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from calcium.data import C_INIT, DT, H_INIT, NOISE_SD, make_dataset, n_steps
from calcium.model import LI_RINZEL_PARAMS, serca_hill, simulate

torch.set_default_dtype(torch.float64)

TRAIN, TEST = [0, 2], [1]      # indices into IP3_LEVELS = (0.4, 0.5, 0.6)
SCHEDULE = [(10, 200), (20, 200), (30, 200), (40, 200)]
QUICK_SCHEDULE = [(10, 20), (40, 20)]
FIG_DIR = Path(__file__).resolve().parent / "figures"


class SercaNet(nn.Module):
    """SERCA flux J(c) = c * softplus(MLP(c / c_scale)).

    The factor c and the softplus encode two physical constraints:
    no pumping without cytosolic Ca2+ (J(0) = 0) and no reverse flux
    (J >= 0). The shape of the flux in between is learned from data.
    """

    def __init__(self, hidden=16, c_scale=0.5):
        super().__init__()
        self.c_scale = c_scale
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, c):
        z = self.mlp((c / self.c_scale).unsqueeze(-1)).squeeze(-1)
        return c * F.softplus(z)


def to_numpy(x):
    """Tensor -> array without torch's NumPy bridge (works with any NumPy version)."""
    return np.array(x.tolist())


def run_hybrid(net, ip3, window):
    batch = ip3.shape[0]
    return simulate(
        LI_RINZEL_PARAMS, ip3, torch.full((batch,), C_INIT), torch.full((batch,), H_INIT),
        DT, n_steps(window), torch.stack, serca=lambda c, p: net(c),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short smoke test")
    parser.add_argument("--seed", type=int, default=0, help="network initialisation seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    ds = make_dataset(seed=0)
    t = ds["t"]
    ip3 = torch.tensor(ds["ip3"].tolist())
    data = torch.tensor(ds["noisy"].tolist())

    net = SercaNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    print(f"SercaNet: {sum(p.numel() for p in net.parameters())} trainable parameters")

    losses = []
    start = time.time()
    for window, iterations in QUICK_SCHEDULE if args.quick else SCHEDULE:
        target = data[: n_steps(window) + 1, TRAIN]
        for _ in range(iterations):
            optimizer.zero_grad()
            c = run_hybrid(net, ip3[TRAIN], window)
            loss = ((c - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"window {window:>2} s  loss {loss.item():.6f}  "
              f"({time.time() - start:.0f} s elapsed)")
    print(f"noise floor (sigma^2) = {NOISE_SD**2:.6f}")

    with torch.no_grad():
        c_all = to_numpy(run_hybrid(net, ip3, t[-1]))
        c_grid = torch.linspace(0.0, 0.7, 200)
        j_learned = to_numpy(net(c_grid))
    c_grid = to_numpy(c_grid)
    j_true = serca_hill(c_grid, LI_RINZEL_PARAMS)

    k = TEST[0]
    rmse_clean = np.sqrt(np.mean((c_all[:, k] - ds["clean"][:, k]) ** 2))
    c_lo, c_hi = ds["clean"][:, TRAIN].min(), ds["clean"][:, TRAIN].max()
    inside = (c_grid >= c_lo) & (c_grid <= c_hi)
    flux_err = np.max(np.abs(j_learned[inside] - j_true[inside]) / j_true[inside])
    print(f"held-out IP3 = {ds['ip3'][k]:.1f} uM: RMSE vs noise-free truth = {rmse_clean:.4f} uM")
    print(f"learned vs true SERCA flux, Ca2+ range seen in training "
          f"[{c_lo:.2f}, {c_hi:.2f}] uM: max relative error = {100 * flux_err:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, k in zip([axes[0, 0], axes[0, 1], axes[1, 0]], range(3)):
        role = "held-out test" if k in TEST else "training"
        ax.plot(t, ds["noisy"][:, k], ".", ms=2.5, color="0.6", label="noisy data")
        ax.plot(t, ds["clean"][:, k], "--", color="k", lw=0.8, label="ground truth")
        ax.plot(t, c_all[:, k], color="tab:green" if k in TEST else "tab:blue",
                lw=1.6, label="hybrid model")
        ax.set_title(f"IP$_3$ = {ds['ip3'][k]:.1f} µM ({role})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("cytosolic Ca$^{2+}$ (µM)")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    ax.axvspan(c_lo, c_hi, color="tab:blue", alpha=0.08, label="Ca$^{2+}$ range in training data")
    ax.plot(c_grid, j_true, "--", color="k", label="true Hill flux")
    ax.plot(c_grid, j_learned, color="tab:red", lw=1.6, label="learned flux")
    ax.set_xlabel("cytosolic Ca$^{2+}$ (µM)")
    ax.set_ylabel("SERCA flux (µM/s)")
    ax.set_title("Recovered mechanism")
    ax.legend(fontsize=8)

    fig.suptitle("Hybrid Li-Rinzel model with a neural SERCA term")
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "part2_hybrid_model.png", dpi=150)
    print("\nsaved figures/part2_hybrid_model.png")


if __name__ == "__main__":
    main()
