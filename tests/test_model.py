"""Sanity checks. Run with:  python -m pytest   (or  python tests/test_model.py)"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calcium.data import make_dataset  # noqa: E402
from calcium.model import LI_RINZEL_PARAMS, simulate  # noqa: E402
from fit_hybrid import SercaNet  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_numpy_and_torch_agree():
    """The same solver code gives the same trajectory on both backends."""
    ip3 = np.array([0.4, 0.6])
    c_np = simulate(LI_RINZEL_PARAMS, ip3, np.full(2, 0.1), np.full(2, 0.7), 0.1, 200, np.stack)
    c_t = simulate(LI_RINZEL_PARAMS, torch.tensor(ip3.tolist()), torch.full((2,), 0.1),
                   torch.full((2,), 0.7), 0.1, 200, torch.stack)
    assert np.allclose(c_np, np.array(c_t.tolist()), atol=1e-12)


def test_autograd_through_solver():
    """Autograd gradients through RK4 match finite differences."""
    def trace_from(log_v1):
        p = dict(LI_RINZEL_PARAMS, v1=torch.exp(log_v1))
        c = simulate(p, torch.tensor([0.5]), torch.tensor([0.1]), torch.tensor([0.7]),
                     0.1, 50, torch.stack)
        return c.sum()

    log_v1 = torch.log(torch.tensor(6.0)).requires_grad_(True)
    assert torch.autograd.gradcheck(trace_from, (log_v1,))


def test_synthetic_data_oscillates():
    ds = make_dataset()
    late = ds["clean"][ds["t"] > 10]
    assert (late.max(axis=0) - late.min(axis=0) > 0.15).all()


def test_serca_net_constraints():
    net = SercaNet()
    c = torch.linspace(0.0, 1.0, 50)
    j = net(c)
    assert j[0].abs().item() == 0.0
    assert (j >= 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
