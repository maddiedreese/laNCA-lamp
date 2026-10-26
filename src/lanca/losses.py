"""Differentiable losses for training.

Matching frames alone rewards blurry averages: when the model can't know exactly where a blob
will be, the safest guess is a smear. So training also matches behavior statistics that a
smear gets wrong, computed from batches of frames (T, B, H, W):

- sharpness: mean w(1 - w) over the vessel. Crisp blobs have little; smears have a lot.
- edges: outline per unit of wax (total variation / wax). Round blobs have little; speckle
  and porous blobs have a lot. Needed because speckle is perfectly sharp, so sharpness alone
  rewards it (measured: real 0.13, a model trained without this term 0.62).
- spectrum: radially binned power spectrum of the wax field, a proxy for blob sizes.
- activity: mean |w(t+1) - w(t)| per step, a proxy for how fast wax moves.
- profile: time-averaged wax per row, where in the lamp the wax spends its time.
"""

import itertools

import torch

Tensor = torch.Tensor


def masked_mse(pred: Tensor, target: Tensor, vessel: Tensor) -> Tensor:
    """Mean squared error over vessel cells, averaged over time and batch."""
    t, b = pred.shape[:2]
    return (((pred - target) ** 2) * vessel).sum() / (vessel.sum() * t * b)


def sharpness(w: Tensor, vessel: Tensor) -> Tensor:
    return (w * (1 - w) * vessel).sum((-2, -1)).mean() / vessel.sum()


def edges(w: Tensor, vessel: Tensor) -> Tensor:
    """Total variation inside the vessel per unit of wax."""
    gx = (w[..., :, 1:] - w[..., :, :-1]).abs() * (vessel[:, 1:] * vessel[:, :-1])
    gy = (w[..., 1:, :] - w[..., :-1, :]).abs() * (vessel[1:, :] * vessel[:-1, :])
    return (gx.sum() + gy.sum()) / ((w * vessel).sum() + 1e-6)


def activity(w: Tensor, vessel: Tensor) -> Tensor:
    return ((w[1:] - w[:-1]).abs() * vessel).sum((-2, -1)).mean() / vessel.sum()


def profile(w: Tensor, vessel: Tensor) -> Tensor:
    """Wax fraction per row, averaged over time and batch (H,)."""
    return (w * vessel).mean(dim=tuple(range(w.dim() - 2))).sum(-1) / vessel.sum(-1).clamp(min=1)


class Spectrum:
    """Radially binned log power spectrum of the mean-removed wax field inside the vessel."""

    def __init__(self, shape: tuple[int, int], bins: int = 12, device=None):
        # Built on the CPU (some of these ops are missing on Apple's GPU), then moved.
        h, w = shape
        fy = torch.fft.fftfreq(h)[:, None]
        fx = torch.fft.rfftfreq(w)[None, :]
        radius = torch.sqrt(fx**2 + fy**2) / 0.5  # 0 at DC, ~1 at Nyquist
        edges = torch.logspace(-2, 0.15, bins + 1)
        onehot = torch.stack(
            [(radius >= lo) & (radius < hi) for lo, hi in itertools.pairwise(edges)]
        ).float()
        self.onehot = onehot.to(device)
        self.counts = onehot.sum((1, 2)).clamp(min=1).to(device)

    def to(self, device) -> "Spectrum":
        self.onehot, self.counts = self.onehot.to(device), self.counts.to(device)
        return self

    def __call__(self, w: Tensor, vessel: Tensor) -> Tensor:
        flat = w.reshape(-1, *w.shape[-2:])
        mean = (flat * vessel).sum((-2, -1), keepdim=True) / vessel.sum()
        power = torch.fft.rfft2((flat - mean) * vessel).abs() ** 2
        binned = torch.einsum("nhw,khw->nk", power, self.onehot.to(power.dtype)) / self.counts
        return torch.log(binned.mean(0) + 1e-6)


def behavior(w: Tensor, vessel: Tensor, spectrum: Spectrum) -> dict[str, Tensor]:
    return {
        "sharpness": sharpness(w, vessel),
        "edges": edges(w, vessel),
        "activity": activity(w, vessel),
        "profile": profile(w, vessel),
        "spectrum": spectrum(w, vessel),
    }


# Chosen so each term's typical gap between a simulation and the footage contributes ~0.1
# (measured gaps: edges ~0.4, activity ~0.025, spectrum ~1 in log power per bin).
WEIGHTS = {"sharpness": 20.0, "edges": 1.0, "activity": 200.0, "profile": 2.0, "spectrum": 0.2}


def behavior_loss(
    pred: dict[str, Tensor], target: dict[str, Tensor]
) -> tuple[Tensor, dict[str, float]]:
    """Weighted squared differences between two behavior summaries. Returns total and parts."""
    parts = {k: WEIGHTS[k] * ((pred[k] - target[k]) ** 2).mean() for k in WEIGHTS}
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
