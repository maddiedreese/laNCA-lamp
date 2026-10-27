import torch
import torch.nn.functional as F

from lanca import losses

H, W = 64, 32


def _blobs(n: int, size: int, t: int = 4) -> torch.Tensor:
    """t frames of n square blobs of the given size, moving down one row per frame."""
    gen = torch.Generator().manual_seed(n * 100 + size)
    frames = torch.zeros(t, 1, H, W)
    ys = torch.randint(0, H - size - t, (n,), generator=gen)
    xs = torch.randint(0, W - size, (n,), generator=gen)
    for i in range(t):
        for y, x in zip(ys, xs):
            frames[i, 0, y + i : y + i + size, x : x + size] = 1.0
    return frames


def test_spectrum_tells_big_blobs_from_small_ones():
    spec = losses.Spectrum((H, W))
    vessel = torch.ones(H, W)
    big, small = spec(_blobs(2, 12), vessel), spec(_blobs(30, 3), vessel)
    low, high = slice(0, 4), slice(-4, None)
    # Big blobs put relatively more power at low frequencies than small ones.
    assert (big[low].mean() - big[high].mean()) > (small[low].mean() - small[high].mean())


def test_sizes_tell_blobs_from_fragments_and_wall_sheets():
    vessel = torch.zeros(H, W)
    vessel[2:-2, 2:-2] = 1
    blob = torch.zeros(1, H, W)
    blob[0, 20:36, 8:24] = 1.0
    fragments = torch.zeros(1, H, W)
    fragments[0, 4:60:4, 4:28:4] = 1.0  # single-cell specks
    sheet = torch.zeros(1, H, W)
    sheet[0, 4:60, 2:4] = 1.0  # two cells thick, against the glass
    # Log of the share of wax removed at each radius: about nothing for a big blob, all of it
    # for specks and the sheet.
    lost = {
        k: losses.sizes(v, vessel).exp()
        for k, v in [("blob", blob), ("frag", fragments), ("sheet", sheet)]
    }
    assert lost["blob"][:3].max() < 0.01
    assert lost["frag"].min() > 0.99 and lost["sheet"].min() > 0.99
    w = (blob + 0.3 * fragments).requires_grad_()
    losses.sizes(w, vessel).sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0


def test_edges_penalize_speckle_but_not_round_blobs():
    vessel = torch.ones(H, W)
    blob = _blobs(1, 16)
    speckle = (torch.rand(blob.shape, generator=torch.Generator().manual_seed(0)) < 0.15).float()
    assert losses.edges(speckle, vessel) > 5 * losses.edges(blob, vessel)


def test_sharpness_and_activity_behave():
    vessel = torch.ones(H, W)
    crisp = _blobs(3, 8)
    smear = F.avg_pool2d(crisp, 7, 1, 3)
    assert losses.sharpness(smear, vessel) > losses.sharpness(crisp, vessel)
    still = crisp[:1].expand(4, -1, -1, -1)
    assert losses.activity(still, vessel) == 0
    assert losses.activity(crisp, vessel) > 0


def test_behavior_loss_is_zero_for_identical_and_differentiable():
    spec = losses.Spectrum((H, W))
    vessel = torch.ones(H, W)
    w = _blobs(4, 6).requires_grad_(True)
    target = {k: v.detach() for k, v in losses.behavior(_blobs(4, 6), vessel, spec).items()}
    total, _ = losses.behavior_loss(losses.behavior(w, vessel, spec), target)
    assert float(total) < 1e-8
    other = F.avg_pool2d(w, 5, 1, 2)
    total, parts = losses.behavior_loss(losses.behavior(other, vessel, spec), target)
    total.backward()
    assert float(total) > 0 and torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0
    assert set(parts) == set(losses.WEIGHTS)
