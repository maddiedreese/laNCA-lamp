import torch
from test_physics import H, W, _vessel

from lanca import physics as ph
from lanca.model import LampNCA


def _setup(seed: int = 0, random_last: float = 0.0):
    torch.manual_seed(seed)
    geom = ph.Geometry(_vessel()).to("cpu", torch.float64)
    model = LampNCA().double()
    if random_last:
        last = model.net[-1]
        torch.nn.init.normal_(last.weight, std=random_last)
        torch.nn.init.normal_(last.bias, std=random_last)
    gen = torch.Generator().manual_seed(seed)
    wax = (torch.rand(2, H, W, generator=gen, dtype=torch.float64) > 0.6).double() * geom.vessel
    temp = torch.rand(2, H, W, generator=gen, dtype=torch.float64) * geom.vessel
    return geom, model, model.init_state(wax, temp)


def test_parameter_count_is_small():
    n = sum(p.numel() for p in LampNCA().network_parameters())
    assert 10_000 <= n <= 20_000


def test_untrained_model_is_exactly_the_physics():
    geom, model, state = _setup()
    p = {k: v.detach() for k, v in model.physics.values().items()}
    a, b = state, ph.State(wax=state.wax, temp=state.temp, flow=state.flow)
    with torch.no_grad():
        for _ in range(20):
            a = model.step(a, geom, 0.5)
            b = ph.step(b, geom, p, 0.5)
    assert (a.wax - b.wax).abs().max() < 1e-10
    assert (a.temp - b.temp).abs().max() < 1e-10


def test_any_network_output_conserves_wax_and_respects_walls():
    geom, model, state = _setup(seed=1, random_last=0.5)
    total = state.wax.sum((1, 2))
    with torch.no_grad():
        for _ in range(100):
            state = model.step(state, geom, 0.5)
    assert (state.wax.sum((1, 2)) - total).abs().max() < 1e-9 * total.max()
    assert state.wax.min() >= -1e-12 and state.wax.max() <= 1 + 1e-12
    assert (state.wax * (1 - geom.vessel)).abs().max() == 0
    assert (state.hidden * (1 - geom.vessel)).abs().max() == 0
    assert state.hidden.abs().max() <= 1


def test_gradients_reach_network_and_constants():
    geom, model, state = _setup(seed=2, random_last=0.1)
    target = torch.rand_like(state.wax) * geom.vessel
    _, waxes = model.rollout(state, geom, 5, 0.5, torch.Generator().manual_seed(0))
    ((waxes[-1] - target) ** 2).mean().backward()
    for name, param in model.net.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name
        assert torch.isfinite(param.grad).all(), name
    grads = dict(zip(model.physics.names, model.physics.raw.grad))
    # Constants used after the flow solve get gradients even without flow_grad, including
    # tension through its last substeps...
    assert grads["compress"] != 0 and grads["heat_rate"] != 0 and grads["tension"] != 0
    # ...but the ones that only shape the flow need flow_grad.
    assert grads["mobility"] == 0

    model.zero_grad()
    _, waxes = model.rollout(state, geom, 5, 0.5, torch.Generator().manual_seed(0), flow_grad=True)
    ((waxes[-1] - target) ** 2).mean().backward()
    grads = dict(zip(model.physics.names, model.physics.raw.grad))
    assert grads["mobility"] != 0 and grads["visc_wax"] != 0


def test_checkpointed_rollout_matches_plain_rollout():
    geom, model, state = _setup(seed=3, random_last=0.1)
    target = torch.rand_like(state.wax) * geom.vessel
    results = []
    for every in (0, 3):
        model.zero_grad()
        _, waxes = model.rollout(
            state, geom, 7, 0.5, torch.Generator().manual_seed(5), checkpoint_every=every
        )
        ((waxes - target) ** 2).mean().backward()
        grad = torch.cat([p.grad.flatten() for p in model.net.parameters()])
        results.append((waxes.detach(), grad))
    assert (results[0][0] - results[1][0]).abs().max() < 1e-12
    assert (results[0][1] - results[1][1]).abs().max() < 1e-10 * results[0][1].abs().max()


def test_randomness_is_reproducible_from_a_seed():
    geom, model, state = _setup(seed=4, random_last=0.3)
    with torch.no_grad():
        runs = [
            model.rollout(state, geom, 5, 0.5, torch.Generator().manual_seed(s))[0].hidden
            for s in (7, 7, 8)
        ]
    assert torch.equal(runs[0], runs[1])
    assert not torch.equal(runs[0], runs[2])
