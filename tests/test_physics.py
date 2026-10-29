import torch
import torch.nn.functional as F

from lanca import physics as ph

H, W = 32, 16


def _vessel() -> torch.Tensor:
    """A small tapered glass, narrow at the top, with empty cells on both sides."""
    v = torch.zeros(H, W, dtype=torch.bool)
    for r in range(1, H - 1):
        half = 4 + 3 * r / (H - 1)
        v[r, round(W / 2 - half) : round(W / 2 + half)] = True
    return v


def _geom(dtype=torch.float64) -> ph.Geometry:
    return ph.Geometry(_vessel()).to("cpu", dtype)


def _params(**overrides) -> dict[str, torch.Tensor]:
    return {k: v.detach().double() for k, v in ph.Params(**overrides).values().items()}


def _divergence(fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    return (fx[:, :, 1:] - fx[:, :, :-1]) + (fy[:, 1:, :] - fy[:, :-1, :])


def _smooth_psi(gen: torch.Generator, scale: float, batch: int = 1) -> torch.Tensor:
    psi = torch.randn(batch, 1, H + 1, W + 1, generator=gen, dtype=torch.float64)
    psi = F.avg_pool2d(F.pad(psi, (2, 2, 2, 2), mode="replicate"), 5, 1)[:, 0]
    return psi * scale


def test_stream_flow_is_divergence_free_and_walls_closed():
    g = _geom()
    gen = torch.Generator().manual_seed(0)
    fx, fy = ph.faces_from_stream(_smooth_psi(gen, 5.0), g)
    assert _divergence(fx, fy).abs().max() < 1e-12
    assert (fx * (1 - g.open_x)).abs().max() == 0
    assert (fy * (1 - g.open_y)).abs().max() == 0
    assert fx.abs().max() > 0.1  # the flow is not trivially zero


def test_wax_is_conserved_exactly_over_100k_steps():
    g = _geom()
    gen = torch.Generator().manual_seed(1)
    wax = torch.rand(1, H, W, generator=gen, dtype=torch.float64) * g.vessel
    total = wax.sum()
    rate = torch.tensor(0.05, dtype=torch.float64)
    for i in range(100_000):
        if i % 100 == 0:  # a fresh random flow every 100 steps
            fx, fy = ph.faces_from_stream(_smooth_psi(gen, 0.3), g)
            n = ph.substeps(ph.outflow(fx, fy))
            fx, fy = fx / n, fy / n
        for _ in range(n):
            wax = ph.advect(wax, fx, fy)
        wax = ph.apply_exchange(wax, *ph.tension_faces(wax, g, rate, 1.0))
    assert abs(float(wax.sum() - total)) < 1e-9 * float(total)
    assert wax.min() >= -1e-12 and wax.max() <= 1 + 1e-12
    assert (wax * (1 - g.vessel)).abs().max() == 0


def test_bounds_hold_under_violent_flow():
    g = _geom()
    gen = torch.Generator().manual_seed(2)
    wax = (torch.rand(4, H, W, generator=gen, dtype=torch.float64) > 0.5).double() * g.vessel
    for _ in range(50):
        fx, fy = ph.faces_from_stream(_smooth_psi(gen, 40.0, batch=4), g)
        n = ph.substeps(ph.outflow(fx, fy))
        assert n > 1
        for _ in range(n):
            wax = ph.advect(wax, fx / n, fy / n)
    assert wax.min() >= -1e-12 and wax.max() <= 1 + 1e-12


def test_interface_compression_keeps_transported_blob_sharp():
    g = _geom()
    gen = torch.Generator().manual_seed(8)
    blob = torch.zeros(1, H, W, dtype=torch.float64)
    blob[0, 18:24, 5:11] = 1.0
    blob = blob * g.vessel
    plain, sharp = blob.clone(), blob.clone()
    strength = torch.tensor(2.0, dtype=torch.float64)
    for _ in range(40):
        fx, fy = ph.faces_from_stream(_smooth_psi(gen, 1.5), g)
        n = ph.substeps(ph.outflow(fx, fy))
        fx, fy = fx / n, fy / n
        for _ in range(n):
            plain = ph.advect(plain, fx, fy)
            sharp = ph.advect(sharp, fx, fy)
            sharp = ph.apply_exchange(sharp, *ph.compression_faces(sharp, g, fx, fy, strength))
    assert abs(float(sharp.sum() - blob.sum())) < 1e-9 * float(blob.sum())
    assert sharp.min() >= -1e-12 and sharp.max() <= 1 + 1e-12

    def fuzz(q):
        return float((q * (1 - q)).sum())

    # Chaotic test flows genuinely stretch the blob, so blur can't vanish; what matters is
    # that the edges stay much tighter and the core stays solid wax instead of dissolving.
    assert fuzz(sharp) < 0.6 * fuzz(plain)
    assert sharp.max() > 0.95 and plain.max() < 0.7


def test_full_step_conserves_wax_and_keeps_temperature_in_range():
    g = _geom()
    p = _params()
    gen = torch.Generator().manual_seed(3)
    wax = torch.rand(1, H, W, generator=gen, dtype=torch.float64) * g.vessel
    s = ph.State(wax=wax, temp=torch.rand(1, H, W, generator=gen, dtype=torch.float64) * g.vessel)
    total = wax.sum()
    for _ in range(2000):
        s = ph.step(s, g, p, dt=0.5)
    assert abs(float(s.wax.sum() - total)) < 1e-9 * float(total)
    assert s.temp.min() >= -1e-9 and s.temp.max() <= 1 + 1e-9


def test_compression_keeps_gradients_tame():
    # Nearly flat part-wax (tiny ripples on a soft blob) is where a differentiated unit normal
    # used to explode: about 120x the gradient without compression, after one step.
    g = _geom()
    gen = torch.Generator().manual_seed(11)
    s = _blob_state(g, 0.9, 0.3)
    readout = torch.randn(1, H, W, generator=gen, dtype=torch.float64)
    ripple = 1e-7 * torch.rand(1, H, W, generator=gen, dtype=torch.float64)
    norms = []
    for compress in (0.0, 2.0):
        p = _params(compress=compress, mobility=0.2)
        wax = ((0.5 * s.wax + 0.25 + ripple) * g.vessel).requires_grad_()
        out = ph.step(ph.State(wax=wax, temp=s.temp), g, p, dt=0.5)
        (grad,) = torch.autograd.grad((out.wax * readout).sum(), wax)
        norms.append(float(grad.norm()))
    assert norms[1] < 3 * norms[0]


def test_heat_without_wax_settles_warm_at_the_bottom():
    g = _geom()
    p = _params(tension=0.0)  # no wax, so skip the (costly) surface tension substeps
    s = ph.State(
        wax=torch.zeros(1, H, W, dtype=torch.float64),
        temp=torch.zeros(1, H, W, dtype=torch.float64),
    )
    temps = []
    for i in range(20_000):
        s = ph.step(s, g, p, dt=1.0)
        if i % 1000 == 999:
            temps.append(float((s.temp * g.vessel).sum() / g.vessel.sum()))
    assert 0 < temps[-1] < 1
    assert abs(temps[-1] - temps[-2]) < 0.01 * temps[-1]  # settled
    rows = (s.temp[0] * g.vessel).sum(1) / g.vessel.sum(1).clamp(min=1)
    assert rows[g.bottom] > rows[(g.top + g.bottom) // 2] > rows[g.top]


def _blob_height(wax: torch.Tensor, g: ph.Geometry) -> float:
    return float((wax * g.height).sum() / wax.sum())


def _blob_state(g: ph.Geometry, blob_temp: float, liquid_temp: float) -> ph.State:
    wax = torch.zeros(1, H, W, dtype=torch.float64)
    wax[0, 13:19, 6:10] = 1.0
    temp = torch.full((1, H, W), liquid_temp, dtype=torch.float64)
    temp[wax > 0] = blob_temp
    return ph.State(wax=wax * g.vessel, temp=temp * g.vessel)


def test_hot_wax_rises_and_cold_wax_sinks():
    g = _geom()
    p = _params(mobility=0.2)  # viscous wax is slow; speed it up so 10 s shows clear motion
    for blob_temp, direction in [(0.9, 1), (0.05, -1)]:
        s = _blob_state(g, blob_temp, liquid_temp=0.3)
        h0 = _blob_height(s.wax, g)
        for _ in range(20):
            s = ph.step(s, g, p, dt=0.5)
        assert direction * (_blob_height(s.wax, g) - h0) > 0.01, (
            blob_temp,
            h0,
            _blob_height(s.wax, g),
        )


def _outline(wax: torch.Tensor, g: ph.Geometry) -> float:
    """Total variation inside the vessel: the length of the wax outline."""
    gx = (wax[..., :, 1:] - wax[..., :, :-1]).abs() * g.open_x[:, 1:-1]
    gy = (wax[..., 1:, :] - wax[..., :-1, :]).abs() * g.open_y[1:-1]
    return float(gx.sum() + gy.sum())


def _relax(wax: torch.Tensor, g: ph.Geometry, seconds: float, **params) -> torch.Tensor:
    p = _params(**params)
    for _ in range(int(seconds / 0.5)):
        wax = ph.apply_tension(wax, g, p, 0.5)
    return wax


def test_surface_tension_shortens_ragged_outlines():
    g = _geom()
    gen = torch.Generator().manual_seed(11)
    blob = torch.zeros(1, H, W, dtype=torch.float64)
    blob[0, 10:22, 4:12] = 1.0
    ragged = (blob * (torch.rand(blob.shape, generator=gen, dtype=torch.float64) > 0.3)) * g.vessel
    total = ragged.sum()
    relaxed = _relax(ragged, g, 40.0)
    assert abs(float(relaxed.sum() - total)) < 1e-12 * float(total)
    assert relaxed.min() >= -1e-12 and relaxed.max() <= 1 + 1e-12
    assert (relaxed * (1 - g.vessel)).abs().max() == 0
    assert _outline(relaxed, g) < 0.5 * _outline(ragged, g)
    # The holes fill in rather than the blob dissolving: the core ends up solid wax.
    assert float(relaxed[0, 13:19, 6:10].min()) > 0.8


def test_surface_tension_gradients_pass_straight_through_by_default():
    g = _geom()
    p = _params()
    wax = (torch.rand(1, H, W, dtype=torch.float64) * g.vessel).requires_grad_(True)
    out = ph.apply_tension(wax, g, p, 0.5)
    assert not torch.equal(out.detach(), wax.detach())  # forward really applies tension
    out.sum().backward()
    assert torch.equal(wax.grad, torch.ones_like(wax))  # backward is the identity
    exact = (torch.rand(1, H, W, dtype=torch.float64) * g.vessel).requires_grad_(True)
    ph.apply_tension(exact, g, p, 0.5, grad=True).pow(2).sum().backward()
    assert not torch.equal(exact.grad, 2 * exact.detach())  # grad=True differentiates through


def test_surface_tension_partial_gradients_see_the_last_substeps():
    g = _geom()
    p = _params()
    base = torch.rand(1, H, W, dtype=torch.float64) * g.vessel
    outs, grads = [], []
    for k in (0, 8):
        wax = base.clone().requires_grad_(True)
        out = ph.apply_tension(wax, g, p, 0.5, grad_substeps=k)
        out.pow(2).sum().backward()
        outs.append(out.detach())
        grads.append(wax.grad)
    assert torch.allclose(outs[0], outs[1], atol=1e-12)  # same forward result
    assert torch.allclose(grads[0], 2 * outs[0])  # k = 0: straight through
    assert not torch.allclose(grads[1], grads[0])  # k = 8: tension shapes the gradient


def test_surface_tension_rounds_a_long_bar():
    g = _geom()
    bar = torch.zeros(1, H, W, dtype=torch.float64)
    bar[0, 15:17, 3:13] = 1.0  # 2 x 10: far longer outline than a disc of the same area
    bar = bar * g.vessel
    relaxed = _relax(bar, g, 60.0, tension=4.0)
    rows = relaxed[0].sum(1)
    tall = int((rows > 0.5).sum())
    assert _outline(relaxed, g) < 0.85 * _outline(bar, g)
    assert tall >= 3  # the bar has pulled in and thickened


def test_poisson_matches_discrete_laplacian():
    g = _geom()
    f = torch.randn(2, H - 1, W - 1, dtype=torch.float64)
    x = F.pad(g.poisson(f), (1, 1, 1, 1))
    lap = x[:, :-2, 1:-1] + x[:, 2:, 1:-1] + x[:, 1:-1, :-2] + x[:, 1:-1, 2:] - 4 * x[:, 1:-1, 1:-1]
    assert (lap - f).abs().max() < 1e-9


def test_variable_solver_matches_direct_solver_at_uniform_viscosity():
    g = ph.Geometry(torch.ones(H, W, dtype=torch.bool))  # a box, so the direct solve is exact
    gen = torch.Generator().manual_seed(5)
    b = torch.randn(2, H, W, generator=gen, dtype=torch.float64)
    direct, _ = ph.stokes_stream(b, g, 1.0)
    iterative, _ = ph.stokes_stream(b, g, 1.0, mu=torch.ones(2, H, W, dtype=torch.float64))
    assert (direct - iterative).abs().max() < 1e-6 * direct.abs().max()


def test_vessel_biharmonic_is_exact_on_the_vessel_shape():
    g = _geom()
    gen = torch.Generator().manual_seed(9)
    f = torch.randn(2, H - 1, W - 1, generator=gen, dtype=torch.float64) * g.free_inner
    x = g.vessel_biharmonic(f)
    lhs = ph.stokes_operator(x, torch.ones(2, H, W, dtype=torch.float64), g)
    assert (lhs - f).abs().max() < 1e-8 * f.abs().max()
    assert (x * (1 - g.free_inner)).abs().max() == 0


def test_stokes_operator_is_symmetric_positive_definite():
    g = _geom()
    gen = torch.Generator().manual_seed(6)
    mu = torch.exp(3 * torch.rand(1, H, W, generator=gen, dtype=torch.float64))
    x, y = (
        torch.randn(1, H - 1, W - 1, generator=gen, dtype=torch.float64) * g.free_inner
        for _ in range(2)
    )
    ax, ay = ph.stokes_operator(x, mu, g), ph.stokes_operator(y, mu, g)
    assert abs(float((ax * y).sum() - (x * ay).sum())) < 1e-9 * float((ax * y).abs().sum())
    assert float((ax * x).sum()) > 0


def test_solver_converges_with_viscosity_contrast():
    g = _geom()
    gen = torch.Generator().manual_seed(7)
    s = _blob_state(g, 0.9, 0.3)
    mu = ph.viscosity(s, _params(visc_wax=300.0))
    f = ph.buoyancy_rhs(torch.randn(1, H, W, generator=gen, dtype=torch.float64), g)
    x, iters = ph.solve_stokes(f, mu, g, tol=1e-8)
    resid = ph.stokes_operator(x, mu, g) - f * g.free_inner
    assert float(resid.norm()) < 1e-7 * float((f * g.free_inner).norm())
    assert iters < 300


def test_adjoint_solver_gradients_match_finite_differences():
    g = _geom()
    gen = torch.Generator().manual_seed(8)
    s = _blob_state(g, 0.9, 0.3)
    mu = ph.viscosity(s, _params(visc_wax=300.0)).requires_grad_()
    f = ph.buoyancy_rhs(torch.randn(1, H, W, generator=gen, dtype=torch.float64), g)
    f.requires_grad_()
    weight = torch.randn(1, H - 1, W - 1, generator=gen, dtype=torch.float64)

    def loss(f, mu):
        return (ph.solve_stokes_adjoint(f, mu, g, tol=1e-13) * weight).sum()

    loss(f, mu).backward()
    df = torch.randn(f.shape, generator=gen, dtype=torch.float64) * g.free_inner
    dlog = torch.randn(mu.shape, generator=gen, dtype=torch.float64) * g.vessel
    with torch.no_grad():
        h = 1e-5
        fd_f = (loss(f + h * df, mu) - loss(f - h * df, mu)) / (2 * h)
        fd_mu = (loss(f, mu * torch.exp(h * dlog)) - loss(f, mu * torch.exp(-h * dlog))) / (2 * h)
    assert abs(float((f.grad * df).sum() - fd_f)) < 1e-6 * abs(float(fd_f))
    assert abs(float((mu.grad * mu.detach() * dlog).sum() - fd_mu)) < 1e-5 * abs(float(fd_mu))


def test_solver_stays_finite_when_iterating_past_convergence():
    # Mixed batch in float32: an easy item, a hard item, and one with no force at all.
    # With tol=0 every item keeps going until rounding noise, which used to produce NaN.
    g = _geom(torch.float32)
    gen = torch.Generator().manual_seed(10)
    f = torch.randn(3, H - 1, W - 1, generator=gen) * g.free_inner
    f[2] = 0
    mu = torch.ones(3, H, W)
    mu[1] = torch.exp(5 * torch.rand(H, W, generator=gen))
    x, _ = ph.solve_stokes(f, mu, g, tol=0.0, max_iter=300)
    assert torch.isfinite(x).all()
    assert x[2].abs().max() == 0
    resid = ph.stokes_operator(x, mu, g) - f * g.free_inner
    assert float(resid[:2].norm()) < 1e-3 * float(f[:2].norm())


def _blob_velocity_spread(visc_wax: float) -> float:
    """Spread of vertical velocity inside a rising blob, relative to its mean."""
    g = _geom()
    p = _params(visc_wax=visc_wax, visc_melt=0.0)
    s = _blob_state(g, 0.9, 0.3)
    psi, _ = ph.stokes_stream(ph.buoyancy(s, g, p), g, p["mobility"], ph.viscosity(s, p))
    _, fy = ph.faces_from_stream(psi, g)
    vy = 0.5 * (fy[:, 1:, :] + fy[:, :-1, :])  # downward velocity per cell
    inside = vy[s.wax > 0.5]
    return float(inside.std() / inside.mean().abs())


def test_viscous_wax_moves_as_a_body():
    assert _blob_velocity_spread(300.0) < 0.5 * _blob_velocity_spread(1.0)


def test_params_round_trip_defaults():
    vals = _params(mobility=0.2)
    assert abs(float(vals["mobility"]) - 0.2) < 1e-4
    assert abs(float(vals["heat_rate"]) - ph.PARAM_BOUNDS["heat_rate"][2]) < 1e-4


def test_params_load_checkpoints_saved_before_a_constant_was_added():
    old = {k: v[:-1].clone() for k, v in ph.Params(mobility=0.2).state_dict().items()}
    params = ph.Params()
    params.load_state_dict(old)
    vals = params.values()
    assert abs(float(vals["mobility"]) - 0.2) < 1e-4
    assert abs(float(vals["expand"]) - ph.PARAM_BOUNDS["expand"][2]) < 1e-3


def test_neumann_poisson_flow_has_the_requested_divergence():
    g = _geom()
    gen = torch.Generator().manual_seed(12)
    f = torch.randn(2, H, W, generator=gen, dtype=torch.float64) * g.vessel
    f = f - f.sum((1, 2), keepdim=True) / g.vessel.sum() * g.vessel
    fx, fy = ph.potential_faces(ph.neumann_poisson(f, g, tol=1e-10), g)
    div = (fx[:, :, 1:] - fx[:, :, :-1]) + (fy[:, 1:, :] - fy[:, :-1, :])
    assert (div - f).abs().max() < 1e-8


def test_melting_wax_swells_by_expand_and_stays_bounded():
    g = _geom()
    p = _params(expand=0.3)
    s = _blob_state(g, 0.0, 0.0)
    cold, hot = torch.zeros_like(s.temp), torch.ones_like(s.temp) * g.vessel
    wax, temp = s.wax, cold
    mass = ph.wax_mass(wax, temp, p)
    # Melt in 100 increments (a real step changes temperature far less).
    for k in range(1, 101):
        new = hot * (k / 100)
        wax, _, _ = ph.expand(wax, new, None, temp, g, p, mass)
        temp = new
    ratio = float(wax.sum() / s.wax.sum())
    assert abs(ratio - 1.3) < 0.02
    assert wax.min() >= -1e-12 and wax.max() <= 1 + 1e-9
    assert (wax * (1 - g.vessel)).abs().max() == 0
    # Cooling back shrinks it to about where it started.
    for k in range(99, -1, -1):
        new = hot * (k / 100)
        wax, _, _ = ph.expand(wax, new, None, temp, g, p, mass)
        temp = new
    assert abs(float(wax.sum() / s.wax.sum()) - 1) < 0.02


def test_expanding_lamp_keeps_its_wax_mass():
    # Full physics steps heating cold wax: tension and compression move wax without its heat,
    # which used to ratchet the volume up; the lamp's wax mass must stay put.
    g = _geom()
    p = _params(expand=0.3, mobility=0.2, heat_rate=0.05)
    s = _blob_state(g, 0.0, 0.0)
    s.wax = s.wax.clone()
    s.wax[0, 24:30, 4:12] = 1.0 * g.vessel[24:30, 4:12]
    start = ph.wax_mass(s.wax, s.temp, p)
    for _ in range(400):
        s = ph.step(s, g, p, dt=0.5)
    assert abs(float(ph.wax_mass(s.wax, s.temp, p) / start) - 1) < 0.01
    assert float(s.wax.sum() / start) > 1.1  # and it did melt and swell
    assert s.wax.min() >= -1e-9 and s.wax.max() <= 1 + 1e-9


def test_wetting_pulls_a_film_off_the_side_glass_but_not_the_floor():
    g = _geom()
    left = torch.stack([torch.nonzero(r)[0, 0] for r in g.vessel[1:-1].bool()])
    film = torch.zeros(1, H, W, dtype=torch.float64)
    for i, c in enumerate(left.tolist()):
        if 4 <= i < 20:
            film[0, i + 1, c : c + 2] = 1.0  # two cells thick against the left glass
    film[0, H - 4 : H - 1] = 1.0  # and a pool on the floor
    film = film * g.vessel
    out = {}
    for wet in (0.0, 0.5):
        p = _params(tension=10.0, wet=wet)
        wax = film
        for _ in range(20):
            wax = ph.apply_tension(wax, g, p, 0.5)
        assert abs(float(wax.sum() - film.sum())) < 1e-9
        out[wet] = wax
    on_glass = lambda w: float((w * g.side)[0, : H - 2].sum())
    assert on_glass(out[0.5]) < 0.1 * on_glass(out[0.0])
    floor = lambda w: float((w * (1 - g.side))[0, H - 2].sum())  # on the floor, off the glass
    assert floor(out[0.5]) > 0.9 * floor(out[0.0])


def test_sink_ratio_scales_only_cold_wax_buoyancy():
    g = _geom()
    wax = g.vessel[None].double()
    temp = torch.full_like(wax, 0.3) * g.vessel
    temp[:, : H // 2] = 0.6 * g.vessel[: H // 2]  # hot above, cold below; float_temp is 0.45
    full = ph.buoyancy(ph.State(wax=wax, temp=temp), g, _params(buoy_liquid=0.0, sink_ratio=1.0))
    half = ph.buoyancy(ph.State(wax=wax, temp=temp), g, _params(buoy_liquid=0.0, sink_ratio=0.5))
    hot, cold = temp > 0.45, (temp < 0.45) & g.vessel.bool()
    assert torch.allclose(half[hot], full[hot])
    assert torch.allclose(half[cold], 0.5 * full[cold], rtol=1e-3)
