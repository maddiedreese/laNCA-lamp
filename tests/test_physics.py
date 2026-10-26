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
    rate = torch.tensor(2.0, dtype=torch.float64)
    for i in range(100_000):
        if i % 100 == 0:  # a fresh random flow every 100 steps
            fx, fy = ph.faces_from_stream(_smooth_psi(gen, 0.3), g)
            n = ph.substeps(ph.outflow(fx, fy))
            fx, fy = fx / n, fy / n
        for _ in range(n):
            wax = ph.advect(wax, fx, fy)
        wax = ph.apply_exchange(wax, *ph.cohesion_faces(wax, g, rate))
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


def test_heat_without_wax_settles_warm_at_the_bottom():
    g = _geom()
    p = _params()
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
        assert direction * (_blob_height(s.wax, g) - h0) > 0.02, (
            blob_temp,
            h0,
            _blob_height(s.wax, g),
        )


def test_cohesion_sharpens_edges_without_losing_wax():
    g = _geom()
    wax = torch.zeros(1, H, W, dtype=torch.float64)
    wax[0, 12:20, 5:11] = 1.0
    wax = F.avg_pool2d(F.pad(wax[:, None], (2, 2, 2, 2)), 5, 1)[:, 0] * g.vessel
    total, fuzz = wax.sum(), (wax * (1 - wax)).sum()
    rate = torch.tensor(1.0, dtype=torch.float64)
    for _ in range(200):
        wax = ph.apply_exchange(wax, *ph.cohesion_faces(wax, g, rate))
    assert abs(float(wax.sum() - total)) < 1e-12 * float(total)
    assert (wax * (1 - wax)).sum() < 0.5 * fuzz


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
