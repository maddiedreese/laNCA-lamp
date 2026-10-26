"""The hard-coded physics of the lamp: wax and heat move, but never appear or vanish.

Fields live on the same 128x64 grid as the masks. Row 0 is the top of the glass. Each step:

1. Flow. Velocities come from a stream function psi on cell corners. Any psi gives a
   divergence-free flow, and psi is pinned to zero on corners that touch a wall, so nothing
   crosses the glass. The built-in psi is Stokes (slow, viscous) flow driven by buoyancy, with
   wax far more viscous than the liquid, and stiffer still when cold. That contrast is what
   makes blobs move as bodies instead of being sheared into streaks. In the learned model
   (Phase 4), the network adds its own psi on top.
2. Transport. Wax fraction and temperature are moved by conservative fluxes. With a
   divergence-free flow, upwind transport makes each cell's new value a weighted average of
   its old value and its neighbors', so wax stays in [0, 1] and the total is exact up to float
   rounding. Upwind transport smears edges in proportion to the flow, so each transport
   substep is followed by interface compression, an exchange that pushes wax back toward its
   own side of the edge at the same rate.
3. Surface tension, Cahn-Hilliard style: wax trades places with liquid across faces, flowing
   down a chemical potential that is high on bumps and thin bits. This shortens interfaces,
   rounding blobs and driving necking and pinch-off, the way real surface tension does. The
   exchange form is conservative and bounded.
4. Heat. Conduction (wax and liquid conduct differently), a heater under the base (the bulb),
   and cooling through the walls and the cap.

Buoyancy: hot wax is lighter than the liquid and rises, cold wax is heavier and sinks. The
liquid itself convects weakly.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor


class Geometry:
    """Everything derived from the vessel shape, precomputed once.

    vessel: (H, W) bool, True inside the glass.

    Built in float64 so the solver matrices are accurate; call .to(device, dtype) to move it.
    """

    def __init__(self, vessel: Tensor, heater_rows: int = 5, top_rows: int = 6):
        v = vessel.to(torch.float64)
        self.shape = h, w = v.shape
        self.vessel = v
        pad = F.pad(v, (1, 1, 1, 1))
        # Corner (i, j) sits between cells (i-1..i, j-1..j); it can carry flow only if all four
        # of those cells are inside the vessel.
        self.corner_free = pad[:-1, :-1] * pad[:-1, 1:] * pad[1:, :-1] * pad[1:, 1:]
        self.free_inner = self.corner_free[1:-1, 1:-1]  # the unknowns of the flow solver
        # Faces between two vessel cells: x faces are (H, W+1), y faces are (H+1, W).
        self.open_x = pad[1:-1, :-1] * pad[1:-1, 1:]
        self.open_y = pad[:-1, 1:-1] * pad[1:, 1:-1]

        rows = torch.nonzero(v.any(1)).squeeze(1)
        self.top, self.bottom = int(rows.min()), int(rows.max())
        r = torch.arange(h, dtype=v.dtype, device=v.device)[:, None].expand(h, w)
        self.height = ((self.bottom - r) / (self.bottom - self.top)).clamp(0, 1) * v

        # Heater: the bottom rows, strongest in the middle where the bulb is.
        cols = torch.arange(w, dtype=v.dtype, device=v.device)
        bottom_cols = torch.nonzero(v[self.bottom]).squeeze(1).to(v.dtype)
        center, half = bottom_cols.mean(), (bottom_cols.max() - bottom_cols.min() + 1) / 2
        across = torch.exp(-0.5 * ((cols - center) / (0.6 * half)) ** 2)
        up = torch.exp(-(self.bottom - r).clamp(min=0) / heater_rows)
        self.heater = up * across[None, :] * v
        self.heater = self.heater / self.heater.max()

        # Cooling: cells touching the side walls, and the top rows under the cap.
        neighbors = pad[:-2, 1:-1] * pad[2:, 1:-1] * pad[1:-1, :-2] * pad[1:-1, 2:]
        self.wall = (1 - neighbors) * v
        self.cap = torch.exp(-(r - self.top).clamp(min=0) / top_rows) * v

        # Dirichlet Poisson solver on the interior corners, by sine transform.
        n, m = h - 1, w - 1
        self._sn, self._sm = _sine_matrix(n, v.device), _sine_matrix(m, v.device)
        k_n = torch.arange(1, n + 1, dtype=v.dtype, device=v.device)
        k_m = torch.arange(1, m + 1, dtype=v.dtype, device=v.device)
        ln = 2 * torch.cos(math.pi * k_n / (n + 1)) - 2
        lm = 2 * torch.cos(math.pi * k_m / (m + 1)) - 2
        self._eig = ln[:, None] + lm[None, :]
        self._norm = 4.0 / ((n + 1) * (m + 1))

        # Capacitance matrix, for exact uniform-viscosity solves on the vessel's own shape (see
        # vessel_biharmonic). Built from the response to a unit impulse at each pinned corner.
        self._pinned = torch.nonzero(self.free_inner.flatten() == 0).squeeze(1)
        k = len(self._pinned)
        if k:
            cap = torch.empty(k, k, dtype=v.dtype, device=v.device)
            for i in range(0, k, 512):
                rows = self._pinned[i : i + 512]
                impulses = torch.zeros(len(rows), n * m, dtype=v.dtype, device=v.device)
                impulses[torch.arange(len(rows)), rows] = 1.0
                response = self.poisson(self.poisson(impulses.view(-1, n, m))).flatten(1)
                cap[i : i + len(rows)] = response[:, self._pinned]
            self._cap_inv = torch.linalg.inv(cap)
        else:
            self._cap_inv = torch.zeros(0, 0, dtype=v.dtype, device=v.device)

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "Geometry":
        for k, val in vars(self).items():
            if isinstance(val, Tensor):
                if val.is_floating_point():
                    setattr(self, k, val.to(device=device, dtype=dtype))
                else:
                    setattr(self, k, val.to(device=device))
        return self

    def poisson(self, f: Tensor) -> Tensor:
        """Solve laplacian(x) = f on the (H-1, W-1) interior corners, x = 0 on the border."""
        fh = self._sn @ f @ self._sm
        return self._norm * (self._sn @ (fh / self._eig) @ self._sm)

    def vessel_biharmonic(self, f: Tensor) -> Tensor:
        """Exactly solve the uniform-viscosity flow equation on the vessel's own shape.

        The sine transform solves it on the bounding box. Pinning stream values to zero at the
        corners outside the vessel adds a force at those corners; the capacitance matrix
        (box response at pinned corners to forces at pinned corners) gives exactly the forces
        that cancel the box solution there. f is (B, H-1, W-1); the result is zero off-vessel.
        """
        y = self.poisson(self.poisson(f))
        if len(self._pinned):
            b = f.shape[0]
            force = y.flatten(1)[:, self._pinned] @ self._cap_inv.T
            spread = torch.zeros(b, y[0].numel(), dtype=f.dtype, device=f.device)
            spread[:, self._pinned] = force
            y = y - self.poisson(self.poisson(spread.view_as(y)))
        return y * self.free_inner


def _sine_matrix(n: int, device) -> Tensor:
    k = torch.arange(1, n + 1, dtype=torch.float64, device=device)
    return torch.sin(math.pi * k[:, None] * k[None, :] / (n + 1))


@dataclass
class State:
    """Lamp state. wax: wax fraction per cell in [0, 1]; temp: 0 = room, 1 = bulb.

    Both are (B, H, W). hidden is (B, C, H, W) of extra channels for the learned model; the
    physics carries it along with the flow but otherwise leaves it alone. flow is the last
    buoyancy flow solution (B, H-1, W-1), kept only to warm-start the next solve.
    """

    wax: Tensor
    temp: Tensor
    hidden: Tensor | None = None
    flow: Tensor | None = None


PARAM_BOUNDS = {
    # name: (low, high, default). Rates are per second, lengths are cells. Defaults are the
    # best hand-tuned baseline (see README); Phase 5 fits them to the footage.
    "heat_rate": (1e-4, 1e-1, 1.5e-2),  # how fast the bulb pulls nearby cells toward T = 1
    "cool_wall": (1e-5, 1e-2, 6e-4),  # loss through the side walls
    "cool_cap": (1e-5, 1e-1, 3e-2),  # loss through the cap
    "kappa_wax": (0.0, 2.0, 0.05),  # conduction, cells^2 / s
    "kappa_liquid": (0.0, 2.0, 0.15),
    "float_temp": (0.05, 0.95, 0.45),  # wax is neutrally buoyant at this temperature
    "buoy_wax": (0.0, 50.0, 6.0),  # buoyancy of wax per unit (T - float_temp)
    "buoy_liquid": (0.0, 50.0, 0.5),  # thermal buoyancy of the liquid
    "mobility": (0.0, 20.0, 4.0),  # Stokes flow per unit buoyancy (1 / liquid viscosity)
    "tension": (0.0, 20.0, 1.0),  # surface tension: exchange rate per unit potential difference
    "interface": (0.5, 3.0, 1.0),  # interface width in cells (Cahn-Hilliard epsilon)
    "visc_wax": (1.0, 1000.0, 100.0),  # wax viscosity / liquid viscosity at float_temp
    "visc_melt": (0.0, 10.0, 3.0),  # wax stiffens by exp(visc_melt) per unit drop in T
    "compress": (0.0, 4.0, 2.0),  # interface compression, relative to the flow through a face
}

MAX_VISCOSITY = 1e4  # cap on the viscosity ratio, to keep the flow solver well conditioned


class Params(nn.Module):
    """Physical constants, stored unconstrained and mapped into their bounds (for Phase 5)."""

    def __init__(self, **overrides: float):
        super().__init__()
        self.names = list(PARAM_BOUNDS)
        lo = torch.tensor([PARAM_BOUNDS[k][0] for k in self.names])
        hi = torch.tensor([PARAM_BOUNDS[k][1] for k in self.names])
        init = torch.tensor([overrides.get(k, PARAM_BOUNDS[k][2]) for k in self.names])
        frac = ((init - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.raw = nn.Parameter(torch.logit(frac))

    def values(self) -> dict[str, Tensor]:
        v = self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw)
        return dict(zip(self.names, v))


# --- flow -----------------------------------------------------------------------------------


def faces_from_stream(psi: Tensor, geom: Geometry) -> tuple[Tensor, Tensor]:
    """Turn a corner stream function (B, H+1, W+1) into face flows.

    fx (B, H, W+1): flow to the right through each vertical face.
    fy (B, H+1, W): flow downward (toward larger row index) through each horizontal face.
    Units are cell volumes per unit time. The discrete divergence is exactly zero.
    """
    psi = psi * geom.corner_free
    fx = psi[:, :-1, :] - psi[:, 1:, :]
    fy = psi[:, :, 1:] - psi[:, :, :-1]
    return fx * geom.open_x, fy * geom.open_y


def viscosity(state: State, p: dict[str, Tensor]) -> Tensor:
    """Viscosity per cell (B, H, W), relative to the liquid.

    Wax is visc_wax times more viscous than the liquid at its float temperature, and stiffer
    when colder, so cold wax at the bottom acts almost solid. Mixed cells interpolate in log
    space, so a half-wax cell is the geometric mean of the two.
    """
    stiff = p["visc_wax"] * torch.exp(p["visc_melt"] * (p["float_temp"] - state.temp))
    return torch.exp(state.wax * torch.log(stiff.clamp(1.0, MAX_VISCOSITY)))


def stokes_operator(x: Tensor, mu: Tensor, geom: Geometry) -> Tensor:
    """The variable-viscosity Stokes operator on interior corner stream values (B, H-1, W-1).

    In stream-function form, viscous flow satisfies
        (dxx - dyy)[mu (dxx - dyy) psi] + 4 dxy[mu dxy psi] = force term,
    written here as L1^T mu L1 + 4 L2^T mu L2, which is symmetric positive definite. L1 lives
    on corners, L2 on cells. With mu = 1 it is exactly the discrete biharmonic, the square of
    the 5-point Laplacian.

    Corner viscosity is the geometric mean of the 4 cells around it. An arithmetic mean would
    let a single wax cell make the liquid corners beside it nearly as stiff as wax, wrapping
    every blob in a sticky halo that slows it far more than real wax would.
    """
    x = x * geom.free_inner
    xp = F.pad(x, (1, 1, 1, 1))

    def l1(t):  # (dxx - dyy) on interior corners, t padded with the zero border
        dxx = t[:, 1:-1, 2:] + t[:, 1:-1, :-2] - 2 * t[:, 1:-1, 1:-1]
        dyy = t[:, 2:, 1:-1] + t[:, :-2, 1:-1] - 2 * t[:, 1:-1, 1:-1]
        return dxx - dyy

    log_mu = torch.log(mu)
    mu_corner = torch.exp(
        0.25 * (log_mu[:, :-1, :-1] + log_mu[:, :-1, 1:] + log_mu[:, 1:, :-1] + log_mu[:, 1:, 1:])
    )
    a1 = l1(F.pad(mu_corner * l1(xp), (1, 1, 1, 1)))
    c = (xp[:, 1:, 1:] - xp[:, 1:, :-1] - xp[:, :-1, 1:] + xp[:, :-1, :-1]) * mu
    cp = F.pad(c, (1, 1, 1, 1))
    a2 = cp[:, 1:, 1:] - cp[:, 1:, :-1] - cp[:, :-1, 1:] + cp[:, :-1, :-1]
    return (a1 + 4 * a2[:, 1:-1, 1:-1]) * geom.free_inner


def solve_stokes(
    rhs: Tensor,
    mu: Tensor,
    geom: Geometry,
    x0: Tensor | None = None,
    tol: float = 1e-2,
    max_iter: int = 300,
) -> tuple[Tensor, int]:
    """Solve stokes_operator(x) = rhs by conjugate gradients, to a relative residual of tol.

    The default of 1% is plenty for a flow that the learned model corrects anyway, and
    roughly halves the cost of 0.01%.

    Preconditioned with the exact uniform-viscosity inverse on the vessel's shape
    (Geometry.vessel_biharmonic), so a uniform mu converges in one iteration and only the
    viscosity contrast costs extra iterations. Returns the solution and iterations used.

    Each batch item stops updating as soon as it converges. Without that, items that converge
    early keep iterating on rounding noise until the slowest one finishes, and the ratio of two
    near-zero numbers eventually blows up to inf and then NaN. An item whose CG quantities
    stop being positive and finite is frozen at its last good iterate for the same reason.
    """
    m = geom.free_inner
    b = rhs * m

    def dot(u, v):
        return (u * v).sum((1, 2), keepdim=True)

    def precondition(r):
        return geom.vessel_biharmonic(r * m)

    x = torch.zeros_like(b) if x0 is None else x0 * m
    r = b - stokes_operator(x, mu, geom)
    b_norm = dot(b, b).sqrt()
    z = precondition(r)
    d, rz = z, dot(r, z)
    active = torch.ones_like(b_norm, dtype=torch.bool)
    for i in range(max_iter):
        active = active & (dot(r, r).sqrt() > tol * b_norm)
        if not bool(active.any()):
            return x, i
        ad = stokes_operator(d, mu, geom)
        dad = dot(d, ad)
        ok = active & (rz > 0) & (dad > 0) & torch.isfinite(rz) & torch.isfinite(dad)
        alpha = torch.where(
            ok, rz / torch.where(ok, dad, torch.ones_like(dad)), torch.zeros_like(dad)
        )
        x_new = x + alpha * d
        r_new = r - alpha * ad
        z = precondition(r_new)
        rz_new = dot(r_new, z)
        # Reject any update that overflowed, so a finite warm start always yields a finite
        # flow, whatever rounding does to a chaotic simulation.
        finite = torch.isfinite(x_new).all(dim=(1, 2), keepdim=True) & torch.isfinite(rz_new)
        ok = ok & finite
        beta = torch.where(
            ok, rz_new / torch.where(ok, rz, torch.ones_like(rz)), torch.zeros_like(rz)
        )
        x = torch.where(ok, x_new, x)
        r = torch.where(ok, r_new, r)
        d = torch.where(ok, z + beta * d, d)
        rz = torch.where(ok, rz_new, rz)
        active = ok
    return x, max_iter


def buoyancy_rhs(buoyancy: Tensor, geom: Geometry) -> Tensor:
    """Force term of the stream-function equation: d(buoyancy)/dx on interior corners."""
    b = buoyancy * geom.vessel
    return 0.5 * (b[:, :-1, 1:] + b[:, 1:, 1:] - b[:, :-1, :-1] - b[:, 1:, :-1])


def stokes_stream(
    buoyancy: Tensor,
    geom: Geometry,
    mobility: Tensor | float,
    mu: Tensor | None = None,
    x0: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Stream function (B, H+1, W+1) of slow viscous flow driven by an upward force (B, H, W).

    With mu None, the viscosity is uniform and the solution is direct and exact on the
    vessel's shape. With mu, the variable-viscosity equation is solved iteratively. Also
    returns the raw solution, to warm-start the next call.
    """
    f = buoyancy_rhs(buoyancy, geom)
    if mu is None:
        x = geom.vessel_biharmonic(f)
    else:
        x, _ = solve_stokes(f, mu, geom, x0)
    # Sign convention checked by test_hot_wax_rises_and_cold_wax_sinks: with rows growing
    # downward and faces_from_stream's orientation, this sign makes positive buoyancy rise.
    return mobility * F.pad(x, (1, 1, 1, 1)), x


def buoyancy(state: State, geom: Geometry, p: dict[str, Tensor]) -> Tensor:
    wax, temp = state.wax, state.temp
    v = geom.vessel
    mean_t = (temp * v).sum((1, 2), keepdim=True) / v.sum()
    b_wax = p["buoy_wax"] * wax * (temp - p["float_temp"])
    b_liq = p["buoy_liquid"] * (1 - wax) * (temp - mean_t)
    return (b_wax + b_liq) * v


# --- transport ------------------------------------------------------------------------------


def outflow(fx: Tensor, fy: Tensor) -> Tensor:
    """Total outgoing flow of each cell (B, H, W)."""
    return (
        fx[:, :, 1:].clamp(min=0)
        + (-fx[:, :, :-1]).clamp(min=0)
        + fy[:, 1:, :].clamp(min=0)
        + (-fy[:, :-1, :]).clamp(min=0)
    )


def advect(q: Tensor, fx: Tensor, fy: Tensor) -> Tensor:
    """Move a cell quantity q (B, H, W) or (B, C, H, W) through face flows, upwind.

    Requires outflow(fx, fy) <= 1 everywhere; see substeps().
    """
    if q.dim() == 4:
        fx, fy = fx[:, None], fy[:, None]
    qp = F.pad(q, (1, 1, 1, 1))
    left, right = qp[..., 1:-1, :-1], qp[..., 1:-1, 1:]
    up, down = qp[..., :-1, 1:-1], qp[..., 1:, 1:-1]
    tx = fx * torch.where(fx > 0, left, right)
    ty = fy * torch.where(fy > 0, up, down)
    return q - (tx[..., :, 1:] - tx[..., :, :-1]) - (ty[..., 1:, :] - ty[..., :-1, :])


def substeps(rate_per_cell: Tensor, limit: float = 0.9) -> int:
    """How many equal substeps keep each one's per-cell rate under the limit."""
    return max(1, math.ceil(float(rate_per_cell.detach().max()) / limit))


def chemical_potential(wax: Tensor, geom: Geometry, width: Tensor | float) -> Tensor:
    """Cahn-Hilliard chemical potential (B, H, W) of the wax field.

    mu = f'(w) - width^2 * laplacian(w), with the double well f(w) = w^2 (1 - w)^2. The first
    term pushes cells toward pure wax or pure liquid; the second is high on convex bumps and
    low in dents, so wax flowing down mu shortens the interface. Neighbors across the glass
    count as the cell itself (no flux through walls).
    """
    wp = F.pad(wax, (1, 1, 1, 1))
    lap = (
        (wp[:, 1:-1, :-2] - wax) * geom.open_x[:, :-1]
        + (wp[:, 1:-1, 2:] - wax) * geom.open_x[:, 1:]
        + (wp[:, :-2, 1:-1] - wax) * geom.open_y[:-1]
        + (wp[:, 2:, 1:-1] - wax) * geom.open_y[1:]
    )
    return 2 * wax * (1 - wax) * (1 - 2 * wax) - width**2 * lap


def tension_faces(
    wax: Tensor, geom: Geometry, rate: Tensor | float, width: Tensor | float
) -> tuple[Tensor, Tensor]:
    """Surface tension as wax/liquid exchange down the chemical potential.

    Wax moves across each face from higher to lower mu, at rate * (mu difference) times
    wax_source * (1 - wax_target). That last factor is the "degenerate mobility" of
    Cahn-Hilliard models: wax only moves where there is wax to move and room to receive it,
    so motion concentrates along interfaces (surface diffusion), wax is conserved exactly, and
    it stays in [0, 1].
    """
    mp = F.pad(chemical_potential(wax, geom, width), (1, 1, 1, 1))
    sx = rate * (mp[:, 1:-1, :-1] - mp[:, 1:-1, 1:])  # positive moves wax right
    sy = rate * (mp[:, :-1, 1:-1] - mp[:, 1:, 1:-1])  # positive moves wax down
    return exchange_faces(wax, geom, sx, sy)


def apply_tension(
    wax: Tensor,
    geom: Geometry,
    p: dict[str, Tensor],
    dt: float,
    grad: bool = False,
    grad_substeps: int = 0,
) -> Tensor:
    """Apply surface tension for dt seconds, in as many substeps as stability needs.

    Explicit Cahn-Hilliard is stiff: its fastest (checkerboard) mode grows the potential
    difference across a face by up to 64 * width^2 per unit of wax moved. The mobility here is
    w_source * (1 - w_target), which reaches 1 across a sharp edge (not the 1/4 of the textbook
    w(1 - w) form), so each substep keeps rate * (64 width^2 + 2) under 1, or it oscillates and
    makes outlines longer instead of shorter. That is hundreds of substeps per step.

    Differentiating through all of them is impractical (it ran the GPU out of memory), so:
    - grad=True differentiates through every substep exactly;
    - grad_substeps=k > 0 runs the earlier substeps without gradients and the last k with
      them, so the backward pass sees how tension reshapes wax (speckle is smoothed away, so
      producing it gains nothing) at memory cost proportional to k. The earlier substeps pass
      gradients straight through;
    - otherwise all substeps pass gradients straight through, as if tension did nothing.
    """
    width = p["interface"].detach()
    stiffness = float((p["tension"].detach() * dt * (64 * width**2 + 2)).max())
    n = max(1, math.ceil(stiffness))

    def relax(w: Tensor, tension: Tensor, width: Tensor, steps: int) -> Tensor:
        rate = tension * dt / n
        for _ in range(steps):
            w = apply_exchange(w, *tension_faces(w, geom, rate, width))
        return w

    if grad or not torch.is_grad_enabled():
        return relax(wax, p["tension"], p["interface"], n)
    k = min(grad_substeps, n)
    with torch.no_grad():
        early = relax(wax, p["tension"].detach(), width, n - k)
    early = wax + (early - wax).detach()
    return relax(early, p["tension"], p["interface"], k)


def compression_faces(
    wax: Tensor, geom: Geometry, fx: Tensor, fy: Tensor, strength: Tensor
) -> tuple[Tensor, Tensor]:
    """Interface compression: undo the smearing of one upwind transport substep.

    Upwind transport blurs an edge in proportion to how much fluid crosses it, so faster flow
    means more blur. This exchange pushes wax back along the interface normal (toward more
    wax) at a rate proportional to the flow through each face, as in volume-of-fluid methods,
    so the correction scales with the blur it cancels. Normals come from the lightly blurred
    wax field and are averaged onto faces.
    """
    blur = F.avg_pool2d(F.pad(wax[:, None], (1, 1, 1, 1), mode="replicate"), 3, 1)[:, 0]
    bp = F.pad(blur, (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (bp[:, 1:-1, 2:] - bp[:, 1:-1, :-2])
    gy = 0.5 * (bp[:, 2:, 1:-1] - bp[:, :-2, 1:-1])
    # The epsilon goes inside the root: sqrt has an infinite derivative at 0, which would turn
    # gradients into NaN wherever the wax is flat.
    mag = torch.sqrt(gx**2 + gy**2 + 1e-12)
    nx, ny = gx / mag, gy / mag
    nxp = F.pad(nx, (1, 1, 0, 0))
    nyp = F.pad(ny, (0, 0, 1, 1))
    nx_face = 0.5 * (nxp[:, :, :-1] + nxp[:, :, 1:])
    ny_face = 0.5 * (nyp[:, :-1, :] + nyp[:, 1:, :])
    return exchange_faces(wax, geom, strength * fx.abs() * nx_face, strength * fy.abs() * ny_face)


def exchange_faces(wax: Tensor, geom: Geometry, sx: Tensor, sy: Tensor) -> tuple[Tensor, Tensor]:
    """Net wax flows for signed exchange rates on faces (positive moves wax right / down).

    Wax can only move from a cell that has it into a cell that has room for it, so the flow
    from a to b is s * wax_a * (1 - wax_b). Rates are capped at 1/4 per face.
    """
    sx, sy = sx.clamp(-0.25, 0.25), sy.clamp(-0.25, 0.25)
    wp = F.pad(wax, (1, 1, 1, 1))
    left, right = wp[:, 1:-1, :-1], wp[:, 1:-1, 1:]
    up, down = wp[:, :-1, 1:-1], wp[:, 1:, 1:-1]
    fx = sx.clamp(min=0) * left * (1 - right) - (-sx).clamp(min=0) * right * (1 - left)
    fy = sy.clamp(min=0) * up * (1 - down) - (-sy).clamp(min=0) * down * (1 - up)
    return fx * geom.open_x, fy * geom.open_y


def apply_exchange(wax: Tensor, fx: Tensor, fy: Tensor) -> Tensor:
    """Apply net wax flows (already in amounts, not rates) across faces."""
    return wax - (fx[:, :, 1:] - fx[:, :, :-1]) - (fy[:, 1:, :] - fy[:, :-1, :])


# --- heat -----------------------------------------------------------------------------------


def conduct(temp: Tensor, wax: Tensor, geom: Geometry, p: dict[str, Tensor], dt: float) -> Tensor:
    kappa = p["kappa_liquid"] + (p["kappa_wax"] - p["kappa_liquid"]) * wax
    kp = F.pad(kappa, (1, 1, 1, 1))
    kx = 0.5 * (kp[:, 1:-1, :-1] + kp[:, 1:-1, 1:]) * geom.open_x
    ky = 0.5 * (kp[:, :-1, 1:-1] + kp[:, 1:, 1:-1]) * geom.open_y
    total = (kx[:, :, 1:] + kx[:, :, :-1] + ky[:, 1:, :] + ky[:, :-1, :]) * dt
    n = substeps(total)
    tp_kx, tp_ky = kx * (dt / n), ky * (dt / n)
    for _ in range(n):
        tp = F.pad(temp, (1, 1, 1, 1))
        qx = tp_kx * (tp[:, 1:-1, :-1] - tp[:, 1:-1, 1:])  # heat flowing right
        qy = tp_ky * (tp[:, :-1, 1:-1] - tp[:, 1:, 1:-1])  # heat flowing down
        temp = temp - (qx[:, :, 1:] - qx[:, :, :-1]) - (qy[:, 1:, :] - qy[:, :-1, :])
    return temp


def heat_exchange(temp: Tensor, geom: Geometry, p: dict[str, Tensor], dt: float) -> Tensor:
    """Bulb heating toward T = 1 and cooling toward T = 0, integrated exactly per cell."""
    heat = p["heat_rate"] * geom.heater
    cool = p["cool_wall"] * geom.wall + p["cool_cap"] * geom.cap
    rate = heat + cool
    target = torch.where(rate > 0, heat / rate.clamp(min=1e-12), torch.zeros_like(rate))
    return (target + (temp - target) * torch.exp(-rate * dt)) * geom.vessel


# --- one step -------------------------------------------------------------------------------


def physics_flow(state: State, geom: Geometry, p: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """The buoyancy-driven stream function (B, H+1, W+1), per second, and the raw solver
    solution (to warm-start the next solve)."""
    return stokes_stream(
        buoyancy(state, geom, p), geom, p["mobility"], viscosity(state, p), state.flow
    )


def cell_velocity(psi: Tensor, geom: Geometry) -> tuple[Tensor, Tensor]:
    """Velocity at cell centers (B, H, W) for a stream function: rightward and upward."""
    fx, fy = faces_from_stream(psi, geom)
    vx = 0.5 * (fx[:, :, :-1] + fx[:, :, 1:])
    vy = -0.5 * (fy[:, :-1, :] + fy[:, 1:, :])  # fy points down the rows
    return vx, vy


def step(
    state: State,
    geom: Geometry,
    p: dict[str, Tensor],
    dt: float,
    psi_extra: Tensor | None = None,
    exchange_extra: tuple[Tensor, Tensor] | None = None,
) -> State:
    """Advance the lamp by dt seconds.

    psi_extra: an additional stream function rate (B, H+1, W+1), added to the buoyancy flow.
    exchange_extra: additional wax exchange rates (sx, sy) on faces, in the same signed form
    as exchange_faces takes (positive moves wax right / down).
    """
    psi, flow = physics_flow(state, geom, p)
    if psi_extra is not None:
        psi = psi + psi_extra
    return advance(state, geom, p, dt, psi, flow, exchange_extra)


def advance(
    state: State,
    geom: Geometry,
    p: dict[str, Tensor],
    dt: float,
    psi: Tensor,
    flow: Tensor | None = None,
    exchange_extra: tuple[Tensor, Tensor] | None = None,
    tension_grad_substeps: int = 0,
) -> State:
    """Advance by dt seconds with a given total stream function psi (per second).

    Everything after the flow solve: transport, compression, surface tension, any extra exchange,
    and heat. flow is the solver solution to carry forward for warm starts.
    tension_grad_substeps: how many final surface-tension substeps to differentiate through
    (see apply_tension).
    """
    fx, fy = faces_from_stream(psi * dt, geom)
    n = substeps(outflow(fx, fy))
    fx, fy = fx / n, fy / n
    wax, temp, hidden = state.wax, state.temp, state.hidden
    for _ in range(n):
        wax, temp = advect(wax, fx, fy), advect(temp, fx, fy)
        wax = apply_exchange(wax, *compression_faces(wax, geom, fx, fy, p["compress"]))
        if hidden is not None:
            hidden = advect(hidden, fx, fy)

    wax = apply_tension(wax, geom, p, dt, grad_substeps=tension_grad_substeps)
    if exchange_extra is not None:
        wax = apply_exchange(wax, *exchange_faces(wax, geom, *exchange_extra))

    temp = conduct(temp, wax, geom, p, dt)
    temp = heat_exchange(temp, geom, p, dt)
    # No clamping: bounds hold by construction, and clamping would break exact conservation.
    flow = None if flow is None else flow.detach()
    return State(wax=wax, temp=temp, hidden=hidden, flow=flow)
