"""The learned part of the lamp: a neural cellular automaton on top of the physics core.

Each step:

1. The physics computes its buoyancy-driven flow.
2. Every cell looks at its 3x3 neighborhood through fixed filters (identity, Sobel x and y,
   Laplacian) over wax, temperature, the hidden channels, the physics flow's velocity, and the
   fixed vessel and height channels.
3. A small per-cell network outputs:
   - a flow correction, as a stream function averaged onto cell corners. It goes through the
     same machinery as the physics flow, so it is divergence-free and never crosses the glass;
   - wax exchange rates on faces, for shape: rounding, necking, pinch-off, merging;
   - updates to the hidden channels, the model's memory.
4. physics.advance applies it all, so wax is conserved exactly and stays in [0, 1] whatever
   the network outputs.

Hidden updates are stochastic: each step only a random subset of cells applies them, with a
little noise. Flow and exchange apply everywhere: a random on/off pattern on wax exchange
injects cell-sized speckle, which over long runs shreds blobs into foam. The last layer starts at zero, so an untrained model is exactly
the physics baseline.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from lanca import physics as ph

Tensor = torch.Tensor

# Fixed 3x3 perception filters. Rows grow downward, so Sobel-y responds to "down".
_FILTERS = (
    torch.tensor(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[1, 2, 1], [2, -12, 2], [1, 2, 1]],
        ],
        dtype=torch.float32,
    )
    / torch.tensor([1.0, 8.0, 8.0, 4.0])[:, None, None]
)


class LampNCA(nn.Module):
    """Physics-structured NCA. The physical constants live in self.physics.

    hidden: number of hidden channels. width, width2: sizes of the per-cell network.
    fire_rate: share of cells that apply their hidden updates each step.
    noise: standard deviation of the noise added to hidden updates.
    psi_scale, exchange_scale: output ranges (cells^2/s, and exchange rate per second).
    speed_scale: typical flow speed in cells/s, used to normalize the velocity inputs.
    tension_grad_substeps: final surface-tension substeps to differentiate through, so the
    network learns how tension will reshape its output (see physics.apply_tension).
    """

    def __init__(
        self,
        hidden: int = 8,
        width: int = 128,
        width2: int = 64,
        fire_rate: float = 0.5,
        noise: float = 0.02,
        psi_scale: float = 10.0,
        exchange_scale: float = 2.0,
        speed_scale: float = 3.0,
        tension_grad_substeps: int = 8,
        physics: ph.Params | None = None,
    ):
        super().__init__()
        self.n_hidden = hidden
        self.fire_rate = fire_rate
        self.noise = noise
        self.psi_scale = psi_scale
        self.exchange_scale = exchange_scale
        self.speed_scale = speed_scale
        self.tension_grad_substeps = tension_grad_substeps
        self.physics = physics if physics is not None else ph.Params()

        self.n_in = 2 + hidden + 2 + 2  # wax, temp, hidden, vx, vy, vessel, height
        self.register_buffer("filters", _FILTERS.repeat(self.n_in, 1, 1)[:, None])
        last = nn.Conv2d(width2, 3 + hidden, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(
            nn.Conv2d(4 * self.n_in, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, width2, 1),
            nn.ReLU(),
            last,
        )

    def network_parameters(self) -> list[nn.Parameter]:
        return list(self.net.parameters())

    def init_state(self, wax: Tensor, temp: Tensor | None = None) -> ph.State:
        """A state from a wax field (B, H, W): temperature defaults to cold, hidden to zero."""
        b, h, w = wax.shape
        return ph.State(
            wax=wax,
            temp=torch.zeros_like(wax) if temp is None else temp,
            hidden=torch.zeros(b, self.n_hidden, h, w, dtype=wax.dtype, device=wax.device),
            flow=torch.zeros(b, h - 1, w - 1, dtype=wax.dtype, device=wax.device),
        )

    def draw_randomness(
        self, state: ph.State, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """The random inputs of one step: a fire mask (B, 1, H, W) and hidden noise."""
        b, c, h, w = state.hidden.shape
        kw = {"generator": generator, "device": state.wax.device, "dtype": state.wax.dtype}
        fire = (torch.rand(b, 1, h, w, **kw) < self.fire_rate).to(state.wax.dtype)
        noise = torch.randn(b, c, h, w, **kw) * self.noise
        return fire, noise

    def step(
        self,
        state: ph.State,
        geom: ph.Geometry,
        dt: float,
        fire: Tensor | None = None,
        noise: Tensor | None = None,
        flow_grad: bool = False,
    ) -> ph.State:
        """Advance by dt seconds.

        fire, noise: this step's random inputs (from draw_randomness); drawn fresh if omitted.
        flow_grad: differentiate through the physics flow solve. It is the expensive part, and
        without it gradients still reach the network through everything else, including how
        the flow moves wax; they just don't account for the physics flow reacting to the
        network's earlier changes.
        """
        if fire is None or noise is None:
            fire, noise = self.draw_randomness(state)
        p = self.physics.values()
        with torch.set_grad_enabled(flow_grad and torch.is_grad_enabled()):
            psi_phys, flow = ph.physics_flow(state, geom, p)

        vx, vy = ph.cell_velocity(psi_phys, geom)
        b = state.wax.shape[0]
        fixed = torch.stack([geom.vessel, geom.height])[None].expand(b, -1, -1, -1)
        x = torch.cat(
            [
                state.wax[:, None],
                state.temp[:, None],
                state.hidden,
                (vx / self.speed_scale)[:, None],
                (vy / self.speed_scale)[:, None],
                fixed,
            ],
            1,
        )
        features = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.filters, groups=self.n_in)
        out = self.net(features) * geom.vessel

        psi_cell = self.psi_scale * torch.tanh(out[:, 0])
        pc = F.pad(psi_cell, (1, 1, 1, 1))
        psi_net = 0.25 * (pc[:, :-1, :-1] + pc[:, :-1, 1:] + pc[:, 1:, :-1] + pc[:, 1:, 1:])

        ex = self.exchange_scale * torch.tanh(out[:, 1])
        ey = self.exchange_scale * torch.tanh(out[:, 2])
        exp, eyp = F.pad(ex, (1, 1, 0, 0)), F.pad(ey, (0, 0, 1, 1))
        sx = 0.5 * (exp[:, :, :-1] + exp[:, :, 1:]) * dt
        sy = 0.5 * (eyp[:, :-1, :] + eyp[:, 1:, :]) * dt

        new = ph.advance(
            state, geom, p, dt, psi_phys + psi_net, flow, (sx, sy), self.tension_grad_substeps
        )
        dh = (torch.tanh(out[:, 3:]) + noise) * fire
        new.hidden = (new.hidden + dh).clamp(-1, 1) * geom.vessel
        return new

    def rollout(
        self,
        state: ph.State,
        geom: ph.Geometry,
        steps: int,
        dt: float,
        generator: torch.Generator | None = None,
        checkpoint_every: int = 0,
        flow_grad: bool = False,
    ) -> tuple[ph.State, Tensor]:
        """Run several steps. Returns the final state and the wax after each step (T, B, H, W).

        checkpoint_every > 0 recomputes segments of that many steps during backward instead of
        storing them, trading time for memory on long rollouts. Randomness is drawn up front,
        so the recomputation sees exactly the same fire masks and noise.
        """
        waxes = []
        done = 0
        while done < steps:
            n = min(steps - done, checkpoint_every) if checkpoint_every > 0 else steps - done
            rand = [self.draw_randomness(state, generator) for _ in range(n)]

            def segment(wax, temp, hidden, flow, *flat, _n=n):
                s = ph.State(wax=wax, temp=temp, hidden=hidden, flow=flow)
                outs = []
                for i in range(_n):
                    s = self.step(s, geom, dt, flat[2 * i], flat[2 * i + 1], flow_grad)
                    outs.append(s.wax)
                return (s.wax, s.temp, s.hidden, s.flow, torch.stack(outs))

            flat = [t for pair in rand for t in pair]
            args = (state.wax, state.temp, state.hidden, state.flow, *flat)
            if checkpoint_every > 0 and torch.is_grad_enabled():
                result = checkpoint(segment, *args, use_reentrant=False)
            else:
                result = segment(*args)
            wax, temp, hidden, flow, seg = result
            state = ph.State(wax=wax, temp=temp, hidden=hidden, flow=flow)
            waxes.append(seg)
            done += n
        return state, torch.cat(waxes)
