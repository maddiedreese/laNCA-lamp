"""Export the trained model for the live browser lamp, with hot starting states.

    uv run python -m lanca.export_live --checkpoint models/stage_b_refit4_iter300.pt \
        --runs b4_i300_s1 b4_i300_s2 b4_i300_s5 --out /tmp/live

The flow preconditioner's capacitance matrix is kept only for the pinned corners that a free
corner's 13-point stencil can reach. Corners further out can't affect the free corners, so the
solution inside the glass is unchanged, and the matrix shrinks enough to download.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lanca import physics as ph
from lanca.extract import ROOT
from lanca.model import LampNCA

DT = 0.5


def reachable_pinned(free: torch.Tensor) -> torch.Tensor:
    """Indices of pinned corners within the biharmonic stencil of a free corner."""
    grown = F.max_pool2d(free[None, None].float(), 5, 1, 2)[0, 0] > 0
    # The 13-point stencil is a diamond of radius 2, narrower than the 5x5 box above.
    n, m = free.shape
    near = torch.zeros_like(free, dtype=torch.bool)
    idx = torch.nonzero(free)
    for di in range(-2, 3):
        for dj in range(-2 + abs(di), 3 - abs(di)):
            i, j = idx[:, 0] + di, idx[:, 1] + dj
            ok = (i >= 0) & (i < n) & (j >= 0) & (j < m)
            near[i[ok], j[ok]] = True
    assert bool((near & ~grown).sum() == 0)
    return torch.nonzero((near & (free == 0)).flatten()).squeeze(1)


def capacitance_inverse(geom: ph.Geometry, pinned: torch.Tensor) -> torch.Tensor:
    n, m = geom.free_inner.shape
    k = len(pinned)
    cap = torch.empty(k, k, dtype=torch.float64)
    for i in range(0, k, 256):
        rows = pinned[i : i + 256]
        impulses = torch.zeros(len(rows), n * m, dtype=torch.float64)
        impulses[torch.arange(len(rows)), rows] = 1.0
        response = geom.poisson(geom.poisson(impulses.view(-1, n, m))).flatten(1)
        cap[i : i + len(rows)] = response[:, pinned]
    return torch.linalg.inv(cap)


def pack_state(s: ph.State) -> np.ndarray:
    parts = [s.wax[0], s.temp[0], s.hidden[0].flatten(), s.flow[0]]
    return torch.cat([p.flatten() for p in parts]).numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--minute", type=int, default=100)
    ap.add_argument("--settle-steps", type=int, default=240)
    ap.add_argument("--wax", type=float, default=1830.0, help="wax cells in the starting frames")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = LampNCA()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    first = np.load(ROOT / "runs" / "physics" / args.runs[0] / "sim.npz")
    vessel = torch.from_numpy(first["vessel"])
    geom64 = ph.Geometry(vessel)

    # The trimmed preconditioner must give the same answer inside the glass as the full one.
    pinned = reachable_pinned(geom64.free_inner.bool())
    cap_inv = capacitance_inverse(geom64, pinned)
    f = torch.randn(1, *geom64.free_inner.shape, dtype=torch.float64) * geom64.free_inner
    full = geom64.vessel_biharmonic(f)
    y = geom64.poisson(geom64.poisson(f))
    spread = torch.zeros(1, y[0].numel(), dtype=torch.float64)
    spread[:, pinned] = y.flatten(1)[:, pinned] @ cap_inv.T
    trimmed = (y - geom64.poisson(geom64.poisson(spread.view_as(y)))) * geom64.free_inner
    gap = float((full - trimmed).abs().max() / full.abs().max())
    print(f"pinned {len(geom64._pinned)} -> {len(pinned)}, trimmed solve differs by {gap:.2e}")
    if gap > 1e-6:
        raise SystemExit("trimmed preconditioner disagrees")

    geom = geom64.to("cpu", torch.float32)
    weights = torch.cat([q.detach().flatten() for q in model.net.parameters()])
    (args.out / "weights.bin").write_bytes(weights.numpy().astype(np.float32).tobytes())
    (args.out / "cap_inv.bin").write_bytes(cap_inv.numpy().astype(np.float32).tobytes())
    h, w = vessel.shape
    meta = {
        "height": h,
        "width": w,
        "dt": DT,
        "hidden": model.n_hidden,
        "layers": [4 * model.n_in, 128, 64, 3 + model.n_hidden],
        "fire_rate": model.fire_rate,
        "noise": model.noise,
        "psi_scale": model.psi_scale,
        "exchange_scale": model.exchange_scale,
        "speed_scale": model.speed_scale,
        "thermal_noise": 2e-3,
        "params": {k: float(v.detach()) for k, v in model.physics.values().items()},
        "vessel": vessel.flatten().int().tolist(),
        "pinned": pinned.tolist(),
        "states": [],
    }

    gen = torch.Generator().manual_seed(0)
    used: list[int] = []
    with torch.no_grad():
        for i, name in enumerate(args.runs):
            # Start from real hot frames holding the footage's typical amount of wax, not from
            # the 2-hour run: the footage shows about 2.4x the wax of the cold start.
            run = np.load(ROOT / "runs" / "physics" / name / "sim.npz")
            footage = np.load(ROOT / "masks" / "lamp-2026-10-25_2fps.npz")["masks"]
            frames = np.arange(75 * 120, 112 * 120, 4)
            # Measured where the warm-up ends: the masks' wax amount flickers frame to frame.
            amounts = (footage[frames + 16].astype(np.float32) / 255 * vessel.numpy()).sum((1, 2))
            order = frames[np.argsort(np.abs(amounts - args.wax))]
            frame = next(int(f) for f in order if all(abs(f - u) > 600 for u in used))
            used.append(frame)
            real = torch.from_numpy(footage[frame : frame + 17].astype(np.float32) / 255)
            real = real * geom.vessel
            temp = torch.from_numpy(run["temps"][args.minute].astype(np.float32))[None]
            state = model.init_state(real[:1], temp)
            for k in range(16):
                state = model.step(state, geom, DT, *model.draw_randomness(state, gen))
                state.wax = real[k + 1 : k + 2]
            for _ in range(args.settle_steps):
                state = model.step(state, geom, DT, *model.draw_randomness(state, gen))
            file = f"state-{i}.bin"
            (args.out / file).write_bytes(pack_state(state).tobytes())
            meta["states"].append(file)
            print(name, "settled, wax", round(float(state.wax.sum()), 1), flush=True)

        # A check for the browser port: one step from a known state, with known randomness.
        fire, noise = model.draw_randomness(state, gen)
        after = model.step(state, geom, DT, fire, noise)
        np.savez(
            args.out / "check.npz",
            fire=fire[0, 0].numpy(), noise=noise[0].numpy(),
            wax=after.wax[0].numpy(), temp=after.temp[0].numpy(),
            hidden=after.hidden[0].numpy(), flow=after.flow[0].numpy(),
        )
        (args.out / "check_before.bin").write_bytes(pack_state(state).tobytes())
        (args.out / "check_after.bin").write_bytes(pack_state(after).tobytes())
        (args.out / "check_rand.bin").write_bytes(
            fire[0, 0].numpy().astype(np.float32).tobytes()
            + noise[0].numpy().astype(np.float32).tobytes()
        )
    (args.out / "model.json").write_text(json.dumps(meta) + "\n")


if __name__ == "__main__":
    main()
