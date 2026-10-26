"""End-to-end check that the model learns from real footage (Phase 4's acceptance test).

    uv run python -m lanca.smoke --iters 150

Trains only the network (physical constants frozen) on short windows from the cycling part
of the recording, and reports prediction error on held-out windows from the last 15 minutes
for three predictors: nothing moves (persistence), the untrained model (the physics alone),
and the trained model. This is a wiring check, not real training; that is Phase 5.
"""

import argparse
import json
import time

import numpy as np
import torch

from lanca import physics as ph
from lanca.data import Footage
from lanca.extract import ROOT
from lanca.model import LampNCA

DT = 0.5  # seconds per step; the 2 fps masks give one target per step


def masked_mse(pred: torch.Tensor, target: torch.Tensor, vessel: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * vessel).sum() / (vessel.sum() * pred.shape[0] * pred.shape[1])


def evaluate(model, footage, geom, starts, steps, device) -> float:
    wax, temp, target = footage.windows(starts, steps, device)
    with torch.no_grad():
        _, pred = model.rollout(
            model.init_state(wax, temp),
            geom,
            steps,
            DT,
            torch.Generator(device=device).manual_seed(0),
        )
    return float(masked_mse(pred, target, geom.vessel))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--steps", type=int, default=8, help="rollout length per window (0.5 s each)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    footage = Footage.load()
    stats = json.loads((ROOT / "stats" / "lamp-2026-10-25.json").read_text())
    cycling = footage.frame_index(stats["phases"]["cycling_s"])
    held_out = len(footage.masks) - footage.frame_index(15 * 60)
    train_range = (cycling, held_out - args.steps - 1)
    test_starts = np.linspace(held_out, len(footage.masks) - args.steps - 2, 16).astype(int)

    dev = args.device
    geom = ph.Geometry(torch.from_numpy(footage.vessel)).to(dev, torch.float32)
    model = LampNCA().to(dev)
    model.physics.requires_grad_(False)
    opt = torch.optim.Adam(model.network_parameters(), lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(1)

    wax, _, target = footage.windows(test_starts, args.steps, dev)
    persistence = float(masked_mse(wax[None].expand_as(target), target, geom.vessel))
    before = evaluate(model, footage, geom, test_starts, args.steps, dev)
    print(
        f"held-out error before training: persistence {persistence:.5f}, physics alone {before:.5f}"
    )

    start = time.time()
    for it in range(1, args.iters + 1):
        starts = rng.integers(*train_range, size=args.batch)
        wax, temp, target = footage.windows(starts, args.steps, dev)
        _, pred = model.rollout(model.init_state(wax, temp), geom, args.steps, DT, gen)
        loss = masked_mse(pred, target, geom.vessel)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network_parameters(), 1.0)
        opt.step()
        if it % 25 == 0 or it == 1:
            print(
                f"iter {it:4d}  train loss {loss.item():.5f}  ({time.time() - start:.0f}s)",
                flush=True,
            )

    after = evaluate(model, footage, geom, test_starts, args.steps, dev)
    print(f"held-out error after {args.iters} iterations: {after:.5f}")
    print(f"  persistence {persistence:.5f} | physics alone {before:.5f} | trained {after:.5f}")
    out = ROOT / "runs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    result = {"persistence": persistence, "physics": before, "trained": after, **vars(args)}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
