"""Run a lamp from a hot footage frame, so it holds the footage's amount of wax.

    uv run python scripts/hot_wax.py --temps refit4_s0 --seed 0 --minutes 60
    uv run python scripts/hot_wax.py --temps refit4_s0 --physics-only --set tension=60

Prints the same comparison with the real lamp as lanca.baseline. Timing targets don't apply, since
the lamp starts hot. --temps names a finished 2-hour run, for the temperature at that point.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from lanca import measure
from lanca import physics as ph
from lanca.baseline import compare
from lanca.extract import ROOT
from lanca.model import LampNCA

CLIP = "lamp-2026-10-25"
DT = 0.5
THERMAL_NOISE = 2e-3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temps", required=True)
    ap.add_argument("--checkpoint", default="models/stage_b_refit4_iter300.pt")
    ap.add_argument("--physics-only", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--start-minute", type=int, default=85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    data = np.load(ROOT / "masks" / f"{CLIP}_2fps.npz")
    geom = ph.Geometry(torch.from_numpy(data["vessel"])).to("cpu", torch.float32)
    model = LampNCA()
    model.load_state_dict(torch.load(ROOT / args.checkpoint, map_location="cpu"))
    model.eval()
    values = {k: float(v.detach()) for k, v in model.physics.values().items()}
    values.update({k: float(v) for k, v in (kv.split("=") for kv in args.set)})
    model.physics.raw.data = ph.Params(**values).raw.data.clone()
    p = {k: v.detach() for k, v in model.physics.values().items()}

    temps = np.load(ROOT / "runs" / "physics" / args.temps / "sim.npz")["temps"]
    frame = args.start_minute * 120 + args.seed * 600
    wax = torch.from_numpy(data["masks"][frame].astype(np.float32) / 255)[None] * geom.vessel
    temp = torch.from_numpy(temps[args.start_minute].astype(np.float32))[None] * geom.vessel
    state = model.init_state(wax, temp)

    gen = torch.Generator().manual_seed(args.seed)
    n = int(args.minutes * 60 / DT)
    frames = np.zeros((n, *geom.shape), np.uint8)
    with torch.no_grad():
        for i in range(n):
            jitter = torch.randn((1, 1, *geom.shape), generator=gen)
            jitter = F.avg_pool2d(jitter, 9, 1, 4)[:, 0] * (THERMAL_NOISE * 9)
            state.temp = (state.temp + jitter * geom.vessel).clamp(0, 1)
            if args.physics_only:
                state = ph.step(state, geom, p, DT)
            else:
                state = model.step(state, geom, DT, *model.draw_randomness(state, gen))
            frames[i] = (state.wax[0] * 255).round().clamp(0, 255).byte().numpy()

    stats, _ = measure.summarize(frames.astype(np.float32) / 255, data["vessel"], 1 / DT)
    real = json.loads((ROOT / "stats" / f"{CLIP}.json").read_text())
    print(f"wax {float(wax.sum()):.1f} -> {float(state.wax.sum()):.1f}")
    print(compare(real, stats))


if __name__ == "__main__":
    main()
