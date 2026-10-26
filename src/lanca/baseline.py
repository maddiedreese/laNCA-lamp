"""Run the physics core alone (no learning) from the real cold start and compare to the footage.

    uv run python -m lanca.baseline                          # defaults, full 2 hours
    uv run python -m lanca.baseline --set mobility=0.08 --name faster

Outputs go to runs/physics/<name>/: simulated masks, temperature snapshots (starting points
for training), behavior stats, and a side-by-side video with the real lamp.
"""

import argparse
import json
import time

import cv2
import numpy as np
import torch

from lanca import measure
from lanca import physics as ph
from lanca.extract import ROOT

CLIP = "lamp-2026-10-25"
COMPARE = [
    ("first blob (s)", lambda s: s["phases"]["first_blob_s"]),
    ("cycling starts (s)", lambda s: s["phases"]["cycling_s"]),
    ("cycle period (s)", lambda s: s["cycling"]["cycle_period_s"]),
    ("free blobs p50", lambda s: s["cycling"]["free_blobs"]["p50"]),
    ("wax height p50", lambda s: s["cycling"]["height"]["p50"]),
    ("wax area p50", lambda s: s["cycling"]["wax_area"]["p50"]),
    ("rise speed p50", lambda s: (s["cycling"]["rise_speed"] or {}).get("p50")),
    ("sink speed p50", lambda s: (s["cycling"]["sink_speed"] or {}).get("p50")),
    ("blob area p50", lambda s: (s["cycling"]["blob_area"] or {}).get("p50")),
    ("merges / min", lambda s: s["cycling"]["merges_per_min"]),
]


def run(
    params: ph.Params, minutes: float, dt: float, device: str, seed: int = 0, noise: float = 2e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate from the first real mask. Returns (wax at 1/dt fps as uint8, temp snapshots, vessel)."""
    data = np.load(ROOT / "masks" / f"{CLIP}_2fps.npz")
    vessel = torch.from_numpy(data["vessel"])
    geom = ph.Geometry(vessel).to(device, torch.float32)
    p = {k: v.detach().to(device) for k, v in params.values().items()}
    wax0 = (
        torch.from_numpy(data["masks"][0].astype(np.float32) / 255).to(device)[None] * geom.vessel
    )
    state = ph.State(wax=wax0, temp=torch.zeros_like(wax0))
    gen = torch.Generator(device=device).manual_seed(seed)

    n = int(minutes * 60 / dt)
    snap_every = int(60 / dt)
    frames = np.zeros((n, *geom.shape), np.uint8)
    temps = []
    with torch.no_grad():
        for i in range(n):
            # A little smooth thermal noise breaks the left-right symmetry, as real convection
            # does. Per-cell noise would seed grid-sized wax fragments, so it is blurred first.
            jitter = torch.randn((1, 1, *geom.shape), generator=gen, device=device)
            jitter = torch.nn.functional.avg_pool2d(jitter, 9, 1, 4)[:, 0] * (noise * 9)
            state.temp = (state.temp + jitter * geom.vessel).clamp(0, 1)
            state = ph.step(state, geom, p, dt)
            frames[i] = (state.wax[0] * 255).round().clamp(0, 255).byte().cpu().numpy()
            if i % snap_every == 0:
                temps.append(state.temp[0].cpu().numpy().astype(np.float16))
    return frames, np.stack(temps), data["vessel"]


def side_by_side(
    real: np.ndarray, sim: np.ndarray, path, fps_in: float, speed: float = 30.0
) -> None:
    """Write real | simulated masks, upscaled 3x, as a sped-up video."""
    n = min(len(real), len(sim))
    h, w = real.shape[1:]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in * speed / 2, (6 * w + 8, 3 * h)
    )
    for i in range(0, n, 2):
        a = cv2.resize(real[i], (3 * w, 3 * h), interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(sim[i], (3 * w, 3 * h), interpolation=cv2.INTER_NEAREST)
        frame = np.concatenate([a, np.full((3 * h, 8), 128, np.uint8), b], 1)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
        cv2.putText(
            frame,
            f"{i / fps_in / 60:5.1f} min",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        writer.write(frame)
    writer.release()


def _fmt(x: float | None) -> str:
    return f"{'-':>10s}" if x is None else f"{x:10.4g}"


def compare(real: dict, sim: dict) -> str:
    lines = [f"{'':22s} {'real':>10s} {'baseline':>10s}"]
    for name, get in COMPARE:
        lines.append(f"{name:22s} {_fmt(get(real))} {_fmt(get(sim))}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="default")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument(
        "--dt", type=float, default=0.5, help="seconds per step; also the recording interval"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: PyTorch default)")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="NAME=VALUE", help="override physical constants"
    )
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    overrides = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    unknown = set(overrides) - set(ph.PARAM_BOUNDS)
    if unknown:
        ap.error(f"unknown constant(s): {', '.join(sorted(unknown))}")
    params = ph.Params(**overrides)

    out = ROOT / "runs" / "physics" / args.name
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    frames, temps, vessel = run(params, args.minutes, args.dt, args.device)
    print(f"simulated {args.minutes:.0f} min in {time.time() - start:.0f}s")

    fps = 1 / args.dt
    sim_stats, _ = measure.summarize(frames.astype(np.float32) / 255, vessel, fps)
    real_stats = json.loads((ROOT / "stats" / f"{CLIP}.json").read_text())
    np.savez_compressed(out / "sim.npz", masks=frames, temps=temps, vessel=vessel, fps=fps)
    used = {k: float(v.detach()) for k, v in params.values().items()}
    (out / "stats.json").write_text(json.dumps({"params": used, **sim_stats}, indent=2) + "\n")

    real = np.load(ROOT / "masks" / f"{CLIP}_2fps.npz")
    step = max(1, round(float(real["fps"]) / fps))
    side_by_side(real["masks"][::step], frames, out / "real_vs_baseline.mp4", fps)
    report = compare(real_stats, sim_stats)
    (out / "comparison.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
