"""Phase 5, stage A: fit the physical constants to the lamp's timing and behavior.

    uv run python -m lanca.fit_constants --generations 6 --population 8 --workers 4
    uv run python -m lanca.fit_constants --init fits/tension.json --set cool_wall=0.0022 \
        --sigma 0.25 --seeds 2 --name refit

Timing only shows up over whole runs (warm-up takes ~45 minutes), and differentiating through
thousands of flow solves is impractical, so this is a derivative-free search: the
cross-entropy method in log space. Each candidate set of constants runs the physics alone for
the full 2 hours from the real cold start, and is scored by how far its behavior stats are
from the footage's. The best constants go to runs/fit/constants.json for stages B and C
(runs/fit/<name>/constants.json with --name).

Runs are chaotic, so one run per candidate rewards luck; --seeds averages each candidate's
score over several noise seeds. --init starts the search from a constants file (constants
not searched are kept fixed at its values) instead of the defaults.
"""

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lanca.extract import ROOT

# Constants to fit, searched in log space (float_temp is searched linearly).
SEARCH = [
    "mobility",
    "buoy_wax",
    "heat_rate",
    "cool_cap",
    "cool_wall",
    "visc_wax",
    "visc_melt",
    "tension",
    "float_temp",
    "wet",
    "sink_ratio",
]
LINEAR = {"float_temp"}

# Behavior stats compared to the footage, with weights. Each term is (log(sim / real))^2.
TARGETS = [
    ("first_blob_s", 1.0, lambda s: s["phases"]["first_blob_s"]),
    ("cycling_s", 1.0, lambda s: s["phases"]["cycling_s"]),
    ("cycle_period_s", 2.0, lambda s: s["cycling"]["cycle_period_s"]),
    ("rise_speed", 2.0, lambda s: (s["cycling"]["rise_speed"] or {}).get("p50")),
    ("sink_speed", 1.0, lambda s: (s["cycling"]["sink_speed"] or {}).get("p50")),
    # Blob size relative to the lamp's own wax: see measure.track.
    ("blob_share", 2.0, lambda s: (s["cycling"]["blob_share"] or {}).get("p50")),
    ("free_blobs", 0.5, lambda s: s["cycling"]["free_blobs"]["p50"] + 1),
    ("height", 1.0, lambda s: s["cycling"]["height"]["p50"]),
    ("merges", 0.5, lambda s: s["cycling"]["merges_per_min"] + 0.5),
]
MISSING = 4.0  # penalty when a stat doesn't exist at all (e.g. the lamp never starts cycling)


def score(sim: dict, real: dict) -> tuple[float, dict]:
    parts = {}
    for name, weight, get in TARGETS:
        try:
            s, r = get(sim), get(real)
        except (KeyError, TypeError):
            s, r = None, None
        if s is None or r is None or s <= 0 or r <= 0:
            parts[name] = weight * MISSING
        else:
            parts[name] = weight * math.log(s / r) ** 2
    return sum(parts.values()), parts


def evaluate(args: tuple[dict, int, float]) -> dict:
    """Run one candidate in a worker process and return its stats and score."""
    overrides, threads, minutes, seed = args
    import torch

    from lanca import measure
    from lanca import physics as ph
    from lanca.baseline import run

    torch.set_num_threads(threads)
    real = json.loads((ROOT / "stats" / "lamp-2026-10-25.json").read_text())
    start = time.time()
    try:
        frames, _, vessel = run(ph.Params(**overrides), minutes, 0.5, "cpu", seed=seed)
        stats, _ = measure.summarize(frames.astype(np.float32) / 255, vessel, 2.0)
        total, parts = score(stats, real)
    except RuntimeError as err:  # a non-finite run counts as a failure, not a crash
        stats, total, parts = None, float("inf"), {"error": str(err)}
    return {"constants": overrides, "score": total, "parts": parts, "seconds": time.time() - start}


def average(results: list[dict]) -> dict:
    """One candidate's runs over several seeds: mean score and parts."""
    parts = {
        k: float(np.mean([r["parts"].get(k, MISSING) for r in results]))
        for k in results[0]["parts"]
    }
    return {
        "constants": results[0]["constants"],
        "score": float(np.mean([r["score"] for r in results])),
        "seed_scores": [r["score"] for r in results],
        "parts": parts,
        "seconds": float(sum(r["seconds"] for r in results)),
    }


def to_vector(constants: dict) -> np.ndarray:
    return np.array([constants[k] if k in LINEAR else math.log(constants[k]) for k in SEARCH])


def from_vector(v: np.ndarray) -> dict:
    from lanca.physics import PARAM_BOUNDS

    out = {}
    for k, x in zip(SEARCH, v):
        lo, hi, _ = PARAM_BOUNDS[k]
        val = x if k in LINEAR else math.exp(x)
        out[k] = float(min(max(val, lo + 1e-6 * (hi - lo)), hi - 1e-6 * (hi - lo)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per worker")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--sigma", type=float, default=0.5, help="initial spread (log space)")
    ap.add_argument("--seeds", type=int, default=1, help="noise seeds per candidate")
    ap.add_argument("--init", type=Path, default=None, help="constants file to start from")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="NAME=VALUE", help="override starting constants"
    )
    ap.add_argument("--name", default=None, help="write to runs/fit/<name>/")
    args = ap.parse_args()

    from lanca.physics import PARAM_BOUNDS

    out = ROOT / "runs" / "fit" / (args.name or "")
    out.mkdir(parents=True, exist_ok=True)
    start = {k: v[2] for k, v in PARAM_BOUNDS.items()}
    if args.init:
        saved = json.loads(args.init.read_text())["constants"]
        start.update({k: v for k, v in saved.items() if k in PARAM_BOUNDS})
    start.update({k: float(v) for k, v in (kv.split("=") for kv in args.set)})
    fixed = {k: v for k, v in start.items() if k not in SEARCH}
    mean = to_vector(start)
    sigma = np.full(len(SEARCH), args.sigma)
    sigma[SEARCH.index("float_temp")] = 0.16 * args.sigma
    rng = np.random.default_rng(0)
    history: list[dict] = []
    best: dict | None = None

    with ProcessPoolExecutor(args.workers) as pool:
        for gen in range(args.generations):
            # Always include the current mean, so the search can't get worse than it.
            vectors = [mean] + [
                mean + sigma * rng.standard_normal(len(SEARCH)) for _ in range(args.population - 1)
            ]
            candidates = [{**fixed, **from_vector(v)} for v in vectors]
            jobs = [
                (c, args.threads, args.minutes, s) for c in candidates for s in range(args.seeds)
            ]
            runs = list(pool.map(evaluate, jobs))
            results = [
                average(runs[i * args.seeds : (i + 1) * args.seeds]) for i in range(len(candidates))
            ]
            for r in results:
                r["generation"] = gen
            history += results
            ranked = sorted(results, key=lambda r: r["score"])
            if best is None or ranked[0]["score"] < best["score"]:
                best = ranked[0]
            elite = np.stack([to_vector(r["constants"]) for r in ranked[: args.elite]])
            mean = elite.mean(0)
            sigma = np.maximum(0.7 * sigma + 0.3 * elite.std(0), 0.03)
            print(
                f"generation {gen}: best {ranked[0]['score']:.3f}, median "
                f"{np.median([r['score'] for r in results]):.3f}, overall best {best['score']:.3f}",
                flush=True,
            )
            Path(out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
            Path(out / "constants.json").write_text(json.dumps(best, indent=2) + "\n")
    print("best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
