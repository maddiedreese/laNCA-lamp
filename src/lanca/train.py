"""Phase 5, stages B and C: train the network, then fine-tune network and constants together.

    uv run python -m lanca.train --stage B --name stage_b --iters 600
    uv run python -m lanca.train --stage C --name stage_c --init runs/train/stage_b/best.pt --iters 200

Each iteration mixes two kinds of rollouts:

- Real windows from anywhere in the recording, cold start included. Each starts from the
  baseline's temperature at that minute, then warms up: the model steps forward with its wax
  overwritten by the real masks, letting temperature and hidden channels settle around the
  real blobs. Then it runs free, and is scored on matching the real frames and their
  behavior statistics.
- Pool lamps: the model's own earlier end states, continued further. Each remembers its age,
  the point in the recording it corresponds to, and is scored on behavior statistics against
  the real lamp at that same age (exact positions can't be expected to match after a long
  free run). This is what keeps long runs lamp-like instead of slowly melting, freezing or
  shredding. A lamp leaves the pool after a number of continuations or when it reaches the
  held-out part of the recording.

Free-run length grows over training (a curriculum), and the last 15 minutes of footage are
held out for evaluation. Stage B trains the network with the constants fixed (from stage A's
runs/fit/constants.json when present). Stage C fine-tunes both at lower learning rates;
without flow gradients, only constants that act after the flow solve (surface tension, compression,
heat) can move, which is all a short fine-tune can afford.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from lanca import losses
from lanca import physics as ph
from lanca.data import Footage
from lanca.extract import ROOT
from lanca.model import LampNCA

DT = 0.5


def load_constants(path: Path | None) -> dict:
    if path and path.exists():
        return json.loads(path.read_text())["constants"]
    return {}


class Trainer:
    def __init__(self, args):
        self.args = args
        dev = self.dev = args.device
        self.footage = Footage.load(baseline=args.temps)
        fps = self.footage.fps
        n = len(self.footage.masks)
        self.held_out = n - round(15 * 60 * fps)
        self.geom = ph.Geometry(torch.from_numpy(self.footage.vessel)).to(dev, torch.float32)
        self.vessel = self.geom.vessel
        self.spectrum = losses.Spectrum(self.geom.shape, device=dev)

        constants = load_constants(args.constants)
        self.model = LampNCA(physics=ph.Params(**constants)).to(dev)
        if args.init:
            self.model.load_state_dict(torch.load(args.init, map_location=dev))
        if args.stage == "B":
            self.model.physics.requires_grad_(False)
            groups = [{"params": self.model.network_parameters(), "lr": args.lr}]
        else:
            groups = [
                {"params": self.model.network_parameters(), "lr": args.lr},
                {"params": self.model.physics.parameters(), "lr": args.lr_physics},
            ]
        self.opt = torch.optim.Adam(groups)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, args.iters, eta_min=args.lr * 0.1
        )
        self.gen = torch.Generator(device=dev).manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        # Pool lamps: (state, age as a frame index into the recording, continuations so far).
        self.pool: list[tuple[ph.State, int, int]] = []

    def free_steps(self, it: int) -> int:
        """Curriculum: free-run length grows in stages over training."""
        lengths = self.args.curriculum
        return lengths[min(len(lengths) - 1, it * len(lengths) // self.args.iters)]

    def warm_start(self, starts: np.ndarray, free: int):
        """Warm up on real frames (no gradients), return the state and the free-run targets."""
        warm = self.args.warmup
        wax, temp, frames = self.footage.windows(starts, warm + free, self.dev)
        state = self.model.init_state(wax, temp)
        with torch.no_grad():
            for k in range(warm):
                state = self.model.step(
                    state, self.geom, DT, *self.model.draw_randomness(state, self.gen)
                )
                state.wax = frames[k]
        return state, frames[warm:]

    def behavior_target(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            k: v.detach() for k, v in losses.behavior(frames, self.vessel, self.spectrum).items()
        }

    def iteration(self, it: int) -> dict:
        a = self.args
        free = self.free_steps(it)
        # Pool lamps that would run past the training part of the recording can't be used.
        usable = [i for i, (_, age, _) in enumerate(self.pool) if age + free + 2 < self.held_out]
        n_pool = min(a.pool_batch, len(usable))
        n_real = a.batch - n_pool
        starts = self.rng.integers(0, self.held_out - a.warmup - free - 2, size=n_real)
        state, target = self.warm_start(starts, free)
        picks = random.Random(it).sample(usable, n_pool) if n_pool else []
        if picks:
            pooled = [self.pool[i][0] for i in picks]
            state = ph.State(
                wax=torch.cat([state.wax] + [s.wax for s in pooled]),
                temp=torch.cat([state.temp] + [s.temp for s in pooled]),
                hidden=torch.cat([state.hidden] + [s.hidden for s in pooled]),
                flow=torch.cat([state.flow] + [s.flow for s in pooled]),
            )
        final, pred = self.model.rollout(
            state, self.geom, free, DT, self.gen, checkpoint_every=a.checkpoint
        )

        frame = losses.masked_mse(pred[:, :n_real], target, self.vessel)
        real_b, real_parts = losses.behavior_loss(
            losses.behavior(pred[:, :n_real], self.vessel, self.spectrum),
            self.behavior_target(target),
        )
        loss = frame + a.behavior_weight * real_b
        pool_b = torch.zeros((), device=self.dev)
        for j, i in enumerate(picks):
            # Score each pool lamp against the real lamp at the same age.
            age = self.pool[i][1]
            real_then = self.footage.frames(np.array([age]), free, self.dev)[1:]
            b, _ = losses.behavior_loss(
                losses.behavior(pred[:, n_real + j : n_real + j + 1], self.vessel, self.spectrum),
                self.behavior_target(real_then),
            )
            pool_b = pool_b + b / len(picks)
        if picks:
            loss = loss + a.behavior_weight * pool_b

        self.opt.zero_grad()
        loss.backward()
        params = [p for g in self.opt.param_groups for p in g["params"]]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if not np.isfinite(grad_norm):
            self.opt.zero_grad()  # skip a bad batch rather than poison the weights
        else:
            self.opt.step()
        self.sched.step()

        # Update the pool: continued lamps age by this rollout (or retire), and this batch's
        # real windows join as new lamps.
        def detached(b: int) -> ph.State:
            return ph.State(
                wax=final.wax[b : b + 1].detach(),
                temp=final.temp[b : b + 1].detach(),
                hidden=final.hidden[b : b + 1].detach(),
                flow=final.flow[b : b + 1].detach(),
            )

        retired = set()
        for j, i in enumerate(picks):
            _, age, uses = self.pool[i]
            if uses + 1 >= a.pool_life:
                retired.add(i)
            else:
                self.pool[i] = (detached(n_real + j), age + free, uses + 1)
        self.pool = [p for i, p in enumerate(self.pool) if i not in retired]
        for b in range(n_real):
            entry = (detached(b), int(starts[b]) + a.warmup + free, 0)
            if len(self.pool) < a.pool_size:
                self.pool.append(entry)
            elif self.rng.random() < 0.25:  # keep most of the pool's long-lived lamps
                self.pool[int(self.rng.integers(len(self.pool)))] = entry
        return {
            "iter": it,
            "free": free,
            "loss": float(loss.detach()),
            "frame": float(frame.detach()),
            "behavior_real": float(real_b.detach()),
            "behavior_pool": float(pool_b.detach()),
            "grad_norm": grad_norm,
            "pool_size": len(self.pool),
            "pool_max_age_min": max((age for _, age, _ in self.pool), default=0)
            / self.footage.fps
            / 60,
            **{f"b_{k}": v for k, v in real_parts.items()},
        }

    @torch.no_grad()
    def evaluate(self, free: int = 32) -> dict:
        """Held-out windows from the last 15 minutes: frame error and behavior error."""
        n = len(self.footage.masks)
        starts = np.linspace(self.held_out, n - self.args.warmup - free - 2, 16).astype(int)
        gen_state = self.gen.get_state()
        self.gen.manual_seed(12345)
        state, target = self.warm_start(starts, free)
        _, pred = self.model.rollout(state, self.geom, free, DT, self.gen)
        self.gen.set_state(gen_state)
        persistence = losses.masked_mse(state.wax[None].expand_as(target), target, self.vessel)
        b, _ = losses.behavior_loss(
            losses.behavior(pred, self.vessel, self.spectrum),
            losses.behavior(target, self.vessel, self.spectrum),
        )
        return {
            "frame": float(losses.masked_mse(pred, target, self.vessel)),
            "behavior": float(b),
            "persistence": float(persistence),
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=["B", "C"], default="B")
    ap.add_argument("--name", default=None)
    ap.add_argument("--init", type=Path, default=None, help="checkpoint to start from")
    ap.add_argument("--constants", type=Path, default=ROOT / "runs" / "fit" / "constants.json")
    ap.add_argument(
        "--temps",
        default="fitted",
        help="baseline run (runs/physics/<name>) for starting temperatures",
    )
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pool-batch", type=int, default=2, help="pool states per batch")
    ap.add_argument("--pool-size", type=int, default=32)
    ap.add_argument("--pool-life", type=int, default=20, help="continuations before a lamp retires")
    ap.add_argument("--warmup", type=int, default=16, help="teacher-forced steps (0.5 s each)")
    ap.add_argument("--curriculum", type=int, nargs="+", default=[8, 16, 24, 32])
    # At 1.0 the behavior terms are comparable to frame error (measured on a shakedown run);
    # at 0.1 they barely moved training.
    ap.add_argument("--behavior-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-physics", type=float, default=3e-3)
    ap.add_argument("--checkpoint", type=int, default=8, help="gradient checkpoint segment length")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out = ROOT / "runs" / "train" / (args.name or f"stage_{args.stage.lower()}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n"
    )
    trainer = Trainer(args)
    log = (out / "log.jsonl").open("a")

    first = trainer.evaluate()
    print(f"start: held-out {first}", flush=True)
    log.write(json.dumps({"iter": 0, "eval": first}) + "\n")
    best = first["frame"] + args.behavior_weight * first["behavior"]
    torch.save(trainer.model.state_dict(), out / "best.pt")
    start = time.time()
    for it in range(1, args.iters + 1):
        row = trainer.iteration(it)
        if it % args.eval_every == 0 or it == args.iters:
            ev = trainer.evaluate()
            row["eval"] = ev
            score = ev["frame"] + args.behavior_weight * ev["behavior"]
            if score < best:
                best = score
                torch.save(trainer.model.state_dict(), out / "best.pt")
            torch.save(trainer.model.state_dict(), out / "last.pt")
            consts = {k: round(float(v), 5) for k, v in trainer.model.physics.values().items()}
            print(
                f"iter {it:4d} free {row['free']:2d} | train {row['loss']:.4f} (frame {row['frame']:.4f}) | "
                f"held-out frame {ev['frame']:.4f} behavior {ev['behavior']:.4f} "
                f"(persistence {ev['persistence']:.4f}) | {time.time() - start:.0f}s",
                flush=True,
            )
            if args.stage == "C":
                print(f"  constants: {consts}", flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    print(f"done; best held-out score {best:.4f}; checkpoints in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
