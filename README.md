# laNCA-lamp

A lava lamp simulator powered by a **physics-structured neural cellular automaton**.

**[See it running live in your browser](https://www.maddiedreese.com/lava/)**

The network decides how the wax flows; hard-coded physics moves the wax and heat. Wax is
conserved exactly, and blobs rise because they are hot, not because the model learned that blobs
near the bottom tend to go up. The model is trained on wax masks from a two-hour recording of a
real lava lamp, starting from cold. A second recording, never used to train it, serves as a test.

## Results

Each lamp runs for two simulated hours from a cold start, and is scored on how closely its
behavior matches each recording (cycle timing, rise and sink speed, blob sizes, merges, height
over time). Lower is better. The two real recordings score 0.41 against each other, which is
about the best a model could reach.

| 8 seeds, both recordings | mean | median | worst | first rise (real 816 s) |
|---|---|---|---|---|
| Physics alone (`fits/refit4.json`) | 2.21 | 2.16 | 3.29 | 826-1240 s |
| **Physics + network (`models/stage_b_refit4_iter300.pt`)** | **1.67** | **1.48** | 3.18 | 682-1100 s |

Once hot, the model is close to the real lamp on cycle period, rise and sink speed, and free blob
count. Its known limitations:

- **The warm-up is wrong.** The real lamp takes about 90 minutes to climb, while the model is
  near full height by 60. Latent heat, slower liquid, layered liquid and lower noise don't explain
  it.
- **Runs vary more than the real lamp.** The real first rise repeats within 2 s; simulated ones
  differ by hundreds of seconds.
- **Too little wax by default.** A cold start holds about 40% of the wax the footage shows once
  hot. Runs started hot with the footage's amount match it better on several measures, but this
  hasn't been tested from cold.
- **Blobs are stringier than real ones.** The model is a flat slice of a 3D lamp, and real blobs
  are rounder. The web page smooths the wax it draws to hide this.

The [research log](docs/research-log.md) has the full history, including what didn't work.

## Quick start

Developed on Apple Silicon with Python 3.12 and [uv](https://docs.astral.sh/uv/). Tests also run
on CPU.

```bash
uv sync
uv run pytest
uv run python -m lanca.baseline --checkpoint models/stage_b_refit4_iter300.pt --name best --minutes 120
```

A 2-hour run takes about 45 minutes. It writes simulated masks, stats, and a side-by-side video
with the real lamp to `runs/physics/best/`.

To train the network and score its checkpoints, first run the physics alone for its starting
temperatures:

```bash
uv run python -m lanca.baseline --constants fits/refit4.json --name refit4_s0
uv run python -m lanca.train --stage B --name stage_b --constants fits/refit4.json --temps refit4_s0 \
    --flow-grad --checkpoint 2 --curriculum 16 32 64 128 --eval-free 128 --iters 300 --eval-every 50
```

That takes about 6 hours. To train on both recordings, run `scripts/temps_for_new_clip.py` and add
`--clips lamp-2026-10-25 lamp-2026-11-04 --temps refit4_s0 refit4_s0_on_new`.

`scripts/hot_wax.py` runs a lamp from hot footage frames, with the footage's amount of wax.
`lanca.export_live` exports a model and hot starting states for the browser, and needs a finished
2-hour run for its temperatures.

## Running in a browser

The [live lamp](https://www.maddiedreese.com/lava/) is a JavaScript port of the physics and network
(`assets/lava/sim.mjs` in [the website's repo](https://github.com/maddiedreese/maddie)). It runs
in a worker at about 3 steps per second on a Mac, where real time needs 2. One step from the same
state and randomness usually differs from PyTorch by about 1e-4 in wax. The largest gap seen was
2e-3, where the iterative flow solve took a different path.

It runs the best model above, from hot states with the footage's amount of wax.

## Layout

```
footage/      manifest.json, CREDITS.md, processing.json, fetch.py (clips are gitignored)
masks/        per-clip wax masks (.npz) and segmentation settings (.json)
stats/        measured behavior targets per clip (cycle period, rise speed, blob sizes, ...)
src/lanca/    extract, segment, measure, physics, model, losses, train, baseline, fit_constants,
              export_live
fits/         fitted physical constants (.json)
models/       best trained model (network plus its constants)
scripts/      experiments outside the main pipeline
docs/         research log
tests/        physics-core, model, training and environment tests
runs/         checkpoints, rollout videos, plots (gitignored)
```

## Footage

All data comes from two self-recorded clips of the same lamp, listed in `footage/manifest.json`
and credited in `footage/CREDITS.md`. The raw videos (10.8 and 12.4 GB) aren't distributed, so
extracting masks from scratch isn't possible without them. The 2 fps masks are in this repo, and
they're all that training and scoring need. The best model trained on the first recording alone.

## License

The code, model weights, masks and measurements are under the [MIT License](LICENSE). The raw
videos aren't part of this repo and remain all rights reserved.

## Roadmap

- [x] Footage, masks and behavior measurements
- [x] Physics core: conservative transport, heat, buoyancy, surface tension
- [x] Learned flow rules (the NCA)
- [x] Training: fit physical constants, train the network
- [x] Running live in a browser
- [ ] Evaluation against heat-off and heat-up recordings
- [ ] A cold start with the footage's amount of wax
