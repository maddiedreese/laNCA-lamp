# laNCA-lamp

A lava lamp simulator powered by a **physics-structured neural cellular automaton**.

The network decides how the wax flows; hard-coded physics moves the wax and heat.
Wax is conserved exactly, and blobs rise because they are hot, not because the model
learned that blobs near the bottom tend to go up.

The model is trained on wax masks derived from a two-hour recording of a real lava lamp,
starting from cold.

## Layout

```
footage/      manifest.json, CREDITS.md, processing.json, fetch.py (clips are gitignored)
masks/        per-clip wax masks (.npz) and segmentation settings (.json)
stats/        measured behavior targets per clip (cycle period, rise speed, blob sizes, ...)
src/lanca/    extract, segment, measure, physics, model, train, eval
tests/        physics-core and environment tests
runs/         checkpoints, rollout videos, plots (gitignored)
```

## Setup

Requires macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv sync
uv run python -m lanca.device   # confirms PyTorch can use the GPU (MPS)
uv run pytest
```

## Phase 2: masks and measurements

```bash
uv run python -m lanca.prepare lamp-2026-10-25              # crops, masks, qa, stats
uv run python -m lanca.prepare lamp-2026-10-25 masks qa     # re-run later stages only
```

1. **crops**: decode at 10 fps, stabilize each frame against the lamp's nose cone and base
   (the camera drifts about 100 px over the recording), and cache the glass at 256x128.
   Settings live in `footage/processing.json`.
2. **masks**: segment wax from liquid in Lab color: b* against a per-pixel threshold (corrects
   for the bulb's glow on the liquid), plus a brightness rule for the white-hot wax over the
   bulb, with fixed glare streaks filled in first. Output is a 128x64 soft mask per frame.
3. **qa**: side-by-side preview videos and a contact sheet in `runs/qa/`.
4. **stats**: behavior targets in `stats/<clip>.json`: warm-up milestones, then distributions
   over the cycling phase and per 10-minute window.

`masks/<clip>_2fps.npz` is committed; the 10 fps file exceeds GitHub's size limit and is
regenerated locally.

## Phase 3: physics core

`src/lanca/physics.py` holds the parts of the lamp that are hard-coded rather than learned:

- **Flow** from a stream function on cell corners: divergence-free by construction, with zero
  flow through the glass. The built-in flow is slow viscous (Stokes) flow driven by buoyancy,
  with wax ~100x more viscous than the liquid and stiffer when cold, solved by preconditioned
  conjugate gradients, preconditioned with an exact uniform-viscosity solve on the vessel's
  shape. The learned model adds its own stream function on top.
- **Transport** of wax and heat by conservative upwind fluxes: wax stays in [0, 1] and its
  total is exact to float rounding. Interface compression after each substep cancels the
  smearing that upwind transport causes, in proportion to the flow.
- **Cohesion**: a bounded, conservative wax/liquid exchange that pulls wax toward denser wax
  nearby, standing in for surface tension.
- **Heat**: conduction (different for wax and liquid), a bulb under the base, and cooling
  through the walls and cap. Hot wax rises, cold wax sinks.

The 13 physical constants live in `physics.Params`, bounded and ready to fit in Phase 5.

```bash
uv run python -m lanca.baseline --name default            # physics alone, from the real cold start
uv run python -m lanca.baseline --set cohesion=8 mobility=6 --name test --threads 2
```

The baseline writes simulated masks, temperature snapshots, stats and a side-by-side video to
`runs/physics/<name>/`. A 2-hour run takes about 4-8 minutes on the Mac mini's CPU.

**Status.** The guarantees hold (`tests/test_physics.py`). With the default constants, the
physics alone warms up, sends wax up and down for the whole run with a pool at the bottom,
and keeps wax as sharp-edged blobs. The simulation is chaotic, so repeated runs differ a
little; the range below is from two runs. Against the footage (cycling phase):

| | real | baseline |
|---|---|---|
| first blob | 812 s | 465-500 s |
| regular cycling from | 2831 s | 986-1322 s |
| cycle period | 66 s | 129-140 s |
| rise speed (lamp heights/s) | 0.023 | 0.0075-0.0078 |
| typical blob size (fraction of glass) | 0.083 | 0.005-0.008 |
| merges per minute | 8.5 | 6.6-9.5 |

What is still wrong: wax breaks into many small, angular pieces instead of two or three big
round blobs, and it moves at about a third of the real speed. Phase 4's learned flow and
exchange, and Phase 5's constant fitting, are meant to close those gaps.

## Phase 4: the learned model

`src/lanca/model.py` is a neural cellular automaton (NCA) that runs on top of the physics.
Each step, every cell looks at its 3x3 neighborhood (identity, Sobel and Laplacian filters)
over wax, temperature, 8 hidden channels, the physics flow's velocity, and the fixed vessel
and height channels. A small per-cell network (about 16k weights) outputs:

- a **flow correction**, as a stream function, so it is divergence-free and never crosses
  the glass;
- **wax exchange rates** on faces, for blob shape: rounding, necking, pinch-off, merging;
- **hidden-channel updates**, the model's memory.

`physics.advance` applies everything, so wax is conserved exactly and stays in [0, 1]
whatever the network outputs. Each step a random half of the cells apply their exchange and
hidden updates, with a little noise; the flow correction applies everywhere so the flow stays
smooth. The last layer starts at zero, so an untrained model is exactly the physics baseline.

`model.rollout` runs many steps, with optional gradient checkpointing for long rollouts.
Gradients reach the network and the physical constants; `flow_grad=True` also
differentiates through the flow solve, which is slower but needed to fit the constants that
only shape the flow (mobility, buoyancy, viscosity).

Speed on the Mac mini, batch of 8 lamps: about 190 ms per step on the GPU (MPS), 360 ms on
CPU. Most of it is the viscous flow solve, which is preconditioned with an exact
uniform-viscosity solve on the vessel's shape (capacitance-matrix method).

```bash
uv run python -m lanca.smoke --iters 150     # end-to-end check on real footage
```

The smoke test trains only the network (constants frozen) on 4-second windows from the
cycling phase, and measures error on fixed held-out windows from the last 15 minutes:

| predictor | held-out error |
|---|---|
| nothing moves (persistence) | 0.171 |
| physics alone (untrained model) | 0.150 |
| after 150 iterations (~3.5 min on the GPU) | 0.146 |
| after 600 iterations (~15 min) | 0.142 |

So the wiring works end to end: the model learns from real footage, steadily and stably.
These are short windows from a smooth starting temperature; the real training (longer
rollouts, temperature warm-up, behavior statistics, fitting the constants) is Phase 5.

## Footage

All training data comes from one self-recorded clip, listed in `footage/manifest.json` and
credited in `footage/CREDITS.md`. The raw video (10.8 GB) is not committed; place it at
`footage/clips/<filename>` and run `uv run python footage/fetch.py` to verify its hash.

## Roadmap

0. Repo setup
1. Footage (self-recorded)
2. Masks and behavior measurements
3. Physics core (conservative transport, heat, buoyancy), no learning
4. Learned flow rules (the NCA)
5. Training: fit physical constants, train the network, fine-tune jointly
6. Evaluation, including heat-off, cold-start and heat-up tests
