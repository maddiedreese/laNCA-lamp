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
