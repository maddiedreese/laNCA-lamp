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
fits/         fitted physical constants (.json)
models/       best trained model (network plus its constants)
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
  shape. Gradients through the solve come from its adjoint (one more solve), not from the
  iterations. The learned model adds its own stream function on top.
- **Transport** of wax and heat by conservative upwind fluxes: wax stays in [0, 1] and its
  total is exact to float rounding. Interface compression after each substep cancels the
  smearing that upwind transport causes, in proportion to the flow. Gradients skip the
  direction of its interface normals, whose derivative blows up where wax is nearly flat.
- **Surface tension**, Cahn-Hilliard style: wax flows down a chemical potential that is high
  on bumps and thin bits, which shortens outlines, rounds blobs and drives pinch-off. It is a
  conservative, bounded exchange, explicit and stiff (hundreds of cheap substeps per step);
  gradients flow through only its last few substeps (8 in the model) and pass straight
  through the rest, which keeps memory bounded.
- **Heat**: conduction (different for wax and liquid), a bulb under the base, and cooling
  through the walls and cap. Hot wax rises, cold wax sinks.

The 20 physical constants live in `physics.Params`, bounded and ready to fit in Phase 5.

```bash
uv run python -m lanca.baseline --name default            # physics alone, from the real cold start
uv run python -m lanca.baseline --constants fits/refit4.json --name refit4   # best physics
```

The baseline writes simulated masks, temperature snapshots, stats and a side-by-side video to
`runs/physics/<name>/`. A 2-hour run takes about 20 minutes on the Mac mini's CPU (surface
tension's substeps; it was 4-8 minutes before).

**Status (before surface tension, with cohesion instead).** The guarantees hold
(`tests/test_physics.py`). The physics alone warmed up, sent wax up and down for the whole
run with a pool at the bottom, and kept wax as sharp-edged blobs. Runs are chaotic, so
repeats differ a little; the range is from two runs. See Phase 5 for the current numbers.

| | real | baseline |
|---|---|---|
| first blob | 812 s | 465-500 s |
| regular cycling from | 2831 s | 986-1322 s |
| cycle period | 66 s | 129-140 s |
| rise speed (lamp heights/s) | 0.023 | 0.0075-0.0078 |
| typical blob size (fraction of glass) | 0.083 | 0.005-0.008 |
| merges per minute | 8.5 | 6.6-9.5 |

Wax broke into many small, angular pieces instead of two or three big round blobs, and moved
at about a third of the real speed; surface tension (Phase 5) fixed most of the first.

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
whatever the network outputs. Each step a random half of the cells apply their hidden
updates, with a little noise; flow and exchange apply everywhere (a random on/off pattern on
exchange shredded blobs into foam over long runs). The last layer starts at zero, so an untrained model is exactly the physics baseline.

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

## Phase 5: training

```bash
uv run python -m lanca.fit_constants --generations 6 --population 8 --workers 4   # stage A
uv run python -m lanca.train --stage B --name stage_b --constants fits/refit4.json \
    --temps refit4 --flow-grad --checkpoint 2 --curriculum 16 32 64 128 --eval-free 128 \
    --iters 300  # stage B
uv run python -m lanca.baseline --checkpoint runs/train/stage_b/best.pt --name trained
```

- **Stage A** (`fit_constants.py`): derivative-free search (cross-entropy method) over 15
  constants, scoring full 2-hour physics runs against the footage's stats. Best constants go
  to `runs/fit/constants.json` (copied to `fits/stage_a.json`). The search's best score was partly luck (each candidate ran
  once, and runs are chaotic); a fresh run showed modest gains (rise speed 0.0078 -> 0.0097,
  period 129-140 s -> 122 s). Better fits need several runs per candidate.
- **Stage B** (`train.py`): trains the network on windows from the whole recording, each
  started at the baseline's temperature for that minute and warmed up on real frames, plus a
  pool of the model's own long runs scored against the real lamp at the same age.
  Losses: frame error plus behavior statistics (`losses.py`: sharpness, edges, spectrum,
  activity, vertical profile, blob size distribution). With `--flow-grad`, gradients pass through the flow solve, so
  the network learns how its changes move the buoyancy-driven flow later on. Free runs grow
  to 128 steps (64 s, about one lamp cycle).
- **Stage C**: joint fine-tune. Without gradients through the flow solve it can only move
  the minor constants, and it didn't improve on stage B.

- **Surface tension.** Measurements showed the physics itself was too ragged (edge per unit
  wax 0.46 vs 0.13 real), because cohesion pulled wax together without shortening outlines.
  Replacing it with Cahn-Hilliard surface tension, then setting tension and mobility from
  short runs and two 2-hour runs, gave the best physics so far
  (`fits/tension.json`).

Results, 2-hour free runs from the cold start (score: stage A's weighted log-error, lower is
better; it scores blob size as a share of each lamp's own wax, see below):

| | real | fitted physics, cohesion | tension physics | trained, before the gradient fix | trained, flow gradients, after the fix | trained, size loss | refit physics | trained on refit physics |
|---|---|---|---|---|---|---|---|---|
| score | - | 7.3 | 5.1 | 5.1-5.9 | 8.6 | 4.0 | 4.7 | **2.5** |
| score, absolute blob size (older) | - | 13.4 | 9.0 | 9.9-10.1 | 17.9 | **7.0** | - | - |
| first blob | 812 s | 487 s | 1126 s | 1421 s | 1152 s | 1779 s | 1550 s | 1836 s |
| regular cycling from | 2831 s | 1353 s | 2028 s | 1958 s | 1794 s | 2250 s | 2043 s | 1966 s |
| cycle period | 66 s | 122 s | 78 s | 159 s | 91 s | 161 s | 47 s | **54 s** |
| rise speed (lamp heights/s) | 0.023 | 0.0097 | 0.0071 | 0.0080 | 0.0128 | 0.0122 | 0.0142 | **0.0227** |
| sink speed | 0.019 | 0.0089 | 0.0068 | 0.0081 | 0.0158 | 0.0099 | **0.0146** | 0.0270 |
| typical blob size | 0.083 | 0.009 | 0.018 | 0.018 | 0.005 | **0.023** | 0.009 | 0.014 |
| typical blob share of all wax | 0.28 | 0.076 | 0.15 | 0.12-0.15 | 0.043 | **0.20** | 0.080 | 0.13 |
| merges per minute | 8.5 | 6.3 | 5.0 | 5.2 | 5.4 | 4.7 | 12.0 | 12.8 |
| edge per unit wax | 0.13 | 0.46 | 0.32 | 0.32 | 0.31 | **0.30** | 0.39 | 0.38 |

With surface tension the blobs are smooth and solid, the cold pool holds through warm-up, and
the cycle period is within 20% of the real lamp. Still off: blobs are about 5x too small and
tend to hug the walls, and wax rises about 3x too slowly.

**Why the first trained networks didn't learn.** Three stage B runs on the tension physics
(straight-through tension, gradients through the last tension substeps, and gradients
through the flow with 64 s windows) all left held-out behavior error flat. The cause was a
gradient bug, not the model: interface compression normalizes the wax gradient to get a unit
normal, and that normal's derivative is about 1 / |grad wax|, enormous where wax is nearly
flat. Through it, gradients grew about a million-fold in one step. In training, gradient
norms reached about 30,000 at 32-step windows, 10^12 at 64 and infinity at 128; clipped to 1
they were mostly noise, and at 128 steps every batch was skipped. Gradients now skip the
normals' direction (the simulation is unchanged), and through 16 physics steps they stay
around 10 instead of reaching 10^10.

**After the fix**, the network trained with flow gradients and windows growing to 128 steps
does learn: held-out behavior error on 64 s windows fell from 0.78 (physics alone) to 0.62
over 300 iterations (4.5 hours), with no skipped batches. In 2-hour runs, wax rises and
sinks almost twice as fast as the physics alone, closer to the real lamp, and the cycle
period is 91 s. But the network also sheds a haze of tiny fragments, so the median blob is a
speck (0.005), and early in warm-up a thin sheet of wax runs up the right wall. Blob size
dominates the score, so it scores 17.9. Its main blobs are larger than that number suggests;
the behavior losses don't penalize small debris directly (the spectrum term barely sees it),
which is the next thing to fix.

**Blob size loss.** The behavior losses barely saw fragments, so a size distribution joined
them (`losses.sizes`): the log of the share of wax removed by morphological openings (erode,
then dilate back) of radius 1, 2, 4 and 7 cells. Specks and thin sheets vanish at small radii,
real blobs survive. Trained the same way with it, the model is the first to beat the physics
alone in 2-hour runs (score 7.0 vs 9.0): no haze, no sheet up the wall, the cold pool holds
for half an hour, and blobs are the largest yet (0.023), with wax rising and sinking faster.
Its cycle is slow (161 s), and wax collects near the top.

**Why blobs were still small, and blob share.** In the footage the wax's visible area grows
as the lamp heats: 0.12 of the glass in the first 10 minutes, 0.25-0.46 once it cycles. A
simulation that conserves wax keeps 0.12, so equal blob counts meant blobs 2-3x smaller.
Thermal expansion (`expand`: melted wax takes 1 + expand times the volume) was tried first.
Matching the footage took expand = 1.5, far beyond paraffin's real 10-20%, and it stalled the
lamp: the swollen pool sat on the heater for over an hour while the lamp overheated. With a
realistic 15% it adds little area (0.14 vs 0.125) and its flow stirs wax into fragments (score
11.0 vs 5.1 without it), so it stays off (expand defaults to 0, capped at 0.3). The extra
visible area is mostly the view: the cold slab is thin and half hidden behind the base, the
melted wax is 3D blobs seen side-on.

So blob size is now scored as a share of the lamp's own wax (`blob_share` in `measure.py`):
each tracked blob's area over all the wax in that frame. The size loss already worked that
way (shares of wax removed by openings). On it, the size-loss model is best by a wider margin:
4.0 vs 5.1 for the physics alone, its blobs 0.20 of its wax against 0.28 real. Its remaining
gaps are the slow cycle (161 s vs 66 s) and rise speed (half the real lamp's).

**Refitting the constants for the cycle.** The size-loss model's cycle was slow (161 s vs
66 s) and wax rose at half the real speed. The cause was a thermostat: wax ran at almost
exactly its float temperature, so buoyancy stayed weak and wax hung near the top. Raising
wall cooling 3x alone helped (score 4.5, period 91 s), but hand-tuning further kept tipping
the lamp into frozen or overheated states. So stage A was rerun from that point
(`--init fits/tension.json --set cool_wall=0.0022`), scoring every candidate on blob share
and averaging two noise seeds per candidate (`--seeds 2`). The best constants
(`fits/refit.json`) raise the float temperature (0.43 to 0.50) and buoyancy (to 30), with
stronger heating and cap cooling and wall cooling at 0.0016. Wax now has to heat further
before it floats, then rises hard.

Alone, this physics cycles fast (47 s) and wax rises and sinks twice as fast as before, but
blobs stay small (score 4.7 in the 2-hour run; 3.7 in the search, from a run at a different
thread count, which shows how much one chaotic run can vary). Stage B trained on it, the same
way as the size-loss model, gives the best lamp so far: score 2.5, cycle period 54 s, rise
speed matching the real lamp (0.0227 vs 0.0228), large round blobs that pinch off the pool.
Its gaps: blobs are about half the real share of wax (0.13 vs 0.28), with small fragments
between them; it merges too often; the first blob comes late (31 vs 14 minutes); and a thin
sheet of wax runs up the right wall around 30-45 minutes.

**Wall wetting, sinking and stronger tension.** Three of those gaps came from the physics
itself; the physics alone shows the same fragments and wall sheets, more of them.

- *Wall sheets.* The sheet wax is cooler than its float temperature, so it is heavier than
  the liquid, and hardly stiffer than hot wax, yet it stays up: a film one or two cells thick
  barely moves against the no-slip glass. It favors the right side because that side of the
  glass leans in more (19 columns over the lamp's height vs 11), so rising blobs run into it.
  In a real lamp the liquid wets the glass and wax never touches it. `wet` adds that: extra
  chemical potential in cells against the side glass (not the floor), so surface tension
  moves wax off it. At wet = 0.5 the share of wax within 2 cells of the glass falls from
  4-11% to 1-1.4% (real: 1-2%), and the physics alone scores 2.6 instead of 4.7.
- *Sinking.* `sink_ratio` scales the buoyancy of wax colder than its float temperature. At
  0.6 it slows sinking only a little (0.026 vs 0.029); the search kept it near 1.
- *Warm-up.* A separate melting temperature was considered, but in this physics nothing
  else is tied to melting: buoyancy and stiffness both follow float_temp. The pool passes its
  float temperature within 9-15 minutes but rises as one neutral slab. Making cold wax stiffer
  (visc_melt 1.2 to 6) moved the first blob by about a minute; faster heating moves it most
  but starts cycling at 14 minutes instead of 47.
- *Tension.* The refit raised buoyancy 5x but kept surface tension, so rising wax stretched
  into necks that snapped into fragments, which then merged back (13 merges/min vs 8.5).
  With wet = 0.5, tension 40 scores 2.1: 2 free blobs (real 2), 9.2 merges/min, edge per unit
  wax 0.30. Tension 80 overshoots (blobs 0.53 of the wax, 2 merges/min). Explicit tension
  substeps scale with it, so a 2-hour run takes ~50 minutes at 40 instead of 20.

Stage A from there (`--init fits/refit.json --set wet=0.5 tension=40`, 3 generations of 5,
2 seeds each, now also searching wet and sink_ratio) gives `fits/refit2.json`: tension 37,
wet 0.37, sink_ratio 0.97, float temperature 0.49. It is the best lamp so far, and it is the
physics alone: 1.3 and 0.7 on two seeds (search: 1.0 and 0.95).

| | real | refit physics | trained on refit physics | refit2 physics (2 seeds) | trained on refit2 physics (2 seeds) |
|---|---|---|---|---|---|
| score | - | 4.7 | 2.5 | **1.3, 0.7** | 1.6, 1.2 |
| first blob | 812 s | 1550 s | 1836 s | 1892 s, **1192 s** | 2221 s, 1614 s |
| regular cycling from | 2831 s | 2043 s | 1966 s | 2074 s, 2070 s | **2596 s**, 2132 s |
| cycle period | 66 s | 47 s | 54 s | 50 s, 52 s | 54 s, 51 s |
| rise speed | 0.023 | 0.014 | **0.023** | 0.026, 0.025 | 0.029, 0.028 |
| sink speed | 0.019 | 0.015 | 0.027 | 0.027, 0.026 | 0.026, **0.023** |
| free blobs | 2 | 5 | 4 | **2, 2** | **2, 2** |
| blob share of all wax | 0.28 | 0.08 | 0.13 | 0.21, 0.21 | 0.38, 0.40 |
| merges per minute | 8.5 | 12.0 | 12.8 | 7.6, **9.4** | 5.6, 5.8 |
| edge per unit wax | 0.13 | 0.39 | 0.38 | 0.30, 0.30 | **0.29, 0.29** |

Stage B on refit2 lowers held-out behavior error on 64 s windows (0.37 to 0.29), but its
2-hour runs are no better than the physics alone: it overshoots blob size (0.38-0.40 of the
wax vs 0.28 real, with too few merges) and delays the first blob. Training also ran the GPU
out of memory once windows reached 64 steps: the faster flow and stronger tension mean more
substeps per step, so each 8-step gradient checkpoint segment held more. Segments of 2 steps
(`--checkpoint 2`) fixed it at no extra compute, since every step is recomputed once either way.

Still off in that lamp: the first blob comes late (20-32 vs 14 minutes), wax sinks about
35% too fast and sits a little high (median height 0.55 vs 0.46), and blobs are 0.21 of the
wax vs 0.28.

**Heater shape and liquid buoyancy.** Diagnostics first:

- *First blob.* The real lamp's first blob is the top of a tall plume on the left (height
  0.62), the physics' first blobs come off low on the right. Over the rest of the run, though,
  real wax rises evenly across the width (by quarter of the width, L to R: 0.09, 0.15, 0.16,
  0.09 of the rising wax), as the physics does, so the heater stays centred.
- *Height.* Free blobs sit at the right height (median 0.76 real vs 0.74-0.75). What is off is
  the pool: 39% of the real lamp's wax stays in it from 50 minutes on, 16% in the physics, and
  more of the physics' wax hangs under the cap.
- *Liquid buoyancy* (`buoy_liquid` 0, 0.25, 1) mostly moves the first blob (905, 1404, 2045 s);
  sinking stays fast at all three, and the scores (1.9, 0.9, 1.3) are within seed noise.

The heater footprint is now two constants, `heat_width` (Gaussian width across the floor, in
half-widths, default 0.6) and `heat_depth` (exponential depth up from it, in cells, default 5),
at fixed total power, so they move heat around without adding any; the defaults reproduce the
old heater exactly. Stage A from refit2 with these and `buoy_liquid` added (14 constants,
3 generations of 5, 3 seeds per candidate, 10 hours) gives `fits/refit3.json`: a narrower
heater (0.46), gentler sinking (sink_ratio 0.71), float temperature 0.46, buoy_liquid 0.62,
stiffer cold wax (visc_melt 1.95). Search score 0.64 (seeds 0.50, 0.59, 0.84).

| | real | refit2 physics (2 seeds) | refit3 physics (3 seeds) | trained on refit3, iteration 200 (2 seeds) |
|---|---|---|---|---|
| score | - | 1.3, 0.7 | **0.50**, 7.3, **0.70** | 1.5, 0.9 |
| first blob | 812 s | 1892 s, 1192 s | **888 s**, 1036 s, 1082 s | 2252 s, 1742 s |
| regular cycling from | 2831 s | 2074 s, 2070 s | 1639 s, 3913 s, 1458 s | 2505 s, 2284 s |
| cycle period | 66 s | 50 s, 52 s | **62 s**, 243 s, **67 s** | 50 s, 53 s |
| rise speed | 0.023 | 0.026, 0.025 | 0.019, **0.022**, **0.022** | 0.029, 0.028 |
| sink speed | 0.019 | 0.027, 0.026 | **0.020**, 0.021, **0.019** | 0.026, 0.024 |
| free blobs | 2 | **2, 2** | **2**, 6, **2** | **2, 2** |
| blob share of all wax | 0.28 | 0.21, 0.21 | **0.27**, 0.08, **0.31** | 0.24, 0.26 |
| merges per minute | 8.5 | 7.6, 9.4 | 6.1, 21, 5.6 | 7.5, 7.4 |
| wax height | 0.46 | 0.54, 0.56 | 0.61, 0.64, 0.62 | 0.55, 0.57 |

On two of three seeds refit3 is the best lamp yet: the first blob at 15-18 minutes, the
cycle period, sinking and blob share all within about 10% of the real lamp. The third seed
breaks into fragments (6 free blobs, 21 merges/min) and cycles every 4 minutes, so these
constants sit near a tipping point. Wax also sits higher than before (0.61-0.64 vs 0.46): only
7% of the wax is in the bottom eighth of the lamp, against 30% real.

Stage B on refit3 (from the refit2 network, `--checkpoint 2`, 200 iterations) saves a
checkpoint at every evaluation (`iter<N>.pt`), and each was scored on two 2-hour runs instead
of trusting held-out error: iterations 50, 100, 150 and 200 average 1.30, 1.32, 1.25 and 1.23
(best single run 0.94 at 200). All are worse than the physics alone: the network brings back a
late first blob (29-38 minutes), fast sinking and a short cycle (about 52 s).

Still off in that lamp: one seed in three fragments, wax sits high with too little left in
the pool (median height 0.61 vs 0.46), and regular cycling starts early (24-27 vs 47 minutes).

**The fragmenting seed, the cap, and scoring for robustness.** Following the runs in 5-minute
bins (`runs/diverge.py`, not committed) shows what the failing seed does. In every refit3 seed
the pool is gone by about an hour, where the real lamp still holds 20-60% of its wax in it
until the end. With no pool, all the wax moves as one giant blob between floor and cap. In
seed 1, at 76 minutes, it reheats on the floor and rises as a thin crescent that tears into a
dozen droplets; with no pool to catch them they never merge back. Seeds 0 and 2 survive the
same phase by luck.

The cap was not the cause: wax under it sits 0.01-0.08 below its float temperature, and the
share of wax there (10-17%) matches the footage. What is off is the warm-up. The real lamp is
still warming through the whole recording, its median wax height climbing from 0.20 at 60
minutes to 0.55 at 90; refit3 is at 0.64-0.69 by 60 minutes. The score's single height median
barely registers that, so it now also compares height at 60 and 90 minutes of lamp age
(`height_by_age` in the stats: 10-minute medians from the start of the clip, so a lamp that
starts cycling late is not counted as missing). A cooling zone of adjustable depth was added
too (`cap_depth`, at fixed total cap cooling), but single changes from refit3 show how narrow
the heat balance is: doubling cap cooling freezes the lamp, halving it parks all the wax in
one blob at the top, depths of 2 and 15 cells score worse, and 20% less heating keeps a
real-looking pool but wax no longer reaches the cap.

`fit_constants --worst W` scores each candidate as (1 - W) x its mean over seeds + W x its
worst seed, so constants that only work on average lose. Stage A from refit3 with
`--worst 0.5` (15 constants, 3 generations of 5, 3 seeds, 9 hours) gives `fits/refit4.json`:
slightly more heating and cap cooling (0.018 each), float temperature 0.47, the rest close
to refit3. Search score 0.94 (seeds 1.01, 0.70, 0.94). Scores below use the new terms, so
they are higher than before for the same runs (refit2: 1.6 and 1.1).

| | real | refit3 physics (3 seeds) | refit4 physics (3 seeds) |
|---|---|---|---|
| score (new terms) | - | 1.03, **7.9**, 1.18 | 0.97, 1.24, 0.94 |
| first blob | 812 s | 888 s, 1036 s, 1082 s | 1266 s, 1065 s, 802 s |
| regular cycling from | 2831 s | 1639 s, 3913 s, 1458 s | 3523 s, 1350 s, 1582 s |
| cycle period | 66 s | 62 s, 243 s, 67 s | 55 s, 58 s, 59 s |
| rise speed | 0.023 | 0.019, 0.022, 0.022 | 0.023, 0.025, 0.026 |
| sink speed | 0.019 | 0.020, 0.021, 0.019 | 0.024, 0.025, 0.025 |
| free blobs | 2 | 2, 6, 2 | 2, 2, 2 |
| blob share of all wax | 0.28 | 0.27, 0.08, 0.31 | 0.33, 0.20, 0.23 |
| wax height at 60 min | 0.20 | 0.65, 0.69, 0.64 | 0.64, 0.52, 0.59 |
| wax height at 90 min | 0.55 | 0.70, 0.67, 0.66 | 0.59, 0.58, 0.61 |
| pool share of wax, 100-120 min | 0.18-0.30 | 0.00-0.18 | 0.08-0.22 |

No seed of refit4 fragments, and a pool now survives to the end, but it gives back some of
refit3's best-seed accuracy: sinking is about 30% fast again and the cycle a little short.
Stage B was not rerun: on refit3 every checkpoint did worse than the physics alone.

Still off in the best lamp: the warm-up is too fast (height 0.52-0.64 at 60 minutes vs 0.20),
wax sinks about 30% too fast, and the cycle is about 10% short.

**Eight seeds, latent heat, and stage B on refit4.** Three seeds were too few to rank
anything. Over eight, refit4 scores 0.97, 1.24, 0.94, 0.94, 1.25, 4.38, 1.15 and 1.09 (mean
1.50, median 1.12): on seed 5 one giant blob holds 60-76% of the wax. The same seed does not
reproduce its score between a search and a rerun, because the runs are chaotic.

The warm-up looked like missing latent heat, so `physics` now has an optional `latent`
constant (in units of T; paraffin's is worth about 100 K, default 0). With it on, the flow
carries energy `E = T + latent x wax x melted_share(T)` and temperature is recovered from E
afterwards; heat added or lost, including the baseline's noise, pays latent heat through
`absorb_latent`. The first version stalled the lamp, but that was a leak: surface tension and
the noise moved or heated melted wax without moving its energy, removing heat about as fast
as the heater added it. With those fixed, a sealed lamp matches its energy budget at latent 0
and 0.5. The lamp still stalls with refit4's other constants, this time for a physical reason:
a ball of wax sits beside the heater.

| refit4 with | score |
|---|---|
| latent 0.25 | 18.9 |
| latent 0.5, wider heater | 20.9 |
| latent 0.5, wax conduction 0.15 | 40.9 |
| latent 0.5, both | 27.9 |

Two more searches did not beat refit4, and their constants were not added to `fits/`. A narrow
search from refit4 (`--only`, over wax viscosity, wax buoyancy, `sink_ratio` and tension) has
an 8-seed mean of 2.14, with blob share failing on 4 seeds. A full search that was free to use
latent heat (18 constants, now including `latent`, `kappa_wax` and `kappa_liquid`) drove it
down to 0.017 and has an 8-seed mean of 2.96, with blobs rising too slowly. So latent heat does
not explain the warm-up within this model. The remaining guesses, untested, are the heater's
own warm-up and heat held by the glass base.

Stage B on refit4 starts from a fresh network, because `--init` would load older physics
constants over refit4's (`--constants fits/refit4.json --temps refit4_s0 --flow-grad
--checkpoint 2 --curriculum 16 32 64 128 --eval-free 128`, 300 iterations, 6 hours). Two 2-hour
runs per checkpoint give means of 2.20, 1.72, 1.45, 1.38, 1.02 and 0.83 at iterations 50 to
300, so the last checkpoint was run on all eight seeds. It is committed as
`models/stage_b_refit4_iter300.pt`; run it with `uv run python -m lanca.baseline --checkpoint
models/stage_b_refit4_iter300.pt --name best --seed 0`.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | mean | median |
|---|---|---|---|---|---|---|---|---|---|---|
| refit4 physics | 0.97 | 1.24 | 0.94 | 0.94 | 1.25 | 4.38 | 1.15 | 1.09 | 1.50 | 1.12 |
| refit4 + network, iteration 300 | 1.08 | 0.58 | 0.71 | 0.65 | 1.49 | **1.17** | 1.16 | 2.03 | **1.11** | 1.12 |

| 8-seed mean | real | refit4 physics | refit4 + network |
|---|---|---|---|
| cycle period | 66 s | 57 s | **62 s** |
| rise speed | 0.023 | **0.025** | 0.027 |
| sink speed | 0.019 | 0.025 | **0.024** |
| wax height | 0.46 | 0.58 | 0.58 |
| wax transport (median) | 0.027 | 0.016 | 0.016 |

Unlike on refit3, this network beats the physics alone, but only by failing less badly. It wins
4 seeds and loses 4, and the medians match; without seed 5 the physics averages 1.08. What it
adds is no giant-blob disaster (worst run 2.03, against 4.38) and a cycle period closer to real.

Still off in the best lamp: height at 60 minutes is a top error on all eight seeds, wax sits
high (about 15% in the bottom eighth vs 30%), and wax transport is about 60% of real. A second
recording, ideally a cold start through at least the first hour, would show whether the warm-up
gap is in the physics or only in this one clip.

**A second recording, and a blob counter that works on both.** `lamp-2026-11-04` is the same
lamp filmed from a true cold start: the bulb comes on 2 s in, and it runs 2 h 18 min. Nothing
has been fitted to it. It shows the slow warm-up is real and repeatable, so the fault is in the
model. Median wax height per 10 minutes of lamp age:

| minutes | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | 100-110 | 110-120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lamp-2026-10-25` | 0.10 | 0.28 | 0.15 | 0.21 | 0.34 | 0.20 | 0.33 | 0.45 | 0.46 | 0.55 | 0.57 | 0.60 |
| `lamp-2026-11-04` | 0.12 | 0.19 | 0.19 | 0.19 | 0.25 | 0.26 | 0.32 | 0.43 | 0.48 | 0.52 | 0.58 | 0.61 |

In the new clip the wax first rises at about 13.5 minutes, as in the first clip, then stands as
a column that barely moves for over 20 minutes, slumps into a dome by 46 minutes, and swells into
a large balloon by 55. No model does this yet.

The new camera sits lower, so the lamp leans about 6 degrees and wax under the cap shows clearly.
Both broke the free-blob count, so `measure.components` now tests for the pool against the floor
of each column (`POOL_ROWS` is 4) and does not count wax entirely within the top 14 rows
(`CAP_ROWS`). This changes the first clip's targets too: regular cycling starts at 72 minutes, not
47, and over that later window wax height goes from 0.46 to 0.54, blob share from 0.28 to 0.41
and merges from 8.5 to 11.2 per minute. Scores below are not comparable with those above.

| target | `lamp-2026-10-25` | `lamp-2026-11-04` |
|---|---|---|
| first free blob | 812 s | 2532 s |
| regular cycling from | 4330 s | 4500 s |
| cycle period | 66 s | 72 s |
| rise speed | 0.025 | 0.022 |
| sink speed | 0.019 | 0.018 |
| blob share of all wax | 0.41 | 0.41 |
| free blobs | 2 | 2 |
| wax height | 0.54 | 0.57 |
| merges per minute | 11.2 | 14.9 |

Scoring one clip against the other gives 1.38, of which 1.29 is first blob and only 0.09
everything else. That 0.09 is the best score a model could expect, and it shows first blob is
partly chance: the first clip's plume threw off a blob that lasted 46 s, while the new clip's
stayed attached.

| 8 seeds | `lamp-2026-10-25` mean, median, worst | `lamp-2026-11-04` mean, median, worst |
|---|---|---|
| refit4 physics | 1.84, 1.73, 3.01 | 2.83, 2.87, 4.06 |
| refit4 + network, iteration 300 | **1.43, 1.25, 2.97** | **2.40, 2.25, 3.79** |

With the corrected counter, the network's lead is no longer only about one bad seed. It wins 6
of 8 seeds on the first clip, though 4 of 8 on the new one. Its lower scores on the clip it never
saw show it generalizes, though most of the higher error there is first blob and merges. Height
at 60 minutes remains wrong for both, at 0.52-0.66 against 0.32-0.33 real.

## Footage

All data comes from two self-recorded clips, listed in `footage/manifest.json` and credited in
`footage/CREDITS.md`; training has used only the first so far. The raw videos (10.8 and 12.4 GB)
are not committed; place them at `footage/clips/<filename>` and run
`uv run python footage/fetch.py` to verify their hashes.

## Roadmap

0. Repo setup
1. Footage (self-recorded)
2. Masks and behavior measurements
3. Physics core (conservative transport, heat, buoyancy), no learning
4. Learned flow rules (the NCA)
5. Training: fit physical constants, train the network, fine-tune jointly
6. Evaluation, including heat-off, cold-start and heat-up tests
