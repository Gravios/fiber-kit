# Biophysical waveform modelling (`fiber-morpho`)

Forward modelling of extracellular spike waveforms from reconstructed hippocampal
morphologies: reconstruction → compartmental cable simulation with active
channels → line-source extracellular field → a `.spk`-convention footprint on a
probe. Plus the afferent topology of CA1, and what it does to the recorded
waveform.

This exists to answer questions the sort itself cannot: **how much of the
waveform variance a sorter sees is the cell, and how much is everything else.**

---

## Contents
- [Why](#why)
- [Modules](#modules)
- [Getting morphologies](#getting-morphologies)
- [Sub-commands](#sub-commands)
- [Results on the reference geometry](#results-on-the-reference-geometry)
- [Afferent topology](#afferent-topology)
- [Constraining merges: the physiological envelope](#constraining-merges-the-physiological-envelope)
- [Shape variance and intra-chunk over-merging](#shape-variance-and-intra-chunk-over-merging)
- [The session's actual feature space](#the-sessions-actual-feature-space)
- [Confronting the model with the sort](#confronting-the-model-with-the-sort)
- [What this model does not do](#what-this-model-does-not-do)

---

## Why

Three questions from the curation work have no answer inside the recording:

1. **Is footprint shape diagnostic of cell type?** If yes, a template library
   indexed by morphology class is worth building. If no, the sorter's job is
   geometry, and typing by waveform is a category error.
2. **How large is non-drift within-unit waveform variance, and does it point in
   the same direction as drift?** A nuisance that mimics drift will be absorbed
   by a drift model and silently distort it.
3. **What sets the dendritic part of the footprint** — the part on the channels
   furthest from the peak, where drift correction has the least signal?

All three need a ground truth about the *cell*, which only a forward model
provides.

## Modules

| module | what it does |
|---|---|
| `morpho_geom` | SWC / NEURON-`.hoc` reconstruction loading, `d_lambda` compartmentalization, orientation to the probe's depth axis |
| `morpho_chan` | Na, Kdr, A-type (proximal/distal) kinetics, transcribed from the published mod files |
| `morpho_cable` | Hines cable solver, spike simulation, back-propagation profiles |
| `morpho_eap` | line-source extracellular field, band-pass, resample, `.spk`-convention windowing |
| `morpho_input` | CA1 afferent topology: pathway table, allocation onto compartments, laminar profile, synaptic drive |
| `morpho_chan_ca1` | per-cell-type CA1 channel kinetics and biophysical presets (Bezaire 2016) |
| `morpho_features` | the session's real feature space: PCAE basis, lagged projection, realignment |
| `morpho_validate` | confronts the model with a curated sort (`fiber-morpho validate`) |
| `morpho_envelope` | spike trains, per-spike footprints, and the **admissible merge envelope** |
| `morpho_archetype` | parametric cells for one-factor-at-a-time sweeps |
| `morpho_study` | the `fiber-morpho` CLI |

`morpho_eap.load_probe` delegates to `fiber_localize.load_geometry`, so a
simulated footprint and a recorded one are indexed by the same channel order.
There is deliberately no second `.probe` reader.

## Getting morphologies

The model reads **SWC** and **NEURON geometry `.hoc`** (`create` / `connect` /
`pt3dadd`, including the `sec { ... }` block form, float array indices, and
simple `for` loops).

Reconstructions used to develop this (all rat hippocampus, all public):

| source | cell | sections / compartments |
|---|---|---|
| ModelDB 87545 | CA1 pyramidal `c62564` | 158 / 1086 |
| ModelDB 55035 (Migliore, Ferrante & Ascoli 2005) | CA1 pyramidal `5038804` | 173 / 561 |
| ModelDB 101629 (Hemond et al. 2008) | CA3b pyramidal `cell1zr` | 135 / 801 |
| `github.com/mbezaire/ca1` (Bezaire et al. 2016) | reduced interneuron set | 4–17 sections |

NeuroMorpho.Org is the obvious primary source and its SWC files load directly;
it was not reachable from the sandbox this was built in, so the above are
GitHub-mirrored ModelDB accessions instead.

**Three properties of that particular set are worth knowing before trusting any
number derived from it:**

- All three detailed reconstructions ship an **identical 97 µm axon stub** (byte-
  identical `pt3dadd` coordinates). The axon in these files is a modelling
  convention, not the cell. Since the AIS is a dense sink ~30 µm from the soma,
  **axonal contribution to waveform variance is not sampled** by this set, and
  its direction after `orient()` is arbitrary — in these files it ascends with
  the apical tree, which is anatomically backwards.
- **Diameter conventions differ per archive**: `c62564` is quantized at 0.30 µm
  (5th percentile = median = 0.3); `5038804` is floored at 1.35 µm. Amplitude
  scales with membrane area, so part of any apparent cross-cell amplitude
  difference is tracing convention, not biology.
- **Seven of nine Bezaire interneuron classes share one identical 51-point
  morphology** — they differ in channels and connectivity only. That set cannot
  speak to interneuron *shape* variance, which is why `morpho_archetype` exists.

## Sub-commands

```bash
fiber-morpho variance --cells a.hoc,b.hoc,archetype:multipolar \
    --ka-scales 0.5,1.0,2.0 --rotations 0,90 --depths 0,40,80 --laterals 20,40,70

fiber-morpho bap   --cells a.hoc --ka-scales 0.25,1.0,4.0
fiber-morpho input --cells a.hoc --post pyramidalcell
fiber-morpho state --cells a.hoc --pathways ca3cell,eccell --drift 2,5,10,20

fiber-morpho envelope --cells a.hoc --detect 50 --out envelope.npz
fiber-morpho gate --envelope envelope.npz --templates candidates.npz
```

Pass `--probe <session>.probe.0.probe --channels 24,25,...` to place the real
session geometry; without it a staggered octrode stand-in is used, and a
stand-in probe biases every distance-dependent quantity downstream.

Every sub-command takes `--out results.npz` and writes the waveforms and factor
labels, so a claim can be re-checked without re-running the simulation.

## Results on the reference geometry

Numbers below are from an 8-site staggered octrode stand-in (21.65 µm pitch,
±12.5 µm stagger), σ = 0.3 S/m, 300–6000 Hz, resampled to 32552 Hz, 42-sample
window with the trough at index 21. **They have not been compared against the
reference session's sorted units** — see the caveats at the end.

### Waveform variance decomposition

6 morphologies × 3 A-current levels × 2 rotations × 9 probe positions
(324 footprints), variance in percent of total sum of squares on unit-normalized
footprints:

| factor | share |
|---|---|
| probe position (depth × lateral) | **38.3%** |
| morphology | **22.2%** |
| rotation about the depth axis | 1.4% |
| dendritic A-current (`ka_scale`) | 0.1% |
| interactions | 38.0% |

Cosine distance `1 − cos(a,b)` between footprints:

| pair | median | 5–95% |
|---|---|---|
| same morphology, different nuisance | 0.290 | 0.022 – 0.957 |
| different morphology | 0.423 | 0.151 – 0.971 |

**94.5% of different-morphology pairs fall below the within-morphology 95th
percentile.** The two distributions overlap almost completely.

The direct reading: on an octrode, **where the cell sits dominates what kind of
cell it is**, and footprint shape does not separate morphology classes. This is
consistent with the working assumption already in `fiber_geometry` — that a
single mean-template vector cannot discriminate units, and that identity has to
come from shared spikes and curve geometry rather than from template shape.

Per-cell medians over the nuisance factors:

| cell | amp µV | width ms | decay µm | spread µm |
|---|---|---|---|---|
| `ca1_5038804` | 66.1 | 0.415 | 73 | 43 |
| `ca1_c62564` | 18.4 | 0.553 | 94 | 43 |
| `ca3_cell1zr` | 42.7 | 0.553 | 133 | 70 |
| `archetype:multipolar` | 30.5 | 0.430 | 56 | 43 |
| `archetype:bipolar` | 19.0 | 0.369 | 76 | 43 |
| `archetype:granule` | 20.1 | 0.430 | 138 | 56 |

Trough-to-peak width spans 0.37–0.55 ms across morphologies with **identical
channel densities**. That is most of the width range normally used to call
narrow-spiking interneurons apart from broad-spiking pyramidal cells — so on
this model, width is not a clean cell-type axis either.

### Back-propagation

Median bAP amplitude (mV above local baseline) by path distance, sweeping the
dendritic A-current:

| cell | `ka` | V soma | 0–50 | 50–100 | 100–200 | 200–300 | 300–400 | 400–600 |
|---|---|---|---|---|---|---|---|---|
| `ca1_5038804` | 0.25 | 37.4 | 102 | 102 | 99 | 95 | 90 | 90 |
| | 1.00 | 28.1 | 90 | 89 | 79 | 72 | 60 | 8 |
| | 4.00 | 4.0 | 68 | 64 | 30 | 7 | 1 | 0 |
| `ca1_c62564` | 0.25 | 52.8 | 117 | 102 | 91 | 90 | 84 | 83 |
| | 1.00 | 44.4 | 109 | 90 | 69 | 64 | 54 | 52 |
| | 4.00 | 29.2 | 93 | 69 | 14 | 3 | 2 | 0 |

At the published density (`ka=1`) the bAP reaches the distal apical tree at
roughly half its somatic amplitude; a 4× A-current confines it to the proximal
100 µm. That is the experimentally reported strong-/weak-propagator dichotomy
(Golding, Kath & Spruston 2001), reproduced here by moving one parameter.

Note `ca1_5038804` at `ka=4` barely spikes (V soma 4.0 mV) at this stimulus — a
high dendritic A-current loads the soma as well as blocking propagation, so that
row is at the edge of its validity.

### State-dependent waveform change vs. drift

The payoff analysis. Drive one pathway, measure how far the footprint moves, and
calibrate against a probe displacement of known size (`ca1_5038804`, soma 40 µm
below site 0, 30 µm off the shank plane):

| condition | co-active synapses | ΔV dendrite | 1 − cos |
|---|---|---|---|
| CA3 (Schaffer) | 24 | 2.4 mV | 0.0023 |
| | 120 | 11.8 mV | 0.0107 |
| | 239 | 23.8 mV | 0.0283 |
| | 598 | 50.6 mV | 0.1127 |
| EC (perforant path) | 260 | 9.2 mV | 0.0060 |
| PV basket (somatic) | 19 | −0.4 mV | 0.0135 |
| OLM / Ivy / bistratified | — | ≈0 | 0.0000 |
| **probe moved +2 µm** | | | **0.0014** |
| **probe moved +5 µm** | | | **0.0064** |
| **probe moved +10 µm** | | | **0.0319** |
| **probe moved +20 µm** | | | **0.0775** |

**~120 synchronous Schaffer synapses move the footprint as much as ~5 µm of
drift; ~240 as much as ~9 µm.** That is well inside the range of synchrony a
sharp-wave ripple or a strong theta cycle delivers, and well inside the drift
range the linker is built to track.

Consequences worth taking seriously:

- This is a **within-unit, state-dependent nuisance of the same magnitude as the
  drift being corrected**, and it is correlated with firing rate and behaviour
  rather than with time. A drift model fitted to template drift alone will
  absorb some of it.
- It is **asymmetric across pathways**: Schaffer input (proximal, dense) matters;
  perforant path input (distal, sparse) is ~4× weaker per synapse-count and
  essentially invisible; dendritic inhibition does not move the spike waveform at
  all. So the effect is specific to proximal apical/basal excitation.
- It is **anisotropic across channels** in a different way from drift — it is
  carried by the dendritic sink, i.e. by the channels away from the peak, while
  drift shifts the whole footprint coherently. That difference is the handle:
  the drift-subspace Jacobian analysis already run on g5 group 4 gives the drift
  direction, and this effect should project largely *outside* it. Testing that
  is the obvious next step and has **not** been done.

## Afferent topology

`morpho_input` carries the CA1 afferent map distilled from the Bezaire et al.
(2016) full-scale model's own datasets (`cellnumbers_101` / `conndata_430` /
`syndata_120`, that model's defaults), shipped as `hippocampal_pathways.tsv`.
Regenerate from a checkout with `fiber-morpho input --bezaire <ca1>/datasets`
rather than editing the derived file.

Each pathway is a `(section list, path-distance window)` targeting rule plus a
convergence and a per-synapse conductance. `laminar_profile` re-expresses that in
the depth coordinate a probe measures. Onto `ca1_5038804`:

| pre | region | E/I | nsyn | so | sp | sr | slm | outside |
|---|---|---|---|---|---|---|---|---|
| CA3 | dendrite 50–200 | E | 11970 | 5186 | 99 | 5353 | 0 | 1331 |
| EC | dendrite 200–1000 | E | 2598 | 10 | 0 | 554 | 707 | 1327 |
| Ivy | dendrite 50–200 | I | 420 | 182 | 3 | 188 | 0 | 47 |
| CA1 recurrent | apical 100–1000 | E | 197 | 0 | 0 | 91 | 43 | 63 |
| PV basket | soma | I | 187 | 0 | 187 | 0 | 0 | 0 |
| NGF | apical 200–1000 | I | 140 | 0 | 0 | 34 | 43 | 63 |
| CCK | soma + dend <50 | I | 208 | 10 | 187 | 11 | 0 | 0 |
| Bistratified | dendrite 50–200 | I | 100 | 43 | 1 | 45 | 0 | 11 |
| OLM | apical 200–1000 | I | 80 | 0 | 0 | 19 | 25 | 36 |
| Axo-axonic | axon | I | 36 | 0 | 15 | 21 | 0 | 0 |
| SCA | dendrite 50–200 | I | 12 | 5 | 0 | 5 | 0 | 1 |

The path-distance rule splits Schaffer input almost evenly between oriens (basal)
and radiatum (proximal apical), which is anatomically right and is not something
the source model states directly — it falls out of applying its rule to real
geometry.

**Three caveats in this table, all real:**

- ~18% of synapses fall **outside** the 450 µm layer stack, because these
  reconstructions are taller than the source model's laminar coordinate. Treat
  that fraction as unassigned, not absent.
- The **CA1→CA1 recurrent weight in `conndata_430` is 0.07 µS per synapse**,
  ~350× the Schaffer value, giving 197 recurrent synapses more total conductance
  than 11970 Schaffer ones. That is almost certainly a tuning artifact of that
  particular dataset rather than biology; it is left at the published value
  rather than silently corrected, but do not use the recurrent pathway
  quantitatively without checking another `conndata_*`.
- The **axo-axonic row lands 21 synapses "in radiatum"** because the axon in
  these reconstructions ascends with the apical tree (see the axon-stub caveat
  above). It is a morphology-file artifact, not a topology error.

`Drive` turns pathways into conductance for `morpho_cable.simulate`; the
GABA-B component of the NGF `ExpGABAab` synapse is dropped (GABA-A only).

## Constraining merges: the physiological envelope

The reason the rest of this stack exists. **A merge gate calibrated on the
recording is circular** — its threshold is fitted to the same sort whose errors
it is meant to catch, so it inherits that sort's over-splitting and its false
merges alike. A gate calibrated on a forward model is not: it states what a
single neuron *can do*, independent of any clustering, and a pair of fragments
requiring more change than that cannot be the same cell however convincing the
feature-space overlap.

### The envelope is a function of amplitude ratio, not a scalar

Two fragments at the same energy are allowed almost no shape difference; two
fragments differing 1.4× in energy — one a burst's first spike, one its fourth —
are allowed considerably more, because Na availability really does reshape the
spike over that range. A single cosine threshold has to be loose enough for the
second case and is then useless for the first. That is the specific way a scalar
gate fails, and it is why `Envelope.allowed(ratio)` returns a curve.

`ca1_5038804`, 24 trains over 6 ISI patterns × 2 Na slow-inactivation floors ×
2 plateau levels × 3 probe positions, 98 detected spikes, 585 within-cell pairs,
99th percentile, **detection threshold 50 µV**:

| amplitude ratio | max admissible 1 − cos | pairs |
|---|---|---|
| 1.00 – 1.05 | 0.0143 | 248 |
| 1.05 – 1.10 | 0.0267 | 107 |
| 1.10 – 1.15 | 0.0452 | 74 |
| 1.15 – 1.21 | 0.0556 | 72 |
| 1.21 – 1.27 | 0.0585 | 42 |
| 1.27 – 1.39 | 0.0585 | 33 |
| **> 1.39** | **unreachable** | — |

Two rejection modes, reported separately because they mean different things:
the amplitude ratio exceeds anything one cell reaches, or the *shape* differs by
more than that ratio licenses.

### Detection threshold is what bounds the envelope

The largest amplitude ratio a merge can legitimately span is not set by the cell
— it is set by (largest spike) / (detection threshold), because a spike below
threshold is never detected and so never joins a cluster to be merged. Sweeping
it on the same simulations:

| `--detect` | spikes | reachable ratio (max) | 99th-pct 1 − cos (max) | variance along d(r) |
|---|---|---|---|---|
| 0 µV | 471 | 6.45 | 1.81 | 0.60 |
| 20 µV | 318 | 2.95 | 1.18 | 0.62 |
| 30 µV | 250 | 2.24 | 1.18 | 0.61 |
| **50 µV** | 98 | **1.39** | **0.059** | **0.77** |

**Set `--detect` to the pipeline's actual detection threshold.** Using 0 licenses
ratios no sorter can encounter and drags in the wild shapes of near-threshold
events.

### Unresolved: the sub-50 µV regime

Between roughly 30 and 50 µV the model produces footprints reaching
`1 − cos ≈ 1.18`, i.e. *anti-correlated* with the same cell's large spikes. This
was checked and is **not** an alignment artifact — allowing ±4 samples of shift
only moves it from 1.184 to 1.097, and the offending pair has the same peak
channel and the same trough index.

The likely explanation is real: in a heavily attenuated burst spike the somatic
sink collapses while the dendritic return current does not, so the footprint's
dominant phase inverts, and a trough-aligned extraction then cuts an essentially
arbitrary feature. Whether recorded spikes do this — and whether a real detector
would even fire on them — **has not been established.** Until it is, use the
envelope at `--detect ≥ 50 µV`, where the behaviour is clean and monotone, and
treat the low-amplitude tail as unknown rather than as licence.

### The fiber premise, tested from first principles

`along_curve_fraction` reports how much of the direction variance the energy
curve d(r) accounts for. At a 50 µV detection threshold it is **0.77** —
physiological modulation slides a unit mostly *along* one smooth curve rather
than scattering it. That is the assumption `fiber_geometry` is built on, arrived
at here from biophysics rather than from the sort.

The remaining 23% is real, though, and it is the part a curve-distance gate
cannot see. It sets a floor on how tight `DEFAULT_GEO_THR` can usefully be.

### Using it

```python
from fiber_kit.morpho_envelope import Envelope
env = Envelope.load("envelope.npz")
ok, ratio, cosd, allowed = env.admissible(template_a, template_b)
```

**Admissible is a necessary condition for a merge, never a sufficient one.** It
says only that one cell *could* have produced both. Refractory-period evidence,
drift budget and feature-space overlap all still apply — and the envelope
deliberately excludes drift, noise and spike overlap, so that a pair needing
"8 µm of drift" stays distinguishable from a pair needing a physiologically
impossible spike. Combine them downstream where the drift budget is known.

### What went into the envelope

- **Na slow inactivation** (`na3`'s `s` gate, `na_ar < 1`) — sets the decrement
  across a complex-spike burst and recovery over ~1.5 s at rest.
- **h-gate incomplete recovery** — sets the second spike of a doublet.
- **Burst plateau** (`--plateaus`) — a labelled *stand-in* for the dendritic
  calcium plateau, since this model has no calcium channels. Without it the soma
  repolarizes fully between spikes and the model reports a decrement several
  times smaller than the recorded one. Because an under-estimated envelope
  *rejects legitimate merges*, that error runs in the dangerous direction for
  over-splitting, so the substitute is provided and swept rather than omitted.
  Above ~2 nA the cell enters depolarization block and the later "spikes" are
  failures, not spikes — the `--v-thresh` and `--detect` filters remove them.
- **Probe position** is *not* pooled into a pair: pairs are formed within one
  cell at one electrode, because the gate must not contain the geometry variance
  that `variance` measures separately.


## Shape variance and intra-chunk over-merging

Within one chunk the electrode does not move and the cell does not change type,
so the only thing varying is the cell's own state. The amplitude-ratio-indexed
envelope above is therefore the *wrong tool* for the intra-chunk question: two
fragments of the same size are exactly the case it says nothing about. What
governs a within-chunk merge is **how much the shape can change with amplitude
held fixed**.

`fiber-morpho shape` measures that, using the Bezaire per-cell-type biophysics
rather than pyramidal kinetics applied to every cell. States pooled: 5 ISI
patterns × 5 synaptic conditions (none, CA3 and EC drive at 1% and 2% synchrony),
3 probe positions, pairs restricted to amplitude ratio ≤ 1.05, detection 50 µV.

| morphology | cell type | pairs | median 1−cos | p99 | max |
|---|---|---|---|---|---|
| `ca1_5038804` | pyramidal | 2079 | 0.0026 | **0.0976** | 0.0996 |
| `ca1_5038804` | pvbasket * | 2191 | 0.0086 | **0.0317** | 0.0575 |

**Physiological shape floor (99th pct): 1 − cos = 0.089.**

### The result that matters, and it is a negative

Between cell **types**, same morphology and same position: median 1 − cos =
**0.060**, 5th percentile 0.025.

**Within-cell shape variance exceeds between-type distance** — separation ratio
0.7×. At matched amplitude, two different CA1 cell types sitting at the same
place can be *closer* in shape than one cell is to itself across firing and
synaptic states.

The consequence is direct and it constrains the whole approach: **no cosine
threshold can simultaneously hold one cell together and keep two cell types
apart.** The operating threshold of 0.90 (1 − cos = 0.10) is not badly placed —
it sits just above the within-cell p99 of 0.089, so it is about right for *not
splitting* a cell. But it is intrinsically insufficient to *prevent*
over-merging, because the two distributions overlap. Tightening it will start
splitting cells before it stops merging them.

So preventing intra-chunk over-merging cannot be done on waveform shape alone.
It needs evidence orthogonal to shape — amplitude-ratio structure, ISI-dependent
amplitude (the recovery curve), and refractory violations — which is what the
`envelope` and recovery machinery supply, and why they are worth combining
rather than choosing between.

### On the real probe, in the real feature space

The numbers above use an octrode stand-in and raw footprints. With the session's
own `probe.0.probe` geometry for group 5 (channels 32–39: staggered ±11 µm,
20 µm pitch, 140 µm span — **tighter** than the stand-in's ±12.5 / 21.65 / 151.5)
and the session's order-5 `sdiffPairs` applied so simulated footprints live in
the same stderiv space as the recorded `.spk`/`.fet`:

| | median 1−cos | p99 |
|---|---|---|
| pvbasket, within-cell, matched amplitude | 0.0119 | 0.084 |
| **pooled within-cell floor** | | **0.198** |
| between cell types, same morphology and position | 0.149 | 5th pct 0.047 |

Separation ratio 0.8× — the overlap conclusion survives the move to real
geometry and real features, and is marginally worse.

**The threshold comparison flips.** In raw space 1 − cos = 0.10 sat just above
the 0.089 within-cell floor. In stderiv space the floor is **0.198**, so the
0.90 operating threshold is *below* what one cell's own state variation
produces. The transform is linear but not orthogonal — it does not preserve
angles — and removing the common mode roughly doubles the within-cell angular
spread, because common mode was much of what made two footprints of the same
cell look alike. A threshold calibrated in raw space is not the threshold that
applies to stderiv features.

Two defects in `probe.0.probe`, found while wiring this up. Both are on shanks
this work does not use, but both would silently corrupt anything run on them:

- **Shank 0, site 1 is `[-78.5179, 977.166]`** where every other shank has
  `[x0−11, 20]`. It is 977 µm out of plane.
- **Shank 3's origin is x = 532.482**, breaking the exact 200 µm shank spacing
  of the other seven (0, 200, 400, …). 532.482 + 67.518 = 600.

Groups 4 and 5 (shanks 4 and 5, x ≈ 794.5 and 994.5) are clean.

One more observation from implementing the transform: the session's order-5
pattern has **rank 7 of 8**. `sdiff_pairs.h` says order-5 sets are "generally
full rank" and treats the `SDIFF_PASS` last-channel drop as a convention for
orders 4/5 rather than a rank necessity. For *this* pattern it is a necessity —
the dropped channel is genuinely redundant, not merely least informative.


### Caveats that move this number in known directions

- **All cell types were run on the same pyramidal morphology.** Only the
  channels varied. Real PV baskets have their own dendritic geometry, and
  `variance` shows morphology is worth 22% of shape variance — so the true
  between-type distance is **larger** than 0.060, and the separation ratio is
  pessimistic. How much larger is not established here.
- **Ca and Ca-dependent K are not transcribed** (`morpho_chan_ca1.INCOMPLETE`).
  The AHP they shape lies mostly outside the 1.29 ms window and below the 300 Hz
  corner, but calcium accumulates across a burst, so a real state-dependent shape
  term is missing. The 0.089 floor is a **lower bound**.
- `bistratified` yielded one usable pair at these settings — it barely spikes
  under this drive. Its row is not usable and is shown only so its absence is
  not mistaken for a small number.
- **A safety-direction correction to the envelope section above.** Patch 0325
  argued that under-estimating the envelope was the dangerous direction, because
  it would reject legitimate merges. That reasoning was for over-*splitting*.
  For intra-chunk over-*merging* the direction inverts: an over-estimated
  envelope is what licenses fusing two real cells. The burst plateau stand-in
  (`--plateaus`) inflates the envelope, so **for intra-chunk work run it at
  `--plateaus 0.0`** and treat the plateau-inflated envelope as applying to
  cross-chunk linking only.

### Per-cell-type biophysics

`morpho_chan_ca1` transcribes the Nav family (`nav`, `navbis`, `navcck`,
`navngf`), the fast delayed rectifiers (`kdrfast`, `kdrfastngf`, fourth-order,
unlike the pyramidal first-order `kdrca1`) and the Boltzmann A-types (`kva`,
`kvangf`), with per-type densities and passive constants from each template's
`mechinit()`. A PV basket cell carries ~5× the somatic sodium of a pyramidal
cell and almost no A-current; that, plus the fourth-order rectifier, is why its
spike is narrow, and none of it is reachable by re-parameterizing a pyramidal
model.

The model's own pyramidal cell uses `ch_Navp` / `ch_Kdrp` / `ch_KvAproxp` /
`ch_KvAdistp`, which **are** the Migliore kinetics already in `morpho_chan`, so
`CA1_TYPES["pyramidal"]` routes there rather than offering a rival pyramidal
cell.


## The session's actual feature space

Everything this repo measured before `morpho_features` used cosine distance on
**waveforms**. The sort does not cluster on waveforms. It clusters on the
`.fet`, and that file is neither the waveform nor a straightforward PCA of it.

Read off g5's `.pca.stderiv.C5.D34.5` (PCAE v2): `nCh=8, data2use=31,
nComp=4, recShift=6, centered=False, method=8`.

- **Method 8 is `StderivCustomCar`** — the per-group `sdiffPairs` reference-set
  form, and `hasTemporalDiff` is true for it, so the extractor applies the
  channel difference **and** a temporal first-difference.
- **Both are already applied on disk.** Projecting `.spk` onto the left-null
  vector of the channel-mixing matrix gives 0.0033 of signal scale — zero up to
  int16 rounding. A consumer must not transform again.
- **The four "components" per channel are not four principal components.** They
  are **PC1@−3, PC1@0, PC1@+3, PC2@0**, with the lag baked into the stored
  31-sample vectors by zero-padding: `corr(c0,c1)` at lag −3 = 1.000,
  `corr(c1,c2)` at lag −3 = 1.000, `corr(c0,c2)` at lag −6 = 1.000. With
  `recShift=6`, PC1's 25-sample support sits at absolute samples 6–31, 9–34,
  12–37 — centred on peak−3, peak, peak+3. `PcaBasis.lag_structure()` recovers
  this from the file rather than assuming it.

**Verified:** projecting the recorded `.spk` through `load_pca` + `project`
reproduces the on-disk `.fet` **exactly** — 100% of 5000 × 32 values match after
rounding, max error 0.500.

### What this does to distance

Three of every four dimensions per channel are one filter at three time offsets,
so the space encodes **timing explicitly**. The 32 nominal dimensions have a
**participation ratio of 6.1** (90% of variance in 13 dims, 99% in 25). A cosine
threshold calibrated on waveforms is calibrated in the wrong geometry, and every
shape number earlier in this document is subject to that.

`to_features()` puts a simulated footprint through the identical chain —
channel difference, then temporal difference, then the lagged projection — so
model and recording land in the same 32 columns.

### Realignment: measured, and it made things worse

`fiber-morpho` can realign each cluster to its own template and reproject.
On g5's `anchor_linked` sort, with two template re-estimation passes and
`max_shift=3`:

| | |
|---|---|
| spikes moved | 25.7% |
| within-cluster RMS feature radius | 1033 → **1055** (2.2% *looser*) |
| clusters that tightened | 8.9% |

So realignment is **not** shipped as a pipeline step. Two things are established
about it and one is not:

- **The shifts are real, not estimator noise.** Split-half agreement of the
  shift estimator is 99.8% against a chance level of ~14%, with 18.8% of spikes
  non-zero.
- **The trough sits at sample 20, not the declared `peakSampleIndex=21`** —
  76% of cluster 2103's spikes trough at 20, only 9.5% at 21. The temporal
  first-difference moves the extremum by a sample. Simulated footprints are cut
  with the trough at 21, so **model and recording are offset by one sample**,
  which is a third of the lag step.
- **The mechanism is not established.** A tempting explanation — that a window
  shift is a different operation from reading a different lag — was asserted,
  written as a test, and *falsified*: in the window interior the two move along
  the same axis to corr 1.000. Whatever loosens the clusters, it is not that.

Realigning on the raw waveform rather than the differenced one is the obvious
next thing to try, by the same argument that keeps localization on
`.spk.standard`; it needs `.spk.standard.5`, which is not available here.


## Interneuron morphologies, and separation in feature space

### Cable templates

`morpho_geom.load_cable_template` loads NEURON cell templates that give **L and
diam plus a `connect` topology and no `pt3dadd`** — the form the Santhakumar /
Cutsuridis / Bezaire CA1 interneuron lineage uses. `load_hoc` still refuses
these, on purpose; this is the deliberate path.

From ModelDB 181967, all five load with a single root and total cable lengths
matching each template's own `geom()` exactly:

| template | sections | comps | total L | note |
|---|---|---|---|---|
| CA1PC | 19 | 75 | 2060 µm | obliques, SLM tuft |
| CA1BC (basket) | 17 | 73 | 1820 µm | |
| CA1AAC (axo-axonic) | 17 | 73 | 1820 µm | **identical geometry to BC** |
| CA1BSC (bistratified) | 13 | 53 | 1420 µm | no `lm*` sections — correct, it does not reach SLM |
| CA1OLM | 4 | 22 | 670 µm | two opposed horizontal dendrites |

3-D placement comes from the **section names**: `rad*` → radiatum (+y), `lm*` →
lacunosum-moleculare (+y, distal), `ori*` → oriens (−y), bare `dend` →
horizontal. That is real anatomy the cable dimensions do not carry.

Two warnings that are load-bearing:

- **This is a stylized layout, not a reconstruction.** Topology and cable
  dimensions are published; the coordinates are laid out here. Sibling fanning
  is a drawing choice with no anatomical content, and these cells have none of
  the branch-point clutter that shapes a real extracellular footprint.
- **Do not re-orient them.** The layout is already in laminar coordinates, so
  `orient()`'s default axis search stands a symmetric cell upright — it rotated
  the OLM's two horizontal oriens dendrites to vertical. Use
  `orient(c, axis=(0,1,0))`.

Simulated on g5's real geometry, soma 40 µm below site 0 and 30 µm off the
shank plane:

| cell | biophys | V soma | p2p | trough-to-peak |
|---|---|---|---|---|
| CA1PC | pyramidal | 66.3 | 58.4 µV | 0.430 ms |
| CA1BC | pvbasket | 58.4 | 85.3 µV | **0.614 ms** |
| CA1BSC | bistratified | 53.7 | 79.0 µV | 0.430 ms |
| CA1OLM | pvbasket | 60.0 | 67.0 µV | 0.369 ms |

**The basket cell comes out broader than the pyramidal cell, which is backwards**
from the narrow-spiking/broad-spiking dichotomy. It is recorded here rather than
explained; the interneuron widths should not be used until it is understood.
Candidates: `CA1_TYPES["pvbasket"]` uses `nav` at 0.15 S/cm² where the source
template uses `ch_Navaxonp`, and BC and PC are running on stylized somata of the
same 10 µm diameter.

### Separation in the sort's own metric

Redone in the 32-dim `.fet` rather than on waveforms. Within-2103 covariance is
well conditioned (cond 220, participation ratio 16.2), so Mahalanobis is
well posed. The principled threshold is **χ²(32, 0.9999) = 70.6** — and
χ²(30, 0.9999) = 67.6, matching the value already established for 30-dim data.

| clu | n | D²(frag → 2103) | D²(2103 → frag) |
|---|---|---|---|
| 2102 | 796 | 1.0 | 1.2 |
| 2108 | 744 | 1.7 | 4.4 |
| 2101 | 892 | 2.0 | 3.2 |
| 2094 | 728 | 2.0 | 2.6 |
| 2097 | 2496 | 2.2 | 1.5 |
| 2098 | 865 | 2.9 | 4.0 |
| 2105 | 1324 | 2.1 | 3.7 |
| **2100** | 142 | **35.3** | **79.7** |
| 2093 | 32 | 3.0 | 170.0 |
| 2095 | 29 | 5.0 | 442.6 |

**The asymmetry is the whole story.** Every fragment's centroid sits deep inside
2103's cloud (D² ≤ 5 for all the well-sampled ones, against a threshold of 70.6),
while 2103's centroid can be far outside *theirs*. That is the signature of a
compact sub-blob inside a broad cloud — over-splitting, not two cells.

It also names the over-merge risk directly: **a gate that uses only the large
cluster's covariance will absorb anything**, because a small tight cluster always
looks close in a broad metric. Use `max` of the two directions, not the pooled
distance, which is dominated by the larger cluster's n.

**But the reverse direction needs n ≫ 32.** The 170 and 443 for clusters of 32
and 29 spikes are artifacts — you cannot estimate a 32-dim covariance from 29
samples, and the regularizer is then setting the answer. Applying an n ≥ 10 d
rule leaves 2094, 2097, 2098, 2101, 2102, 2104, 2105, 2108, all with max D² ≤ 5.5.
Only **2100** is separated by this test as well as by shape.


## The constant per-cell span, and where ISI changes sit inside it

The covariance tests above need n ≫ 32 and are useless on the small clusters
where the question is sharpest. That was never a sample-size problem to work
around — it was a sign that a **location** statistic was the wrong tool. What
distinguishes a fragment from a cell is **dispersion**, and dispersion has one
useful number in it.

### The span is constant

Across 275 clusters with n ≥ 200 in g5's `anchor_linked` sort, the RMS distance
of a cluster's spikes from its own centroid in the 32-dim `.fet`:

| | |
|---|---|
| median radius | **993** |
| IQR | 908 – 1097 |
| CoV | **0.20** |
| radius vs template energy | log-log slope **+0.057**, R² **0.020** |

Flat. This is the feature-space form of the constant absolute scatter radius
already established on raw amplitudes (~240 ADU, 13% CoV, R² = 0.00) — the same
phenomenon, measured in the space the sort actually clusters in.

Note the population median is inflated by contaminated clusters. The **curated**
cell 2103 sits at **893**, which is the better estimate of one cell's span.

### ISI-dependent changes live inside it

Pooling 2103 with its 14 fragments (56,068 spikes) and binning by preceding ISI:

- ISI bin membership explains **0.21%** of the cell's feature variance
- that is a radius of **41** inside a cell radius of **897**

So the ISI-dependent modulation is a component *within* the constant span, at
~4.6% of the radius — not an extra tolerance a merge gate has to add on top.

### The operational consequence: under-dispersion is over-splitting

A fragment is a **compact sub-region** of a cell, so it is *tighter* than a cell,
not merely nearer. Every one of 2103's fragments is under-dispersed:

| clu | n | radius | × population median |
|---|---|---|---|
| 2108 | 744 | 794 | 0.80 |
| 2106 | 185 | 806 | 0.81 |
| 2101 | 892 | 810 | 0.82 |
| 2105 | 1324 | 831 | 0.84 |
| 2094 | 728 | 843 | 0.85 |
| 2098 | 865 | 864 | 0.87 |
| 2102 | 796 | 878 | 0.88 |
| 2104 | 366 | 886 | 0.89 |
| 2107 | 268 | 885 | 0.89 |
| 2097 | 2496 | 956 | 0.96 |

And **merging all 14 into 2103 grows the radius by 0.5%** — 893 → 897 while n
goes 47,254 → 56,068. Adding 8,814 spikes from fourteen "separate" clusters
barely moves the dispersion, which is what one cell looks like and is not what
absorbing fourteen other cells would look like.

`dispersion_verdict()` reads this out: under → fragment, over → contamination,
neither → one cell. It needs **no covariance**, so it works down to ~20 spikes,
because a radius averages n × d squared residuals rather than n of them.

This is what the biophysical model is for: to predict that span, and its
per-channel structure, from first principles rather than from the sort. The
empirical target it has to hit is the near-uniform per-channel variance
(0.113–0.135 of the total, against amplitude fractions of 0.068–0.157) reported
in the previous section.


### Cell types have different spans — and the data cannot currently show it

`fiber-morpho span` runs each cell type's state ensemble through the identical
chain into the session's 32 feature columns. Reported as **radius / |template|**,
which is dimensionless — the model produces microvolts and the recording ADU, so
an absolute comparison would be reporting the acquisition gain.

| cell | biophys | states | radius / \|template\| | rel | per-channel variance fraction |
|---|---|---|---|---|---|
| CA1PC | pyramidal | 85 | **0.434** | 1.00 | 0.08 0.09 0.22 0.27 0.07 0.09 0.04 0.15 |
| CA1BSC | bistratified | 85 | **0.431** | 0.99 | 0.08 0.17 0.37 0.12 0.09 0.01 0.14 0.03 |
| CA1OLM | pvbasket | 80 | **0.216** | 0.50 | 0.10 0.10 0.48 0.21 0.00 0.01 0.08 0.02 |
| CA1BC | pvbasket | 80 | **0.196** | 0.45 | 0.13 0.08 0.37 0.13 0.09 0.01 0.15 0.05 |

**A 2.2× spread in span between cell types** — the cells with dendrites spanning
both oriens and radiatum (pyramidal, bistratified) vary about twice as much as
the compact ones (basket, OLM), which is what a footprint dominated by a
state-dependent dendritic contribution should do.

Against the data, dimensionlessly: the curated cell 2103 sits at **0.72**, the
population median at 0.87 (IQR 0.63–1.14). So the model's physiological span is
**0.20–0.43 against an observed 0.72** — roughly 8–36% of the observed
*variance* depending on type. Physiology is **not** negligible here, which
corrects a first, unscaled reading of these numbers that put it near 1%.

**But the per-channel signature says the observed span is not mostly
physiological.** The model concentrates variance on the channels carrying the
state-dependent current (0.003 to 0.48 across channels); the data is nearly
uniform (0.113–0.135). Uniform is the signature of additive noise. Together with
the radius being constant and energy-independent, the observed ~890 reads as a
noise floor with a physiological component inside it — which is also why
over-split fragments come out *under*-dispersed: they are sub-samples of a noise
ball.

**Typing the clusters to test this directly is not currently possible.** Of 197
clusters with ≥ 300 spikes and > 10 min lifespan, **zero** meet an interneuron
firing-rate criterion — median rate is **0.17 Hz**, because on an over-split sort
a cluster's rate is the *fragment's* rate, not the cell's. Trough-to-peak width
is no help either: it is measured here on the differenced waveform, where the
literature's narrow/broad thresholds do not apply, and the model's own widths
come out backwards (see above). So the type-dependent span is a **prediction
awaiting a curated sort**, not a measured result.

Two things would test it: a sort with cells assembled, so rates become
meaningful; and `.spk.standard.5`, so width is measured where the thresholds
apply and a noise estimate can be propagated analytically through the known
linear transform to separate the uniform component from the concentrated one.


## Separating noise from physiology

> **Superseded in part.** The baseline-derived noise estimate below is unsafe on
> detected spikes, and the "79–93% recording noise" figure it produced is both
> mislabelled and low. See *What the variance actually decomposes into*, which
> replaces it. The `.spk.standard` machinery is retained because the noise
> spectrum it measures is still the right input for a forward noise model — it
> is the *share* attributed to it that was wrong.


`.spk.standard.5` makes this a measurement rather than an argument. The noise is
estimated from the **raw** pre-spike baseline — never the transformed waveform,
since the transform is exactly what we want to propagate it through — and then
pushed through the identical chain into the 32 feature columns.

Measured baseline noise on g5: SD **119–166 ADU** rising with channel index,
median inter-channel |r| **0.34**, and a strongly oscillatory autocorrelation
(lag 1 **+0.67**, lag 5 **−0.78**) — band-passed, not white, so it must be
modelled as coloured and channel-correlated.

| clu | n | observed r | \|template\| | noise r | residual r | residual/\|t\| | noise share of variance |
|---|---|---|---|---|---|---|---|
| 2103 | 47254 | 864 | 1268 | 769 | 395 | **0.312** | 79% |
| 2102 | 796 | 848 | 1318 | 766 | 363 | 0.276 | 82% |
| 2097 | 2496 | 829 | 1159 | 768 | 312 | 0.269 | 86% |
| 2105 | 1324 | 828 | 1267 | 771 | 301 | 0.237 | 87% |
| 2094 | 728 | 841 | 1218 | 809 | 230 | 0.189 | 93% |
| 2101 | 892 | 805 | 1268 | 778 | 208 | 0.164 | 93% |
| 2108 | 744 | 790 | 1304 | 763 | 204 | 0.156 | 93% |

**79–93% of the within-cluster feature variance is recording noise.** And the
per-channel signature matches: the propagated noise gives 0.110–0.136 across the
eight channels against an observed 0.114–0.136. The near-uniformity that looked
like a puzzle is simply what this noise does after the transform.

### The model and the residual agree

Residual span (radius / |template|, noise removed): **0.156 – 0.312**.
Model-predicted physiological span, same chain and probe: **0.196 – 0.434**.

The ranges overlap, and the comparison is between an **upper bound and a lower
bound**, which is the only honest way to read it: the residual still contains
drift as well as physiology, so it bounds physiology from above; the model's
state ensemble is five ISI patterns and two pathways at one probe position, so
it bounds physiology from below. They are consistent, not equal.

The ordering is also right. The curated cell 2103 has the **largest** residual
(0.312) and its fragments sit below it (0.156–0.276) — a fragment samples less
of the cell's state space, which is the same under-dispersion seen in the raw
radius, now with the noise floor removed.

Realignment shows up too: 2103's radius is **864** on the re-extracted files
against **893** before, 3.2% tighter — the opposite of what realigning on the
differenced waveform achieved.

### A bias worth recording

The first version of `baseline_noise_cov` centred the baseline **within each
spike**, across its 8 samples. Because the noise is strongly autocorrelated,
those 8 samples' mean carries far more than 1/8 of the variance, and the
estimator came out **16% low in SD and 31% low in propagated radius** — which
under-stated the noise share as 68–81% instead of 79–93%. Centring across spikes
at each (sample, channel) instead costs 1/N of a degree of freedom and removes
any residual template baseline. On the synthetic ground truth the recovered
noise radius went 0.69× → **0.99×** of the true value.

### Peak indices

Three different sample positions are in play and they are not interchangeable:

| | trough at |
|---|---|
| raw `.spk.standard` | **23** (57%), 22 (40%) |
| transformed `.spk.stderiv.C5.D34` | **20** (83%), 19 (11%) |
| declared `peakSampleIndex` | 21 |

The temporal first-difference moves the extremum 3 samples earlier. Simulated
footprints are currently cut with the trough at 21, so **the model is offset from
the raw recording by 2 samples** — two-thirds of the lag step in the D34 basis.
That is uncorrected.


## What the variance actually decomposes into

Estimating noise from the pre-spike samples of a `.spk` window is unsafe, for
reasons visible in the data itself:

- **The leading samples are not baseline.** `|template − baseline|` rises to 21
  ADU by sample 5 and 73 by sample 11 — the spike's rising phase is inside the
  window.
- **Detected-spike windows are selected events.** In a dense band the pre-spike
  region routinely contains other units' spikes, so the estimate is of
  multi-unit activity, not a noise floor.
- The block-Toeplitz truncation left **102 of 336 eigenvalues negative**, and
  clipping them *raises* the trace.

### An estimator that touches none of that

Two spikes of the same cell less than a second apart share the electrode
position, the dendritic state and the behavioural context. Whatever differs
between them is spike-independent:

`V_fast = E[ |F_i − F_j|² ] / 2` over temporally adjacent pairs.

Pooled cell 2103 + its 14 fragments, 56,068 spikes, |template| = 1256:

| component | share of variance | radius | radius / \|template\| |
|---|---|---|---|
| **fast, spike-independent** | **94.6%** | 848 | 0.675 |
| drift (between 12 time blocks) | 3.2% | 157 | 0.125 |
| slow, non-drift (≈ physiology) | **1.7%** | 128 | **0.102** |

Trimming the top 5% of residuals moves the fast share by ≤1 point, so this is
the bulk and not a handful of outliers. Pooling the fragments does not recover
more slow variance (0.162 → 0.161), so over-splitting is not truncating it.

### Two corrections this forces

**The model over-predicts physiological span by 2–4×.** Predicted 0.196–0.434;
the slow non-drift component is **0.102**, and even the whole slow term is 0.161.
Last section's claim that model and residual *agree* was an artifact of the
baseline estimator under-counting the fast term — it left an inflated residual
(0.156–0.312) that happened to overlap the model's range. It does not.

**`V_fast` is not "recording noise".** It is electrode noise **plus spike
superposition plus cluster contamination**, and the decomposition does not
separate them. Two facts rule out the easy readings: the residual is *not*
concentrated in the timing subspace (28.8% of variance against 26.4% by chance;
correlation with the measured trough index is +0.08), so it is not alignment
jitter; and the tails are far from Gaussian (99.9th percentile of the squared
residual norm is **4.4×** its mean, where χ²₃₂ gives 2.0), so a Gaussian noise
model is wrong in the tail. Only the claim that it is **spike-independent** is
supported — which is enough for the merge question, because a component that
decorrelates completely between adjacent spikes cannot be the cell's
physiological state.

### What survives

The operational conclusion is unchanged and sharper: **only ~2–5% of a cluster's
feature variance is slow enough to be physiology or drift.** A merge gate
thresholding total feature distance is thresholding a quantity that is ~95%
spike-independent — a noise-and-contamination threshold wearing a physiology
label. The dispersion test of the previous section still works, because it
compares like with like: every cluster carries the same fast term, so a cluster
that is *under*-dispersed is short of it, which is the fragment signature.


## Positively identifying the physiological component

Every measure above works by subtraction — total minus noise, total minus fast.
Subtraction cannot say what the remainder *is*. Splitting a cluster along a
feature direction and cross-correlating the halves can: **noise cannot produce a
CCG asymmetry**, because which half a spike falls in is independent of when it
fired. A state variable can, because the state evolves in time, so one half
systematically precedes the other.

On cluster 2103, splitting at the median along each residual principal axis,
after local ±60 s centring:

| PC | var % | asym 0–10 ms | asym **10–50 ms** | asym 100–150 ms | short-gap state variance |
|---|---|---|---|---|---|
| 0 | 10.9% | +0.004 | +0.017 | +0.002 | 0.02% |
| 1 | 9.2% | −0.014 | −0.043 | +0.004 | 0.02% |
| 2 | 7.6% | +0.041 | **+0.042** | +0.004 | **0.67%** |
| 3 | 7.1% | −0.022 | **−0.043** | −0.010 | **0.81%** |
| 4 | 6.7% | +0.007 | **+0.052** | −0.004 | **0.56%** |
| 5 | 6.0% | — | — | — | **0.81%** |

(shuffled controls run −0.02 to +0.02)

**Two independent signatures agree.** The axes with a CCG asymmetry are the same
axes whose adjacent-pair difference is *smaller* at 10–50 ms than at 0.3–1 s —
spikes close in time share the state. PCs 0 and 1 show neither and are noise
axes despite carrying the most variance.

### The timescale is 10–50 ms, which is neither burst nor theta period

- Not **0–10 ms**: burst-gap variance is flat (690k at 2–10 ms against 699–729k
  at 0.1–1 s), and the ISI share is 0.21%.
- Not **100–150 ms**: no asymmetry there. The split-half CCG *is* 90% theta-band
  power peaking at 6.7 Hz, but that is the cell's own firing rhythmicity, which
  any split inherits — it is not evidence about the split.
- **10–50 ms** is consistent with within-theta-cycle progression, dendritic
  integration decay, or AHP recovery. Distinguishing them needs the LFP for
  theta phase, which is not available here.

### Magnitude, and what it fixes

Summing the state-carrying axes: **~2–3% of cluster variance**. That closes a
real flaw — `V_fast` used a 1-second gap, so a 10–50 ms component was booked as
spike-independent. It was, however, small enough not to change the totals: the
figure agrees with the 1.7% slow non-drift estimate reached by subtraction.

The physiological component is now **positively identified** rather than left as
a residual, and the model comparison stands: predicted 0.196–0.434 in radius,
i.e. 8–39% of variance, against ~2–3% measured. The model over-predicts by
3–13× in variance.

### A confound that turned out not to be one

I assumed monotone drift could manufacture CCG asymmetry. It cannot, and the
test says why: a drifting feature puts early spikes in one half and late spikes
in the other, so the halves barely co-occur and there are almost no cross-pairs
at short lags — a full-range linear ramp gives −0.004. `local_center` is still
applied by default, and the test now checks the property that matters, that it
does not destroy a genuine state signal.


## Extracting LFP channels

Use **neurosuite-3's `process_extractchannels`**, which patch 0337 extends with a
record range:

```bash
process_extractchannels -s 750000 -e 3000000 \
    <session>.lfp seg32.lfp 96 64 65 66 ... 95      # 20 min, all 32 probe-1 sites

process_extractchannels <session>.lfp theta.lfp 96 72 84 43
```

Records, not seconds — `seconds × samplingRate` gives the index. The channel-spec
grammar is that tool's: `5`, `5*1.5`, `5-2`, and both orders of the combined form.

An earlier version of this work put a Python reimplementation in fiber-kit. That
was wrong. It created a second implementation of the channel-spec grammar, in
another language and another repo, with no conformance vectors — the dominant
failure mode this codebase has, and it was already incomplete (it accepted
`5*1.5-2` but not `5-2*1.5`). `.dat`/`.fil`/`.lfp` are SessionWide artifacts and
belong to ndmanager; fiber-kit reads per-group sorted artifacts.

On this side only two small helpers remain, neither duplicating ns3:
`neuro_io.session_rates()` reads nChannels and both sampling rates from the
session yaml, and `neuro_io.lfp_index()` maps a `.res` timestamp to an LFP sample,
subtracting the offset of an extracted segment. The extracted file itself opens
with the existing `neuro_io.open_signal(path, nchan)`.

## Theta phase, from a bipolar LFP — and the answer is no

The two-channel LFP slice is **channels 56 and 63**: top and bottom of probe 0's
shank 7, 140 µm apart, full session at 1250 Hz. Verified end to end — 21,108 s
by both clocks, ratio 1.0000, so `res × 1250/32552` holds at the endpoints.

### The bipolar derivation works

| | ch56 (y=0) | ch63 (y=140) |
|---|---|---|
| theta peak | 7.63 Hz | 7.63 Hz |
| theta / 1–25 Hz power | 0.43 | **0.57** |

Phase difference **−78°**, not 180° — the two sit on the *same* side of the
reversal and sample the progressive depth gradient, with ch63 deeper (more
theta). The **bipolar theta SD is 1.88× the common-mode SD**, so subtracting
genuinely suppresses the volume-conducted far field rather than just halving the
signal.

What it does *not* do is identify a layer. 140 µm constrains the gradient
between the two sites; naming its generator needs the fissure located, which
that span cannot do. It is the local gradient on shank 7.

### The positive control passes emphatically

Spike count of cluster 2103 across theta phase: modulation depth **2.22**,
max/min **21×**. The phase estimate, the derivation and the timestamp mapping
are all sound, and the cell is strongly theta-locked.

### The state axis is not theta phase

| PC | var % | state % | r circ-lin | shuffled |
|---|---|---|---|---|
| **0** | 10.9% | **0.02%** | **0.0395** | 0.0064 |
| 1 | 9.2% | 0.02% | 0.0231 | 0.0083 |
| 2 | 7.6% | **0.67%** | 0.0241 | 0.0075 |
| 3 | 7.1% | **0.81%** | 0.0250 | 0.0088 |
| 4 | 6.7% | **0.56%** | 0.0175 | 0.0094 |
| 5 | 6.0% | **0.81%** | 0.0304 | 0.0086 |

**The axis with the strongest phase correlation carries the least state**, and
the state-carrying axes are no more phase-related than the noise axes. All the
correlations are 2–5× their shuffles — significant at n ≈ 24,000, and explaining
under 0.1% of variance each.

So of the two mechanisms proposed for the 10–50 ms structure, **theta phase is
ruled out and dendritic integration is not**. That is a negative result about
theta, not a positive one about dendrites: 10–50 ms is also the timescale of AHP
recovery, and nothing here distinguishes those two.

### Two caveats that bound it

- The LFP is from **shank 7** (x ≈ 1400 µm); the units are on **shank 5**
  (x ≈ 1000 µm), 400 µm away. Theta is coherent over that distance and the 21×
  spike modulation proves the phase is meaningful for these cells — but a
  *local dendritic* signal would want the units' own shank.
- `ndm_lfp` defaults `subtractSpikes: auto`, and this session has `.spk.*`
  files, so the LFP is probably **spike-cleaned** by `ndm_stripdat`. For phase
  that is preferable — no spike bleed-through — but it means a spike-triggered
  signal has been removed, so residuals are correlated with the spike train.
  Which was used is not recorded in the slice's `.info`.


## Gamma sub-bands: also no

Theta is volume-conducted and global, so it localises nothing. Gamma is
pathway-specific — ~25–50 Hz tracks CA3→radiatum, ~60–100 Hz entorhinal→SLM,
180–240 Hz is the ripple band — so band power at spike time is a direct test of
the surviving dendritic-integration hypothesis, and it can say *which* input.

Correlation of each residual axis with log band power of the bipolar signal at
spike time, cluster 2103, n = 47,254:

| band | mean amp | PC0 | PC2 | PC3 | PC5 | max\|r\| / null |
|---|---|---|---|---|---|---|
| 25–30 Hz | 437 | −0.018 | −0.016 | +0.007 | −0.007 | 1.08 |
| 30–40 Hz | 545 | −0.012 | −0.006 | +0.002 | −0.004 | 0.92 |
| 50–60 Hz | 344 | **−0.023** | −0.014 | +0.005 | +0.002 | 1.88 |
| 60–80 Hz | 377 | −0.006 | −0.007 | +0.010 | −0.004 | 0.82 |
| 80–100 Hz | 275 | −0.012 | −0.013 | +0.012 | +0.001 | 1.13 |
| 100–120 Hz | 204 | −0.013 | −0.010 | +0.009 | −0.003 | 1.27 |
| 180–240 Hz | 157 | **−0.024** | −0.016 | +0.012 | −0.010 | 1.75 |

state variance: PC0 0.02%, PC2 0.67%, PC3 0.81%, PC5 0.81%

**Every correlation is within, or barely above, the circular-shift null.** And
the same pattern as theta repeats: the largest values sit on **PC0**, the
highest-variance axis carrying essentially no state, while the state-carrying
axes stay at |r| ≤ 0.016 in every band.

### The null has to be circular shifts, not permutation

Both series are strongly autocorrelated — band power is smooth over hundreds of
ms, a feature projection drifts over minutes. A permutation null sits near
0.006; the circular-shift null reaches 0.013–0.028. That is exactly the
difference between calling r = 0.024 a five-sigma effect and calling it nothing,
and a test asserts the shift null is the wider of the two.

### What it leaves

Local synaptic drive, as indexed by gamma in any of these bands, does not
explain the 10–50 ms state axis. Of the mechanisms on the table that leaves
**intrinsic dynamics — AHP recovery** — a cell-autonomous process with the right
time constant that need not track the local field at all.

One small positive: PC0's largest correlations are at 50–60 Hz and 180–240 Hz,
the ripple band. PC0 carries no state variance, and ripples coincide with dense
population firing, so this is consistent with part of `V_fast` being **spike
superposition** — one of the three components that decomposition could not
separate.

### The caveat that bounds this null

The whole rationale was that gamma is *local* where theta is global — and this
gamma is measured on **shank 7**, 400 µm from the units on shank 5. Gamma is
spatially structured at that scale, so a null measured 400 µm away is weaker
than a null measured on the units' own shank. **This does not rule out dendritic
integration; it rules out a relationship to gamma 400 µm away.** Two channels
from shank 5 would make it decisive.


## Correction: which channels group 5 is

`.clu.stderiv.C5.D34.**5**` is the **1-based** group 5 — `spikeDetection`
group index 4 — which is **channels 32–39**, shank index 4 at x ≈ 794.5 µm.
Not channels 40–47. Earlier sections used 40–47.

Every variance, span and shape number stands: all eight shanks are identical
staggered octrodes (±11 µm, 20 µm pitch, 140 µm span) and the geometry is
re-referenced to site 0, so the relative layout used in the simulations is
unchanged. What changes is the channel list to pass on a command line, and one
caveat that was understated — the first LFP slice (`g8rc`, channels 56–63) is
group 8, shank index 7 at x ≈ 1400 µm, so it sat **605 µm** from the units, not
400.

## Gamma on the units' own shank

With the local slice (`g5rc`, channels 32 and 39), correlations do clear the
circular-shift null — unlike at 605 µm:

| band | PC0 | PC2 | PC3 | PC5 | null p95 |
|---|---|---|---|---|---|
| 25–30 Hz | −0.010 | −0.005 | +0.004 | −0.004 | 0.010 |
| 30–40 Hz | −0.016 | −0.008 | +0.001 | −0.001 | 0.009 |
| 60–80 Hz | −0.014 | **−0.013** | +0.006 | **−0.013** | 0.008 |
| 80–100 Hz | −0.017 | **−0.014** | **+0.010** | +0.008 | 0.007 |
| 180–240 Hz | **−0.029** | **−0.019** | **+0.019** | **−0.014** | 0.008 |

state variance: PC0 0.02%, PC2 0.67%, PC3 0.81%, PC5 0.81%

**|r| grows monotonically with frequency**, strongest in the ripple band — which
is the signature of spike bleed-through into the LFP, not of synaptic drive.

### The superposition control

Detected multi-unit coincidences within ±1.5 ms are rare: mean 0.017 per spike,
1.7% of spikes have any. They correlate with **PC0 (−0.018) and PC1 (+0.014)**,
both above null, and **not** with the state axes (|r| ≤ 0.004, below null). So
superposition is real, and it lives on the no-state axes — a second, independent
identification of one `V_fast` component.

But partialling MU out barely moves the band correlations (180–240 Hz PC0:
−0.029 → −0.028; PC2: −0.019 → −0.019), and corr(band power, MU) is only +0.066
at 180–240 Hz. **Discrete superposition is not the mediator.** Undetected
multi-unit activity — spikes below threshold, or from cells off this shank —
would raise high-band power without producing a detected coincidence, and is not
excluded.

### What it amounts to

There *is* a local-LFP relationship for the state axes at 60–240 Hz, and it is
tiny: |r| ≤ 0.019 is under 0.04% of cluster variance, against state axes that
carry 0.56–0.81% each. So local high-frequency field explains a few percent of
the state variance at most. Dendritic integration is weakly supported at high
frequency and cannot account for the bulk; **AHP recovery remains the leading
candidate for the 10–50 ms structure.**


## What the state axes do to the waveform

Variance numbers cannot say *what* changes. Splitting cluster 2103 at the
quartiles of each residual axis and averaging the **raw** `.spk.standard`
waveforms does:

| PC | state % | Δp2p on peak ch | Δt_trough | largest Δ |
|---|---|---|---|---|
| **0** | 0.02% | **+10.9%** | 0 samples | ch3 (peak), 244 ADU |
| 2 | 0.67% | −3.4% | 0 | ch7 |
| 3 | 0.81% | −2.3% | 0 | ch4, 233 ADU |
| 5 | 0.81% | +13.7% | 0 | ch4 |

For PC3, the difference as a fraction of **each channel's own** peak-to-peak:

| ch0 | ch1 | ch2 | **ch3 (peak)** | ch4 | ch5 | **ch6** | ch7 |
|---|---|---|---|---|---|---|---|
| 25.6% | 5.7% | 3.8% | **5.5%** | 18.9% | 29.9% | **64.2%** | 39.3% |

**A dissociation.** The noise axis PC0 scales the *peak* channel — a gain change.
The state axes change the *distal, low-amplitude* channels, up to 64% of their
own p2p, while barely touching the peak. On ch3 the PC3 difference peaks at
sample 22, one **before** the trough at 23 — on the edge, which a gain change
cannot produce. `Δt_trough` is 0 everywhere, so none of this is alignment.

Total per-channel variance looked uniform because noise dominates it. The
*state-specific* part is strongly non-uniform and sits away from the soma —
which is where a dendritic contribution shows. So the signature is dendritic
even though the *timing* does not track local gamma.

## The real noise floor

The 20-minute `.emgclean.dat` slice (channels 32 and 39, t = 10800–12000 s)
gives the estimate the `.spk` leading samples could not. Filtering replicates
`ndm_bandpass` exactly — subtract a 33-sample moving average (`windowHalfLength
16`), then low-pass at 6 kHz. A Butterworth is not a substitute: their responses
differ by 61% below 2 kHz, and the moving-average high pass passes ~fully at
986 Hz where a Butterworth would be well into its stopband.

Excluding every sample within 30 of a detected spike leaves 95.5% of the slice
(29,334 spikes, 24.4 Hz on the group):

| | group ch0 | group ch7 |
|---|---|---|
| **true, spike-free wideband** | 149.9 (MAD 144.0) | 199.1 (MAD **174.5**) |
| `.spk` baseline, across-spike centring | 157.5 (**+5%**) | 232.8 (**+17%**) |
| `.spk` baseline, within-spike centring | 119.0 (**−21%**) | 162.7 (**−18%**) |

The centring fix of patch 0333 moved the estimator from ~20% low to 5–17% high —
so it errs conservatively now, over-attributing variance to noise.

**And SD exceeds MAD by 14% on ch7** against 4% on ch0. Gaussian noise gives
~0%. That gap is residual **undetected** multi-unit activity surviving the
spike guard, measured rather than merely conceded — and it is worse on the far,
low-amplitude channel, which is exactly where the state-axis waveform difference
is largest. The two cannot be separated with two channels.


## Fast-spiking interneurons: morphologies and Kv3

### Six reconstructed basket cells

`k_140789` (Nörenberg, Hu, Vida, Bartos & Jonas 2010, PNAS 107:894) supplies six
**detailed reconstructions of fast-spiking basket cells**, 19,791–61,922 3-D
points each, including the full axonal arbor:

| cell | sections | comps | axon | dendrite | axon share of length |
|---|---|---|---|---|---|
| BC1 | 1506 | 2678 | 26,682 µm | 1,718 | **94%** |
| BC2 | 1017 | 2201 | 17,461 | 3,756 | 82% |
| BC3 | 3171 | 3171 | 26,715 | 4,283 | 86% |
| BC4 | 3154 | 3154 | 32,564 | 5,602 | 85% |
| BC5 | 2888 | 2990 | 45,549 | 3,673 | 92% |
| BC6 | 2717 | 2923 | 51,122 | 4,238 | 92% |

**These are dentate gyrus, not CA1.** PV+ basket morphology is broadly
comparable across subfields, but the region differs and results should say so.

**82–94% of the cable is axon**, which matters directly. Simulated on the
session's real geometry, the axon contributes **22–27% of the extracellular
field on the somatic peak channels but 46–109% on distal ones** — the same
spatial pattern as the state axis (5.5% relative change on the peak channel,
64% on ch6). A basket cell's off-peak field is largely axonal, so
activity-dependent axonal excitability is a mechanism that would move exactly
those channels.

### The loader fix that made them usable

Neurolucida exports use hoc's **brace-less single-statement loop**:

```
for i = 1, 5 connect axon[i](0), axon[i-1](1)
```

`_expand_for` handled only the braced form, so 920 connect lines were partly
ignored and BC1 loaded as **977 disconnected roots**. It failed loudly rather
than simulating fragments — the `require_connected` refusal from patch 0320
doing its job — but it failed. Both forms are now expanded.

### Kv3 was a missing channel, not a wrong density

Sweeping the existing fast rectifier `Kdrfast` over **7.7×** (0.013 → 0.100
S/cm²) on BC1 moved the extracellular trough-to-peak only 0.614 → 0.461 ms and
saturated. The real g5 interneuron measures **0.271 ms**. Conductance density
cannot buy a time constant.

`Kv3` is transcribed from Akemann et al. (2009) as implemented by Zang &
De Schutter (2021), whose constants are least-squares fits to the interneuron K
current data of Martina et al. (2007). n⁴, high threshold, and much faster:

| v | Kv3 τ | Kdrfast τ |
|---|---|---|
| −70 mV | **0.177 ms** | 1.926 ms |
| −40 mV | **0.48 ms** | 3.505 ms |
| 0 mV | 0.581 ms | 0.706 ms |

The 11× faster deactivation at rest is what permits high-frequency firing.
Its absence is why patch 0330 gave basket cells *broader* spikes than pyramidal
cells — flagged there as backwards and unexplained. The model had no mechanism
for fast spiking at all.

It is placed uniformly, soma and axon: in a cell that is 90% axon by membrane, a
soma-only placement would leave the spike broad exactly where most of the field
is generated.

**Not established: whether Kv3 closes the width gap.** A 20× sweep of `gkv3`
left the measured trough-to-peak at exactly 0.430 ms while peak-to-peak rose
64.7 → 75.6 µV — so the channel acts, but the width measure is integer-sample
quantised (0.031 ms at 32552 Hz) and cannot resolve the change. The two sweeps
above are also not directly comparable: the first passed `axon_na_mult=1.0`, the
second used the default 5.0. Sub-sample width estimation on both is the next
step, and until then no claim is made that the model reproduces 0.271 ms.


### Ih — the best timescale match, and its sign problem

`HCN` transcribes the `ch_HCN` the Bezaire basket, axo-axonic and bistratified
templates insert: `g = gmax·h²`, `hinf = 1/(1+exp((v+91)/10))`,
`τ = (120 + 129.5/(1+exp((v+59.3)/0.83)))/q10`.

| v | h∞ | τ |
|---|---|---|
| −100 mV | 0.711 | 250 ms |
| −80 | 0.250 | 249 ms |
| −60 | 0.043 | 211 ms |
| −50 | 0.016 | 120 ms |

**τ runs 120–250 ms — the best match in the module to the measured adaptation
profile**, whose marginal R² peaks at τ = 50–200 ms and which also carries an
independent slow (0.5–10 s) component. One channel spanning both is more
parsimonious than two mechanisms, and second-order gating slows the effective
onset further. It is ~2000× slower than Kv3, so the two cannot substitute.

`INCOMPLETE` for `pvbasket` and `axoaxonic` narrows from `(HCN, Ca, KCa)` to
`(Ca, KCa)`.

**Two arguments against it being the dominant term, neither settled:**

- **The sign.** With half-activation at −91 mV and a fast-spiking duty cycle
  near 2%, a train leaves the cell hyperpolarized far more than depolarized, so
  Ih should *activate*, depolarize between spikes, and reduce Na availability —
  predicting **smaller** spikes at short ISI. The measurement is **+11% larger**.
  That is the same wrong sign as Kv3 and Na inactivation. Resurgent Na
  (Nav1.6/β4) remains the only candidate with the right direction.
- **State dependence.** Ih is cAMP-modulated and should therefore vary with
  neuromodulatory tone, yet the measured state component is invariant across
  theta/non-theta and across 162 minutes of session time.

So Ih is added because it is a real, documented gap with the right kinetics —
not because it explains the facilitation. It does not.


### Resurgent sodium — necessary, not sufficient

`NaRsg` transcribes the 13-state Raman & Bean scheme from `narsg.mod` (Khaliq,
Gouwens & Raman 2003, as distributed with Zang & De Schutter 2021): five closed
states, open, a **blocked** state, and six inactivated. It is not a product of
independent gates, so it needs the Markov path added to `morpho_cable`.

**Why it was worth the machinery.** The measured cell shows spikes ~11% *larger*
after short intervals. Kv3, Na inactivation and Ih all predict the opposite
sign. Resurgent sodium is the only candidate that facilitates: during
repolarisation the blocked state unbinds back *through* the open state rather
than through inactivation, so a preceding spike leaves channels poised to reopen.

**Channel validation**, independent of any cell: `alfac` 3.500 and `btfac` 0.316
match the published reversibility factors (derived from `Oon/Con` and
`Ooff/Coff`, not free); generator columns sum to zero to 3.6e-12; and the
voltage-step protocol reproduces the resurgent current — 50% of channels
accumulate blocked at +30 mV, and on repolarisation to −30 the open probability
rises again to **7.2×** its end-of-step value. Removing the O↔B transition
collapses that to **1.0×**, so it comes from the block and not from the
activation chain.

**Two-parameter sweep**, ratio of 2nd to 1st spike at 5 ms ISI (target 1.11):

| gna \ gnarsg | 0.000 | 0.004 | 0.008 | 0.015 | 0.030 |
|---|---|---|---|---|---|
| 0.040 | 0.850 | 0.894 | 0.962 | 1.024 | 0.931 |
| 0.060 | 0.884 | 0.947 | 1.011 | 1.037 | 0.938 |
| 0.090 | 0.934 | 1.002 | 1.047 | 1.037 | 0.954 |
| 0.120 | 0.977 | 1.031 | 1.051 | 1.028 | 0.973 |
| 0.150 | 0.994 | 1.040 | **1.061** | 1.029 | 0.982 |

**Without resurgent sodium nothing facilitates** — the whole `gnarsg = 0` column
depresses, approaching 1.0 as `gna` rises but never crossing it. With it, a
facilitation ridge appears at `gnarsg` ≈ 0.008–0.015 and collapses beyond,
presumably because too much charge sits blocked to be recovered.

**Best is 1.061 against a measured 1.11.** Over half the effect, right sign,
right mechanism — and a real gap. Candidates for the remainder, none tested: the
model cell is a *dentate* basket cell, the drive is somatic current injection
rather than synaptic, Ca and KCa are still absent, and part of the measured
+11% at 2–5 ms could be ringing from the preceding spike through the 33-sample
moving-average high pass.

**Cost.** A batched 13×13 solve per compartment per step, so the Markov path is
opt-in per *simulation* via the `Biophys` density rather than per cell type —
`CA1_TYPES` describes the cell, not the run. Integration is implicit because the
fastest rates reach ~10⁴/ms at spike potentials, where an explicit step at
dt = 0.02 ms diverges rather than merely blurring.


## Acquiring morphologies: `fiber-morpho-fetch`

Adding a reconstruction is not "download a file". Three things have gone wrong
in this work in ways that only surfaced later:

- a **dentate** basket cell was used where a **CA1** one was needed, and the
  substitution had to be carried as a caveat through five patches;
- **four of twelve** NeuroMorpho files tested here have **no axon**, and a
  dendrite-only interneuron is useless for this — the axon carries 46–109% of
  the off-peak extracellular field;
- a reconstruction loaded as **977 disconnected roots** because of a parser gap,
  and only failed loudly because `morpho_geom` refuses that.

So every acquisition passes a **validation gate** and lands in a **manifest**
with provenance. A morphology not in the manifest is not one this project uses.

The console script comes from `pyproject.toml`, so it only appears after an
install: `pip install -e .`. Without that, use `python3 -m fiber_kit.morpho_fetch`
from `src/` — identical behaviour.

```bash
# find candidates (no download)
fiber-morpho-fetch search --fq brain_region:CA1 --fq cell_type:interneuron \
    --q species:rat --json hits.json

# download, gate on axon length, record provenance
fiber-morpho-fetch fetch --fq brain_region:CA1 --fq cell_type:interneuron \
    --dest morph/ --min-axon 5000

# files you fetched by hand — no network needed
fiber-morpho-fetch adopt morph/*.swc --manifest morph/morphologies.tsv --min-axon 5000

# re-check that files still match their recorded hashes
fiber-morpho-fetch verify --manifest morph/morphologies.tsv --dir morph/
```

The manifest records `axon_um` and `dend_um` alongside archive, species, region
and a SHA-1, so the region substitution and the dendrite-only case are
*checkable* rather than remembered. It is written sorted and de-duplicated, so a
diff between runs shows what changed rather than how the API ordered its reply.

Demonstrated on the real files to hand:

| file | axon | dend | verdict at `--min-axon 5000` |
|---|---|---|---|
| `EC2-609291-4.CNG.swc` | 31,245 µm | 12,037 | **adopted** |
| `AKO60sdax2lay.CNG.swc` | 6,171 µm | 2,343 | **adopted** |
| `l22.swc` | 0 | 2,900 | rejected |
| `A1-May29-IR1-5-O.CNG.swc` | 0 | — | rejected |

### The network code is unverified

`neuromorpho.org` is unreachable from the environment this was written in — the
egress proxy blocks the host and its `robots.txt` disallows automated access —
so **`search()` and `download()` have never been run against the live service**.
Their URL construction follows the documented API and one observed file URL
(`dableFiles/hamad/CNG version/int27_3_1.CNG.swc`), and is unit-tested against
those; the HTTP round trip is not tested.

The archive path segment is inferred to be the archive name lower-cased with
spaces removed, from that single example rather than from documentation, so
`swc_urls` returns **four candidates** — standardised and source versions of
both the slugified and the raw archive string — and `download` tries each. If a
multi-word archive breaks all four, the neuron's own page carries the link.

Everything downstream of a file existing on disk **is** tested, which is the
path that matters if you fetch by hand — and `adopt` exists precisely so that
route is first-class rather than a workaround.


## Localisation: position as a merge and link criterion

`morpho_localize` turns a cluster's per-channel amplitude profile into a
**position**. That is evidence built on none of what the existing criteria use —
not waveform cosine, not refractory statistics, not feature-space distance.

**Affordable because the transfer matrix is time-independent.** One simulation
per morphology serves the whole position grid: 1,372 positions built in **6 s**,
after which every cluster and every atom is a table search. Nothing re-simulates
per cluster.

### Measured on g5 group 5

| | |
|---|---|
| resolution | split-half floor **2.5 µm** at ~4,000 spikes; RMSE triples over 5 µm of depth |
| co-localisation | clusters **262 and 263 agree to 2.5 µm** across eight 24-min blocks — exactly the floor |
| drift | one cluster's atoms trace **152.5 → 175 µm** over five hours |
| two cells in one cluster | three chunks of 263 hold atoms **14.6, 26.6 and 21.4 µm** apart, against floors of 2.5, 6.7, 4.6 |

### The distinction that makes it work

Atoms are **chunk-aligned**, so comparing them *across* chunks measures drift
and will flag every drifting unit as a split. I made that mistake once: 13 of
15 atoms in cluster 262 "flagged", when what they trace is a smooth 25 µm
trajectory. `within_chunk_split` therefore requires atoms from **one time
window**, where drift cannot account for a separation — and a test asserts that
a drifting unit's first and last blocks *do* look like two positions, so the
constraint is about something real.

### What this changed about 262 + 263

Three independent tests supported that merge: refractory dip 0.916, template
1−cos 0.036, co-localisation to 2.5 µm. All were measured on **263 as a whole**,
and 263 is a mixture — 161 atoms for 17,056 spikes against 262's 15 for 51,222.
So they establish that *part* of 263 is 262's cell, not all of it. Split 263
first, then re-test each part. Its elevated contamination (viol/exp 0.11 against
262's 0.03) stops being mysterious and becomes a prediction.

### Pipeline integration

Build the table once per session from the group's `.probe` geometry, then fit
per cluster and per atom and write the positions as a sidecar. Linking across
chunks can then match on **position continuity** — a physical criterion — rather
than only on feature-space similarity. `trajectory()` returns each block's fit
alongside its own floor and the step from the previous block, so a trace is
readable against its resolution rather than by eye.

### Running it as a pipeline stage

```bash
fiber-morpho localize \
    --base <session> --group 5 \
    --variant stderiv.C5.D34 --tag anchor_linked \
    --probe <session>.probe.0.probe --channels 32,33,34,35,36,37,38,39 \
    --morphologies "morph/fs-basket-cell-12-09-07-4.CNG.swc,morph/BC-S-01-03-2011_Z02.CNG.swc" \
    --table morph/postable.npz \
    --out <session>.pos.stderiv.C5.D34.5.tsv
```

**The table is cached and reused.** It depends only on the probe geometry and the
morphology set, never on the sort — 4,788 positions from three morphologies took
115 s to build and 0 s on rerun. Re-cluster and re-run without re-simulating.

Output is one row per cluster (`atom = -1`) and one per atom, carrying
morphology, rotation, depth, lateral, RMSE and that population's **own**
split-half floor. The floor is per-row because thresholds must scale with spike
count.

**`--spk-variant` defaults to `standard` and must stay untransformed.** The
first version of this stage inherited the clustering variant from `Sort()` and
fitted positions to the **stderiv** waveform, where the channel difference has
removed exactly the amplitude-distance relationship a position fit reads — the
same reason `read_pca` refuses to fall back to stderiv. Cluster 262 came out at
RMSE 0.179, depth 182.5, lateral 2.5; on the raw waveform it is **0.0134, depth
170, lateral 20**, and the session median goes 0.229 → 0.0688 with the
resolution reading 1.2 µm instead of 0. The stage now opens the raw file
explicitly and prints which one it used.

**`--max-rmse 0.06` gates the split scan.** A large separation between two bad
fits means nothing; without the gate the scan reported 12 flags including
separations of 106–148 µm on a 140 µm grid. With it, 4.

**`--morphologies` takes a directory, a glob, or a comma-separated list — and
each entry of the list is itself resolved as a directory, glob or file**, so
`--morphologies morph/,morph_pyr/` works. The first version treated every comma
entry as a *file path*: two directories passed the existence check, produced two
entries whose basename was the empty string, and failed several steps later with
`2 morphologies have no biophysics rule` and two blank names — a message
blaming the manifest for a path-resolution fault. A missing entry, an unmatched
glob and an empty directory are all now refused up front, naming the entry.

**`--manifest` is repeatable**, one per morphology directory, later ones winning
on a name collision. With a single manifest, cells from the other directory come
back unmapped and `--strict-kind` refuses them.

```bash
fiber-morpho localize ... \
    --morphologies morph/,morph_pyr/ \
    --manifest morph/morphologies.tsv --manifest morph_pyr/morphologies.tsv \
    --strict-kind
```

**`--max-morph` caps the count after selection and never chooses which.**
It previously took the first N alphabetically, which silently selected a CCK
basket cell and produced RMSE 0.22 with no error.

**Filter on RMSE before using a position.** Only clusters the morphology set can
represent will fit: with three basket cells, non-basket units land at 0.1–0.3
and their positions are meaningless. On g5, 4 of 344 clusters reach RMSE ≤ 0.10.


### Biophysics from the manifest

A single `--kind` across a mixed morphology set runs pyramidal reconstructions
with interneuron conductances — a plausible-looking waveform that is simply
wrong, and which nothing downstream can flag. The manifest already records
`cell_type`, so the preset is read from it:

```bash
fiber-morpho localize ... --morphologies morph/ --manifest morph/morphologies.tsv
```

`--manifest` supersedes `--kind`, which becomes the fallback. `--strict-kind`
refuses instead of falling back, which is what a pipeline wants.

On the 84-cell CA1 interneuron set: **zero unmapped** — `pvbasket` 25, `sca` 25,
`cck` 20, `bistratified` 13, `axoaxonic` 1.

**Rules are checked in order, most specific class first — not by key length.**
That distinction is load-bearing: `parvalbumin (pv)-positive` is 25 characters
and `chandelier` is 10, so length-ordering routed a *PV-positive Chandelier*
cell to `pvbasket` when it is axo-axonic. A multiply-marked NPY/SOM
bistratified cell failed the same way. Anatomical class now precedes marker,
because the class is the more specific statement — a PV+ chandelier cell is an
axo-axonic cell that happens to be PV+.

**Two interpretations are baked in and should be known:**

- **CB1R-negative → PV, CB1R-positive → CCK.** The standard CA1 basket
  dichotomy, but an inference from what the Scanziani label says the cell
  *lacks*, not something it states. **32 of the 84 cells** depend on it.
- **`APPROXIMATED`** names the classes with no preset of their own — O-LM,
  trilaminar, perforant-path associated, back-projecting — and what they route
  to. The CLI prints which were used, so a substitution is never mistaken for an
  identity. **25 of 84** run on `sca` this way, which is a gap in the biophysics
  rather than in the mapping.

`infer_kind` returns `None` on no match rather than guessing, and
`kinds_from_manifest` returns the unmapped rows to the caller rather than
quietly defaulting them.


**The manifest gates the morphology set.** `fetch` leaves rejected downloads on
disk — in the reference set, **110 of 194** `.swc` files never entered the
manifest: 83 with no axon, 20 that load as disconnected trees, 7 below the
`--min-axon` threshold. A directory glob picks all of them up, so when a
manifest is given, files absent from it are dropped with a count.
`--allow-unmanifested` keeps them, on `--kind`.

That separates two failures which previously arrived as one message. *Not in the
manifest* means the file never passed the fetch gate; *in the manifest but its
`cell_type` maps to no preset* is what `--strict-kind` is for. Reporting the
first as the second sent a path-resolution problem to the biophysics subsystem:

```
  [warn] 110 morphologies have no biophysics rule; refusing (--strict-kind)
    050103b: <not in manifest>
```

`--max-morph` now caps **after** the gate filter, so a cap of N yields N usable
cells rather than N entries of which most are rejects.


### The O-LM preset

`CA1_TYPES` had no O-LM entry, so 25 of the 84 fetched interneurons ran on
`sca`. Bezaire et al. (2016) does define one (`class_olmcell.hoc`), and it is
not a variant of anything already present:

| | O-LM | pvbasket |
|---|---|---|
| A-type | **KvAolm** (τ_a fixed 5 ms) | KvABez (τ voltage-dependent) |
| Ih | **HCNolm**, first order, e = −32.9 | HCN, squared, e = −30 |
| Ih τ at −70 mV | **2,972 ms** | 249 ms |
| Rm | **100,000** | 5,555 |
| Na gradient | dendrite **2×** soma, axon 1.6× | uniform |

The slow Ih is the property these cells are known for, and the `sca`
substitution captured none of it. `KvAolm` and `HCNolm` are transcribed from
the Bezaire mod files; densities are his, with the multipliers evaluated.

`gka_dist` and `gna_dend_mult` are new `Biophys` fields, since O-LM is the first
type here with an inverted somatodendritic gradient.

### What remains approximated

Three classes, down from five. None has published CA1 biophysics in the Bezaire
model, so each is routed on marker grounds and reported as an approximation:

| class | routed to | on what grounds |
|---|---|---|
| trilaminar | bistratified | SOM+ dendrite-targeting |
| back-projecting | **olm** | SOM+, mGluR1α+ |
| perforant-path associated | **cck** | CCK+, SLM-targeting |

`back-projecting` and `perforant-path` previously went to `sca`; both are better
served now. `Ca` and `KCa` remain declared missing for every interneuron type
including O-LM.


### Limits

It is **blind to genuinely co-located cells** — two somata at one point give the
same profile whatever their spikes look like — and that is precisely the
population left after the spatial criteria are exhausted. It needs enough spikes
(the floor scales as 1/√n, which is why the floor is returned rather than a
fixed threshold applied). And it inherits the model's systematic error: a good
fit is RMSE ~0.011 against a split-half floor of ~0.002, so the model is ~6×
from describing the data, and a position is only as good as the morphology being
roughly right. Rotation was searched coarsely and is likely far less constrained
than depth.


## Confronting the model with the sort

```bash
fiber-morpho validate --base <session> --group 5 \
    --variant stderiv.C5.D34 --tag anchor_linked --main 2103
```

Run on g5's curated `anchor_linked` sort, cluster 2103 (47,254 spikes) and the
13 neighbouring fragments the curator kept separate "because they show
interesting deviations from the main waveform".

### The recovery prediction is falsified

Pooled over 2103 + 13 fragments, amplitude normalised to 9,274 spikes with
preceding ISI > 200 ms:

| preceding ISI | n | amp ratio | model | 1−cos vs rested |
|---|---|---|---|---|
| 2–4 ms | 2430 | **1.030** | 0.589 | 0.0058 |
| 4–6 ms | 3314 | 1.035 | 0.758 | 0.0038 |
| 8–12 ms | 5287 | 1.033 | 0.894 | 0.0030 |
| 100–200 ms | 10058 | 1.019 | 0.953 | 0.0013 |
| > 200 ms | 9274 | 1.000 | — | 0.0000 |

Spikes after a short interval are 3% **larger**. `morpho_envelope.Recovery`
predicts 0.589 at 4 ms — wrong sign, an order of magnitude out. Do not gate
merges on predicted amplitude ratio; measure the curve on the unit.

Why it over-predicted is **not established**. Candidates, none tested: the unit
may not be a bursting pyramidal cell; the stderiv common-average reference may
remove the component carrying most of the amplitude change; or the
extracellular amplitude, dominated by a somato-axonal sink, may be more robust
to Na inactivation than the somatic membrane potential is.

### Firing state is not what a merge gate competes with

The ISI-dependent *shape* change is real but small — 1−cos rises monotonically
to **0.0058**. Against that:

- the fragments sit **0.011–0.071** from 2103, 2–12× larger;
- 2103's own template moves up to **0.0257** across six time blocks — the
  within-unit variation this sort already accepts as one cell.

So drift and estimation noise, not adaptation, set the budget here.

### The fragments

| clu | n | 1−cos | split-half noise | d/noise | d/budget | lat < 10 ms | t5–t95 (min) |
|---|---|---|---|---|---|---|---|
| 2102 | 796 | 0.0113 | 0.0141 | 0.8 | 0.4 | 15.3% | 165–280 |
| 2101 | 892 | 0.0227 | 0.0105 | 2.2 | 0.9 | 28.7% | 164–178 |
| 2108 | 744 | 0.0226 | 0.0125 | 1.8 | 0.9 | 23.4% | 164–249 |
| 2105 | 1324 | 0.0284 | 0.0108 | 2.6 | 1.1 | 24.3% | 164–267 |
| 2098 | 865 | 0.0301 | 0.0085 | 3.5 | 1.2 | 13.4% | 172–197 |
| 2097 | 2496 | 0.0530 | 0.0422 | 1.3 | 2.1 | 15.7% | 168–280 |
| **2100** | 142 | **1.0853** | 0.0371 | **29.2** | **42.2** | 26.8% | 164–178 |

(chance for lat < 10 ms is 4.8% at 2103's own rate over its own lifetime)

Six of thirteen are inside their own split-half noise; five more are inside
2103's time budget. Every fragment is enriched 3–6× over chance for firing
within 10 ms of a 2103 spike, and merging any of them moves the ISI < 2 ms
fraction by at most 0.091% → 0.103%.

**Only 2100 is distinct** — 29× its own noise, 42× the unit's time budget — and
its difference is in time course, not spatial profile.

Two classes by time: **localised** (2094, 2098, 2100, 2101, 2107 — 14–30 min
windows from ~164 min) versus **session-spanning** (2097, 2102, 2104, 2105,
2108 — matching 2103's own extent). The first group looks like one instability
episode, which is a drift problem rather than a merge decision.

### The command prints no verdict, deliberately

The refractory column has almost no power at a few hundred spikes; latency
enrichment is shared by a burst continuation and by a synaptically driven
partner; and a fragment inside its own split-half noise has not been shown to
deviate at all. These are inputs to a decision, not one.


## What this model does not do

- **Nothing here has been validated against the reference session's sorted
  units.** Every number above is internal to the model. The comparison to make
  is simulated footprints against the 29 curated g5 ground-truth cells, on the
  real `.probe` geometry, and it has not been made.
- Homogeneous isotropic extracellular medium, no frequency dependence. Real
  hippocampal tissue is anisotropic and the assumption inflates distant-channel
  amplitude.
- No calcium, no Ih, no calcium-dependent K. Ih matters for the resting profile
  of distal dendrites; its absence makes the distal tree slightly too excitable.
- No calcium, so complex-spike bursting is driven by a stand-in plateau current
  rather than by its real mechanism (see the envelope section).
- Spikes are evoked by a somatic pulse, so the axonal initiation site and the
  resulting early waveform phase are approximate.
- The archetypes are **factor-sweep instruments, not cells**. A smooth tapering
  cylinder tree has none of the branch-point clutter that shapes a real
  dendrite's extracellular signature; never report an archetype as a morphology
  result.
