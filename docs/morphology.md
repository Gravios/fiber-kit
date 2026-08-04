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
own `probe.0.probe` geometry for group 5 (channels 40–47: staggered ±11 µm,
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
