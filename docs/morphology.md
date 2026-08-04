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
- The Na model omits the `s` slow-inactivation state, which is identically 1 at
  the published `ar=1` — the only simplification of the transcribed kinetics.
- Spikes are evoked by a somatic pulse, so the axonal initiation site and the
  resulting early waveform phase are approximate.
- The archetypes are **factor-sweep instruments, not cells**. A smooth tapering
  cylinder tree has none of the branch-point clutter that shapes a real
  dendrite's extracellular signature; never report an archetype as a morphology
  result.
