# fiber-kit

![python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![version](https://img.shields.io/badge/version-0.2.0-orange)

Drift-stable **fiber** reorganization of over-split spike sorts, for the
[neurosuite-3](https://github.com/Gravios/neurosuite-3) electrophysiology toolchain.

A **fiber** is an energy-direction manifold in whitened feature space: a single
neuron traces one smooth curve `d(r)` — spike *direction* as a function of
*energy/radius* `r`. Drift, bursting, and amplitude modulation move a unit along
its own curve rather than scattering it, so reorganizing a sort around fibers is
robust to the slow waveform changes that fragment conventional clusters.

fiber-kit clusters chunk spikes into fibers, links them across a recording,
groups fibers into neurons using the validated direction-profile signal, and
emits per-fiber geometry plus quality / firing / drift statistics for curation.

---

## Contents
- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Inputs & outputs](#inputs--outputs)
- [Per-fiber statistics](#per-fiber-statistics)
- [Curation: merge candidates & validation](#curation-merge-candidates--validation)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Design notes (what was tried and rejected)](#design-notes-what-was-tried-and-rejected)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [License](#license)

---

## Install

```bash
pip install fiber-kit            # core: numpy, scipy, scikit-learn
pip install "fiber-kit[dip]"     # + diptest, enables DipSplit (bimodal splitting)
```

From a built artifact:

```bash
pip install fiber_kit-0.2.0.tar.gz             # sdist
pip install fiber_kit-0.2.0-py3-none-any.whl   # wheel
```

## Quick start

**Full session (CLI).** Cluster every chunk, link across chunks, write a `.clu`
and a per-fiber `.fibers` table:

```bash
# channels, sampling rate, nChannels and nSamples are read from <session>.yaml
# (looked up as <session>.yaml or <session>/<session>.yaml); group is 1-based.
fiber-session sirotaA-jg-000005-20120312 5 \
    --chunk-min 12 --overlap-min 4 --min-group 200 \
    --fine-method rkk --inclusion-k 3 \
    --merge-method sliding --merge-corr 0.90 \
    --collision-flag --quality-metrics
```

**Python.**

```python
import fiber_kit as fk
from fiber_kit import fiber_lib as fl

# W, nmean: whitener for this chunk (fk.fil_chunk_whitener or fl.chunk_whitener)
fine, geoms = fk.cluster_chunk_fine(
    waves, res, W, nmean, coarse_mg=200, mask=fl.MASK_FULL, sr=32552,
    method="rkk", merge_method="sliding", merge_corr=0.90,
    collision_flag=True, quality_metrics=True)
# fine: per-spike fiber label (-1 = noise); geoms: per-fiber dict of geometry + stats
```

## How it works

The pipeline runs per chunk, then links chunks:

1. **Whitener** — a per-chunk noise whitener from occupancy-masked off-spike
   `.fil` baseline, in stderiv space (`fil → ALLPAIRS common-mode reject →
   temporal first-difference`).
2. **Coarse fibers** — mean-shift clustering for stable cross-chunk anchors.
3. **Fine split** — within each coarse fiber: `rkk` (standalone KlustaKwik CEM),
   `gmm` (BIC mixture), `fiber` (mean-shift), or `none`; then an optional
   **DipSplit** pass (Hartigan dip test on lowered dims) catches bimodal
   clusters the BIC penalty left merged.
4. **Inclusion radius** — per fiber, `median + k·MAD` of the whitened residual to
   its own trajectory; outliers go to noise.
5. **Adaptation cleaning** *(optional)* — rejects high-energy-at-short-ISI spikes,
   gated to fibers with a *real* fast-adaptation law (`corr/snr/tau`).
6. **Consolidation** — (a) template / sliding-direction correlation merge, then
   (b) **direction-profile** same-neuron grouping (apply, or emit candidates).
7. **Isolation** — nearest-neighbour direction-profile distance per fiber;
   optional L-ratio / isolation distance.
8. **Collision flag** *(optional)* — routes recoverable collisions out of the
   noise bin into one dedicated collision cluster.
9. **Linking** — overlap-anchor mutual-majority matching joins per-chunk fibers
   into global ids; rows sharing a `gid` are one fiber over time.

## Inputs & outputs

**Inputs** (neurosuite binary formats):

| File | Format |
|------|--------|
| `<base>.res.<elec>` | spike times, little-endian `int64`, no header |
| `<base>.spkD.<elec>` (or `.spk.<elec>`) | waveforms, `int16`, reshaped `(n, nsamp, nchan)` |
| `<base>.fil` | filtered wideband, `int16`, `ntotal` channels interleaved |
| `<session>.yaml` | session parameters (nChannels, samplingRate, spikeDetection groups) |

**Outputs:**

| File | Contents |
|------|----------|
| `<base>.clu.<elec>` | `int32` cluster ids (header = nClusters; `0` = noise, fiber ids `+1`) |
| `<base>.fibers.<method>.<elec>` | `npz` of per-(chunk,fiber) geometry + statistics (below) |
| `<base>.merge_candidates.<elec>.tsv` | proposed same-neuron merges (with `--emit-merge-candidates`) |

## Per-fiber statistics

Every stat in `.fibers` is per chunk, so rows sharing a `gid` form **time series**
across the session (drift, rate stability, isolation over time).

| Group | Columns |
|-------|---------|
| Identity / time | `gid`, `chunk`, `tmin`, `coarse`, `nspk` |
| Geometry | `radius`, `depth`, `width_ms`, `radius_incl`, `n_rejected`, `template`, `grid`, `dir` |
| Firing / cell-type | `rate`, `presence`, `burst`, `isi_cv` |
| Contamination | `refrac` (% ISI<2 ms), `hill_fp` (refractory false-positive fraction; **NaN when too sparse to estimate**) |
| Isolation | `resid_med`, `resid_mad`, `nn_dist` + `nn_gid` (closest fiber in direction-profile space), `lratio`, `iso_dist` (`--quality-metrics` only) |
| Within-chunk drift | `radius_slope` (/min), `depth_slope` (/min), `dir_drift` (first- vs second-half direction change) |
| Adaptation | `adapt_corr`, `adapt_tau`, `adapt_snr`, `adapt_meanabsz`, `adapt_fracz3` |

```python
import numpy as np
f = np.load("sirotaA-jg-000005-20120312.fibers.stderiv.5", allow_pickle=True)
for g in np.unique(f['gid'][f['gid'] >= 0]):
    m = f['gid'] == g; o = np.argsort(f['tmin'][m])
    depth_t = f['depth'][m][o]    # drift trajectory across the session
    iso_t   = f['nn_dist'][m][o]  # isolation over time
```

Curation flags: low `nn_dist` → merge candidate; high `hill_fp`/`refrac` →
contamination; high `|radius_slope|`/`dir_drift` → chunk too long; low
`presence` → unit drifts in/out.

## Curation: merge candidates & validation

Grouping fibers into neurons uses the **direction profile** `d(r)` — the
validated signal (within-fiber halves vs distinct fibers separate at AUC ~0.98).
The threshold auto-calibrates to the within-fiber-half distance floor.

```bash
# 1. consolidate normally, then propose same-neuron merges WITHOUT applying them
fiber-session <base> 5 ... --merge-method sliding --merge-corr 0.90 \
              --emit-merge-candidates           # -> <base>.merge_candidates.5.tsv

# 2. gather independent full-session evidence per proposed pair
fiber-validate-merges <base> 5 --sr 32552
#    cross-correlogram by band + 30 s rate correlation:
#    - low [2,5] ms vs baseline  -> relative refractory survives -> same neuron
#    - rate_r < 0                -> drift handoff               -> same neuron
#    ([0,2] ms is confounded: cross-neuron coincidences are removed as collisions)

# 3. apply once trusted (auto threshold, or --profile-thr / hand-merge the gids)
fiber-session <base> 5 ... --merge-method profile
```

## CLI reference

`fiber-session <base> <elec>` (key flags; see `--help` for all):

| Flag | Default | Purpose |
|------|---------|---------|
| `<session>` `<group>` | — | positional: session (finds `<session>.yaml`) and 1-based group |
| `--channels` / `--ntotal` / `--nchan` / `--nsamp` / `--sr` | from YAML | override the YAML-derived probe geometry & sampling |
| `--chunk-min` / `--overlap-min` | 12 / 4 | chunk length & overlap (minutes) |
| `--min-group` | 200 | coarse min spikes/fiber (linking anchors) |
| `--fine-method` | `gmm` | `rkk` \| `gmm` \| `fiber` \| `none` |
| `--inclusion-k` | 3.0 | per-fiber radius = `median + k·MAD`; `0` disables |
| `--merge-method` | `template` | `template` \| `sliding` \| `profile` |
| `--merge-corr` | 0.0 | consolidation threshold (`0`=off; ~0.95 template / ~0.90 sliding) |
| `--profile-thr` / `--profile-floor-pct` / `--profile-min-n` | auto / 90 / 120 | profile-merge threshold, auto-floor percentile, min spikes to be a merge anchor |
| `--emit-merge-candidates` | off | write proposals, don't merge (curation) |
| `--collision-flag` | off | route recoverable collisions to a dedicated cluster |
| `--quality-metrics` / `--quality-dims` | off / 10 | L-ratio + isolation distance (O(N·K)) |
| `--adapt-clean` | off | reject high-energy-at-short-ISI on real fast adapters |
| `--no-link` / `--no-fine` / `--no-dipsplit` | off | disable linking / refinement / DipSplit |

Other tools: `fiber-validate-merges <base> <elec>`, `fiber-raw-vs-stderiv <base>
<elec> --channels ... --ntotal ...`.

## Python API

```python
import fiber_kit as fk
fk.cluster_chunk(...)         # coarse mean-shift fibers
fk.cluster_chunk_fine(...)    # full per-chunk pipeline -> (labels, geoms)
fk.fiber_geom(...)            # geometry + stats for one fiber
fk.link_chunks(...)           # cross-chunk overlap-anchor linking
fk.trajectory(X); fk.predict(traj, r)     # fiber tracer
fk.read_res(...); fk.open_spkD(...); fk.fil_chunk_whitener(...)
fk.fiber_lib, fk.fiber_tracer, fk.fiber_adapt, fk.fiber_collision, fk.laplacian_link
```

## Design notes (what was tried and rejected)

These choices are baked in; the rejected alternatives are documented so they
aren't re-attempted:

- **Direction only** for same-neuron grouping. Trajectory *curvature* and
  *tangent* are noise-dominated (AUC ~0.65 vs direction's ~0.98) and *degrade*
  the result when combined — deliberately unused.
- **No derivative features.** A temporal/spatial/mixed derivative of the stderiv
  waveform is a linear re-encoding; after re-whitening it is an orthogonal
  rotation, so a whitened-L2 classifier is invariant to it (rank-deficient ones
  only lose information). Confirmed empirically.
- **Collision detection, not decomposition.** Flagging recoverable collisions by
  matching-pursuit gain is reliable; decomposing a collision to a specific fiber
  pair is separability-bound (~15–19%) and kept advisory only. Amplitude-pinned
  decomposition is least-squares-dominated and not recommended on raw output.
- **Refractory validation is full-session.** Within one shank the cross-CG
  `[0,2] ms` bin is confounded by collision removal; `fiber-validate-merges`
  keys on the `[2,5] ms` band and the across-session rate handoff instead.
- **`hill_fp` returns NaN** for units too sparse to estimate, rather than a
  misleading 0.5 clamp.

## Project layout

```
src/fiber_kit/
  fiber_lib.py        primitives: whitener, realign, features, constants
  fiber_tracer.py     trajectory(), predict(), assign, temperature calibration
  klustakwik.py       standalone KlustaKwik CEM (the "rkk" split)
  fiber_adapt.py      EWMA adaptation fit / residual / de-adapt
  fiber_collision.py  collision templates, matching-pursuit decompose/detect
  laplacian_link.py   curve-continuity fragment linking; energy-banding report
  fiber_session.py    pipeline + CLI (cluster_chunk_fine, link, I/O, main)
  raw_vs_stderiv.py   raw .fil vs stderiv discrimination test
  validate_merge_candidates.py  full-session evidence for merge candidates
  WORKFLOW.md         end-to-end recipe (also shipped as package data)
```

## Requirements

Python ≥ 3.9 · numpy ≥ 1.21 · scipy ≥ 1.7 · scikit-learn ≥ 1.0 ·
pyyaml ≥ 5.3 · optional: diptest ≥ 0.5 (for DipSplit).

## License

MIT — see [LICENSE](LICENSE).
