# fiber_kit — profile-merge testing workflow

Same-neuron fiber grouping by **energy-resolved direction profile** `d(r)`, the
validated signal (AUC ~0.98 same-fiber-halves vs distinct fibers; curvature/
tangent and derivative features were tested and add nothing — see notes).

## Files
- `session_yaml.py` — **new** reads `<session>.yaml`; all CLIs take `<session> <group>`.
- `fiber_lib.py` `fiber_tracer.py` `klustakwik.py` — primitives (unchanged).
- `fiber_session.py` — pipeline; `merge_method="profile"` + `--emit-merge-candidates`.
- `fiber_adapt.py` `fiber_collision.py` `laplacian_link.py` — adaptation / collision / link.
- `validate_merge_candidates.py` — full-session evidence for proposed merges.
- `raw_vs_stderiv.py` — raw-`.fil` vs stderiv discrimination test (separate question).

Inputs expected on disk: `<session>.res.<group>`, `<session>.spkD.<group>`,
`<session>.fil`, and `<session>.yaml` (or `<session>/<session>.yaml`). The CLIs
read **channels, sampling rate, nChannels, and nSamples straight from the YAML** —
you only pass `<session> <group>`. Any `--channels/--ntotal/--sr/...` flag still
works as an override, and if no YAML is found the tools fall back to requiring them.

```bash
SESSION=sirotaA-jg-000005-20120312 ; G=5     # group is 1-based (spikeDetection.channelGroups)
COMMON="--chunk-min 12 --overlap-min 4 --min-group 200 --fine-method rkk --inclusion-k 3"
```

## 1. Baseline + curation candidates (recommended path)
Consolidate normally (sliding), then emit profile candidates **on the clean fibers
without merging** — review-only. Writes `.clu`, `.fibers.*`, and the candidate tsv.
```bash
fiber-session $SESSION $G $COMMON \
    --merge-method sliding --merge-corr 0.90 \
    --emit-merge-candidates
# -> <session>.merge_candidates.5.tsv   (gid columns VALID in emit mode)
```
The tsv: `chunk  gid_a  gid_b  local_a  local_b  profile_dist  threshold`,
sorted by distance (smaller = higher-confidence same-neuron). Threshold is the
auto same-neuron floor (90th pct of within-fiber-half distances); override with
`--profile-thr 0.13` or change `--profile-floor-pct` / `--profile-min-n`.

In-sandbox reference (10-min chunk): g5 → 7 candidates @ thr 0.13; g7 → 19 @ 0.13.

## 2. Validate candidates against full-session evidence
```bash
fiber-validate-merges $SESSION $G
```
Per pair it prints the cross-correlogram by band and the 30 s rate correlation:
- **`[2,5] ms` low vs baseline** → relative refractory survives → same-neuron evidence
  (the `[0,2] ms` bin is confounded: cross-neuron coincidences were removed as
  collisions, so it reads ~0 for distinct pairs too — judge on `[2,5]`).
- **`rate_r < 0`** → drift handoff (amplitude wanders between the two fibers) → same neuron.
- **`rate_r > 0` with filled `[2,5]`** → co-active distinct neurons → do NOT merge.

Flags are heuristics; eyeball the CCG for borderline pairs.

## 3. Apply the merge (after you trust the threshold)
Auto-apply profile-merge as the consolidation step (no sliding):
```bash
fiber-session $SESSION $G $COMMON --merge-method profile        # auto threshold
fiber-session $SESSION $G $COMMON --merge-method profile --profile-thr 0.12
```
Or keep sliding for fragments and hand-merge only the validated gids from step 2.
In-sandbox reference: g5 27→14, g7 58→18 (≥120-spike fibers, floor threshold).

## 4. (Separate) raw-`.fil` vs stderiv discrimination
Tests whether the un-reduced raw waveform separates the confusable fibers better
than stderiv (the one feature route the whitening-invariance theorem doesn't forbid):
```bash
fiber-raw-vs-stderiv $SESSION $G \
    --chunk-min-start 183 --chunk-min 10 --min-spikes 60
```
Read the final line: `mean raw-minus-stderiv on confusable pairs`. Positive ⇒ stderiv
is discarding separating power; flat ⇒ the ceiling is intrinsic (use temporal context).

## Per-fiber statistics in `.fibers.<method>.<elec>` (npz)
Each row is one (chunk, fiber); rows sharing `gid` are the same global fiber over
time, so these become **time series** across the session.

Firing / cell-type: `rate` (Hz), `presence` (frac of 20 chunk-bins with a spike),
`burst` (frac ISI<6 ms), `isi_cv`, `refrac` (% ISI<2 ms), `hill_fp` (refractory
false-positive fraction; **NaN when the unit is too sparse to estimate**).

Isolation / compactness: `resid_med`/`resid_mad` (whitened residual to own
trajectory), `nn_dist` + `nn_gid` (closest other fiber in direction-profile space
and which one — small `nn_dist` on a clean fiber = over-split / merge candidate),
and `lratio`/`iso_dist` (Schmitzer-Torbert; populated only with `--quality-metrics`).

Within-chunk drift: `radius_slope` (whitened-radius/min), `depth_slope` (channels/
min), `dir_drift` (first- vs second-half direction distance — high = morphing or a
split/merge mid-chunk; unreliable for n<~120).

Waveform / position: `depth`, `width_ms`, `radius`, plus `template`, `grid`, `dir`.
Adaptation: `adapt_corr`, `adapt_tau`, `adapt_snr`, `adapt_meanabsz`, `adapt_fracz3`.

Quick reads of the time series (Python):
```python
import numpy as np
f = np.load("sirotaA-jg-000005-20120312.fibers.stderiv.5", allow_pickle=True)
for g in np.unique(f['gid'][f['gid'] >= 0]):
    m = f['gid'] == g; o = np.argsort(f['tmin'][m])
    depth_t  = f['depth'][m][o]      # drift trajectory
    rate_t   = f['rate'][m][o]       # rate stability / presence drops
    iso_t    = f['nn_dist'][m][o]    # isolation over time
# curation flags: low nn_dist (merge), high hill_fp / refrac (contamination),
# high |radius_slope|/dir_drift (chunk too long), low presence (drift in/out).
```

## Notes / decisions baked in
- Direction profile only; **curvature/tangent deliberately unused** (noise-dominated, AUC ~0.65).
- `--profile-min-n 120`: small fibers have unstable trajectories → never a merge anchor.
- Profile step runs **after** template/sliding, so it reviews/merges already-clean fibers.
- Within one shank the refractory test is partially confounded by collision removal;
  the trustworthy validators are the `[2,5] ms` CCG band and full-session rate handoff.

## Post-hoc re-linking by evolving geometry (`fiber-relink`)

`link_chunks` links fibers across chunks only by shared overlap-window spikes. On
long sessions that strands almost every fiber as its own unit (group 5 of
`sirotaA-jg-000005-20120312`: 2400 / 2465 gids were single-chunk) because a unit
firing < ~8 spikes in the 4-min overlap never anchors. Yet the fiber geometry
itself is an almost-perfect cross-chunk signal — for genuine same-unit
consecutive chunks the direction-profile distance is ~0.045, template
correlation ~0.99, |Δdepth| ~0.04 ch (AUC 0.993 vs different units), and it
drifts smoothly. `fiber-relink` exploits that, post-hoc, on a finished
`.fibers` run (no re-run needed):

```
fiber-relink <base>.fibers.<method>.<elec>.npz --clu <base>.clu.<elec> \
             --out <base>.clu.<elec>.relinked --report relink_report.tsv
```

It is **strictly additive** (only merges, never splits): it seeds the union-find
with the existing gids (so every original link is preserved — 100% recall), then

1. **bundles within a chunk** the over-split fragments of one neuron, by
   *mutual-best* pairing on tight template + direction-profile agreement
   (iterated, never single-linkage, so it cannot chain distinct neurons), and
2. **chains across chunks by evolving geometry** with a *noise-aware* distance
   `wp·profile + (1-wp)·tmpl_dist`, `wp = q/(q+300)`, `q = min(nspk)` — so the
   sharp direction profile dominates only when both fibers are well sampled and
   the robust template carries the sparse ones. Matching is **one-to-one forward
   chaining** (≤1 successor / ≤1 predecessor) under a **chunk-disjoint** union
   (two tracks sharing a chunk are different neurons), with reciprocal-best +
   uniqueness margin, a per-step template/depth gate, and a one-chunk gap bridge.

Matchability scales with spike count (the geometry-estimate quality): nspk ≥ 1k
chains at ~0.03, but the ~87 % of fibers with nspk < 300 are too noisy to link
on geometry alone and are left as singletons — those need the in-pipeline
overlap anchors and CCG validation, not geometry. On the group-5 run this took
2465 → ~1868 units while preserving the longest tracks (the persistent
interneuron spans 28 chunks) and keeping every geometry-created step below the
template gate.

The TSV report has one row per re-linked unit (`n_chunks`, `spikes`, end-to-end
template/depth drift, worst consecutive step) with a `suspect` flag for any unit
whose worst consecutive step or end-to-end drift is large — which also surfaces
**inherited** bad links from the original `.clu` for review.

## Spike-time correction to the fiber template (`fiber-realign`)

After re-linking, correct each spike's time so it is aligned to the template of
the unit it now belongs to, and save the per-spike offsets:

```
fiber-realign <base> <elec> --nsamp 32 --nch 8 --clu <base>.clu.<elec>.relinked
# -> <base>.res.<elec>.realigned   (int64 LE, = res + round(offset))
# -> <base>.offsets.<elec>.npy     (float32 sub-sample offset per spike)
```

Each spike's waveform (`.spkD`/`.spk`) is aligned to its unit's multichannel
template by cross-correlation (integer lag within `--max-shift`), refined to
sub-sample resolution by a parabola through the correlation peak; the template is
recomputed from the aligned spikes and the alignment repeated (`--iters`). Units
with `< --min-n` spikes are left at zero offset. `res_corrected = res +
round(offset)`; the full sub-sample offset is saved separately (the integer `.res`
grid cannot hold it).

Two uses: it removes residual detection jitter, and — run with the **re-linked**
`.clu` — it forces every spike of a merged unit onto one canonical template, so a
unit assembled from fibers detected against different per-chunk references gets a
single consistent spike-time convention. Where the extractor already peak-aligned
(`process_extractspikes_stderiv`), the integer offsets are correctly small (on the
group-5 min-183..193 excerpt: 8.9 % nonzero, max ±5 samples, +0.06 % template
sharpening) while ~88 % of spikes still carry a non-zero **sub-sample** offset —
so the saved offsets are the substantive output and the corrected `.res` differs
only for the minority needing an integer shift.

Recommended chain: `fiber-session` → `fiber-relink` (merge) → `fiber-realign`
(align times to the merged-unit templates).
## Physical localization from waveform spread (`fiber-localize`)

Recover each unit's position relative to the probe by fitting a point-source field
to its RAW per-channel peak-to-peak amplitudes:

```
fiber-localize <base> <elec> --nsamp 32 --nchan 8 \
               --probe <session>_probe_0.probe [more .probe ...] \
               --channels 32,33,34,35,36,37,38,39 --clu <base>.clu.<elec>.relinked
# -> <base>.localize.<elec>.tsv
```

Model: `a_c = A/d_c + B·((y_c−y0)/d_c)/d_c²`, `d_c = √((x0−x_c)²+(y0−y_c)²+z0²)`.
The **monopole** term reads distance from the spatial SPREAD of the footprint
(FWHM ≈ 3.46·z0 — steep falloff = near, broad = far), independent of source
strength. The **dipole** term absorbs an ASYMMETRIC footprint and recovers
distance a symmetric monopole would miss. Distances come with a bootstrap CI over
spikes; an energy-stratified depth (`depth_shift`) reports axial extent — ~0 for a
compact source, non-zero for a soma–dendrite axis (the d(r) curvature as a length).

**Localize on RAW amplitudes** (`.spk` / `.fil`), never `.spkD`/whitened — the
stderiv transform breaks the amplitude–distance law. Pass the **re-linked** `.clu`
so a unit's full spike set is used.

Each row is flagged `reliable` only if none of: `one_flank` (peak on a terminal
channel — perpendicular distance unidentifiable on a linear probe), `at_bound`
(distance pinned to the fit limit), `low_n`, or `high_resid`. Validated on the
Buzsaki64L octrode (group 5): interior well-sampled units localize tightly
(z ≈ 24 µm, CI ±1 µm), the dipole term rescues asymmetric units a monopole pins
to the bound, and edge units are correctly flagged degenerate. On a single linear
shank depth is solid but perpendicular distance is identifiable only for interior
both-flanks units; a 2-D-site probe removes that limit.

Full chain: `fiber-session` → `fiber-relink` → `fiber-realign` → `fiber-localize`.
## Probe drift tracking (`fiber-drift`)

Track the probe's drift over the session from the fiber files of its groups — no
raw data, just the per-(chunk,fiber) depths:

```
fiber-drift <base>.fibers.stderiv.0 <base>.fibers.stderiv.1 ...   # a probe's groups
# -> fiber_drift.tsv   (chunk, t_min, n_units, drift_um[, per-group])
```

Each unit tracked across chunks is a drift fiducial. The tool solves a
decentralised registration `depth_u(c) = base_u + D(c)` (robust median
alternation) for the shared drift curve `D(c)`, separating it from each unit's own
depth. Several groups on one probe feed a joint `D(c)` (rigid probe) while each
group also gets its own `D_g(c)`, so cross-shank spread exposes tilt/bending.
After removing `D(c)`, the slope of the residual vs a unit's base depth is the
**depth-gradient of drift** — the signature that triggers position-dependent
(non-rigid) correction.

Re-linked units are used by default (more fiducials); `--no-relink` falls back to
raw `.fibers` gid. Depth is in channel units, converted to µm via `--pitch` (20).
Because cross-chunk tracking gates on small per-step depth change, this resolves
SLOW drift well; drift faster than the link gate breaks tracks and is better
estimated by raster registration. Validated on group 5 of
sirotaA-jg-000005-20120312 (118 fiducial units, 30 chunks over 348 min): a gradual
~3 µm drift rising to +2 µm by 336 min, with a −0.19 µm/ch depth-gradient flagging
mild non-rigid motion.

## Normalized position along the fiber (`fiber-position`)

A drift-independent per-spike feature: where each spike sits along its unit's
fiber manifold, `s ∈ [0,1]` (0 = low-energy / most-adapted end, 1 = high-energy
end), for studying input and adaptation dynamics.

```
fiber-position <base> <elec> --fibers <base>.fibers.stderiv.<elec> \
    --nsamp 32 --nchan 8 --ntotal 96 --clu <base>.clu.<elec> [--session <s>.yaml]
# -> <base>.position.<elec>      (binary float32, parallel to .res)
# -> <base>.position.<elec>.npz  (res, unit, s, conf, energy, chunk)
```

The manifold is the one already **estimated in the `.fibers` file**: each
re-linked unit's stored per-chunk `d(r)` curves are consolidated (arc-length-
normalized, nspk-weighted) into ONE manifold `d̂(u)`, direction as a function of
normalized arc length. It is not re-estimated per chunk.

Position is read from the spike's **direction** (footprint shape), not its
energy: `s` is the arc length `u` whose manifold direction best matches the
spike. This is what makes it drift-independent — drift slowly rescales a unit's
amplitude (so energy *walks with the drift*), but the shape-position along the
manifold is a physiological coordinate that adaptation/input drive, invariant to
that rescaling. Energy `‖X‖` is reported alongside as a drift-DEPENDENT
reference. Per-spike direction is noisy over a curve that spans only a modest
rotation, so `s` is most informative aggregated (per ISI bin / time window) —
in line with this pipeline treating geometry as a per-population quantity.

`--no-relink` uses raw `.fibers` gid units; `--n-u` sets the manifold arc-length
resolution; `--min-nspk` the minimum chunk-curve spike count to enter a manifold.
Synthetic validation (energy drift injected across chunks): manifold recovery
cos ≈ 1.0, position recovery corr(s, true)=0.96, and for a fixed true position
the mean `s` is flat across chunks (range 0.03) while energy falls with the
injected drift — i.e. `s` is drift-independent where energy is not.

Full chain: `fiber-session` → `fiber-relink` → `fiber-realign` →
`fiber-localize` / `fiber-drift` / `fiber-position`.
