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

## Tightening fiber membership by per-channel residual variance

When a fiber sorts well but still hides waveform sub-groups (subtle shape
differences that an rkk split in Klusters resolves into cleaner templates, fewer
ISI violations, and channels with much less variance), the measure to target is
the per-channel variance of the RESIDUAL TO THE ENERGY-LOCAL TEMPLATE r·d(r),
read in raw (un-whitened) channel space — `fiber_tracer.channel_residual_profile`.

Crucially this is NOT raw per-channel waveform variance: that is dominated by the
fiber's legitimate energy/adaptation spread (the fiber is a curve), so minimizing
it would just carve the fiber into energy bands. The residual to d(r) removes the
energy axis, leaving genuine shape contamination — its per-channel profile peaks
exactly on the discriminating channels, and a real shape sub-split lowers its
mean sharply while an energy split does not.

Exposed three ways:
- `.fibers` now carries `chan_resid_var_mean` / `chan_resid_var_max` per
  (chunk,fiber): rank fibers by these to find the ones hiding sub-units.
- `fiber-session --cone-channel-k K` tightens the inclusion cone PER CHANNEL:
  after the global `--inclusion-k` radius, spikes that are residual outliers
  (>K MAD) on the discriminative channels are dropped — peeling channel-localized
  contaminants the global residual norm averages away. (Start K≈2.5–3.)
- `fiber-session --split-var-margin M` makes the minimal-mean-per-channel-
  residual-variance the split ACCEPTANCE criterion: a within-fiber split is kept
  only if it lowers that mean by ≥ M (e.g. 0.1), so real shape sub-units are
  split and energy-only splits are rejected.

Both controls default off (validated behavior unchanged); opt in to tighten.

### Automatic variance-driven splitting (`--var-split`)

Rather than ranking fibers by `chan_resid_var_max` and splitting them by hand,
`fiber-session --var-split R` auto-splits any fiber whose per-channel residual
profile is peaked (`max/median channel residual variance >= R`, try R≈2). It
recursively bisects on the trajectory residual WEIGHTED toward the high-variance
channels, and — crucially — the per-channel residual variance is the STOP
criterion: a bisection is taken only while it lowers the mean by the margin
(`--split-var-margin`, default 0.05 when `--var-split` is on). So it finds the
right number of shape sub-units, stops at unimodal, and never splits on energy.

Synthetic validation: 2 / 3 true shape sub-units recovered at RandIndex 1.00;
a single unit with large energy spread and a pair differing in amplitude only
are both correctly left unsplit. `--var-split-depth` caps recursion (≤ 2^depth
sub-units/fiber). Recommended: `--var-split 2 --split-var-margin 0.1`, then
`--cone-channel-k 2.5` to clean residual edges.

### Circular-xcorr alignment makes the residual variance meaningful

The per-channel residual variance is only meaningful AFTER per-spike timing
jitter is removed — otherwise jitter inflates it and mislocates the
discriminating channels. `fiber_lib.align_xcorr` aligns each spike to the
cluster MEDIAN by full (channel-summed) cross-correlation, refined to sub-sample
lags (parabolic peak + Fourier phase shift), ITERATING until the residual
variance stops dropping. Circular shifts are exact and harmless because the
waveforms are high-pass filtered (window edges ~0). `channel_residual_profile`
(and hence `--var-split` / `--split-var-margin`) uses it by default
(align="xcorr"). Validated: injected sub-sample jitter recovered at corr 0.9999,
residual variance driven to the noise floor, and on a jittered 2-sub-unit fiber
the discriminating channels are correctly recovered (dom-channel realign
mislocated them).

### Shape-distinctness gate + re-seed (fiber-refine, v0.15.0)

Validation on a partially-curated `.clu` (g5) showed `fiber-refine` would shatter
high-rate units into energy-level pieces sharing one waveform shape (curated 343
-> 35 sub-units, median-template corr 0.94-0.99), which mechanically lowered the
within-cluster band-ISI count (a small-N artifact, not real isolation gain). Two
fixes:

- `--split-min-corr C` (default 0.93): a split piece or knn-peel energy bucket
  whose NORMALISED median waveform correlates >= C with its parent is not carved
  off (kept with the parent). Genuinely distinct sub-units (corr ~0.70 on g5)
  still split; same-shape energy clones (corr ~0.95) do not. Use 0.95 for more
  aggressive consolidation of high-rate units.
- `merge_back` is now ON by default (`--merge-back`, min_sim 0.92, normalized),
  so a single `refine()` call returns a consolidated sort rather than an
  over-split one.

With both, g5: 202 -> ~35 clusters, curated 343/258 consolidated, the distinct
clu-33 sub-units preserved, and the band-ISI total honest (29 vs 30 curated) --
i.e. beyond dedup, this chunk has little real band contamination to remove.

`--reseed N` re-runs the whole loop using the refined labels as the next seed
(stops early on convergence). On g5 it did NOT help -- split and merge_back
partially undo each other, drifting to slightly more clusters without settling --
so it defaults off (single pass). It is wired for data where the cleaned fibers
genuinely seed a better pass.

### Fit/reassign re-seed loop + fiber-geometry tracking (v0.16.0)

`--reseed N` now runs the loop the way it should: each pass is
`split -> merge_back -> refit fibers -> reassign` and the reassigned labels seed
the next pass (`_refit_reassign` uses `fiber_tracer.run_from_seeds`: per-cluster
trajectory fit, then every labelled spike re-assigned by whiteness residual;
noise preserved). Unlike the old label-refeed, this *converges* -- on g5 two
passes settle at 23 clusters instead of drifting up.

`fiber_tracer.fiber_shape_stats(waves,W,nmean,mask)` returns the shape of one
cluster's spike cloud AROUND its fiber: radius mean/CV/skew/bimodality, cone
angle (median + p95 of per-spike angle to d(r)), whiteness residual med/MAD, and
trajectory bend (total turning of d(r), deg) + per-step smoothness.

`--track-geometry` records every cluster's shape stats at every iteration and,
because the spikes are fixed and only labels move, links each FINAL fiber back
through the snapshots by spike overlap, writing `<base>.geom.<group>.tsv`
(`fiber, iter, host, purity, <stats>`). API: `geometry_tracks(snaps, waves, W,
nmean, mask)` + `write_geometry_tracks`; `refine(..., snaps_out=[])` collects the
per-step labellings.

What it exposed on g5: **trajectory bend + smoothness are an intrinsic per-fiber
fingerprint, near-invariant across all 14 iterations, while radius-bimodality and
cone respond to cleaning.** Clean single units (curated 343, 258) have flat
tracks -- bend ~33-41 deg, smooth ~0.8-1.1, r_cv 0.13, r_bimod steady ~0.47. The
multi-cell footprint (curated 33, which elsewhere splits into 8 units at template
corr ~0.70) is immediately distinct: bend ~95-100 deg (~3x), smooth ~2.5, r_cv
0.21-0.30, and r_bimod rising past the 0.555 bimodality threshold (0.49 -> 0.66)
as reassignment resolves its energy levels. A high *stable* bend is the tell that
a footprint is more than one cell; absolute cone degrees are inflated in whitened
space, so read fibers relative to each other, not the raw angle.

**Reading the residual along the track.** The geom TSV carries `resid_med` (median
per-spike whiteness residual to the energy-local template `r*d(r)`) and
`resid_mad` (its robust spread = along-fiber residual variance). Comparing each
final fiber to its `fine`-iteration ancestor on g5:

  - Clean single units do NOT drop -- nothing to remove. 343->fiber21 resid_mad
    2.35->2.36, 258->fiber22 2.21->2.20; level flat at ~12. A small dip appears
    at the `*.reasgn` rows and merge_back puts it back.
  - Contaminated / multi-cell fibers DO drop. Curated 33->fiber15 resid_mad
    3.15->2.77 (-12%), level 14.7->13.6 (-7%); the small contaminated clusters
    (19,16,2,3) fall 22-41% in mad. Median over all 23 fibers: resid_mad -3.9%,
    resid_med -1.4% -- a modest tightening concentrated entirely where there was
    excess variance to lose.

Two caveats when interpreting it: (1) the reduction is driven by the
refit/reassign step, not the split -- it shows up as a step-change in the
`*.reasgn` rows (fiber15 mad 3.33 through `1.merge`, then 2.76 at `1.reasgn`),
which is the evidence that fitting a fresh per-fiber trajectory and reassigning by
whiteness residual tightens the cloud. (2) It is reduction-by-purification, not a
better fit of the same spikes: the dropping fibers also shed spikes (fiber15
1858->997) as contaminants reassign away, so part of the median/MAD drop is just
losing high-residual outliers. A few small clusters get WORSE (fiber9, 109
spikes, mad +105%) where merge_back over-absorbs or reassignment pulls in poor
matches -- the honest cost on low-count fibers. Watch `resid_mad` rising together
with `purity` falling as the signal that a track is being contaminated rather
than cleaned.

### Drift-aware chunked mode (v0.17.0)

Whole-session `refine` pools every spike of a cluster into ONE trajectory and ONE
whitener, which smears a drifting unit. `--chunk-minutes M` (with
`--chunk-overlap-minutes`, default 1.0) instead windows the session the way
`fiber_session` does -- disjoint CORE windows `[lo,hi)` that tile the session,
plus EXTended windows `[lo-ov, hi+ov)` that overlap their neighbours -- fits a
SEPARATE whitener and runs the full refine loop inside each (quasi-stationary)
window, then links per-window fibers by overlap-anchor (`fs.link_chunks`: the same
physical spikes in adjacent windows' overlap prove identity, mutual-majority,
drift-free). Each spike's final label comes from its CORE window. The iteration
knobs (`--iters`, `--reseed`, `--no-converge`) apply per window, so a high-count
full-session run is meaningful rather than drift-smeared.

`--track-geometry` in chunked mode writes `<base>.geomchunk.<group>.tsv`
(`fiber, chunk, t_min, <stats>`) -- each global fiber's shape stats measured in
EACH window's own frame, i.e. the drift signature over time (radius/cone/bend per
window). API: `refine_chunked(...) -> (global_labels, n_global, tracks)`,
`write_chunk_geometry`.

Validated (g5 10-min chunk split into 3x4-min windows, overlap 1 min, per-window
local ids deliberately scrambled): core windows tile all 21710 spikes; stitching
gives 100% core-spike agreement with the curated labels; 343/258/33 each link to
a single global id across all windows. Two honest caveats:
  - Overlap-anchor only links units with >= min_anchor (8) spikes in the overlap,
    so low-rate units fragment across windows (here 423 globals vs 202 curated,
    85/202 single-id). Lengthen the overlap or lower min_anchor for sparse units.
  - traj_bend is spike-count sensitive (a short tail window inflates it); compare
    the n-robust stats (r_cv, cone_med, resid_mad) across windows for drift, and
    read bend only against windows of comparable n.
