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
