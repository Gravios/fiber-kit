# fiber-kit workflow — group-5 drift-tracking pipeline, start to finish

Reference session: `sirotaA-jg-000005-20120312`, group 5 (8-ch octrode, CA1, 32552 Hz,
~350 min, 30 × 12-min chunks, known depth drift). Every command takes `<session> <group>`
and reads geometry (`nchan`, `nsamp`, channels, `sr`) from `<session>.yaml`; flags override.

Principle that drives the ordering: **raw `.spk` → position only** (monopole inverse);
**all matching happens in stderiv** (`.spkD`). Mutual-center before any cross-cluster cosine.
Position-primary + cosine + offset co-gate. Nothing below re-clusters — fiber-kit tracks and
curates an existing KlustaKwik sort.

---

## 0. Upstream (ndmanager-plugins, C++) — produces the per-group binaries

Detection → stderiv transform → PCA → KlustaKwik. Outputs per group `g`:
`.res.g` (int64 times), `.clu.g` (int32 header+ids), `.spk.g` (raw int16), `.spkD.g`
(stderiv int16), `.fetD.g`, `.pcaD.g` (PCA basis). fiber-kit consumes these; it does not
produce them. (`fiber-pca` can now *read/write* `.pcaD` and *realign* without the C++
`process_alignspikes_pca` — see §5.)

---

## 1. `fiber-refine` — curate the KlustaKwik clusters

```
fiber-refine <session> 5 --in-clu <base>.clu.5 --out-method stderiv --out-stage refine
```

Sub-floor dedup + residual-split + merge-back, iterated to convergence. Important args:
`--no-dedup` (skip ISI-floor duplicate removal), `--no-residual-split`, `--no-merge-back`,
`--merge-budget F` (how freely merge-back may recombine; 1.0 default), `--iters N` /
`--converge` / `--converge-tol` (splitting loop control), `--refr-window-ms` (refractory
window). Writes `<base>.clu.stderiv.5.refine`.

---

## 2. `fiber-cpos` — monopole position + per-cluster templates

```
fiber-cpos <session> 5 --clu-method stderiv --clu-stage refine
```

Inverts the raw `.spk` amplitudes to `(x0,y0,z0,A)` per cluster and stamps each with its
centered stderiv template, offset, `t_mid`, `n`. **Always uses raw `.spk` amplitudes — never
`.spkD`/whitened** (the stderiv transform breaks the amplitude-distance law). Writes
`<base>.cpos.stderiv.5.refine.clusters.npz` (keys: `clu,x0,y0,z0,A,template,offset,t_mid,n,…`),
the input table for linking.

---

## 3. `fiber-intrachunk` — per-chunk fragment matching → per-chunk units

```
fiber-intrachunk <session> 5 --cpos-method stderiv --cpos-stage refine --emit-units
```

Within each 12-min chunk, collapses over-split fragments (energy-ladder + time-shift
duplicates) into per-chunk units via a complete-linkage clique on the co-gate. Important args:
`--chunk-minutes 12`, `--cos-thr 0.85` (template cosine gate), `--off-thr 1.0` (inter-channel
offset RMS gate, samples), `--depth-gate 35` (µm; intra-chunk depth window — set ≥30 so an
energy ladder spanning ~28 µm stays one unit), `--min-n 12`, `--boundary-minutes 3.0`
(half-window of chunk-straddling spikes used to build the overlap backbone), `--emit-units`
(write `<...>.units.npz`, the table §4 links). Writes `<base>.clu.…intrachunk.5` (for eyeballing
in Klusters) and, with `--emit-units`, the unit-signature `.units.npz`.

Core functions (Python API):
- `build_signatures(spkD, clu, t_mid_s, pos, *, chunk_min=12, min_n=12, reserve=(0,1))` →
  per-cluster centered template + offset + x0/y0/z0/A + chunk + n.
- `group_intrachunk(sig, *, cos_thr=0.85, off_thr=1.0, depth_gate=35)` → per-fragment labels.
- `aggregate_units(sig, label)` → per-chunk units (weighted-mean template, mutual-centered).
- `overlap_backbone(units, member_spikes, spkD, t_spike_s, *, chunk_min=12, half_window=3.0,
  cos_thr=0.90, off_thr=0.80)` → `(backbone_links, drift{chunk:µm})`. Boundary-straddling
  spikes have ~0 drift, so these are high-confidence anchors **and** a true per-unit drift
  estimate (vs the composition-fragile density xcorr).
- `member_spike_index(src_ids, members)` → unit → member spike rows.

---

## 4. `fiber-link` — track per-chunk units across chunks

```
fiber-link <session> 5 --from-units <base>.…units.npz --max-gap 2 --refine-trajectory
```

Position-fingerprint + A-anchor + template/offset co-gate, seeded by the overlap-backbone and
its per-unit drift, unioned into bundles (= one neuron tracked across chunks). Important args:
`--from-units PATH` (link per-chunk units rather than raw cpos fragments — the normal path),
`--cos-thr 0.85`, `--pos-thr 1.5` (position fingerprint gate), `--off-thr 1.0`, `--max-gap 2`
(bridge single-chunk dropouts), `--min-n`, `--out-stage` (default `<clu-stage>.linked`),
**`--refine-trajectory`** (the §4b post-pass), **`--traj-ext-min M`** (minutes an attach may
extend beyond a bundle's member span; 0 = interpolation only).

`link_session(frag, *, chunk_min=12, cos_thr=0.85, pos_thr=1.5, off_thr=1.0, max_resid=0.08,
min_n=20, gap=1, drift=None, seed_links=None, refine_trajectory=False, traj_ext_min=0.0)` —
pass `drift=` and `seed_links=` from `overlap_backbone`. Returns
`dict(chunk, chunks, D, links, bundles, link_mask, traj_info)`.

### 4b. `--refine-trajectory` (fiber_trajectory) — clean merges + extend tracks

Fits each bundle's depth path `y0(t)` and template-PCA feature path `F(t)` (+ near-flat
`logA(t)`), then: (1) **resolves same-chunk conflicts** — a bundle with ≥2 units in one chunk
is a provable mis-merge, so predict that chunk from the other chunks and keep the closest unit,
evicting the rest; (2) **attaches** singletons lying on a bundle's path in a free chunk slot.
Tolerances self-calibrate from members' residual quantile.

`refine_bundles(frag, bundles, chunk, *, K=4, quantile=0.95, lat=0.25, ext_min=0.0,
chunk_min=12.0, max_iters=8)` → `(new_bundles, info)`. `K` = PCA-feature dims; `quantile` =
member-residual percentile that sets the depth/feature gate; `lat` = logA gate; `ext_min` =
extrapolation window (0 = interpolation only, the safe default; ~chunk length opts into
extrapolation-based track extension). g5: conflicts 31→0 (91 mis-merges evicted), span≥3 72→92,
25 attaches, all in-family by leave-one-out.

Writes `<base>.clu.stderiv.5.refine.linked` (one id per tracked unit; 0/1 reserved).

---

## 5. (optional) `fiber-pca` / `fiber-realign` — realign before re-clustering

Standalone neurosuite-3 `.pcaD` + the Klusters two-stage realign, so you don't depend on the
C++ `process_alignspikes_pca`. Feed the corrected `res`/`spk`/`fet` to a KlustaKwik re-cluster
(with dedupe) — realign alone won't collapse time-shift duplicates, only re-clustering does.

```
fiber-pca <session> 5 --stderiv          # fit a .pcaD basis from .spkD + .clu
fiber-pca --info <base>.pcaD.5           # inspect a basis header
```

Python API:
- `read_pcad(path)` / `write_pcad(path, means, evec, recShift, centered=0)` — byte-exact I/O,
  follows the `process_pca` writer (grouped means-then-evec, unconditional means).
- `fit_basis(windows, nComp=3, centered=True)` → `(means, evec)` per-channel temporal PCA.
- `extract_windows(spk, recShift, data2use)`; `project(windows, basis)` → `.fet`.
- `stage2_shift(cluster_spikes, basis, *, max_global=4)` → rigid PCA-energy shift.
- `realign_pca(spk, clu, res, basis, *, max_shift=5, iters=2, min_n=20, max_global=4)` →
  `dict(res, spk, fet, ioff, s2)`. Stage 1 = xcorr-to-template (`fiber_realign.template_offsets`),
  Stage 2 = PCA-energy rigid center, then reproject. Without raw `.fil` it circular-shifts the
  `.spk` window (exact for the small shifts realign produces).

`fiber-realign <session> 5 --clu <linked.clu> --max-shift 5 --iters 2` runs Stage 1 alone and
writes a corrected `.res`.

---

## 6. `fiber-localize` and `fiber-drift` — final geometry

```
fiber-localize 5 --nsamp <N> --nchan 8 --session <session> --clu <linked.clu>
fiber-drift <base>.fibers.stderiv.5 --pitch 20 --min-span 3 --npy drift_g5.npy
```

`fiber-localize`: bootstrapped monopole/dipole position per tracked unit (`--nboot 200`,
`--min-n 50`, `--max-resid 0.10`, `--no-dipole` for monopole-only). `fiber-drift`: per-unit and
session drift curve `D(c)` in µm (`--pitch 20` converts channel units, `--min-span 3`,
`--min-nspk 60`, `--npy` saves the curve).

---

## 7. `fiber-view` — inspect

`fiber-view` / `fiber-view-gui` (rotatable GL fiber view, projection-mix sliders, interesting-
projection tour) and `fiber-view-tour` for a geodesic keyframe video.

---

## One-line recap

```
refine → cpos → intrachunk(--emit-units) → link(--from-units --refine-trajectory)
       → [pca/realign → re-cluster] → localize → drift → view
```

Everything is membership/position curation over an existing sort. The two levers that move the
merge/track quality: the intra-chunk co-gate (`--cos-thr/--off-thr/--depth-gate`) and the
trajectory refine (`--refine-trajectory`, `--traj-ext-min`).
