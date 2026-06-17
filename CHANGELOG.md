# Changelog

All notable changes to **fiber-kit**. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic-ish
`0.MINOR.PATCH` versions (each minor adds a tool or a self-contained capability).

## [0.27.0] — pipeline driver: positional electrode + rename to `fiber-pipeline`
- `scripts/run_fiber_pipeline` renamed to `scripts/fiber-pipeline` (installed on PATH
  under the new name). The electrode group is now the FIRST positional argument:
  `fiber-pipeline <elec> [all | <stage> ...]`, e.g. `fiber-pipeline 5 fiber-refine`.
  `FK_ELEC` is retained as a fallback when no leading group is given.
- fiber-refine: `--out-variant` renamed to `--out-stage` (it is the output STAGE tag;
  `--out-variant` is reserved for the feature variant everywhere, as in fiber-realign).
- neuro_io.apply_spike_keep: dedup propagation no longer edits backups (dated snapshots,
  `*_bkp`), byte-split fragments (`.part.*`), or sidecars (`.units.npz`); electrode is
  matched exactly as the first numeric token (fixes a latent cross-group match).

## [0.26.0] — installed pipeline driver (`run_fiber_pipeline`)
- New `scripts/run_fiber_pipeline`, installed on PATH by pip (setup.cfg
  `[options] scripts`). Stage dispatcher: `run_fiber_pipeline [all | <stage> ...]`
  where a stage is a tool name (`fiber-refine`, `fiber-cpos`, ...). Requested
  stages always run in canonical order; unknown names are rejected before any work.
- Config is env-overridable (the installed script is not hand-edited):
  `FK_DIR` (default `$PWD`), `FK_SESS` (default basename), `FK_ELEC` (required),
  plus per-knob `FK_*` for the algorithm/warp thresholds.
- Encodes the verified call conventions (realign in-place; cpos after refine on the
  refine clu, from the raw spk) and wires the optional warp knobs (0083-0086),
  passing them only when set so it still runs on a tree without those patches.

## [0.25.0] — residual split integrated into fiber-refine (opt-out)
- `refine()` runs a final residual-split cleanup by default (`fiber_split`): each
  output fiber is split on the residual to its shared d(r) when that lowers
  held-out residual energy beyond a random split of the same cluster. Disable
  with `--no-residual-split`. Applies in single-pass and per-chunk (chunked) mode.
- Params `residual_split=True`, `residual_margin`; an "rsplit" row is logged.
- On the real g5 chunk it carved envelope-similar siblings out of coarse fibers
  (clu-free) spanning the discriminator types: a sub-sample timing shift
  (diff∥∂/∂t, cos 0.93), a distinct waveform shape (cos 0.26), and amplitude.
- NOTE: acceptance is residual-energy-vs-null only; a refractory-improvement /
  time-stability tiebreaker (esp. for pure-timing-shift splits, the most
  alignment-sensitive) is the recommended next gate.

## [0.24.0] — recursive residual splitting (fiber_split)
- `fiber_split`: refine a fiber by splitting on the residual to its single shared
  d(r) (where an envelope-similar second unit hides). A candidate binary split in
  the residual subspace is ACCEPTED only if it lowers OUT-OF-SAMPLE residual
  energy more than a random split of the same node (per-node null), so it never
  rewards the trivial decrease from adding clusters. `recursive_split`,
  `accept_split`, `total_residual_energy`, `shared_fiber_residual`.
- Validated clu-free on the real g5 chunk: refining a coarse CEM baseline cut
  held-out total residual energy ~32%, while a matched random refinement to the
  same cluster count did not (-2%) -> +33% over random at equal K. The splitter
  refines an existing fiber; it does not bootstrap a full sort from one root.

## [0.23.0] — fiber-realign reads the session YAML
- `fiber-realign` now takes `<session> <group>` (like `fiber-refine`) and reads
  group channels / nchan / nsamp from `<session>.yaml`, with `--channels`,
  `--ntotal`, `--nchan`, `--nsamp`, `--sr` overrides. `--nsamp`/`--nchan` are no
  longer required. Backward compatible: a bare file `base` still works (no-YAML
  fallback), and the old `--nch` is kept as an alias of `--nchan`.
- `--clu` unchanged; pass the refined labels, e.g. `<base>.clu.refine.<group>`.

## [0.22.1] — fix: chunked refine duplicate `min_group`
- `refine_chunked` passed `min_group` both explicitly and inside `refine_kw`,
  raising `TypeError: refine() got multiple values for keyword argument
  'min_group'` on any `--chunk-minutes` run. Pop it from `refine_kw` (the
  explicit arg is authoritative). Single-pass mode was unaffected.

## [0.22.0] — drift-predicted, signature-gated continuity linking
- `fiber-refine --chunk-minutes M --link-continuity`: after the overlap-anchor
  backbone, recover fibers too sparse to share enough overlap spikes by bridging
  global tracks across chunk gaps. Coherent drift Δz(t) is estimated from the
  well-linked multi-chunk globals; a track that *ends* is bridged to one that
  *begins* only if the drift-predicted depth matches AND the template signatures
  agree (cosine >= `--continuity-sig-thr`, default 0.6), within
  `--continuity-depth-gate` per chunk and `--continuity-max-gap` chunks. Bridges
  may refuse, preserving genuine discontinuities.
- API: `fiber_session.link_continuity`; `fiber_refine._chunk_fiber_features`
  (energy-weighted channel centroid + template signature). Opt-in; the overlap
  backbone is unchanged when the flag is off.
- The signature gate is what blocks identity swaps when a different unit appears
  on a vanished unit's drift path (validated on synthetic coherent-drift data:
  sig-gated recovers ground truth with 0 id-mixing; the ablation without the
  gate wrongly merges the swap).

## [0.21.0] — fiber-view: most-interesting projection tour
- `fiber-view-tour <base>.bundles.<group>.npz`: guided projection-pursuit tour
  over selected bundles — pools them into one shared PCA space, scores 3-D
  projections by between-bundle separation + drift (`tr(PᵀCP)`), and animates a
  smooth path through the highest-scoring frames (camera spinning). Writes `.gif`
  (Pillow) or `.mp4` (ffmpeg). GUI gains multi-select + a "tour video" button.
- API: `interesting_tour`, `render_tour`, `_select_bundles`.

## [0.20.0] — projection-mix sliders
- The bundle GUI exposes the projection as an editable `ncomp×3` mixing matrix
  (top-K PC scores → the 3 display axes): a slider grid (identity = PC1/2/3,
  each PC row labelled with its variance %) to rotate higher-PC contributions
  into view. PCA is fit once per bundle; slider moves only re-multiply scores.
- API: `projection_basis`, `default_mix`, `apply_mix`; `bundle_figure(..., mix=)`.

## [0.19.0] — fiber-view: figures + rotatable bundle GUI
- `fiber-view` (matplotlib): per-channel interpolated waveform-template montage,
  local-fiber PCA(3) manifold, ISI/`fiber_shape_stats` panels.
- `fiber-view-gui` (PySide6 + pyqtgraph): a **selectable bundle table** (one row
  per global fiber, sorted by drift score) that renders the chosen bundle in a
  rotatable 3-D view — per-chunk trajectories + transparent lofted drift sheet.
- Bundle data layer (`make_bundle`, `bundle_table`, `bundle_drift_score`,
  `bundle_figure`, `load_bundles_npz`); HDR spike-density isosurfaces helper.
- `fiber-refine --chunk-minutes M --bundles` writes `<base>.bundles.<group>.npz`
  (per-chunk un-whitened template curves, comparable across chunks).
- New `[viz]` extra (matplotlib + pyqtgraph + PySide6).

## [0.18.0] — geometry output as lossless npz
- `<base>.geom.<group>.npz` / `<base>.geomchunk.<group>.npz` replace the lossy
  `%.4g` TSV; long-format, no object arrays. Added `load_geometry`.

## [0.17.0] — drift-aware chunked mode
- `fiber-refine --chunk-minutes M`: window the session (disjoint cores + overlap
  exts), fit a separate whitener and run the full refine loop per quasi-stationary
  window, then link per-window fibers by overlap-anchor (`fs.link_chunks`); final
  label by core window. `--track-geometry` writes the per-window (drift) geometry.

## [0.16.0] — fit/reassign re-seed loop + geometry tracking
- `--reseed N`: each extra pass is split → merge_back → **refit fibers →
  reassign** (`run_from_seeds`), converging instead of drifting.
- `fiber_tracer.fiber_shape_stats`: per-fiber radius/cone/bend/smoothness/bimodality.
- `--track-geometry`: link each final fiber back through iteration snapshots by
  spike overlap and record its geometry time series.

## [0.15.0] — shape-distinctness split gate + merge-back on
- `--split-min-corr`: don't carve off a split piece / peeled bucket whose
  normalised median waveform matches its parent (stops energy-level
  over-fragmentation). `merge_back` on by default; `--reseed` (label re-feed).

## [0.14.0] — fiber-refine convergence + contamination-gated merge-back
- Early-stop once nfib/swBand/enCV hold steady; merge-back gated on the imposed
  refractory band.

## [0.13.0] — fiber-refine
- New tool: dedup at the imposed refractory, then an iterative gated
  split/peel/isolate cascade (rkk/dip/knn-peel), each split gated on
  per-channel residual-variance reduction without worsening <1 ms refractory.

## [0.12.0] — iterated circular-xcorr alignment
- `fiber_lib.align_xcorr`: align each spike to the cluster median by channel-summed
  xcorr with sub-sample Fourier phase shift, making the residual-variance measure
  meaningful.

## [0.11.0] — variance-driven auto-split
- Recursive KMeans bisection on the channel-weighted trajectory residual, with
  per-channel residual-variance reduction as the stop criterion.

## [0.10.0] — per-channel residual-variance membership
- `channel_residual_profile` / `split_meanvar`: tighten membership and gauge
  contamination by per-channel variance of the residual to the energy-local
  template; per-channel outlier cone + variance-margin split acceptance.

## [0.9.0] — fiber-position
- `fiber-position`: drift-independent per-spike normalized position along the
  consolidated fiber manifold (direction-based).

## [0.8.0] — fiber-drift
- `fiber-drift`: probe drift tracking from fiber depth trajectories.
  *(Base commit `d981297` for the current patch series.)*

## [0.7.0] — fiber-localize
- Monopole + dipole physical localization from raw-waveform spatial spread.

## [0.6.0] — performance
- Vectorized realign/predict/template alignment; optional CuPy GPU backend for
  the realign/whiten kernels; chunk-level CPU parallelism (`--jobs`); batched
  EWMA and collision decomposition; optimization verification/benchmark harness.

## [0.5.0] — standardized I/O
- `neuro_io`: variant resolution and `.res`/`.clu`/`.fet`/`.spk` readers + writers;
  all fiber-kit I/O routed through it.

## [0.4.0] — fiber-realign
- Per-spike fiber-template offsets + corrected `.res`.

## [0.3.0] — fiber-relink
- Post-hoc geometry-aware re-bundling / re-linking of an existing sort.

## [0.2.1] — whitener memory cap
- Sample the whitener baseline from the memmap; cap memory at O(n_base).

## [0.2.0] — session YAML
- `SESSION.yaml` autoloading; CLIs take `<session> <group>`.

## [0.1.0] — initial
- Coarse mean-shift fibers, cross-chunk overlap-anchor linking, per-fiber
  geometry + quality / firing / drift statistics; `fiber-session` pipeline.
