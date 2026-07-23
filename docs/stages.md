# fiber-kit — stage & parameter reference

Every `fiber-*` command is a pipeline stage or tool.  This file is generated from each stage's
argument parser, so the parameters and defaults below match the code.  A stage's *input* is a `.clu`
(selected by `--clu-method`/`--clu-stage`, or `--in-clu`) plus the session's `.res` and `.spk`; its
*output* is a new `.clu` under `--out-stage` (`--out-tag`).  Session geometry (channels, samples, peak,
sampling rate) is read from `<session>.yaml`.  Positionals are `<session> <elec>` unless noted.

Run any stage with `-h` for the live help, or `fiber-pipeline <elec> <stage> [args…]` to run it in the
pipeline (see [pipeline.md](pipeline.md)).

---

## Flag vocabulary

Three words, one meaning each, across every stage.

| word | means | where it lands in a filename |
|---|---|---|
| **method** | the operation the clusters stem from — `standard`, `stderiv`, `stderiv_C5` | the slot **before** the group: `<base>.<type>.<method>.<group>` |
| **stage** | the post-fiber tag, so each processing step can keep its own cluster files | the slot **after** the group: `…<group>.<stage>` |
| **algo** | a choice of algorithm — nothing to do with file naming | — |

A qualified prefix says which artifact is meant when a stage touches several
(`--clu-method`, `--spk-method`, `--out-stage`); the **suffix** carries the
meaning.  So `--clu-method stderiv_C5 --clu-stage refine` reads
`<base>.clu.stderiv_C5.<group>.refine`.

`--method` used to be spelled `--variant` in some stages and mean the *stage* in
others; `-method` was also used for algorithm choices.  **Every previous spelling
still works as an alias**, so existing scripts and `plans/*.yaml` are unaffected —
the tables below show the canonical name first with the old one in parentheses.

---

## Core pipeline (canonical order)

### `fiber-session`

Cluster a session group into fibers.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--chunk-min` (`--chunk-minutes`) | `12.0` |  |
| `--overlap-min` | `4.0` |  |
| `--min-group` | `200` | COARSE min spikes/fiber (for linking) |
| `--fine-algo` (`--fine-method`) | `gmm` | choices: `gmm`, `rkk`, `fiber`, `none` |
| `--rkk-dims` | `6` |  |
| `--rkk-max` | `50` |  |
| `--rkk-realign` | flag (off) | interleave rkk (CEM) with per-cluster realignment (per-step; default on) |
| `--no-rkk-realign` | flag (on) | legacy: one parent realign + fixed features for the rkk split |
| `--rkk-realign-iters` | `2` | cluster<->realign passes in the rkk realign loop |
| `--rkk-delete` | flag (off) | rkk (CEM) culls sub-min-group sub-clusters during the per-fiber fine split (default on) |
| `--no-rkk-delete` | flag (on) | keep small non-singular rkk sub-clusters -- session should OVER-cluster, leaving the cull to refine; use to stop session shedding fragments into the residual/artifact bin |
| `--refit-iters` | `0` | iterate {realign each fiber's spikes to their own median -> refit template -> re-apply the inclusion gate} this many extra times. Recovers true members rejected for a bad sub-sample offset and tightens the template; re-judges from the full member set each pass. 0 = single pass (original). Marginal on clean fibers; helps contamination-heavy ones. |
| `--reseed-residual-thr` | `0.0` | residual-gated reseed: after the fine split, any fiber whose median-subtracted residual/signal exceeds this is re-split (a welded pair of co-located different-shape cells scores high; ~0.35 separates welded from clean on g5). 0 = off. The re-split is kept only if it lowers the residual by --reseed-min-reduction. |
| `--reseed-min-reduction` | `0.05` | keep a residual-gated reseed only if it lowers the median-subtracted residual/signal by at least this much (a real sub-unit does; noise does not) |
| `--seed-density` | `0.0` | density-preferential coarse seeding: draw ridge seeds toward concentrated modes of the waveform space (p proportional to local_density**this) instead of a uniform stride. 0 = uniform (current); 1 = fully density-weighted; small values a gentle tilt. Concentrates the seed budget on isolated, reproducible modes (isolation, not firing rate). |
| `--coarse-dr` | `0.15` | radial band half-width (fraction of the 1-99%% whitened-radius span) for spike->seed association in the COARSE pass. Lower -> tighter amplitude bands -> more, smaller coarse fibers (amplitude-distinct cells stay separate for the fine splitter); higher -> fewer, fatter coarse groups. Default 0.15. |
| `--no-whiten` | flag (off) | cluster in the RAW mask-selected (mean-centred) feature space instead of the .fil baseline-whitened space: identity whitener, so the radial coordinate is the raw feature norm and the angular metric is raw cosine (no per-channel covariance normalisation) |
| `--merge-corr` | `0.0` | consolidate fibers above this (0=off; 0.95 template / 0.90 sliding) |
| `--resplit-passes` | `0` | iterative residual-gated re-split (em_swap on target-channel residual) + correlation merge; 0=off. Replaces the Block-A/B consolidation when >0. |
| `--resplit-residual-thr` | `0.08` | re-split only fibers whose amplitude-scaled max residual (+-8 @ RMS peak) exceeds this (~0.08 for stderiv, ~0.15 for standard waveforms) |
| `--resplit-topch` | `3` | channels fed to em_swap (top residual-variance) |
| `--resplit-min-reduction` | `0.2` | keep an em_swap split only if it cuts target-channel variance by >= this |
| `--resplit-merge-corr` | `0.99` | correlation merge threshold inside the loop |
| `--resplit-detrend-episode` | flag (off) | before each em_swap, strip the episode-position axis (the direction covarying with spikes-after minus spikes-before in a +-90 ms window) from the residual, so the split cannot cut a cell along its own temporal gradient and manufacture an asymmetric CCG |
| `--resplit-detrend-win` | `90.0` | half-window (ms) for the episode-position count |
| `--resplit-detrend-min-n` | `100` | skip the detrend below this many spikes -- the axis is a covariance estimate and is unreliable on small groups |
| `--cfiber-gate` | flag (off) | veto Block-A fragment merges whose affine-invariant cfiber shape disagrees beyond the per-chunk within-fiber null (precision gate; threshold self-calibrated at --cfiber-q) |
| `--cfiber-q` | `0.9` | quantile of the within-fiber split-half cfiber null used as the --cfiber-gate veto threshold |
| `--merge-algo` (`--merge-method`) | `template` | choices: `template`, `sliding`, `profile` |
| `--sliding-nwin` | `14` |  |
| `--profile-thr` | — | profile-merge direction-distance threshold; default = auto same-neuron floor |
| `--profile-floor-pct` | `90.0` | percentile of within-fiber-half distances used as the auto threshold |
| `--profile-min-n` | `120` | min spikes/fiber to be eligible for a profile merge (trajectory reliability) |
| `--emit-merge-candidates` | flag (off) | write proposed same-neuron merges to <base>.merge_candidates.<elec>.tsv WITHOUT merging (curation) |
| `--refrac-ms` | `0.0` | DEFAULT OFF. >0 gates each within-chunk profile-merge through a refractory cross-correlogram veto: a merge whose two trains coincide at chance level (two neurons, no dip) is blocked. Profile merges are permanent (relink/defrag only ever merge), so this stops irreversible over-merges at the source. Power-aware: ABSTAINS at low rate, only ever removes a false merge. |
| `--refrac-thr` | `0.3` | coincidence ratio above which the pair is 'two neurons' (default 0.3) |
| `--refrac-min-exp` | `5.0` | min expected coincidences for the refractory test to be powered (default 5) |
| `--refrac-censor-ms` | `0.0` | censor window (ms) dropping duplicate detections of one spike (default 0) |
| `--deadapt` | flag (off) | de-adapt (EWMA-tau) RS coarse fibers before splitting |
| `--deadapt-min-corr` | `0.2` |  |
| `--adapt-clean` | flag (off) | reject high-energy-at-short-ISI spikes on real fast adapters |
| `--adapt-z` | `3.0` |  |
| `--adapt-isi-ms` | `10.0` |  |
| `--adapt-clean-corr` | `0.4` |  |
| `--adapt-clean-snr` | `0.5` |  |
| `--adapt-taumax` | `0.5` |  |
| `--collision-flag` | flag (off) | route recoverable collisions from noise to a dedicated collision cluster |
| `--collision-gain` | `0.09` |  |
| `--collision-shift` | `8` |  |
| `--quality-metrics` | flag (off) | also compute L-ratio + isolation distance (O(N*K)) |
| `--quality-dims` | `10` | PCA dims for L-ratio/isolation Mahalanobis |
| `--pca-k` | `6` |  |
| `--max-sub` | `8` |  |
| `--inclusion-k` | `3.0` | per-fiber radius = median+k*MAD of residuals; 0 disables |
| `--no-noise` | flag (off) | sweep every remaining noise spike (below the inclusion radius / rejected / collision junk) into a single UNDEFINED FIBER (one real cluster, not the noise cluster) rather than dropping it. For clean stderiv data; the undefined fiber is cleaned/re-split in later steps. |
| `--incl-assign-rejected` | flag (off) | assign spikes beyond the per-fiber inclusion radius to that fiber (kept in the sort) instead of dropping them to the unsorted bin. Geometry/templates still use the pure core; this only rescues the good high-amplitude tail spikes the radius would otherwise discard. |
| `--energy-band` | flag (off) | energy-band split: partition each ENERGY-CONFOUNDED coarse fiber into overlapping log10-energy bands, BIC-GMM per band (global features), relink by overlap-anchor; surfaces shape sub-units the drift axis masks |
| `--eband-width` | `0.45` | energy-band width in decades (default 0.45) |
| `--eband-overlap` | `0.2` | energy-band overlap in decades for overlap-anchor linking (default 0.2) |
| `--eband-confound` | `0.4` | only band a fiber when PC1 R^2 vs log-energy >= this (default 0.4) |
| `--eband-min-span` | `0.6` | only band a fiber spanning >= this many decades (default 0.6) |
| `--eband-min-band` | `60` | min spikes per band to cluster (default 60) |
| `--eband-low-assign` | `0.0` | fraction of the energy range (from the bottom) made ASSIGNMENT-ONLY: in that low-SNR floor the direction is noise, so its spikes are assigned to units from the bands above instead of independently split (default 0.0 = split every band) |
| `--cone-channel-k` | `0.0` | tighten the cone per channel: drop spikes that are residual outliers (>k MAD) on the discriminative channels; 0 disables |
| `--split-var-margin` | `0.0` | accept a within-fiber split only if it lowers the mean per-channel residual variance by >= this fraction (e.g. 0.1); 0 accepts all splits |
| `--var-split` | `0.0` | auto-split fibers whose per-channel residual profile is peaked: trigger when max/median channel residual variance >= this ratio (e.g. 2.0); 0 disables. Bisects on the high-variance channels, accepting only variance-reducing splits. |
| `--var-split-depth` | `4` | max recursion depth for --var-split (max 2^depth sub-units per fiber) |
| `--dipsplit` | flag (off) |  |
| `--no-dipsplit` | flag (on) |  |
| `--dip-dim` | `4` |  |
| `--dip-alpha` | `0.01` |  |
| `--dip-min` | `40` |  |
| `--dip-realign` | flag (off) | realign each dipsplit node to its own median before splitting (per-step alignment; default on) |
| `--no-dip-realign` | flag (on) | legacy: one parent realign + fixed features for the whole dipsplit recursion |
| `--nudge-split` | flag (off) | for low-amp clusters, split temporally-offset overlaid units by alignment lag (similar-shape neurons a few samples apart that median realign merges); default on |
| `--no-nudge-split` | flag (on) |  |
| `--nudge-max` | `3` | max +/- sample lag tested for offset overlays |
| `--nudge-amp-pct` | `40.0` | only clusters below this template-amplitude percentile are nudge-split |
| `--nudge-min-channels` | `4` | min signal channels for the broad-noise condition |
| `--nudge-alpha` | `0.01` | dip-test p for the lag-bimodality split |
| `--fine-kappa` | `40.0` |  |
| `--fine-dedup-deg` | `5.0` |  |
| `--fine-min-group` | `40` |  |
| `--no-fine` | flag (off) | coarse fibers only, no within-chunk refinement |
| `--min-anchor` | `8` |  |
| `--no-link` | flag (off) |  |
| `--n-grid` | `40` |  |
| `--method` | `stderiv` | extraction method tag in the .fibers filename |
| `--no-cluster-basis` | flag (off) | ignore the global .pca basis for the fine-split shape features and use a per-call local SVD (legacy behaviour) |
| `--clu-stage` | `fiber_session` | post-group stage tag for the clu: <base>.clu.<method>.<elec>[.<stage>] (default 'fiber_session'); pass --clu-stage '' for an untagged .clu |
| `--emit-hierarchy` / `--no-emit-hierarchy` | flag (on) | emit the .clu/.clc/.clp microfiber triple (atoms = pre-link fine fragments, fibers = linked global ids) via FiberHierarchy, instead of a flat .clu only. --no-emit-hierarchy writes just the flat .clu (legacy). Ignored with --out. |
| `--gpu` | flag (off) | run the realign/whiten kernels on GPU (CuPy; needs the [gpu] extra) |
| `--jobs` (`-j`) | `1` | parallel worker processes over chunks (default 1 = serial; chunks are independent) |
| `--feature-align` | — | feature-building alignment: xcorr (default) or centroid (pure, no refine -- adds the trough-position-vs-asymmetry structure to the clustering/linking features). Does NOT touch committing alignment or fiber-realign. Overrides the FIBER_ALIGN env var. — choices: `xcorr`, `centroid` |
| `--subsample` / `--no-subsample` | flag (off) | enable (--subsample) or disable (--no-subsample) realign's per-spike sub-sample (parabolic) refine in the feature build; default leaves the FIBER_SUBSAMPLE env var / lever untouched (off). Reaches pool workers. |
| `--out` | — |  |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-realign`

Per-spike Klusters-style realignment + corrected .res spike times, with optional re-extraction from .fil and re-featurisation (commit-and-reextract finalize).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu` | — | cluster file (default <base>.clu.<group>; pass the refined/relinked one, e.g. <base>.clu.stderiv.<group>.refine) |
| `--align-algo` (`--align-method`) | `klusters` | alignment algorithm (runs on the clu's variant waveform): klusters = iterative normalised-xcorr vs pre-aligned mean; template = legacy median/un-normalised; centroid = reference-free per-spike energy-centroid (fiber_lib.centroid_shift), no peak/template/labels needed. (Named --align-method to avoid colliding with the --method extraction-variant flag used by other tools.) — choices: `klusters`, `template`, `centroid` |
| `--max-shift` | `8` |  |
| `--iters` | `4` |  |
| `--min-n` | `20` |  |
| `--min-score` | `0.0` | klusters: leave a spike unshifted if its best cosine score is below this |
| `--upsample` | `1` | klusters: cubic-spline upsample factor for sub-sample matching (2 = half-sample lags); .res is rounded back to whole samples at save |
| `--reextract` | flag (off) | re-extract each spike's window from .fil at the corrected timestamp -> new .spk |
| `--shift-spk` | flag (off) | commit WITHOUT a .fil: circularly roll the existing .spk of each variant by the per-spike integer offset (valid for high-pass .spk; the stderiv transform commutes with a time shift). The no-.fil equivalent of --reextract; pair with --refeaturize to reproject the rolled windows onto .pca. Mutually exclusive with --reextract. |
| `--refeaturize` | flag (off) | reproject the re-extracted/rolled windows onto .pca.<variant> -> new .fet (implies --reextract, or --shift-spk when that is set) |
| `--fil` | — | filtered signal path (default <base>.fil) |
| `--variants` | — | comma list of feature spaces to refresh from .fil (default: standard + stderiv if present). Each is re-derived from the re-extracted raw window: standard=raw; stderiv=SDIFF_ALLPAIRS+temporal-diff; stderiv_C4/_C5 use the session's own spikeDetection.channelGroups[N].sdiffPairs pattern (partner map / reference sets); then projected onto its .pca |
| `--out-stage` (`--out-tag`) | `""` | stage tag for committed outputs (default: empty -> overwrite the canonical .res/.clu/.spk/.fet[.<variant>].<group> in place; the realign IS the commit). Pass a tag only if you want a side-by-side copy, e.g. --out-tag realigned |
| `--out-variant` | — | variant the .res/.clu adhere to (default: inferred from --clu, e.g. stderiv; falls back to standard). There is one .res/.clu under this variant; .spk/.fet are written per feature space in --variants (standard raw, stderiv transform) |
| `--emit-clu` | flag (off) | re-emit the (label-unchanged) clu next to the committed .res/.spk/.fet so the set opens in Klusters as a unit (default on) |
| `--no-emit-clu` | flag (on) | do NOT write the clu. Use when the input --clu is a stage-tagged clu but the outputs commit canonically (--out-tag ''): re-emitting would overwrite the BASE .clu.<variant>.<group> with this stage's labels. The stage clu already exists and its labels are unchanged, so skipping the write keeps the base over-cluster intact. |
| `--out-res` | — |  |
| `--out-off` | — |  |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-refine`

Dedup at the imposed refractory, then iteratively split/peel a fine sort into clean units; writes a refined .clu (+ deduped .res).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--in-clu` | — | path to the input sort to refine; default = canonical .clu if present, else a fresh fine sort |
| `--out-method` | `stderiv` | feature space written BEFORE the group (standard\|stderiv\|...); refine operates in stderiv space, so default stderiv |
| `--no-cluster-basis` | flag (off) | ignore the global .pca basis for the split-stage shape features and use a per-call local SVD (legacy behaviour) |
| `--out-stage` | `refine` | fiber STAGE written AFTER the group, e.g. 'refine' -> <base>.clu.<out-method>.<group>.refine ('' for none). (was --out-variant; renamed so --out-variant is free for the feature variant, as in fiber-realign) |
| `--emit-hierarchy` / `--no-emit-hierarchy` | flag (on) | after writing the .clu, also emit the .clc/.clp microfiber triple via the fiber-microfiberize identity lift (each fiber = one atom) so the refine output is a hierarchy intrachunk/link can consume; --no-emit-hierarchy writes the flat .clu only |
| `--refr-floor` | — | imposed detection refractory (samples); default = from yaml |
| `--refr-window-ms` | `2.0` | biological/ISI-violation window upper bound (ms); contamination is [floor, window) |
| `--dedup` / `--no-dedup` | flag (off) | run the sub-floor dedup pass (drops near-coincident sub-threshold duplicate detections, ~200-300 spikes). OFF by default: dedup re-indexes the spike list so a deduped clu no longer aligns 1:1 with the canonical .res, which breaks round-tripping of curated clu files. --no-dedup is the explicit off; --dedup re-enables. |
| `--no-residual-split` | flag (on) | disable the final residual-split cleanup (on by default): each output fiber is split on the residual to its shared d(r) when that lowers held-out residual energy beyond a random split of the same cluster |
| `--iters` | `10` | max splitting iterations (cap) |
| `--converge` | flag (off) | stop the splitting phase early once nfib/swBand/enCV are steady (default on) |
| `--no-converge` | flag (on) |  |
| `--converge-tol` | `0.01` | nfib change (fraction) below which an iteration counts as steady |
| `--converge-patience` | `2` | number of consecutive steady iters required to stop |
| `--merge-back` | flag (off) | final contamination-gated merge-back to a reasonable count (default on) |
| `--no-merge-back` | flag (on) |  |
| `--merge-budget` | `1.0` | max merged-cluster [floor,window) band%% to accept a merge |
| `--merge-min-sim` | `0.92` | min median-waveform similarity to consider a merge |
| `--merge-warp-thr` | — | final-merge WARP gate (Omlor-Giese group delay): require the cross-channel correlation of the two clusters' median-template group-delay profiles >= this. On clean well-populated clusters the warp is a very clean signature (g5: same neuron ~0.99, different ~0; it morphs continuously with drift, adjacent-bin change ~0.004), so ~0.9 is safe -- lower --merge-min-sim and set this to recover the last merges without false ones. None (default) = off. |
| `--merge-warp-recall` | — | DRIFT-FRAGMENT recall path for merge-back: also admit a merge when the median-template group-delay (WARP) correlation >= this AND the per-channel amplitude-profile correlation >= --merge-amp-thr, EVEN IF cosine is low. Drift changes waveform shape (dropping cosine) but preserves the per-channel delay+amplitude structure, so this recovers same-neuron fragments cosine misses (g5 over-clusters: warp>=0.9 & amp>=0.7 recovered 323/395 such merges at >=0.976 precision). Refractory budget remains the final gate. None (default) = off; ~0.9. |
| `--merge-amp-thr` | `0.7` | amplitude-profile correlation floor for the --merge-warp-recall path (Omlor-Giese magnitude term) |
| `--merge-warp-resid-thr` | — | single-channel warp-incongruity SUB-gate layered on --merge-warp-thr / --merge-warp-recall: among pairs whose overall group-delay (warp) correlation is already coherent (>=0.85), veto the merge if any ONE channel's group-delay residual (robust Theil-Sen per-channel delay line) exceeds this many samples. warp_correlation is a cross-channel Pearson, so a couple of strong channels can hold it high while one channel betrays a different co-located source; this catches that (especially on the low-cosine warp-recall admits). g5: vetoes ~9%% of warp-coherent merge-admissible pairs at ~1.0. None (default) = off. |
| `--merge-warp-resid-thr-int` | — | cell-type-aware warp-incongruity SUB-gate: threshold for pairs touching an INTERNEURON (narrow trough-to-peak). Interneurons fire fast so their per-channel timing is very stable (offset RMS ~0.23), making a single-channel group-delay residual a real different-source signature -> tighter than pyr (~0.7). Needs raw .spk for cell-typing. Set BOTH _INT and _PYR to enable the dual gate (each falls back to --merge-warp-resid-thr if only one is set). |
| `--merge-warp-resid-thr-pyr` | — | cell-type-aware warp-incongruity SUB-gate: threshold for PYRAMIDAL pairs (wide trough-to-peak). Pyramidal timing jitters more (offset RMS ~0.72), so a larger single-channel residual is benign (~1.3). A pair touching an interneuron uses the stricter _INT threshold. |
| `--merge-mode` | `normalized` | normalized = merge energy levels (neuron count); amplitude = keep them — choices: `normalized`, `amplitude` |
| `--split-min-corr` | `0.93` | shape-distinctness gate: do NOT carve off a split piece / energy bucket whose normalised median waveform correlates >= this with its parent (stops over-fragmenting high-rate units into energy-level clones); 1.0 disables |
| `--fold-off-thr` | — | inter-channel TIMING veto on the knn contaminant-fold: when set, a small group is NOT folded into an amplitude/shape-similar target if their robust (raw xcorr-lag) inter-channel offset profiles differ by more than this many samples -- protects small but timing-distinct cells (e.g. 294 vs 295) from being shattered into a look-alike. g5 calibration: same-cell ~0.11, distinct co-located cells ~0.26, so ~0.2-0.25 catches them. None (default) = off. |
| `--reseed` | `0` | re-run the whole loop (split -> merge -> refit fibers -> reassign) using the refined labels as the next seed, up to N extra passes (e.g. 1 = 2 passes); 0 = single pass |
| `--track-geometry` | flag (off) | record per-fiber geometry (radius/cone/smoothness/bend) at every iteration and write <base>.geom.<group>.npz tracking each final fiber back through the loop |
| `--large` | `800` | only clusters >= this are split each iter |
| `--min-group` | `40` |  |
| `--drop-min` | — | cluster-KEEP floor: clusters smaller than this are dropped to the artifact bin each iteration. Decoupled from --min-group (the split-PIECE floor). Default None = use --min-group (legacy). Set LOW (e.g. 5) so session's small over-split fragments survive to be stitched by intrachunk instead of being discarded. |
| `--var-margin` | `0.05` | min per-channel residual-variance reduction to accept a gated sub-split |
| `--brr-tol` | `0.3` | max allowed increase (pp) in [floor,window) refractory for a gated sub-split |
| `--var-peak` | `2.0` | var-split trigger (max/median channel variance) |
| `--split-var-mult` | `0.0` | curator split-variance gate: only attempt to split a cluster whose top-3 whitened-feature variance exceeds this multiple of the median over splittable clusters. From the g5 curation log, split-source clusters carry ~3.6x the median feature variance (and kurtosis/ISI are NOT the trigger), so ~1.5-2.0 matches the curator; below it clusters pass through unsplit. 0 (default) = off (no variance prefilter). |
| `--var-depth` | `4` |  |
| `--dip-realign` | flag (off) | realign each dipsplit node to its own median (per-step; default on) |
| `--no-dip-realign` | flag (on) |  |
| `--rkk-realign` | flag (off) | interleave rkk (CEM) with per-cluster realignment (default on) |
| `--no-rkk-realign` | flag (on) |  |
| `--rkk-realign-iters` | `2` |  |
| `--rkk-delete` | flag (off) | rkk (CEM) culls sub-min-group clusters during the split (default on) |
| `--no-rkk-delete` | flag (on) | keep small (non-singular) rkk sub-clusters instead of dissolving them -- stops the size-cull from shedding spikes that the cascade then sends to the residual/artifact bin. Use when too many spikes land in the artifact cluster. |
| `--ccg-refrac-ms` | `0.0` | curation-independent veto: reject a split the refractory cross-correlogram calls spurious (one neuron) where it has power; 0 = off. ~1.5 to enable. Abstains (no effect) at low firing rates. |
| `--ab-reclaim` / `--no-ab-reclaim` | flag (off) | after merge_back, run the targeted A/B contamination reclaim: move a host cluster's spikes into a clean, DISTINCT donor when their realigned shape matches the donor (refractory-safe, per-spike, reclaim-only). Recovers foreign spikes a non-co-firing contaminant leaves invisible to the refractory metric. Default off. |
| `--ab-distinct` | `0.93` | A/B reclaim: max donor/host template shape-corr to treat them as two cells (>= this is one cell -> merge_back, not reclaim). Validated knee ~0.93. |
| `--ab-abs` | `0.5` | A/B reclaim: absolute shape-corr a spike must reach to the donor template to move. |
| `--ab-margin` | `0.05` | A/B reclaim: how much better a spike must match the donor than its host to move (bounds false-grab of the host's own spikes). |
| `--ab-min` | `10` | A/B reclaim: minimum spikes a donor must reclaim from one host to commit the move. |
| `--ab-sigcap` | `2000` | A/B reclaim: cap on spikes used to estimate each cluster's median TEMPLATE (not its membership/scoring). A ~2000-spike sample matches the full template to ~1e-3 while avoiding the iterated align on whole-session merged clusters (tens of thousands of spikes) -- the pass's bottleneck in whole-session mode. 0 = no cap (use all spikes). |
| `--ab-jobs` | `1` | A/B reclaim: worker threads for the per-cluster template precompute (the align bottleneck). Result is identical for any value (templates are sampled before the parallel work). 1 = serial. Helps in WHOLE-SESSION mode (big clusters); in chunked mode clusters are small so it barely matters. |
| `--rkk-first` | flag (on) | restore the old cascade order (rkk before dip-bisection); default is dip-first, which targets the single high-margin dip axis welds separate on |
| `--nudge-split` | flag (off) | split temporally-offset overlaid units in low-amp clusters by alignment lag (residual-neutral; for from-scratch/coarse sorts, default off) |
| `--nudge-max` | `3` |  |
| `--nudge-amp-pct` | `40.0` |  |
| `--nudge-min-channels` | `4` |  |
| `--nudge-alpha` | `0.01` |  |
| `--knn-k` | `20` |  |
| `--knn-thr` | `0.3` | K-NN majority fraction to peel a spike |
| `--knn-minref` | `50` |  |
| `--knn-minnew` | `30` |  |
| `--knn-dims` | `16` |  |
| `--fold-thr` | `0.9` | non-normalised median-xcorr above which a peeled bucket is folded (else kept as new) |
| `--fine-algo` (`--fine-method`) | `gmm` | method for the initial fine sort when no --in-clu is given — choices: `gmm`, `fiber`, `none` |
| `--chunk-minutes` (`--chunk-min`) | `0.0` | drift-aware mode: window the session into CORE chunks of this many minutes, refine each in its own whitened frame, and link fibers across windows by overlap-anchor; 0 = single whole-session pass (assumes stationary) |
| `--chunk-overlap-minutes` | `1.0` | overlap between adjacent windows used for overlap-anchor linking (drift-aware mode) |
| `--chunk-jobs` | `1` | parallel worker PROCESSES over chunks in drift-aware mode (default 1 = serial). Chunks are independent (own whitener + refine), so this is the main speedup for a chunked run; the cross-window link runs serially after. Workers re-open the .spkD/.fil memmaps, so memory is bounded. No effect in whole-session mode (--chunk-minutes 0). |
| `--bundles` | flag (off) | drift-aware mode: also write <base>.bundles.<group>.npz (per-chunk un-whitened template curves per global fiber) for the fiber-view-gui bundle table |
| `--legacy-link` | flag (off) | use the old overlap-only link_chunks (no geometry/timing veto) |
| `--min-anchor` | `20` | min shared overlap spikes to link two per-chunk fibers (strict linker) |
| `--link-continuity` | flag (off) | drift-aware mode: after overlap-anchor linking, bridge sparse fibers that share too few overlap spikes using a drift-predicted, signature-gated continuity fallback |
| `--continuity-sig-thr` | `0.6` | min template cosine to allow a continuity bridge (signature gate; default 0.6) |
| `--continuity-depth-gate` | `14.0` | max drift-predicted depth error (per chunk of gap) for a continuity bridge |
| `--continuity-max-gap` | `2` | max chunk gap a continuity bridge may span (default 2) |
| `--gpu` | flag (off) |  |
| `--feature-align` | — | feature-building alignment: xcorr (default) or centroid (pure, no refine -- adds the trough-position-vs-asymmetry structure to the clustering/linking features). Does NOT touch committing alignment or fiber-realign. Overrides the FIBER_ALIGN env var. — choices: `xcorr`, `centroid` |
| `--dedup-strict` / `--no-dedup-strict` | flag (on) | when a dedup removes spikes, every live per-spike file of the group is subset too. A live file at a third row count (neither pre- nor post-dedup) is a stale leftover from an earlier extraction that CANNOT be subset by this mask; by default it is quarantined aside (.stalebkp) and regenerated. --no-dedup-strict leaves such files in place instead; --dedup-stale error restores a hard failure. |
| `--dedup-stale` | — | explicit policy for stale leftover per-spike files at a third row count. Default (unset) QUARANTINES them aside as <file>.stalebkp -- non-destructive, leaves the group consistent, and the stage regenerates them. error = hard-fail (old strict behavior); skip = leave them (= --no-dedup-strict). Default follows --dedup-strict. — choices: `error`, `skip`, `quarantine` |
| `--subsample` / `--no-subsample` | flag (off) | enable (--subsample) or disable (--no-subsample) realign's per-spike sub-sample (parabolic) refine in the feature build; default leaves the FIBER_SUBSAMPLE env var / lever untouched (off). Reaches pool workers. |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-peel`

fiber-peel: consolidate over-split refine fragments by footprint cosine gated by the refractory cross-CCG.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--cpos-method` | `stderiv` |  |
| `--cpos-stage` | `refine` |  |
| `--clu-method` | — |  |
| `--clu-stage` | — |  |
| `--out-stage` | — | output clu stage (default: same as input == in-place relabel) |
| `--foot-hi` | `0.97` | anneal start: strict footprint-cosine for the first, most certain merges |
| `--foot-lo` | `0.9` | anneal floor: loosest footprint-cosine accepted (best g5 result at 0.90) |
| `--anneal-steps` | `4` |  |
| `--refrac-ms` | `2.0` | refractory half-window (ms) for the veto cross-CCG |
| `--refrac-thr` | `0.3` | coincidence-ratio above which the pair is two cells (veto) |
| `--refrac-min-exp` | `5.0` | min expected coincidences for the veto to be powered (else abstain -> merge allowed) |
| `--refrac-censor-ms` | `0.0` | censor window (ms) dropping duplicate detections of one spike |
| `--min-n` | `15` | min spikes for a fragment to participate |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-cpos`

Write a per-spike cluster-position sidecar (.cpos) by localizing each cluster's median RAW template (monopole+dipole).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--spk` | — | path to a STANDARD/raw .spk (preferred over .fil); never the stderiv .spkD |
| `--spk-method` | `standard` | method of the raw .spk to resolve: <base>.spk.<spk-method>.<elec> |
| `--fil` | — | path to raw .fil (fallback if no standard .spk) |
| `--fil-offset` | `0` | first absolute sample of the .fil (0 for full recording) |
| `--clu-method` | `stderiv` | source-clu feature space BEFORE the group (standard\|stderiv\|...) |
| `--clu-stage` | `refine` | source-clu fiber STAGE AFTER the group: read <base>.clu.<clu-method>.<elec>.<clu-stage> |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--clu-stage) |
| `--out-method` | — | cpos method BEFORE the group (default: mirror --clu-method) |
| `--out-stage` | — | cpos fiber STAGE AFTER the group (default: mirror --clu-stage) |
| `--min-spikes` | `15` |  |
| `--no-dipole` | flag (off) |  |
| `--no-templates` | flag (off) | skip per-cluster median templates in the .clusters.npz |
| `--amp-algo` (`--amp-method`) | `pc1` | per-channel amplitude profile for the position inverse: pc1=rank-1 denoised template (default, sharpest footprint + most precise), wave=median-waveform ptp, ptp=median per-spike ptp (legacy; ~4-sigma noise floor on far channels flattens the footprint). — choices: `pc1`, `wave`, `ptp` |
| `--amp-basis` | `auto` | amplitude basis the gate-facing positions use: 'pca'=read .pca.standard.<elec> (PC1 score per channel = the .fet amplitude); 'fit'=group basis from .spk; 'auto'=pca if present else fit; 'none'=per-cluster SVD — choices: `auto`, `pca`, `fit`, `none` |
| `--no-amp-basis` | flag (off) | alias for --amp-basis none (per-cluster SVD) |
| `--nboot` | `0` | bootstrap draws for the depth/distance percentile CIs (z_lo/z_hi/y_lo/y_hi). This loop is ~5x the rest of the cost (the dominant runtime); positions (x0,y0,z0,A) and the energy-tercile depth-shift do NOT use it. Use --nboot 0 for identical positions ~5x faster (analytic sig_y is still written; the percentile CIs become NaN). |
| `--probe` | — | probe file(s) for geometry (else from chunk xy via YAML) |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-intrachunk`

Collapse over-split fragments within each chunk into units (stderiv cosine + offset + depth).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--cpos-method` | `stderiv` |  |
| `--cpos-stage` | — | positions (.cpos) stage tag; default follows --clu-stage, else 'refine' |
| `--clu-method` | — |  |
| `--clu-stage` | — |  |
| `--chunk-minutes` (`--chunk-min`) | `12.0` |  |
| `--gate` | `==SUPPRESS==` | shape gate: cosine\|mmd\|kcov\|cfiber\|band (default band: energy-scaled median+/-sigma overlap) — choices: `cosine`, `mmd`, `kcov`, `cfiber`, `band` |
| `--cos-thr` | `==SUPPRESS==` | cosine recall prefilter |
| `--off-thr` | `==SUPPRESS==` | inter-channel offset RMS gate (samples) |
| `--depth-gate` | `==SUPPRESS==` | depth gate (um) |
| `--amp-gate` | `==SUPPRESS==` | absolute log-amplitude (energy) gate, natural log; ln(3)=1.1 -> 3x (0=off) |
| `--refrac-ceiling` | `==SUPPRESS==` | reject merge if combined 2ms-ISI violation > this percent (empty=off) |
| `--pre-merge-cos` | `==SUPPRESS==` | pre-collapse obvious mutual-NN pairs at cosine>=this (0=off) |
| `--iter` | `==SUPPRESS==` | iterate group->re-estimate->regroup this many passes (1=single pass); >1 keeps the tight gate but re-merges DENOISED units across passes, consolidating over-split fragments a single pass leaves. Early-converges when a pass merges nothing (g5: 5 -> ~1124). Left at 1 in production; the exp config opts in (FK_INTRA_ITER). |
| `--linkage` | `==SUPPRESS==` | complete\|dynamic\|ms — choices: `complete`, `dynamic`, `ms` |
| `--align-lag` | `==SUPPRESS==` | merge-time best-lag half-window, NATIVE samples (0=off) |
| `--align-upsample` | `==SUPPRESS==` | cubic-spline upsampling factor for the align-lag search |
| `--cfiber-q` | `==SUPPRESS==` | cfiber self-calibration quantile |
| `--cfiber-null` | `==SUPPRESS==` | cfiber split-half null basis: order\|energy — choices: `order`, `energy` |
| `--band-thr` | `==SUPPRESS==` | gate='band': min energy-scaled median+/-sigma band-overlap IoU to merge (empty -> 0.5) |
| `--cfiber-thr-floor` | `==SUPPRESS==` | absolute floor on the self-calibrated cfiber threshold (0=off) |
| `--sig-cap` | `==SUPPRESS==` | per-fragment spikes for the mean template (empty = no cap) |
| `--warp-thr` | `==SUPPRESS==` | group-delay WARP coherence gate (Omlor-Giese): merge only if the cross-channel correlation of the two fragments' per-channel group-delay profiles >= this. Same-neuron warps cohere; co-located different cells anti-correlate. Group-delay is noisy at low spike count -> use LOW (~0.3). empty=off. |
| `--warp-resid-thr` | `==SUPPRESS==` | single-channel warp-incongruity SUB-GATE (layers on warp_thr): among already-coherent pairs (corr>=0.85), veto if any ONE centroid-range channel's group-delay residual (Theil-Sen line) > this many samples -- a strong-channel-masked different source. g5 knee ~1.0. empty=off. |
| `--off-thr-int` | `==SUPPRESS==` | DUAL gate: offset RMS threshold for suspected INTERNEURON pairs (narrow trough-to-peak). Fast cells have stable offsets (~0.23) so off_thr=1.0 is inert; tighten to ~0.5. Needs raw .spk for cell-typing. empty=off (use off_thr). |
| `--off-thr-pyr` | `==SUPPRESS==` | DUAL gate: offset RMS threshold for suspected PYRAMIDAL pairs (wide trough-to-peak); ~1.0. Set BOTH off_thr_int and off_thr_pyr to enable the dual gate; mixed pairs use the stricter. empty=off. |
| `--profile` | `default` | fallback profile for any intrachunk knob left unset in CLI/env/<session>.yaml: 'recommended' = the tuned pipeline baseline (cfiber gate, amp-gate 1.1, refrac 1.0, pre-merge 0.97, sig-cap 8000); 'default' = the conservative library baseline. — choices: `default`, `recommended` |
| `--off-n-ref` | — | SNR-adaptive offset gate: spike count at which --off-thr applies as-is; loosens ~1/sqrt(n) below it (recommend ~150). Omit for flat off_thr. |
| `--off-ceil` | `2.0` | cap on the adaptive offset tolerance (default 2.0; ~95%% same-neuron knee). |
| `--split-min-sil` | `0.12` | ms linkage: min silhouette to accept a split. |
| `--split-min-n` | `40` | ms linkage: min spikes per split sub-unit. |
| `--var-env-mult` | `3.0` | dynamic linkage: single-unit variance envelope = this * median fragment variance; blocks merges that push a unit's spread past it (permits the growth de-fragmentation needs, caps over-growth). |
| `--ccg-thr` | `1000000000.0` | dynamic linkage: max cross-CCG refractory ratio to admit a merge.  DEFAULT OFF (1e9): a simple refractory-dip requirement is WRONG-SIGNED for de-fragmentation -- same-neuron over-split fragments (time-shift dups, and amplitude-splits of bursting cells whose spikes attenuate through the burst) show a short-lag cross-CCG PEAK, not a dip, so a low threshold rejects true merges (g5: collapses 1243->1850).  Left as scaffolding; a correct temporal term needs duplicate-coincidence vs distinct-co-activity. |
| `--ccg-win` | `2.0` | dynamic linkage: cross-CCG refractory half-window (ms). |
| `--cfiber-thr` | — | gate='cfiber' shape-distance threshold; default None self-calibrates from per-fragment split-half nulls at --cfiber-q. |
| `--min-n` | `12` |  |
| `--boundary-minutes` | `3.0` | half-window (min) of straddling spikes for the overlap backbone anchor (--emit-units) |
| `--backbone-std-cos` | `0.75` | STANDARD median-template cosine a backbone pair must also clear (--emit-units). The stderiv boundary match is near-blind to depth/amplitude; this co-gate on the raw axis rejects the different-unit anchors it lets through. 0 disables. |
| `--backbone-warp` | — | optional Omlor-Giese group-delay coherence a backbone pair must also clear. OFF by default: on octrodes most units span <=2 channels so group-delay is degenerate (non-discriminative). Set only if energy spreads over >=3 channels. |
| `--out-stage` | — | output .clu stage (default: <clu-stage>_intrachunk) |
| `--emit-units` | flag (off) | also write a <...>.units.npz unit-signature table for fiber-link |
| `--no-provenance` | flag (on) | skip the .merge.tsv per-merge provenance sidecar (default: write it) |

### `fiber-link`

Link per-chunk fragments into tracked units (position fingerprint + A anchor + template co-gate).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--cpos-method` | `stderiv` |  |
| `--cpos-stage` | `refine` |  |
| `--clu-method` | — | source .clu method (default: mirror --cpos-method) |
| `--clu-stage` | — | source .clu stage (default: mirror --cpos-stage) |
| `--chunk-minutes` (`--chunk-min`) | `12.0` |  |
| `--cos-thr` | `0.975` |  |
| `--pos-thr` | `1.5` |  |
| `--max-shift` | — | cap (um) on the drift shift a single cross-chunk link may bridge: skip a chunk-pair whose estimated drift \|D(c+g)-D(c)\| exceeds this. Restricts linking to low-drift spans and leaves high-drift boundaries split into separate collinear time-bands (rejoin by ISI). Off by default (unbounded); set small to only auto-link the low-drift regions. |
| `--off-thr` | `1.0` | inter-channel offset RMS co-gate (samples); <=0 disables |
| `--amp-gate` | `0.0` | absolute log-amplitude gate (natural-log units): veto a cross-chunk link whose two fragments differ in log-energy by more than this (0 = off, default). A is treated as a drift-stable anchor, so a large energy jump between linked fragments is suspect; the cosine/cfiber co-gates are amplitude-invariant and cannot catch it, and logA is otherwise only one standardized term in the pos_thr fingerprint. This is an absolute, un-pooled cap. |
| `--align-lag` | `0` | sub-sample template re-registration half-window (native samples; 0 = off) applied before the cosine gate. The integer mutual_center leaves a fractional-sample residual that drops a true same-neuron cross-chunk cosine under threshold; re-registering recovers it (g5: +25% of admitted links). Mirrors fiber-intrachunk's --align-lag. |
| `--align-upsample` | `1` | cubic-spline upsampling factor for the --align-lag search (1 = native-rate). |
| `--primary-amp-frac` | `0.0` | restrict the cosine gate to the channels BOTH fragments treat as primary (peak-to-peak >= this fraction of the template's own max; 0 = off, full template). On an octrode the near-threshold channels carry mostly noise and decorrelate true same-neuron pairs; the intersection (g5 median 5 of 8 channels) lifts those links over threshold. ~0.3. |
| `--tan-thr` | — | energy-tangent (microfiber) co-gate: require the cosine of the two fragments' energy-direction tangents (high-energy minus low-energy template) >= this. A precision guard on the recall lifted by --primary-amp-frac; needs a per-fragment 'tangent' array in the cpos/units table (emitted by fiber-cpos/-intrachunk). None = off. ~0.5. |
| `--linkage` | `cogated` | merge method: 'cogated' (default; per-pair mutual-NN position + offset + cosine veto stack) or 'spectral' (global graph_link affinity + normalized-Laplacian eigengap partition -- transitivity-aware, robust to the per-pair gate miscalibration that leaves high-cosine blocks unmerged). EXPERIMENTAL. — choices: `cogated`, `spectral` |
| `--bundle` | `chunkexcl` | cross-chunk bundling: 'chunkexcl' (default; cosine-ordered union-find with the same-chunk-collision guard only) or 'varbound' (energy-seeded, variance-bounded agglomeration -- orders high-energy fragments first and refuses a merge that spreads a bundle's template variance past a single-neuron envelope, stopping the single-linkage chaining where a high-cosine-but-distinct cross edge welds two units that never share a chunk). — choices: `chunkexcl`, `varbound` |
| `--var-allow` | — | varbound: explicit template PC-variance boundary (default: self-calibrate from the high-energy backbone edges at link time). |
| `--var-scale` | `1.0` | varbound: multiply the variance boundary to dial the operating point (>1 merges more, <1 splits more); applies to both the self-calibrated and explicit boundary. |
| `--n-pc` | `12` | varbound: number of template principal components defining the variance space. |
| `--warp-amp-thr` | — | Omlor-Giese amplitude-profile floor (eq.10) -- full warp criterion with --warp-thr (more discriminative on co-located look-alikes than group-delay alone; ''/unset=off) |
| `--warp-resid-thr` | — | single-channel warp-incongruity ceiling (samples) -- sub-gate on warp-coherent pairs (catches a co-located source one channel's group delay betrays; unset=off) |
| `--warp-thr` | — | spatio-temporal WARP continuity co-gate (Omlor-Giese group delay): require the cross-channel correlation of two candidates' per-channel group-delay profiles >= this to link them. A neuron's warp morphs CONTINUOUSLY with drift (g5: adjacent-chunk change ~0.004) while different co-located cells anti-correlate (260x separation), so this vetoes false links that share gross shape. None (default) = off; ~0.9 is safe on the clean per-chunk unit templates the linker sees. |
| `--max-gap` | `2` | max chunk skip for the fingerprint pass (2 bridges single-chunk dropouts) |
| `--max-resid` | `0.08` |  |
| `--min-n` | `20` |  |
| `--min-snr` | `0.0` | gate linkable fragments on waveform SNR (needs snr in the cpos table; 0=off) |
| `--min-a` | `0.0` | absolute amplitude (A) floor on linkability, seeds included: drop noise-floor (A~1) fragments so a backbone pair cannot weld a noise unit into a huge-amplitude bundle (0=off). 'auto' picks the floor from the gap between the clip pile-up and the real-unit mass (per-session; leaves a clean session ungated) |
| `--amp-span` | `0.0` | varbound bundler: refuse a non-seed union whose bundle logA span would exceed this (natural-log units, e.g. ln(6)=1.79 for 6x; caps <=4x-per-link amplitude chaining; 0=off) |
| `--from-units` | — | link a fiber-intrachunk <...>.units.npz (per-chunk units) instead of raw cpos fragments |
| `--refine-trajectory` | flag (off) | post-pass: fit per-bundle depth + PCA-feature trajectories, resolve same-chunk-conflict merges, and attach units lying on a bundle's path |
| `--allow-chunk-clash` | flag (off) | disable chunk-exclusive bundling (default OFF: a bundle may not hold two same-chunk units; the exclusion vetoes provable chained over-merges). |
| `--traj-ext-min` | `0.0` | minutes an attach may extend beyond a bundle's member time span (0=interpolation only; ~chunk length allows extrapolation-based extension) |
| `--cfiber-thr` | — | cfiber shape co-gate: veto a candidate whose affine-invariant cfiber shape distance exceeds this (drift-invariant complement to the cosine gate). Fixed value. |
| `--cfiber-q` | — | enable the cfiber co-gate with the threshold self-calibrated at link time to this quantile of the overlap-backbone same-unit shape distances (e.g. 0.90; needs --from-units). |
| `--out-stage` | — | output .clu stage (default: <clu-stage>_linked) |
| `--gt-clu` | — | ground-truth .clu to score the clustering before vs after linking |
| `--gt-res` | — | .res for the ground truth (timestamp alignment if it covers a window) |
| `--overlap-refrac-ms` | `0.0` | DEFAULT OFF. >0 enables a power-aware refractory veto on the chunk-OVERLAP region: a candidate cross-chunk link is vetoed if, in the time window the two fragments share, their spikes coincide at chance level (two neurons) rather than showing a refractory dip (one neuron). Censors zero-lag duplicate detections; abstains (keeps the link) when underpowered, so it only ever removes well-supported wrong links. Needs sr + .res; abstains on sparse data (g5). Only the default 'cogated' linkage is gated. |
| `--overlap-refrac-thr` | `0.3` | coincidence ratio above which the overlap test reads 'two neurons' and vetoes (default 0.3) |
| `--overlap-censor-ms` | `0.4` | zero-lag censor band for the overlap test -- removes the same-spike duplicate detections that the overlapping chunks make of one neuron (default 0.4 ms) |
| `--overlap-min-exp` | `5.0` | min expected overlap-window coincidences to have power; below this the test abstains (default 5) |
| `--dr-candidates` | flag (off) | NEW: generate cross-chunk candidates by template-DR (PCA) nearest-neighbour instead of physical-position mutual-NN (g5: candidate cosine 0.966/69%% clean vs 0.888/23%%). The full co-gate stack still filters; this only improves the candidate set. |
| `--dr-thr` | — | DR-space NN distance cap (default none; co-gate filters) |
| `--dr-k` | `10` | template-DR dimensionality (default 10 ~ 96%% var on g5) |
| `--drift-algo` (`--drift-method`) | `accumulated` | NEW: 'global' solves drift by maximising template-anchor collinearity with a Laplacian-smoothness term (+ distance attenuation if the cpos table has 'dist'); 'accumulated' = legacy consecutive xcorr (compounds on sparse partitions) — choices: `accumulated`, `global` |
| `--no-provenance` | flag (on) | skip the .merge.tsv per-merge provenance sidecar (default: write it) |
| `--complete-edge` | flag (off) | NEW: rescue truncated / channel-shifted footprints in the cosine gate by spatial drift registration + own-structure off-probe completion (g5: lifts a truncated pair 0.83->0.91; taken as max with the plain cosine, so it never lowers a good match) |
| `--channel-pitch` | `20.0` | axial channel pitch (um) for --complete-edge |
| `--complete-field` | `inv_sq` | spatial field model for off-probe completion (1/r^2 default) — choices: `inv_sq`, `inv` |
| `--complete-edge-frac` | `0.5` | array-end amplitude fraction above which a footprint counts as truncated |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |

## Alternative / drift linkers

### `fiber-backbone-link`

Link fiber-session fragments across chunks on the invariant backbone (median+/-sigma CI-overlap) with the Omlor-Giese warp veto.

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | fragment .clu feature space (before the group) |
| `--clu-stage` (`--variant`) | `fiber_session` | fragment .clu stage tag (the fiber-session output) |
| `--in-clu` | — | explicit fragment .clu path (overrides --clu-method/--clu-stage) |
| `--spk-method` (`--spk-variant`) | `standard` | waveform axis for templates/warp (standard = curation axis) |
| `--channels` | — | pin backbone channels (global ids, e.g. 33,34); default = per-pair shared primary |
| `--out-stage` (`--out-tag`) | `backbone_linked` | output .clu stage tag (single token) |
| `--hierarchy` | flag (off) | also write the Klusters hierarchy siblings: `.clc` (per-spike child id) + `.clp` (child->parent map); an input `.clc` is carried through so repeated passes keep the original fiber-session fragments as the leaves. |
| `--gt-stage` (`--gt-clu`) | — | curated .clu to score purity+completeness against |
| `--gt-res` | — | reserved: .res for the GT (unused when GT shares the session res) |
| `--spk-cap` | `600` | spikes per fragment for the template |
| `--chunk-min` | — | chunk length (min); default from <session>.yaml or 12 |
| `--seed` | `0` |  |
| `--z` | `1.0` | FK_BBLINK_Z (default 1.0) |
| `--win` | `8` | FK_BBLINK_WIN (default 8) |
| `--slide` | `4` | FK_BBLINK_SLIDE (default 4) |
| `--iou-thr` | `0.5` | FK_BBLINK_IOU_THR (default 0.5) |
| `--floor` | `0.55` | FK_BBLINK_FLOOR (default 0.55) |
| `--prim-frac` | `0.3` | FK_BBLINK_PRIM_FRAC (default 0.3) |
| `--warp-thr` | `0.5` | FK_BBLINK_WARP_THR (default 0.5) |
| `--amp-thr` | `0.85` | FK_BBLINK_AMP_THR (default 0.85) |
| `--resid-thr` | `1.0` | FK_BBLINK_RESID_THR (default 1.0) |
| `--min-frag` | `40` | FK_BBLINK_MIN_FRAG (default 40) |
| `--max-gap` | `1` | FK_BBLINK_MAX_GAP (default 1) |
| `--complexity-scale` | `0.0` | FK_BBLINK_CX_SCALE (default 0.0) |
| `--min-snr-q` | `0.0` | FK_BBLINK_MIN_SNR_Q (default 0.0) |

### `fiber-xcorr-merge`

Confidence-ordered Klusters roll-shift cosine merge (realign after each merge).

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` |  |
| `--clu-stage` (`--variant`) | `backbone_linked` | input .clu stage tag (e.g. the fiber-backbone-link output) |
| `--in-clu` | — | explicit input .clu path (overrides --clu-method/--clu-stage) |
| `--spk-method` (`--spk-variant`) | `standard` | waveform axis for templates (curation axis) |
| `--out-stage` (`--out-tag`) | `xcorr_merged` | output .clu stage tag |
| `--refrac-censor-ms` | `0.0` | detection censor window (ms) |
| `--nsamp` | — | override; default from <session>.yaml |
| `--nchan` | — | override; default from <session>.yaml |
| `--ref-sample` | — | override; default = peak from <session>.yaml |
| `--gt-stage` (`--gt-clu`) | — | curated .clu tag/path to score purity+completeness |
| `--seed` | `0` |  |
| `--cos-thr` | `0.99` | FK_XCM_COS_THR (default 0.99) |
| `--shift` | `4` | FK_XCM_SHIFT (default 4) |
| `--refrac-ms` | `2.0` | FK_XCM_REFRAC_MS (default 2.0) |
| `--refrac-thr` | `0.3` | FK_XCM_REFRAC_THR (default 0.3) |
| `--refrac-min-exp` | `5.0` | FK_XCM_REFRAC_MIN_EXP (default 5.0) |
| `--min-n` | `40` | FK_XCM_MIN_N (default 40) |
| `--spk-cap` | `300` | FK_XCM_SPK_CAP (default 300) |
| `--complexity-scale` | `0.0` | FK_XCM_CX_SCALE (default 0.0) |
| `--band-thr` | `0.5` | FK_XCM_BAND_THR (default 0.5) |

### `fiber-relink`

Geometry-aware re-bundling/re-linking of a .fibers run (no re-run needed).

Positional: `fibers`

| flag | default | description |
|---|---|---|
| `--clu` | — | existing .clu to remap (gid+1; 0=noise) |
| `--out` | — | output .clu (default <clu>_relinked) |
| `--report` | — | per-unit drift report TSV |
| `--prof-thr` | `0.1` |  |
| `--tcorr-min` | `0.96` |  |
| `--prof-gate` | `0.18` |  |
| `--tdist-gate` | `0.055` |  |
| `--depth-gate` | `0.12` |  |
| `--q-half` | `300.0` |  |
| `--thr` | `0.14` |  |
| `--margin` | `0.7` |  |
| `--max-gap` | `2` |  |
| `--consec-guard` | `0.08` |  |
| `--e2e-guard` | `0.35` |  |
| `--refrac-ms` | `0.0` | DEFAULT OFF. >0 enables a curation-independent refractory cross-correlogram veto on every bundle/link: a merge whose two trains coincide at chance level on their temporal overlap (two neurons, no refractory dip) is blocked. Needs --res. Power-aware: ABSTAINS at low firing rates, only ever removes false linkages. |
| `--res` | — | per-spike .res (sample times) aligned to --clu; required for --refrac-ms |
| `--sr` | — | sample rate (Hz) for --refrac-ms (e.g. 32552) |
| `--refrac-thr` | `0.3` | coincidence ratio above which the overlap is 'two neurons' (default 0.3) |
| `--refrac-min-exp` | `5.0` | min expected coincidences for the test to be powered (default 5) |
| `--refrac-censor-ms` | `0.0` | censor window (ms) to drop duplicate detections of the same spike (default 0) |
### `fiber-refiber`

Re-fiber microfiber atoms (.clc) into fibers by raw-position + template-shape co-gating; write the .clu/.clc/.clp triple.

Positional: `base`, `group`

| flag | default | description |
|---|---|---|
| `--method` (`--variant`) | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 (default stderiv) |
| `--stage` (`--tag`) | `microfiber` | post-fiber stage tag of the input atom layer |
| `--out-stage` (`--out-tag`) | — | output tag (default: same as --tag, i.e. overwrite in place with .bak) |
| `--session` | — |  |
| `--probe` | — |  |
| `--channels` | — | comma-separated global channel ids |
| `--nsamp` | — |  |
| `--nchan` | — |  |
| `--dy-um` | `6.0` | max depth disagreement (um) |
| `--dlogA` | `0.25` | max log-amplitude disagreement |
| `--dz-um` | `8.0` | max distance disagreement (um) |
| `--min-cos` | `0.85` | min template cosine similarity |
| `--clique` | flag (off) | require an atom to gate to ALL members of a fiber (stricter; no shape chaining) |
| `--max-spikes` | `2000` | cap per-atom spikes for templates |
| `--no-renumber` | flag (off) |  |
| `--no-backup` | flag (off) |  |
| `--dry-run` | flag (off) |  |
### `fiber-microfiberize`

Lift a flat .clu fiber sort into the microfiber triple (.clc atom layer + .clp child->parent map) so it opens in the Klusters hierarchical / fiber-refiberize machinery.

Positional: `base`, `group`

| flag | default | description |
|---|---|---|
| `--method` (`--variant`) | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 (default stderiv) |
| `--stage` | `refine_linked` | pipeline stage of the input .clu (default refine_linked) |
| `--out-stage` | — | pipeline stage for the output triple (default: same as --stage) |
| `--atoms-stage` (`--atoms`) | — | stage of a FINER per-spike .clu to use as the atom (.clc) layer; omit for the identity lift (each fiber = one microfiber) |
| `--write-clu` | flag (off) | also (re-)write the derived .clu; by default only .clc/.clp are written (the input .clu is left untouched unless --out-stage differs) |
| `--no-renumber` | flag (off) | keep original fiber ids (leave gaps) instead of compacting to consecutive |
| `--no-backup` | flag (off) | do not write .bak copies of overwritten files |
| `--dry-run` | flag (off) | report only; write nothing |
### `fiber-refiberize`

Edit the fiber<-microfiber hierarchy and regenerate the aligned .clu/.clc/.clp from the child->parent map.

Positional: `base`, `group`

| flag | default | description |
|---|---|---|
| `--method` (`--variant`) | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 (default stderiv) |
| `--stage` (`--tag`) | `microfiber` | post-fiber stage tag of the input triple (default microfiber) |
| `--out-stage` (`--out-tag`) | — | post-fiber stage tag for the output triple (default: overwrite --stage) |
| `--ops` | — | edit-script file to apply before refiberizing |
| `--no-renumber` | flag (off) | keep original fiber ids (leave gaps) |
| `--no-backup` | flag (off) | do not write .bak copies |
| `--dry-run` | flag (off) | report only; write nothing |
### `fiber-defrag`

De-fragment an over-clustered sort: reunite a neuron's drift/amplitude fragments by mutual-nearest-neighbour template merging, gated by cosine AND the time-warp (spike width) so distinct same-shape cells are held apart.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | feature space before the group (default stderiv) |
| `--clu-stage` (`--variant`) | `refine` | fiber stage after the group (default refine; '' = none) |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--variant) |
| `--cos-thr` | `0.92` | template cosine merge candidate (default 0.92) |
| `--warp-max` | `0.06` | \|alpha-1\| width gate; above this keep separate (default 0.06) |
| `--amp-gate` | `1.4` | \|delta log\|F1\|\| energy gate (default 1.4, wide) |
| `--min-cluster` | `40` | fragments smaller than this are left untouched |
| `--var-budget` | — | path to a fiber-calibrate .npz; adds a curated PC-variance stopping gate (merge rejected once a merged cluster would be more spread than a real unit) |
| `--var-scale` | `1.0` | multiply the loaded variance allowance (dial the operating point: >1 looser/more merging, <1 tighter; floor is conservative, ~1.5-2x reaches the baseline) |
| `--out-stage` (`--out-tag`) | — | post-fiber stage tag for the merged result (default 'defrag', single token) |
| `--ccg-refrac-ms` | `0.0` | refractory cross-correlogram veto window (ms); 0 disables. ~1.5 to enable. Power-aware: abstains where firing rates are too low to show a dip (e.g. g5). |
| `--ccg-thr` | `0.3` | cross-CCG ratio above which a powered pair is vetoed |
| `--ccg-min-exp` | `5.0` | min expected coincidences for the veto to act |
| `--ccg-censor-ms` | `0.3` | duplicate censor band for the cross-CCG (ms) |
| `--gt-clu` | — | ground-truth .clu to score before/after the merge against |
| `--gt-res` | — | .res for the ground truth (timestamp alignment if it covers a window) |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |

## Positions & geometry

### `fiber-cpos`

Write a per-spike cluster-position sidecar (.cpos) by localizing each cluster's median RAW template (monopole+dipole).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--spk` | — | path to a STANDARD/raw .spk (preferred over .fil); never the stderiv .spkD |
| `--spk-method` | `standard` | method of the raw .spk to resolve: <base>.spk.<spk-method>.<elec> |
| `--fil` | — | path to raw .fil (fallback if no standard .spk) |
| `--fil-offset` | `0` | first absolute sample of the .fil (0 for full recording) |
| `--clu-method` | `stderiv` | source-clu feature space BEFORE the group (standard\|stderiv\|...) |
| `--clu-stage` | `refine` | source-clu fiber STAGE AFTER the group: read <base>.clu.<clu-method>.<elec>.<clu-stage> |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--clu-stage) |
| `--out-method` | — | cpos method BEFORE the group (default: mirror --clu-method) |
| `--out-stage` | — | cpos fiber STAGE AFTER the group (default: mirror --clu-stage) |
| `--min-spikes` | `15` |  |
| `--no-dipole` | flag (off) |  |
| `--no-templates` | flag (off) | skip per-cluster median templates in the .clusters.npz |
| `--amp-algo` (`--amp-method`) | `pc1` | per-channel amplitude profile for the position inverse: pc1=rank-1 denoised template (default, sharpest footprint + most precise), wave=median-waveform ptp, ptp=median per-spike ptp (legacy; ~4-sigma noise floor on far channels flattens the footprint). — choices: `pc1`, `wave`, `ptp` |
| `--amp-basis` | `auto` | amplitude basis the gate-facing positions use: 'pca'=read .pca.standard.<elec> (PC1 score per channel = the .fet amplitude); 'fit'=group basis from .spk; 'auto'=pca if present else fit; 'none'=per-cluster SVD — choices: `auto`, `pca`, `fit`, `none` |
| `--no-amp-basis` | flag (off) | alias for --amp-basis none (per-cluster SVD) |
| `--nboot` | `0` | bootstrap draws for the depth/distance percentile CIs (z_lo/z_hi/y_lo/y_hi). This loop is ~5x the rest of the cost (the dominant runtime); positions (x0,y0,z0,A) and the energy-tercile depth-shift do NOT use it. Use --nboot 0 for identical positions ~5x faster (analytic sig_y is still written; the percentile CIs become NaN). |
| `--probe` | — | probe file(s) for geometry (else from chunk xy via YAML) |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-localize`

Localize fibers (distance + depth + orientation) from raw waveform spread.

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--nchan` | — | override: channels in this group (default from YAML) |
| `--probe` | — | NeuroSuite .probe YAML(s), in global-channel order (default: the probe named in <session>.yaml) |
| `--channels` | — | comma-separated global channel ids of this group (else read <session>.yaml) |
| `--session` | — | session for channel/probe lookup if --channels/--probe omitted |
| `--clu` | — | cluster file (pass the re-linked .clu) |
| `--no-dipole` | flag (off) |  |
| `--min-n` | `50` |  |
| `--nboot` | `0` | spike bootstrap draws for depth/distance CIs; 0 (default) uses the analytic Gaussian sigma (matches the bootstrap on isolated clusters, ~Nx cheaper). |
| `--amp-algo` (`--amp-method`) | `pc1` | per-channel amplitude profile: pc1=rank-1 denoised template (default, sharpest + most precise); wave=median-waveform ptp; ptp=median per-spike ptp (legacy, carries a ~4-sigma noise floor on far channels). — choices: `pc1`, `wave`, `ptp` |
| `--max-resid` | `0.1` |  |
| `--amp-basis` | `auto` | amplitude denoising basis: 'pca' = read .pca.standard.<elec> eigenvectors (PC1 score per channel = the .fet amplitude); 'fit' = fit one group-wide basis from .spk; 'auto' = pca if present else fit; 'none' = per-cluster SVD — choices: `auto`, `pca`, `fit`, `none` |
| `--no-amp-basis` | flag (off) | alias for --amp-basis none (per-cluster SVD; unstable at low n) |
| `--out` | — |  |
### `fiber-position`

Per-spike drift-independent position along the fiber manifold (from a .fibers file).

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--fibers` | — | <base>.fibers.<method>.<elec> (the estimated manifold) |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--nchan` | — | override: channels in this group (default from YAML) |
| `--ntotal` | — | override: total channels in the .fil (default from YAML) |
| `--channels` | — | comma-separated global channel ids (else from --session/.fibers) |
| `--session` | — | session .yaml for channel/sr lookup |
| `--sr` | — |  |
| `--clu` | — | cluster file (the re-linked .clu) |
| `--no-relink` | flag (off) | use raw .fibers gid units instead of re-linked tracks |
| `--min-nspk` | `60` | min spikes for a chunk-curve to enter a manifold |
| `--n-u` | `64` | manifold arc-length resolution |
| `--chunk-min` | `20.0` |  |
| `--min-n` | `20` |  |
### `fiber-drift`

Track probe drift over time from the fiber files of a probe's groups.

Positional: `fibers`

| flag | default | description |
|---|---|---|
| `--no-relink` | flag (off) | use raw .fibers gid instead of re-linked units |
| `--min-nspk` | `60` |  |
| `--min-span` | `3` |  |
| `--pitch` | `20.0` | site pitch in µm (depth is in channel units) |
| `--out` | — | drift table TSV |
| `--npy` | — | save the drift curve D(c) in µm as .npy |
### `fiber-cfiber`

Complex-fiber drift/identity test: per curated unit, bin by time and report whether the affine-invariant loop SHAPE stays flat (identity) while the AFFINE rotation/scale drift.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | source .clu feature space before the group |
| `--clu-stage` | `refine_linked` | source .clu stage after the group |
| `--in-clu` | — | explicit curated .clu path (overrides method/stage) |
| `--spk-method` | `standard` | raw .spk variant to read (.spk.<m>.N) |
| `--bins` | `8` | time bins across the session |
| `--min-spikes` | `200` |  |
| `--min-bins` | `5` | require a unit populate >= this many time bins |
| `--win-pre` | `10` |  |
| `--win-post` | `12` |  |
| `--no-align` | flag (on) | skip per-unit realign before building the loop |
| `--modes` | `2,3,4,-1,-2,-3` | comma list of Fourier modes for the shape |
| `--out` | — | write per-unit metrics .tsv (default <base>.cfiber.<elec>.tsv) |
| `--fig` | — | write a shape_flatness vs rotation_drift summary figure |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-chan-svd`

Per-channel SVD/PCA of cluster mean templates: which channels are invariant vs vary across the clusters (a merge/curation aid).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--in-clu` | — | sort to analyse (default canonical .clu) |
| `--clu-method` (`--variant`) | — | method the staged .clu stems from (e.g. stderiv, stderiv_C5) instead of --in-clu |
| `--clu-stage` (`--stage`) | — | post-fiber stage tag of the staged .clu (e.g. refine); pair with --clu-method |
| `--clusters` | — | comma list of .clu ids to include (default: all ids >= 2 with >= --min-n spikes) |
| `--spk` | `standard` | waveform space (default standard = the RAW waveform curation sees) — choices: `standard`, `stderiv` |
| `--n-comp` | `3` | components plotted per channel (default 3) |
| `--min-n` | `30` | skip clusters below this many spikes (noisy template) |
| `--sig-cap` | `2000` | spikes sampled per cluster for the template |
| `--within` | — | examine ONE cluster: split its spikes into --bins time-ordered sub-templates and run the per-channel SVD across them (single-unit view, not across clusters) |
| `--bins` | `12` | --within: time-ordered sub-templates (default 12) |
| `--local-frac` | — | --within: use only a contiguous central fraction of the cluster's timespan (drift-minimal -> the residual per-channel variation is physiological, not drift) |
| `--normalize` | flag (off) | p2p-normalize each template first -> SVD sees SHAPE variation, amplitude drift removed |
| `--out` | — | output PNG path or directory (default next to the session) |
| `--tsv` | — | also write the per-channel metric table here |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-pca`

neurosuite-3 .pca/.pcaD basis: fit a per-channel PCA basis from .spk+.clu and write the binary, or inspect one.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--info` | — | print the header of an existing .pcaD and exit |
| `--clu-method` | `stderiv` |  |
| `--clu-stage` | `refine` |  |
| `--data2use` | — | PCA window length (default: nSamples) |
| `--rec-shift` | — | window start sample (default: peak - data2use//2) |
| `--ncomp` | `3` |  |
| `--centered` | flag (off) | store centered flag (projection subtracts the mean) |
| `--stderiv` | flag (off) | fit on .spkD (stderiv) instead of .spk |
| `--out` | — | output .pcaD path (default <base>.pca<D>.<group>) |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-branch`

Flag units whose spikes branch off the single fiber d(r) (a second, energy-independent, depth-coherent waveform class).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu` | — | unit-defining .clu (e.g. the .intrachunk or .linked clu); default resolves the canonical sort |
| `--clu-method` | `stderiv` |  |
| `--clu-stage` | `refine_linked` |  |
| `--min-n` | `400` | skip units with fewer spikes (branch test needs samples) |
| `--pitch` | `20.0` | probe site pitch (um) for the depth-coherence gate |
| `--depth-um` | `8.0` |  |
| `--sep-min` | `2.2` |  |
| `--ecorr-max` | `0.25` |  |
| `--dbic-min` | `20.0` |  |
| `--out` | — | write a per-unit branch report .npz |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |

## Curation, QC & scoring

### `fiber-refit`

Refit the fiber model (per-unit, per-chunk templates / positions / signatures) to a MANUALLY CURATED .clu, taking the curator's grouping as final (no re-merge/split).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | feature space of the curated clu |
| `--clu-stage` (`--variant`) | `curated` | stage tag of the curated .clu to refit (e.g. 'curated') |
| `--in-clu` | — | explicit curated .clu path |
| `--cpos-method` | — | cpos method for positions (default: --clu-method) |
| `--cpos-stage` | — | cpos stage for positions (default: --variant) |
| `--relocalize` | flag (off) | re-localize the curated units here from raw .spk/.fil instead of reading a cpos table |
| `--spk-method` | `standard` | raw .spk method for --relocalize |
| `--gate` | `cfiber` | shape descriptor to attach to each signature (default cfiber) — choices: `cfiber`, `wave`, `none` |
| `--chunk-minutes` (`--chunk-min`) | `12.0` |  |
| `--min-n` | (from config) | min spikes for a per-chunk signature |
| `--out-stage` | — | stage tag for the refit units (default '<variant>_refit') |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-reject`

Reassign per-cluster outlier spikes to a better-fitting cluster or to noise.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | variant of the input .clu (before the group) |
| `--clu-stage` | `refine` | stage of the input .clu (after the group) |
| `--feat-method` | `standard` | .pca/.spk variant for the feature space (standard=amplitude, stderiv=clustering) |
| `--spk` | — | explicit .spk path (else <base>.spk.<feat-method>.<group>) |
| `--out-stage` | — | output .clu stage (default: <clu-stage>_reject) |
| `--noise-id` | `1` | cluster id outliers fall to when no cluster fits |
| `--chi2-p` | `0.9999` | per-spike outlier gate (chi^2 quantile) |
| `--shrink` | `0.1` | covariance shrinkage toward diagonal (0=full cov, 1=diagonal) for conditioning |
| `--support-fraction` | `0.75` | MCD clean-core fraction (1 - max contamination the robust cov tolerates) |
| `--no-robust` | flag (off) | use a plain (non-robust) covariance instead of MCD (faster; masks contaminants) |
| `--no-reassign` | flag (off) | send every outlier to noise (skip cross-cluster reassignment) |
| `--min-n` | `50` | min spikes for a cluster to get a robust model |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-stats`

Extract per-(chunk,cluster) fiber statistics from an existing sort (no re-clustering).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | feature space BEFORE the group (standard\|stderiv\|...); default stderiv |
| `--clu-stage` (`--variant`) | `refine` | fiber STAGE AFTER the group: read <base>.clu.<clu-method>.<elec>.<variant> (default: refine; '' = no stage) |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--variant) |
| `--chunk-min` | `12.0` |  |
| `--overlap-min` | `4.0` |  |
| `--whole-session` | flag (off) | one row per cluster over the whole session (single whitener) |
| `--min-cluster` | `20` | skip clusters smaller than this |
| `--n-grid` | `40` |  |
| `--fibers-stage` (`--method`) | — | post-fiber stage tag of the written .fibers file (<base>.fibers.<clu-method>.<group>.<stage>). Default: mirror --clu-stage, since the stats describe that clu |
| `--out` | — |  |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-qc`

Per-group QC report (rate, ISI violation, SNR, amplitude, presence) rendered as an interactive HoloViz/Bokeh HTML, with a metrics CSV.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` |  |
| `--clu-stage` (`--variant`) | `refine` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path |
| `--refrac-ms` | `1.5` |  |
| `--censor-ms` | `0.3` |  |
| `--presence-bins` | `120` |  |
| `--min-spikes` | `20` |  |
| `--isi-thr` | `0.01` | ISI-violation fraction to flag (default 0.01) |
| `--snr-thr` | `4.0` | SNR below this is flagged lowSNR (default 4) |
| `--presence-thr` | `0.5` | presence below this is flagged intermittent (lower for sparse data; default 0.5) |
| `--gt-clu` | — | ground-truth .clu to score against (fiber-score) |
| `--gt-res` | — | .res for the ground truth (timestamp alignment) |
| `--out` | — | output HTML path (default <base>.qc.<elec>.html) |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-score`

Score a candidate .clu against a ground-truth .clu (ARI, V-measure, pairwise precision/recall, per-unit and split/merge diagnostics).

| flag | default | description |
|---|---|---|
| `--candidate` (`--cand`) | — | candidate .clu file |
| `--gt` (`--ground-truth`) | — | ground-truth .clu file |
| `--candidate-res` | — | .res for the candidate (for timestamp alignment) |
| `--gt-res` | — | .res for the ground truth (for timestamp alignment) |
| `--gt-noise` | `0,1` | GT cluster ids to exclude from scope (default 0,1) |
| `--cand-noise` | `""` | candidate cluster ids to drop from scope (default none) |
| `--split-tol` | `0.2` | largest-piece shortfall to call a unit split |
| `--merge-frac` | `0.2` | min share for a GT unit to count in a merge |
| `--top` | `8` | how many worst units / merges to list |
| `--json` | — | also write the full summary as JSON |
### `fiber-contam`

Contamination QC for an existing sort: flag two-cell mixtures that ISI checks miss, via the per-channel derivative-distribution bimodality of the stderiv spikes (amplitude/burst axis rejected).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | feature space before the group (default stderiv) |
| `--clu-stage` (`--variant`) | `refine` | fiber stage after the group: read <base>.clu.<clu-method>.<elec>.<variant> (default refine; '' = no stage) |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--variant) |
| `--min-cluster` | `60` | skip clusters smaller than this |
| `--n-pc` | `4` | top within-cluster SVD components scanned |
| `--n-null` | `16` | single-mode surrogate draws for the null |
| `--burst-cos` | `0.92` | sub-template shape cosine at/above this is one bursting cell, not flagged |
| `--seed` | `0` |  |
| `--out` | — | write the ranked table to this TSV path |
| `--split` | flag (off) | write a new staged .clu (tag '<variant>.csplit') with each flagged two-cell cluster recursively QC-split into sub-ids on its bimodal axis |
| `--max-split` | `6` | max sub-clusters a single flagged cluster may be split into (default 6) |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-calibrate`

Learn the variance/energy envelope of a curated group's single units and write an .npz budget for `fiber-defrag --var-budget`.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | feature space before the group (default stderiv) |
| `--clu-stage` (`--variant`) | `""` | curated fiber stage tag after the group (default none) |
| `--in-clu` | — | explicit curated .clu path (overrides --clu-method/--variant) |
| `--n-pc` | `10` | number of PCs for the feature space (default 10) |
| `--min-cluster` | `60` | ignore curated units below this many spikes |
| `--floor` | flag (off) | set allowance to the confusable-pair merged-variance floor (tighter, recommended) |
| `--cos-thr` | `0.85` | candidate cosine for the confusable-pair gate |
| `--warp-max` | `0.06` | width gate for the confusable-pair gate |
| `--out` | — | output .npz (default '<base>.calib.<group>.npz') |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-validate-merges`

Full-session evidence for proposed same-neuron merges; reads <session>.yaml for sr.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--cand` | — | candidates tsv (default <base>.merge_candidates.<group>.tsv) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-ccg`

Refractory QC for a group's clustering: per-cluster ISI-violation fraction, and the cluster pairs whose refractory cross-correlogram shows a dip (merge-consistent).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` |  |
| `--clu-stage` (`--variant`) | `refine` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path |
| `--refrac-ms` | `1.5` | refractory window (ms, default 1.5) |
| `--censor-ms` | `0.3` | duplicate censor band (ms, default 0.3) |
| `--thr` | `0.3` | ratio at/below which a pair shows a dip |
| `--min-exp` | `5.0` | min expected coincidences to have power |
| `--min-cluster` | `40` | ignore clusters smaller than this |
| `--top` | `15` | how many merge-consistent pairs to list |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |

## Pipeline runner, plans & tooling

### `fiber-plan-edit`

*(parameters not introspectable: SystemExit; run `fiber-plan-edit -h`)*

Positional: `plans`

| flag | default | description |
|---|---|---|
| `--strict` | flag (off) | treat warnings as failures (CI gate) |
| `-q` (`--quiet`) | flag (off) | print only failures |
### `fiber-kit-init`

Copy the template fiber-kit.yaml (fiber-pipeline tuning config) into the current directory.

| flag | default | description |
|---|---|---|
| `-o` (`--output`) | `fiber-kit.yaml` | destination path (default: ./fiber-kit.yaml) |
| `-f` (`--force`) | flag (off) | overwrite the destination if it already exists |
| `--exp` | flag (off) | copy the EXPERIMENTAL template (fiber-kit-exp.yaml: the newest gates turned on) to ./fiber-kit.yaml instead of the production default |
### `fiber-raw-vs-stderiv`

raw .fil vs stderiv discrimination of the original fibers; reads <session>.yaml.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--chunk-min-start` | `0.0` |  |
| `--chunk-min` | `10.0` |  |
| `--min-spikes` | `60` |  |
| `--min-group` | `200` |  |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |

## Visualization

### `fiber-view`

Visualise fibers: template montages, a 3-D manifold of local fiber curves, and ISI/geometry panels.

Positional: `bundles`, `session`, `group`

| flag | default | description |
|---|---|---|
| `--fibers` | `all` | comma list of bundle ids, 'top:N', or 'all' |
| `-o` (`--out`) | — | output .gif/.mp4 (default <bundles>.tour.gif) |
| `--ncomp` | `6` |  |
| `--keypoints` | `4` |  |
| `--steps` | `24` | interpolation frames per leg |
| `--fps` | `20` |  |
| `--spin` | `0.5` |  |
| `--seed` | `0` |  |
| `--in-clu` | — | sort to view (default canonical .clu) |
| `--fibers` | `top:6` | comma list of .clu cluster ids, 'top:N', or 'all' (default top:6) |
| `--mode` | `all` | choices: `templates`, `manifold`, `stats`, `all` |
| `--npos` | `80` | positions sampled along the fiber |
| `--geom` | — | a .geom/.geomchunk npz for the stats geometry track |
| `--no-dedup` | flag (off) |  |
| `--out` | — | output path or directory (default next to the session) |
| `--channels-override` | — |  |
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | (from config) | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--peak` | `16` | peak sample index within the window |
| `--probe` | — | probe file(s) for geometry |
### `fiber-view-gui`

fiber-view-gui: a standalone, rotatable bundle viewer.

*(parameters not introspectable: SystemExit; run `fiber-view-gui -h`)*



### `fiber-stochastic`  (diagnostic — not part of the sort)

Ensemble / consensus fibering by resampling.  Instead of clustering each chunk once, it draws
`--stochastic-draws` random subsamples of the chunk (fraction `--stochastic-frac`), fibers each draw with
the **ordinary per-chunk clusterer** (same `.fil` whitener + `cluster_chunk_fine` + `fiber_geom` as the
production path — it reuses that worker directly), and finds the fibers that **recur across draws** (the
consensus set).  With `--stochastic-peel-rounds > 0` it freezes the recurring fibers, removes their
spikes, and re-runs the ensemble on the residual to expose the next tier.

It writes **`<base>.fiberens.<elec>.npz`** and changes no sort output.  The file is a row-store of *every*
fiber instance from *every* draw: geometry columns match the production `.fibers.npz` (`template`, `grid`,
`dir`, `depth`, `width_ms`, `radius`, adaptation fingerprint, drift slopes, isolation) plus diagnostic
columns — `draw`, `frac`, `peel_round`, `consensus_gid`, `recovery_freq`, `match_corr`, `match_corr2`
(nearest-rival / merge-proneness) — and a per-round `peellog_*` trajectory.

| flag | default | meaning |
| --- | --- | --- |
| `--stochastic` | off | run this diagnostic instead of the normal single pass. |
| `--stochastic-draws` | `20` | resampled draws per chunk (per peel round). |
| `--stochastic-frac` | `0.8` | subsample fraction per draw. |
| `--stochastic-match-corr` | `0.95` | template corr above which two draw-fibers are the same consensus fiber. |
| `--stochastic-stable-freq` | `0.6` | recurrence fraction to call a consensus fiber stable. |
| `--stochastic-peel-rounds` | `0` | 0 = single pass; N = freeze stable fibers, peel, re-run on residual. |
| `--stochastic-link` | `average` | how draw-fibers are grouped into consensus fibers: `average` (default), `complete`, `single`. `single` chains anticorrelated sub-modes through intermediate shapes; agglomerative modes do not. |
| `--stochastic-jobs` | `1` | parallelise the independent resampling draws across N worker processes; deterministic regardless of N. |
| `--stochastic-write-clu` | flag (off) | also write a Klusters `.clu`/`.clc`/`.clp` triplet from the per-spike majority vote (`.clu` = consensus fibers, `.clc` = sub-modes/branches) for inspection in Klusters. |
| `--stochastic-clu-tag` | `fiber_stochastic` | tag for the written triplet. |
| `--stochastic-chunks` | all | restrict to these chunk indices for a quick look. |
| `--stochastic-seed` | `0` | RNG seed for the draws. |

Positional: `bundles`

| flag | default | description |
|---|---|---|
