<picture>
  <source media="(max-width: 600px)" srcset="images/doc-stages-banner-narrow.png">
  <img src="images/doc-stages-banner.png" width="100%" alt="">
</picture>

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

**The clustering a stage reads and writes, in three short names** (0.71.16).  A plan says `in:`
and `out:`; every stage that has them takes the same on the command line:

| short | stands for | absent on |
|---|---|---|
| `--in` | the stage of the clustering read: `--clu-stage` (`--in-stage` in fiber-session and fiber-stochastic, where `--clu-stage` names the output) | stages that read no clustering |
| `--out` | the stage of the clustering written: `--out-stage` or `--out-tag` | fiber-session, fiber-stochastic and fiber-rdr, whose `--out` names a file |
| `--method` | the method token of that clustering: `--clu-method` | — |

So `fiber-polish <base> 8 --in fc_session --out fc_polish --method stderiv_C5` and
`fiber-xcorr-merge <base> 8 --in fc_polish --out fc_xcm --method stderiv_C5` read alike.  The
longer names keep working; `fiber_kit/stages.py` (`IO`) holds the table.

---

## Core pipeline (canonical order)

### `fiber-session`

Cluster a session group into fibers.

**End-of-run realign.**  After the sort, `--realign` (`FK_SESSION_REALIGN`, default `auto`) aligns
every spike to its fiber's template and commits it: the shifted times to the shared `.res` (offsets to
`<base>.offsets.<group>.npy`), and every waveform variant's `.spk` rebuilt at those times with a fresh
`.fet` from its `.pca` -- fiber-realign's own `--reextract --refeaturize --variants all` on the `.clu`
just written, byte for byte.  `auto` re-extracts from the `.fil` when it exists and rolls the `.spk`
otherwise; `off` leaves `.res`/`.spk`/`.fet` as extracted.  See docs/pipeline.md.

**Re-sorting around kept units.**  `--in-stage <stage> --exclude-units <ids>` takes the clu
`<base>.clu.<in-method>.<elec>.<stage>` (in-method defaults to `--method`) and keeps the listed
clusters' spikes out of every chunk, so they shape no cluster.  With `--exclude-as carry` (default)
each listed unit is written back as one cluster and one atom of its own, after the new fibers; the log
prints `input -> output` ids.  `--exclude-as noise` makes them cluster 0.  Ids 0 and 1 (artefact,
noise) always go to 0; ids absent from the input are reported and ignored.  The input clu must have
one label per spike of the `.res` fiber-session reads (a stage with its own `.res`, e.g. decollided, is
refused), and the run refuses to write over it.  In a plan, a fiber-session step's `in:` is passed as
`--in-stage` and the list rides `params:`:

    - {stage: fiber-session, in: curated, out: resort, params: {exclude-units: "3,7,12-15"}}

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from YAML) |
| `--chunk-min` (`--chunk-minutes`) | `12.0` |  |
| `--overlap-min` | `4.0` |  |
| `--min-group` | `200` | COARSE min spikes/fiber (for linking) |
| `--fine-algo` (`--fine-method`) | `rkk` | per-fiber fine clusterer: rkk (default, masked-EM KlustaKwik with per-cluster realign + BIC model selection, robust to unequal covariances), gmm (Gaussian-mixture split), fiber (fiber-kit's native cluster_chunk), none (no fine split -- keep the coarse cluster whole) — choices: `gmm`, `rkk`, `fiber`, `none` |
| `--rkk-dims` | `6` | FK_SESSION_RKK_DIMS. SVD components of the fine-split feature space handed to the rkk CEM |
| `--rkk-max` | `50` |  |
| `--rkk-realign` | flag (off) | interleave rkk (CEM) with per-cluster realignment (per-step; default on) |
| `--no-rkk-realign` | flag (on) | legacy: one parent realign + fixed features for the rkk split |
| `--rkk-realign-iters` | `2` | FK_SESSION_RKK_REALIGN_ITERS. cluster<->realign passes in the rkk realign loop (each pass re-aligns every cluster to its own median and re-runs the CEM; stops early when the cluster count no longer changes) |
| `--rkk-delete` | flag (off) | rkk (CEM) culls sub-min-group sub-clusters during the per-fiber fine split (default on) |
| `--no-rkk-delete` | flag (on) | keep small non-singular rkk sub-clusters -- session should OVER-cluster, leaving the cull to refine; use to stop session shedding fragments into the residual/artifact bin |
| `--refit-iters` | `0` | iterate {realign each fiber's spikes to their own median -> refit template -> re-apply the inclusion gate} this many extra times. Recovers true members rejected for a bad sub-sample offset and tightens the template; re-judges from the full member set each pass. 0 = single pass (original). Marginal on clean fibers; helps contamination-heavy ones. |
| `--reseed-residual-thr` | `0.0` | residual-gated reseed: after the fine split, any fiber whose median-subtracted residual/signal exceeds this is re-split (a welded pair of co-located different-shape cells scores high; ~0.35 separates welded from clean on g5). 0 = off. The re-split is kept only if it lowers the residual by --reseed-min-reduction. |
| `--reseed-min-reduction` | `0.05` | keep a residual-gated reseed only if it lowers the median-subtracted residual/signal by at least this much (a real sub-unit does; noise does not) |
| `--seed-density` | `0.0` | density-preferential coarse seeding: draw ridge seeds toward concentrated modes of the waveform space (p proportional to local_density**this) instead of a uniform stride. 0 = uniform (current); 1 = fully density-weighted; small values a gentle tilt. Concentrates the seed budget on isolated, reproducible modes (isolation, not firing rate). |
| `--coarse-dr` | `0.15` | radial band half-width (fraction of the 1-99%% whitened-radius span) for spike->seed association in the COARSE pass. Lower -> tighter amplitude bands -> more, smaller coarse fibers (amplitude-distinct cells stay separate for the fine splitter); higher -> fewer, fatter coarse groups. Default 0.15. |
| `--whiten-source` | `auto` | FK_SESSION_WHITEN_SOURCE. Where the noise whitener comes from. nst = the .nst.<variant>.<group> ndm_extractspikes writes beside the .spk: noise measured in the .spk's own transform on spike-free windows, outliers rejected (one whitener for the session). fil = per chunk, off-spike .fil baseline taken into the .spk's waveform space (fiber_stderiv.space_transform; needs the .fil, and the sdiffPairs for a _C variant). spk = per chunk, the snippets' own pre-peak samples (stationary block-Toeplitz covariance): needs neither file and follows any variant, including ones no transform describes (fiber-adlp's per-spike filtered snippets). auto (default) = nst when the .spk's .nst exists, else fil. --deconv needs the .fil either way. — choices: `auto`, `nst`, `fil`, `spk` |
| `--no-whiten` | flag (off) | cluster in the RAW mask-selected (mean-centred) feature space instead of the .fil baseline-whitened space: identity whitener, so the radial coordinate is the raw feature norm and the angular metric is raw cosine (no per-channel covariance normalisation) |
| `--merge-corr` | `0.0` | consolidate fibers above this (0=off; 0.95 template / 0.90 sliding) |
| `--neigh-passes` | `0` | FK_SESSION_NEIGH_PASSES. passes of the neighbourhood re-cluster (Block N): rkk over the union of a random set of 1..--neigh-k neighbouring clusters (template cosine >= --neigh-cos), seeded from the current labels, adopted only when KK's penalised score improves. 0 = off. |
| `--neigh-k` | `3` | FK_SESSION_NEIGH_K. most neighbours drawn per seed cluster |
| `--neigh-cos` | `0.7` | FK_SESSION_NEIGH_COS. template cosine (masked channels, each realigned) that makes two clusters neighbours |
| `--neigh-seed` | `0` | FK_SESSION_NEIGH_SEED. seed of the neighbourhood draw |
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
| `--off-thr` | `0.0` | inter-channel-TIMING co-veto on Block-A fragment merges: refuse a footprint-correlation merge whose two fibers' RAW-template inter-channel offsets disagree by more than this (xcorr LAG samples, ~0.2-0.3; NOT the trough ~1.0). Reads .spk.standard for raw templates (stderiv timing is channel-mixed); inert if that file is absent. FK_SESSION_OFF_THR. 0 = OFF (default) |
| `--off-amp-frac` | `0.3` | channels below this fraction of the dominant peak-to-peak carry no reliable timing for --off-thr. FK_SESSION_OFF_AMP_FRAC |
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
| `--deconv` / `--no-deconv` | flag (off) | detection-anchored sub-sample template deconvolution per chunk (fiber_deconv): per-fiber deconv_agree contamination QC in .fibers and a .deconv.tsv manifest of labels, collision partners and undetected spikes. Changes no label unless --deconv-assign |
| `--deconv-assign` / `--no-deconv-assign` | flag (off) | with --deconv: move confidently attributed noise / collision-cluster spikes into their fiber as a parked child atom. Off by default: on synthetic sessions with look-alike cells its precision fell to 70-90 percent (docs/methods.md) |
| `--deconv-phases` | `4` | sub-sample phases per template (1 = whole samples) |
| `--deconv-order` | `4` | order of the VAR noise-whitening filter |
| `--deconv-refrac-ms` | `1.0` | a unit is not placed twice within this many ms |
| `--deconv-floor` | `16.0` | lowest pursuit gain (2 x log-likelihood) accepted, whatever the noise null says |
| `--deconv-null-q` | `0.995` | quantile of the off-spike first-step gain used as the stopping threshold |
| `--deconv-agree` | `0.999` | margin calibration: agreement with the clustering required among clean spikes above it |
| `--deconv-margin-min` | `10.0` | floor on the calibrated label margin (2 x log-likelihood ratio, best vs other unit) |
| `--deconv-min-n` | `30` | min spikes for a fiber to get a template |
| `--deconv-fiber-agree` | `0.95` | a fiber whose spikes the pursuit labels with an equivalent template at least this often is corroborated: the margin is calibrated on such fibers and only they receive rescues |
| `--deconv-equiv-cos` | `0.93` | fibers whose templates reach this whitened cosine (noise-corrected, best over +/-4 samples) are equivalent: the over-split fragments of a cell. Agreement, margin and collision partners are judged up to equivalence. >1 = identity only |
| `--quality-metrics` | flag (off) | also compute L-ratio + isolation distance (O(N*K)) |
| `--quality-dims` | `10` | PCA dims for L-ratio/isolation Mahalanobis |
| `--pca-k` | `6` |  |
| `--max-sub` | `8` |  |
| `--inclusion-k` | `3.0` | per-fiber radius = median+k*MAD of residuals; 0 disables |
| `--no-noise` | flag (off) | sweep every remaining noise spike (below the inclusion radius / rejected / collision junk) into a single UNDEFINED FIBER (one real cluster, not the noise cluster) rather than dropping it. For clean stderiv data; the undefined fiber is cleaned/re-split in later steps. |
| `--keep-small-groups` | flag (off) | a fine group below --fine-min-group (or whose inclusion core falls below it) becomes a fiber of its own -- all its spikes -- instead of going to the noise cluster. Over-split is recoverable downstream; a dropped unit is not. Groups below --keep-small-min are left to --assign-unsorted |
| `--keep-small-min` | `5` | smallest group --keep-small-groups keeps as its own fiber (default 5) |
| `--assign-unsorted` | flag (off) | place every spike still unsorted at the end of a chunk (coarse-unassigned, groups too small to keep): within its nearest fiber's own spread -> that fiber; outside every fiber's spread -> GMM groups of their own (with --keep-small-groups), the remainder -> nearest fiber (whitened, realigned features: shape and amplitude). With fiber-flag-artifacts -> --exclude-clu, only flagged artifacts reach cluster 0 |
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
| `--out-variant` | — | ALSO write .clu/.clc/.clp under this variant token, in addition to the --method-named copies. READS are unaffected (the .pca basis and .spk still resolve under --method), so an alias like stderiv_C5_D34 can name a feature space without needing its own basis or waveform file. |
| `--emit-pca` | flag (off) | with --emit-fet: also write <base>.pca.<out-variant or method>.<elec>, the basis the .fet is the projection of, so Klusters can realign/reproject in that space |
| `--emit-fet` | flag (off) | write <base>.fet.<out-variant or method>.<elec> -- the FULL-WIDTH session-wide projection on the global basis (the feature space the fine split used, before its per-chunk reduction), so Klusters can display it. |
| `--feat-lag` | `0` | FK_SESSION_FEAT_LAG. >0 = fine-split shape features become PC1 sampled at -N/0/+N samples plus PC2 (4 per channel) instead of PC1,PC2,PC3. 0 = classic. |
| `--feat-lag-pc2` | `1` | with --feat-lag: 1 (default) keep PC2 as a 4th column per channel, 0 = lags only |
| `--feat-hybrid` | `0` | FK_SESSION_FEAT_HYBRID. 1 = HYBRID fine-split features: the --method (stderiv) features as configured (lag space with --feat-lag) PLUS the leading standard-basis PCs of the paired RAW .spk.standard window of each spike, one realign shifting both. The derivative spaces lose broad, low-energy units (the loss grows with spike width: d' x0.6 at 1.5x the pyramidal width, x0.44 at 2x) that the raw space keeps; the raw block puts them back. Needs .spk.standard beside the method .spk. |
| `--hybrid-pcs` | `2` | FK_SESSION_HYBRID_PCS. standard-basis components per channel in the raw block (1-3) |
| `--hybrid-scale` | `1.0` | FK_SESSION_HYBRID_SCALE. weight of the raw block relative to the method block after the two are put on equal total variance (1 = equal; <1 derivative-led, >1 raw-led) |
| `--hybrid-gain-fit` | `all` | FK_SESSION_HYBRID_GAIN_FIT. energy half of the spikes the block-equalising gain is fit over: all (default) \| low \| high. Over all, the big spikes set the gain and the low-energy spikes load more on the derivative block than the nominal share. — choices: `all`, `low`, `high` |
| `--hybrid-deriv-basis` | `session` | FK_SESSION_HYBRID_DERIV_BASIS. basis of the method (derivative) block: session (the .pca.<method> basis, default) \| high (refit per channel on the high-energy half of a spike sample -- the low half's derivative variance is nearly unstructured). — choices: `session`, `high` |
| `--exclude-clu` | — | per-spike exclusion mask in .clu form (nonzero = drop), aligned to the .res -- e.g. fiber-flag-artifacts' output. Excluded spikes never enter a chunk, so they cannot shape a cluster; they are emitted as cluster 0. |
| `--in-stage` (`--in`) | — | input stage: the clu <base>.clu.<in-method>.<elec>.<in-stage> that --exclude-units names clusters of ('' = the stage-less <base>.clu.<in-method>.<elec>). It must be aligned to the .res this run reads (same spike count); a stage with its own .res (e.g. decollided) is refused |
| `--in-method` | — | variant token of the input-stage clu (default: --method), e.g. the --out-variant alias a curated clu was saved under |
| `--exclude-units` | — | clusters of the input-stage clu whose spikes are excluded from the sort: comma-separated ids and ranges, e.g. 3,7,12-15. Excluded spikes never enter a chunk; see --exclude-as for what the output holds for them |
| `--exclude-as` | `carry` | carry (default): each excluded unit is written back as one cluster of its own (one atom), after the new fibers; ids 0 and 1 of the input go to 0. noise: excluded spikes are cluster 0, like --exclude-clu — choices: `carry`, `noise` |
| `--no-cluster-basis` | flag (off) | ignore the global .pca basis for the fine-split shape features and use a per-call local SVD (legacy behaviour) |
| `--clu-stage` | `fiber_session` | post-group stage tag for the clu: <base>.clu.<method>.<elec>[.<stage>] (default 'fiber_session'); pass --clu-stage '' for an untagged .clu |
| `--skip-extant` | `0` | FK_SESSION_SKIP_EXTANT. 1 = if the destination .clu this run would write already exists (--out, else <base>.clu.<out-variant or method>.<elec>[.<clu-stage>]), log it and exit before any compute -- a plan re-run resumes past its fiber-session step instead of regenerating hours of sort over the top of an extant (possibly hand-curated) one. Delete the file, or set 0 (the default), to rebuild. |
| `--emit-hierarchy` / `--no-emit-hierarchy` | flag (on) | emit the .clu/.clc/.clp microfiber triple (atoms = pre-link fine fragments, fibers = linked global ids) via FiberHierarchy, instead of a flat .clu only. --no-emit-hierarchy writes just the flat .clu (legacy). Ignored with --out. |
| `--gpu` | flag (off) | run the realign/whiten kernels on GPU (CuPy; needs the [gpu] extra) |
| `--jobs` (`-j`) | `1` | FK_SESSION_JOBS. parallel worker processes over chunks (1 = serial, the default; 0 = one per chunk, capped at the CPUs this process may use). Chunks are independent, and this is the axis that scales: with --jobs 1 the only parallelism is BLAS inside one chunk, whose matrices are far too small to repay a 64-way fork/join barrier -- and NumPy's bundled OpenBLAS has a build-time thread ceiling (openblas_get_config() prints MAX_THREADS) that no fiber-kit setting can raise. |
| `--job-mem-gib` | `4.0` | FK_SESSION_JOB_MEM_GIB. resident memory to budget per chunk worker, GiB, used only to bound --jobs 0 (0 = no memory cap). Measured 2.5-3.4 GiB for a 50k-spike 12-min g6 chunk with the hybrid stack; an explicit --jobs N is never reduced. |
| `--feature-align` | — | feature-building alignment: xcorr (default), centroid (pure, no refine -- adds the trough-position-vs-asymmetry structure to the clustering/linking features), or off (skip the sub-sample align entirely; features are built from the windows exactly as extracted). Does NOT touch committing alignment or fiber-realign. Overrides the FIBER_ALIGN env var. — choices: `xcorr`, `centroid`, `off` |
| `--subsample` / `--no-subsample` | flag (off) | enable (--subsample) or disable (--no-subsample) realign's per-spike sub-sample (parabolic) refine in the feature build; default leaves the FIBER_SUBSAMPLE env var / lever untouched (off). Reaches pool workers. |
| `--out` | — |  |
| `--cons-mode` | `off` | FK_SESSION_CONS_MODE > FK_CONS_MODE: which consolidation passes run (default off) — choices: `off`, `strip`, `knn`, `both`, `shed` |
| `--cons-max-dist` | `0.5` | FK_CONS_MAX_DIST (default 0.5) |
| `--cons-gmin` | `0.0` | FK_CONS_GMIN (default 0.0) |
| `--cons-gmax` | `10.0` | FK_CONS_GMAX (default 10.0) |
| `--cons-chan` | `0.0` | FK_CONS_CHAN (default 0.0) |
| `--cons-margin` | `0.85` | FK_CONS_MARGIN (default 0.85) |
| `--cons-own-q` | `0.9` | FK_CONS_OWN_Q (default 0.9) |
| `--cons-tgt-q` | `0.9` | FK_CONS_TGT_Q (default 0.9) |
| `--cons-tgt-scale` | `1.25` | FK_CONS_TGT_SCALE (default 1.25) |
| `--cons-twin` | `0.93` | FK_CONS_TWIN (default 0.93) |
| `--cons-tpl-cap` | `1024` | FK_CONS_TPL_CAP (default 1024) |
| `--cons-min-tpl` | `8` | FK_CONS_MIN_TPL (default 8) |
| `--cons-iters` | `1` | FK_CONS_ITERS (default 1) |
| `--cons-cap-amp` | `0.0` | FK_CONS_CAP_AMP (default 0.0) |
| `--cons-pure-iqr` | `0.0` | FK_CONS_PURE_IQR (default 0.0) |
| `--cons-agg-iqr` | `0.0` | FK_CONS_AGG_IQR (default 0.0) |
| `--cons-agg-every` | `3` | FK_CONS_AGG_EVERY (default 3) |
| `--cons-agg-minnew` | `30` | FK_CONS_AGG_MINNEW (default 30) |
| `--cons-agg-vac` | `0` | FK_CONS_AGG_VAC (default 0) |
| `--cons-tpl-align` | `0` | FK_CONS_TPL_ALIGN (default 0) |
| `--cons-park-min` | `0` | FK_CONS_PARK_MIN (default 0) |
| `--cons-shed-k` | `3.0` | FK_CONS_SHED_K (default 3.0) |
| `--cons-shed-min` | `40` | FK_CONS_SHED_MIN (default 40) |
| `--cons-knn-k` | `20` | FK_CONS_KNN_K (default 20) |
| `--cons-knn-thr` | `0.3` | FK_CONS_KNN_THR (default 0.3) |
| `--cons-knn-minref` | `50` | FK_CONS_KNN_MINREF (default 50) |
| `--cons-knn-minnew` | `30` | FK_CONS_KNN_MINNEW (default 30) |
| `--cons-knn-dims` | `16` | FK_CONS_KNN_DIMS (default 16) |
| `--cons-fold-thr` | `0.9` | FK_CONS_FOLD_THR (default 0.9) |
| `--cons-scorr` | `0.93` | FK_CONS_SCORR (default 0.93) |
| `--cons-off-thr` | `0.0` | FK_CONS_OFF_THR (default 0.0) |
| `--cons-resid-k-on` | `0.0` | FK_CONS_RESID_K_ON (default 0.0) |
| `--cons-resid-k-off` | `0.0` | FK_CONS_RESID_K_OFF (default 0.0) |
| `--cons-resid-win` | `0` | FK_CONS_RESID_WIN (default 0) |
| `--cons-resid-mingain` | `0.25` | FK_CONS_RESID_MINGAIN (default 0.25) |
| `--cons-resid-min` | `200` | FK_CONS_RESID_MIN (default 200) |
| `--cons-resid-iqr-max` | `0.0` | FK_CONS_RESID_IQR_MAX (default 0.0) |
| `--cons-resid-iqr-min` | `0.0` | FK_CONS_RESID_IQR_MIN (default 0.0) |
| `--cons-resid-align` | `chunk` | FK_CONS_RESID_ALIGN: which alignment the residual strip scores in (default chunk) — choices: `chunk`, `cluster`, `stored` |
| `--realign` | `auto` | FK_SESSION_REALIGN. At the end of the run, align every spike to its fiber's template (fiber-realign's Klusters per-spike offsets, on the .clu this run wrote), commit the shifted spike times to the shared .res (+ <base>.offsets.<group>.npy), and rebuild EVERY waveform variant's .spk on disk at the shifted times (standard, stderiv*, ...) and its .fet (from its .pca; _D alias .fet included) -- what `fiber-realign --reextract --refeaturize --variants all` does as a separate step. reextract = re-read the .fil; shift-spk = circularly roll the existing .spk (no .fil); auto (default) = reextract when the .fil exists, else shift-spk; off = leave .res/.spk/.fet as extracted. — choices: `auto`, `reextract`, `shift-spk`, `off` |
| `--realign-algo` | `klusters` | FK_REALIGN_ALGO. fiber-realign --align-algo for the end-of-run commit; its other knobs (FK_REALIGN_MAX_SHIFT / _ITERS / _MIN_SCORE / _RECENTER_RMS / _RMIN) apply as they do there — choices: `klusters`, `template`, `centroid` |
| `--realign-fil` | — | filtered signal the end-of-run re-extraction reads (default <base>.fil) |
| `--chunks` | `""` | FK_CHUNKS: process only these 0-based chunks, e.g. 14-16 or 3,7 ('' = all); the other chunks' spikes go to the reserved clusters 16777116+chunk |
### `fiber-realign`

Per-spike Klusters-style realignment + corrected .res spike times, with optional re-extraction from .fil and re-featurisation (commit-and-reextract finalize).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from YAML) |
| `--peak` | — | override: the group's peakSampleIndex, 1-BASED like the YAML's (default the YAML's value; fiber-session passes one when the session has none). Windows hold the spike time at index peak-1 -- neurosuite-3's cut -- which is also where the klusters algorithm aligns each template's peak. With a .fil, the windows on disk are located in it first: windows cut elsewhere (a 0.38.1-0.50 commit cut them at peak) are aligned in their own geometry and re-cut at peak-1 |
| `--clu` | — | cluster file (default <base>.clu.<group>; pass the refined/relinked one, e.g. <base>.clu.stderiv.<group>.refine) |
| `--align-algo` (`--align-method`) | `klusters` | alignment algorithm (runs on the clu's variant waveform): klusters = iterative normalised-xcorr vs pre-aligned mean; template = legacy median/un-normalised; centroid = reference-free per-spike energy-centroid (fiber_lib.centroid_shift), no peak/template/labels needed. (Named --align-method to avoid colliding with the --method extraction-variant flag used by other tools.) — choices: `klusters`, `template`, `centroid` |
| `--max-shift` | — | klusters: lag search radius (default Klusters': max(1, (peak-1)//2)) [FK_REALIGN_MAX_SHIFT] |
| `--iters` | `2` | klusters: template/xcorr passes (Klusters --iterations, default 2) [FK_REALIGN_ITERS] |
| `--min-n` | `20` |  |
| `--min-score` | `0.7` | klusters: leave a spike unshifted in a pass if its best normalised xcorr is below this (Klusters --threshold, default 0.70) [FK_REALIGN_MIN_SCORE] |
| `--recenter-rms` | `0` | klusters: 1 = Klusters realign mode 2 (--recenter-rms): after the passes, shift each unit so its channel-summed energy centroid lands on the peak [FK_REALIGN_RECENTER_RMS] |
| `--rmin` | `0.4` | klusters --recenter-rms: minimum mean resultant length of the energy profile; below it the recenter is skipped (Klusters --rmin, default 0.4) [FK_REALIGN_RMIN] |
| `--kernel` | `hamming` | klusters: the correlation window (Klusters --kernel, Preferences > Realignment > Correlation window): hamming (default) / hann / gauss = a bell of --kernel-width samples each side of the peak sample; template = the template's own energy profile; none = the whole window (the pre-0.67.9 kernel). Under a window the score is normalised by the energy the window admits at each lag, so a second spike in the window (a collision) cannot steer the lag [FK_REALIGN_KERNEL] — choices: `none`, `gauss`, `hann`, `hamming`, `template` |
| `--kernel-width` | `16` | klusters --kernel-width: half-width of the gauss / hann / hamming window in samples of the recording (default 16 = 0.49 ms at 32,552 Hz; 0 = half the peak position, Klusters' automatic) [FK_REALIGN_KERNEL_WIDTH] |
| `--upsample` | `1` | klusters: cubic-spline upsample factor for sub-sample matching (2 = half-sample lags); .res is rounded back to whole samples at save |
| `--reextract` | flag (off) | re-extract each spike's window from .fil at the corrected timestamp -> new .spk |
| `--shift-spk` | flag (off) | commit WITHOUT a .fil: circularly roll the existing .spk of each variant by the per-spike integer offset (valid for high-pass .spk; the stderiv transform commutes with a time shift). The no-.fil equivalent of --reextract; pair with --refeaturize to reproject the rolled windows onto .pca. Mutually exclusive with --reextract. |
| `--refeaturize` | flag (off) | reproject the re-extracted/rolled windows onto .pca.<variant> -> new .fet (implies --reextract, or --shift-spk when that is set) |
| `--fil` | — | filtered signal path (default <base>.fil) |
| `--variants` | — | 'all' = every base-rate waveform variant with a .spk for the group (+ the --variant's waveform token; a super-sampled _U<factor> space is never written by a commit -- Klusters' Sync Feature Spaces or ndm_extractspikes --supersample rebuilds it); or a comma list of feature spaces to refresh from .fil (default: standard + stderiv if present). Each is re-derived from the re-extracted raw window: standard=raw; stderiv=SDIFF_ALLPAIRS+temporal-diff; stderiv_C4/_C5 use the session's own spikeDetection.channelGroups[N].sdiffPairs pattern (partner map / reference sets); then projected onto its .pca |
| `--out-stage` (`--out-tag`, `--out`) | `""` | stage tag for committed outputs (default: empty -> overwrite the canonical .res/.clu/.spk/.fet[.<variant>].<group> in place; the realign IS the commit). Pass a tag only if you want a side-by-side copy, e.g. --out-tag realigned |
| `--variant` | `""` | feature variant the .clu/.spk/.fet adhere to (e.g. stderiv_C5): drives which .spk is aligned and which .fet is refreshed, OVERRIDING the --clu-name inference. The pipeline hard-codes 'stderiv' in the clu path, so a custom-sdiffPairs session (stderiv_C5) needs this to name its real variant instead of leaning on family-matching. The .res stays the single shared copy (resolve_any). (FK_REALIGN_VARIANT) |
| `--out-variant` | — | variant the .res/.clu adhere to (default: inferred from --clu, e.g. stderiv; falls back to standard). There is one .res/.clu under this variant; .spk/.fet are written per feature space in --variants (standard raw, stderiv transform) |
| `--emit-clu` | flag (off) | re-emit the (label-unchanged) clu next to the committed .res/.spk/.fet so the set opens in Klusters as a unit (default on) |
| `--no-emit-clu` | flag (on) | do NOT write the clu. Use when the input --clu is a stage-tagged clu but the outputs commit canonically (--out-tag ''): re-emitting would overwrite the BASE .clu.<variant>.<group> with this stage's labels. The stage clu already exists and its labels are unchanged, so skipping the write keeps the base over-cluster intact. |
| `--recut` | flag (off) | re-cut every variant's windows from the .fil at the CURRENT .res and move no spike (zero offsets): the repair for windows committed with another geometry (before 0.51 a commit cut them one sample after its alignment target) and the refresh after any .res edit. Implies --reextract; pair with --refeaturize and --variants all. The .clu is not re-emitted (no label changes) |
| `--check` | flag (off) | report what a realign of this sort would do and write NOTHING: where the windows on disk hold the spike time against the group's peakSampleIndex - 1 (with a .fil), the spikes it would move, where each unit's template peaks against the target on the windows as they are, and the same-sample spike pairs and inversions in the .res now and after the shift. The commit flags are ignored |
| `--park-duplicates` / `--no-park-duplicates` | flag (off) | after the shift, move the later spike of every same-unit pair that landed on one sample (two detections of one spike; identical windows) to a new duplicates cluster (the next free id; never 0) in the --clu, in place, as one child atom under it when the clu has its .clc/.clp; with --out-tag (the canonical .res is left alone) only in the clu emitted under the tag. Rows are never removed. Cross-unit pairs are only listed (<base>.realign.<variant>.<group>[.<tag>].tsv). fiber-session's own commit parks; a stand-alone run only lists unless this is set [FK_REALIGN_PARK_DUPS] |
| `--out-res` | — |  |
| `--out-off` | — |  |
| `--chunks` | `""` | FK_CHUNKS: process only these 0-based chunks, e.g. 14-16 or 3,7 ('' = all); the other chunks' spikes go to the reserved clusters 16777116+chunk |
| `--chunk-length` | `12.0` | FK_SESSION_CHUNK_MIN: chunk length in minutes that --chunks counts in |
### `fiber-refine`

Dedup at the imposed refractory, then iteratively split/peel a fine sort into clean units; writes a refined .clu (+ deduped .res).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from YAML) |
| `--in-clu` | — | path to the input sort to refine; default = canonical .clu if present, else a fresh fine sort |
| `--out-method` | `stderiv` | feature space written BEFORE the group (standard\|stderiv\|...); refine operates in stderiv space, so default stderiv |
| `--no-cluster-basis` | flag (off) | ignore the global .pca basis for the split-stage shape features and use a per-call local SVD (legacy behaviour) |
| `--out-stage` (`--out`) | `refine` | fiber STAGE written AFTER the group, e.g. 'refine' -> <base>.clu.<out-method>.<group>.refine ('' for none). (was --out-variant; renamed so --out-variant is free for the feature variant, as in fiber-realign) |
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
| `--fine-algo` (`--fine-method`) | `gmm` | method for the initial fine sort when no --in-clu is given (rkk = masked-EM KlustaKwik, the same cluster_chunk_fine backend fiber-session's --fine-algo uses) — choices: `gmm`, `rkk`, `fiber`, `none` |
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
### `fiber-peel`

fiber-peel: consolidate over-split refine fragments by footprint cosine gated by the refractory cross-CCG.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--cpos-method` | `stderiv` |  |
| `--cpos-stage` | `refine` |  |
| `--clu-method` (`--method`) | — |  |
| `--clu-stage` (`--in`) | — |  |
| `--out-stage` (`--out`) | — | output clu stage (default: same as input == in-place relabel) |
| `--foot-hi` | `0.97` | anneal start: strict footprint-cosine for the first, most certain merges |
| `--foot-lo` | `0.9` | anneal floor: loosest footprint-cosine accepted (best g5 result at 0.90) |
| `--anneal-steps` | `4` |  |
| `--refrac-ms` | `2.0` | refractory half-window (ms) for the veto cross-CCG |
| `--refrac-thr` | `0.3` | coincidence-ratio above which the pair is two cells (veto) |
| `--refrac-min-exp` | `5.0` | min expected coincidences for the veto to be powered (else abstain -> merge allowed) |
| `--refrac-censor-ms` | `0.0` | censor window (ms) dropping duplicate detections of one spike |
| `--min-n` | `15` | min spikes for a fragment to participate |
| `--burst-foot-lo` | `0.0` | burst pass: pairs with this <= footprint cosine < --foot-lo merge only on a powered refractory dip AND the complex-spike burst signature in the cross-CCG (0 = off) |
| `--burst-ms` | `2,15` | burst window (ms): the smaller fragment follows the larger by lo..hi |
| `--burst-base-ms` | `50,150` | flank window (ms, both sides) giving the expected cross-CCG rate |
| `--burst-alpha` | `0.01` | significance for the forward peak (Poisson) and the ordering (binomial) |
| `--burst-ratio` | `2.0` | forward count must reach this multiple of the flank expectation |
| `--burst-min-count` | `10` | min forward+backward pairs to test the burst signature (else abstain) |
### `fiber-cpos`

Write a per-spike cluster-position sidecar (.cpos) by localizing each cluster's median RAW template (monopole+dipole).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
| `--spk` | — | path to a STANDARD/raw .spk (preferred over .fil); never the stderiv .spkD |
| `--spk-method` | `standard` | method of the raw .spk to resolve: <base>.spk.<spk-method>.<elec> |
| `--fil` | — | path to raw .fil (fallback if no standard .spk) |
| `--fil-offset` | `0` | first absolute sample of the .fil (0 for full recording) |
| `--clu-method` (`--method`) | `stderiv` | source-clu feature space BEFORE the group (standard\|stderiv\|...) |
| `--clu-stage` (`--in`) | `refine` | source-clu fiber STAGE AFTER the group: read <base>.clu.<clu-method>.<elec>.<clu-stage> |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--clu-stage) |
| `--out-method` | — | cpos method BEFORE the group (default: mirror --clu-method) |
| `--out-stage` (`--out`) | — | cpos fiber STAGE AFTER the group (default: mirror --clu-stage) |
| `--min-spikes` | `15` |  |
| `--max-spikes` | `20000` | spikes per cluster its template and position are taken from (seeded random sample; 0 = all). Bounds memory: a 380k-spike cluster in float64 alone is ~1.3 GB |
| `--no-dipole` | flag (off) |  |
| `--no-templates` | flag (off) | skip per-cluster median templates in the .clusters.npz |
| `--amp-algo` (`--amp-method`) | `pc1` | per-channel amplitude profile for the position inverse: pc1=rank-1 denoised template (default, sharpest footprint + most precise), wave=median-waveform ptp, ptp=median per-spike ptp (legacy; ~4-sigma noise floor on far channels flattens the footprint). — choices: `pc1`, `wave`, `ptp` |
| `--amp-basis` | `auto` | amplitude basis the gate-facing positions use: 'pca'=read .pca.standard.<elec> (PC1 score per channel = the .fet amplitude); 'fit'=group basis from .spk; 'auto'=pca if present else fit; 'none'=per-cluster SVD — choices: `auto`, `pca`, `fit`, `none` |
| `--no-amp-basis` | flag (off) | alias for --amp-basis none (per-cluster SVD) |
| `--nboot` | `0` | bootstrap draws for the depth/distance percentile CIs (z_lo/z_hi/y_lo/y_hi). This loop is ~5x the rest of the cost (the dominant runtime); positions (x0,y0,z0,A) and the energy-tercile depth-shift do NOT use it. Use --nboot 0 for identical positions ~5x faster (analytic sig_y is still written; the percentile CIs become NaN). |
| `--probe` | — | probe file(s) for geometry (else from chunk xy via YAML) |
### `fiber-intrachunk`

Collapse over-split fragments within each chunk into units (stderiv cosine + offset + depth).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--probe` | — | probe file(s) for on-the-fly localisation when no fiber-cpos table exists (else the probe named in <session>.yaml) |
| `--cpos-method` | `stderiv` |  |
| `--cpos-stage` | — | positions (.cpos) stage tag; default follows --clu-stage, else 'refine' |
| `--clu-method` (`--method`) | — |  |
| `--clu-stage` (`--in`) | — |  |
| `--chunk-minutes` (`--chunk-min`) | `12.0` |  |
| `--gate` | (from config) | shape gate: cosine\|mmd\|kcov\|cfiber\|band (default band: energy-scaled median+/-sigma overlap) — choices: `cosine`, `mmd`, `kcov`, `cfiber`, `band` |
| `--cos-thr` | (from config) | cosine recall prefilter |
| `--off-thr` | (from config) | inter-channel offset RMS gate (samples) |
| `--depth-gate` | (from config) | depth gate (um) |
| `--amp-gate` | (from config) | absolute log-amplitude (energy) gate, natural log; ln(3)=1.1 -> 3x (0=off) |
| `--amp-band-gap` | (from config) | post-linkage amplitude-band partition (0=off): split a unit's atoms where the sorted log2 CEILING gap exceeds this. Ceilings = median best-channel p2p of each atom's top AMP_BAND_TOP fraction on the raw .spk.standard -- the un-adapted top of the adaptation ladder -- so the same-cell tolerance is tight (~1.3x = 0.33 log2 measured) where whole-cluster amplitude is adaptation-confounded (the 3x amp_gate). Covers EVERY linkage incl. 'ms', which carries no pairwise amplitude gate at all. 0.4 measured best on g6 (+2.6pp GT purity, completeness unchanged). Inert without .spk.standard. |
| `--amp-band-top` | (from config) | top fraction of an atom's spikes (by best-channel p2p) defining its ceiling |
| `--amp-band-min` | (from config) | min atom spikes to ANCHOR a band; smaller atoms attach to the nearest band |
| `--recut-gate` | (from config) | post-linkage median-residual KlustaKwik denoise of every unit: '' off \| all \| dip \| giqr. 'all' recuts every unit >= RECUT_MIN_SPK (surveyed g6: 40 units improved GT purity >2pp, none hurt -- kk's model selection + the component floor are the quality control at UNIT level, but chain-scored 'all' over-splits drifting cells: ARI 0.188 -> 0.144). 'giqr' (own-gain IQR) is the measured sweet spot -- it recuts only amplitude-impure units: +4.3pp GT purity with ARI UP (0.190), vs 'dip' +2.5pp/0.189 and 'all' +5.7pp at heavy ARI/completeness cost (terminal pre-curation use only) — choices: ``, `all`, `dip`, `giqr` |
| `--recut-dims` | (from config) | residual-PCA features fed to the recut KlustaKwik round |
| `--recut-rounds` | (from config) | max recut passes per unit lineage (products are re-tested -- the curator's double round) |
| `--recut-min-n` | (from config) | min spikes for a recut component to become a unit; smaller components and kk's noise bucket shed to reserve |
| `--recut-min-spk` | (from config) | min unit spikes to attempt a recut |
| `--recut-alpha` | (from config) | gate 'dip': Hartigan dip p on the residual PCs below this = recut |
| `--recut-giqr` | (from config) | gate 'giqr': own-gain IQR above this = recut |
| `--refrac-ceiling` | (from config) | reject merge if combined 2ms-ISI violation > this percent (empty=off) |
| `--pre-merge-cos` | (from config) | pre-collapse obvious mutual-NN pairs at cosine>=this (0=off) |
| `--iter` | (from config) | iterate group->re-estimate->regroup this many passes (1=single pass); >1 keeps the tight gate but re-merges DENOISED units across passes, consolidating over-split fragments a single pass leaves. Early-converges when a pass merges nothing (g5: 5 -> ~1124). Left at 1 in production; the exp config opts in (FK_INTRA_ITER). |
| `--linkage` | (from config) | complete\|dynamic\|ms\|kk — choices: `complete`, `dynamic`, `ms`, `kk` |
| `--kk-dims` | (from config) | kk linkage (MW-cluster + median-residual KlustaKwik): PCs of the median-residual waveform used to split within each MW group (0 or >=ncol = all). Reduced dims keep the split sensitive to the dominant between-cell deviation. |
| `--kk-mw-thr` | (from config) | kk linkage: median-waveform grouping cut = 1 - cosine; atoms whose windowed median waveforms are within this cosine distance group together (complete-linkage, so EVERY pair must be within it). 0.10 -> group only atoms with median-waveform cosine >= 0.90. Lower = stricter grouping (less merging); this wall is what prevents pooling distinct cells. |
| `--kk-win` | (from config) | kk linkage: half-window (samples each side of the peak) the waveform is cropped to BEFORE the median-waveform cosine and the median-residual SVD -- focuses both on the peak region and drops the flat baseline tails (validated: cleaner refractory than the full 42-sample window). |
| `--kk-min-size` | (from config) | kk linkage: KlustaKwik min cluster size for the within-group split. |
| `--kk-big` | (from config) | kk linkage: atoms with >= this many spikes are 'big' (kept as grouping anchors); smaller atoms are folded into the nearest big atom by gated median-waveform cosine, or reserved if none matches. fiber_stochastic emits mostly tiny atoms, so this fold is what brings the singles into the fold and prevents the over-split. |
| `--kk-fold-thr` | (from config) | kk linkage: a small atom folds into the nearest big atom only if their windowed median-waveform cosine >= this; else it goes to the reserve (id 1). This gate is the reject option a Mahalanobis assignment lacks -- a tiny fragment of a real cell folds in, a tiny distinct/noise atom stays out (holds zero false-merge). |
| `--kk-strip` | (from config) | kk linkage: per-channel polynomial-coefficient contamination strip run after the split (0 = off, 1 = on; ON by default). Fits each spike's per-channel waveform with a Legendre polynomial (smooths per-sample stderiv noise into stable coeffs) and moves robust-z outlier spikes to the reserve. Cleans most contaminated units while keeping clean units usable (raw per-sample strips cannot -- stderiv self-correlation ~0.6 drowns the signal). |
| `--kk-strip-deg` | (from config) | kk linkage: Legendre polynomial degree for the per-channel strip (validated: 4 best; 3 under-cleans, 5 over-strips). |
| `--kk-strip-z` | (from config) | kk linkage: robust-z (MAD-scaled) threshold on the per-channel polynomial coefficients; a spike is a per-channel outlier if any coefficient exceeds this. |
| `--kk-strip-maxbad` | (from config) | kk linkage: a spike is stripped to the reserve if it is a polynomial-coeff outlier on MORE than this many signal channels (0 = strictest). |
| `--align-lag` | (from config) | merge-time best-lag half-window, NATIVE samples (0=off) |
| `--align-upsample` | (from config) | cubic-spline upsampling factor for the align-lag search |
| `--cfiber-q` | (from config) | cfiber self-calibration quantile |
| `--cfiber-null` | (from config) | cfiber split-half null basis: order\|energy — choices: `order`, `energy` |
| `--band-thr` | (from config) | gate='band': min energy-scaled median+/-sigma band-overlap IoU to merge (empty -> 0.5) |
| `--cfiber-thr-floor` | (from config) | absolute floor on the self-calibrated cfiber threshold (0=off) |
| `--sig-cap` | (from config) | per-fragment spikes for the mean template (empty = no cap) |
| `--warp-thr` | (from config) | group-delay WARP coherence gate (Omlor-Giese): merge only if the cross-channel correlation of the two fragments' per-channel group-delay profiles >= this. Same-neuron warps cohere; co-located different cells anti-correlate. Group-delay is noisy at low spike count -> use LOW (~0.3). empty=off. |
| `--warp-resid-thr` | (from config) | single-channel warp-incongruity SUB-GATE (layers on warp_thr): among already-coherent pairs (corr>=0.85), veto if any ONE centroid-range channel's group-delay residual (Theil-Sen line) > this many samples -- a strong-channel-masked different source. g5 knee ~1.0. empty=off. |
| `--off-thr-int` | (from config) | DUAL gate: offset RMS threshold for suspected INTERNEURON pairs (narrow trough-to-peak). Fast cells have stable offsets (~0.23) so off_thr=1.0 is inert; tighten to ~0.5. Needs raw .spk for cell-typing. empty=off (use off_thr). |
| `--off-thr-pyr` | (from config) | DUAL gate: offset RMS threshold for suspected PYRAMIDAL pairs (wide trough-to-peak); ~1.0. Set BOTH off_thr_int and off_thr_pyr to enable the dual gate; mixed pairs use the stricter. empty=off. |
| `--cons-mode` | `off` | FK_INTRA_CONS_MODE > FK_CONS_MODE: which consolidation passes run (default off) — choices: `off`, `strip`, `knn`, `both`, `shed` |
| `--cons-max-dist` | `0.5` | FK_CONS_MAX_DIST (default 0.5) |
| `--cons-gmin` | `0.0` | FK_CONS_GMIN (default 0.0) |
| `--cons-gmax` | `10.0` | FK_CONS_GMAX (default 10.0) |
| `--cons-chan` | `0.0` | FK_CONS_CHAN (default 0.0) |
| `--cons-margin` | `0.85` | FK_CONS_MARGIN (default 0.85) |
| `--cons-own-q` | `0.9` | FK_CONS_OWN_Q (default 0.9) |
| `--cons-tgt-q` | `0.9` | FK_CONS_TGT_Q (default 0.9) |
| `--cons-tgt-scale` | `1.25` | FK_CONS_TGT_SCALE (default 1.25) |
| `--cons-twin` | `0.93` | FK_CONS_TWIN (default 0.93) |
| `--cons-tpl-cap` | `1024` | FK_CONS_TPL_CAP (default 1024) |
| `--cons-min-tpl` | `8` | FK_CONS_MIN_TPL (default 8) |
| `--cons-iters` | `1` | FK_CONS_ITERS (default 1) |
| `--cons-cap-amp` | `0.0` | FK_CONS_CAP_AMP (default 0.0) |
| `--cons-pure-iqr` | `0.0` | FK_CONS_PURE_IQR (default 0.0) |
| `--cons-agg-iqr` | `0.0` | FK_CONS_AGG_IQR (default 0.0) |
| `--cons-agg-every` | `3` | FK_CONS_AGG_EVERY (default 3) |
| `--cons-agg-minnew` | `30` | FK_CONS_AGG_MINNEW (default 30) |
| `--cons-agg-vac` | `0` | FK_CONS_AGG_VAC (default 0) |
| `--cons-tpl-align` | `0` | FK_CONS_TPL_ALIGN (default 0) |
| `--cons-park-min` | `0` | FK_CONS_PARK_MIN (default 0) |
| `--cons-shed-k` | `3.0` | FK_CONS_SHED_K (default 3.0) |
| `--cons-shed-min` | `40` | FK_CONS_SHED_MIN (default 40) |
| `--cons-knn-k` | `20` | FK_CONS_KNN_K (default 20) |
| `--cons-knn-thr` | `0.3` | FK_CONS_KNN_THR (default 0.3) |
| `--cons-knn-minref` | `50` | FK_CONS_KNN_MINREF (default 50) |
| `--cons-knn-minnew` | `30` | FK_CONS_KNN_MINNEW (default 30) |
| `--cons-knn-dims` | `16` | FK_CONS_KNN_DIMS (default 16) |
| `--cons-fold-thr` | `0.9` | FK_CONS_FOLD_THR (default 0.9) |
| `--cons-scorr` | `0.93` | FK_CONS_SCORR (default 0.93) |
| `--cons-off-thr` | `0.0` | FK_CONS_OFF_THR (default 0.0) |
| `--cons-resid-k-on` | `0.0` | FK_CONS_RESID_K_ON (default 0.0) |
| `--cons-resid-k-off` | `0.0` | FK_CONS_RESID_K_OFF (default 0.0) |
| `--cons-resid-win` | `0` | FK_CONS_RESID_WIN (default 0) |
| `--cons-resid-mingain` | `0.25` | FK_CONS_RESID_MINGAIN (default 0.25) |
| `--cons-resid-min` | `200` | FK_CONS_RESID_MIN (default 200) |
| `--cons-resid-iqr-max` | `0.0` | FK_CONS_RESID_IQR_MAX (default 0.0) |
| `--cons-resid-iqr-min` | `0.0` | FK_CONS_RESID_IQR_MIN (default 0.0) |
| `--cons-resid-align` | `chunk` | FK_CONS_RESID_ALIGN: which alignment the residual strip scores in (default chunk) — choices: `chunk`, `cluster`, `stored` |
| `--profile` | `default` | fallback profile for any intrachunk knob left unset in CLI/env/<session>.yaml: 'recommended' = the tuned pipeline baseline (cfiber gate, amp-gate 1.1, refrac 1.0, pre-merge 0.97, sig-cap 8000); 'default' = the conservative library baseline. — choices: `default`, `recommended` |
| `--off-n-ref` | — | SNR-adaptive offset gate: spike count at which --off-thr applies as-is; loosens ~1/sqrt(n) below it (recommend ~150). Omit for flat off_thr. |
| `--off-ceil` | `2.0` | cap on the adaptive offset tolerance (default 2.0; ~95%% same-neuron knee). |
| `--split-min-sil` | `0.12` | ms linkage: min silhouette to accept a split. |
| `--split-min-n` | `40` | ms linkage: min spikes per split sub-unit. |
| `--var-env-mult` | `3.0` | dynamic linkage: single-unit variance envelope = this * median fragment variance; blocks merges that push a unit's spread past it (permits the growth de-fragmentation needs, caps over-growth). |
| `--ccg-thr` | `1000000000.0` | dynamic linkage: max cross-CCG refractory ratio to admit a merge. DEFAULT OFF (1e9): a simple refractory-dip requirement is WRONG-SIGNED for de-fragmentation -- same-neuron over-split fragments (time-shift dups, and amplitude-splits of bursting cells whose spikes attenuate through the burst) show a short-lag cross-CCG PEAK, not a dip, so a low threshold rejects true merges (g5: collapses 1243->1850). Left as scaffolding; a correct temporal term needs duplicate-coincidence vs distinct-co-activity. |
| `--ccg-win` | `2.0` | dynamic linkage: cross-CCG refractory half-window (ms). |
| `--cfiber-thr` | — | gate='cfiber' shape-distance threshold; default None self-calibrates from per-fragment split-half nulls at --cfiber-q. |
| `--min-n` | `12` |  |
| `--boundary-minutes` | `3.0` | half-window (min) of straddling spikes for the overlap backbone anchor (--emit-units) |
| `--backbone-std-cos` | `0.75` | STANDARD median-template cosine a backbone pair must also clear (--emit-units). The stderiv boundary match is near-blind to depth/amplitude; this co-gate on the raw axis rejects the different-unit anchors it lets through. 0 disables. |
| `--backbone-warp` | — | optional Omlor-Giese group-delay coherence a backbone pair must also clear. OFF by default: on octrodes most units span <=2 channels so group-delay is degenerate (non-discriminative). Set only if energy spreads over >=3 channels. |
| `--out-stage` (`--out`) | — | output .clu stage (default: <clu-stage>_intrachunk) |
| `--emit-units` | flag (off) | also write a <...>.units.npz unit-signature table for fiber-link |
| `--no-provenance` | flag (on) | skip the .merge.tsv per-merge provenance sidecar (default: write it) |
### `fiber-link`

Link per-chunk fragments into tracked units (position fingerprint + A anchor + template co-gate).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--cpos-method` | `stderiv` |  |
| `--cpos-stage` | `refine` |  |
| `--clu-method` (`--method`) | — | source .clu method (default: mirror --cpos-method) |
| `--clu-stage` (`--in`) | — | source .clu stage (default: mirror --cpos-stage) |
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
| `--out-stage` (`--out`) | — | output .clu stage (default: <clu-stage>_linked) |
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

## Alternative / drift linkers

### `fiber-anchor-link`

Link fiber-session fragments across chunks, SEEDED by the overlap anchors (links established from
the same physical spikes in the chunk overlap, correct by construction) and gated on a null
calibrated from the session itself: within-chunk pairs are different cells by construction and set
the floor at the requested false-positive rate.  Gates run on the candidate pool BEFORE
mutual-nearest-neighbour selection; writes .clu plus the .clc/.clp hierarchy.

Positional: `args`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--clu-method` (`--method`) | `stderiv` | fragment .clu feature space (before the group) |
| `--clu-stage` (`--in`) | `fiber_session` | fragment .clu stage tag |
| `--in-clu` | — | explicit fragment .clu path (overrides --clu-method/--clu-stage) |
| `--frag-layer` | `atoms` | which layer of the source stage is the fragment layer: 'atoms' (default) prefers the .clc sibling -- the overlap anchors index .clc labels; 'units' reads the stage's FLAT .clu, so the linker consolidates the stage's UNITS (e.g. fiber-intrachunk output) instead of re-linking its atoms from scratch. An explicit --in-clu overrides both. — choices: `atoms`, `units` |
| `--spk-method` (`--spk-variant`) | `standard` | waveform axis for templates/warp (standard = curation axis) |
| `--anchor-stage` | `fiber_session` | stage whose .clc atoms fiber-session's anchors index (used when the anchors come from the unstaged .fibers): fiber_session, or fs_clean after fiber-consensus |
| `--anchor-majority` | `0.6` | an anchor atom maps to the fragment holding at least this fraction of its spikes |
| `--fibers` | — | path to the .fibers npz carrying the overlap anchors (default: derive from --clu-method/--clu-stage) |
| `--out-stage` (`--out-tag`, `--out`) | `anchor_linked` | post-fiber stage tag of the output .clu (single token) |
| `--hierarchy` | `1` | 1 = also write the .clc/.clp sibling hierarchy |
| `--promote-noise` | `1` | 1 (default) = treat cluster 0 as a real cell (its own atom + cell); 0 = leave it as noise |
| `--artifact-clu` | — | per-spike artifact mask in .clu form (nonzero = artifact), the SAME mask given to fiber-session --exclude-clu. Masked spikes are withheld from --promote-noise and left in cluster 0, so a promoted cell is the genuine unassigned population only. |
| `--chunk-min` | — | chunk length (min); default from yaml or 12 |
| `--seed` | `0` |  |
| `--prim-frac` | `0.15` | FK_ALINK_PRIM_FRAC (default 0.15) |
| `--z` | `0.5` | FK_ALINK_Z (default 0.5) |
| `--win` | `12` | FK_ALINK_WIN (default 12) |
| `--slide` | `2` | FK_ALINK_SLIDE (default 2) |
| `--iou-thr` | `0.3` | FK_ALINK_IOU_THR (default 0.3) |
| `--target-fpr` | `2.0` | FK_ALINK_TARGET_FPR (default 2.0) |
| `--rate-dev` | `2.0` | FK_ALINK_RATE_DEV (default 2.0) |
| `--hi-nspk` | `300` | FK_ALINK_HI_NSPK (default 300) |
| `--max-gap` | `4` | FK_ALINK_MAX_GAP (default 4) |
| `--spk-cap` | `1500` | FK_ALINK_SPK_CAP (default 1500) |
| `--sample-keyed` | `1` | FK_ALINK_SAMPLE_KEYED (default 1) |
| `--warp-thr` | `0.0` | FK_ALINK_WARP_THR (default 0.0) |
| `--amp-thr` | `0.85` | FK_ALINK_AMP_THR (default 0.85) |
| `--min-frag` | `15` | FK_ALINK_MIN_FRAG (default 15) |
| `--feat-lag` | `0` | FK_ALINK_FEAT_LAG (default 0) |
| `--cons-mode` | `off` | FK_ALINK_CONS_MODE > FK_CONS_MODE: which consolidation passes run (default off) — choices: `off`, `strip`, `knn`, `both`, `shed` |
| `--cons-max-dist` | `0.5` | FK_CONS_MAX_DIST (default 0.5) |
| `--cons-gmin` | `0.0` | FK_CONS_GMIN (default 0.0) |
| `--cons-gmax` | `10.0` | FK_CONS_GMAX (default 10.0) |
| `--cons-chan` | `0.0` | FK_CONS_CHAN (default 0.0) |
| `--cons-margin` | `0.85` | FK_CONS_MARGIN (default 0.85) |
| `--cons-own-q` | `0.9` | FK_CONS_OWN_Q (default 0.9) |
| `--cons-tgt-q` | `0.9` | FK_CONS_TGT_Q (default 0.9) |
| `--cons-tgt-scale` | `1.25` | FK_CONS_TGT_SCALE (default 1.25) |
| `--cons-twin` | `0.93` | FK_CONS_TWIN (default 0.93) |
| `--cons-tpl-cap` | `1024` | FK_CONS_TPL_CAP (default 1024) |
| `--cons-min-tpl` | `8` | FK_CONS_MIN_TPL (default 8) |
| `--cons-iters` | `1` | FK_CONS_ITERS (default 1) |
| `--cons-cap-amp` | `0.0` | FK_CONS_CAP_AMP (default 0.0) |
| `--cons-pure-iqr` | `0.0` | FK_CONS_PURE_IQR (default 0.0) |
| `--cons-agg-iqr` | `0.0` | FK_CONS_AGG_IQR (default 0.0) |
| `--cons-agg-every` | `3` | FK_CONS_AGG_EVERY (default 3) |
| `--cons-agg-minnew` | `30` | FK_CONS_AGG_MINNEW (default 30) |
| `--cons-agg-vac` | `0` | FK_CONS_AGG_VAC (default 0) |
| `--cons-tpl-align` | `0` | FK_CONS_TPL_ALIGN (default 0) |
| `--cons-park-min` | `0` | FK_CONS_PARK_MIN (default 0) |
| `--cons-shed-k` | `3.0` | FK_CONS_SHED_K (default 3.0) |
| `--cons-shed-min` | `40` | FK_CONS_SHED_MIN (default 40) |
| `--cons-knn-k` | `20` | FK_CONS_KNN_K (default 20) |
| `--cons-knn-thr` | `0.3` | FK_CONS_KNN_THR (default 0.3) |
| `--cons-knn-minref` | `50` | FK_CONS_KNN_MINREF (default 50) |
| `--cons-knn-minnew` | `30` | FK_CONS_KNN_MINNEW (default 30) |
| `--cons-knn-dims` | `16` | FK_CONS_KNN_DIMS (default 16) |
| `--cons-fold-thr` | `0.9` | FK_CONS_FOLD_THR (default 0.9) |
| `--cons-scorr` | `0.93` | FK_CONS_SCORR (default 0.93) |
| `--cons-off-thr` | `0.0` | FK_CONS_OFF_THR (default 0.0) |
| `--cons-resid-k-on` | `0.0` | FK_CONS_RESID_K_ON (default 0.0) |
| `--cons-resid-k-off` | `0.0` | FK_CONS_RESID_K_OFF (default 0.0) |
| `--cons-resid-win` | `0` | FK_CONS_RESID_WIN (default 0) |
| `--cons-resid-mingain` | `0.25` | FK_CONS_RESID_MINGAIN (default 0.25) |
| `--cons-resid-min` | `200` | FK_CONS_RESID_MIN (default 200) |
| `--cons-resid-iqr-max` | `0.0` | FK_CONS_RESID_IQR_MAX (default 0.0) |
| `--cons-resid-iqr-min` | `0.0` | FK_CONS_RESID_IQR_MIN (default 0.0) |
| `--cons-resid-align` | `chunk` | FK_CONS_RESID_ALIGN: which alignment the residual strip scores in (default chunk) — choices: `chunk`, `cluster`, `stored` |
### `fiber-backbone-link`

Link fiber-session fragments across chunks on the invariant backbone (median+/-sigma CI-overlap) with the Omlor-Giese warp veto.

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` | fragment .clu feature space (before the group) |
| `--clu-stage` (`--variant`, `--in`) | `fiber_session` | fragment .clu stage tag (the fiber-session output) |
| `--in-clu` | — | explicit fragment .clu path (overrides --clu-method/--clu-stage) |
| `--spk-method` (`--spk-variant`) | `standard` | waveform axis for templates/warp (standard = curation axis) |
| `--channels` | — | pin backbone channels (global ids, e.g. 33,34); default = per-pair shared primary |
| `--out-stage` (`--out-tag`, `--out`) | `backbone_linked` | post-fiber stage tag of the output .clu (single token) |
| `--gt-stage` (`--gt-clu`) | — | post-fiber stage tag (or path) of the curated .clu to score purity+completeness against |
| `--gt-res` | — | reserved: .res for the GT (unused when GT shares the session res) |
| `--spk-cap` | `600` | spikes per fragment for the template |
| `--chunk-min` | — | chunk length (min); default from <session>.yaml or 12 |
| `--seed` | `0` |  |
| `--drift-fibers` | — | path to the .fibers npz carrying fiber-session's overlap-anchor drift transform (default: derive from --clu-method); only consulted when --drift=1 |
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
| `--drift` | `0` | FK_BBLINK_DRIFT (default 0) |
| `--drift-recenter` | `1` | FK_BBLINK_DRIFT_RECENTER (default 1) |
| `--hierarchy` | `1` | FK_BBLINK_HIERARCHY (default 1) |
### `fiber-xcorr-merge`

Confidence-ordered Klusters roll-shift cosine merge (realign after each merge).

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` |  |
| `--clu-stage` (`--variant`, `--in`) | `backbone_linked` | input .clu stage tag (e.g. the fiber-backbone-link output) |
| `--in-clu` | — | explicit input .clu path (overrides --clu-method/--clu-stage) |
| `--spk-method` (`--spk-variant`) | `standard` | waveform axis for templates (curation axis) |
| `--out-stage` (`--out-tag`, `--out`) | `xcorr_merged` | post-fiber stage tag of the output .clu |
| `--refrac-censor-ms` | `0.0` | detection censor window (ms) |
| `--nsamp` | — | override; default from <session>.yaml |
| `--nchan` | — | override; default from <session>.yaml |
| `--ref-sample` | — | override; default = peak from <session>.yaml |
| `--gt-stage` (`--gt-clu`) | — | post-fiber stage tag (or path) of the curated .clu to score purity+completeness |
| `--seed` | `0` |  |
| `--cos-thr` | `0.99` | FK_XCM_COS_THR (default 0.99) |
| `--shift` | `4` | FK_XCM_SHIFT (default 4) |
| `--refrac-ms` | `2.0` | FK_XCM_REFRAC_MS (default 2.0) |
| `--refrac-thr` | `0.3` | FK_XCM_REFRAC_THR (default 0.3) |
| `--refrac-min-exp` | `5.0` | FK_XCM_REFRAC_MIN_EXP (default 5.0) |
| `--min-n` | `40` | FK_XCM_MIN_N (default 40) |
| `--spk-cap` | `300` | FK_XCM_SPK_CAP (default 300) |
| `--sample-keyed` | `1` | FK_XCM_SAMPLE_KEYED (default 1) |
| `--complexity-scale` | `0.0` | FK_XCM_CX_SCALE (default 0.0) |
| `--band-thr` | `0.5` | FK_XCM_BAND_THR (default 0.5) |
| `--off-thr` | `0.0` | FK_XCM_OFF_THR (default 0.0) |
| `--off-amp-frac` | `0.3` | FK_XCM_OFF_AMP_FRAC (default 0.3) |
| `--merge-align` | `peak` | FK_XCM_MERGE_ALIGN (default peak) |
| `--merge-align-min-score` | `0.1` | FK_XCM_MERGE_ALIGN_MIN_SCORE (default 0.1) |
| `--giqr-thr` | `0.0` | FK_XCM_GIQR_THR (default 0.0) |
| `--giqr-rise` | `0.0` | FK_XCM_GIQR_RISE (default 0.0) |
| `--rvu-thr` | `0.0` | FK_XCM_RVU_THR (default 0.0) |
| `--rvu-rise` | `0.0` | FK_XCM_RVU_RISE (default 0.0) |
| `--rvu-win` | `12` | FK_XCM_RVU_WIN (default 12) |
| `--rvu-knn` | `0` | FK_XCM_RVU_KNN (default 0) |
| `--rvu-knn-k` | `10` | FK_XCM_RVU_KNN_K (default 10) |
| `--rvu-knn-vote` | `0.6` | FK_XCM_RVU_KNN_VOTE (default 0.6) |
| `--rvu-knn-refs` | `8` | FK_XCM_RVU_KNN_REFS (default 8) |
| `--rvu-knn-fold` | `0.9` | FK_XCM_RVU_KNN_FOLD (default 0.9) |
| `--rvu-knn-max` | `20000` | FK_XCM_RVU_KNN_MAX (default 20000) |
| `--emit-drift` | `1` | FK_XCM_EMIT_DRIFT (default 1) |
| `--drift-k` | `6` | FK_XCM_DRIFT_K (default 6) |
| `--drift-chunk-min` | `12.0` | FK_XCM_DRIFT_CHUNK_MIN (default 12.0) |
| `--chunks` | `""` | FK_CHUNKS: process only these 0-based chunks, e.g. 14-16 or 3,7 ('' = all); the other chunks' spikes go to the reserved clusters 16777116+chunk |
| `--chunk-length` | `12.0` | FK_SESSION_CHUNK_MIN: chunk length in minutes that --chunks counts in |
### `fiber-residual-merge`

Confidence-ordered reverse-residual-strip merge (re-template after each merge).

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` |  |
| `--clu-stage` (`--variant`, `--in`) | `backbone_linked` | input .clu stage tag (e.g. the linker's output) |
| `--in-clu` | — | explicit input .clu path (overrides --clu-method/--clu-stage) |
| `--spk-method` (`--spk-variant`) | `standard` | waveform axis for templates and residuals (RAW: the strip was measured there) |
| `--out-stage` (`--out-tag`, `--out`) | `resid_merged` | post-fiber stage tag of the output .clu |
| `--refrac-censor-ms` | `0.0` | detection censor window (ms) |
| `--nsamp` | — | override; default from <session>.yaml |
| `--nchan` | — | override; default from <session>.yaml |
| `--ref-sample` | — | override; default = peak from <session>.yaml |
| `--gt-stage` (`--gt-clu`) | — | post-fiber stage tag (or path) of the curated .clu to score purity+completeness |
| `--seed` | `0` |  |
| `--replay` | — | the .rsm table of an earlier run on the SAME input at a LOWER --acc-thr and otherwise the same settings: this run is that one truncated at --acc-thr (exact), and scores nothing. One run at e.g. 0.4, then any threshold above it for free |
| `--acc-thr` | `0.675` | FK_RSM_ACC_THR (default 0.675) |
| `--cos` | `0.5` | FK_RSM_COS (default 0.5) |
| `--shift` | `2` | FK_RSM_SHIFT (default 2) |
| `--strip-k-on` | `6.0` | FK_RSM_STRIP_K_ON (default 6.0) |
| `--strip-k-off` | `3.0` | FK_RSM_STRIP_K_OFF (default 3.0) |
| `--win` | `0` | FK_RSM_WIN (default 0) |
| `--mingain` | `0.25` | FK_RSM_MINGAIN (default 0.25) |
| `--refrac-ms` | `2.0` | FK_RSM_REFRAC_MS (default 2.0) |
| `--refrac-thr` | `0.3` | FK_RSM_REFRAC_THR (default 0.3) |
| `--refrac-min-exp` | `5.0` | FK_RSM_REFRAC_MIN_EXP (default 5.0) |
| `--min-n` | `40` | FK_RSM_MIN_N (default 40) |
| `--spk-cap` | `1000` | FK_RSM_SPK_CAP (default 1000) |
| `--sample-keyed` | `1` | FK_RSM_SAMPLE_KEYED (default 1) |
| `--merge-align` | `stored` | FK_RSM_MERGE_ALIGN (default stored) |
| `--merge-align-min-score` | `0.1` | FK_RSM_MERGE_ALIGN_MIN_SCORE (default 0.1) |
| `--emit-drift` | `1` | FK_RSM_EMIT_DRIFT (default 1) |
| `--drift-k` | `6` | FK_RSM_DRIFT_K (default 6) |
| `--drift-chunk-min` | `12.0` | FK_RSM_DRIFT_CHUNK_MIN (default 12.0) |
| `--chunks` | `""` | FK_CHUNKS: process only these 0-based chunks, e.g. 14-16 or 3,7 ('' = all); the other chunks' spikes go to the reserved clusters 16777116+chunk |
| `--chunk-length` | `12.0` | FK_SESSION_CHUNK_MIN: chunk length in minutes that --chunks counts in |
### `fiber-psame-merge`

Merge by a calibrated P(same cell) from history-conditioned cell models (docs/methods.md "P(same cell)").

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` |  |
| `--clu-stage` (`--in`) | `xcorr_merged` | input .clu stage tag |
| `--in-clu` | — | explicit input .clu path (overrides --clu-method/--clu-stage) |
| `--spk-method` (`--spk-variant`) | — | waveform variant the cell models are fitted on (default: the WAVEFORM token of --clu-method, e.g. stderiv_C5_D34 -> stderiv_C5; the shipped coefficients were calibrated on stderiv_C5) |
| `--out-stage` (`--out-tag`, `--out`) | `psame_merged` |  |
| `--cap` | — | spikes per cluster model (default: all) |
| `--seed` | `0` |  |
| `--chunk-min` | `12.0` | chunk length (min) for --per-chunk's fallback partition and --chunk-aware's chunks (default FK_SESSION_CHUNK_MIN) |
| `--theta` | `0.5` | FK_PSAME_THETA (default 0.5) |
| `--cannot-link` | `0.3` | FK_PSAME_CANNOT_LINK (default 0.3) |
| `--knn` | `8` | FK_PSAME_KNN (default 8) |
| `--cos-floor` | `0.75` | FK_PSAME_COS_FLOOR (default 0.75) |
| `--min-n` | `50` | FK_PSAME_MIN_N (default 50) |
| `--sample-keyed` | `1` | FK_PSAME_SAMPLE_KEYED (default 1) |
| `--shift` | `4` | FK_PSAME_SHIFT (default 4) |
| `--w-rc` | `22.2143` | FK_PSAME_W_RC (default 22.2143) |
| `--w-dbic` | `0.6144` | FK_PSAME_W_DBIC (default 0.6144) |
| `--bias` | `-17.773` | FK_PSAME_BIAS (default -17.773) |
| `--geo` | `0` | FK_PSAME_GEO (default 0) |
| `--geo-w-rc` | `21.5792` | FK_PSAME_GEO_W_RC (default 21.5792) |
| `--geo-w-dbic` | `0.5606` | FK_PSAME_GEO_W_DBIC (default 0.5606) |
| `--geo-w-z` | `-0.4395` | FK_PSAME_GEO_W_Z (default -0.4395) |
| `--geo-bias` | `-17.0015` | FK_PSAME_GEO_BIAS (default -17.0015) |
| `--per-chunk` | `0` | FK_PSAME_PER_CHUNK (default 0) |
| `--chunk-aware` | `0` | FK_PSAME_CHUNK_AWARE (default 0) |
| `--min-chunk-n` | `30` | FK_PSAME_MIN_CHUNK_N (default 30) |
| `--max-gap` | `1` | FK_PSAME_MAX_GAP (default 1) |
| `--align-pairs` | `1` | FK_PSAME_ALIGN_PAIRS (default 1) |
| `--scale-veto` | `0` | FK_PSAME_SCALE_VETO (default 0) |
| `--scale-thr` | `0.2` | FK_PSAME_SCALE_THR (default 0.2) |
| `--scale-min-n` | `100` | FK_PSAME_SCALE_MIN_N (default 100) |
| `--w-sresid` | `0.0` | FK_PSAME_W_SRESID (default 0.0) |
| `--sresid-ref` | `split` | FK_PSAME_SRESID_REF (default split) |
| `--sresid-ref-k` | `1.0` | FK_PSAME_SRESID_REF_K (default 1.0) |
| `--sresid-ref-max` | `0.0` | FK_PSAME_SRESID_REF_MAX (default 0.0) |
| `--sresid-min-half` | `8` | FK_PSAME_SRESID_MIN_HALF (default 8) |
### `fiber-relink`

Geometry-aware re-bundling/re-linking of a .fibers run (no re-run needed).

Positional: `fibers`

| flag | default | description |
|---|---|---|
| `--clu` | — | existing .clu to remap (the .fibers' clu_id names each fiber's cluster; 0=noise) |
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
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
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

## Positions & geometry

### `fiber-cpos`

Write a per-spike cluster-position sidecar (.cpos) by localizing each cluster's median RAW template (monopole+dipole).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
| `--spk` | — | path to a STANDARD/raw .spk (preferred over .fil); never the stderiv .spkD |
| `--spk-method` | `standard` | method of the raw .spk to resolve: <base>.spk.<spk-method>.<elec> |
| `--fil` | — | path to raw .fil (fallback if no standard .spk) |
| `--fil-offset` | `0` | first absolute sample of the .fil (0 for full recording) |
| `--clu-method` (`--method`) | `stderiv` | source-clu feature space BEFORE the group (standard\|stderiv\|...) |
| `--clu-stage` (`--in`) | `refine` | source-clu fiber STAGE AFTER the group: read <base>.clu.<clu-method>.<elec>.<clu-stage> |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--clu-stage) |
| `--out-method` | — | cpos method BEFORE the group (default: mirror --clu-method) |
| `--out-stage` (`--out`) | — | cpos fiber STAGE AFTER the group (default: mirror --clu-stage) |
| `--min-spikes` | `15` |  |
| `--max-spikes` | `20000` | spikes per cluster its template and position are taken from (seeded random sample; 0 = all). Bounds memory: a 380k-spike cluster in float64 alone is ~1.3 GB |
| `--no-dipole` | flag (off) |  |
| `--no-templates` | flag (off) | skip per-cluster median templates in the .clusters.npz |
| `--amp-algo` (`--amp-method`) | `pc1` | per-channel amplitude profile for the position inverse: pc1=rank-1 denoised template (default, sharpest footprint + most precise), wave=median-waveform ptp, ptp=median per-spike ptp (legacy; ~4-sigma noise floor on far channels flattens the footprint). — choices: `pc1`, `wave`, `ptp` |
| `--amp-basis` | `auto` | amplitude basis the gate-facing positions use: 'pca'=read .pca.standard.<elec> (PC1 score per channel = the .fet amplitude); 'fit'=group basis from .spk; 'auto'=pca if present else fit; 'none'=per-cluster SVD — choices: `auto`, `pca`, `fit`, `none` |
| `--no-amp-basis` | flag (off) | alias for --amp-basis none (per-cluster SVD) |
| `--nboot` | `0` | bootstrap draws for the depth/distance percentile CIs (z_lo/z_hi/y_lo/y_hi). This loop is ~5x the rest of the cost (the dominant runtime); positions (x0,y0,z0,A) and the energy-tercile depth-shift do NOT use it. Use --nboot 0 for identical positions ~5x faster (analytic sig_y is still written; the percentile CIs become NaN). |
| `--probe` | — | probe file(s) for geometry (else from chunk xy via YAML) |
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
| `--session` | — | session .yaml for channel/sr/sdiffPairs lookup |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from --session) |
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
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
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
### `fiber-chan-svd`

Per-channel SVD/PCA of cluster mean templates: which channels are invariant vs vary across the clusters (a merge/curation aid).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
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
### `fiber-adlp`

Amplitude-dependent low-pass of the spike snippets, written as a new waveform variant: every spike gets a zero-phase low-pass whose cutoff is a power law of its amplitude, `fc = clip(fc_ref * (A / A_ref) ** gamma, fc_min, fc_max)`, so small, noise-dominated spikes are smoothed and large ones keep their detail.  Writes `.spk.<out>.<g>`, a `.pca.<out>.<g>` fitted on the filtered snippets in the input basis's window, and `.adlp.<out>.<g>.npz` (per-spike amplitude and cutoff).  Sort the variant with `fiber-session --method <out> --whiten-source spk`: the noise of a per-spike filtered variant is not that of any `.fil` or `.nst`.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--in-variant` | `standard` | waveform variant to filter (exact token; default standard) |
| `--out-variant` | `standard_ADLP` | variant token to write |
| `--fc-ref` | `9000.0` | cutoff (Hz) at A = A_ref. Default 9000: where the median g7 standard spike's spectrum meets the pre-peak noise floor |
| `--a-ref` | — | reference amplitude (default: file median ptp) |
| `--gamma` | `1.2` | cutoff exponent: fc ~ A**gamma (0 = uniform low-pass at --fc-ref) |
| `--fc-min` | `2500.0` | lowest cutoff (Hz) |
| `--fc-max` | — | highest cutoff (Hz); default Nyquist, so large spikes pass unfiltered |
| `--order` | `2` | Butterworth order (applied forward+backward) |
| `--scope` | `spike` | spike = one cutoff from the largest channel; channel = a cutoff per channel — choices: `spike`, `channel` |
| `--amp-lo` | `-6` | amplitude window start, samples from peak |
| `--amp-hi` | `10` | amplitude window end, samples from peak |
| `--block` | `40000` | spikes per processing block |
| `--no-pca` | flag (off) | do not fit/write .pca.<out>.<g> |
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

## Curation, QC & scoring

### `fiber-refit`

Refit the fiber model (per-unit, per-chunk templates / positions / signatures) to a MANUALLY CURATED .clu, taking the curator's grouping as final (no re-merge/split).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--clu-method` (`--method`) | `stderiv` | feature space of the curated clu |
| `--clu-stage` (`--variant`, `--in`) | `curated` | stage tag of the curated .clu to refit (e.g. 'curated') |
| `--in-clu` | — | explicit curated .clu path |
| `--cpos-method` | — | cpos method for positions (default: --clu-method) |
| `--cpos-stage` | — | cpos stage for positions (default: --variant) |
| `--relocalize` | flag (off) | re-localize the curated units here from raw .spk/.fil instead of reading a cpos table |
| `--spk-method` | `standard` | raw .spk method for --relocalize |
| `--gate` | `cfiber` | shape descriptor to attach to each signature (default cfiber) — choices: `cfiber`, `wave`, `none` |
| `--chunk-minutes` (`--chunk-min`) | `12.0` |  |
| `--min-n` | `12` | min spikes for a per-chunk signature |
| `--out-stage` (`--out`) | — | stage tag for the refit units (default '<variant>_refit') |
### `fiber-reject`

Reassign per-cluster outlier spikes to a better-fitting cluster or to noise.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--peak` | `16` | peak sample index within the window |
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
### `fiber-stats`

Extract per-(chunk,cluster) fiber statistics from an existing sort (no re-clustering).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from YAML) |
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
### `fiber-qc`

Per-group QC report (rate, ISI violation, SNR, amplitude, presence) rendered as an interactive HoloViz/Bokeh HTML, with a metrics CSV.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--clu-method` | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
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
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
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
### `fiber-lowmix`

Split low-amplitude units that are mixtures of sources.  After noise whitening a unit at or below
`--amp-quantile` is split into groups; a group that copies the shape of another unit is returned to
it, a group that copies nothing is quarantined (to the noise, kept as `.clc` children, or as a unit
of its own: `--novel`).  Writes `.<in>_lowmix` and the `.lmx` table.

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--signal` | — | continuous filtered int16 file (BASE.fil) for the noise |
| `--signal-start` | `0` | absolute sample of the signal's first row |
| `--total-channels` | — |  |
| `--channels` | — | group channels, e.g. 40-47 (default: session yaml) |
| `--clu-method` (`--method`) | `stderiv_C5_D34` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` (`--in`) | `""` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--spk-variant` | `standard` | waveforms to split on (raw: the amplitude law holds) |
| `--res-variant` | `""` | pin the .res variant ('' = the shared .res) |
| `--units` | — | comma-separated units to examine (overrides the amplitude pick) |
| `--amp-quantile` | `0.3333333333333333` | units at or below this quantile of peak-to-peak amplitude (among units with >= --min-n spikes) are examined; 1 = every unit [FK_LOWMIX_AMP_QUANTILE; default 0.3333333333333333] |
| `--min-n` | `150` | spikes a unit needs to be examined [FK_LOWMIX_MIN_N; default 150] |
| `--min-group` | `30` | smallest group acted on; smaller groups stay with the unit [FK_LOWMIX_MIN_GROUP; default 30] |
| `--deg` | `20` | Legendre degree per channel (20 for 42 samples: split-half optimum) [FK_LOWMIX_DEG; default 20] |
| `--k-max` | `4` | most groups per unit [FK_LOWMIX_K_MAX; default 4] |
| `--null-mult` | `2.0` | split only if BIC gain > NULL_MULT * null gain + NULL_ADD [FK_LOWMIX_NULL_MULT; default 2.0] |
| `--null-add` | `0.02` | see NULL_MULT (BIC gain per spike) [FK_LOWMIX_NULL_ADD; default 0.02] |
| `--noise-n` | `20000` | spike-free .fil snippets for the noise covariance [FK_LOWMIX_NOISE_N; default 20000] |
| `--copy-cos` | `0.95` | shape cosine that makes a group a copy of another unit [FK_LOWMIX_COPY_COS; default 0.95] |
| `--novel-cos` | `0.9` | below this a non-core group matches no unit: quarantined (--novel noise) or kept [FK_LOWMIX_NOVEL_COS; default 0.9] |
| `--max-shift` | `6` | +-samples searched when matching templates [FK_LOWMIX_MAX_SHIFT; default 6] |
| `--template-min-n` | `100` | spikes a unit needs to serve as a match template [FK_LOWMIX_TEMPLATE_MIN_N; default 100] |
| `--refrac-ms` | `1.5` | a copy returns only if its coincidences with the target in (0.6, REFRAC] ms stay below max(0.3, 1.5 x the target's own ratio), against the local 3-6 ms rate (0 = off: shape alone decides) [FK_LOWMIX_REFRAC_MS; default 1.5] |
| `--refrac-min-exp` | `3.0` | coincidences that must be expected before the refractory test may veto [FK_LOWMIX_REFRAC_MIN_EXP; default 3.0] |
| `--copy-on-abstain` | `0` | 1 = a shape copy also returns when the refractory test has no power; 0 = it stays its own unit (measured on g6: shape-only copies sit at 0.68 of the local rate against their target -- half of them are not the same neuron) [FK_LOWMIX_COPY_ON_ABSTAIN; default 0] |
| `--novel` | `noise` | noise = novel groups to noise (1), as a reviewable .clc child; keep = each novel group becomes its own unit [FK_LOWMIX_NOVEL; default noise] |
| `--seed` | `0` | seed for subsampling, noise snippets and the GMM [FK_LOWMIX_SEED; default 0] |
| `--out-tag` (`--out`) | — | output stage tag (default <clu-stage>_lowmix) |
### `fiber-rdr`

Per-cluster contamination score from the RMS of the residual's temporal derivative, against the clean amplitude line (a QC table; moves no spike).

Each cluster's raw windows are aligned as Klusters' realign does (threshold 0.10), a median template is
taken per `--chunk-min` chunk, and D = RMS over the window of diff(spike − template) is computed per spike
and channel, in units of the noise's derivative RMS (exact from the group's `.nst`, else from the
windows' lead-in).  A clean cell's D still grows with the channel's amplitude A, as
median D² = D0² + k·A² (g6 0316 clean units: 1.105 + 0.00153·A²).  The line is the lower envelope of every
cluster's channels, a quantile regression at `--fit-quantile` (0.25), so it needs no labels and holds when
most clusters are dirty (`--fit clip` is the 0.53.0 least-squares fit; `--line` fixes it).  The score is the largest
observed/expected excess, √(D² − D0²) / (√k·A), over channels of at least `--min-amp` SD; clean
clusters sit near 1.  Measured: g6 0316, dirty units 1.77–2.57 against clean ≤ 1.65; g7 0312 against the
old curation (below vs at least 90% purity), AUC 0.87 / 0.79 / 0.85 on three sorts, and `--thr 2.0` flags
50–78% of the mixed and 9–22% of the pure clusters.  Knobs `FK_RDR_*`; step `rdr` of the exp plan.  Writes
`<base>.rdr.<clu-method>.<group>[.<stage>].tsv`: cluster, n, largest amplitude, score, its channel, flag,
and per-channel amplitude and median D.

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` |  |
| `--clu-stage` (`--in`) | `""` | post-fiber stage tag of the .clu to score |
| `--in-clu` | — | explicit .clu path (overrides --clu-method/--clu-stage) |
| `--spk-method` | `standard` | waveform axis (raw windows; default standard) |
| `--nst` | — | .nst path (default <base>.nst.<spk-method>.<elec> if present) |
| `--chunk-min` | `12.0` | FK_RDR_CHUNK_MIN: one template per chunk of this length, minutes (0: one per cluster) |
| `--min-n` | `40` | FK_RDR_MIN_N: clusters with fewer spikes are skipped |
| `--cap` | `4000` | FK_RDR_CAP: spikes sampled per cluster |
| `--min-amp` | `4.0` | FK_RDR_MIN_AMP: channels below this amplitude (noise SD) are not scored |
| `--align-min-score` | `0.1` | FK_RDR_ALIGN_MIN_SCORE: Klusters realign threshold (Klusters' own: 0.70) |
| `--fit` | `quantile` | FK_RDR_FIT: clean-line fit -- quantile (lower envelope) or clip (least squares, 3-MAD clip) — choices: `quantile`, `clip` |
| `--fit-quantile` | `0.25` | FK_RDR_FIT_QUANTILE: quantile of the quantile-regression fit |
| `--thr` | `2.0` | FK_RDR_THR: flag clusters whose ratio reaches this |
| `--line` | `""` | FK_RDR_LINE: fixed clean line D0SQ,K instead of fitting it |
| `--out` | — | output TSV (default <base>.rdr.<clu-method>.<elec>[.<stage>].tsv) |
| `--seed` | `0` |  |
| `--chunks` | `""` | FK_CHUNKS: process only these 0-based chunks, e.g. 14-16 or 3,7 ('' = all); the other chunks' spikes go to the reserved clusters 16777116+chunk |
| `--chunk-length` | `12.0` | FK_SESSION_CHUNK_MIN: chunk length in minutes that --chunks counts in |
### `fiber-manifold`

The off-spike-channel field manifold of a unit, in sliding windows, as children and as a feature space
(a refinement stage; moves no spike between units).  After the unit's own template is subtracted,
the channels it has little power on still carry the fields of neighbouring cells that fired within the
window (before / with / after the spike) and the unit's own far field -- structure the per-channel PCA
features give no budget.  Per spike one field event is fitted on those channels,
A·g((t−τ)/w)·exp(−|ch−p|/λ) (g the unit's own dominant-channel shape; a matched filter, noise-whitened
on the off channels from the group's `.nst` or `--signal`'s spike-free snippets, a coarse grid of lag ×
position × decay then a refinement), in sliding windows of `--window-min` every `--step-min`, each with
its own template (the per-sample median of the window's best-fitting spikes, so the unit's drift is
followed and a minority's fields do not leak into it).  `--channels-set full` fits the same
model over every channel of the group instead (the unit's own misfit and collisions become fields;
position over the whole group).  Children: time bin × class (plain, or position
bin × before / coincident / after), at most `--max-children` per unit.  Writes `.clu` (unchanged),
`.clc`/`.clp` at `<clu-stage>_manifold` (or `--out-tag`), `.fet.manifold.<g>` -- snr×100 | lag_µs |
position×100 | decay×100 | amplitude | window | class | isi_flag | amp_dom | time, a feature space
Klusters lists under *Space* -- and `.mft.<method>.<g>.<tag>.tsv` (children and windows).  The ISI
flag marks, of each pair closer than `--refrac-ms`, the member farther from the unit's median waveform.
Units run in parallel (`--jobs`, one process each); `FIBER_KIT_GPU=1` puts the matched-filter products on
the GPU (CuPy).  g6 0312 unit 5 (311k spikes, 67 windows): 19% of spikes carry a field at SNR ≥ 4
(noise 0.7%), 20 s serial.  Knobs `FK_MANIFOLD_*`; profile `klusters-manifold` runs it alone on a
Klusters selection (`fk install klusters-plugin`: the `fiber-kit-manifold` plugin).

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--signal` | — | continuous filtered int16 file (BASE.fil) for per-window noise; default: the group's .nst, else the unit's own baseline |
| `--signal-start` | `0` | absolute sample of the signal's first row |
| `--total-channels` | — |  |
| `--channels` | — | group channels, e.g. 40-47 (default: session yaml) |
| `--clu-method` (`--method`) | `stderiv_C5_D34` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` (`--in`) | `""` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--res-variant` | `""` | pin the .res variant ('' = the shared .res) |
| `--units` | — | comma-separated units to examine (default: every unit with >= --min-n) |
| `--off-channels` | — | off-spike channels as group indices, e.g. 3,4,5,6,7 (default: by --off-frac) |
| `--window-min` | `20.0` | sliding window length (minutes) [FK_MANIFOLD_WINDOW_MIN; default 20.0] |
| `--step-min` | `5.0` | window step (minutes); each spike takes the nearest window's fit [FK_MANIFOLD_STEP_MIN; default 5.0] |
| `--min-n` | `2000` | spikes a unit needs to be examined [FK_MANIFOLD_MIN_N; default 2000] |
| `--min-window-n` | `300` | spikes a window needs for its own template; fewer: the neighbour's [FK_MANIFOLD_MIN_WINDOW_N; default 300] |
| `--spk-variant` | `standard` | the waveform space the fields are fitted in: any <base>.spk.<variant>.<g> (standard = the raw waveforms; stderiv, a whitened space...). The noise covariance comes from the matching .nst, else --signal, else white [FK_MANIFOLD_SPK; default standard] |
| `--channels-set` | `off` | channels the field is fitted on: off = the off-spike channels (by --off-frac / --off-channels), full = every channel of the group [FK_MANIFOLD_CHANNELS; default off] — choices: `off`, `full` |
| `--off-frac` | `0.4` | a channel is off-spike when its template trough is below this fraction of the deepest channel's (used when --off-channels is unset) [FK_MANIFOLD_OFF_FRAC; default 0.4] |
| `--min-off` | `3` | at least this many off-spike channels (the shallowest are added) [FK_MANIFOLD_MIN_OFF; default 3] |
| `--snr` | `4.0` | field present when the matched-filter SNR (noise-sd units) is at least this [FK_MANIFOLD_SNR; default 4.0] |
| `--lag-ms` | `0.9` | largest \|lag\| searched (ms) [FK_MANIFOLD_LAG_MS; default 0.9] |
| `--dead-ms` | `0.6` | \|lag\| below this is 'coincident' (the detector's dead time) [FK_MANIFOLD_DEAD_MS; default 0.6] |
| `--pos-bins` | `3` | position bins over the off-channel span (classes = bins x 3 timings) [FK_MANIFOLD_POS_BINS; default 3] |
| `--max-children` | `200` | most children per unit (time bins coarsen to fit) [FK_MANIFOLD_MAX_CHILDREN; default 200] |
| `--time-bins` | `12` | time bins the plain class is split into (0 = one plain child) [FK_MANIFOLD_TIME_BINS; default 12] |
| `--refrac-ms` | `2.0` | ISI below this flags the pair's farther-from-median member (0 = off) [FK_MANIFOLD_REFRAC_MS; default 2.0] |
| `--noise-n` | `20000` | noise snippets for the SNR calibration (drawn from the .nst or the .fil) [FK_MANIFOLD_NOISE_N; default 20000] |
| `--write-fet` | `1` | write the .fet.manifold.<g> feature space (1) or not (0) [FK_MANIFOLD_WRITE_FET; default 1] |
| `--seed` | `0` | random seed (noise draws) [FK_MANIFOLD_SEED; default 0] |
| `--jobs` | `0` | units examined in parallel (processes); 0 = one per CPU, 1 = serial [FK_MANIFOLD_JOBS; default 0] |
| `--out-tag` (`--out`) | — | output stage tag (default <clu-stage>_manifold) |
| `--fet-method` | `manifold` | method token of the written feature space |
### `fiber-flag-artifacts`

Flag acquisition artifacts before clustering, from the unwhitened temporal derivative.  Writes an
exclusion mask `.clu` (`<base>.clu.<spk-method>.<elec>.<out-stage>`) with one label per spike of the
`.res`; `fiber-session --exclude-clu` leaves the flagged spikes out of the sort and emits them as
cluster 0.  `fiber-flag-collisions` adds to the same mask, so it runs after this stage.

Positional: `args`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--spk-method` (`--spk-variant`) | `standard` | waveform axis (standard = the unwhitened curation axis these features assume) |
| `--out-stage` (`--out-tag`, `--out`) | `artifact` | stage tag of the emitted mask .clu |
| `--dry-run` | flag (off) | report the flagged count, write nothing |
| `--chunks` | `""` | FK_CHUNKS: process only these 0-based chunks, e.g. 14-16 or 3,7 ('' = all); the other chunks' spikes go to the reserved clusters 16777116+chunk |
| `--chunk-length` | `12.0` | FK_SESSION_CHUNK_MIN: chunk length in minutes that --chunks counts in |
| `--hf-frac` | `0.9` | FK_FLAG_HF_FRAC (default 0.9) |
| `--d-crest` | `0.0` | FK_FLAG_D_CREST (default 0.0) |
| `--sat-adu` | `32000` | FK_FLAG_SAT_ADU (default 32000) |
| `--batch` | `50000` | FK_FLAG_BATCH (default 50000) |
### `fiber-flag-collisions`

Flag probable collisions BEFORE clustering (0.71.0), so the sort does not spend its effort on
windows that hold two spikes -- they sit between their two units and pull the clusters together --
and `fiber-unmix` explains them afterwards against units built without them.

The score needs only the `.spk`: `p(s)` = the noise-normalised energy over the channels at sample
`s`; spikes are binned by their peak energy, and a spike's score is how far, in MADs of its bin,
its `p(s)` rises above the bin's median anywhere more than `--guard` samples from the peak.  A
spike scoring above `--score-min` is flagged; so is, whatever its score, a spike with another
detection within `--neighbour` samples.  The threshold is on the score, not a quantile of it, so it
does not depend on how many collisions a group holds; `--max-share` caps the flagged share.

Measured on sirotaA-jg-000005-20120312 group 8 (815,728 spikes; known two-spike windows = the 7,418
spikes with another detection 9-26 samples away; kept / noise = what `fiber-clean` later made of
the spike), the score alone:

| `--score-min` | of all spikes | of the known pairs | of kept, isolated spikes | of noise |
|---|---|---|---|---|
| 8 | 10.7 % | 81 % | 3.5 % | 16.5 % |
| 12 (default) | 5.8 % | 66 % | 1.5 % | 9.4 % |
| 15 | 3.9 % | 55 % | 0.9 % | 6.5 % |
| 20 | 2.2 % | 41 % | 0.4 % | 3.8 % |

With the neighbour rule 6.2 % of that group is flagged.  Those known pairs are collisions in which
BOTH spikes were detected; how often a collision with a smaller, undetected partner is flagged was
not measured there (on the test fixture, partners at 0.6-1.1 of their template: 99 %).  A partner
within `--guard` samples of the peak is not seen.

What it buys is the flagged spikes kept out of the units, not a better sort of the rest: on
chunks 2-5 of that group `fiber-session` reached 76.5 % time-local purity without the flagged
spikes and 76.4 % when they were sorted along and dropped afterwards (75.0 % with them in).

Output: the exclusion mask `<base>.clu.<spk-method>.<group>.<out-stage>` -- by default the one
`fiber-flag-artifacts` writes, kept and added to (collisions as 2).  With `FK_SESSION_EXCLUDE=1`
`fiber-session` leaves masked spikes out of every chunk and emits them as cluster 0;
`fiber-unmix --noise-ids 0,1` then explains them.  Run it after `fiber-flag-artifacts`.

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--spk-method` (`--spk-variant`) | `standard` | waveform file the score is taken on (default standard) |
| `--out-stage` (`--out-tag`, `--out`) | `artifact` | stage tag of the mask .clu (default artifact: fiber-flag-artifacts' mask, which is kept and added to) |
| `--score-min` | `12.0` | a spike whose off-spike score exceeds this is flagged (default 12, in MADs of its peak-energy bin; the module header has the measured rates for 8-20) |
| `--max-share` | `0.0` | most of the spikes the score may flag: the threshold is raised to keep to it (default 0 = no cap) |
| `--guard` | `8` | samples either side of the peak the score ignores |
| `--neighbour` | `-1` | a spike with another detection within this many samples is flagged whatever its score (default -1 = half a window; 0 = off) |
| `--bins` | `20` | peak-energy bins the expected tail is taken in |
| `--dry-run` | flag (off) | report the flagged counts, write nothing |
### `fiber-clean`

Keep the clean part of a sort (0.69.0): the two Klusters tools as one pass over a flat `.clu`.

1. **Core cut** (`fiber_corecut`, a numpy port of Klusters' "Split Off Spurious Spikes",
   `corecut.h`) on every cluster of at least `--cc-min` spikes, on the RAW `.spk.standard` windows:
   the spikes past the cut go to the noise cluster.
2. **Quality value** of what is left (`fiber_corecut.residual_variance`, the waveform view's quality
   dot), on the session's method `.spk`: a cluster at or above `--q-dirty` goes to the noise whole.
3. A cluster left with fewer than `--min-keep` spikes, and every cluster in `--junk`, goes to the noise.

Cluster 0, the noise cluster and reserved clusters are left as they are.  Writes a flat `.clu` (the
`.clc`/`.clp` hierarchy is not rebuilt) and `<base>.clean.<method>.<g>.<out>.tsv` (cluster, spikes
in, spikes kept, the cut's channel, quality value, kept | passed | dirty | small | junk).

It trades yield for purity.  On sirotaA-jg-000005-20120312 group 8 (fs_polish, against the curated
file, purity within 10-minute bins): purity 93.0 -> 95.8 %, clusters >= 90 % pure 57 -> 66 %,
curated spikes held 77.1 -> 53.7 %.  The cut took 29 % of the kept clusters' spikes, most of them
spikes of the same unit -- what `fiber-unmix` is for.  The loop:

```yaml
pipeline:
  - {stage: fiber-clean,   in: fs_polish,       out: fs_clean, params: {junk: "<polish residue id>"}}
  - {stage: fiber-unmix,   in: fs_clean,        out: fs_clean_unmix}
  - {stage: fiber-session, in: fs_clean_unmix,  out: round2, params: {exclude-units: "<every id above 1>", exclude-as: noise}}
```

The port gives the header's keep mask spike for spike on the clusters it was compared on
(test/test_clean.py states which).

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` (`--in`) | `fiber_session` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--res-variant` | `""` | pin the .res variant ('' = the shared .res) |
| `--noise-id` | `1` | cluster the removed spikes go to (default 1) |
| `--junk` | `""` | clusters sent to the noise whole: ids, and/or the word `residue` for the residue cluster fiber-polish recorded for the input stage |
| `--keep-ids` | `""` | clusters passed through untouched |
| `--core-cut` | `1` | 1 = core cut per cluster (default), 0 = quality only |
| `--cc-min` | `200` | smallest cluster the core cut runs on (default 200) |
| `--cc-max` | `40000` | most spikes of a cluster that are ordered; above it an evenly spaced subset (default 40000) |
| `--q-dirty` | `0.52` | a cluster whose quality value is at or above this is dirty (default 0.52, the red of Klusters' quality dot; 0 = no quality test) |
| `--q-spikes` | `1000` | spikes per cluster the quality value is taken on |
| `--min-keep` | `50` | a cluster left with fewer spikes goes to the noise |
| `--out-tag` (`--out`) | — | output stage tag (default <clu-stage>_clean) |
### `fiber-spkc`

Write a stage's sparse `.spkc` (0.70.0): every collision row -- two or more template classes in
its `.eap` row -- with its partners' fitted templates subtracted.  The `.spk` is not changed; a
reader that wants the corrected waveform of row *i* takes it from the `.spkc` when the row is
listed (neurosuite-3 `doc/ndmanager-plugins/formats/spkc.md`), which is what Klusters shows for it.

The `.eap` says which partners and about where; the amplitude and the exact shift are fitted,
jointly with the row's own template, within `--radius` samples of the stored offset.  A partner is
subtracted only for an amplitude in `--amp-min`..`--amp-max` that removes at least `--min-gain` of
the residual energy.  Templates are the mean of the 300 rows of a unit that are not collisions,
nearest in time.  Which cluster a class is: the one its `.tcl` entry was made from while that still
holds 30 rows naming only the class, else the cluster holding most of them.  A row whose own unit
has no template is left as it is.  One file per waveform
variant (`--variants`, default `standard` and the clu method).  `--write-offsets` writes the
fitted whole-sample offsets back to the stage's `.eap`.

`fiber-unmix` writes the file itself (`--spkc 1`); run this after a stage's `.clu` or `.eap` was
changed by something else.  The numpy fit mirrors neurosuite-3's `core/decollide_fit.hpp`: the
same offsets, accept decisions and int16 records on 2,000 random cases, and the same file as
Klusters on a synthetic session (measured 2026-11-23, outside the test suite).

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` | `fiber_session` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--variants` | — | waveform variants to write a .spkc for (default: standard and the clu method) |
| `--write-offsets` | flag (off) | write the fitted whole-sample offsets of the subtracted partners back to the stage's .eap (from the FIRST variant listed) |
| `--radius` | `4` | samples searched either side of the stored offset |
| `--amp-min` | `0.3` |  |
| `--amp-max` | `2.5` |  |
| `--min-gain` | `0.05` | least share of the residual energy a partner must remove to be subtracted |
### `fiber-unmix`

Explain the spikes OUTSIDE the kept units (0.68.0).  Input: a stage whose noise cluster holds what a
cleaning pass removed (`--noise-ids`, default 0 and 1); every other non-reserved cluster is a kept
unit.  Kept spikes are a mask: never pursued, never relabelled, they only supply the templates.  Each
noise detection is pursued on the continuous `.fil` with `fiber_deconv` (noise-whitened, sub-sample
phases, amplitude priors, joint refit) over a segment that also covers its neighbours; templates,
noise model, pursuit threshold and the residual bound are taken per `--block-min` of recording.
Nothing is subtracted from any file.

Each pursued detection gets a kind:

| kind | meaning | `.clu` |
|---|---|---|
| `one` | one kept unit explains it, with a residual no larger than that unit's own spikes leave | `--one-as`: one new cluster, one per unit, or into the unit |
| `collision` | a kept unit plus a second, non-equivalent kept unit in the same waveform (detected itself, or hidden) | `--collision-as`: into the unit on the detection when the label is confident (default), or one new cluster |
| `shadow` | no unit on the detection; its window is explained by neighbouring spikes | one new cluster |
| `partial` | a kept unit on it and a remainder: a candidate collision with a cell that has no template | keeps its label |
| `unexplained` | nothing known fits | keeps its label |

A collision that goes into its unit is that unit's spike with a partner in it: the `.eap` names the
partner, and the stage's sparse `.spkc` (`--spkc 1`, see `fiber-spkc`) holds the row with the
partner subtracted, for the standard and the method `.spk`.

`--children 1` (0.71.8) also writes the stage as a hierarchical session under `<out>_h`
(`fiber-unmix-children`, which runs on a finished stage too):

```
fiber-unmix-children <base> <g> --method stderiv_C5 --stage fc_unmix     # -> .clu/.clc/.clp ...fc_unmix_h
```

In the `.clu` every `one` spike is in the unit it was named for; in the `.clc` the unit's own
spikes are its self child (unit + 10000) and the `one` spikes a second child (20000 and up), so
Klusters shows each unit with what unmix would add to it as one child -- to keep, cut or move.
The result does not depend on `--one-as`.  A row no longer where fiber-unmix put it (moved by hand)
is left alone; `--collisions` adds the parked collision spikes as a third child; the stage's
`.eap` / `.spkc` are linked under the new tag.

`--spkc 2` (0.71.7) writes the cleaned rows the way YASS and DARTsort do instead: **residual
add-back**.  The pursuit has already placed every spike it found in the segment; a row's cleaned
window is the raw window with every OTHER placement subtracted -- any number of them, at the
pursuit's own sub-sample times and amplitudes, with templates taken in the raw signal at those
placements.  It covers every row the pursuit saw that has another placement in its window (kept
spikes near a noise detection included), needs no `.eap`, and is written for the raw waveforms
only (`.spkc.standard`).  On the synthetic recording the error against the record before the
partner is 1.9 % of the raw row's (median).  Klusters does not read this file: it rebuilds its own
corrected rows from the `.eap`.  Not yet compared with the two-template fit on a real session.

So the next round's sort (`fiber-session --in-stage <out> --exclude-units ... --exclude-as noise`)
sees only `partial` and `unexplained`.  Outputs: the flat `.clu` (the `.clc`/`.clp` hierarchy is not
rebuilt), `<base>.unmix.<method>.<g>.<out>.tsv` (one row per pursued detection: kind, first and
second unit with amplitude and offset in samples, gain, margin, residual, whether the second unit
was detected), and with `--eap all` the stage's `.eap`/`.tcl`: every kept spike holds its own class
at offset 0, every explained detection the classes found in it.  The `.eap` is spikes x classes
bytes (one class per kept unit, in steps of `--n-cells`): about 0.6 GB for 800,000 spikes and 660
units; `--eap resolved` writes the explained rows only, `--eap off` none.  `--skip-marked` leaves
out the noise rows the input stage's `.eap` already explains.

Limits: only DETECTED events are examined (a spike under a kept spike that never crossed the
threshold needs a subtraction from the recording); two spikes within a few samples are mostly read
as `one`; an impure kept unit explains what it should not.  Checked on synthetic recordings only
(test/test_unmix.py) -- not yet run on a real session.

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` (`--in`) | `fiber_session` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--res-variant` | `""` | pin the .res variant ('' = the shared .res) |
| `--signal` | `""` | the filtered recording (default <base>.fil) |
| `--noise-ids` | `0,1` | clusters whose spikes are explained (default 0,1); every other non-reserved cluster is a kept unit |
| `--skip-marked` | flag (off) | leave out the noise rows the INPUT stage's .eap already holds a class for |
| `--block-min` | `30.0` | minutes per block: templates, noise model and thresholds are per block (default 30) |
| `--blocks` | `""` | run only these 0-based blocks, e.g. 0-2,7 (default all) |
| `--phases` | `4` | sub-sample phases per template (1 = whole samples) |
| `--order` | `4` | order of the VAR noise-whitening filter |
| `--refrac-ms` | `1.0` | a unit is not placed twice within this |
| `--floor` | `16.0` | lowest pursuit gain threshold |
| `--null-q` | `0.995` | quantile of the off-spike gain taken as threshold |
| `--min-n` | `30` | spikes a kept unit needs in a block for a template |
| `--equiv-cos` | `0.93` | templates at least this similar count as one unit (a second component must differ) |
| `--resid-q` | `0.95` | a fit is clean when its residual is within this quantile of the kept spikes' own |
| `--extend` | `8` | samples the templates reach beyond the .spk window on each side (default 8) |
| `--calib-n` | `300` | kept spikes per block the residual bound is measured on |
| `--one-as` | `cluster` | where one-unit spikes go: one new cluster (default), one new cluster per unit, or into the unit itself when --margin-min is met — choices: `cluster`, `units`, `assign` |
| `--collision-as` | `unit` | where collision spikes go: into the unit on the detection when --margin-min is met (default; the .eap names the partner and the .spkc subtracts it), or all to one new cluster — choices: `unit`, `cluster` |
| `--margin-min` | `10.0` | --one-as assign / --collision-as unit: least gain margin over the best other unit |
| `--spkc` | `1` | 1 = write the stage's sparse .spkc (fiber_spkc: collision rows with the partner subtracted) for the standard and the method .spk (default); needs --eap. 2 = residual add-back instead: every row the pursuit covered with ALL other placements subtracted in the raw signal (the standard .spk only; kept spikes included). 0 = none — choices: `0`, `1`, `2` |
| `--eap` | `all` | the stage's .eap membership: every kept spike (own class, offset 0) and the explained detections (default); resolved = the explained detections and the kept spikes of the units they name (fewer classes, a narrower matrix); or none. The matrix is spikes x template classes bytes — choices: `all`, `resolved`, `off` |
| `--n-cells` | `128` | template-class columns a fresh .eap starts with / grows by |
| `--out-tag` (`--out`) | — | output stage tag (default <clu-stage>_unmix) |
| `--children` | `0` | 1 = also write the stage as a hierarchical session under <out-tag>_h (fiber-unmix-children: .clu + .clc + .clp, every one-unit spike under its unit as a child of its own). The flat .clu is written as always — choices: `0`, `1` |
### `fiber-unmix-children`

A `fiber-unmix` stage as a hierarchical session (`.clu` + `.clc` + `.clp`): every one-unit spike sits
under the unit it was named for, as a child of its own beside the unit's own spikes, so Klusters
shows a unit with what the unmix added to it.  `fiber-unmix --children 1` runs it.

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `""` | the stage's method token (stderiv_C5, standard, ...) |
| `--clu-stage` (`--stage`) | — | the fiber-unmix stage to read (its .clu and its .unmix...tsv) |
| `--out-tag` | — | output stage tag (default <clu-stage>_h) |
| `--collisions` | flag (off) | also put the collision rows fiber-unmix parked in its collision cluster under their first unit, as one more child |
| `--no-collision-files` | flag (off) | do not link the stage's .eap / .spkc files under the new tag |
### `fiber-decollide`

Write collisions resolved by matching pursuit back to a derived stage.

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` | — | feature variant of the .clu (e.g. stderiv_C5_D34) |
| `--res-method` | `stderiv` | detection variant of the .res (default stderiv) |
| `--spk-variant` | `standard` | raw .spk variant to subtract in (default standard) |
| `--spk-variants` | — | write new units' .spk + .fet in EVERY listed variant (e.g. standard stderiv_C5_D34) via decollide_all_variants |
| `--no-fet` | flag (off) | with --spk-variants, skip the .fet projection |
| `--tag` | `decollided` | shared stage tag for the derived files |
| `--parent-tag` | `""` | stage of the parent being de-collided ('' = base files) |
| `--no-eap` | flag (off) | skip writing the .eap membership (+ .tcl) on the parent stage |
| `--n-cells` | `128` | template-class columns (T) when a fresh .eap must be built (default 128) |
| `--nsamp` | — |  |
| `--nch` | — |  |
| `--decomp` | — | npz with 'sel' + k1,a1,tau1,k2,a2,tau2,gain (fiber_collision output) |
### `fiber-polish`

Polish an over-split sort (0.67.33): a residual strip with a pooled full-dimensional rkk, a small-leaf
t-SNE island pass, and the reverse residual strip as a merge.  Every score is taken on the RAW
(`.spk.standard`) windows at their stored alignment; the pool rkk and the island embedding use the
session's feature space (hybrid when the method `.spk` and both `.pca` bases are present).  Writes a
flat `.clu` -- the `.clc`/`.clp` hierarchy is not rebuilt.  The island pass needs `fiber-kit[consensus]`.
Untuned: the defaults are the values of the 2026-11-18 measurements (module docstring;
`FK_POLISH_*` in docs/config.md).

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` (`--method`) | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` (`--in`) | `fiber_session` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--res-variant` | `""` | pin the .res variant ('' = the shared .res) |
| `--gt-clu` | `""` | curated .clu path: score every stage against it (nothing is fitted to it) |
| `--loops` | `2` | strip -> pool -> rkk rounds (0 = skip; rounds 3-5 moved `good` <= 0.5 pt) [FK_POLISH_LOOPS; default 2] |
| `--strip-k-on` | `3.0` | strip: robust-tail multiplier of the on-channel residual (0 = term off) [FK_POLISH_STRIP_K_ON; default 3.0] |
| `--strip-k-off` | `3.0` | strip: robust-tail multiplier of the off-channel residual (0 = term off) [FK_POLISH_STRIP_K_OFF; default 3.0] |
| `--strip-min` | `100` | strip: smallest cluster that is stripped [FK_POLISH_STRIP_MIN; default 100] |
| `--core-cut` | `0` | strip: 0 = the residual strip alone, 1 = Klusters' core cut as well (a spike either of them parks is parked), 2 = the core cut in its place [FK_POLISH_CORE_CUT; default 0] |
| `--cc-min` | `200` | strip: smallest cluster the core cut runs on [FK_POLISH_CC_MIN; default 200] |
| `--cc-max` | `40000` | strip: most spikes of a cluster the core cut orders (above it an even subset) [FK_POLISH_CC_MAX; default 40000] |
| `--pool-max` | `50` | pool: clusters below this many spikes join the pool whole [FK_POLISH_POOL_MAX; default 50] |
| `--pool-min` | `50` | pool: smallest cluster the pool rkk may create (smaller -> the residue cluster) [FK_POLISH_POOL_MIN; default 50] |
| `--pool-k` | `50` | pool: starting cluster count of the full-dimensional rkk [FK_POLISH_POOL_K; default 50] |
| `--pool-cap` | `30000` | pool: most spikes the rkk is run on (the rest go to the residue cluster) [FK_POLISH_POOL_CAP; default 30000] |
| `--rounds` | `2` | island -> merge rounds (0 = skip both) [FK_POLISH_ROUNDS; default 2] |
| `--island` | `1` | 1 = run the island pass (needs the [consensus] extra), 0 = merge only [FK_POLISH_ISLAND; default 1] |
| `--leaf` | `300` | islands: most spikes per t-SNE leaf (the 200-300 regime) [FK_POLISH_LEAF; default 300] |
| `--island-runs` | `8` | islands: resampled t-SNE runs per leaf [FK_POLISH_ISLAND_RUNS; default 8] |
| `--island-min` | `10` | islands: smallest community treated as an island [FK_POLISH_ISLAND_MIN; default 10] |
| `--island-min-clu` | `60` | islands: smallest cluster (and leaf) that is embedded [FK_POLISH_ISLAND_MIN_CLU; default 60] |
| `--island-margin` | `0.5` | islands: move only if residual to the target < MARGIN x residual to the parent [FK_POLISH_ISLAND_MARGIN; default 0.5] |
| `--merge` | `1` | 1 = run the reverse-strip merge, 0 = islands only [FK_POLISH_MERGE; default 1] |
| `--merge-thr` | `0.7` | merge: min share of each cluster that survives the other's strip (0.3 collapses a sort) [FK_POLISH_MERGE_THR; default 0.7] |
| `--merge-min` | `50` | merge: smallest cluster that takes part [FK_POLISH_MERGE_MIN; default 50] |
| `--merge-q` | `-1.0` | merge: quality guard -- a union is refused when its quality value (Klusters' quality dot, on the raw windows) exceeds the worse of its two parts' by more than this (0 = no worse than the worse part; -1 = no guard) [FK_POLISH_MERGE_Q; default -1.0] |
| `--merge-q-max` | `0.0` | merge: quality bound -- a union whose quality value exceeds this is refused (0 = no bound; 0.2 moved purity by 0.2 pt on 0312 g8: the module header) [FK_POLISH_MERGE_Q_MAX; default 0.0] |
| `--realign-topch` | `0` | realign: > 0 = before every loop and every round each cluster is realigned to its own median, only its N largest channels voting (Klusters' Realign top-ch; N >= the channel count = all of them), and every pass reads the windows at that alignment. 0 = the stored alignment, as before [FK_POLISH_REALIGN_TOPCH; default 0] |
| `--realign-maxlag` | `3` | realign: furthest a spike is moved from its stored position (samples) [FK_POLISH_REALIGN_MAXLAG; default 3] |
| `--realign-min` | `20` | realign: smallest cluster that is realigned (smaller ones stay as stored) [FK_POLISH_REALIGN_MIN; default 20] |
| `--jobs` | `1` | worker processes that find the islands (1 = none, 0 = one per core); the labels do not depend on it [FK_POLISH_JOBS; default 1] |
| `--cos` | `0.5` | template cosine a pair needs to be a candidate (islands and merge) [FK_POLISH_COS; default 0.5] |
| `--tpl-cap` | `1000` | spikes per cluster used for a median template and its tails [FK_POLISH_TPL_CAP; default 1000] |
| `--hybrid-pcs` | `3` | features: standard-basis components per channel in the raw block of the hybrid space [FK_POLISH_HYBRID_PCS; default 3] |
| `--hybrid-scale` | `2.0` | features: weight of the raw block (the hybrid-search winner's 2.0) [FK_POLISH_HYBRID_SCALE; default 2.0] |
| `--seed` | `0` | random seed (subsamples, t-SNE runs, rkk) [FK_POLISH_SEED; default 0] |
| `--out-tag` (`--out`) | — | output stage tag (default <clu-stage>_polish) |
### `fiber-bench`

Run, score and compare pipeline-parameter variants under a config hash.  Sub-commands `run` (the config's
plan up to `--upto`, one run per `--set`/`--grid`/`--sweep` variant, every clustering step's output tagged
`<out>_<hash>`; fiber-session's in-place realign forced off), `score` (truth metrics per truth unit, typed
pyramidal / interneuron, and truth-free per-cluster measures), `report` (HTML) and `list`.  See
[bench.md](bench.md).

Positional: `cmd`

| flag | default | description |
|---|---|---|
### `fiber-tune`

Search a plan's knobs against a ground truth on a subset of chunks, then confirm the best.  Sub-commands
`search` (coordinate search over the knob groups of a spec file, each trial a `fiber-bench run` of the plan
up to `score_at` on the spec's `chunks`, in-place realign steps left out, scored on the truth), `confirm`
(the best configurations again with the FULL plan, realign included, each in a private workspace with
its own copy of the .res/.spk/.fet), `show` and `rescore` (every recorded trial scored again from its
stored .clu, nothing re-run).  Writes `fk-bench/tune/<name>/` (trials, summary,
`best.yaml`).  Example spec: `plans/tune/exp-search.yaml`.  See [bench.md](bench.md#fiber-tune).

Positional: `cmd`

| flag | default | description |
|---|---|---|
### `fiber-gt-review`

Lists where a sort (`--clu-stage`) and a curated ground truth disagree, with waveform (roll-cosine of the raw templates) and refractory (cross-refractory ratio) evidence for which one is wrong: curated units that violate their own refractory period, sort clusters holding two curated units or curator-noise spikes, and curated units split over sort clusters.  Writes `<base>.gtreview.<group>.<stage>.tsv` and a review `.clu` (`<stage>_gtreview`) with every listed spike set as a cluster.  See [bench.md](bench.md#reviewing-the-ground-truth-fiber-gt-review).

Positional: `session`, `elec`

| flag | default | description |
|---|---|---|
| `--clu-method` | `stderiv_C5_D34` | method token of the sort's .clu |
| `--clu-stage` | — | stage tag of the sort to compare (e.g. exp_psame) |
| `--gt-clu` | — | curated .clu |
| `--gt-res` | — | its .res |
| `--gt-units` | `""` | curated units (default every unit > 1) |
| `--gt-exclude` | `""` | units to leave out |
| `--gt-tol` | `5` | spike match tolerance (samples) |
| `--min-n` | `50` | spikes a set needs to be listed |
| `--frac` | `0.1` | ... and this share of its sort cluster |
| `--spk-variant` | `standard` | waveforms for the cosine (default raw standard) |
| `--no-clu` | flag (off) | do not write the review .clu |
### `fiber-envelope`

Collision-robust per-unit variance envelope sigma(t,c) (noise floor from off-spike windows + amplitude + jitter + robust excess) and near-collision detection: flags spikes carrying a second, non-coincident spike inside the window, explains each with another unit's template at a non-circular shift, and writes the per-unit `.env` table (with the cell layer: amplitude-dependent shape, adaptation law, spike width / burst / rate), the per-spike `.nco` table and, with `--eap`, the explained partners as `.eap` membership. `--merge THETA` also merges units on envelope d' with merge inflation as a cannot-link and writes a `.envmerge` clu stage plus the per-pair evidence. See docs/methods.md, "Collision-robust envelope" and "Cell layer".

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--signal` | — | continuous filtered int16 file to read windows from (e.g. BASE.fil) |
| `--total-channels` | — | channels in the signal file (default: from the session yaml) |
| `--channels` | — | group channels, e.g. 40-47 or 40,41,... (default: session yaml) |
| `--signal-start` | `0` | absolute sample of the signal file's first row (for chunks) |
| `--space` | — | waveform space: standard \| stderiv \| stderiv_C<order> (default: the --clu-method token without its _D suffix, i.e. the clu's own space) |
| `--sdiff-pairs` | — | sdiffPairs pattern for a _C space (default: the session yaml's group entry) |
| `--res-variant` | `""` | pin the .res variant (e.g. stderiv_S3); default: resolve the shared .res like the other stages |
| `--clu-method` | `stderiv_C5_D34` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` | `""` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--pre` | `21` | core window samples before the detected trough |
| `--post` | `21` | core window samples from the trough on |
| `--margin` | `32` | extended-window margin for partners |
| `--min-n` | `50` | skip units with fewer spikes |
| `--max-per` | `3000` | spikes per unit used to FIT the envelope (all spikes are scored) |
| `--q` | `0.99` | null quantile of the bump threshold (off-spike noise windows) |
| `--out-tag` | `""` | tag for the .env/.nco tables |
| `--sr` | — | sampling rate (default: session yaml, else 32552) |
| `--merge` | — | also merge units: average linkage on envelope d' cut at THETA (7 on g6), with merge inflation as a cannot-link; writes a new .clu stage + .envmerge table |
| `--infl-thr` | `0.1` | merge inflation above which a pair cannot be merged (default 0.1) |
| `--type-veto` | flag (off) | never merge a narrow- with a broad-spiking unit (off by default: on g6 fiber_session fragments the per-fragment width is too noisy and the veto cost 0.11 ARI) |
| `--refrac-ms` | `1.5` | refractory cannot-link for --merge: two units whose trains coincide within (0.6, REFRAC] ms at chance are two neurons (power-aware; 0 = off; default 1.5) |
| `--refrac-min-exp` | `3.0` | coincidences that must be expected before the refractory test may veto (default 3) |
| `--merge-tag` | — | stage tag of the merged .clu (default: <clu-stage>.envmerge) |
| `--eap` | flag (off) | write explained near collisions as .eap membership on the analysed --clu-stage (Klusters' collision ring) |
### `fiber-consensus`

Resampled-t-SNE consensus over a neighbourhood of similar clusters (the ones a curator would lasso together): many t-SNE runs of random subsets in the stage's `.fet` space, Louvain communities on the co-neighbour probability, communities joined into cells by the envelope merge, and a QUARANTINE cluster for spikes that carry a near collision or sit on the rim of their cell's 2-D cloud with a deviant waveform (cell switching opt-in), with a per-spike reason table (`.qrn`). Needs the `consensus` extra (openTSNE, networkx). See docs/methods.md, "Resampled t-SNE consensus and quarantine".

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--signal` | — | continuous filtered int16 file (e.g. BASE.fil) |
| `--total-channels` | — | channels in the signal file (default: session yaml) |
| `--channels` | — | group channels, e.g. 40-47 (default: session yaml) |
| `--signal-start` | `0` | absolute sample of the signal's first row |
| `--space` | — | waveform space for the envelope (default: --clu-method's) |
| `--sdiff-pairs` | — | sdiffPairs for a _C space (default: session yaml) |
| `--clu-method` (`--method`) | `stderiv_C5_D34` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` (`--in`) | `""` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path, overriding --clu-method/--clu-stage |
| `--fet-variant` | — | variant of the .fet to embed (default: --clu-method) |
| `--units` | — | comma-separated units to study together (repeat with ';' for several groups); default: every envelope d' neighbourhood |
| `--sr` | — |  |
| `--tsne-backend` | `auto` | FK_CONSENSUS_TSNE_BACKEND. auto (default) = RAPIDS cuML on the GPU when importable, else openTSNE on the CPU; the runs then go serially on the one device instead of the worker pool — choices: `auto`, `opentsne`, `cuml` |
| `--quarantine-only` | `0` | 1 = clean, do not merge: kept spikes keep their input labels, quarantined spikes go to noise (1); the .clc/.clp hierarchy is carried; island / rim clusters are written to a <tag>_qreview stage for review. For use right after fiber-session, before intrachunk and the linkers [FK_CONSENSUS_QUARANTINE_ONLY; default 0] |
| `--res-variant` | `""` | pin the .res variant (e.g. stderiv_S3); '' = resolve the shared .res the way fiber-session and the other stages do [FK_CONSENSUS_RES_VARIANT; default ] |
| `--fet-hybrid` | `0` | 1 = embed a HYBRID feature space: the --fet-variant block plus --hybrid-fet-variant's, each z-scored per column and weighted to carry the same variance (fiber-session's FK_SESSION_FEAT_HYBRID, one stage later and at the .fet level instead of the waveform's) [FK_CONSENSUS_FET_HYBRID; default 0] |
| `--hybrid-fet-variant` | `standard` | variant of the .fet stacked beside --fet-variant under --fet-hybrid 1; 'standard' falls back to the canonical <base>.fet.<elec> when no .fet.standard is staged [FK_CONSENSUS_HYBRID_FET_VARIANT; default standard] |
| `--hybrid-scale` | `1.0` | weight of the appended block relative to the first after the two are put on equal total variance (1 = equal; <1 clustering-space-led, >1 appended-led; the square is the variance share), as FK_SESSION_HYBRID_SCALE weights fiber-session's raw block [FK_CONSENSUS_HYBRID_SCALE; default 1.0] |
| `--hybrid-pcs` | `0` | components per channel kept from the appended block (0 = all), as FK_SESSION_HYBRID_PCS trims fiber-session's raw block [FK_CONSENSUS_HYBRID_PCS; default 0] |
| `--fet-drop-time` | `auto` | drop the .fet's last column -- the spike TIME Klusters requires there, and the one KlustaKwik's useFeatures masks off -- before embedding: auto = only under --fet-hybrid (two blocks would carry two time ramps), 1 = always, 0 = never (what every run before 0.67.26 did, a z-scored coordinate worth 1/d of the variance whatever the chunk length). Only a column that is non-decreasing over the whole .fet is dropped, so a .fet with no time column (fiber-decollide's) keeps every component [FK_CONSENSUS_FET_DROP_TIME; default auto] |
| `--neighbourhood-theta` | `12.0` | envelope d' cut for automatic neighbourhoods (ignored with --units) [FK_CONSENSUS_NEIGHBOURHOOD_THETA; default 12.0] |
| `--max-spikes` | `0` | re-cut a neighbourhood holding more spikes at a lower d' until it fits (0 = no cap; a cap costs completeness: it splits confusable cells apart) [FK_CONSENSUS_MAX_SPIKES; default 0] |
| `--runs` | `40` | resampled t-SNE runs [FK_CONSENSUS_RUNS; default 40] |
| `--frac` | `0.7` | fraction of the neighbourhood's spikes in each run [FK_CONSENSUS_FRAC; default 0.7] |
| `--k` | `15` | 2-D neighbours per spike [FK_CONSENSUS_K; default 15] |
| `--perplexity` | `30.0` | t-SNE perplexity (runs and watershed maps) [FK_CONSENSUS_PERPLEXITY; default 30.0] |
| `--tsne-ee-iter` | `250` | t-SNE early-exaggeration iterations per run (openTSNE 250) [FK_CONSENSUS_TSNE_EE_ITER; default 250] |
| `--tsne-iter` | `500` | t-SNE main-phase iterations per run (openTSNE 500; 250 measured to give the same consensus at 1.8x the speed) [FK_CONSENSUS_TSNE_ITER; default 500] |
| `--theta-cell` | `7.0` | envelope d' cut joining communities into cells [FK_CONSENSUS_THETA_CELL; default 7.0] |
| `--refrac-ms` | `1.5` | refractory cannot-link in the cell merge: communities whose trains coincide within (0.6, REFRAC] ms at chance are two neurons (power-aware; 0 = off) [FK_CONSENSUS_REFRAC_MS; default 1.5] |
| `--refrac-min-exp` | `3.0` | coincidences that must be expected before the refractory test may veto [FK_CONSENSUS_REFRAC_MIN_EXP; default 3.0] |
| `--switch-thr` | `0.3` | cell-switch rate that quarantines a spike [FK_CONSENSUS_SWITCH_THR; default 0.3] |
| `--perim-thr` | `0.8` | average 2-D radial rank within the cell (0 centre, 1 rim) for the perimeter rule [FK_CONSENSUS_PERIM_THR; default 0.8] |
| `--dev-z` | `2.0` | robust z of residual energy or bump score (within the cell) for the perimeter rule [FK_CONSENSUS_DEV_Z; default 2.0] |
| `--rules` | `collision,perimeter` | quarantine criteria, any of: switch, collision, perimeter (switch is off by default: 0.13-0.22 precision where a neighbourhood's cells are close) [FK_CONSENSUS_RULES; default collision,perimeter] |
| `--ws-runs` | `5` | full t-SNE maps voting the watershed category; 0 = no categories, one quarantine cluster per neighbourhood [FK_CONSENSUS_WS_RUNS; default 5] |
| `--ws-grid` | `256` | KDE grid side for the watershed basins [FK_CONSENSUS_WS_GRID; default 256] |
| `--ws-island` | `0.5` | quarantined fraction that makes a basin an island [FK_CONSENSUS_WS_ISLAND; default 0.5] |
| `--window` | `""` | restrict to spikes in START,END minutes of the signal ('' = the whole signal) [FK_CONSENSUS_WINDOW; default ] |
| `--chunk-min` | — | process the sort chunk by chunk, this many minutes each (unset = FK_SESSION_CHUNK_MIN or 12, fiber-session's chunking; 0 = the whole window at once) [FK_CONSENSUS_CHUNK_MIN; default None] |
| `--keep-ids` | `1` | 1 = a cell keeps the id of the input unit holding most of its kept spikes (so a linked sort stays linked); 0 = every cell gets a fresh id [FK_CONSENSUS_KEEP_IDS; default 1] |
| `--min-spikes` | `300` | skip a neighbourhood with fewer spikes in the chunk (t-SNE at perplexity 30 needs a few hundred) [FK_CONSENSUS_MIN_SPIKES; default 300] |
| `--workers` | `0` | parallel processes for the independent t-SNE runs and watershed maps (0 = every core, 1 = serial); results do not depend on it [FK_CONSENSUS_WORKERS; default 0] |
| `--jobs` | `1` | openTSNE threads inside each run (keep 1 when workers > 1) [FK_CONSENSUS_JOBS; default 1] |
| `--seed` | `0` | seed for subsampling, initialisation and Louvain [FK_CONSENSUS_SEED; default 0] |
| `--out-tag` (`--out`) | — | stage tag of the output .clu (default: <clu-stage>.consensus) |
### `fiber-calibrate`

Learn the variance/energy envelope of a curated group's single units and write an .npz budget for `fiber-defrag --var-budget`.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--clu-method` | `stderiv` | feature space before the group (default stderiv) |
| `--clu-stage` (`--variant`) | `""` | curated fiber stage tag after the group (default none) |
| `--in-clu` | — | explicit curated .clu path (overrides --clu-method/--variant) |
| `--n-pc` | `10` | number of PCs for the feature space (default 10) |
| `--min-cluster` | `60` | ignore curated units below this many spikes |
| `--floor` | flag (off) | set allowance to the confusable-pair merged-variance floor (tighter, recommended) |
| `--cos-thr` | `0.85` | candidate cosine for the confusable-pair gate |
| `--warp-max` | `0.06` | width gate for the confusable-pair gate |
| `--out` | — | output .npz (default '<base>.calib.<group>.npz') |
### `fiber-validate-merges`

Full-session evidence for proposed same-neuron merges; reads <session>.yaml for sr.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--sr` | — | override: sampling rate |
| `--cand` | — | candidates tsv (default <base>.merge_candidates.<group>.tsv) |
### `fiber-ccg`

Refractory QC for a group's clustering: per-cluster ISI-violation fraction, and the cluster pairs whose refractory cross-correlogram shows a dip (merge-consistent).

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--clu-method` | `stderiv` | method the clu stems from: standard \| stderiv \| stderiv_C5 |
| `--clu-stage` (`--variant`) | `refine` | post-fiber stage tag at the end of the .clu name |
| `--in-clu` | — | explicit .clu path |
| `--refrac-ms` | `1.5` | refractory window (ms, default 1.5) |
| `--censor-ms` | `0.3` | duplicate censor band (ms, default 0.3) |
| `--thr` | `0.3` | ratio at/below which a pair shows a dip |
| `--min-exp` | `5.0` | min expected coincidences to have power |
| `--min-cluster` | `40` | ignore clusters smaller than this |
| `--top` | `15` | how many merge-consistent pairs to list |
### `fiber-template`

Linked per-unit median templates along time or energy: one `.wtf` per variant, indexed by the
`.wti`.

Positional: `base`, `elec`

| flag | default | description |
|---|---|---|
| `--variants` | — | spk variants to template, e.g. standard stderiv_C5_D34 |
| `--clu-variant` | — | variant of the units .clu |
| `--clu-tag` | `""` | tag/stage of the units .clu |
| `--out-tag` | `""` | stage tag for the written .wtf/.wti |
| `--links` | `['drift']` | template link axes to generate (default: drift) — choices: `drift`, `adapt` |
| `--nsamp` | — |  |
| `--nchan` | — |  |
| `--sr` | `32552.0` |  |
| `--chunk-min` | — | drift chunk duration in minutes |
| `--n-chunks` | — | drift: fixed number of chunks (default 6) |
| `--n-energy` | `5` | adapt: number of energy bins |
| `--energy-variant` | — | variant whose energy defines the adapt bins (default: standard if present, else the first) |
| `--max-per` | `800` |  |
| `--units` | — | restrict to these ids (clu ids, or .eap class columns with --eap) |
| `--eap` | flag (off) | membership from the .eap matrix (class = column, offset-aligned), not the .clu; the stable class id is stamped into .wti |

## Pipeline runner, plans & tooling

### `fiber-plan-edit`

*(parameters not introspectable: SystemExit; run `fiber-plan-edit -h`)*

Positional: `plans`

| flag | default | description |
|---|---|---|
| `--strict` | flag (off) | treat warnings as failures (CI gate) |
| `-q` (`--quiet`) | flag (off) | print only failures |
### `fiber-plan-lint`

Statically check plan files, or a `fiber-kit.yaml` with a `pipeline:` section: stage names (against
`fiber_kit/stages.py`), tag wiring, cycles and, against the stage modules themselves, unknown or
mistyped `params` flags.  Exits 1 on errors; with `--strict` also on warnings.

Positional: `plans`

| flag | default | description |
|---|---|---|
| `--strict` | flag (off) | treat warnings as failures (CI gate) |
| `-q` (`--quiet`) | flag (off) | print only failures |
### `fiber-config-edit`

Qt form editor for a `fiber-kit.yaml`: the file's sections as drop-downs, typed fields, off knobs
greyed, each knob's comment in a help pane on hover.  It saves losslessly: only the edited values
change.

### `ndm_fiber-kit`

The NDManager plugin entry: `ndm_fiber-kit <session>.yaml [<group>] [stages...]` runs
`fiber-pipeline` on one group, or on every spike group of the session when none is given.
`--ndm-version` prints the format versions.

### `fk`

Plan steps by name, and the config profiles.  `fk list` prints the steps of the plan `fiber-pipeline` would run (`--plan`, else the `pipeline:` section of `$FK_CONFIG` / `./fiber-kit.yaml`) with their unique ids (`<stage>-<k>`, the k-th run of that stage) and `name:` aliases; `fk run <elec>` runs the whole plan (as `fiber-pipeline <elec>`: the canonical stages when there is no plan), and `fk run <elec> <step>...` / `--from` / `--to` run chosen steps (`fiber-pipeline <elec> --plan P --step ...`).  `fk list config` lists the shipped config profiles (`base` fiber-kit.yaml, `exp` fiber-kit-exp.yaml, `best` fiber-kit-best.yaml -- the tuned one) and which one `./fiber-kit.yaml` is; `fk set config --profile <name>` writes one as `./fiber-kit.yaml` in the working directory (refuses to overwrite without `--force`; a profile is its short name, its file name or a path).  `fk run <elec> --profile <name>` / `--config <file>` run with that config instead of the session folder's fiber-kit.yaml.  `fk install klusters-plugin` installs the Klusters plugin descriptor (`fiber-kit-recluster`: a `recluster` plugin that runs the plan on the selected clusters' sub-session).  `fk run <elec> --session <base> --final-clu <path>` runs the plan on a session named by its base path (a sub-session) and copies the run's final `.clu` (with its `.clc`/`.clp`) to `<path>`; `fk subsession export|clean` writes / removes a sub-session: the spikes of chosen clusters as a session of their own (see [pipeline.md](pipeline.md#reclustering-a-selection-sub-sessions)).  `fk completion bash|zsh` prints the shell completion for all of this (subcommands, profiles, the plan's step names): `eval "$(fk completion bash)"`.  See [pipeline.md](pipeline.md#step-names) and [config.md](config.md).

Positional: `cmd`

| flag | default | description |
|---|---|---|
### `fiber-kit-init`

Write a shipped config profile (`--profile base|exp|best`, or a path) into the current directory as fiber-kit.yaml -- the same as `fk set config`; `--list` lists the profiles as `fk list config` does.

| flag | default | description |
|---|---|---|
| `-p` (`--profile`) | — | profile: base (fiber-kit.yaml), exp (fiber-kit-exp.yaml), best (fiber-kit-best.yaml), klusters (fiber-kit-klusters.yaml), klusters-manifold (fiber-kit-klusters-manifold.yaml), or a path to a yaml (default: base; --exp = exp) |
| `-o` (`--output`) | `fiber-kit.yaml` | destination path (default: ./fiber-kit.yaml) |
| `-f` (`--force`) | flag (off) | overwrite the destination if it already exists |
| `--exp` | flag (off) | the same as --profile exp (kept from earlier versions) |
| `--list` | flag (off) | list the profiles (as `fk list config`) and exit |
### `fiber-clu-header`

Check or repair the int32 header of `.clu` / `.clc` / `.clp` files.  The header is the number of distinct ids (a `.clp`: its number of entries).  That is what Klusters writes, and it keeps a chunk run's files (`FK_CHUNKS`) within the 0-65536 Klusters accepts.  fiber-kit 0.56.0-0.57.2 wrote the highest id + 1 instead: about 16.8 million once the reserved clusters are in the file, so Klusters refused those files.  Rewrites the 4 header bytes in place and leaves the ids alone; names that are not `<base>.<clu|clc|clp>[.<variant>].<elec>[.<tag>]`, text `.clu` files and odd sizes are skipped.  `--check` writes nothing and exits 1 when a header is off.

Positional: `files`

| flag | default | description |
|---|---|---|
| `--check` | flag (off) | report only, write nothing; exit 1 when a header is off |
### `fiber-raw-vs-stderiv`

raw .fil vs stderiv discrimination of the original fibers; reads <session>.yaml.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from YAML) |
| `--chunk-min-start` | `0.0` |  |
| `--chunk-min` | `10.0` |  |
| `--min-spikes` | `60` |  |
| `--min-group` | `200` |  |

## Visualization

### `fiber-view`

Visualise fibers: template montages, a 3-D manifold of local fiber curves, and ISI/geometry panels.

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from YAML) |
| `--in-clu` | — | sort to view (default canonical .clu) |
| `--fibers` | `top:6` | comma list of .clu cluster ids, 'top:N', or 'all' (default top:6) |
| `--mode` | `all` | choices: `templates`, `manifold`, `stats`, `all` |
| `--npos` | `80` | positions sampled along the fiber |
| `--geom` | — | a .geom/.geomchunk npz for the stats geometry track |
| `--no-dedup` | flag (off) |  |
| `--out` | — | output path or directory (default next to the session) |
| `--channels-override` | — |  |
### `fiber-view-gui`

fiber-view-gui: a standalone, rotatable bundle viewer.

*(parameters not introspectable: SystemExit; run `fiber-view-gui -h`)*



Positional: `bundles`

| flag | default | description |
|---|---|---|
### `fiber-view-tour`

Interesting-projection tour video of the fibers.

Positional: `bundles`

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

Positional: `session`, `group`

| flag | default | description |
|---|---|---|
| `--channels` | — | override: comma-separated physical channels |
| `--ntotal` | — | override: total channels in the recording |
| `--nchan` (`--nch`) | — | override: channels in this group |
| `--nsamp` | — | override: samples per spike (default from YAML) |
| `--sr` | — | override: sampling rate |
| `--sdiff-pairs` | — | override: the group's sdiffPairs pattern, which a stderiv_C<n> .spk was built with (default from YAML) |
| `--chunk-min` (`--chunk-minutes`) | `12.0` |  |
| `--overlap-min` | `4.0` |  |
| `--min-group` | `200` | COARSE min spikes/fiber (for linking) |
| `--fine-algo` (`--fine-method`) | `rkk` | per-fiber fine clusterer: rkk (default, masked-EM KlustaKwik with per-cluster realign + BIC model selection, robust to unequal covariances), gmm (Gaussian-mixture split), fiber (fiber-kit's native cluster_chunk), none (no fine split -- keep the coarse cluster whole) — choices: `gmm`, `rkk`, `fiber`, `none` |
| `--rkk-dims` | `6` | FK_SESSION_RKK_DIMS. SVD components of the fine-split feature space handed to the rkk CEM |
| `--rkk-max` | `50` |  |
| `--rkk-realign` | flag (off) | interleave rkk (CEM) with per-cluster realignment (per-step; default on) |
| `--no-rkk-realign` | flag (on) | legacy: one parent realign + fixed features for the rkk split |
| `--rkk-realign-iters` | `2` | FK_SESSION_RKK_REALIGN_ITERS. cluster<->realign passes in the rkk realign loop (each pass re-aligns every cluster to its own median and re-runs the CEM; stops early when the cluster count no longer changes) |
| `--rkk-delete` | flag (off) | rkk (CEM) culls sub-min-group sub-clusters during the per-fiber fine split (default on) |
| `--no-rkk-delete` | flag (on) | keep small non-singular rkk sub-clusters -- session should OVER-cluster, leaving the cull to refine; use to stop session shedding fragments into the residual/artifact bin |
| `--refit-iters` | `0` | iterate {realign each fiber's spikes to their own median -> refit template -> re-apply the inclusion gate} this many extra times. Recovers true members rejected for a bad sub-sample offset and tightens the template; re-judges from the full member set each pass. 0 = single pass (original). Marginal on clean fibers; helps contamination-heavy ones. |
| `--reseed-residual-thr` | `0.0` | residual-gated reseed: after the fine split, any fiber whose median-subtracted residual/signal exceeds this is re-split (a welded pair of co-located different-shape cells scores high; ~0.35 separates welded from clean on g5). 0 = off. The re-split is kept only if it lowers the residual by --reseed-min-reduction. |
| `--reseed-min-reduction` | `0.05` | keep a residual-gated reseed only if it lowers the median-subtracted residual/signal by at least this much (a real sub-unit does; noise does not) |
| `--seed-density` | `0.0` | density-preferential coarse seeding: draw ridge seeds toward concentrated modes of the waveform space (p proportional to local_density**this) instead of a uniform stride. 0 = uniform (current); 1 = fully density-weighted; small values a gentle tilt. Concentrates the seed budget on isolated, reproducible modes (isolation, not firing rate). |
| `--coarse-dr` | `0.15` | radial band half-width (fraction of the 1-99%% whitened-radius span) for spike->seed association in the COARSE pass. Lower -> tighter amplitude bands -> more, smaller coarse fibers (amplitude-distinct cells stay separate for the fine splitter); higher -> fewer, fatter coarse groups. Default 0.15. |
| `--whiten-source` | `auto` | FK_SESSION_WHITEN_SOURCE. Where the noise whitener comes from. nst = the .nst.<variant>.<group> ndm_extractspikes writes beside the .spk: noise measured in the .spk's own transform on spike-free windows, outliers rejected (one whitener for the session). fil = per chunk, off-spike .fil baseline taken into the .spk's waveform space (fiber_stderiv.space_transform; needs the .fil, and the sdiffPairs for a _C variant). spk = per chunk, the snippets' own pre-peak samples (stationary block-Toeplitz covariance): needs neither file and follows any variant, including ones no transform describes (fiber-adlp's per-spike filtered snippets). auto (default) = nst when the .spk's .nst exists, else fil. --deconv needs the .fil either way. — choices: `auto`, `nst`, `fil`, `spk` |
| `--no-whiten` | flag (off) | cluster in the RAW mask-selected (mean-centred) feature space instead of the .fil baseline-whitened space: identity whitener, so the radial coordinate is the raw feature norm and the angular metric is raw cosine (no per-channel covariance normalisation) |
| `--merge-corr` | `0.0` | consolidate fibers above this (0=off; 0.95 template / 0.90 sliding) |
| `--neigh-passes` | `0` | FK_SESSION_NEIGH_PASSES. passes of the neighbourhood re-cluster (Block N): rkk over the union of a random set of 1..--neigh-k neighbouring clusters (template cosine >= --neigh-cos), seeded from the current labels, adopted only when KK's penalised score improves. 0 = off. |
| `--neigh-k` | `3` | FK_SESSION_NEIGH_K. most neighbours drawn per seed cluster |
| `--neigh-cos` | `0.7` | FK_SESSION_NEIGH_COS. template cosine (masked channels, each realigned) that makes two clusters neighbours |
| `--neigh-seed` | `0` | FK_SESSION_NEIGH_SEED. seed of the neighbourhood draw |
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
| `--off-thr` | `0.0` | inter-channel-TIMING co-veto on Block-A fragment merges: refuse a footprint-correlation merge whose two fibers' RAW-template inter-channel offsets disagree by more than this (xcorr LAG samples, ~0.2-0.3; NOT the trough ~1.0). Reads .spk.standard for raw templates (stderiv timing is channel-mixed); inert if that file is absent. FK_SESSION_OFF_THR. 0 = OFF (default) |
| `--off-amp-frac` | `0.3` | channels below this fraction of the dominant peak-to-peak carry no reliable timing for --off-thr. FK_SESSION_OFF_AMP_FRAC |
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
| `--deconv` / `--no-deconv` | flag (off) | detection-anchored sub-sample template deconvolution per chunk (fiber_deconv): per-fiber deconv_agree contamination QC in .fibers and a .deconv.tsv manifest of labels, collision partners and undetected spikes. Changes no label unless --deconv-assign |
| `--deconv-assign` / `--no-deconv-assign` | flag (off) | with --deconv: move confidently attributed noise / collision-cluster spikes into their fiber as a parked child atom. Off by default: on synthetic sessions with look-alike cells its precision fell to 70-90 percent (docs/methods.md) |
| `--deconv-phases` | `4` | sub-sample phases per template (1 = whole samples) |
| `--deconv-order` | `4` | order of the VAR noise-whitening filter |
| `--deconv-refrac-ms` | `1.0` | a unit is not placed twice within this many ms |
| `--deconv-floor` | `16.0` | lowest pursuit gain (2 x log-likelihood) accepted, whatever the noise null says |
| `--deconv-null-q` | `0.995` | quantile of the off-spike first-step gain used as the stopping threshold |
| `--deconv-agree` | `0.999` | margin calibration: agreement with the clustering required among clean spikes above it |
| `--deconv-margin-min` | `10.0` | floor on the calibrated label margin (2 x log-likelihood ratio, best vs other unit) |
| `--deconv-min-n` | `30` | min spikes for a fiber to get a template |
| `--deconv-fiber-agree` | `0.95` | a fiber whose spikes the pursuit labels with an equivalent template at least this often is corroborated: the margin is calibrated on such fibers and only they receive rescues |
| `--deconv-equiv-cos` | `0.93` | fibers whose templates reach this whitened cosine (noise-corrected, best over +/-4 samples) are equivalent: the over-split fragments of a cell. Agreement, margin and collision partners are judged up to equivalence. >1 = identity only |
| `--quality-metrics` | flag (off) | also compute L-ratio + isolation distance (O(N*K)) |
| `--quality-dims` | `10` | PCA dims for L-ratio/isolation Mahalanobis |
| `--pca-k` | `6` |  |
| `--max-sub` | `8` |  |
| `--inclusion-k` | `3.0` | per-fiber radius = median+k*MAD of residuals; 0 disables |
| `--no-noise` | flag (off) | sweep every remaining noise spike (below the inclusion radius / rejected / collision junk) into a single UNDEFINED FIBER (one real cluster, not the noise cluster) rather than dropping it. For clean stderiv data; the undefined fiber is cleaned/re-split in later steps. |
| `--keep-small-groups` | flag (off) | a fine group below --fine-min-group (or whose inclusion core falls below it) becomes a fiber of its own -- all its spikes -- instead of going to the noise cluster. Over-split is recoverable downstream; a dropped unit is not. Groups below --keep-small-min are left to --assign-unsorted |
| `--keep-small-min` | `5` | smallest group --keep-small-groups keeps as its own fiber (default 5) |
| `--assign-unsorted` | flag (off) | place every spike still unsorted at the end of a chunk (coarse-unassigned, groups too small to keep): within its nearest fiber's own spread -> that fiber; outside every fiber's spread -> GMM groups of their own (with --keep-small-groups), the remainder -> nearest fiber (whitened, realigned features: shape and amplitude). With fiber-flag-artifacts -> --exclude-clu, only flagged artifacts reach cluster 0 |
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
| `--out-variant` | — | ALSO write .clu/.clc/.clp under this variant token, in addition to the --method-named copies. READS are unaffected (the .pca basis and .spk still resolve under --method), so an alias like stderiv_C5_D34 can name a feature space without needing its own basis or waveform file. |
| `--emit-pca` | flag (off) | with --emit-fet: also write <base>.pca.<out-variant or method>.<elec>, the basis the .fet is the projection of, so Klusters can realign/reproject in that space |
| `--emit-fet` | flag (off) | write <base>.fet.<out-variant or method>.<elec> -- the FULL-WIDTH session-wide projection on the global basis (the feature space the fine split used, before its per-chunk reduction), so Klusters can display it. |
| `--feat-lag` | `0` | FK_SESSION_FEAT_LAG. >0 = fine-split shape features become PC1 sampled at -N/0/+N samples plus PC2 (4 per channel) instead of PC1,PC2,PC3. 0 = classic. |
| `--feat-lag-pc2` | `1` | with --feat-lag: 1 (default) keep PC2 as a 4th column per channel, 0 = lags only |
| `--feat-hybrid` | `0` | FK_SESSION_FEAT_HYBRID. 1 = HYBRID fine-split features: the --method (stderiv) features as configured (lag space with --feat-lag) PLUS the leading standard-basis PCs of the paired RAW .spk.standard window of each spike, one realign shifting both. The derivative spaces lose broad, low-energy units (the loss grows with spike width: d' x0.6 at 1.5x the pyramidal width, x0.44 at 2x) that the raw space keeps; the raw block puts them back. Needs .spk.standard beside the method .spk. |
| `--hybrid-pcs` | `2` | FK_SESSION_HYBRID_PCS. standard-basis components per channel in the raw block (1-3) |
| `--hybrid-scale` | `1.0` | FK_SESSION_HYBRID_SCALE. weight of the raw block relative to the method block after the two are put on equal total variance (1 = equal; <1 derivative-led, >1 raw-led) |
| `--hybrid-gain-fit` | `all` | FK_SESSION_HYBRID_GAIN_FIT. energy half of the spikes the block-equalising gain is fit over: all (default) \| low \| high. Over all, the big spikes set the gain and the low-energy spikes load more on the derivative block than the nominal share. — choices: `all`, `low`, `high` |
| `--hybrid-deriv-basis` | `session` | FK_SESSION_HYBRID_DERIV_BASIS. basis of the method (derivative) block: session (the .pca.<method> basis, default) \| high (refit per channel on the high-energy half of a spike sample -- the low half's derivative variance is nearly unstructured). — choices: `session`, `high` |
| `--exclude-clu` | — | per-spike exclusion mask in .clu form (nonzero = drop), aligned to the .res -- e.g. fiber-flag-artifacts' output. Excluded spikes never enter a chunk, so they cannot shape a cluster; they are emitted as cluster 0. |
| `--in-stage` (`--in`) | — | input stage: the clu <base>.clu.<in-method>.<elec>.<in-stage> that --exclude-units names clusters of ('' = the stage-less <base>.clu.<in-method>.<elec>). It must be aligned to the .res this run reads (same spike count); a stage with its own .res (e.g. decollided) is refused |
| `--in-method` | — | variant token of the input-stage clu (default: --method), e.g. the --out-variant alias a curated clu was saved under |
| `--exclude-units` | — | clusters of the input-stage clu whose spikes are excluded from the sort: comma-separated ids and ranges, e.g. 3,7,12-15. Excluded spikes never enter a chunk; see --exclude-as for what the output holds for them |
| `--exclude-as` | `carry` | carry (default): each excluded unit is written back as one cluster of its own (one atom), after the new fibers; ids 0 and 1 of the input go to 0. noise: excluded spikes are cluster 0, like --exclude-clu — choices: `carry`, `noise` |
| `--no-cluster-basis` | flag (off) | ignore the global .pca basis for the fine-split shape features and use a per-call local SVD (legacy behaviour) |
| `--clu-stage` | `fiber_session` | post-group stage tag for the clu: <base>.clu.<method>.<elec>[.<stage>] (default 'fiber_session'); pass --clu-stage '' for an untagged .clu |
| `--skip-extant` | `0` | FK_SESSION_SKIP_EXTANT. 1 = if the destination .clu this run would write already exists (--out, else <base>.clu.<out-variant or method>.<elec>[.<clu-stage>]), log it and exit before any compute -- a plan re-run resumes past its fiber-session step instead of regenerating hours of sort over the top of an extant (possibly hand-curated) one. Delete the file, or set 0 (the default), to rebuild. |
| `--emit-hierarchy` / `--no-emit-hierarchy` | flag (on) | emit the .clu/.clc/.clp microfiber triple (atoms = pre-link fine fragments, fibers = linked global ids) via FiberHierarchy, instead of a flat .clu only. --no-emit-hierarchy writes just the flat .clu (legacy). Ignored with --out. |
| `--gpu` | flag (off) | run the realign/whiten kernels on GPU (CuPy; needs the [gpu] extra) |
| `--jobs` (`-j`) | `1` | FK_SESSION_JOBS. parallel worker processes over chunks (1 = serial, the default; 0 = one per chunk, capped at the CPUs this process may use). Chunks are independent, and this is the axis that scales: with --jobs 1 the only parallelism is BLAS inside one chunk, whose matrices are far too small to repay a 64-way fork/join barrier -- and NumPy's bundled OpenBLAS has a build-time thread ceiling (openblas_get_config() prints MAX_THREADS) that no fiber-kit setting can raise. |
| `--job-mem-gib` | `4.0` | FK_SESSION_JOB_MEM_GIB. resident memory to budget per chunk worker, GiB, used only to bound --jobs 0 (0 = no memory cap). Measured 2.5-3.4 GiB for a 50k-spike 12-min g6 chunk with the hybrid stack; an explicit --jobs N is never reduced. |
| `--feature-align` | — | feature-building alignment: xcorr (default), centroid (pure, no refine -- adds the trough-position-vs-asymmetry structure to the clustering/linking features), or off (skip the sub-sample align entirely; features are built from the windows exactly as extracted). Does NOT touch committing alignment or fiber-realign. Overrides the FIBER_ALIGN env var. — choices: `xcorr`, `centroid`, `off` |
| `--subsample` / `--no-subsample` | flag (off) | enable (--subsample) or disable (--no-subsample) realign's per-spike sub-sample (parabolic) refine in the feature build; default leaves the FIBER_SUBSAMPLE env var / lever untouched (off). Reaches pool workers. |
| `--out` | — |  |
| `--cons-mode` | `off` | FK_SESSION_CONS_MODE > FK_CONS_MODE: which consolidation passes run (default off) — choices: `off`, `strip`, `knn`, `both`, `shed` |
| `--cons-max-dist` | `0.5` | FK_CONS_MAX_DIST (default 0.5) |
| `--cons-gmin` | `0.0` | FK_CONS_GMIN (default 0.0) |
| `--cons-gmax` | `10.0` | FK_CONS_GMAX (default 10.0) |
| `--cons-chan` | `0.0` | FK_CONS_CHAN (default 0.0) |
| `--cons-margin` | `0.85` | FK_CONS_MARGIN (default 0.85) |
| `--cons-own-q` | `0.9` | FK_CONS_OWN_Q (default 0.9) |
| `--cons-tgt-q` | `0.9` | FK_CONS_TGT_Q (default 0.9) |
| `--cons-tgt-scale` | `1.25` | FK_CONS_TGT_SCALE (default 1.25) |
| `--cons-twin` | `0.93` | FK_CONS_TWIN (default 0.93) |
| `--cons-tpl-cap` | `1024` | FK_CONS_TPL_CAP (default 1024) |
| `--cons-min-tpl` | `8` | FK_CONS_MIN_TPL (default 8) |
| `--cons-iters` | `1` | FK_CONS_ITERS (default 1) |
| `--cons-cap-amp` | `0.0` | FK_CONS_CAP_AMP (default 0.0) |
| `--cons-pure-iqr` | `0.0` | FK_CONS_PURE_IQR (default 0.0) |
| `--cons-agg-iqr` | `0.0` | FK_CONS_AGG_IQR (default 0.0) |
| `--cons-agg-every` | `3` | FK_CONS_AGG_EVERY (default 3) |
| `--cons-agg-minnew` | `30` | FK_CONS_AGG_MINNEW (default 30) |
| `--cons-agg-vac` | `0` | FK_CONS_AGG_VAC (default 0) |
| `--cons-tpl-align` | `0` | FK_CONS_TPL_ALIGN (default 0) |
| `--cons-park-min` | `0` | FK_CONS_PARK_MIN (default 0) |
| `--cons-shed-k` | `3.0` | FK_CONS_SHED_K (default 3.0) |
| `--cons-shed-min` | `40` | FK_CONS_SHED_MIN (default 40) |
| `--cons-knn-k` | `20` | FK_CONS_KNN_K (default 20) |
| `--cons-knn-thr` | `0.3` | FK_CONS_KNN_THR (default 0.3) |
| `--cons-knn-minref` | `50` | FK_CONS_KNN_MINREF (default 50) |
| `--cons-knn-minnew` | `30` | FK_CONS_KNN_MINNEW (default 30) |
| `--cons-knn-dims` | `16` | FK_CONS_KNN_DIMS (default 16) |
| `--cons-fold-thr` | `0.9` | FK_CONS_FOLD_THR (default 0.9) |
| `--cons-scorr` | `0.93` | FK_CONS_SCORR (default 0.93) |
| `--cons-off-thr` | `0.0` | FK_CONS_OFF_THR (default 0.0) |
| `--cons-resid-k-on` | `0.0` | FK_CONS_RESID_K_ON (default 0.0) |
| `--cons-resid-k-off` | `0.0` | FK_CONS_RESID_K_OFF (default 0.0) |
| `--cons-resid-win` | `0` | FK_CONS_RESID_WIN (default 0) |
| `--cons-resid-mingain` | `0.25` | FK_CONS_RESID_MINGAIN (default 0.25) |
| `--cons-resid-min` | `200` | FK_CONS_RESID_MIN (default 200) |
| `--cons-resid-iqr-max` | `0.0` | FK_CONS_RESID_IQR_MAX (default 0.0) |
| `--cons-resid-iqr-min` | `0.0` | FK_CONS_RESID_IQR_MIN (default 0.0) |
| `--cons-resid-align` | `chunk` | FK_CONS_RESID_ALIGN: which alignment the residual strip scores in (default chunk) — choices: `chunk`, `cluster`, `stored` |
| `--stochastic` | flag (off) | run the ensemble/consensus fibering diagnostic instead of the normal single pass |
| `--stochastic-draws` | `20` | number of resampled draws per chunk (per peel round) |
| `--stochastic-frac` | `0.8` | subsample fraction per draw. HIGH by default: a low fraction starves most fibers below min_group (a 0.5 draw drops 50-70%% of fibers on this data) |
| `--stochastic-link` | `average` | how draw-fibers are grouped into consensus fibers from pairwise template correlation. 'single' (transitive union-find) CHAINS anticorrelated sub-modes through intermediate shapes into one component -- it undercounts real units; 'average' (default) and 'complete' are agglomerative and will not weld a chain of distinct co-located cells. — choices: `average`, `complete`, `single` |
| `--stochastic-no-npz` | flag (off) | skip the diagnostic .fiberens.npz. It holds every fiber INSTANCE (draws x chunks) and both compressing and writing it cost real time on a large run; the clu/clc/clp triplet is unaffected |
| `--stochastic-fast-stats` | flag (off) | skip the per-instance curation metrics (adaptation residual, per-channel residual profile, within-chunk drift slopes), leaving them NaN. Clustering, consensus and the triplet do not read them, so this only empties columns of the diagnostic npz |
| `--stochastic-min-fiber-spikes` | `3` | consensus fibers owning fewer than this many spikes are purged to NOISE (cluster 0) in the written triplet; 1 disables. Not-yet-processed spikes go to cluster 1 and are unaffected (default 3) |
| `--stochastic-clu-method` | — | variant token for the written clu/clc/clp (default: the variant of the .spk actually resolved, e.g. stderiv_C5) |
| `--stochastic-match-corr` | `0.95` | template correlation above which two fibers from different draws are the SAME consensus fiber |
| `--stochastic-stable-freq` | `0.6` | a consensus fiber is 'stable' (and gets frozen when peeling) if it appears in at least this fraction of draws |
| `--stochastic-peel-rounds` | `0` | 0 = single ensemble pass (no peeling). N>0 = freeze stable fibers, remove their spikes, and re-run the ensemble on the residual, up to N times |
| `--stochastic-chunks` | — | restrict to these chunk indices (default: all) — useful for a quick look |
| `--stochastic-seed` | `0` | RNG seed for the draws |
| `--stochastic-jobs` | `1` | parallelise the (independent) resampling draws across this many worker processes (the same per-chunk worker fiber_session uses). Draws are deterministic regardless of job count (per-draw seeds spawned from --stochastic-seed). 1 = serial. |
| `--stochastic-write-clu` | flag (off) | also write a Klusters clu/clc/clp triplet from the per-spike majority vote, so the consensus fibers (.clu) and their sub-modes/branches (.clc) can be inspected in Klusters |
| `--stochastic-clu-tag` | `fiber_stochastic` | tag for the written triplet: <base>.clu.stderiv.<elec>.<tag> etc |
| `--stochastic-verbose` | flag (off) | per-chunk progress |

## Morphology modelling

Forward modelling, apart from the sort.  [morphology.md](morphology.md) is the manual.

### `fiber-morpho`

Morphology modelling experiments, as sub-commands; each writes an `.npz` beside its report.

Positional: `cmd`

| flag | default | description |
|---|---|---|
### `fiber-morpho-fetch`

Acquire, validate and record the provenance of neuron reconstructions.

Positional: `cmd`

| flag | default | description |
|---|---|---|
