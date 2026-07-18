# fiber-kit — configuration knobs (`FK_*`)

Stages read tuning knobs with precedence **CLI flag / plan `params:` > `FK_*` environment variable >
`$FK_CONFIG` yaml > built-in default**.  Set them in your `fiber-kit.yaml` (pointed to by `$FK_CONFIG`;
`fiber-kit-exp.yaml` is the shipped experimental profile), in the environment, or per-step in a plan.
The pipeline runner (`fiber-pipeline`) also maps a few `FK_SESSION_*` knobs to fiber-session flags.
Values below are the defaults in the shipped `fiber-kit-exp.yaml`.

## fiber-session

| knob | default | meaning |
|---|---|---|
| `FK_SESSION_FINE_METHOD` | `rkk` | rkk \| kk.  Per-fiber fine clusterer; rkk = masked-EM KlustaKwik with BIC model selection (robust to unequal covariances). Categorical. |
| `FK_SESSION_CHUNK_MIN` | `18` | 12 \| 8-20 (min).  Drift must be ~stationary within a chunk: choose so the probe drifts << site pitch (20 um) per chunk.  Too long smears templates across drift; too short starves per-chunk templates of spikes. |
| `FK_SESSION_RKK_DELETE` | `0` | 0 \| 0/1.  0 = session OVER-clusters: keep small non-singular rkk sub-clusters so refine/intrachunk adjudicate them.  1 = cull sub-min-group pieces (old behaviour; sheds fragments into the artifact bin -- not what session is for). |
| `FK_SESSION_MERGE_METHOD` | `sliding` | sliding \| template \| profile.  Coarse merge method; INERT when MERGE_CORR=0. |
| `FK_SESSION_MERGE_CORR` | `0` | 0 \| 0 or 0.88-0.93.  0 = session does NOT merge across chunks -- fiber-intrachunk stitches later WITH its refractory gate.  Set 0.90 to restore the old session sliding merge (premature: it welds co-active cells before refine can clean). |
| `FK_SESSION_RESPLIT_PASSES` | `0` (exp: 3) | iterative within-chunk residual-gated re-split (em_swap on target-channel residual) + correlation merge, to convergence; replaces Block A/B when >0. 0=off. |
| `FK_SESSION_RESPLIT_RESIDUAL_THR` | `0.08` | re-split only fibers whose amplitude-scaled max residual (+-8 @ RMS peak) exceeds this (~0.08 stderiv, ~0.15 standard). |
| `FK_SESSION_RESPLIT_TOPCH` | `3` | channels fed to em_swap (top residual variance). |
| `FK_SESSION_RESPLIT_MIN_REDUCTION` | `0.20` | keep an em_swap split only if it cuts target-channel variance by >= this. |
| `FK_SESSION_RESPLIT_MERGE_CORR` | `0.99` | correlation merge threshold inside the loop. |
| `FK_SESSION_RESPLIT_DETREND_EPISODE` | `0` (exp: 1) | strip the episode-position axis (direction covarying with spikes-after minus spikes-before in ±90 ms) from the residual before each em_swap, so a split cannot cut a cell along its own temporal gradient and manufacture an asymmetric CCG. |
| `FK_SESSION_RESPLIT_DETREND_WIN` | `90.0` | half-window (ms) for the episode-position count. |
| `FK_SESSION_RESPLIT_DETREND_MIN_N` | `100` | skip the detrend below this many spikes (covariance estimate too noisy). |
| `FK_STOCH_DRAWS` | `20` | fiber-stochastic: resampled draws per chunk. |
| `FK_STOCH_FRAC` | `0.8` | fiber-stochastic: subsample fraction per draw (high — a low fraction starves fibers below min_group). |
| `FK_STOCH_MATCH_CORR` | `0.95` | fiber-stochastic: template corr = "same fiber" across draws. |
| `FK_STOCH_STABLE_FREQ` | `0.6` | fiber-stochastic: recurrence fraction to call a consensus fiber stable / freeze it. |
| `FK_STOCH_PEEL_ROUNDS` | `0` | fiber-stochastic: 0 = single pass; N = peel stable fibers and re-run on residual. |
| `FK_STOCH_SEED` | `0` | fiber-stochastic: RNG seed for the draws. |
| `FK_SESSION_LINK` | `0` | 0 \| 0/1.  0 = --no-link: fiber-session does NOT assemble per-chunk fragments across chunks; the downstream stages (intrachunk/link) do all the stitching.  1 = restore the overlap-anchor per-fiber set-up linking (fiber SET-UP, not cross-fiber merging). |
| `FK_SESSION_CFIBER_GATE` | `1` | 1 \| 0/1.  cfiber affine-invariant SHAPE veto on coarse fragment merges (precision). |
| `FK_SESSION_DIPSPLIT` | `1` | 1 \| 0/1.  Dip-bimodal split within fibers -- over-cluster welds at session so the downstream stages have the real cells to stitch, not a pre-merged blob. |
| `FK_SESSION_INCL_ASSIGN` | `1` | 1 \| 0/1.  1 = keep spikes beyond the per-fiber inclusion radius by ASSIGNING them to that fiber (kept in the sort) instead of dropping them to the unsorted/artifact bin. FK_INCLUSION_K still sets the pure CORE used for geometry/templates; this only stops the good high-amplitude tail (~6-7% of spikes here) being lost.  0 = legacy drop. |
| `FK_SESSION_NO_NOISE` | `0` | 0 \| 0/1.  1 = sweep every remaining noise spike (below the inclusion radius / rejected / collision junk) into a single UNDEFINED FIBER -- a real cluster, not the noise cluster -- instead of dropping it.  For clean stderiv data; the undefined fiber is heterogeneous and gets cleaned/re-split in later steps.  0 = leave in the noise cluster. |

## fiber-refine

| knob | default | meaning |
|---|---|---|
| `FK_REFINE_LARGE` | `100` | 150 \| 100-300.  Split only clusters with >= this many spikes (need enough to estimate sub-structure).  Stock 800 skips most 60-700-spike units -> subtle pairs never split.  Lower = more aggressive splitting. |
| `FK_REFINE_MIN_GROUP` | `30` | 30 \| 25-50.  Min spikes per split piece; a piece below this is noise, not a unit. |
| `FK_REFINE_DROP_MIN` | `0` | 5 \| 0-30.  Cluster-KEEP floor (decoupled from MIN_GROUP, the split-PIECE floor): each refine iteration drops clusters SMALLER than this to the artifact bin.  At MIN_GROUP=30 this discarded session's small over-split fragments (8-30 spikes) wholesale -- tens of thousands of real spikes lost to artifact.  5 keeps real fragments for fiber-intrachunk to stitch and sheds only 1-4 spike noise; 0 keeps everything (nothing dropped); set =MIN_GROUP to restore the old lossy behaviour. |
| `FK_REFINE_DEDUP` | `0` | 0 \| 0/1.  0 = do NOT run the sub-floor spike-dedup pass (default).  It drops only ~200-300 near-coincident sub-threshold duplicate detections, but it RE-INDEXES the spike list, so the output clu no longer aligns 1:1 with the canonical .res and round-tripping curated clu files back into the session breaks.  1 (--dedup) re- enables it (only worth it if those duplicates are actually corrupting templates). |
| `FK_REFINE_MERGE_MIN_SIM` | `0.98` | 0.96 \| 0.92-0.97.  Merge-back similarity after splitting.  0.96 KEEPS over-splits for intrachunk to adjudicate locally; 0.92 re-merges eagerly here. |
| `FK_REFINE_MERGE_BUDGET` | `1.0` | 1.0 \| 0.5-1.0.  Merge-back refractory budget (%): a pair merges only if the merged unit's [floor,window) contamination stays <= this.  TO REDUCE AGGRESSIVE MERGING lower toward 0.5 (stricter refractory gate) and/or raise MERGE_MIN_SIM toward 0.97; the two gates are AND-ed, so either tightens it. |
| `FK_REFINE_SPLIT_MIN_CORR` | `0.95` | 0.93 \| 0.90-0.95.  Keep a split only if each piece is internally coherent (within-piece template correlation >= this); else the split was noise. |
| `FK_REFINE_CHUNK_MINUTES` | `18` | 12 \| 8-20 (min).  Refine chunk length; match FK_SESSION_CHUNK_MIN. |
| `FK_REFINE_CHUNK_JOBS` | `1` | 1 \| 1-N (processes).  Parallel worker processes over chunks in drift-aware mode (chunk-minutes>0).  Chunks are independent (own whitener + refine), so this is the main speedup for a chunked run; the cross-window link runs serially after. Workers re-open the .spkD/.fil memmaps (bounded memory).  No effect whole-session. |
| `FK_REFINE_AB_RECLAIM` | `1` | 0 \| 0/1.  0 = off (default; reproduces current output).  1 = run the reclaim pass. |
| `FK_REFINE_AB_DISTINCT` | `0.98` | 0.93 \| 0.90-0.95.  Max host-RESIDUAL/donor template shape-corr to treat them as two cells; >= this is ONE cell -> merge-back's job, not a reclaim.  Validated knee ~0.93 (recall ~100% below it, collapsing into the continuum above). |
| `FK_REFINE_AB_ABS` | `0.50` | 0.50 \| 0.40-0.70.  Absolute donor-shape floor a spike must reach (a loose noise floor only; the real DECISION is the relative margin below, since a per-spike stderiv corr tops out ~0.69 even against its own template). |
| `FK_REFINE_AB_MARGIN` | `0.05` | 0.05 \| 0.02-0.10.  How much more a spike must match the donor than its host to move -- THE precision/recall dial.  Higher = fewer moves, less false-grab (and less recall); 0.05 gave 0% false-grab on g5 ground-truth injection.  Lower only under curation. |
| `FK_REFINE_AB_MIN` | `10` | 10 \| 5-30.  Min spikes a donor must reclaim from one host to commit the move. |
| `FK_REFINE_AB_SIGCAP` | `2000` | 2000 \| 1000-8000 (0=no cap).  Spikes used to estimate each cluster's median TEMPLATE (not membership): the iterated align is the reclaim's bottleneck, and on a whole-session merged cluster (tens of thousands of spikes) a ~2000-spike sample matches the full template to ~1e-3 (validated) -- ~38x faster on the big clusters. Matches the intrachunk FK_INTRA_SIG_CAP convention. |
| `FK_REFINE_AB_JOBS` | `1` | 1 \| 1-N (threads).  Worker threads for the reclaim's per-cluster template precompute (the align bottleneck).  Result is IDENTICAL for any value -- templates are sampled before the parallel work, so it is a pure speedup, not a tuning knob.  Helps in WHOLE-SESSION mode (big clusters); in chunked mode clusters are small so it barely matters -- there the lever is FK_CHUNK_JOBS (the per-chunk pool), not this. |

## fiber-peel

| knob | default | meaning |
|---|---|---|
| `FK_PEEL_FOOT_HI` | `0.97` | 0.97 \| 0.95-0.99.  Anneal START: strictest footprint cosine; the most certain merges (the interneuron's own fragments, cos ~1.0) fuse first. |
| `FK_PEEL_FOOT_LO` | `0.90` | 0.90 \| 0.88-0.95.  Anneal FLOOR: loosest cosine accepted.  Best g5 result at 0.90; lower re-merges more (recall) but risks fusing same-site distinct cells (lean on the refractory veto), higher leaves more over-split for intrachunk/link. |
| `FK_PEEL_REFRAC_MS` | `2.0` | 2.0 \| 1.5-2.5 (ms).  Refractory half-window for the veto cross-CCG. |
| `FK_PEEL_REFRAC_THR` | `0.3` | 0.3 \| 0.2-0.4.  Coincidence ratio (obs/exp) above which the pair is two cells -> veto. |
| `FK_PEEL_REFRAC_MIN_EXP` | `5.0` | 5.0 \| 3-10.  Min expected coincidences for the veto to be POWERED; below this it abstains (merge allowed on footprint alone) -- so it only ever removes a false merge. |
| `FK_PEEL_REFRAC_CENSOR_MS` | `0.0` | 0.0 \| 0-0.5 (ms).  Censor window dropping duplicate detections of one spike. |
| `FK_PEEL_MIN_N` | `15` | 15 \| 10-40.  Min spikes for a fragment to participate (footprint/CCG reliability). |

## fiber-intrachunk

| knob | default | meaning |
|---|---|---|
| `FK_INTRA_GATE` | `band` | band \| cfiber \| cosine \| mmd \| kcov.  Shape gate.  band = energy-scaled median+/-sigma mutual-centre-dependent cosine and self-calibrates from the split-half null; it is marginally better on the hard look-alike subset (cosine>0.85) -- the regime that matters under drift.  Set cosine to revert. |
| `FK_INTRA_COS_THR` | `0.90` | 0.85 \| 0.82-0.88.  Cheap cosine recall PREFILTER for merge candidates (the real shape decision is the gate above). |
| `FK_INTRA_OFF_THR` | `1.0` | 1.0 \| 0.8-1.2 (samples).  Inter-channel offset RMS gate: fragments of one neuron share per-channel timing; > ~1 sample mismatch implies a different source. |
| `FK_INTRA_DEPTH_GATE` | `10.0` | 35 \| 25-45 (um).  Depth gate.  On 20-um pitch a single neuron's localized depth wanders < ~1.5 sites; 35 um ~ 1.75 sites caps drift and rejects far-apart merges. |
| `FK_INTRA_AMP_GATE` | `0.90` | 1.10 \| 0.92-1.39 (nat-log; 0=off).  Absolute log-amplitude (energy) gate.  ln(3)=1.10 caps a merge to <=3x amplitude ratio (a single unit's drift amplitude range). |
| `FK_INTRA_LINKAGE` | `complete` | complete \| dynamic \| ms.  Agglomeration linkage.  complete (farthest-point) yields tight, conservative clusters; dynamic/ms are density-adaptive. |
| `FK_INTRA_CFIBER_Q` | `0.95` | 0.90 \| 0.85-0.95.  cfiber self-calibration quantile: threshold = Q-quantile of the per-fragment split-half null.  Higher = stricter. |
| `FK_INTRA_SIG_CAP` | `8000` | 8000 \| "" = no cap.  Per-fragment spikes used for the mean template; beyond ~8000 the template is already stable, so the cap only bounds compute. |
| `FK_INTRA_WARP_THR` | `0.3` | 0.3 \| 0.2-0.5 ("" off).  Group-delay (warp) coherence gate (Omlor-Giese): merge only if cross-channel corr of the two per-channel group-delay profiles >= this. Group- delay is noisy at low spike count -> keep LOW (~0.3). |
| `FK_INTRA_WARP_RESID_THR` | `1.0` | 1.0 \| 0.75-1.5 ("" off).  Single-channel warp-incongruity SUB-gate: among already- coherent pairs, veto if any ONE centroid-range channel's group-delay residual (Theil-Sen line) > this many samples (a strong-channel-masked different source). |
| `FK_INTRA_OFF_THR_INT` | `0.5` | 0.5 \| 0.4-0.6 ("" off).  DUAL gate: offset RMS threshold for INTERNEURON pairs (narrow trough-to-peak). Fast cells have stable offsets (~0.23) so off_thr=1.0 is inert; tighten to ~0.5. Needs raw .spk for cell-typing. |
| `FK_INTRA_OFF_THR_PYR` | `1.0` | 1.0 \| 0.9-1.1 ("" off).  DUAL gate: offset RMS threshold for PYRAMIDAL pairs (wide trough-to-peak). Set BOTH _INT and _PYR to enable the dual gate; mixed -> stricter. |
| `FK_INTRA_ITER` | `5` | 5 \| 1 = single pass.  Iterate group->re-estimate->regroup to convergence (cap 5; early-stops when a pass merges nothing).  NOT a gate: each pass AGGREGATES the partial merges and re-signs the units, so the same tight gate keeps finding the true merges a single pass left -- denoising, not gate-loosening.  Production default is 1 (config.py); this experiment runs the within-chunk merge to convergence to shed the residual over-split before linking. |

## fiber-link

| knob | default | meaning |
|---|---|---|
| `FK_LINK_COS_THR` | `0.75` | 0.75 \| 0.70-0.85.  Cosine PREFILTER for cross-chunk candidates.  Relaxed to 0.75 (vs 0.85) added 7 clean bundles: recall matters here because the precision is carried by the position, cfiber and amplitude gates below. |
| `FK_LINK_CFIBER_Q` | `0.90` | 0.90 \| 0.85-0.95 ("" = off).  cfiber CO-GATE (cosine AND cfiber); threshold self-calibrates from the overlap-backbone same-unit shape distances at link time. |
| `FK_LINK_WARP_THR` | `0.3` | 0.3 \| 0.2-0.5 ("" = off).  Group-delay (warp) co-gate: link only if per-channel timing profiles also agree.  Earlier "g5 no-op" was PRE-DR-candidates (precision saturated); with template-DR candidates ON the widened candidate set over-merges, and warp discriminates same/different units at AUC ~0.87 on these units -- an independent cross-timing veto cosine lacks, useful in the co-located dense regions. |
| `FK_LINK_WARP_AMP_THR` | `0.7` | 0.7 \| 0.6-0.8 ("" = off).  Omlor-Giese amplitude-profile floor (eq.10) -- the FULL two-term warp criterion (group-delay eq.11 AND amp-profile eq.10).  On g5 co-located look-alikes eq.10 (AUC ~0.76) out-discriminates group-delay alone (~0.65); adding it trims a few more over-merges at zero recall cost (multi 311->303, kept 85.9%). |
| `FK_LINK_WARP_RESID_THR` | `1.0` | 1.0 \| 0.75-1.5 ("" = off).  Single-channel warp-incongruity SUB-gate: among warp-coherent pairs, veto one whose worst single-channel group-delay residual exceeds this (samples) -- the co-located source the aggregate correlation masks. |
| `FK_LINK_POS_THR` | `1.5` | 1.5 \| 1.0-2.0.  Position gate: link across chunks only if localized positions agree within this (in the localizer's units).  Tighter rejects drift-jumped mismatches. |
| `FK_LINK_OFF_THR` | `1.0` | 1.0 \| 0.8-1.2 (samples).  Inter-channel offset gate (same timing-signature logic as intra). |
| `FK_LINK_MAX_GAP` | `2` | 2 \| 1-4 (chunks).  Max chunk gap a unit may vanish for and still be bridged (set by how long a neuron can fall silent or drift off-probe before returning). |
| `FK_LINK_AMP_GATE` | `1.39` | 1.39 \| 1.10-1.61 (nat-log; 0=off).  Absolute log-amplitude gate across the WHOLE recording's drift.  ln(4)=1.386 -> <=4x amplitude cap (wider than intrachunk's 3x: more total drift). |
| `FK_LINK_MIN_A` | `50` | 50 \| 0=off.  Absolute amplitude (A) floor on linkability, SEEDS INCLUDED.  ~115 g5 units sit at the localization floor A~1 (next percentile ~490); a backbone seed touching one welds a noise unit into a huge-amplitude bundle (worst 52750x A ratio).  Floor drops them: worst A-ratio 52750x -> 21x, backbone-pair recall 93.5% -> 85.9% (the 7.6% lost are exactly those noise pairs).  Session-scaled: set between the noise floor and real units. |
| `FK_LINK_AMP_SPAN` | `1.79` | 1.79 \| 0=off (nat-log).  Varbound bundler: refuse a NON-seed union whose bundle logA span would exceed this.  ln(6)=1.79 -> 6x.  amp_gate caps 4x PER LINK but a chain of <=4x links accumulates unbounded amplitude; this bounds the accrued bundle span (seeds bypass). |
| `FK_LINK_ALIGN_LAG` | `6` | 0=off \| 6 recommended (native samples).  Sub-sample template re-registration before the cosine gate.  Integer mutual_center leaves a fractional-sample residual that drops a true same-neuron cross-chunk cosine under threshold; re-registering recovers it (+25% of links). |
| `FK_LINK_ALIGN_UPSAMPLE` | `4` | 4 \| 1-4.  Cubic-spline upsample factor for the align-lag search (sub-sample precision). |
| `FK_LINK_PRIMARY_AMP_FRAC` | `0.30` | 0=off \| 0.30 recommended.  Restrict the cosine gate to channels BOTH fragments treat as primary (p2p >= frac*max).  The near-threshold channels carry mostly noise and decorrelate true same-neuron pairs; the intersection (g5 median 5 of 8) lifts those links over threshold. Recall lever -- keep the FK_LINK_OFF_THR (and FK_LINK_TAN_THR) precision guards ON with it. |
| `FK_LINK_TAN_THR` | `""` | "" = off \| 0.5 recommended.  Energy-tangent (microfiber) CO-GATE: require the cosine of the two fragments' energy-direction tangents (high- minus low-energy template) >= this.  Precision guard on the primary-amp-frac recall.  Needs a per-fragment 'tangent' array in the cpos/units table (fiber-cpos/-intrachunk emit; inert until then). |
| `FK_LINK_DR_CANDIDATES` | `1` | 1 \| 0/1.  1 = template-DR candidate space; 0 = legacy physical position-NN. |
| `FK_LINK_DR_K` | `10` | 10 \| 6-16.  Template-DR dimensionality (~96% variance on g5). |
| `FK_LINK_DR_THR` | `""` | "" = no DR-space NN distance cap (the cosine/cfiber/pos/amp co-gates do the filtering). |
| `FK_LINK_BUNDLE` | `varbound` | varbound \| chunkexcl.  varbound = the new anti-chaining bundler (this first attempt); chunkexcl = the legacy cosine-ordered union-find (set this to revert exactly). |
| `FK_LINK_VAR_SCALE` | `1.0` | 1.0 \| 0.7-3.0.  Dial on the self-calibrated variance boundary.  1.0 = the conservative floor (gate the worst over-spread merges); <1 splits more (tighter, for stubborn over-merge); >1 merges more (toward the un-gated baseline, ~3x ~ legacy behaviour). |
| `FK_LINK_N_PC` | `12` | 12 \| 6-16.  Template principal components defining the variance space the boundary is measured in (matches the fiber-calibrate/-defrag convention). |
| `FK_LINK_VAR_ALLOW` | `""` | "" = self-calibrate at link time from the high-energy backbone edges (recommended). Set an explicit float only to pin the boundary across runs/sessions. |

## fiber-backbone-link

| knob | default | meaning |
|---|---|---|
| `FK_BBLINK_MIN_SNR_Q` | `0.5` | 0.5 \| 0.0-0.8.  START-WITH-HIGH-SNR floor: link only clusters whose SNR (dom-channel amplitude / spike-to-spike spread) is at/above this QUANTILE; below it are left as singletons for the refinement pass.  0 = link all. |
| `FK_BBLINK_Z` | `1.0` | 1.0 \| 0.8-1.5 (sigma).  Band half-width; median +/- z*sigma.  1 sigma is tight enough to drop co-located impostors; the literal SEM CI goes pencil-thin and fails. |
| `FK_BBLINK_WIN` | `8` | 8 \| 6-10 (samples).  Half-window around the fragment's RMS-energy centre. |
| `FK_BBLINK_SLIDE` | `4` | 4 \| 2-6 (samples).  +/- lag search for the max-overlap alignment. |
| `FK_BBLINK_IOU_THR` | `0.5` | 0.5.  Per (sample x channel) interval-IoU counted at >= this (the '50% overlap'). |
| `FK_BBLINK_FLOOR` | `0.55` | 0.55 \| 0.5-0.7.  Min mean-IoU to accept a mutual-NN link (mutual-NN + warp veto carry precision; this is a floor, not the main gate). |
| `FK_BBLINK_PRIM_FRAC` | `0.30` | 0.30 \| 0.2-0.4.  Backbone = channels with p2p >= frac*max in BOTH fragments (the shared primary footprint); --channels pins an explicit set instead. |
| `FK_BBLINK_WARP_THR` | `0.5` | 0.5 \| 0.3-0.9.  Omlor-Giese group-delay coherence (eq.11).  NEAR-DEGENERATE on the octrode (units span few channels) so RELAXED; the amp-profile term does the gating. |
| `FK_BBLINK_AMP_THR` | `0.85` | 0.85 \| 0.75-0.92.  Omlor-Giese amplitude-profile correlation (eq.10) -- the EFFECTIVE veto here (spatial-footprint agreement across the full 8 channels). |
| `FK_BBLINK_RESID_THR` | `1.0` | 1.0 \| 0.75-1.5 (samples).  Single-channel warp-incongruity ceiling (self-gates at warp>=0.85; a no-op below, so it only vetoes already-coherent look-alikes). |
| `FK_BBLINK_MIN_FRAG` | `40` | 40 \| 30-60.  Min spikes for a fragment to carry a template/CI (else left as-is). |
| `FK_BBLINK_MAX_GAP` | `1` | 1 \| 1-2 (chunks).  Bridge a cell silent for up to this many chunks. |
| `FK_BBLINK_CX_SCALE` | `0.0` | 0.0 \| 0.0-1.0.  COMPLEXITY scaling of the mutual-NN overlap floor: simpler (shift- insensitive) fragments must overlap harder.  0=off (opt-in; mild effect, tune on review). |

## fiber-xcorr-merge

| knob | default | meaning |
|---|---|---|
| `FK_XCM_COS_THR` | `0.99` | 0.99 \| 0.985-0.999.  Min roll-shift cosine to merge; start HIGH. |
| `FK_XCM_SHIFT` | `4` | 4 \| 2-6 (samples).  +/- circular-shift half-window (Klusters xcorr). |
| `FK_XCM_REFRAC_MS` | `2.0` | 2.0 \| 1.5-3.0.  Refractory window; 0 disables the CCG veto. |
| `FK_XCM_REFRAC_THR` | `0.3` | 0.3.  CCG obs/exp ratio above which a powered pair is vetoed as two cells. |
| `FK_XCM_REFRAC_MIN_EXP` | `5.0` | 5.0.  Min expected coincidences for the veto to have power (else it abstains). |
| `FK_XCM_MIN_N` | `40` | 40 \| 30-100.  Min spikes for a cluster to participate (else carried as-is). |
| `FK_XCM_SPK_CAP` | `300` | 300.  Spikes per cluster used to build/realign the template. |
| `FK_XCM_BAND_THR` | `0.5` | 0.5 \| 0.0-0.8.  Median+/-sigma BAND-OVERLAP co-gate (fiber-backbone-link's method), ON: a candidate merge must also have band-overlap IoU >= this.  0=off (roll-cosine only). |
| `FK_XCM_CX_SCALE` | `0.0` | 0.0 \| 0.0-1.0.  COMPLEXITY scaling: raise the required cosine for LOW-complexity (shift-insensitive) pairs -- their high roll-shift cosine is weak evidence (the best circular shift inflates it).  0=off.  Moderate signal on g5 (AUC 0.73 correct-vs- over-merge) but a MILD practical effect; opt-in and tune on review. |

## other

| knob | default | meaning |
|---|---|---|
| `FK_CFIBER_Q` | `0.90` | 0.90 \| 0.85-0.95.  Within-fiber shape veto: reject a candidate whose cfiber (affine-invariant Fourier descriptor) distance exceeds the Q-quantile of the fiber's OWN split-half null.  0.90 = tolerate up to the 90th-pct self-jitter. Higher = stricter veto (purer, fewer accepted). |
| `FK_FEATURE_ALIGN` | `xcorr` | xcorr \| centroid \| off.  Sub-sample align of the channel-summed waveform before featurizing.  A misalignment of d samples decorrelates two identical spikes by ~(d/W)^2 (W = spike width); for narrow interneurons (W~4 samp) even d=1 hurts. The #1 lever for separating subtle look-alike shapes.  xcorr aligns to the summed template; centroid uses the energy centroid; off disables. |
| `FK_INCLUSION_K` | `2.0` | 2.5 \| 2.0-3.0.  Per-fiber core radius = median + k*MAD of within-fiber distance. Gaussian MAD=0.6745*sigma, so k=2.5 -> median+1.69*sigma (~91% of a unimodal core).  Lower k = purer cores (drop contaminant tails); higher = more inclusive. |
| `FK_DIP_DIM` | `4` | 6 \| 4-8.  PCA dims the Hartigan dip test runs on.  More dims expose footprint / secondary PCs that separate co-located pairs; too many adds pure-noise dims. |
| `FK_DIP_ALPHA` | `0.02` | 0.02 \| 0.01-0.03.  Dip-test p to REJECT unimodality and bisect.  Statistical gate: >0.05 starts splitting on sampling noise; <0.01 misses real bimodality. |
| `FK_DIP_MIN` | `30` | 30 \| 30-60.  Min spikes to attempt a dip bisection (need enough for a stable dip statistic; below this the test is underpowered). |
| `FK_EBAND` | `1` | 1=on (validated) \| 0=off (falls back to --fine-method per fiber). |
| `FK_EBAND_WIDTH` | `0.40` | 0.45 \| 0.35-0.60 (decades).  Band width; 0.45 dec ~ factor 2.8 in amplitude. Narrower = more, finer but sparser bands. |
| `FK_EBAND_OVERLAP` | `0.2` | 0.20 \| 0.15-0.25 (decades).  Band overlap so the overlap-anchor relink has shared spikes to stitch adjacent bands. |
| `FK_EBAND_CONFOUND` | `0.4` | 0.40 \| 0.30-0.50.  Band a fiber only if PC1 R^2 vs log-energy >= this (its main shape axis IS energy).  Avoids banding shape-clean fibers. |
| `FK_EBAND_MIN_SPAN` | `0.6` | 0.60 \| 0.50-0.80 (decades).  Band only if the fiber spans >= this many decades (~4x); below that there is nothing to band. |
| `FK_EBAND_LOW_ASSIGN` | `0.2` | 0.20 \| 0.0-0.3.  Bottom fraction of the energy range made ASSIGNMENT-ONLY (noise- floor spikes are assigned from the bands above, never used to seed a band). |
| `FK_RKK_DELETE` | `1` | 1 \| 0/1.  1 = rkk's CEM culls sub-min-group pieces mid-split (they fall through to the residual/artifact bin).  Set 0 (--no-rkk-delete) to keep small, non-singular rkk sub-clusters as real units when too many spikes are landing in the artifact cluster.  Trades a fuller sort for some extra over-split. |
| `FK_REALIGN_AFTER` | `"fiber-refine fiber-peel fiber-intrachunk fiber-link"` | stages after which to re-roll each new group's spikes onto its unit template (Klusters method).  A split/merge across drift leaves a group's troughs spread; a per-unit realign re-tightens them.  Set "" to disable.  (fiber-session is already followed by the stage-2 fiber-realign.) |
| `FK_REALIGN_AFTER_MODE` | `shift-spk` | shift-spk \| reextract.  shift-spk circular-rolls the existing .spk (no .fil, fast, validated trough-std 2.00->1.05); reextract re-reads the .fil (recovers samples that drifted out of the extraction window, at the cost of .fil I/O each step). |
| `FK_REALIGN_FINAL_MODE` | `reextract` | mode for the LAST realign of the run (the artefacts that go to curation): reextract re-reads the .fil at the corrected times for the cleanest final .spk/.fet.  Set to shift-spk to make the final step match the intermediate ones (no .fil read). |
| `FK_SPLIT_VAR_MULT` | `1.5` | 1.5 \| 1.3-2.0.  Split only clusters whose top-3 feature variance exceeds this x the median cluster's -- a variance-ratio that targets the bloated (mixed) clusters. |
| `FK_FOLD_OFF_THR` | `0.22` | 0.22 \| 0.20-0.25 (samples).  Inter-channel offset (timing) veto on a contaminant fold: reject folding in a contaminant whose per-channel offset RMS differs by more than this -- timing is a source-identity signature cosine ignores. |
| `FK_DEDUP_STALE` | `quarantine` | quarantine \| error \| skip.  Handling of stale leftover per-spike files (.clu/.fet from a prior, differently-counted run) a fresh dedup cannot subset.  quarantine renames each aside as <file>.stalebkp (non-destructive, re-runnable); error blocks; skip leaves them.  (Legacy FK_DEDUP_STRICT: 1->error, 0->skip; superseded.) |
| `FK_MERGE_WARP_RECALL` | `""` | "" = OFF (default).  Warp-RECALL merge path (Omlor-Giese group-delay, patch 0086): reunite drift fragments cosine misses when median-template warp >= this AND amplitude-profile corr >= FK_MERGE_AMP_THR.  It admits merges BELOW the cosine floor -- i.e. drift-fragment STITCHING -- and on a full session that over-merges (~2x extra sub-cosine edges; the g5 0.976 precision does not generalise).  That reunion is fiber-intrachunk's job (it has the offset/depth/ cfiber/refractory-ceiling gates).  Set 0.9 only if curation shows refine must reunite drift fragments locally. |
| `FK_MERGE_AMP_THR` | `0.7` | 0.7 \| 0.6-0.8.  Amplitude-profile correlation floor for the warp-recall path (the second of the two Omlor-Giese terms; magnitude eq.10 + group-delay eq.11). |
| `FK_MERGE_WARP_THR` | `""` | "" = off.  Warp PRECISION gate on cosine-selected merges (reject a cosine merge whose group-delay disagrees).  Tighter precision at some recall cost; tune vs curation. |
| `FK_MERGE_WARP_RESID_THR` | `1.0` | 1.0 \| 0.75-1.5 ("" off).  Single-channel warp-incongruity SUB-gate on the warp gate/recall: among coherent-overall pairs, veto if any ONE channel's group-delay residual (Theil-Sen line) > this many samples (a strong-channel-masked source). Self-gates at warp >= 0.85 (no-op below). |
| `FK_MERGE_WARP_RESID_THR_INT` | `""` | "" = off.  Cell-type-aware warp-resid gate: threshold for pairs touching an INTERNEURON (narrow trough-to-peak).  Interneurons fire fast -> very stable per- channel timing, so a single-channel group-delay residual is a real different-source signature -> tighter (~0.7).  Needs raw .spk (loaded when set).  Set BOTH _INT+_PYR to enable the dual gate; each falls back to FK_MERGE_WARP_RESID_THR if only one set. |
| `FK_MERGE_WARP_RESID_THR_PYR` | `""` | "" = off.  PYRAMIDAL threshold (wide trough-to-peak): timing jitters more, so a larger residual is benign (~1.3).  A pair touching an interneuron uses the stricter _INT. |
| `FK_REFRAC_CEILING` | `0.5` | 1.0 \| 0.5-2.0 (%, ""=off).  Reject a merge whose COMBINED 2 ms-ISI refractory violation exceeds this percent -- two genuine units merged manufacture cross-refractory spikes. |
| `FK_PRE_MERGE_COS` | `0.98` | 0.97 \| 0.96-0.98 (0=off).  Pre-collapse obvious mutual-NN pairs at cosine >= this before the full gate (cheap, safe consolidation that shrinks the agglomeration). |
| `FK_ALIGN_LAG` | `6` | 6 \| 4-8 (native samples; 0=off).  Best-lag half-window: align two templates within +/- this before scoring, so a sub-sample jitter is not read as a shape difference. |
| `FK_ALIGN_UPSAMPLE` | `1` | 1 \| 1-4.  Cubic-spline upsampling factor for the align-lag search (sub-sample precision); 1 = native-rate search.  Higher = finer alignment at compute cost. |
| `FK_CFIBER_NULL` | `order` | order \| energy.  Split-half null basis.  order splits a fragment by spike time (drift- robust); energy splits by amplitude (sensitive to amplitude structure). |
| `FK_CFIBER_THR_FLOOR` | `0.0` | 0.0 = off.  Absolute floor on the self-calibrated cfiber threshold, guarding against a pathologically tight null on a very clean fragment. |
| `FK_BACKBONE_STD_COS` | `0.75` | 0.75 \| 0.5-0.85 (0=off).  STANDARD median-template cosine a seed must also clear. On g4 keeps 100% of same-unit seeds, rejects 100% of the anti-correlated ones. |
| `FK_BACKBONE_WARP` | `0.3` | 0.3 \| 0.2-0.5 ("" = off).  Group-delay (warp) coherence co-gate on the seed, ON here. NOTE (validated g4): on this octrode most units span <=2 channels, so the per-channel group-delay is degenerate and warp is NOT discriminative (same/diff both ~0) -- at 0.3 this REJECTS most seeds, shrinking the backbone.  That is deliberately conservative: it pushes the linker toward under-linking (collinear time-bands to rejoin by ISI) rather than trusting stderiv-similar anchors.  Set "" to fall back to the std-cos gate alone. |

