# fiber-kit — core methods & primitives

The reusable numerical primitives behind the stages, with their signatures and parameters.
These live in `fiber_geometry` (shape/warp) and `fiber_ccg` (spike-train refractory tests), and are
shared by the linkers (`fiber-backbone-link`, `fiber-xcorr-merge`) and the within-chunk merger
(`fiber-intrachunk`).  See [stages.md](stages.md) for the CLI parameters that expose them.

## Waveform shape & matching

### `band_overlap(med_a, sd_a, med_b, sd_b, chans=None, *, z=1.0, win=8, slide=4, iou_thr=0.5)`

Energy-scaled median+/-z*sigma band-overlap between two cluster templates: the per-sample
interval IoU of their [median +/- z*sigma] bands, each windowed on its RMS-energy centroid
(+/- win samples), the second slid +/- slide samples for the best overlap, each band normalised
to unit energy over the compared window x channels (spike-to-spike variance scales with waveform
energy).  Returns the mean IoU (higher = more consistent shape at matched scale), or nan.  This is
fiber-backbone-link's ci_overlap as a reusable primitive.  med_*, sd_*: (nsamp, nchan) template and
per-sample std.  chans: channel subset (None = all channels).

### `waveform_complexity(template, shift=1)`

Shift-sensitivity of a mean waveform: 1 - cos(t, roll(t, shift)) on the DC-removed template
(flattened over samples x channels).  A simple, broad waveform is shift-INSENSITIVE (low value)
-- a high roll-shift cosine to it is weak evidence, since the best circular shift inflates the
match -- while a complex, peaked waveform is distinctive (high value).  Downstream merge/link
thresholds scale with this: demand more similarity from low-complexity clusters.  On g5 it
separates co-located over-merges from correct merges at AUC ~0.73.  template: (nsamp, nch).

### `dispersion_profile(waves, sigma=1.0, aligned=False)`

Per-channel spike-to-spike DISPERSION profile sigma(t) of a cluster -- the (nsamp, nch)
half-width of the per-channel confidence band, the second-moment companion to the MEDIAN
template.  From an (nspk, nsamp, nch) raw stack: realign (fl.realign) -> denoise ->
mutual_center_spikes -> per-sample std over spikes, so sigma(t) is measured on the SAME
aligned stack the median template is (pass aligned=True to skip that when the caller already
aligned, e.g. beside _boundary_std_template).

This is sigma(t), NOT the SEM sigma(t)/sqrt(n): deliberately n-INDEPENDENT so it is a
cross-fragment identity signature and not a sample-size artefact (low-count fragments would
otherwise read a wide band purely from small n).  Where the band is WIDE is where the
cluster's spikes disagree: amplitude variability (bursting -> wide at the peak/trough),
temporal jitter (-> wide at the edges), or -- on peripheral channels -- background noise.
Returns None for < 2 spikes.

The plain std is biased UP by near collisions (a second spike inside the window); see
[Collision-robust envelope](#collision-robust-envelope-fiber_envelope) for the robust counterpart.

### `mutual_center_spikes(waveforms, ref_sample=16)`

mutual_center applied to a (nspk, nsamp, nchan) stack: shift every spike by
the single offset that brings the cluster-mean dominant trough to ref_sample
(rigid whole-cluster shift; preserves within-cluster structure realign set up).

### `denoise(waveforms, sigma=1.0)`

Strip the high-frequency noise floor off RAW footprints before building
the curve, by Gaussian-smoothing along the sample axis (-2), per channel.
waveforms: (..., nsamp, nchan) realigned (un-whitened) waveforms.

The geometry curve is built from raw footprints (cross-chunk comparable,
unlike whitened features), so it carries the recording noise floor; that
floor is what spreads same-unit curves and caps the strict link recall.
Calibrated on real g5 (curated identity, temporal split): sigma~1.0 lifts the
perfectly-separable same-unit fraction 0.68 -> 0.95 AND pushes the nearest
different-unit pair 1.23 -> 1.61 (it denoises without collapsing the fine
timing structure that separates near-duplicate units).  A linear Gaussian
beats a 5-pt median here because the noise is ~white and the median's
nonlinear peak/trough clipping distorts the discriminative shape.  Over-
smoothing is the failure mode: sigma>=2 starts merging fine-structure
near-duplicates (the different-unit floor falls back below the unfiltered
value), so keep sigma ~1.  sigma<=0 disables.

### `interchannel_offsets(template, amp_frac=0.3, method='trough', up=8, maxlag=None)`

Per-channel sub-sample timing (samples) of each channel relative to the dominant
channel.  Channels below amp_frac of the dominant peak-to-peak carry no reliable
timing -> NaN.  template: (nsamp, nchan) realigned (ideally denoised) mean template.

method="trough" (default, unchanged): parabolic sub-sample minimum of the trough.
    Cheap, but the trough sample is jittery at low spike count and on the stderiv
    waveform (multi-extremum) -- measured split-half noise ~8 samples on a 27-spike
    cluster, which swamps the genuine sub-sample inter-channel timing.
method="xcorr": upsampled cross-correlation LAG of the FULL channel waveform against
    the dominant channel (a matched filter -- uses the whole shape, not one sample).
    Far more stable: on g5 the same-neuron split-half noise drops to ~0.1 samples on
    a RAW template (use raw, not stderiv -- the stderiv trough is noisy), so a real
    0.26-sample inter-channel difference between two co-located cells becomes a clean
    2-3 sigma discriminator even at ~30 spikes.  NOTE the lag scale differs from the
    trough scale, so off_thr must be re-calibrated (~0.2-0.3, not 1.0) when using it.

## Drift-warp veto (Omlor–Giese)

### `group_delay_profile(template, sr=32552.0, band=(300.0, 9000.0), amp_frac=0.3)`

Per-channel GROUP DELAY (samples, relative to the dominant channel) of a template, from the
cross-spectrum phase slope:  gd_c = -d/domega arg(F[c] * conj(F[ref])).  This is the per-channel
delay ('warp') of the Omlor-Giese anechoic mixing model x_c(t)=alpha_c*s(t-tau_c) -- a neuron's
octrode footprint is one delayed source, so gd_c is its spatial-temporal signature.  Uses the
whole phase spectrum, so it is steadier than a single trough/lag.  Channels below amp_frac of the
dominant peak-to-peak carry no reliable phase -> NaN.  Best on RAW templates.

### `warp_correlation(gd_a, gd_b)`

Cross-channel Pearson correlation of two group-delay profiles.  ~1 when the per-channel delay
structure matches (same neuron -- a fixed geometric signature, drift-robust); low/incoherent for
two different co-located cells.  Complements cosine: it catches high-cosine look-alikes (validated
on g5: same-neuron ~0.93, high-cosine-different ~0.67), so a relaxed cosine + warp gate recovers
the last few real merges without the false ones.

### `amp_profile_correlation(ta, tb)`

Cross-channel correlation of two templates' per-channel peak-to-peak AMPLITUDE profiles -- the
magnitude term of the Omlor-Giese anechoic model (eq. 10), the companion to warp_correlation's
group-delay term (eq. 11).  The full same-neuron criterion demands BOTH.  On g5 co-located
look-alikes (cosine>=0.9) this term (AUC ~0.76) separates same/different better than the group-delay
term alone (~0.65).  Time-shift invariant (per-channel p2p), so no mutual_center needed.

### `warp_channel_incongruity(gd_a, gd_b, warp_hi=0.85, min_local=4)`

Worst single-channel group-delay incongruity -- a SUB-GATE statistic for pairs whose
overall warp is ALREADY coherent.  warp_correlation is a cross-channel Pearson, so a couple
of strong channels can hold it high while ONE channel's group delay disagrees: the single-
channel signature of a different co-located source the aggregate masks.  Over the channels
within BOTH clusters' centroid range (finite group delay in both profiles -- i.e. supra-
amp_frac at each centroid), fit the per-channel delay relationship robustly (Theil-Sen, so
the outlier channel does not tilt the line) and return the largest single-channel residual,
in samples.

Returns 0.0 (nothing to veto) when the pair is NOT already warp-coherent (corr < warp_hi) or
has < min_local shared centroid-range channels -- so a caller thresholding the return value
only ever vetoes high-warp, well-populated merges.  Validated on g5: among merge-passing pairs
(class+offset+warp all pass), a 1.0-sample threshold vetoes ~2% at mid-range offset (~0.55) --
look-alikes the offset gate does NOT catch, i.e. an INDEPENDENT precision veto.

## Refractory cross-correlogram gate

### `refractory_gate(t_a, t_b, duration, refrac, thr=0.3, min_exp=5.0, censor=0)`

Power-aware verdict for a proposed merge of two spike trains (see module docstring).

### `overlap_refractory_gate(t_a, t_b, refrac, thr=0.3, min_exp=5.0, censor=0)`

Refractory verdict for a proposed CROSS-CHUNK link, evaluated ONLY on the temporal overlap of
the two fragments' spikes.  Cross-chunk fragments normally occupy disjoint time windows (no power),
but adjacent chunks OVERLAP: in that window the same neuron's spikes are detected in BOTH chunks.
The censor band removes those zero-lag duplicate detections; what remains in (censor, refrac] is
the refractory shoulder -- empty for one neuron (a dip -> 'allow'), at chance for two independent
neurons (no dip -> 'veto').  The overlap window is taken empirically as the intersection of the two
spike-time spans, so it needs no chunk-geometry assumptions; if the fragments do not overlap in time,
or too few coincidences are expected, the test has no power and ABSTAINS (never vetoes).  t_a, t_b,
refrac and censor are all in samples.  Returns the refractory_gate dict plus ov_lo/ov_hi/n_a/n_b.

### `refrac_samples(refrac_ms, sr)`

*(no docstring)*

## Collision-robust envelope (`fiber_envelope`)

The per-unit band a single cell should stay inside, estimated so that near collisions -- a second
spike from another unit inside the window but not coincident with the detected one -- cannot
inflate it.  Stage: [`fiber-envelope`](stages.md).  Model, per unit, channel c and sample t:

    sigma^2(t,c) = var_noise(c) + var_a * T(t,c)^2 + var_b * T'(t,c)^2 + excess(t,c)

* `var_noise` -- from OFF-SPIKE windows of the continuous trace (no detection within the guard),
  never from cluster members.  Undetected sub-threshold spikes stay in those windows: they are
  the background every spike rides on, so they belong to the floor.
* `var_a`, `var_b` -- robust (MAD) spreads of the per-spike least-squares fit
  `x_i ~ a_i T + b_i T'` (amplitude scale and first-order jitter, `b = -a*delta`).
* `excess` -- robust residual variance above the floor on the CLEAN spikes: the unit's own shape
  variability.  `Envelope.excess_ratio()` (per channel, mean over the window) is ~1 for a
  well-modelled unit.

### `fit_envelope(X, core, var_noise=None, noise_windows=None, *, maxlag=4, width=8, q=0.99, iters=3, k_mad=4.0, ext=None)`

Iterates: integer-align each spike to the median (the extraction margin means no wrap-around) ->
fit a, b -> residual -> BUMP score = max over a `width`-sample sliding window of the
channel-summed residual energy normalised by noise + excess -> drop spikes above the threshold
-> re-estimate.  The threshold is the `q` quantile of the same statistic on the off-spike noise
windows (or a median + `k_mad`*MAD rule on the cluster's own scores when no windows are given).
A near collision is time-LOCALISED; a merged second cell's difference is spread over the whole
template, so it is (correctly) NOT flagged as a collision -- see Limits.

### `find_partners(Xext, own, env, others, var_noise, *, core_in_ext, width=8, smax=None, min_shift=2, amp=(0.4, 2.5), gain_thr=0.0)`

Explains a flagged spike's bump by another unit's EXTENDED template at every shift up to `smax`,
zero-padded rather than rolled (np.roll would wrap the partner's tail back into the window).
Accepted when the post-subtraction bump score is below the unit's threshold, the amplitude
coefficient is within `amp`, and the explained energy beats `gain_thr` = `partner_gain_null`,
the q-quantile of the best dictionary fit to pure noise (without it, low-amplitude templates
"explain" any bump).  Explained partners map onto `.eap` membership (`eap_manifest` ->
`fiber_decollide.write_eap_membership`), i.e. Klusters' amber collision ring.

### `separation(ea, eb, maxlag=3, sigma="robust")`

Merge-gate d' between two units: sqrt(sum D^2/s^2 - L*C*(1/na + 1/nb)), D the template difference
at the best lag and s^2 the mean of the two envelopes.  The subtracted term is the expectation for
two estimates of one template, so a same-cell pair sits near 0.  Also returns the worst single
channel's d'.

### `Envelope.channel_ratio(which="kept")` -- the amplitude-scaled variance check

Per channel, observed / predicted variance summed over the window, with the prediction
`amp_model() = noise_c + var_a T^2 + var_b T'^2` using ONE var_a / var_b for every channel.  The
band therefore widens with the channel's amplitude: a near unit is allowed a large spread on its
big channels, a distant one stays noise-dominated, and the check reads ~1 for a single source at
any distance.  A merged second unit shows up as > 1 on the channels where the two differ.
`which`: `kept` (default: all spikes except near collisions a partner explained), `clean` (bump-test
survivors only -- blind to a strongly different minority, which the bump test flags), `all`
(inflated by near collisions).  The noise-relative `excess_ratio()` is kept for comparison.

### Validation

Synthetic (test/test_envelope.py; known templates, amplitude/jitter spreads, a 20-sample detector
dead time and an unsorted background unit):
* plain-std variance is 10-17 % high on average (95th percentile +38-67 %) and the robust envelope
  is within 3-4.5 % of the truth (3 seeds x 3 units);
* 68-73 % of near collisions with a sorted partner are flagged, with no flags on spikes that have
  no second event;
* explained partners carry the right identity and offset (+-1 sample) in 98-100 % of cases,
  including the hidden ones inside the dead time.

Real g6 (sirotaA-jg-000005-20120316, exp_final cells).  Four spaces: the session's own `.fil`
(36-41 min -- the filter `.spk.standard` was cut from) as-is (`standard`) and through the session's
order-5 sdiffPairs pattern (`stderiv_C5`, the fiber feature space; `space_transform` reproduces the
on-disk `.spk.stderiv_C5_D34` exactly for the spikes fiber-realign did not sub-sample-shift); and a
Butterworth 800 Hz trace of 36-47 min, all-pairs stderiv and `stderiv_C5`:

| space | window | plain/robust variance, mean (worst unit) | 95th pct (median unit) | flagged | dead-time hole -> recovered (pair r) |
|---|---|---|---|---|---|
| standard | .fil 36-41 | 1.22 (1.49) | 1.77 | 4.2 % | 133 -> 99 (0.50) |
| stderiv_C5 | .fil 36-41 | 1.05 (1.10) | 1.35 | 4.1 % | 133 -> 118 (0.79) |
| stderiv (all-pairs) | BW 36-47 | 1.10 (1.21) | 1.38 | 4.3 % | 318 -> 289 (0.53) |
| stderiv_C5 | BW 36-47 | 1.08 (1.17) | 1.35 | 4.3 % | 318 -> 294 (0.69) |

* The bias is largest in the untransformed space and roughly a quarter of that in the stderiv
  spaces: the spatial + temporal derivative suppresses a partner's slow rebound tail and shared
  footprint (on the synthetic recording the plain std is not inflated at all in stderiv_C5).
* The recovered HIDDEN partners (|shift| <= 20 samples, no .res row possible) fill the detector's
  dead-time hole: the CCG flank (32-95 samples, per unordered cell pair) predicts the hole, and
  74-92 % of it is recovered.  Attribution per pair is best in stderiv_C5 (r = 0.69-0.79); in the
  untransformed space small partners under large spikes are missed and look-alikes get swapped.
* Partners at |shift| > 21 that should have their own .res row: 19-33 % do, 0 % at the mirrored
  shift (n = 11-30; small).

Amplitude-scaled check on real g6 cells (exp_final, >= 150 spikes; .fil 36-41):

| statistic | Spearman vs unit SNR (C5 / standard) | AUC, 10 % of the most similar cell merged in | 20 % |
|---|---|---|---|
| channel_ratio kept | +0.08 / -0.09 | 0.75 / 0.75 | 0.81 / 0.82 |
| channel_ratio clean | -0.42 / -0.20 | 0.66 / 0.68 | 0.74 / 0.83 |
| channel_ratio all | +0.11 / +0.11 | 0.74 / 0.68 | 0.80 / 0.74 |
| excess_ratio (noise-relative) | +0.48 / +0.50 | 0.55 / 0.53 | 0.59 / 0.63 |

The noise-relative check grows with SNR (near units fail it for being near); the amplitude-scaled
one does not, and it is the better merge detector.  Real single cells still read ~1.3-1.5 (shape
variability the scale + jitter model does not capture), which is what limits a 10 % admixture to
AUC 0.75.  On fiber_session units it flags those < 90 % pure against GT2 at AUC 0.81-0.83, but does
not predict which units the curator split in the t-SNE pass (AUC 0.52-0.53).

### Expected effect on fiber_session output

fiber_session units of the 36-48 min chunk, refined automatically and scored against the curated
stages on the same spikes (ARI / purity / completeness, GM = reference cell > 1):

| | GT2 ARI | GT2 purity | GT2 completeness |
|---|---|---|---|
| fiber_session as is | 0.10-0.12 | 0.84-0.85 | 0.31 |
| near collisions -> noise | -0.005 to -0.007 | +0.006 to +0.010 | -0.008 to -0.011 |
| exp_chunk (the t-SNE pass) | 0.22-0.27 | 0.90 | 0.46-0.47 |
| exp_linked (pipeline final) | 0.61 | 0.86 | 0.68-0.70 |
| average-linkage merge on envelope d' < 7 | 0.61-0.70 | 0.83-0.84 | 0.71-0.77 |

* The robust envelope itself (vs the plain std in the same d') changes the best achievable result
  by <= 0.004 ARI (0.695 vs 0.699, 0.610 vs 0.612, 0.604 vs 0.607) and anything on the 6-8 plateau
  by <= 0.007; off the plateau the two shift the threshold slightly (differences up to +-0.1 in
  either direction).  The merge gain is the d' statistic (envelope-normalised template
  difference), not the collision-robust noise estimate.  d' beats amplitude-preserving cosine
  merging (best 0.39-0.59 ARI, at equal or lower purity).
* The threshold 7 was read off GT2; the plateau (6-8) holds in all three spaces (stderiv_C5 and
  standard on the session .fil 36-41, stderiv_C5 on a Butterworth trace 36-47), but it is one chunk
  of one session, and the merge does not cross chunk boundaries (exp_linked's drift linking does).
* The curator did treat flagged spikes differently (sent to noise 7-9 % vs 3 %; moved 63-70 % vs
  52-54 %), but they are 3.4-4 % of spikes, so automating that is worth ~0.01 purity.

### Limits

* The bias does not change merge decisions on these fragments in any space: same vs different
  fiber_session fragment pairs (GT2-labelled, >= 80 % pure, 5 nearest by cosine) give d' AUC
  0.989-1.000 with the plain std and the same to the third decimal with the envelope; band_overlap
  likewise.  Same-cell d' is ~3-5, different-cell ~19-25.  In the stderiv_C5 space the amplitude-
  preserving cosine is already 0.993-1.000, so d' adds nothing there; it only helps in the
  untransformed Butterworth space (look-alike pairs, cos >= 0.9: 0.994 vs 0.865).
* A 10 % admixture of the most similar cell inside ONE cluster is not detected by either variance
  statistic (the robust excess ignores a minority by construction; the plain std moves < 2 %), and
  the bump test flags only 5-26 % of the foreign spikes (most in stderiv_C5) -- the difference is
  spread over the template, not localised.  Detecting it needs a direction (bimodality along the
  template-difference / residual principal axis), not a variance.

## Cell layer, cell-aware merging and cell type (`fiber_envelope`, 0.34)

### The cell layer: `fit_envelope(..., times=...)`

With spike times the envelope also fits what ONE cell does from spike to spike, on the clean spikes:

    x_i = a_i T + b_i T' + (a_i - <a>) S + (t_i - <t>) D + noise

* `S` (`Envelope.shape_amp`, `shape_gain()` = |S|/|T|) -- shape change per unit of amplitude: a
  burst's later, smaller spikes are also differently shaped (the fiber premise; docs/morphology.md
  puts 77 % of physiological modulation along such a curve).
* `D` (`Envelope.drift`) -- template change per second within the window.
* `adapt` -- the spike-frequency-adaptation law of `fiber_adapt` fitted to the unit's own train:
  a_i ~ b0 + b1 EWMA_tau(prior spikes), best tau by R^2 (`adaptation_law`).

`amp_model()` then becomes noise + the spike-to-spike variance of that whole prediction, so
`channel_ratio()` measures spread beyond what one cell does.  On g6 exp_final cells the single-cell
ratio drops from 1.28 to 1.23 (stderiv_C5, session .fil), 1.27 to 1.20 (stderiv_C5, Butterworth) and
1.48 to 1.33 (standard); most for bursting cells (stderiv_C5, session .fil: cell 28 2.12 -> 1.45,
cell 12 1.46 -> 1.27).
The clearly adapting cells fit tau 0.25-0.47 s (fiber_adapt's prior: 270-620 ms) with R^2
0.24-0.78; the narrow-spiking interneurons fit R^2 ~0.

### `merge_units(...)` / `fiber-envelope --merge THETA`

Average linkage on envelope d' (`separation`), cut at THETA, with **merge inflation** as a
cannot-link: for each unit's 8 nearest neighbours (d' < 15) the union is fitted with the cell layer,
and a pair whose union's amplitude-scaled spread exceeds the larger part's by more than 0.1 can never
end up in one cluster.  Writes `<base>.clu.<method>.<elec>.<stage>.envmerge` and the per-pair evidence
table `<base>.envmerge.<elec>[.<tag>]` (d', inflation, adaptation coupling, shape gain, veto reason).

fiber_session of the 36-48 min chunk, GT2 ARI / purity / completeness:

| | d' only | d' + inflation cannot-link |
|---|---|---|
| stderiv_C5, Butterworth 36-47, theta 7 | 0.699 / 0.836 / 0.769 | 0.699 / 0.836 / 0.770 |
| same, theta 8 | 0.660 / 0.830 / 0.798 | **0.698** / 0.832 / 0.774 |
| same, theta 10 | 0.658 / 0.808 / 0.833 | **0.682** / 0.815 / 0.798 |
| stderiv_C5, session .fil 36-41, theta 8 | 0.572 / 0.822 / 0.750 | **0.633** / 0.828 / 0.749 |
| same, theta 10 | 0.588 / 0.813 / 0.791 | **0.633** / 0.818 / 0.770 |

The cannot-link does not raise the peak; it makes the result insensitive to the threshold (the
plateau extends from 6-7 to 6-10).  It is precise: 31 of the 32 pairs it vetoed (Butterworth run)
were different GT cells.

**Tested and not adopted** (each measured on the same chunk; all would have been plausible):

| evidence | precision as a veto | effect on ARI |
|---|---|---|
| adaptation coupling (one law over the merged train loses > 0.3 R^2) | 81 % | -0.01 |
| point-source position > 3x split-half floor (corrected geometry) | 57 % | -0.02 to -0.34 |
| narrow vs broad spike width (0.35 ms) per fragment | -- | -0.11 |
| split on bimodality of the cell-model residual (11 splits) | -- | -0.017 (purity +0.004) |

The adaptation coupling and type features stay in the outputs as evidence (`.envmerge`, `.env`).

### Cell type: `type_features`, `cell_type_conflict`

Per unit: trough-to-peak width of the UNTRANSFORMED mean waveform, burst index (ISI < 10 ms), rate;
written to `.env`.  On g6, cells 21, 23 and 26 are narrow (0.15-0.25 ms, 21 at 10.9 Hz) and do not
adapt; cells 3, 10, 12, 15, 18, 28, 29 are broad, bursting and adapting.  Type matters for absolute
thresholds -- the cell-model ratio of the narrow cells is 1.33-1.55 against a median 1.18 for the
broad ones, so a split threshold must be per type -- and the merge's remaining errors concentrate in the narrow
and the broad non-bursting cells (22, 23, 25, 30, 33, 19 end up in groups of < 0.35 purity; the
bursting pyramidal cells in groups of 0.90-1.00).  As a cannot-link on fiber_session FRAGMENTS the
width is too noisy (`--type-veto`, off by default).

## Resampled t-SNE consensus and quarantine (`fiber_consensus`, 0.35)

The curator's t-SNE lasso works on one embedding, and one embedding is one random draw.
`fiber-consensus` embeds a neighbourhood of similar clusters R times (default 40; random 70 %
subsets, fresh random initialisation, per-column z-scored `.fet`), and from the runs builds:

* `coneighbour_communities` -- Louvain on P_ij = (runs with j among i's 15 nearest 2-D neighbours) /
  (runs containing both), edges with P >= 0.1;
* cells -- communities joined by `fiber_envelope.merge_units` (d' < 7, inflation cannot-link),
  because t-SNE cuts one cell's noise ball into several stable-looking islands;
* `switch_rate` -- per spike, the fraction of its runs in which the majority of its 2-D neighbours
  belongs to another CELL;
* the quarantine (0.36) -- with the envelope refitted on the cells: the near-collision flag, or the
  PERIMETER rule (`perimeter_rank` >= 0.8, the spike's average 2-D radial rank inside its cell, AND its
  residual energy or bump score >= 2 robust z above the cell's).  Cell switching (rate >= 0.3) is
  opt-in.  Quarantined spikes go to one cluster per neighbourhood; `<base>.qrn.<elec>.<tag>` lists
  each with its cell, switch rate, perimeter rank, residual / bump z, collision flag and the
  explaining partner / shift, for review.

Four g6 neighbourhoods (36-47 min; average linkage on envelope d' < 15 or < 10; GT = exp_final;
"misassigned" = a spike whose GT cell is not its cell's majority, or GT noise):

| neighbourhood | spikes | fiber_session purity / completeness | cells purity / completeness | quarantined | precision (base rate) | purity after |
|---|---|---|---|---|---|---|
| 7 (10 fragments) | 2153 | 0.904 / 0.791 | 0.931 / 0.732 | 8.6 % | 0.34 (0.08) | 0.953 |
| 8 (12 fragments, one cell) | 1458 | 0.906 / 0.513 | 0.910 / 0.979 | 4.7 % | 0.66 (0.11) | 0.932 |
| 18 (11 fragments) | 1953 | 0.768 / 0.355 | 0.807 / 0.901 | 5.9 % | 0.83 (0.42) | 0.832 |
| 9 (8 fragments) | 1520 | 0.870 / 0.719 | 0.900 / 0.593 | 10.2 % | 0.39 (0.11) | 0.929 |

Of the two criteria the collision flag is the more reliable (precision 0.51-0.79 at 3.6-4.9 % of
spikes); cell switching runs 0.20-0.86, best where the neighbourhood holds clearly different cells.
Completeness of the cells beats the fragments where a cell was over-split (8, 18) and loses where
the envelope merge stops short (7, 9) -- the cells are a starting point for the curator, not a
verdict.

**Distance variance is not a suspicion marker.**  The first idea -- spikes whose 2-D distances to
their full-dimensional neighbours vary most across runs -- measures density: inside one cell's
noise ball every neighbour is as good as any other, so core spikes are the unstable ones, and
outliers, collisions included, form small islands that are stable from run to run.  Against
misassigned spikes it ran at AUC 0.25-0.55 (0.39-0.52 even ranked within each cell); against
collision flags 0.35-0.60.

### Where deviant spikes sit in the 2-D map (0.36)

Averaged over 10 runs, a spike's radial rank inside its cell's 2-D cloud (0 centre, 1 rim) follows its
deviation from the cell (8 cells, 312-1562 spikes; Spearman rho): full-dimensional distance from the
cell centre 0.51-0.76, residual energy after a T + b T' 0.04-0.53, bump score -0.07-0.47, amplitude
deviation 0.15-0.31; timing jitter -0.32-0.14 (slightly larger in the core of pure cells -- the D34
lag features absorb a shift).  The outer 20 % against the inner 50 %:

| cell (GT purity) | collision-flagged, outer / inner | misassigned, outer / inner |
|---|---|---|
| 18:2 (0.64) | 24 % / 1.1 % | 78 % / 21 % |
| 7:3 (0.83) | 38 % / 2.0 % | 47 % / 9.6 % |
| 9:3 (0.83) | 34 % / 2.4 % | 39 % / 11 % |
| 8:2 (0.92) | 20 % / 1.2 % | 28 % / 3.0 % |
| 7:2, 9:2, 7:4 (0.97-0.99) | 4.4-6.9 % / 1.8-2.3 % | 2.0-5.6 % / 0.6-2.6 % |

So the rim of a contaminated cell is where its collisions and foreign spikes are; the rim of a pure
cell is mostly its legitimate amplitude extremes -- hence the waveform condition in the rule.

Quarantine rules on the four neighbourhoods (precision against the misassigned base rate):

| rule | quarantined | precision | recall | purity gain |
|---|---|---|---|---|
| collision | 4.6-5.6 % | 0.48-0.84 | 0.11-0.30 | +0.017-0.032 |
| perimeter (0.8, z 2) | 3.9-5.2 % | 0.44-0.90 | 0.11-0.30 | +0.015-0.036 |
| collision OR perimeter (default) | 6.7-7.8 % | 0.40-0.85 | 0.16-0.41 | +0.021-0.045 |
| switch (opt-in) | 0.4-8.6 % | 0.13-0.86 | 0.03-0.14 | +0.003-0.010 |

(base rates 0.077, 0.113, 0.107, 0.418 for neighbourhoods 7, 8, 9, 18).  The perimeter rule adds 24-47
spikes per neighbourhood that the collision flag misses, at 0.23-0.88 precision.

### Categorising the quarantine by t-SNE watershed (0.37)

A quarantined spike is a candidate, not a verdict: a collision group, a cell's own extreme and a
spike that belongs to a neighbour all land there.  `watershed_categories` sorts them.  On each of
`--ws-runs` fresh t-SNE maps of the whole neighbourhood (5 by default), a Gaussian KDE on a 256 x 256
grid (sigma = grid/100) is cut into watershed basins from its density maxima.  A basin's OWNER is the
cell holding most of its kept (non-quarantined) spikes.  Every quarantined spike is

| category | basin | meaning |
|---|---|---|
| island | >= 50 % quarantined (or no kept spikes, or outside every basin) | a collision / noise group of its own |
| own-rim | owned by the spike's own cell | an extreme of that cell |
| other-cell | owned by another cell | a reassignment candidate; the destination is recorded, not applied |

and the category is the vote over the maps.  On the four g6 neighbourhoods (GT = exp_final), share
of the quarantine / fraction whose GT is in fact the spike's own cell:

| neighbourhood | own-rim | island | other-cell | median vote agreement |
|---|---|---|---|---|
| 7 | 78 % / 73 % own | 20 % / 10 % own | 2 % | 1.0 |
| 8 | 61 % / 60 % | 37 % / 34 % | 2 % | 0.8 |
| 9 | 65 % / 52 % | 28 % / 18 % | 8 % (78 % GT = destination) | 0.8 |
| 18 | 60 % / 21 % (41 % GT noise) | 38 % / 2 % (20 % GT noise) | 2 % | 1.0 |

Islands are 66-98 % foreign or noise; own-rim spikes of a clean cell are mostly the cell's own
(neighbourhood 18's base cell is itself 0.64 pure, so its rim is not).  `fiber-consensus` therefore
writes TWO quarantine clusters per neighbourhood -- island, and rim (own-rim + other-cell) -- and the
`.qrn` table carries `category`, `ws_agree` and `ws_dest`.  Review the island cluster as collisions /
noise and the rim cluster as cell extremes; `--ws-runs 0` writes one quarantine cluster as before.
Other-cell is rare (2-8 %) and only neighbourhood 9's are mostly right about the destination, so
they are not reassigned automatically.

A SECOND ROUND -- hold the quarantine out, rebuild the cells, quarantine again -- quarantines 1-2 %
more of the neighbourhood at 0.25-0.65 precision (base 0.06-0.38) and moves kept-spike purity by
-0.002..+0.004 (completeness +0.12..-0.22, mixed).  Diminishing returns; not implemented.

### Running it (0.37)

The stage works chunk by chunk (`--chunk-min`, default fiber-session's `FK_SESSION_CHUNK_MIN` or 12):
neighbourhoods are found and embedded within one chunk, the scale it was measured at (1-3 k spikes per
neighbourhood).  At d' 12 a chunk's confusable fragments form one large group (g6 36-48 min:
65 units, 11.7 k of 19 k spikes, 9 GT cells).  It is kept whole by default (see below);
`--max-spikes N` re-cuts a group over N spikes at theta*0.8, *0.64, ... only to bound run time.  With `--keep-ids 1` (default) each cell, largest first, keeps the id of the input
unit holding most of its kept spikes, unless a larger cell took it; split-off cells get fresh ids.  So
it can run on a chunk-local sort before linking or on a linked sort after it without breaking the
links.  The t-SNE runs and watershed maps (~93 % of the time; ~4 s per run on 1 k points) run in
parallel processes (`--workers`, default every core) with results independent of the worker count.
Knobs are `FK_CONSENSUS_*` (CLI > env > `$FK_CONFIG` > default; section [C] of
fiber-kit-exp.yaml); in a plan it is the stage `fiber-consensus` (`fiber-pipeline <elec> fiber-consensus`
reads `.exp_final` and `<sess>.fil` by default).

### Neighbourhood size, and what did not help (0.37.3)

Fresh fiber-session sort of the 36-47 min chunk (the exp config; 109 units with >= 30 spikes,
19,395 spikes, 14 % misassigned against exp_final), scored against exp_final.

**Keep the big neighbourhood whole.**  A neighbourhood can only fix contamination whose true cell is
inside it.  With the 0.37.1 cap of 4000 spikes that holds for 38 % of misassigned spikes; uncapped,
63 %.  Grouping units by their own GT cell (an oracle) reaches only 32 %: the contamination runs
between different cells whose fragments look alike, which is why d' chains them.  On the 11.7 k-spike
group, scored on spikes neither run quarantined:

| | cells | purity | completeness | ARI | quarantine precision |
|---|---|---|---|---|---|
| fiber-session input | 65 | 0.846 | 0.439 | 0.023 | -- |
| capped at 4000 (7 pieces) | 26 | 0.874 | 0.582 | 0.170 | 0.70 |
| uncapped (one group) | 7 | 0.872 | 0.812 | 0.594 | 0.73 |

Measured and not adopted:

* **Fiber geometry for neighbourhoods.**  fiber-session's direction profile d(r) separates same-cell
  from different-cell fragment pairs at AUC 0.371 (shared energy range) and 0.475 with each profile
  continued by its Jacobian dd/dr; envelope d' does 0.985.  The code reproduces fiber-session's own
  case (two halves of one fragment, AUC 0.979): direction identifies the same fragment, not the same
  cell -- fiber-session split those fragments along direction (same-cell median 0.64, different-cell
  0.44).
* **Contamination orthogonal to the Jacobian.**  A spike's residual to its own fiber with the local
  tangent J(r) removed, projected on the gap toward each of the 10 nearest fibers (t = 0 own fiber,
  1 the other): misassigned spikes sit at median t = 0.22 toward their true cell, 9 % past halfway,
  at every energy; as a flag AUC 0.515 (envelope bump 0.63).  Their true cell is among the 10
  nearest fibers for 48 %.  Pair statistics on the candidate pairs: orthogonal d' 0.75, between-band
  flux 0.81, envelope d' 0.955; neither improves the neighbourhoods.  Likely reason: a fiber is fitted
  through its own contamination, so the foreign spikes are not orthogonal to it.
* **Quarantine held out of a fiber-session rerun.**  fiber-session is deterministic (identical rerun
  ARI 1.0) but its fragment boundaries move with any 5 % of the spikes: holding out the quarantine
  gives ARI 0.30 to the original, holding out 1,080 random spikes (matched per unit) 0.28, and the
  random holdout scores no worse.  Consensus on the rerun: purity 0.870 / completeness 0.571 / ARI
  0.301 against 0.886 / 0.570 / 0.314 for one round, although its new quarantine (538 spikes, none
  from round 1) runs at 0.65 precision.
* **t-SNE stratification of dense cells** (perplexity 15-80 x exaggeration 1-3, scored by seed-to-seed
  ARI x full-dimensional d' between modes).  Cells split stably only at exaggeration 3 (k = 3-8,
  ARI 0.66-1.0).  In pure cells the modes are burst / adaptation states (median preceding ISI 28 ms
  -> 1.5 s across modes with feature norm 0.89 -> 1.17) and drift, 0-6 % foreign; collision-flag
  rates per mode equal the cell's.  A mode's median-waveform residual exceeds the noise null 1.5-3x
  in pure modes too (legitimate shape change), so as a collision test it is not specific.

### Judging without the ground truth (0.37.4)

exp_final was curated with earlier fiber-kit tools, so it carries their merges: agreement with it
cannot tell a cleaner sort from one that repeats its errors.  The evidence below uses spike timing,
which no waveform method shares.

* **The quarantine is real contamination.**  Within the cell it sat in, a round-1 quarantined spike
  is in a refractory-violating pair (another spike 0.6-1.5 ms away) 3.8x as often as random spikes
  of the same cells (p < 0.005); removing the quarantine removes 8 of 25 violating pairs, a matched
  random removal 2.5.  (Round 2's new quarantine shows no excess, 1.07x -- but only 17 pairs were
  left to find.)
* **The envelope merge joins different neurons; a refractory cannot-link stops most of it.**  On the
  11.7 k-spike neighbourhood (merge at d' < 7):

| | cells | violating pairs 0.6-1.0 / 1.0-1.5 ms | Hill contamination | largest cell |
|---|---|---|---|---|
| fiber-session fragments | 53 | 2 / 2 | 0.03 | 543 |
| Louvain communities | 14 | 6 / 9 | 0.25 | 1,852 |
| merge, no veto | 6 | 38 / 69 | 0.43 | 8,482 |
| merge + refractory veto, 1.5 ms (default) | 8 | 13 / 20 | 0.34 | 3,157 |

  `refractory_cannot_link` uses fiber_ccg.refractory_gate: a pair whose trains coincide within
  (0.6, 1.5] ms at chance level is two neurons; it abstains when fewer than 3 coincidences are
  expected (at 1.0 ms nothing had that power).  The exp_final score does not move (0.84 -> 0.83):
  exp_final made the same merges.
* **exp_final cell 21 is multi-unit background.**  7,186 spikes at 10.9 Hz, median p2p 1,880 (all
  spikes 2,560; pyramidal cells 3,800-7,400); its autocorrelogram sits at 1.2-2x chance from 1 to
  10 ms with no refractory gap, where the pyramidal cells (10, 12, 15, 17) are empty at 1-2 ms and
  peak at 30-130x chance at 3-5 ms.  Its narrow average waveform is an average of small spikes, not
  a fast-spiking interneuron.  A region-local oblique library -- degree-2 Legendre fits in 8-sample
  regions (step 4) per channel and across channels, the top 3 principal axes of each, 243 axes --
  finds no axis that splits it into halves with a refractory gap (no bimodality, dip p >= 0.98; best
  excess straddle of the violating pairs p = 0.003 against a Bonferroni 2e-4; each half still
  coincides with itself at 0.6-0.8 of chance).

## Per-spike realignment: parity with Klusters (`fiber_realign`, 0.38.1)

`fiber-realign --align-algo klusters` follows `KlustersDoc::realignSpikes` (neurosuite-3
`klustersdoc_realign.cpp`) around the libklustersshared kernel (`realign_xcorr_omp.cpp`) step for step.
Per unit, up to `--iters` passes:

1. template = mean of the currently aligned spikes (the int64 sum divided by n, truncated toward zero,
   as Klusters' int16 template);
2. the template is rolled **circularly** so its summed-|amplitude| peak lands on the peak sample, which
   is `peakSampleIndex - 1` (Klusters' parameter is 1-based) for every `.spk` variant;
3. each spike is scored by normalised cross-correlation over integer lags in +-`--max-shift`: full
   window, circular lag, constant full-energy denominator, float32, first maximum;
4. a spike whose best score is below `--min-score` is not moved in that pass; the others are rolled and
   the lag added to their cumulative shift (so the total can exceed the search radius); stop when
   nothing moves.

`--recenter-rms 1` is Klusters' realign mode 2: afterwards the unit is shifted as a whole so the
circular centroid of its channel-summed energy lands on the peak sample, unless the energy profile's
mean resultant length is below `--rmin`.

Defaults are Klusters': threshold 0.70, 2 iterations, max shift `max(1, (peakSampleIndex-1) // 2)`, rmin
0.4; knobs `FK_REALIGN_MIN_SCORE`, `FK_REALIGN_ITERS`, `FK_REALIGN_MAX_SHIFT`, `FK_REALIGN_RECENTER_RMS`,
`FK_REALIGN_RMIN` (fiber-kit-exp.yaml sets the RMS mode).

Checked against the compiled Klusters code (origin/main, driven by a verbatim port of the GUI loop) on
the g6 36-47 min fiber-session sort, 115 units / 19,543 spikes: identical shift on every spike, on both
the standard and the stderiv_C5_D34 `.spk`, in both modes.  Before 0.38.1 the pipeline's call agreed on
17.6 % (standard) and 51.6 % (stderiv_C5_D34) of spikes with mode 2: no RMS recenter, threshold 0.0, 4
iterations, max shift 8, an absolute rather than cumulative lag search, a zero-filled template pre-roll,
and on the standard `.spk` the peak one sample late (21 instead of 20).  The old settings gave tighter
units (mean template correlation 0.617 vs 0.592 on stderiv, 0.701 vs 0.652 on standard) because they
also moved the low-correlation spikes Klusters leaves alone; `FK_REALIGN_MIN_SCORE: 0` and
`FK_REALIGN_ITERS: 4` restore that behaviour.  `test/test_realign_parity.py` holds golden shifts from the
compiled code.
