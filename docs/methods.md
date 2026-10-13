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
