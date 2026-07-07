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

