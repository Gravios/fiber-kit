# ════════════════════════════════════════════════════════════════════════════
#  fiber_realign.py — per-spike fiber-template offsets and .res time correction.
#
#  Each spike's waveform is aligned to the template of the unit it belongs to by
#  multichannel cross-correlation (integer lag within +-max_shift), refined to
#  sub-sample resolution by a parabola through the correlation peak.  The per-spike
#  offset (in samples; the "fiber template offset") is saved, and the .res spike
#  times are corrected by the rounded integer part:  res_corrected = res + round(off).
#
#  The template is recomputed from the aligned spikes and the alignment repeated a
#  couple of rounds so it converges; units with too few spikes are left untouched
#  (their template is too noisy to anchor an alignment).
#
#  This both removes residual detection jitter and — importantly after re-linking —
#  forces every spike of a merged unit onto ONE canonical template, so a unit built
#  from fibers detected against different per-chunk references gets a single
#  consistent spike-time convention.  Where the extractor already peak-aligned the
#  spikes, the offsets are correctly small (mostly 0, +-1); the tool still writes
#  them and the corrected .res so the convention is explicit and reproducible.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import numpy as np

_LP = "\u25b8 fiber-realign"
def _log(m=""): print(f"{_LP} \u00b7 {m}" if m else _LP)
def _det(k, v, w=8): print(f"{' ' * (len(_LP) + 3)}{k:<{w}} {v}")

try:
    from . import fiber_session as fs
except ImportError:
    import fiber_session as fs
try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio
try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl
try:
    from . import session_yaml as sy
except ImportError:
    import session_yaml as sy


def _read_clu(path):
    nclu, ids = nio.read_clu_file(path)
    return nclu, ids                                    # nClusters, per-spike ids


def _parse_clu_variant_tag(clu_path, base, group):
    """Infer (variant, tag) from a clu path <base>.clu[.<variant>].<group>[.<tag>] so the realign
    outputs ADHERE to the sort that was passed in (e.g. stderiv / refine) instead of defaulting to
    standard.  Returns ('', '') if the name doesn't parse."""
    import os
    name = os.path.basename(str(clu_path))
    pre = f"{os.path.basename(str(base))}.clu."
    if not name.startswith(pre):
        return "", ""
    toks = name[len(pre):].split(".")
    g = str(group)
    if g not in toks:
        return "", ""
    gi = toks.index(g)
    return ".".join(toks[:gi]), ".".join(toks[gi + 1:])


def template_offsets(spk, labels, max_shift=5, iters=2, min_n=20,
                     subsample=True, noise_label=0):
    """Compute per-spike sub-sample offsets aligning each spike to its unit's
    multichannel template.

    spk     : (N, T, C) float/int waveforms (sample-major, peak near T//2)
    labels  : (N,) per-spike unit id (noise_label is left at offset 0)
    returns : off (N,) float32 sub-sample offset, ioff (N,) int32 rounded offset.
    res_corrected = res + ioff.
    """
    spk = np.asarray(spk, np.float32)
    N, T, C = spk.shape
    ms = int(max_shift)
    off = np.zeros(N, np.float32)
    Tcore = T - 2 * ms
    lags = np.arange(-ms, ms + 1)

    by = {}
    for i, l in enumerate(labels):
        by.setdefault(int(l), []).append(i)

    for u, rows in by.items():
        if u == noise_label or len(rows) < min_n:
            continue
        idx = np.asarray(rows)
        W = spk[idx]                                     # (n,T,C)
        cur = np.zeros(len(idx), int)                    # current integer lag
        corr = None
        for _ in range(max(1, iters)):
            # template from currently-aligned spikes (robust median), core region.
            # Vectorized gather of each spike's lag-shifted core (was a per-spike
            # Python loop); identical result, O(n) interpreter calls removed.
            gidx = np.arange(Tcore)[None, :] + (ms + cur)[:, None]   # (n, Tcore)
            al = np.take_along_axis(W, gidx[:, :, None], axis=1).astype(np.float32)
            templ = np.median(al, axis=0)                # (Tcore,C)
            tc = templ - templ.mean(axis=0, keepdims=True)
            # correlation at every lag (n, nLags)
            corr = np.empty((len(idx), len(lags)), np.float32)
            for k, L in enumerate(lags):
                seg = W[:, ms + L:T - ms + L, :]
                seg = seg - seg.mean(axis=1, keepdims=True)
                corr[:, k] = np.einsum('ntc,tc->n', seg, tc)
            cur = lags[np.argmax(corr, axis=1)]
        # sub-sample refinement: parabola through (k-1,k,k+1) of the correlation
        kbest = np.argmax(corr, axis=1)
        frac = np.zeros(len(idx), np.float32)
        if subsample:
            ok = (kbest > 0) & (kbest < len(lags) - 1)
            a = corr[np.arange(len(idx)), np.clip(kbest - 1, 0, len(lags) - 1)]
            b = corr[np.arange(len(idx)), kbest]
            c = corr[np.arange(len(idx)), np.clip(kbest + 1, 0, len(lags) - 1)]
            den = (a - 2 * b + c)
            good = ok & (np.abs(den) > 1e-6)
            frac[good] = 0.5 * (a[good] - c[good]) / den[good]
            frac = np.clip(frac, -0.5, 0.5)
        off[idx] = lags[kbest] + frac
    ioff = np.rint(off).astype(np.int32)
    return off, ioff


def _upsample_spline(W, factor):
    """Cubic-spline upsample waveforms (n, T, C) along the sample axis by `factor`,
    returning (n, factor*(T-1)+1, C).  Used to align on a finer grid (factor=2 -> half-
    sample lags) for better template matching; the committed .res is rounded back to
    whole original samples afterwards."""
    from scipy.interpolate import CubicSpline
    n, T, C = W.shape
    t = np.arange(T)
    tu = np.linspace(0.0, T - 1, factor * (T - 1) + 1)
    return CubicSpline(t, np.asarray(W, np.float64), axis=1)(tu).astype(np.float32)


def klusters_offsets(spk, labels, peak, max_shift=8, iters=4, min_n=20,
                     min_score=0.0, upsample=1, noise_label=0):
    """Klusters-faithful per-spike realignment -- a verbatim match to libklustersshared
    realign_xcorr_omp.cpp (the kernel Klusters' Shift+P runs) plus spikerealign.cpp's template
    build.  Per unit:

      * build the cluster MEAN template, then PRE-ALIGN it so its summed-|amplitude| peak sample
        (argmax_s Sigma_ch |tmpl[s,ch]|) lands on `peak` (non-circular zero-fill shift, exactly as
        spikerealign.cpp does -- without this the xcorr pulls spikes off the true peak);
      * score each spike by NORMALISED cross-correlation over INTEGER lags in [-max_shift,max_shift]
        using a FULL-window CIRCULAR shift and a CONSTANT full-spike-energy denominator:

            num(L)   = Sigma_ch Sigma_s  tmpl[s,ch] * spike[(s+L) % N, ch]
            score(L) = num(L) / sqrt(tmplEnergy * spikeEnergy)      (spikeEnergy over ALL N samples)

        bestLag = argmax score; spikes scoring < min_score keep lag 0.

    The full circular window with a constant (not per-lag) denominator is the crucial detail: a
    trimmed core window with a per-lag segment norm -- which earlier versions of this function used --
    inflates the score at large lags and leaves spikes essentially un-tightened (the realign_xcorr_omp
    header calls this out explicitly).  That bug is why committed re-extractions stayed jittered while
    Klusters' own Shift+P tightened the same data.

    The pass is repeated up to `iters` times, re-estimating the mean from the freshly-aligned spikes
    (iters=1 reproduces a single Shift+P press; more iterations sharpen the template and converge).

    With upsample>1 the waveforms are cubic-spline interpolated to that many sub-samples first so the
    integer-lag search runs on a finer grid; the returned offset is in ORIGINAL samples (bestLag/f)
    and ioff = round(offset) for the whole-sample .res commit.

    Returns (off, ioff): off (N,) float32 sub-sample offset in original samples, ioff (N,) int32
    rounded; res_corrected = res + ioff (Klusters' convention: newTs = ts + shift)."""
    spk = np.asarray(spk, np.float32)
    f = max(1, int(upsample))
    if f > 1:
        spk = _upsample_spline(spk, f)
        peak = int(peak) * f
        max_shift = int(max_shift) * f
    N, T, C = spk.shape
    ms = int(max_shift)
    pk = int(peak)
    lags = np.arange(-ms, ms + 1)
    arT = np.arange(T)
    spkE = (spk.astype(np.float64) ** 2).sum((1, 2))          # full per-spike energy (all samples)
    sqrtSpkE = np.sqrt(spkE) + 1e-12                          # constant across lags (no large-lag bias)
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(int(l), []).append(i)
    ioff_up = np.zeros(N, np.int32)
    for u, rows in by.items():
        if u == noise_label or len(rows) < min_n:
            continue
        idx = np.asarray(rows)
        W = spk[idx]                                          # (n,T,C)
        sqE = sqrtSpkE[idx]
        cur = np.zeros(len(idx), int)
        for _ in range(max(1, iters)):
            # MEAN template from spikes circularly rolled by their current shift: aligned[s]=W[(s+cur)%T]
            sidx = (arT[None, :] + cur[:, None]) % T
            al = np.take_along_axis(W, sidx[:, :, None], axis=1).astype(np.float64)
            templ = al.mean(0)                                # (T,C) MEAN (Klusters), not median
            # pre-align: shift template so its summed-|amp| peak lands at `pk` (non-circular zero-fill)
            tpk = int(np.argmax(np.abs(templ).sum(1)))
            tshift = pk - tpk
            if tshift:
                z = np.zeros_like(templ)
                if tshift > 0:
                    z[tshift:] = templ[:T - tshift]
                else:
                    z[:T + tshift] = templ[-tshift:]
                templ = z
            tnorm = float(np.sqrt((templ * templ).sum())) + 1e-12
            denom = tnorm * sqE                               # (n,) constant per spike across lags
            best = np.full(len(idx), -np.inf)
            blag = np.zeros(len(idx), int)
            for L in lags:
                seg = W[:, (arT + L) % T, :]                  # FULL-window CIRCULAR shift: spike[(s+L)%N]
                num = np.einsum('ntc,tc->n', seg.astype(np.float64), templ)
                score = num / denom
                upd = score > best
                best[upd] = score[upd]; blag[upd] = L
            new = np.where(best >= min_score, blag, cur)
            if np.array_equal(new, cur):
                break
            cur = new
        ioff_up[idx] = cur
    off = ioff_up.astype(np.float32) / f                      # back to ORIGINAL samples (sub-sample if f>1)
    ioff = np.rint(off).astype(np.int32)                      # rounded for the whole-sample .res commit
    return off, ioff


def roll_spikes(spk, ioff):
    """Circularly shift each (T,C) spike window by its integer offset -- the no-.fil commit.

    The realign convention is res_corrected = res + ioff and a window that starts at res-peak, so the
    re-extracted-at-res_corr window equals new[t] = old[(t+ioff) % T]; this reproduces that by a per-spike
    circular roll instead of reading the .fil.  Exact for high-pass-filtered .spk windows (the edges are
    ~zero, so the samples rolled in from the far end carry no signal).  Because the stderiv transform
    (SDIFF_ALLPAIRS + temporal first-difference) commutes with a circular time shift, rolling the existing
    stderiv .spk is equivalent (up to the negligible filtered seam) to re-deriving it from a rolled raw
    window -- so both variants can be committed from the uploaded .spk alone.  `spk` (N,T,C); `ioff` (N,)
    int.  Returns the rolled array in the input dtype (int16)."""
    spk = np.asarray(spk)
    N, T, C = spk.shape
    sidx = (np.arange(T)[None, :] + np.asarray(ioff, int)[:, None]) % T      # new[t] = old[(t+ioff)%T]
    return np.take_along_axis(spk, sidx[:, :, None], axis=1)


def reextract_from_fil(filmm, gch, res_corr, nsamp, peak, batch=50000):
    """Re-extract each spike's waveform window from the FILTERED signal at its CORRECTED
    timestamp — the real thing, not a circular roll of the old window.  filmm is the
    (nSamples, ntotal) int16 memmap from neuro_io.open_signal; gch the group's channel
    columns; the window is [ts - peak, ts - peak + nsamp).  Edge spikes are clamped.
    Returns (n, nsamp, len(gch)) int16, sample-major (the .spk layout)."""
    gch = np.asarray(gch, int)
    n = len(res_corr); Tlen = filmm.shape[0]
    out = np.empty((n, nsamp, len(gch)), np.int16)
    win = np.arange(nsamp)
    for b0 in range(0, n, batch):
        b1 = min(b0 + batch, n)
        idx = (np.asarray(res_corr[b0:b1])[:, None] - int(peak)) + win[None, :]   # (m, nsamp)
        np.clip(idx, 0, Tlen - 1, out=idx)
        block = filmm[idx]                                 # (m, nsamp, ntotal)
        out[b0:b1] = block[:, :, gch]
    return out


def refeaturize(spk_new, res_corr, basis):
    """Reproject re-extracted windows onto the PCA `basis` (the on-disk .pca.standard the
    sort used, via fiber_pca.read_pca) to refresh the .fet, and append the corrected
    timestamp as the final feature column (the neurosuite .fet time convention).  Returns
    an int64 (n, nCh*nComp + 1) array ready for neuro_io.write_fet."""
    try:
        from . import fiber_pca as fpca
    except ImportError:
        import fiber_pca as fpca
    win = fpca.extract_windows(np.asarray(spk_new, np.float64), basis["recShift"], basis["data2use"])
    fet = fpca.project(win, basis)                         # (n, nCh*nComp), channel-major
    full = np.empty((len(fet), fet.shape[1] + 1), np.int64)
    full[:, :-1] = np.rint(fet).astype(np.int64)
    full[:, -1] = np.asarray(res_corr, np.int64)           # time feature (last column)
    return full


def _stderiv_transform(raw_ext):
    """ndmanager stderiv transform, ported verbatim from process_extractspikes_stderiv /
    process_alignspikes:sdiff_allpairs.  Two steps per the C++ kernel:
        sdiff[t,c]   = nChanGrp * raw[t,c] - Σ_j raw[t,j]      (SDIFF_ALLPAIRS)
        stderiv[t,c] = sdiff[t,c] - sdiff[t-1,c]               (temporal first-difference)
    clamped to int16.  `raw_ext` is (N, nsamp+1, C): the window PLUS one preceding .fil sample, so
    the t=0 temporal diff uses the TRUE previous sample (matching the continuous g_prev_sdiff of the
    original extraction rather than the zero-baseline of process_alignspikes).  Returns (N, nsamp, C)
    int16, aligned 1:1 with the standard window (raw_ext[:, 1:, :])."""
    r = np.asarray(raw_ext, np.float64)
    C = r.shape[2]
    sd = C * r - r.sum(2, keepdims=True)
    st = sd[:, 1:, :] - sd[:, :-1, :]
    return np.clip(st, -32768.0, 32767.0).astype(np.int16)


def _resolve_variant_token(base, group, variant):
    """The method token actually on disk for a requested one, or the request itself.

    resolve_input family-matches a bare token (asking for 'stderiv' finds
    .spk.stderiv_C5.N), which is what lets the default --variants work on a
    custom-pattern session.  But the caller then needs to know WHICH token it got,
    because the transform to apply and the token to write both depend on it.
    """
    try:
        r = nio.resolve_input(base, "spk", group, [variant])
        if r.found and r.variant:
            return r.variant
    except Exception:
        pass
    return variant


def _variant_present(base, group, variant):
    """True if this session has a <variant> spk or pca on disk (so realign should refresh it)."""
    try:
        if nio.resolve_input(base, "spk", group, [variant]).found:
            return True
    except Exception:
        pass
    try:
        from . import fiber_pca as fpca
    except ImportError:
        import fiber_pca as fpca
    try:
        fpca.read_pca(base, group, prefer=[variant])
        return True
    except Exception:
        return False


def realign(base, elec, nsamp, nch, clu_path=None, max_shift=5, iters=2,
            min_n=20, method="klusters", peak=None, min_score=0.0, upsample=1,
            variant="standard", verbose=True):
    """Read the .spk of the VARIANT the clu/res points to, .res, .clu; compute per-spike
    offsets and corrected .res.

    The alignment waveform follows `variant` (resolved by nio.open_spk(prefer=variant)), NOT the
    method -- so a stderiv clu aligns the stderiv waveforms the curator actually views, instead of
    deriving the shift from the standard .spk and leaving the stderiv jittered.

    method='klusters'  iterative normalised-xcorr to the cluster template (needs a peak: cfg['peak']
                       for the standard variant, else the variant's own pooled energy-peak);
    method='template'  legacy median/un-normalised aligner;
    method='centroid'  reference-free per-spike energy-centroid alignment (fiber_lib.centroid_shift) to
                       the population's own circular-mean centroid -- needs NO peak/template/labels.
    Returns (res, off, ioff, res_corrected, spk, spk_path, labels)."""
    # the clu names the variant exactly (.clu.stderiv.N -> .spk.stderiv.N); load THAT file, not a
    # preference search.  prefer must be a list of variant tokens (a bare string would be walked
    # character-by-character).  There is no canonical .spk -- only .spk.<variant>.N.
    spk, r = nio.open_spk(base, elec, nsamp, nch, prefer=[variant or "standard"])
    spk_path = r.path
    res = fs.read_res(base, elec)
    nclu, labels = _read_clu(clu_path or f"{base}.clu.{elec}")
    n = min(len(res), len(labels), spk.shape[0])
    if not (len(res) == len(labels) == spk.shape[0]):
        if verbose:
            _log(f"WARNING length mismatch: res={len(res):,} clu={len(labels):,} "
                 f"spk={spk.shape[0]:,} → using first {n:,}")
    res, labels, spk = res[:n], labels[:n], np.asarray(spk[:n])
    is_std = variant in ("standard", "", None)
    if method == "centroid":
        T = spk.shape[1]
        pos = np.asarray(fl._centroid_pos(spk.astype(float)))           # per-spike circular centroid
        ang = np.angle(np.exp(2j * np.pi * pos / T).mean())            # population circular-mean centroid
        target = (ang % (2.0 * np.pi)) * T / (2.0 * np.pi)            # reference-free target (no peak arg)
        # centroid_shift returns sh = signed(target - pos); re-extraction applies roll(-ioff), so the
        # committing offset that lands each centroid on target is ioff = pos - target = -sh
        off = -np.asarray(fl.centroid_shift(spk, target), float)
        ioff = np.rint(off).astype(np.int32)
        align_peak = target
    elif method == "klusters":
        align_peak = peak if is_std else int(np.abs(spk).sum(2).mean(0).argmax())
        if align_peak is None:
            raise ValueError("klusters method needs the canonical peak sample (cfg['peak'])")
        off, ioff = klusters_offsets(spk, labels, align_peak, max_shift, iters, min_n, min_score, upsample)
    else:
        off, ioff = template_offsets(spk, labels, max_shift, iters, min_n)
        align_peak = None
    res_corr = res + ioff.astype(np.int64)
    if verbose:
        nz = np.count_nonzero(ioff)
        up = f" upsample={upsample}x" if (method == "klusters" and upsample > 1) else ""
        pk = f" peak={align_peak:.2f}" if align_peak is not None else ""
        _log(f"realign {method}{up}  variant={variant or 'canonical'}{pk}")
        _det("input", f"{n:,} spikes · {nclu - 1:,} units · {spk_path}")
        _det("offsets", f"{nz:,} nonzero ({100 * nz / n:.1f}%) · mean {off.mean():+.3f} · "
                        f"std {off.std():.3f} · max {int(np.abs(ioff).max())} samp")
    return res, off, ioff, res_corr, spk, spk_path, labels


def write_outputs(base, elec, off, res_corr, out_res=None, out_off=None):
    out_res = out_res or f"{base}.res.{elec}.realigned"
    out_off = out_off or f"{base}.offsets.{elec}.npy"
    nio.write_res_file(out_res, res_corr)               # binary LE int64, same as .res
    np.save(out_off, off.astype(np.float32))            # sub-sample per-spike offsets
    return out_res, out_off


def main():
    ap = argparse.ArgumentParser(
        description="Per-spike Klusters-style realignment + corrected .res spike times, with "
                    "optional re-extraction from .fil and re-featurisation (commit-and-reextract "
                    "finalize).  Probe geometry is read from <session>.yaml; flags override.")
    sy.add_session_args(ap)
    ap.add_argument("--clu", default=None,
                    help="cluster file (default <base>.clu.<group>; pass the refined/relinked one, "
                         "e.g. <base>.clu.stderiv.<group>.refine)")
    ap.add_argument("--align-method", dest="align_method", choices=["klusters", "template", "centroid"], default="klusters",
                    help="alignment algorithm (runs on the clu's variant waveform): klusters = iterative "
                         "normalised-xcorr vs pre-aligned mean; template = legacy median/un-normalised; "
                         "centroid = reference-free per-spike energy-centroid (fiber_lib.centroid_shift), "
                         "no peak/template/labels needed.  (Named --align-method to avoid colliding with "
                         "the --method extraction-variant flag used by other tools.)")
    ap.add_argument("--max-shift", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="klusters: leave a spike unshifted if its best cosine score is below this")
    ap.add_argument("--upsample", type=int, default=1,
                    help="klusters: cubic-spline upsample factor for sub-sample matching "
                         "(2 = half-sample lags); .res is rounded back to whole samples at save")
    ap.add_argument("--reextract", action="store_true",
                    help="re-extract each spike's window from .fil at the corrected timestamp -> new .spk")
    ap.add_argument("--shift-spk", dest="shift_spk", action="store_true",
                    help="commit WITHOUT a .fil: circularly roll the existing .spk of each variant by the "
                         "per-spike integer offset (valid for high-pass .spk; the stderiv transform commutes "
                         "with a time shift).  The no-.fil equivalent of --reextract; pair with --refeaturize "
                         "to reproject the rolled windows onto .pca.  Mutually exclusive with --reextract.")
    ap.add_argument("--refeaturize", action="store_true",
                    help="reproject the re-extracted/rolled windows onto .pca.<variant> -> new .fet (implies "
                         "--reextract, or --shift-spk when that is set)")
    ap.add_argument("--fil", default=None, help="filtered signal path (default <base>.fil)")
    ap.add_argument("--variants", default=None,
                    help="comma list of feature spaces to refresh from .fil (default: standard + "
                         "stderiv if present).  Each is re-derived from the re-extracted raw window: "
                         "standard=raw, stderiv=SDIFF_ALLPAIRS+temporal-diff, then projected onto its .pca")
    ap.add_argument("--out-tag", "--out-stage", default="",
                    help="stage tag for committed outputs (default: empty -> overwrite the canonical "
                         ".res/.clu/.spk/.fet[.<variant>].<group> in place; the realign IS the commit). "
                         "Pass a tag only if you want a side-by-side copy, e.g. --out-tag realigned")
    ap.add_argument("--out-variant", default=None,
                    help="variant the .res/.clu adhere to (default: inferred from --clu, e.g. stderiv; "
                         "falls back to standard).  There is one .res/.clu under this variant; .spk/.fet "
                         "are written per feature space in --variants (standard raw, stderiv transform)")
    ap.add_argument("--emit-clu", dest="emit_clu", action="store_true", default=True,
                    help="re-emit the (label-unchanged) clu next to the committed .res/.spk/.fet so the "
                         "set opens in Klusters as a unit (default on)")
    ap.add_argument("--no-emit-clu", dest="emit_clu", action="store_false",
                    help="do NOT write the clu.  Use when the input --clu is a stage-tagged clu but the "
                         "outputs commit canonically (--out-tag ''): re-emitting would overwrite the BASE "
                         ".clu.<variant>.<group> with this stage's labels.  The stage clu already exists and "
                         "its labels are unchanged, so skipping the write keeps the base over-cluster intact.")
    ap.add_argument("--out-res", default=None)
    ap.add_argument("--out-off", default=None)
    a = ap.parse_args()
    # peak is needed for the klusters align and any commit; ntotal (the .fil width) is needed ONLY for
    # the .fil re-extraction path -- --shift-spk rolls the existing .spk, so it never reads the .fil.
    fil_path = a.reextract or (a.refeaturize and not a.shift_spk)
    need = ("nchan", "nsamp") + (("ntotal",) if fil_path else ()) \
        + (("peak",) if (fil_path or a.shift_spk or a.align_method == "klusters") else ())
    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr, require=need)
    base, group, nsamp, nch = cfg["base"], cfg["group"], cfg["nsamp"], cfg["nchan"]
    # the clu's variant (e.g. stderiv) drives BOTH which .spk is aligned and which variant is committed
    cv, _ctag = _parse_clu_variant_tag(a.clu, base, group) if a.clu else ("", "")
    out_variant = a.out_variant if a.out_variant is not None else (cv or "standard")
    res, off, ioff, res_corr, spk, spk_path, labels = realign(
        base, group, nsamp, nch, a.clu, a.max_shift, a.iters, a.min_n,
        method=a.align_method, peak=cfg.get("peak"), min_score=a.min_score, upsample=a.upsample,
        variant=out_variant)

    res_out = nio.write_res(base, group, res_corr, variant=out_variant, tag=a.out_tag)
    np.save(a.out_off or f"{base}.offsets.{group}.npy", off.astype(np.float32))
    if a.out_res:                                          # optional extra explicit copy
        nio.write_res_file(a.out_res, res_corr)
    _log("committed timestamps")
    _det("res", f"{res_out}   (variant {out_variant or 'canonical'}, "
                f"{'overwrote canonical' if not a.out_tag else f'tag={a.out_tag}'})")

    # realignment shifts timestamps/waveforms but NOT cluster assignments, so emit the clu unchanged
    # under the same variant+tag -- this completes the .clu/.res/.spk/.fet set Klusters loads together
    # (without it the realigned .res has no matching .clu and the set can't be opened as a unit).
    # Skipped under --no-emit-clu: when --clu is stage-tagged but outputs commit canonically, this write
    # would overwrite the BASE .clu.<variant>.<group> with the stage's labels (the stage clu already exists).
    if a.emit_clu:
        clu_out = nio.write_clu(base, group, np.asarray(labels, np.int64), variant=out_variant, tag=a.out_tag)
        _det("clu", f"{clu_out}   (ids unchanged)")
    else:
        _det("clu", "re-emit skipped (--no-emit-clu); base over-cluster intact")

    if a.shift_spk:
        try:
            from . import fiber_pca as fpca
        except ImportError:
            import fiber_pca as fpca
        if a.reextract:
            raise SystemExit("[realign] --shift-spk and --reextract are mutually exclusive (one reads the "
                             ".fil, the other rolls the existing .spk)")
        if a.variants:
            want = [v.strip() for v in a.variants.split(",") if v.strip()]
        else:
            want = ["standard"] + [v for v in ("stderiv",) if _variant_present(base, group, v)]
        for v in want:
            try:
                spk_v, _r = nio.open_spk(base, group, nsamp, nch, prefer=[v])
            except Exception:
                _log(f"variant '{v}': no .spk, skipping"); continue
            # Write under the token that was actually RESOLVED, not the one asked
            # for.  resolve_input family-matches a bare request, so asking for
            # 'stderiv' on a custom-pattern session hands back .spk.stderiv_C5.N --
            # and writing that rolled waveform back as .spk.stderiv.N would label a
            # C5-derived file as plain allpairs stderiv.  The roll itself is a
            # circular shift and is transform-agnostic, so any token is fine to
            # roll; only the provenance label has to stay honest.
            vout = _r.variant or v
            wav = roll_spikes(np.asarray(spk_v[:len(res_corr)]), ioff)        # circular per-spike roll
            spk_out = nio.write_spk(base, group, wav, variant=vout, tag=a.out_tag)
            _log(f"rolled {len(wav):,} {vout} spikes by offset (no .fil) → {spk_out}")
            if a.refeaturize:
                try:
                    basis = fpca.read_pca(base, group, prefer=[vout, "standard", ""]
                                          if nio.variant_family(vout) == "standard" else [vout, "D"])
                except FileNotFoundError:
                    _det("fet", f"no .pca basis for {vout}; .fet not written"); continue
                fet = refeaturize(wav, res_corr, basis)
                fet_out = nio.write_fet(base, group, fet, variant=vout, tag=a.out_tag)
                _det("fet", f"{fet_out}   ({fet.shape[1]} features incl. time)")
    elif a.reextract or a.refeaturize:
        try:
            from . import fiber_pca as fpca
        except ImportError:
            import fiber_pca as fpca
        fil = a.fil or f"{base}.fil"
        filmm = nio.open_signal(fil, cfg["ntotal"])
        # read the window PLUS one preceding sample (peak+1, nsamp+1) so the stderiv temporal diff
        # at t=0 uses the true previous .fil sample; standard = raw_ext[:, 1:, :]
        raw_ext = reextract_from_fil(filmm, cfg["channels"], res_corr, nsamp + 1, cfg["peak"] + 1)
        # which feature spaces to refresh: standard always; stderiv (+any listed) when present
        if a.variants:
            want = [v.strip() for v in a.variants.split(",") if v.strip()]
        else:
            want = ["standard"] + [v for v in ("stderiv",) if _variant_present(base, group, v)]
        # Resolve each requested token to the one actually on disk BEFORE choosing a
        # transform.  A bare 'stderiv' family-matches .spk.stderiv_C5.N, and deciding
        # on the requested name would apply the allpairs transform to a session whose
        # waveforms were built from a custom sdiffPairs pattern -- then write the
        # result under a token claiming otherwise.  Deciding on the resolved token is
        # what makes the refusal below reachable at all.
        want = [_resolve_variant_token(base, group, v) for v in want]
        for v in want:
            spec = nio.parse_variant_token(v)
            if spec.family == "standard":
                wav = raw_ext[:, 1:, :]
            elif spec.family in ("stderiv", "D") and spec.kind is None:
                wav = _stderiv_transform(raw_ext)         # SDIFF_ALLPAIRS + temporal diff (verbatim)
            elif spec.family == "stderiv" and spec.kind == "C":
                # _C<order> means the session's own sdiffPairs pattern -- a partner map
                # (order 4) or reference sets (order 5) stored with the session, not
                # anything derivable from the waveform window.  fiber-kit does not
                # hold that pattern and must not invent one: applying allpairs here
                # would silently produce a DIFFERENT feature space under the same
                # token.  Re-extract it with the tool that owns the pattern.
                _log(f"variant '{v}': custom sdiffPairs pattern (order {spec.order}) cannot be "
                     f"re-derived here -- fiber-kit never applies the stderiv transform for a "
                     f"_C token. Re-extract with  ndm_extractspikes -P <pattern>  (or "
                     f"ndm_alignspikes -P) after this realign, so it picks up the corrected .res. "
                     f"NOTE its .spk is now STALE against the realigned timestamps.")
                continue
            elif spec.family == "stderiv" and spec.kind == "S":
                # Orders 1-3 are plain spatial derivatives, but both neurosuite-3
                # aligners refuse _S tokens outright -- they are not implemented there
                # either, so producing one here would be the only writer of that space
                # and nothing downstream could reproduce it.
                _log(f"variant '{v}': _S{spec.order} spatial-derivative order is not implemented "
                     f"(the neurosuite-3 aligners refuse it too); skipping")
                continue
            else:
                _log(f"variant '{v}': no known waveform transform, skipping"); continue
            spk_out = nio.write_spk(base, group, wav, variant=v, tag=a.out_tag)
            _log(f"re-extracted {len(wav):,} {v} spikes from {fil} → {spk_out}")
            if a.refeaturize:
                try:
                    basis = fpca.read_pca(base, group, prefer=[v, "standard", ""] if v == "standard"
                                          else [v, "D"])
                except FileNotFoundError:
                    _det("fet", f"no .pca basis for {v}; .fet not written"); continue
                fet = refeaturize(wav, res_corr, basis)
                fet_out = nio.write_fet(base, group, fet, variant=v, tag=a.out_tag)
                _det("fet", f"{fet_out}   ({fet.shape[1]} features incl. time)")


if __name__ == "__main__":
    main()
