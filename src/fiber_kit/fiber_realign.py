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
import glob
import os
import numpy as np

_LP = "\u25b8 fiber-realign"
def _log(m=""): print(f"{_LP} \u00b7 {m}" if m else _LP)
def _det(k, v, w=8): print(f"{' ' * (len(_LP) + 3)}{k:<{w}} {v}")

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
try:
    from . import fiber_stderiv as fst
except ImportError:
    import fiber_stderiv as fst
try:
    from . import config as cfgmod
except ImportError:
    import config as cfgmod
try:
    from . import fiber_pca as fpca
except ImportError:
    import fiber_pca as fpca


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


def _lround(x):
    """C lround: round half away from zero (Python's round() is banker's)."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def rms_recenter_shift(energy, peak, rmin=0.4):
    """Klusters' --recenter-rms group shift (libklustersshared realign_center::circularRecenterShift):
    circular weighted mean of the per-sample energy profile; when its mean resultant length R >= rmin
    the minimal signed circular shift (roll convention new[t] = old[(t+shift)%T]) that puts the centroid
    on `peak`, else 0.  Returns (shift, centroid, R, applied)."""
    e = np.asarray(energy, np.float64); T = e.size
    th = 2.0 * np.pi * np.arange(T) / T
    W = e.sum(); C = (e * np.cos(th)).sum(); S = (e * np.sin(th)).sum()
    rlen = float(np.hypot(C, S) / W) if W > 0 else 0.0
    if rlen < rmin:
        return 0, 0.0, rlen, False
    c = np.arctan2(S, C)
    if c < 0:
        c += 2.0 * np.pi
    c *= T / (2.0 * np.pi)
    d = np.fmod(c - peak, T)
    if d > T / 2.0:
        d -= T
    if d < -T / 2.0:
        d += T
    return _lround(d), float(c), rlen, True


def klusters_offsets(spk, labels, peak, max_shift=None, iters=2, min_n=20,
                     min_score=0.70, upsample=1, noise_label=0, recenter=False, rmin=0.4):
    """Klusters-faithful per-spike realignment: KlustersDoc::realignSpikes (klustersdoc_realign.cpp)
    around the libklustersshared kernel (realign_xcorr_omp.cpp), step for step.  Per unit, up to
    `iters` times:

      * template = MEAN of the currently-aligned spikes, the int64 sum divided by n and truncated toward
        zero (Klusters' int16 template);
      * pre-roll it CIRCULARLY so its summed-|amplitude| peak sample (argmax_s Sigma_ch |tmpl[s,ch]|,
        first maximum) lands on `peak` -- the 0-BASED peak, i.e. Klusters' peakSampleIndex - 1;
      * score every currently-aligned spike over integer lags in [-max_shift, max_shift] by normalised
        cross-correlation with a FULL-window CIRCULAR shift and a constant full-energy denominator,

            score(L) = Sigma_ch Sigma_s tmpl[s,ch] * spike[(s+L) % T, ch] / sqrt(E_tmpl * E_spike)

        in float32, first maximum wins; a spike whose best score is below `min_score` is not moved this
        pass;
      * roll the moved spikes and add the lag to their CUMULATIVE shift (so the total can exceed
        max_shift over the passes, as in Klusters); stop when nothing moves.

    recenter=True adds Klusters' realign mode 2 (--recenter-rms): after the passes, the whole unit is
    shifted so the circular centroid of its channel-summed energy lands on `peak`, unless the profile's
    mean resultant length is below `rmin` (energy not single-lobed) -- rms_recenter_shift.

    Defaults are Klusters' (threshold 0.70, 2 iterations, max_shift = max(1, peak // 2) clamped to
    T - 1).  Units below `min_n` spikes and `noise_label` are left unshifted (Klusters realigns whatever
    cluster is selected).  With upsample > 1 the windows are cubic-spline interpolated first (a fiber-kit
    extension; the template is then a float mean) and the offsets returned in original samples.

    Returns (off, ioff): off (N,) float32 offset in original samples, ioff (N,) int32 rounded;
    res_corrected = res + ioff (Klusters' newTs = ts + shift)."""
    spk = np.asarray(spk)
    f = max(1, int(upsample))
    exact = f == 1 and np.issubdtype(spk.dtype, np.integer)
    if f > 1:
        spk = _upsample_spline(spk.astype(np.float32), f)
        peak = int(peak) * f
    N, T, C = spk.shape
    pk = int(peak)
    ms = (max(1, pk // 2) if max_shift is None else int(max_shift) * f)
    ms = int(min(max(ms, 1), T - 1))
    lags = np.arange(-ms, ms + 1)
    arT = np.arange(T)
    W_all = spk.astype(np.int64) if exact else spk.astype(np.float64)
    spkE = (W_all.astype(np.float64) ** 2).sum((1, 2))
    sqrtSpkE = np.sqrt(spkE)
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(int(l), []).append(i)
    ioff_up = np.zeros(N, np.int64)
    for u, rows in by.items():
        if u == noise_label or len(rows) < min_n:
            continue
        idx = np.asarray(rows)
        W = W_all[idx]
        n = len(idx)
        cur = np.zeros(n, np.int64)
        for _ in range(max(1, int(iters))):
            al = np.take_along_axis(W, ((arT[None, :] + cur[:, None]) % T)[:, :, None], axis=1)
            acc = al.sum(0)
            templ = (np.sign(acc) * (np.abs(acc) // n)).astype(np.float64) if exact else acc / n
            tpk = int(np.argmax(np.abs(templ).sum(1)))
            ts = pk - tpk
            if ts:
                templ = templ[(arT - ts) % T]                     # circular: src = (t - tShift) % T
            tnorm = np.sqrt((templ * templ).sum())
            denom = tnorm * sqrtSpkE[idx]
            best = np.full(n, -np.inf, np.float32)
            blag = np.zeros(n, np.int64)
            alf = al.astype(np.float64)
            for L in lags:
                num = np.einsum("ntc,tc->n", alf[:, (arT + L) % T, :], templ)
                score = np.where(denom > 1e-12, num / np.where(denom > 1e-12, denom, 1.0), 0.0).astype(np.float32)
                upd = score > best
                best[upd] = score[upd]; blag[upd] = L
            sh = np.where(best >= np.float32(min_score), blag, 0)
            if not sh.any():
                break
            cur = cur + sh
        if recenter:
            al = np.take_along_axis(W, ((arT[None, :] + cur[:, None]) % T)[:, :, None], axis=1)
            dg, _, _, applied = rms_recenter_shift((al.astype(np.float64) ** 2).sum((0, 2)), pk, rmin)
            if applied and dg:
                cur = cur + dg
        ioff_up[idx] = cur
    off = ioff_up.astype(np.float32) / f
    ioff = np.rint(off).astype(np.int32)
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
    win = fpca.extract_windows(np.asarray(spk_new, np.float64), basis["recShift"], basis["data2use"])
    fet = fpca.project(win, basis)                         # (n, nCh*nComp), channel-major
    full = np.empty((len(fet), fet.shape[1] + 1), np.int64)
    full[:, :-1] = fst._round_half_away(fet).astype(np.int64)   # ns3 process_pca uses llround (half-away); np.rint is half-even
    full[:, -1] = np.asarray(res_corr, np.int64)           # time feature (last column)
    return full


def _refresh_lag_alias_fets(base, group, refreshed, res_corr, out_tag=""):
    """Rewrite each on-disk _D<lag><dims> alias .fet from a freshly committed base variant.

    fiber-session --out-variant <method>_D<lag><dims> --emit-fet/--emit-pca names the lag
    feature space as an ALIAS of the extraction method: the alias owns a .clu copy, a .fet
    and a .pca, but deliberately NO waveform file -- its .pca is the lag space re-expressed
    as an ordinary basis on a widened window (fiber_pca.lag_basis), so its .fet is a plain
    projection of the METHOD waveform.  A realign commit rewrites the shared .res and the
    method .spk/.fet, which leaves that alias .fet stale twice over: its feature values
    project pre-commit windows, and its final column carries the PRE-correction timestamps,
    disagreeing with the .res beside it.

    So: for every <base>.pca.<tok>.<group> whose token parses with a _D lag and whose
    D-stripped base token is one of the variants this commit just refreshed, project that
    variant's committed waveform onto the alias basis and rewrite the alias .fet -- the
    exact projection the emitter used, since the widened basis exists to make it one plain
    project() (verified equal to the emitter's lag_project to 1e-14, identically rounded).
    `refreshed` maps the RESOLVED committed token -> its (n, nsamp, nch) waveform.  No
    alias .spk is ever written: the alias names a feature space, not a waveform space, and
    a second same-family .spk would make every bare-family resolve ambiguous."""
    g = str(group)
    stem, tail = f"{base}.pca.", f".{g}"
    for pth in sorted(glob.glob(f"{stem}*{tail}")):
        tok = pth[len(stem):-len(tail)]
        if not tok or "." in tok:                          # stage-tagged / off-slot name
            continue
        spec = nio.parse_variant_token(tok)
        if not spec.lag:
            continue
        base_tok = spec.family + (f"_{spec.kind}{spec.order}" if spec.kind else "")
        wav = refreshed.get(base_tok)
        if wav is None:
            continue                                       # its base was not committed this run
        basis = fpca.read_pcad(pth)
        rs, d2u = int(basis["recShift"]), int(basis["data2use"])
        if rs < 0 or rs + d2u > wav.shape[1]:
            _det("fet", f"alias {tok}: widened window [{rs},{rs + d2u}) does not fit "
                        f"nsamp {wav.shape[1]}; .fet left as it was")
            continue
        fet = refeaturize(wav, res_corr, basis)
        fet_out = nio.write_fet(base, group, fet, variant=tok, tag=out_tag)
        _det("fet", f"{fet_out}   (alias of {base_tok}; {fet.shape[1]} features incl. time)")


def open_waveforms(base, group, nsamp, nch, token):
    """The .spk a realign aligns / commits for `token`, by the custody contract (neurosuite-3
    doc/ndmanager-plugins/formats/naming.md): .spk is Shared -- the exact token, then its WAVEFORM
    token for a _D feature-space token (stderiv_C5_D34 -> .spk.stderiv_C5; no _D waveform set exists),
    then standard, then the untagged name.  The contract allows the standard fallback, but realigning
    RAW waveforms for a stderiv request would align and commit the wrong domain, so that case is
    refused here (the contract's own advice: check the resolved method).  Returns (memmap,
    ResolvedInput)."""
    r = nio.resolve(base, "spk", group, token)
    if not r.found:
        raise FileNotFoundError(f"no waveform file for token '{token}': {r.path} (and the standard / "
                                f"untagged fallbacks) not found")
    if nio.is_stderiv_variant(token) and not nio.is_stderiv_variant(r.variant):
        raise FileNotFoundError(f"token '{token}' is a stderiv space but the only waveforms found are "
                                f"{os.path.basename(r.path)}; refusing to realign raw waveforms in its place")
    return nio.open_spk_file(r.path, nsamp, nch), r


def _variants_on_disk(base, group):
    """Every waveform variant with a .spk for the group: standard first, then each method-tagged
    <base>.spk.<token>.<group> (no stage tag, not a _D feature-space alias -- those own no waveforms).
    The `--variants all` set: a realign re-extracts EVERY waveform space from the .fil, so none is
    left stale against the corrected .res."""
    g = str(group)
    stem, tail = f"{base}.spk.", f".{g}"
    toks = []
    for pth in sorted(glob.glob(f"{stem}*{tail}")):
        tok = pth[len(stem):-len(tail)]
        if not tok or "." in tok or nio.parse_variant_token(tok).lag:
            continue
        toks.append(tok)
    return ["standard"] + [t for t in toks if t != "standard"]


def _resolve_variant_token(base, group, variant):
    """The method token actually on disk for a requested one, or the request itself.

    resolve_input family-matches a bare token (asking for 'stderiv' finds
    .spk.stderiv_C5.N), which is what lets the default --variants work on a
    custom-pattern session.  But the caller then needs to know WHICH token it got,
    because the transform to apply and the token to write both depend on it.
    """
    try:                                   # exact / _D -> waveform token, by the custody contract
        r = nio.resolve(base, "spk", group, variant)
        if r.found and r.variant and nio.variant_family(r.variant) == nio.variant_family(variant):
            return r.variant
    except Exception:
        pass
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
        fpca.read_pca(base, group, prefer=[variant])
        return True
    except Exception:
        return False


def _derive_from_fil(v, raw_ext, cfg, group):
    """Re-derive variant `v`'s waveform from the re-extracted .fil window (`raw_ext`: nsamp+1
    samples, one before the window).  Returns None, with the reason logged, when the .fil cannot
    derive it (an opaque variant such as fiber-adlp's, or a _C token without a usable sdiffPairs);
    the caller then rolls the variant's existing .spk so it still matches the shifted .res."""
    spec = nio.parse_variant_token(v)
    if spec.family == "standard":
        wav = raw_ext[:, 1:, :]
    elif spec.family in ("stderiv", "D") and spec.kind is None:
        wav = fst.apply_stderiv_transform(raw_ext)         # SDIFF_ALLPAIRS + temporal diff (verbatim)
    elif spec.family == "stderiv" and spec.kind == "C":
        # _C<order> is the session's OWN pattern, carried in
        # spikeDetection.channelGroups[<group-1>].sdiffPairs -- a partner map
        # (order 4) or reference sets (order 5) in group-local 0-based
        # positions.  Read from the session rather than assumed, so the
        # re-derived waveform is the one this session's extractor produced.
        pat = cfg.get("sdiff_pairs")
        if not pat:
            _log(f"variant '{v}': token says a custom pattern but "
                 f"spikeDetection.channelGroups[{group - 1}].sdiffPairs is absent from "
                 f"{cfg.get('yaml')}; refusing to guess one. Add it, or re-extract with "
                 f"ndm_extractspikes -P.")
            return None
        # The token records the order the extractor applied; the pattern's own
        # grammar implies one too ('+' => sets => order 5).  If they disagree,
        # one of them is wrong about this session, and picking either would
        # write a file whose name contradicts its contents.
        pat_order = fst.sdiff_spec_order(pat)
        # A session mixing forms runs order 6 for every group, an older-form group
        # extracted as the expressions it computes (the same values): read it so.
        if spec.order == 6:
            pat_order = 6
        if spec.order != pat_order:
            _log(f"variant '{v}': token says order {spec.order} but the session's "
                 f"sdiffPairs parses as order {pat_order}; refusing rather than writing "
                 f"a file whose name contradicts its contents.")
            return None
        try:
            if pat_order == 6:
                wav = fst.apply_stderiv_transform(raw_ext, exprs=fst.parse_sdiff_exprs(pat))
            elif pat_order == 5:
                wav = fst.apply_stderiv_transform(raw_ext, sets=fst.parse_sdiff_sets(pat))
            else:
                partner, _root = fst.parse_sdiff_pairs(pat)
                wav = fst.apply_stderiv_transform(raw_ext, partner=partner)
        except ValueError as e:
            _log(f"variant '{v}': sdiffPairs {pat!r} unusable for this group ({e})")
            return None
        _det("sdiff", f"{v}: order {pat_order} from session sdiffPairs {pat!r}")
    elif spec.family == "stderiv" and spec.kind == "S":
        # _S<order>: the extractor's plain spatial derivative (process_extractspikes_stderiv
        # computeSDiff: 1 nearest-neighbour, 2 Laplacian, 3 all-pairs), then the temporal
        # first-difference -- the same rounding / clamping as the other orders.
        try:
            partner, sets = fst.plain_order_spec(spec.order, raw_ext.shape[2])
        except ValueError as e:
            _log(f"variant '{v}': {e}")
            return None
        wav = fst.apply_stderiv_transform(raw_ext, partner=partner, sets=sets)
        _det("sdiff", f"{v}: plain spatial-derivative order {spec.order}")
    else:
        _log(f"variant '{v}': no .fil transform derives it"); return None
    return wav


def realign(base, elec, nsamp, nch, clu_path=None, max_shift=None, iters=2,
            min_n=20, method="klusters", peak=None, min_score=0.70, upsample=1,
            variant="standard", verbose=True, recenter=False, rmin=0.4):
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
    spk, r = open_waveforms(base, elec, nsamp, nch, variant or "standard")
    spk_path = r.path
    res = nio.read_res(base, elec)
    nclu, labels = _read_clu(clu_path or f"{base}.clu.{elec}")
    n = min(len(res), len(labels), spk.shape[0])
    if not (len(res) == len(labels) == spk.shape[0]):
        if verbose:
            _log(f"WARNING length mismatch: res={len(res):,} clu={len(labels):,} "
                 f"spk={spk.shape[0]:,} → using first {n:,}")
    res, labels, spk = res[:n], labels[:n], np.asarray(spk[:n])
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
        # Klusters aligns to peakSampleIndex - 1 (its parameter is 1-based) whatever .spk variant is
        # loaded; the .fil re-extraction keeps using the yaml value as the window geometry.
        if peak is None:
            raise ValueError("klusters method needs the canonical peak sample (cfg['peak'])")
        align_peak = int(peak) - 1
        off, ioff = klusters_offsets(spk, labels, align_peak, max_shift, iters, min_n, min_score, upsample,
                                     recenter=recenter, rmin=rmin)
    else:
        off, ioff = fl.template_offsets(spk, labels, max_shift, iters, min_n)
        align_peak = None
    res_corr = res + ioff.astype(np.int64)
    if verbose:
        nz = np.count_nonzero(ioff)
        up = f" upsample={upsample}x" if (method == "klusters" and upsample > 1) else ""
        if method == "klusters":
            up += (f" threshold={min_score:g} iters={iters} max_shift="
                   f"{max_shift if max_shift is not None else 'peak//2'}" + (f" recenter-rms rmin={rmin:g}" if recenter else ""))
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


def main(argv=None):
    """CLI entry point.  `argv` (default sys.argv[1:]) lets fiber-session run the same commit
    in-process at the end of a sort (fiber_session --realign), flag for flag."""
    ap = argparse.ArgumentParser(
        prog="fiber-realign",
        description="Per-spike Klusters-style realignment + corrected .res spike times, with "
                    "optional re-extraction from .fil and re-featurisation (commit-and-reextract "
                    "finalize).  Probe geometry is read from <session>.yaml; flags override.")
    sy.add_session_args(ap, sdiff=True)
    ap.add_argument("--peak", type=int, default=None,
                    help="override: peak sample index of the window (default the YAML's peakSampleIndex; "
                         "fiber-session passes the one it inferred when the session has no YAML)")
    ap.add_argument("--clu", default=None,
                    help="cluster file (default <base>.clu.<group>; pass the refined/relinked one, "
                         "e.g. <base>.clu.stderiv.<group>.refine)")
    ap.add_argument("--align-algo", "--align-method", dest="align_method", choices=["klusters", "template", "centroid"], default="klusters",
                    help="alignment algorithm (runs on the clu's variant waveform): klusters = iterative "
                         "normalised-xcorr vs pre-aligned mean; template = legacy median/un-normalised; "
                         "centroid = reference-free per-spike energy-centroid (fiber_lib.centroid_shift), "
                         "no peak/template/labels needed.  (Named --align-method to avoid colliding with "
                         "the --method extraction-variant flag used by other tools.)")
    _g = cfgmod.load_global_config()

    def _kd(name, typ, fb):
        v = os.environ.get(name, "")
        if v in (None, ""):
            v = _g.get(name)
        return fb if v in (None, "") else typ(v)
    ap.add_argument("--max-shift", type=int, default=_kd("FK_REALIGN_MAX_SHIFT", int, None),
                    help="klusters: lag search radius (default Klusters': max(1, (peak-1)//2)) "
                         "[FK_REALIGN_MAX_SHIFT]")
    ap.add_argument("--iters", type=int, default=_kd("FK_REALIGN_ITERS", int, 2),
                    help="klusters: template/xcorr passes (Klusters --iterations, default 2) [FK_REALIGN_ITERS]")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--min-score", type=float, default=_kd("FK_REALIGN_MIN_SCORE", float, 0.70),
                    help="klusters: leave a spike unshifted in a pass if its best normalised xcorr is below "
                         "this (Klusters --threshold, default 0.70) [FK_REALIGN_MIN_SCORE]")
    ap.add_argument("--recenter-rms", type=int, default=_kd("FK_REALIGN_RECENTER_RMS", int, 0),
                    help="klusters: 1 = Klusters realign mode 2 (--recenter-rms): after the passes, shift each "
                         "unit so its channel-summed energy centroid lands on the peak [FK_REALIGN_RECENTER_RMS]")
    ap.add_argument("--rmin", type=float, default=_kd("FK_REALIGN_RMIN", float, 0.4),
                    help="klusters --recenter-rms: minimum mean resultant length of the energy profile; below "
                         "it the recenter is skipped (Klusters --rmin, default 0.4) [FK_REALIGN_RMIN]")
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
                    help="'all' = every waveform variant with a .spk for the group (+ the --variant's "
                         "waveform token); or a comma list of feature spaces to refresh from .fil (default: standard + "
                         "stderiv if present).  Each is re-derived from the re-extracted raw window: "
                         "standard=raw; stderiv=SDIFF_ALLPAIRS+temporal-diff; stderiv_C4/_C5 use the "
                         "session's own spikeDetection.channelGroups[N].sdiffPairs pattern "
                         "(partner map / reference sets); then projected onto its .pca")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default="",
                    help="stage tag for committed outputs (default: empty -> overwrite the canonical "
                         ".res/.clu/.spk/.fet[.<variant>].<group> in place; the realign IS the commit). "
                         "Pass a tag only if you want a side-by-side copy, e.g. --out-tag realigned")
    ap.add_argument("--variant", default="",
                    help="feature variant the .clu/.spk/.fet adhere to (e.g. stderiv_C5): drives which "
                         ".spk is aligned and which .fet is refreshed, OVERRIDING the --clu-name inference. "
                         "The pipeline hard-codes 'stderiv' in the clu path, so a custom-sdiffPairs session "
                         "(stderiv_C5) needs this to name its real variant instead of leaning on "
                         "family-matching.  The .res stays the single shared copy (resolve_any).  "
                         "(FK_REALIGN_VARIANT)")
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
    a = ap.parse_args(argv)
    # peak is needed for the klusters align and any commit; ntotal (the .fil width) is needed ONLY for
    # the .fil re-extraction path -- --shift-spk rolls the existing .spk, so it never reads the .fil.
    fil_path = a.reextract or (a.refeaturize and not a.shift_spk)
    need = ("nchan", "nsamp") + (("ntotal",) if fil_path else ()) \
        + (("peak",) if (a.peak is None and (fil_path or a.shift_spk or a.align_method == "klusters")) else ())
    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr, sdiff_pairs=a.sdiff_pairs,
                                    require=need)
    base, group, nsamp, nch = cfg["base"], cfg["group"], cfg["nsamp"], cfg["nchan"]
    peak = a.peak if a.peak is not None else cfg.get("peak")
    # the clu's variant (e.g. stderiv) drives BOTH which .spk is aligned and which variant is committed
    cv, _ctag = _parse_clu_variant_tag(a.clu, base, group) if a.clu else ("", "")
    # --variant (explicit) wins over the clu-name inference and --out-variant, so a stderiv_C5 session
    # names its real variant instead of the hard-coded 'stderiv' the pipeline puts in the clu path.
    out_variant = a.variant or (a.out_variant if a.out_variant is not None else (cv or "standard"))
    res, off, ioff, res_corr, spk, spk_path, labels = realign(
        base, group, nsamp, nch, a.clu, a.max_shift, a.iters, a.min_n,
        method=a.align_method, peak=peak, min_score=a.min_score, upsample=a.upsample,
        variant=out_variant, recenter=bool(a.recenter_rms), rmin=a.rmin)

    # .res is the ONE shared artifact (resolve_any finds it under whatever token it wears): commit the
    # correction back to THAT file, not a feature-variant copy.  Without this, realigning a session whose
    # res is .res.stderiv.N under a stderiv_C5 clu (out_variant=stderiv_C5) spawns a SECOND .res.stderiv_C5.N
    # -- two res per folder, which breaks the single-res invariant.  A stage-tagged commit keeps the tag.
    _rr = nio.resolve_any(base, "res", group)
    _res_variant = _rr.variant if getattr(_rr, "found", False) else out_variant
    res_out = nio.write_res(base, group, res_corr, variant=_res_variant, tag=a.out_tag)
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
        if a.reextract:
            raise SystemExit("[realign] --shift-spk and --reextract are mutually exclusive (one reads the "
                             ".fil, the other rolls the existing .spk)")
        if a.variants and a.variants.strip() == "all":
            want = _variants_on_disk(base, group) + [out_variant]
        elif a.variants:
            want = [v.strip() for v in a.variants.split(",") if v.strip()]
        else:
            want = ["standard"] + [v for v in ("stderiv",) if _variant_present(base, group, v)]
        refreshed = {}                                     # resolved token -> committed waveform
        want = list(dict.fromkeys(_resolve_variant_token(base, group, v) for v in want))
        for v in want:
            try:
                spk_v, _r = open_waveforms(base, group, nsamp, nch, v)
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
            refreshed[vout] = wav
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
        if a.refeaturize:
            _refresh_lag_alias_fets(base, group, refreshed, res_corr, a.out_tag)
    elif a.reextract or a.refeaturize:
        fil = a.fil or f"{base}.fil"
        filmm = nio.open_signal(fil, cfg["ntotal"])
        # read the window PLUS one preceding sample (peak+1, nsamp+1) so the stderiv temporal diff
        # at t=0 uses the true previous .fil sample; standard = raw_ext[:, 1:, :]
        raw_ext = reextract_from_fil(filmm, cfg["channels"], res_corr, nsamp + 1, peak + 1)
        # which feature spaces to refresh: standard always; stderiv (+any listed) when present
        if a.variants and a.variants.strip() == "all":
            want = _variants_on_disk(base, group) + [out_variant]
        elif a.variants:
            want = [v.strip() for v in a.variants.split(",") if v.strip()]
        else:
            want = ["standard"] + [v for v in ("stderiv",) if _variant_present(base, group, v)]
        # Resolve each requested token to the one actually on disk BEFORE choosing a
        # transform.  A bare 'stderiv' family-matches .spk.stderiv_C5.N, and deciding
        # on the requested name would apply the allpairs transform to a session whose
        # waveforms were built from a custom sdiffPairs pattern -- then write the
        # result under a token claiming otherwise.  Deciding on the resolved token is
        # what makes the refusal below reachable at all.
        want = list(dict.fromkeys(_resolve_variant_token(base, group, v) for v in want))
        refreshed = {}                                     # resolved token -> committed waveform
        for v in want:
            wav = _derive_from_fil(v, raw_ext, cfg, group)
            if wav is not None:
                how = f"re-extracted {len(wav):,} {v} spikes from {fil}"
            else:
                # transform-agnostic, like --shift-spk: better than leaving this .spk/.fet stale
                # against the shifted .res (what skipping used to do)
                try:
                    spk_v, _r = open_waveforms(base, group, nsamp, nch, v)
                except Exception:
                    spk_v, _r = None, None
                if _r is None or _r.variant != v:          # never roll a fallback file under v's name
                    _log(f"variant '{v}': no .spk.{v} to roll either; skipping"); continue
                wav = roll_spikes(np.asarray(spk_v[:len(res_corr)]), ioff)
                how = f"rolled {len(wav):,} {v} spikes by offset (the .fil cannot derive it)"
            spk_out = nio.write_spk(base, group, wav, variant=v, tag=a.out_tag)
            refreshed[v] = wav
            _log(f"{how} → {spk_out}")
            if a.refeaturize:
                try:
                    basis = fpca.read_pca(base, group, prefer=[v, "standard", ""] if v == "standard"
                                          else [v, "D"])
                except FileNotFoundError:
                    _det("fet", f"no .pca basis for {v}; .fet not written"); continue
                fet = refeaturize(wav, res_corr, basis)
                fet_out = nio.write_fet(base, group, fet, variant=v, tag=a.out_tag)
                _det("fet", f"{fet_out}   ({fet.shape[1]} features incl. time)")
        if a.refeaturize:
            _refresh_lag_alias_fets(base, group, refreshed, res_corr, a.out_tag)


if __name__ == "__main__":
    main()
