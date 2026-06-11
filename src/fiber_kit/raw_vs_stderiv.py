#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  raw_vs_stderiv.py  —  does the raw .fil waveform discriminate the ORIGINAL
#  (stderiv-defined) fibers better than the stderiv waveform?
#
#  Rationale: a derivative/integral of stderiv is a LINEAR re-encoding, so after
#  re-whitening it's an orthogonal rotation and cannot change a whitened-L2
#  classifier (proven + confirmed empirically).  The raw bandpassed .fil is NOT
#  a linear function of stderiv — stderiv's ALLPAIRS kills common-mode and its
#  temporal first-diff kills DC, and that information cannot be recovered.  So
#  the raw waveform is the one feature set that *could* add separating power.
#  In-sandbox the .fil is only a 25 s slice (~4% of spikes, too few of the
#  confusable units); run this on the full .fil for a powered answer.
#
#  Method (per chunk, with the original stderiv fibers as the classes):
#    1. cluster the chunk with the stderiv pipeline      -> original fibers
#    2. extract raw .fil waveforms at every spike time
#    3. build whiteners for stderiv space AND raw space
#    4. held-out discrimination (amplitude-invariant direction-centroid):
#         - multiclass over fibers with >= --min-spikes
#         - 2-way over the most-confused pairs
#       reported for BOTH feature spaces.  If raw > stderiv on the confusable
#       pairs, the stderiv reduction is discarding separating information.
#
#  Usage:
#    python3 raw_vs_stderiv.py <FileBase> <ElecNo> \
#        --channels 32,33,34,35,36,37,38,39 --ntotal 96 \
#        --nsamp 32 --nchan 8 --sr 32552 \
#        --chunk-min-start 183 --chunk-min 10 --min-spikes 60
# ═══════════════════════════════════════════════════════════════════════════
import argparse, itertools, numpy as np
from sklearn.covariance import LedoitWolf
try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl
try:
    from .fiber_session import cluster_chunk_fine, read_res, open_spkD
except ImportError:
    from fiber_session import cluster_chunk_fine, read_res, open_spkD
try:
    from . import session_yaml as sy
except ImportError:
    import session_yaml as sy
try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio


def whitener_from(windows, mask):
    bm = windows[:, mask, :].reshape(len(windows), -1); nmean = bm.mean(0)
    C = LedoitWolf().fit(bm - nmean).covariance_
    ev, V = np.linalg.eigh(C); ev = np.maximum(ev, 1e-9)
    return V @ np.diag(1 / np.sqrt(ev)) @ V.T, nmean


def dir_feat(waves, W, nmean, mask):
    w = fl.realign(waves)
    X = (w[:, mask, :].reshape(len(w), -1) - nmean) @ W
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def multiclass(featfn, fibs, fine, inset, seed=1):
    rng = np.random.default_rng(seed); cen = {}; tx = []; ty = []
    for k in fibs:
        idx = np.flatnonzero(fine == k); idx = idx[np.isin(idx, inset)]; rng.shuffle(idx); h = len(idx) // 2
        cen[k] = featfn(idx[:h]).mean(0); cen[k] /= np.linalg.norm(cen[k]) + 1e-9
        Xt = featfn(idx[h:]); tx.append(Xt); ty += [k] * len(Xt)
    tx = np.vstack(tx); ty = np.array(ty); K = list(fibs)
    Cm = np.array([cen[k] for k in K]); pred = np.array(K)[(tx @ Cm.T).argmax(1)]
    return float((pred == ty).mean())


def pair_acc(A, B, featfn, fine, inset, reps=10):
    accs = []
    for tr in range(reps):
        rng = np.random.default_rng(tr); o = {}
        for k in (A, B):
            idx = np.flatnonzero(fine == k); idx = idx[np.isin(idx, inset)]; rng.shuffle(idx); h = len(idx) // 2
            c = featfn(idx[:h]).mean(0); o[k] = (c / (np.linalg.norm(c) + 1e-9), featfn(idx[h:]))
        cA, cB = o[A][0], o[B][0]
        for y, (_, Xt) in ((0, o[A]), (1, o[B])):
            accs.append(np.mean(((Xt @ cB) > (Xt @ cA)).astype(int) == y))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser(description="raw .fil vs stderiv discrimination of the original fibers; reads <session>.yaml.")
    sy.add_session_args(ap)
    ap.add_argument("--chunk-min-start", type=float, default=0.0); ap.add_argument("--chunk-min", type=float, default=10.0)
    ap.add_argument("--min-spikes", type=int, default=60); ap.add_argument("--min-group", type=int, default=200)
    a = ap.parse_args()
    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    a.base = cfg["base"]; a.elec = a.group
    a.ntotal = cfg["ntotal"]; a.nchan = cfg["nchan"]; a.nsamp = cfg["nsamp"]; a.sr = cfg["sr"]
    gch = np.array(cfg["channels"], int); OFF = fl.EXTRACT_OFFSET; mask = fl.MASK_FULL

    res = read_res(a.base, a.elec); spk_mm, _ = open_spkD(a.base, a.elec, a.nsamp, a.nchan)
    fil = nio.open_signal(f"{a.base}.fil", a.ntotal)
    s0 = int(a.chunk_min_start * 60 * a.sr); s1 = int((a.chunk_min_start + a.chunk_min) * 60 * a.sr)
    sel = np.flatnonzero((res >= s0) & (res < s1))
    res_c = res[sel]; spk = np.asarray(spk_mm[sel], float)
    span = np.asarray(fil[s0:s1, :][:, gch], float); rel = res_c - s0
    print(f"chunk {a.chunk_min_start:.0f}-{a.chunk_min_start+a.chunk_min:.0f} min: {len(sel)} spikes")

    # stderiv whitener + clustering = ORIGINAL fibers
    Wsd, nmsd, _ = fl.chunk_whitener(span, rel, mask=mask)
    fine, _ = cluster_chunk_fine(spk, res_c, Wsd, nmsd, coarse_mg=a.min_group, mask=mask, sr=a.sr,
                                 method="rkk", dipsplit=False, rkk_dims=6, incl_k=3.0,
                                 merge_corr=0.90, merge_method="sliding")

    # raw waveforms + raw whitener
    T2 = fl.fil_to_spkD_space(span)           # only to reuse the off-spike baseline mask logic
    forb = np.zeros(len(span), bool)
    for sp in rel:
        forb[max(0, sp - 24):min(len(span), sp + 24)] = True
    rng = np.random.default_rng(0); rawbase = []; tries = 0
    while len(rawbase) < 6000 and tries < 400000:
        s = int(rng.integers(0, len(span) - 32)); tries += 1
        if not forb[s:s + 32].any(): rawbase.append(span[s:s + 32])
    Wraw, nmraw = whitener_from(np.array(rawbase), mask)
    inwin = np.flatnonzero((rel >= OFF) & (rel < len(span) - (a.nsamp - OFF)))
    raw = {int(i): span[rel[i] - OFF: rel[i] - OFF + a.nsamp] for i in inwin}

    sd_feat = lambda idxs: dir_feat(spk[idxs], Wsd, nmsd, mask)
    rw_feat = lambda idxs: dir_feat(np.array([raw[int(i)] for i in idxs]), Wraw, nmraw, mask)

    cnt = {int(k): int(((fine == k) & np.isin(np.arange(len(fine)), inwin)).sum()) for k in np.unique(fine[fine >= 0])}
    fibs = [k for k, c in cnt.items() if c >= a.min_spikes]
    print(f"original fibers: {len(np.unique(fine[fine>=0]))}; with >= {a.min_spikes} raw-covered spikes: {len(fibs)}")
    if len(fibs) < 2:
        print("not enough covered fibers — widen --chunk-min or lower --min-spikes"); return

    print(f"\nmulticlass held-out ({len(fibs)} fibers):")
    print(f"  stderiv  : {100*multiclass(sd_feat, fibs, fine, inwin):.1f}%")
    print(f"  raw .fil : {100*multiclass(rw_feat, fibs, fine, inwin):.1f}%")

    cen = {k: sd_feat(np.flatnonzero(fine == k)[np.isin(np.flatnonzero(fine == k), inwin)]).mean(0) for k in fibs}
    cen = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in cen.items()}
    pp = sorted([(float(cen[x] @ cen[y]), x, y) for x, y in itertools.combinations(fibs, 2)], reverse=True)[:8]
    print("\n2-way held-out on the 8 most-similar fiber pairs:")
    print(f"{'pair':>11} {'cos':>5} {'stderiv':>9} {'raw .fil':>9}  delta")
    d = []
    for c, A, B in pp:
        sa, ra = pair_acc(A, B, sd_feat, fine, inwin), pair_acc(A, B, rw_feat, fine, inwin)
        d.append(ra - sa)
        print(f"{A:>5}-{B:<5} {c:>5.2f} {100*sa:>8.0f}% {100*ra:>8.0f}%  {100*(ra-sa):+5.0f}")
    print(f"\nmean raw-minus-stderiv on confusable pairs: {100*np.mean(d):+.1f}%  "
          f"({'raw helps' if np.mean(d)>0.01 else 'no clear gain'})")


if __name__ == "__main__":
    main()
