#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  fiber_intrachunk.py — collapse over-split fragments WITHIN a chunk into units.
#
#  The over-split sort (deliberately many small clusters) fragments one neuron two
#  ways inside a single 12-min chunk:
#    * energy ladder    — one unit split across amplitude levels (the spatial fiber:
#                         different A, depth spread ~30um, scattered x0/z0).
#    * time-shifted dup — one unit split into copies offset by a few samples
#                         (zero-lag cosine can read negative; see mutual_center).
#  Both must collapse to one unit; genuinely different units at the same depth must
#  stay apart.  This is the INTRA-chunk stage; cross-chunk tracking is fiber_link.
#
#  Matching space is STDERIV — the space the sort ran in (raw .spk is only for the
#  monopole position inverse in fiber-cpos, not for shape).  Three signals, all from
#  fiber_geometry, after mutual-centering the templates:
#    1. template cosine        >= cos_thr (0.85)  — shape; energy-invariant once centred.
#    2. inter-channel offsets  RMS <= off_thr (1.0) — the drift-robust differentiator:
#                              separates same depth-but-different units where cosine
#                              alone overlaps (same-unit ~0.4 vs different ~1.15).
#    3. depth |dy0|            <= depth_gate (35um) — the reliable position axis intra
#                              chunk (x0/z0 scatter with energy, so they are NOT gated
#                              here; the full x0,y0,z0,A fingerprint is fiber_link's job).
#  Grouping is COMPLETE-LINKAGE (a fragment joins a unit only if it agrees with EVERY
#  member, not just one) — the anti-chaining guard that stops an energy ladder from
#  walking into a neighbour.
#
#  Validated on real g5 (sirotaA-jg-000005-20120312, 350 min, 30 chunks): 4781
#  fragments -> 1706 per-chunk units; curated over-splits 343-347 and 553/557 each
#  collapse to one (488, the bent energy-extreme, links at cosine 0.94-0.98 once
#  centred — it was a time-shift, not a shape difference); largest units <=44 frags,
#  depth span <=34um, A span <=3x, max internal offset <=0.84 (coherent, not over-
#  merged).  Same neuron in different chunks keeps DIFFERENT ids here by design.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import numpy as np

try:
    from . import fiber_geometry as fg, fiber_lib as fl, neuro_io as nio, session_yaml as sy
except ImportError:
    import fiber_geometry as fg, fiber_lib as fl, neuro_io as nio, session_yaml as sy

DEFAULT_COS_THR = 0.85
DEFAULT_OFF_THR = 1.0
DEFAULT_OFF_NREF = None   # SNR-adaptive offset gate: spike count at which off_thr applies as-is
DEFAULT_OFF_CEIL = 2.0    #   (None -> flat off_thr).  See _off_thr_eff / the g5 offset<->CCG calibration.
DEFAULT_DEPTH_GATE = 35.0   # um; the energy ladder's depth spread (spatial fiber)
DEFAULT_MIN_N = 12          # fragments below this are too noisy to sign reliably


def _offset_rms(o1, o2):
    m = ~np.isnan(o1) & ~np.isnan(o2)
    return float(np.sqrt(np.nanmean((o1[m] - o2[m]) ** 2))) if m.sum() >= 2 else np.inf



def _off_thr_eff(off_thr, n_i, n_j, n_ref, ceil):
    """SNR-adaptive inter-channel offset tolerance.  The offset-RMS between two fragments of the
    SAME neuron carries an estimation-noise floor that scales ~1/sqrt(n) (validated on g5: median
    same-neuron offset 0.25 at n>=300 rises to 0.77 at n<75), so a flat off_thr over-splits small
    fragments.  This loosens the gate for low-count pairs: off_thr * sqrt(n_ref / min(n_i,n_j)),
    clamped to [off_thr, ceil] -- never tighter than the base, never looser than `ceil`.  n_ref=None
    disables adaptation (returns off_thr).  NOTE the offset still carries identity signal at all n
    (P(diff|offset) is ~n-independent on g5), so this is a recall/precision knob, not free recall:
    raising `ceil` past ~2.0 admits a rising different-neuron fraction.  The principled fix for the
    low-n regime is to re-estimate the offset after fragments are pooled (cross-chunk link /
    iterative cluster merge), where the higher spike count makes the offset reliable."""
    if n_ref is None:
        return off_thr
    f = (n_ref / max(min(int(n_i), int(n_j)), 1)) ** 0.5
    return min(ceil, off_thr * max(1.0, f))


def kernel_twosample(Xp, Xq, kind="kcov"):
    """Kernel two-sample divergence between two feature-sample sets (rows = samples).

    kind='mmd'  : squared MMD with a Gaussian kernel (median-heuristic bandwidth) --
                  the kernel MEAN embedding distance (sensitive to any difference).
    kind='kcov' : the same with the SQUARED kernel (half the bandwidth), i.e. the
                  Hilbert-Schmidt distance between the kernel COVARIANCE embeddings
                  N(0,S_P), N(0,S_Q) -- the separation-of-measure statistic, strongest
                  on hard high-similarity pairs.
    Returns a non-negative divergence (0 = identical distributions); caller gates on
    a calibrated upper threshold."""
    Xp = np.asarray(Xp, float); Xq = np.asarray(Xq, float)
    Z = np.vstack([Xp, Xq]); n = len(Xp)
    G = Z @ Z.T                                            # Gram (no 3-D broadcast tensor)
    sq = np.diag(G)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
    iu = np.triu_indices(len(Z), 1)
    nz = d2[iu][d2[iu] > 0]
    if nz.size == 0:
        return 0.0
    bw = np.median(nz) * (1.0 if kind == "mmd" else 0.5)
    K = np.exp(-d2 / bw)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def build_signatures(spkD, clu, t_mid_s, pos, *, chunk_min=12.0, min_n=DEFAULT_MIN_N,
                     reserve=(0, 1), sigma=fg.DEFAULT_SMOOTH_SIGMA,
                     feats=None, feat_dim=12, feat_n=80, feat_seed=0, realign_lohi=None):
    """Per-cluster stderiv signature for matching.

    spkD     : (nspk, nsamp, nchan) int16 stderiv waveforms (array or memmap).
    clu      : (nspk,) source cluster id per spike.
    t_mid_s  : (nspk,) spike time in seconds (from .res / sampling rate) — its
               per-cluster mean places the cluster in a chunk.
    pos      : {clu_id: (x0, y0, z0, A)} from the fiber-cpos cluster table.

    feats    : if 'wave', also attach a per-cluster sample of up to feat_n spikes
               projected onto a global PCA(feat_dim) of the stderiv waveforms, for the
               kernel two-sample co-gate (group_intrachunk gate='mmd'/'kcov').  Default
               None keeps the old behaviour (cosine gate only).

    Returns a dict of per-cluster arrays keyed by cluster id order in `ids`:
      ids, template (nclu,nsamp,nchan, mutual-centred mean), offset (nclu,nchan),
      x0,y0,z0,A, chunk, t_mid, n[, feat].  Clusters in `reserve`, below min_n, or
      absent from `pos` are skipped (they stay singletons downstream)."""
    order = np.argsort(clu, kind="stable")
    cs = clu[order]
    uq, st = np.unique(cs, return_index=True)
    en = np.r_[st[1:], len(cs)]
    Vt = mu_w = None
    if feats == "wave":                                   # global PCA basis for the gate
        rng = np.random.default_rng(feat_seed)
        Wf = np.asarray(spkD, np.float32).reshape(len(spkD), -1)
        Wf = Wf / (np.linalg.norm(Wf, axis=1, keepdims=True) + 1e-9)   # SHAPE space: unit-norm each
        s = rng.choice(len(Wf), min(len(Wf), 100000), replace=False)   # spike (drop the energy axis the
        mu_w = Wf[s].mean(0)                                            # cosine gate also discards), so an
        Vt = np.linalg.svd(Wf[s] - mu_w, full_matrices=False)[2][:feat_dim]   # energy ladder stays one unit
    ids, T, O, X, Y, Z, A, CH, TM, NS, FT = ([] for _ in range(11))
    for k, c in enumerate(uq):
        c = int(c)
        if c in reserve or c not in pos:
            continue
        idx = order[st[k]:en[k]]
        if len(idx) < min_n:
            continue
        al = fg.mutual_center_spikes(fg.denoise(fl.realign(spkD[idx].astype(float), *(realign_lohi or ())), sigma))
        tmpl = al.mean(0)
        x0, y0, z0, amp = pos[c]
        tm = float(np.mean(t_mid_s[idx]))
        ids.append(c); T.append(tmpl); O.append(fg.interchannel_offsets(tmpl))
        X.append(x0); Y.append(y0); Z.append(z0); A.append(abs(amp))
        CH.append(int(tm / 60.0 // chunk_min)); TM.append(tm); NS.append(len(idx))
        if Vt is not None:
            ridx = idx if len(idx) <= feat_n else np.random.default_rng(feat_seed + c).choice(idx, feat_n, replace=False)
            sp = np.asarray(spkD[ridx], np.float32).reshape(len(ridx), -1)
            sp = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-9)
            FT.append((sp - mu_w) @ Vt.T)
    out = dict(ids=np.array(ids, int), template=np.array(T, np.float32),
               offset=np.array(O, np.float32), x0=np.array(X), y0=np.array(Y),
               z0=np.array(Z), A=np.array(A), chunk=np.array(CH, int),
               t_mid=np.array(TM), n=np.array(NS, int))
    if Vt is not None:
        out["feat"] = np.array(FT, dtype=object)
    return out


def group_intrachunk(sig, *, cos_thr=DEFAULT_COS_THR, off_thr=DEFAULT_OFF_THR,
                     depth_gate=DEFAULT_DEPTH_GATE, gate="cosine", feat_q=0.90,
                     off_n_ref=DEFAULT_OFF_NREF, off_ceil=DEFAULT_OFF_CEIL):
    """Per-chunk complete-linkage clique on (similarity, offset, depth).  Returns a
    per-cluster integer label (dense, 0-based) — one label per per-chunk unit.

    gate : 'cosine' (default) tests mean-template cosine >= cos_thr.  'mmd'/'kcov' instead
           run a kernel two-sample test on each fragment's attached shape features
           (sig['feat']; build_signatures(..., feats='wave')), applied only to cosine-passing
           pairs as a precision filter, with a threshold self-calibrated from per-fragment
           split-half nulls at feat_q.  'kcov' is the covariance-embedding (separation-of-
           measure) statistic.

           CAUTION: as a *grouping* gate the kernel tests OVER-SPLIT -- they are powerful
           enough to separate a single neuron's own sub-populations (energy ladder, SNR
           bands), which the cosine gate is intentionally blunt enough to merge.  On g5 they
           raise the per-chunk unit count ~1.6x and do not reduce downstream conflicts (those
           are a link-stage phenomenon; see fiber_trajectory).  Their validated use is
           discriminating already-formed *units* (e.g. same-chunk conflict pairs, AUC ~0.99),
           not collapsing fragments.  Left available and off by default for that experiment;
           do not enable for routine grouping.  Depth and offset gates apply in every mode."""
    T = sig["template"].reshape(len(sig["ids"]), -1)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    off, Y, chunk = sig["offset"], sig["y0"], sig["chunk"]
    use_kernel = gate in ("mmd", "kcov")
    if use_kernel:
        if "feat" not in sig:
            raise ValueError("gate='%s' needs sig['feat'] -- call build_signatures(..., feats='wave')" % gate)
        feat = sig["feat"]
        rng = np.random.default_rng(0); nulls = []          # self-calibrate from split-half nulls
        cal = range(len(feat)) if len(feat) <= 500 else rng.choice(len(feat), 500, replace=False)
        for fi in cal:
            f = feat[fi]
            if len(f) >= 8:
                r = rng.permutation(len(f)); h = len(r) // 2
                nulls.append(kernel_twosample(f[r[:h]], f[r[h:]], gate))
        thr = float(np.quantile(nulls, feat_q)) if nulls else np.inf
    Ncnt = sig.get("n")
    label = np.full(len(sig["ids"]), -1, int); nxt = 0
    for ch in np.unique(chunk):
        ix = np.flatnonzero(chunk == ch); n = len(ix)
        C = Tn[ix] @ Tn[ix].T
        edges = []
        for a in range(n):
            for b in range(a + 1, n):
                i, j = ix[a], ix[b]
                if abs(Y[i] - Y[j]) > depth_gate:           # depth gate first (cheap prefilter)
                    continue
                if use_kernel:
                    if C[a, b] < cos_thr:                   # cosine is the cheap recall prefilter;
                        continue                            # the kernel test is the precision filter
                    s = kernel_twosample(feat[i], feat[j], gate)
                    if s > thr:
                        continue
                    strength = thr - s                      # smaller divergence -> stronger
                else:
                    if C[a, b] < cos_thr:
                        continue
                    strength = C[a, b]
                o = _offset_rms(off[i], off[j])
                ot = off_thr if (Ncnt is None or off_n_ref is None) else _off_thr_eff(off_thr, Ncnt[i], Ncnt[j], off_n_ref, off_ceil)
                if o <= ot:
                    edges.append((strength - o, a, b))      # strongest agreement first
        edges.sort(reverse=True)
        par = list(range(n)); mem = {k: [k] for k in range(n)}
        def root(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        es = {(a, b) for _, a, b in edges} | {(b, a) for _, a, b in edges}
        for _, a, b in edges:
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            Ma, Mb = mem[ra], mem[rb]
            if all((p, q) in es for p in Ma for q in Mb):   # clique: agree with ALL
                par[rb] = ra; mem[ra] = Ma + Mb; del mem[rb]
        loc = np.array([root(k) for k in range(n)])
        for L in np.unique(loc):
            label[ix[loc == L]] = nxt; nxt += 1
    return label


def aggregate_units(sig, label):
    """Collapse fragment signatures into per-chunk UNIT signatures (n-weighted,
    re-centred template; n-weighted position).  This is the table fiber_link links
    across chunks.  Returns a dict shaped like build_signatures' output plus
    `members` (list of source-cluster-id arrays) and `unit` (0-based unit id)."""
    U = np.unique(label); uN = len(U)
    out = dict(unit=np.arange(uN), template=np.zeros((uN, *sig["template"].shape[1:]), np.float32),
               offset=np.zeros((uN, sig["offset"].shape[1]), np.float32),
               x0=np.zeros(uN), y0=np.zeros(uN), z0=np.zeros(uN), A=np.zeros(uN),
               chunk=np.zeros(uN, int), t_mid=np.zeros(uN), n=np.zeros(uN, int), members=[])
    for k, L in enumerate(U):
        mm = np.flatnonzero(label == L); w = sig["n"][mm].astype(float); wn = w / w.sum()
        t = fg.mutual_center((sig["template"][mm] * w[:, None, None]).sum(0) / w.sum())
        out["template"][k] = t; out["offset"][k] = fg.interchannel_offsets(t)
        for f in ("x0", "y0", "z0", "A", "t_mid"):
            out[f][k] = float((sig[f][mm] * wn).sum())
        out["chunk"][k] = int(sig["chunk"][mm[0]]); out["n"][k] = int(w.sum())
        out["members"].append(sig["ids"][mm])
    return out


def group_intrachunk_iter(sig, *, max_iter=5, **kw):
    """Iterated intra-chunk grouping: group -> aggregate (re-estimate clean unit signatures from the
    pooled spikes) -> regroup, to convergence, composing the fragment->unit map across passes.  Because
    aggregation denoises each partial merge, the SAME tight gate keeps finding merges on later passes
    (g5: 1724 one-pass -> 1124 at max_iter>=4, tight gate, within-unit CCG ~0.07) -- the principled
    alternative to loosening off_thr for low-count fragments.  **kw forward to group_intrachunk every
    pass, so --off-n-ref / --cos-thr / gate compose.  max_iter=1 reproduces a single group_intrachunk
    pass exactly (same partition).  Returns a dense 0-based label over the ORIGINAL fragment rows."""
    cur = sig
    f2u = np.arange(len(sig["ids"]))
    for _ in range(max_iter):
        lab = group_intrachunk(cur, **kw)
        if len(np.unique(lab)) == len(cur["ids"]):     # nothing merged this pass -> converged
            break
        f2u = lab[f2u]                                  # compose orig->cur with cur->new
        u = aggregate_units(cur, lab); u["ids"] = u["unit"]
        cur = u
    return f2u


def _boundary_sig(waves, sigma=fg.DEFAULT_SMOOTH_SIGMA):
    """Centred mean template (flattened, unit-norm), inter-channel offsets, and an
    energy-weighted channel-centroid depth (geometry-free, channel units) from a small
    boundary-window spike stack.  None if too few spikes to sign."""
    al = fg.mutual_center_spikes(fg.denoise(fl.realign(np.asarray(waves, float)), sigma))
    t = al.mean(0)
    e = (t ** 2).sum(0)
    dep = float((np.arange(t.shape[1]) * e).sum() / (e.sum() + 1e-9))
    return t.ravel() / (np.linalg.norm(t) + 1e-9), fg.interchannel_offsets(t), dep


def overlap_backbone(units, member_spikes, spkD, t_spike_s, *, chunk_min=12.0,
                     half_window=3.0, cos_thr=0.90, off_thr=0.80, depth_tol=1.0, min_n=8):
    """Anchor units across each chunk boundary using only the spikes that STRADDLE it.

    Spikes within +/-half_window minutes of a boundary are measured at almost the same
    time, so the drift between the chunk-c side and the chunk-(c+1) side is ~0.  Matching
    boundary-window signatures (template cosine + offset + boundary depth) therefore links
    the same neuron across the boundary with the drift confound removed.  Two outputs:

      * backbone links  : (unit_i, unit_j) pairs -- the high-confidence cross-chunk
                          correspondence (mutual-NN cosine>=cos_thr, offset<=off_thr,
                          boundary depth within depth_tol channels).
      * drift D(c)      : cumulative median of the *whole-cluster* depth difference over
                          backbone pairs -- a true per-unit drift estimate that does not
                          suffer the composition contamination of a density cross-corr
                          (on real g5 it reads a small wandering +/-~20um where the density
                          method read a spurious ~120um).

    member_spikes : list (per unit) of spike indices into spkD / t_spike_s.
    t_spike_s     : per-spike time in seconds.  Returns (links, {chunk_id: D_um})."""
    uC = units["chunk"]; uY = units["y0"]; uN = len(uC)
    chunks = sorted({int(c) for c in uC})
    ut = [t_spike_s[ix] for ix in member_spikes]
    links = []; D = {chunks[0]: 0.0}
    for kk in range(1, len(chunks)):
        tb = chunks[kk] * chunk_min * 60.0                  # boundary time (s)
        lo, hi = tb - half_window * 60.0, tb + half_window * 60.0
        a = [u for u in range(uN) if uC[u] == chunks[kk - 1]]
        b = [u for u in range(uN) if uC[u] == chunks[kk]]
        SA, SB = {}, {}
        for u in a:
            m = (ut[u] >= lo) & (ut[u] < tb)
            if m.sum() >= min_n:
                SA[u] = _boundary_sig(spkD[member_spikes[u][m]])
        for u in b:
            m = (ut[u] >= tb) & (ut[u] < hi)
            if m.sum() >= min_n:
                SB[u] = _boundary_sig(spkD[member_spikes[u][m]])
        if len(SA) < 2 or len(SB) < 2:
            D[chunks[kk]] = D[chunks[kk - 1]]; continue
        al, bl = list(SA), list(SB)
        TA = np.array([SA[u][0] for u in al]); TB = np.array([SB[u][0] for u in bl])
        DA = np.array([SA[u][2] for u in al]); DB = np.array([SB[u][2] for u in bl])
        C = TA @ TB.T; pairs = []
        for ii in range(len(al)):
            jj = int(np.argmax(C[ii]))
            if int(np.argmax(C[:, jj])) == ii and C[ii, jj] >= cos_thr \
                    and _offset_rms(SA[al[ii]][1], SB[bl[jj]][1]) <= off_thr \
                    and abs(DA[ii] - DB[jj]) <= depth_tol:
                pairs.append((al[ii], bl[jj]))
        links += [(int(i), int(j)) for i, j in pairs]
        ds = float(np.median([uY[j] - uY[i] for i, j in pairs])) if pairs else 0.0
        D[chunks[kk]] = D[chunks[kk - 1]] + ds
    return links, D


def member_spike_index(src_ids, members):
    """list (per unit) of spike indices into the source arrays, from each unit's member
    source-cluster ids (the inverse of build_signatures' grouping)."""
    order = np.argsort(src_ids, kind="stable"); cs = src_ids[order]
    uq, st = np.unique(cs, return_index=True); en = np.r_[st[1:], len(cs)]
    by_clu = {int(u): order[st[k]:en[k]] for k, u in enumerate(uq)}
    return [np.concatenate([by_clu[int(c)] for c in mm]) if len(mm) else np.array([], int)
            for mm in members]


def intrachunk_clu(src_ids, sig_ids, label, *, reserve=(0, 1)):
    """Map every source spike's cluster id to its per-chunk unit id.  Reserve ids pass
    through; clusters not signed (reserve / too few spikes / no position) keep fresh
    singleton ids so nothing is silently dropped.  Returns (per-spike ids, n_clusters)."""
    unit_of = {int(sig_ids[k]): int(label[k]) for k in range(len(sig_ids))}
    nid = max(reserve) + 1; remap = {}
    def fresh(key):
        if key not in remap:
            remap[key] = len(remap)
        return remap[key]
    lut = {}
    for c in np.unique(src_ids):
        c = int(c)
        lut[c] = c if c in reserve else (nid + fresh(("U", unit_of[c])) if c in unit_of
                                         else nid + fresh(("S", c)))
    out = np.array([lut[int(c)] for c in src_ids], np.int32)
    return out, int(out.max()) + 1


def main():
    ap = argparse.ArgumentParser(description="Collapse over-split fragments within each "
                                             "chunk into units (stderiv cosine + offset + depth).")
    sy.add_session_args(ap, channels=False, ntotal=False, nsamp=False, nchan=False, sr=False)
    ap.add_argument("--cpos-method", default="stderiv")
    ap.add_argument("--cpos-stage", default="refine")
    ap.add_argument("--clu-method", default=None); ap.add_argument("--clu-stage", default=None)
    ap.add_argument("--chunk-minutes", type=float, default=12.0)
    ap.add_argument("--cos-thr", type=float, default=DEFAULT_COS_THR)
    ap.add_argument("--off-thr", type=float, default=DEFAULT_OFF_THR)
    ap.add_argument("--off-n-ref", type=float, default=DEFAULT_OFF_NREF,
                    help="SNR-adaptive offset gate: spike count at which --off-thr applies as-is; "
                         "loosens ~1/sqrt(n) below it (recommend ~150). Omit for flat off_thr.")
    ap.add_argument("--off-ceil", type=float, default=DEFAULT_OFF_CEIL,
                    help="cap on the adaptive offset tolerance (default 2.0; ~95%% same-neuron knee).")
    ap.add_argument("--iter", type=int, default=1, dest="n_iter",
                    help="iterate group->re-estimate->regroup this many passes (default 1 = single pass). "
                         ">1 keeps the tight gate but re-merges denoised units across passes (g5: 5 -> ~1124).")
    ap.add_argument("--depth-gate", type=float, default=DEFAULT_DEPTH_GATE)
    ap.add_argument("--gate", choices=("cosine", "mmd", "kcov"), default="cosine",
                    help="fragment-merge test: 'cosine' (mean template, default & recommended). "
                         "'mmd'/'kcov' are kernel two-sample tests (precision filter on cosine-passing "
                         "pairs); NOTE they OVER-SPLIT as a grouping gate (they separate a neuron's own "
                         "energy/SNR sub-populations) -- exposed for experimentation on unit-vs-unit "
                         "discrimination, not for routine grouping")
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    ap.add_argument("--boundary-minutes", type=float, default=3.0, help="half-window (min) of straddling spikes for the overlap backbone anchor (--emit-units)")
    ap.add_argument("--out-stage", default=None, help="output .clu stage (default: <clu-stage>.intrachunk)")
    ap.add_argument("--emit-units", action="store_true", help="also write a <...>.units.npz unit-signature table for fiber-link")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, require=("ntotal", "sr"))
    base = cfg.base; elec = a.group; sr = float(cfg.sr)
    nsamp = int(cfg.nsamp); nch = int(cfg.nchan)
    clu_method = a.clu_method if a.clu_method is not None else a.cpos_method
    clu_stage = a.clu_stage if a.clu_stage is not None else a.cpos_stage
    out_stage = a.out_stage if a.out_stage is not None else (f"{clu_stage}.intrachunk" if clu_stage else "intrachunk")

    _, src = nio.read_clu_at(base, elec, variant=clu_method, tag=clu_stage)
    res = nio.read_res(base, elec)
    spkD, _ = nio.open_spkD(base, elec, nsamp, nch)   # open_spkD returns (memmap, path)
    tbl = nio.session_path(base, "cpos", elec, variant=a.cpos_method, tag=a.cpos_stage) + ".clusters.npz"
    z = np.load(tbl)
    pos = {int(c): (float(x), float(y), float(zz), float(A))
           for c, x, y, zz, A in zip(z["clu"], z["x0"], z["y0"], z["z0"], z["A"])}

    _m = fl.build_masks(nsamp, cfg.peak)                  # peak-relative realign window for this nSamples
    sig = build_signatures(spkD, src.astype(np.int64), res.astype(float) / sr, pos,
                           chunk_min=a.chunk_minutes, min_n=a.min_n,
                           feats="wave" if a.gate != "cosine" else None,
                           realign_lohi=(_m.realign_lo, _m.realign_hi))
    label = group_intrachunk_iter(sig, max_iter=a.n_iter, cos_thr=a.cos_thr, off_thr=a.off_thr, depth_gate=a.depth_gate,
                             off_n_ref=a.off_n_ref, off_ceil=a.off_ceil,
                             gate=a.gate)
    newids, ncl = intrachunk_clu(src, sig["ids"], label)
    out_path = nio.session_path(base, "clu", elec, variant=clu_method, tag=out_stage)
    nio.write_clu_file(out_path, newids, n_clusters=ncl)
    nunits = len(np.unique(label))
    print(f"[intrachunk] {len(sig['ids'])} signed fragments over "
          f"{len(np.unique(sig['chunk']))} chunks -> {nunits} per-chunk units")
    print(f"[intrachunk] wrote {out_path}  ({ncl} clusters incl reserve+singletons)")
    if a.emit_units:
        units = aggregate_units(sig, label)
        mspk = member_spike_index(src.astype(np.int64), units["members"])
        bb, D = overlap_backbone(units, mspk, spkD, res.astype(float) / sr,
                                 chunk_min=a.chunk_minutes, half_window=a.boundary_minutes)
        dch = np.array(sorted(D)); dum = np.array([D[c] for c in dch], float)
        upath = out_path + ".units.npz"
        np.savez(upath, **{k: v for k, v in units.items() if k != "members"},
                 members=np.array(units["members"], dtype=object),
                 backbone=np.array(bb, int).reshape(-1, 2),
                 drift_chunks=dch, drift_um=dum)
        print(f"[intrachunk] wrote {upath}  ({nunits} unit signatures, {len(bb)} overlap-backbone "
              f"anchors, drift {dum.min():.0f}..{dum.max():.0f}um for fiber-link)")


if __name__ == "__main__":
    main()
