#  fiber_link_core.py -- shared cross-chunk LINKING MECHANISMS.
#
#  Extracted VERBATIM from fiber_backbone_link.py so more than one caller can reuse the
#  same, validated matcher/veto/linker:
#    * ci_overlap   -- best-lag energy-scaled median+/-z*sigma band-overlap match (identity/recall)
#    * warp_ok      -- Omlor-Giese warp precision veto (warp_correlation + amp-profile + incongruity)
#    * link         -- conservative adjacent-chunk (+ one-chunk gap) mutual-NN linker over those two,
#                      high-SNR fragments anchoring first, union-find labels
#    * pair_channels / _win -- the backbone-channel selection and window helper the above need
#
#  This is a PURE EXTRACTION: the function bodies are byte-for-byte identical to the
#  fiber_backbone_link originals, so behaviour is unchanged.  fiber_backbone_link now imports
#  these; a second consumer (fiber-session's overlap-anchor cross-chunk linker) can import the
#  same core rather than re-implementing it.  A fragment is the dict that build_frag produces
#  (med, sd, c, gd, dom, snr, cx); building it stays with each caller.
try:
    from . import fiber_geometry as fg
except ImportError:
    import fiber_geometry as fg
import numpy as np


def _win(lo, hi, c, win, ns):
    s, e = c - win, c + win
    return (lo[s:e + 1], hi[s:e + 1]) if (s >= 0 and e < ns) else None


def pair_channels(A, B, pinned, prim_frac):
    """Backbone channels for a pair: pinned set, or the SHARED primary channels
    (p2p >= prim_frac*max in both)."""
    if pinned is not None:
        return pinned
    pa = np.ptp(A["med"], 0); pb = np.ptp(B["med"], 0)
    return np.flatnonzero((pa >= prim_frac * pa.max()) & (pb >= prim_frac * pb.max()))


def ci_overlap(A, B, chans, *, z, win, slide, iou_thr):
    """Best-lag ENERGY-SCALED median+/-z*sigma band overlap on `chans`; returns mean IoU.
    Each fragment's band (median and +/-z*sigma) is normalised to unit energy over the compared
    window x channels, because spike-to-spike variance scales with waveform energy -- an absolute
    sigma band would give a high-amplitude cluster an unfairly wide band that overlaps its
    co-located neighbours.  After scaling the band width is RELATIVE variance (sigma/energy) and the
    overlap tests shape consistency at matched scale (amplitude/footprint is the warp veto's job)."""
    if len(chans) == 0:
        return np.nan
    hA = z * A["sd"]; hB = z * B["sd"]; ns = A["med"].shape[0]
    wA = _win(A["med"] - hA, A["med"] + hA, A["c"], win, ns)
    if wA is None:
        return np.nan
    aL, aH = wA[0][:, chans], wA[1][:, chans]
    eA = float(np.linalg.norm((aL + aH) * 0.5)) + 1e-9       # energy of A's median over the compared region
    aL, aH = aL / eA, aH / eA
    best = np.nan
    for L in range(-slide, slide + 1):
        wB = _win(B["med"] - hB, B["med"] + hB, B["c"] + L, win, ns)
        if wB is None:
            continue
        bL, bH = wB[0][:, chans], wB[1][:, chans]
        eB = float(np.linalg.norm((bL + bH) * 0.5)) + 1e-9
        bL, bH = bL / eB, bH / eB
        inter = np.clip(np.minimum(aH, bH) - np.maximum(aL, bL), 0, None)
        union = np.clip(np.maximum(aH, bH) - np.minimum(aL, bL), 1e-12, None)
        miou = float((inter / union).mean())
        if not np.isfinite(best) or miou > best:
            best = miou
    return best


def warp_ok(A, B, *, warp_thr, amp_thr, resid_thr):
    if warp_thr is not None and fg.warp_correlation(A["gd"], B["gd"]) < warp_thr:
        return False
    if amp_thr is not None and fg.amp_profile_correlation(A["med"], B["med"]) < amp_thr:
        return False
    if resid_thr is not None and fg.warp_channel_incongruity(A["gd"], B["gd"]) > resid_thr:
        return False
    return True


def _compose_transform(R, t, k0, k1):
    """Compose the per-adjacent-pair rigid transforms from chunk k0 to k1 (k1 > k0):
    p_{k1} = p_{k0} @ Rc.T + tc.  Indices outside R's range are skipped (identity)."""
    if R.ndim != 3 or R.shape[0] == 0:
        return None, None
    K = R.shape[1]; Rc = np.eye(K); tc = np.zeros(K)
    for c in range(k0, k1):
        if 0 <= c < R.shape[0]:
            Rc = R[c] @ Rc; tc = tc @ R[c].T + t[c]
    return Rc, tc


def _apply_drift(med, Rc, tc, basis, mean):
    """Drift-correct a RAW template by the predicted between-chunk displacement.  The transform
    lives in the PCA subspace; only that in-subspace displacement is added to the FULL template,
    so waveform detail outside the subspace is preserved (no lossy reconstruction)."""
    flat = med.ravel().astype(float)
    p = (flat - mean) @ basis.T
    delta = ((p @ Rc.T + tc) - p) @ basis
    return (flat + delta).reshape(med.shape)


def fit_drift_transforms(pos, anchor_links, K):
    """Per-adjacent-chunk-pair RIGID (rotation+translation) Procrustes fit of the fiber-template
    constellation, from the overlap-anchor correspondences (the same physical spikes in both
    chunks -> drift-free training data).  `pos` maps (chunk,label) -> a K-vector of RAW-template
    PCA coords (raw only: stderiv breaks the amplitude-distance law and mutes drift).
    `anchor_links` is per-adjacent-pair [(label_c, label_c+1), ...].  Returns per-pair R (KxK),
    t (K), the residual FRACTION of the chunk-to-chunk motion left after the fit, and the anchor
    count used.  Fitting lives HERE (shared) so the consumer can fit in ITS OWN feature space:
    fiber-session discovers the anchors, but backbone-link -- which runs AFTER fiber-realign's
    reextraction -- fits on the aligned waveforms it will actually correct, avoiding the stale
    pre-realign basis."""
    nP = len(anchor_links)
    R = np.stack([np.eye(K) for _ in range(nP)]) if nP else np.zeros((0, K, K))
    t = np.zeros((nP, K)); resid = np.full(nP, np.nan); na = np.zeros(nP, int)
    for c in range(nP):
        keep = [(f, g) for f, g in anchor_links[c] if (c, f) in pos and (c + 1, g) in pos]
        na[c] = len(keep)
        if len(keep) < K + 1:                                    # too few anchors to fit a K-dim rotation
            continue
        P = np.array([pos[(c, f)] for f, g in keep]); Q = np.array([pos[(c + 1, g)] for f, g in keep])
        cP, cQ = P.mean(0), Q.mean(0); Pc, Qc = P - cP, Q - cQ
        U, S, Vt = np.linalg.svd(Pc.T @ Qc)
        Rk = Vt.T @ U.T
        if np.linalg.det(Rk) < 0: Vt = Vt.copy(); Vt[-1] *= -1; Rk = Vt.T @ U.T   # reflect -> proper rotation
        R[c] = Rk; t[c] = cQ - cP @ Rk.T
        resid[c] = float(np.linalg.norm(Qc - Pc @ Rk.T) / (np.linalg.norm(Q - P) + 1e-12))
    return R, t, resid, na


def link(frags, byc, *, pinned, prim_frac, z, win, slide, iou_thr, floor, max_gap, veto, warp_kw, cx_scale=0.0, drift=None):
    """Conservative adjacent-chunk (+ one-chunk gap) MUTUAL-NN CI-overlap links with the warp veto.
    Within each chunk boundary the fragments are considered HIGH-SNR FIRST, so the cleanest clusters
    anchor their links before the noisier ones.  (The high-SNR RESTRICTION -- linking only the clean
    backbone this pass, deferring low-SNR/contaminated fragments -- is applied by the caller via the
    SNR floor.)  Union-find over the accepted links; returns a label per fragment index.

    If `drift` is given (dict with basis (K,D), mean (D,), R (nP,K,K), t (nP,K) -- fiber-session's
    overlap-anchor transform), the SOURCE fragment's RAW template is drift-corrected to the target
    chunk's frame before BOTH the band-overlap score AND the warp veto, so the match is drift-
    invariant.  drift=None (default) reproduces the original behaviour byte-for-byte."""
    uf = list(range(len(frags)))
    cx_ref = (float(np.median([f["cx"] for f in frags])) + 1e-9) if (cx_scale > 0 and frags) else 1.0
    chunk_of = {i: k for k in byc for i in byc[k]}       # fragment index -> chunk (for drift indexing)

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]; x = uf[x]
        return x

    def _corrected(i, j):
        """frag i's template drift-corrected toward frag j's chunk (identity if drift is None)."""
        Ai = frags[i]
        if drift is not None:
            ki, kj = chunk_of.get(i), chunk_of.get(j)
            if ki is not None and kj is not None and kj > ki:
                Rc, tc = _compose_transform(drift["R"], drift["t"], ki, kj)
                if Rc is not None:
                    Ai = dict(Ai, med=_apply_drift(Ai["med"], Rc, tc, drift["basis"], drift["mean"]))
        return Ai

    def score(i, j):
        Ai = _corrected(i, j)
        ch = pair_channels(Ai, frags[j], pinned, prim_frac)
        return ci_overlap(Ai, frags[j], ch, z=z, win=win, slide=slide, iou_thr=iou_thr)

    linked_fwd = set()
    for gap in range(1, max_gap + 1):
        for k in sorted(byc):
            A = [i for i in byc.get(k, []) if not (gap > 1 and i in linked_fwd)]
            Bn = byc.get(k + gap, [])
            if not A or not Bn:
                continue
            for i in sorted(A, key=lambda x: -frags[x]["snr"]):   # high-SNR clusters anchor first
                cand = [(score(i, j), j) for j in Bn]
                cand = [(s, j) for s, j in cand if np.isfinite(s)]
                if not cand:
                    continue
                sc, j = max(cand)
                floor_eff = (floor + cx_scale * (1.0 - floor) * max(0.0, 1.0 - min(frags[i]["cx"], frags[j]["cx"]) / cx_ref)
                             if cx_scale > 0 else floor)   # complexity-scaled: simpler fragments must overlap harder
                if sc < floor_eff:
                    continue
                back = [(score(i2, j), i2) for i2 in A]
                back = [(s, i2) for s, i2 in back if np.isfinite(s)]
                if not back or max(back)[1] != i:                 # mutual-NN
                    continue
                if veto and not warp_ok(_corrected(i, j), frags[j], **warp_kw):
                    continue
                uf[find(i)] = find(j); linked_fwd.add(i)
    return [find(i) for i in range(len(frags))]
