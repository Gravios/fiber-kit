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


def link(frags, byc, *, pinned, prim_frac, z, win, slide, iou_thr, floor, max_gap, veto, warp_kw, cx_scale=0.0):
    """Conservative adjacent-chunk (+ one-chunk gap) MUTUAL-NN CI-overlap links with the warp veto.
    Within each chunk boundary the fragments are considered HIGH-SNR FIRST, so the cleanest clusters
    anchor their links before the noisier ones.  (The high-SNR RESTRICTION -- linking only the clean
    backbone this pass, deferring low-SNR/contaminated fragments -- is applied by the caller via the
    SNR floor.)  Union-find over the accepted links; returns a label per fragment index."""
    uf = list(range(len(frags)))
    cx_ref = (float(np.median([f["cx"] for f in frags])) + 1e-9) if (cx_scale > 0 and frags) else 1.0

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]; x = uf[x]
        return x

    def score(i, j):
        ch = pair_channels(frags[i], frags[j], pinned, prim_frac)
        return ci_overlap(frags[i], frags[j], ch, z=z, win=win, slide=slide, iou_thr=iou_thr)

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
                if veto and not warp_ok(frags[i], frags[j], **warp_kw):
                    continue
                uf[find(i)] = find(j); linked_fwd.add(i)
    return [find(i) for i in range(len(frags))]
