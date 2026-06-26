#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  fiber_link.py — co-gated inter-chunk linker: per-chunk fragments -> tracked units.
#
#  Reads a fiber-cpos cluster table (per-fragment position x0,y0,z0,A + realigned raw
#  median template + t_mid) and the source .clu, places each fragment in a 12-min
#  chunk, and links the same neuron across consecutive chunks by:
#
#    1. depth drift D(c)        : (y0, logA) density cross-correlation per chunk
#                                 (drift on a shank is axial -> depth only).
#    2. candidate links         : mutual nearest-neighbour in the full position
#                                 fingerprint (x0, y0-D, z0, logA) -- A is the
#                                 drift-invariant anchor; x0/z0 are reproducible
#                                 per-unit fingerprints even where not physically
#                                 precise on a thin probe.
#    3. shape CO-GATE           : accept a candidate only if the raw template cosine
#                                 >= cos_thr -- vetoes co-located-but-different units.
#    4. union-find -> bundles   : each bundle = one neuron tracked across chunks.
#
#  Writes a global .clu mapping every spike to its fragment's bundle id (over-split
#  fragments of one neuron collapse to a single unit), under the new naming
#  convention <base>.clu.<method>.<group>.<out-stage>.
#
#  Validated on a real raw 4-chunk g5 window (chunks 15-18, 323 clear fragments):
#  62/68 candidates shape-confirmed, confirmed links cosine 0.98 / |dlogA| 0.026,
#  43 bundles span >=2 chunks (15 span >=3).
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import numpy as np

_LP = "\u25b8 fiber-link"
def _log(m=""): print(f"{_LP} \u00b7 {m}" if m else _LP)
def _det(k, v, w=8): print(f"{' ' * (len(_LP) + 3)}{k:<{w}} {v}")

try:
    from . import fiber_lib as fl, fiber_geometry as fg, neuro_io as nio, session_yaml as sy, fiber_score as fsc, fiber_ccg as cg
except ImportError:
    import fiber_lib as fl, fiber_geometry as fg, neuro_io as nio, session_yaml as sy, fiber_score as fsc, fiber_ccg as cg

try:
    from .fiber_cfiber import channel_angles as _cf_angles, complex_loop as _cf_loop, shape_descriptor as _cf_shape
except ImportError:
    from fiber_cfiber import channel_angles as _cf_angles, complex_loop as _cf_loop, shape_descriptor as _cf_shape

try:                                                  # sub-sample template re-registration (shared with intrachunk)
    from .fiber_intrachunk import _register as _register_lag
except ImportError:
    from fiber_intrachunk import _register as _register_lag

# cfiber shape co-gate: affine-invariant (rotation+scale+translation) Fourier descriptors of each
# unit template's complex channel-loop.  Drift-invariant by construction (no mutual_center needed),
# so it complements the cosine gate where amplitude reweighting across chunks hurts cosine.  Used as
# an OPTIONAL co-gate (not a replacement): a candidate must pass cosine AND, if enabled, cfiber.
_CFIBER_MODES = (2, 3, 4, -1, -2, -3)
_CFIBER_PRE, _CFIBER_POST = 10, 12


def _cfiber_win(nsamp, peak):
    p = int(peak) if peak is not None else nsamp // 2
    return slice(max(0, p - _CFIBER_PRE), min(nsamp, p + _CFIBER_POST))


def _cfiber_shapes(templates, win):
    """templates (M,nsamp,nchan) or (nsamp,nchan) -> (M,ndesc) affine-invariant shape descriptors."""
    t = np.asarray(templates, float)
    if t.ndim == 2:
        t = t[None]
    Z = _cf_loop(t, _cf_angles(t.shape[2]), win)
    S, _, _, _ = _cf_shape(Z, _CFIBER_MODES)
    return np.asarray(S, float)


def _cfiber_peak(templates):
    """Window-centre sample: dominant-channel trough of the mean template (units are mutual_centred,
    so this is the common trough sample)."""
    m = np.asarray(templates, float)
    m = m.mean(0) if m.ndim == 3 else m
    dom = int(np.argmax(m.max(0) - m.min(0)))
    return int(np.argmin(m[:, dom]))


def masked_cos(ta, tb, mask):
    a = ta[mask].ravel(); b = tb[mask].ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _primary_mask(ta, tb, smask, frac):
    """Channels BOTH templates treat as primary (peak-to-peak >= frac * the template's own
    max p2p) within the sample window `smask` -- the shared signal core.  Restricting the
    cosine to it drops the near-threshold channels that carry mostly noise and decorrelate a
    true same-neuron cross-chunk pair (on a g5 octrode, median 5 of 8 channels are shared)."""
    p1 = ta[smask].max(0) - ta[smask].min(0); p2 = tb[smask].max(0) - tb[smask].min(0)
    return (p1 >= frac * (p1.max() + 1e-9)) & (p2 >= frac * (p2.max() + 1e-9))


def _cos2d(ta, tb, smask, cmask):
    """Cosine over the intersection of a sample window (smask) and a channel set (cmask)."""
    a = ta[smask][:, cmask].ravel(); b = tb[smask][:, cmask].ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _tan_cos(ta, tb):
    """Cosine of two energy-direction tangents (high-energy minus low-energy template, flattened)
    -- the microfiber criterion: a true same-neuron pair shares the DIRECTION its waveform moves
    with amplitude, an independent confirmation beyond the static mean shape."""
    a = np.asarray(ta, float).ravel(); b = np.asarray(tb, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _offset_rms(o1, o2):
    m = ~np.isnan(o1) & ~np.isnan(o2)
    return float(np.sqrt(np.nanmean((o1[m] - o2[m]) ** 2))) if m.sum() >= 2 else np.inf


def estimate_drift(y0, logA, w, chunk, chunks, *, span_um=24.0, step=3.0):
    """Depth drift D per chunk by (y0,logA) density cross-correlation, accumulated
    relative to the first chunk.  Returns {chunk_id: D_um}."""
    ybins = np.arange(0, 141, step)
    abins = np.arange(logA.min() - 0.1, logA.max() + 0.25, 0.15)
    nshift = int(span_um / step)
    D = {int(chunks[0]): 0.0}
    for k in range(1, len(chunks)):
        a = chunk == chunks[k - 1]; b = chunk == chunks[k]
        if a.sum() < 8 or b.sum() < 8:
            D[int(chunks[k])] = D[int(chunks[k - 1])]; continue
        Hp, _, _ = np.histogram2d(y0[a], logA[a], bins=[ybins, abins], weights=w[a])
        best = (0, -1.0)
        for s in range(-nshift, nshift + 1):
            Hc, _, _ = np.histogram2d(y0[b] - s * step, logA[b], bins=[ybins, abins], weights=w[b])
            v = float((Hc * Hp).sum())
            if v > best[1]:
                best = (s, v)
        D[int(chunks[k])] = D[int(chunks[k - 1])] + best[0] * step
    return D


def cogated_links(x0, y0, z0, logA, tmpl, chunk, chunks, D, mask, *, cos_thr=0.975,
                  pos_thr=1.5, off_thr=1.0, warp_thr=None, offsets=None, gap=1,
                  cfiber_thr=None, cfiber_win=None, amp_gate=0.0,
                  align_lag=0, align_upsample=1, primary_amp_frac=0.0, tan_thr=None, tangents=None,
                  frag_times=None, ov_refrac=None, ov_thr=0.3, ov_min_exp=5.0, ov_censor=0):
    """Mutual-NN candidates in (x0, y0-D, z0, logA) co-gated by template cosine AND
    inter-channel offset.  Templates are mutual_center'd first -- each is circularly shifted so its
    dominant-channel trough sits at a common sample -- which removes a whole-cluster time-offset
    between two chunks before the cosine gate (verified on real g5: a 2-3 sample offset that drops the
    RAW template cosine to ~0.5-0.75 reads ~1.0 after mutual_center; centring alone, i.e. DC removal,
    does NOT do this -- mutual_center's trough re-alignment does).  The alignment is integer-sample, so
    a sub-sample residual (<=0.5 samp) can remain, but there the cosine still sits >0.98, far above any
    usable gate.  The offset RMS co-gate is the drift-robust differentiator that vetoes a co-located
    different unit that happens to share gross shape; off_thr<=0 disables it.

    gap>1 also matches across a skipped chunk (c -> c+2), but only for source units that
    found NO link at a smaller gap -- bridging single-chunk dropouts without competing with
    the adjacent match.  Returns list of (i, j) index pairs (into the passed arrays)."""
    tc = np.array([fg.mutual_center(t) for t in tmpl])
    if offsets is None:
        offsets = np.array([fg.interchannel_offsets(t) for t in tc])
    gd = [fg.group_delay_profile(t) for t in tmpl] if warp_thr is not None else None
    Scf = _cfiber_shapes(np.asarray(tmpl), cfiber_win) if cfiber_thr is not None else None
    sy_ = y0.std() + 1e-9; sx = x0.std() + 1e-9; sz = z0.std() + 1e-9; sa = logA.std() + 1e-9
    links = []; linked_fwd = set()

    def _ov_ok(i, j):                                     # overlap-refractory veto (default off; only ever vetoes)
        if ov_refrac is None or frag_times is None:
            return True
        return cg.overlap_refractory_gate(frag_times[i], frag_times[j], ov_refrac,
                                          thr=ov_thr, min_exp=ov_min_exp, censor=ov_censor)["verdict"] != "veto"

    for g in range(1, gap + 1):
        for k in range(len(chunks) - g):
            ai = np.flatnonzero(chunk == chunks[k]); bi = np.flatnonzero(chunk == chunks[k + g])
            if g > 1:
                ai = np.array([u for u in ai if u not in linked_fwd], int)
            if len(ai) < 2 or len(bi) < 2:
                continue
            dd = D[int(chunks[k + g])] - D[int(chunks[k])]
            Fa = np.vstack([y0[ai] / sy_, x0[ai] / sx, z0[ai] / sz, logA[ai] / sa]).T
            Fb = np.vstack([(y0[bi] - dd) / sy_, x0[bi] / sx, z0[bi] / sz, logA[bi] / sa]).T
            for u in range(len(bi)):
                dist = np.sum((Fb[u] - Fa) ** 2, 1); v = int(np.argmin(dist))
                if int(np.argmin(np.sum((Fa[v] - Fb) ** 2, 1))) == u and np.sqrt(dist[v]) <= pos_thr:
                    Ti = tc[ai[v]]; Tj = tc[bi[u]]
                    if align_lag > 0:                                      # sub-sample re-registration:
                        Tj = _register_lag(Ti, Tj, align_lag, align_upsample)[0]   # recover a same-neuron pair
                    cmask = (_primary_mask(Ti, Tj, mask, primary_amp_frac)         # whose chunk medians sit a
                             if primary_amp_frac > 0 else None)                    # fraction of a sample apart
                    if cmask is not None and int(cmask.sum()) >= 2:               # (the residual integer
                        shape_cos = _cos2d(Ti, Tj, mask, cmask)                   # mutual_center leaves)
                    else:
                        shape_cos = masked_cos(Ti, Tj, mask)                      # full-template fallback
                    if shape_cos >= cos_thr and \
                            (amp_gate <= 0 or abs(logA[ai[v]] - logA[bi[u]]) <= amp_gate) and \
                            (off_thr <= 0 or _offset_rms(offsets[ai[v]], offsets[bi[u]]) <= off_thr) and \
                            (tan_thr is None or tangents is None or _tan_cos(tangents[ai[v]], tangents[bi[u]]) >= tan_thr) and \
                            (warp_thr is None or fg.warp_correlation(gd[ai[v]], gd[bi[u]]) >= warp_thr) and \
                            (cfiber_thr is None or np.linalg.norm(Scf[ai[v]] - Scf[bi[u]]) <= cfiber_thr) and \
                            _ov_ok(ai[v], bi[u]):
                        links.append((int(ai[v]), int(bi[u]))); linked_fwd.add(int(ai[v]))
    return links


def bundles_chunk_exclusive(n_frag, links, chunk, strength):
    """Union-find that REFUSES any merge putting two units in the same chunk.  A neuron cannot be two
    distinct same-chunk units -- the intra-chunk merger would already have merged them if they were one
    -- so a bundle spanning the same chunk twice is a provable over-merge (one bad cross-chunk link
    chaining in a different neuron).  Links are processed strongest-first (`strength`, higher = earlier;
    pass inf for trusted overlap-backbone seeds) so the best match wins under competition.  This vetoes
    only the provable collisions; different neurons that never share a chunk are not separable here."""
    order = sorted(range(len(links)), key=lambda k: -strength[k])
    par = list(range(n_frag)); chs = [{int(chunk[i])} for i in range(n_frag)]
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for k in order:
        i, j = links[k]; ri, rj = find(i), find(j)
        if ri == rj or (chs[ri] & chs[rj]):
            continue
        par[rj] = ri; chs[ri] |= chs[rj]
    groups = {}
    for i in range(n_frag):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def build_bundles(n_frag, links):
    """Union-find over links; every fragment lands in a bundle (singletons for unlinked)."""
    par = list(range(n_frag))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in links:
        par[find(i)] = find(j)
    groups = {}
    for i in range(n_frag):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _trace_var(n, s, q):
    """Trace of the pooled covariance from weighted sufficient stats (same statistic as
    fiber_defrag/fiber_calibrate: sum of per-dimension variances)."""
    return float(np.sum(q / n - (s / n) ** 2))


def _template_pc_stats(templates, counts, n_pc=12):
    """Per-fragment template -> a point in a top-`n_pc` PC space of the fragment templates, plus a
    spike-count weight.  Returns (P (M,k), w (M,)) where a bundle's pooled `_trace_var` over its
    members' P (weighted by w) measures how far apart the welded fragment SHAPES are -- the quantity
    single-linkage chaining inflates.  Baseline-subtracted and flattened exactly as defrag's
    template_cosine so the coordinate matches the rest of the toolchain."""
    M = np.stack([(np.asarray(t, float) - np.asarray(t, float)[:6].mean(0)).ravel() for t in templates])
    w = np.asarray(counts, float)
    Mc = M - M.mean(0)
    k = int(min(int(n_pc), Mc.shape[0] - 1, Mc.shape[1]))
    if k < 1:
        return np.zeros((len(M), 1)), w
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Mc @ Vt[:k].T, w


def _selfcal_var_allow(links, P, w, energy, scale=1.0, top_frac=0.34, pct=95.0):
    """Self-calibrate the variance boundary from the HIGH-ENERGY backbone, the way the cfiber co-gate
    self-calibrates from the overlap backbone.  Take the accepted edges whose BOTH endpoints sit in the
    top `top_frac` of fragment energy (the trusted, well-localized same-unit links), measure each edge's
    2-fragment template `_trace_var`, and set the boundary to `scale` x its `pct` percentile.  Falls
    back to all edges if too few high-energy ones exist.  Returns +inf (no gating) when there are no
    usable edges."""
    if not links:
        return np.inf
    e = np.asarray(energy, float)
    thr = np.quantile(e, 1.0 - top_frac) if e.size else -np.inf

    def _pair_tv(i, j):
        n = w[i] + w[j]
        return _trace_var(n, w[i] * P[i] + w[j] * P[j], w[i] * P[i] ** 2 + w[j] * P[j] ** 2)

    tv = [_pair_tv(i, j) for i, j in links if e[i] >= thr and e[j] >= thr]
    if len(tv) < 3:
        tv = [_pair_tv(i, j) for i, j in links]
    if not tv:
        return np.inf
    return float(np.percentile(tv, pct)) * float(scale)


def bundles_variance_bounded(n_frag, links, chunk, strength, energy, P, w, var_allow):
    """Energy-seeded, variance-bounded sibling of bundles_chunk_exclusive (the anti-chaining bundler).

    Same chunk-exclusivity guard, but two changes that target single-linkage over-merge:
      * ORDER edges high-energy-first -- trusted overlap-backbone seeds (strength=inf) first, then by the
        weaker endpoint's energy descending (an edge is only as reliable as its faintest fragment, so
        low-energy bridge edges are demoted to last), strength as the final tiebreak.  The high-energy
        backbone agglomerates before any confusable low-energy edge competes.
      * GATE every union by a variance boundary: a merge is refused when the combined bundle's pooled
        template `_trace_var` exceeds `var_allow` -- i.e. when welding the two bundles would spread the
        shape past a single neuron's envelope.  This is what stops a high-cosine-but-distinct cross edge
        (which passes the pairwise cosine gate) from chaining two units that never share a chunk.  Seeds
        (strength=inf) bypass the variance gate; they are trusted same-unit by construction.

    `P`,`w` are the per-fragment template PC points + weights from _template_pc_stats; `energy` is per
    fragment (logA).  Returns (bundles, n_blocked)."""
    e = np.asarray(energy, float)

    def _key(k):
        i, j = links[k]
        return (not np.isinf(strength[k]), -min(e[i], e[j]), -strength[k])

    order = sorted(range(len(links)), key=_key)
    par = list(range(n_frag)); chs = [{int(chunk[i])} for i in range(n_frag)]
    N = w.astype(float).copy(); S = (w[:, None] * P).astype(float); Q = (w[:, None] * P ** 2).astype(float)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    blocked = 0
    for k in order:
        i, j = links[k]; ri, rj = find(i), find(j)
        if ri == rj or (chs[ri] & chs[rj]):
            continue
        n2 = N[ri] + N[rj]; s2 = S[ri] + S[rj]; q2 = Q[ri] + Q[rj]
        if not np.isinf(strength[k]) and var_allow < np.inf and _trace_var(n2, s2, q2) > var_allow:
            blocked += 1                                  # variance boundary -> reject (anti-chaining)
            continue
        par[rj] = ri; chs[ri] |= chs[rj]
        N[ri] = n2; S[ri] = s2; Q[ri] = q2
    groups = {}
    for i in range(n_frag):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), blocked


def _graph_links(method, frag, idx, y0, logA, chunk, D, mask, offs, knn=7):
    """Global graph linkage via graph_link.spectral_partition, an alternative to the per-pair
    cogated_links veto stack (mutual-NN position + pos_thr + off_thr + cosine) which leaves
    high-cosine blocks unmerged when the session geometry/drift differ from the calibration set.

    Builds a discriminative feature per linkable unit -- mutual-centred masked template (+)
    DRIFT-CORRECTED position (+) logA (+) inter-channel offsets, each block z-scored so pos/offset
    enter as weights rather than hard vetoes -- runs the self-tuning-sigma affinity + normalized-
    Laplacian eigengap partition, and returns within-group index pairs in idx-space (exactly like
    cogated_links) so the downstream seed-link union and bundling are untouched.

    EXPERIMENTAL: the spectral core is validated on g5 (eigengap k in the EV-knee band) but this
    wiring is not yet integration-tested on a full pipeline run -- validate before trusting."""
    try:
        from . import graph_link as gl
    except ImportError:
        import graph_link as gl
    if method != "spectral":
        raise ValueError("graph linkage method must be 'spectral' (ev needs per-spike features, "
                         "available only at the fiber-intrachunk stage)")
    yc = y0[idx] - np.array([D[int(c)] for c in chunk[idx]])            # drift-correct to a common frame
    tc = np.array([fg.mutual_center(t) for t in frag["template"][idx]])[:, mask, :].reshape(len(idx), -1)

    def _z(a):
        a = np.asarray(a, float).reshape(len(idx), -1)
        return (a - a.mean(0)) / (a.std(0) + 1e-9)

    blocks = [_z(tc), _z(np.column_stack([frag["x0"][idx], yc, frag["z0"][idx]])), _z(logA[idx])]
    if offs is not None:
        blocks.append(_z(offs))
    F = np.column_stack(blocks)
    if len(F) < 4:
        return []
    A = gl.discriminative_affinity(F, knn=min(knn, len(F) - 1), self_tuning=True)
    lab = gl.spectral_partition(A)["labels"]
    raw = []
    for g in np.unique(lab):
        mem = np.flatnonzero(lab == g)
        raw += [(int(mem[t]), int(mem[t + 1])) for t in range(len(mem) - 1)]    # chain -> union-find group
    return raw


def link_session(frag, *, chunk_min=12.0, cos_thr=0.975, pos_thr=1.5, off_thr=1.0, warp_thr=None,
                 max_resid=0.08, min_n=20, min_snr=0.0, mask=None, gap=1,
                 drift=None, seed_links=None, refine_trajectory=False, traj_ext_min=0.0,
                 chunk_exclusive=True, cfiber_thr=None, cfiber_q=None, linkage="cogated", amp_gate=0.0,
                 align_lag=0, align_upsample=1, primary_amp_frac=0.0, tan_thr=None, tangents=None,
                 frag_times=None, ov_refrac=None, ov_thr=0.3, ov_min_exp=5.0, ov_censor=0,
                 bundle="chunkexcl", var_allow=None, var_scale=1.0, n_pc=12, verbose=True):
    """frag: dict of per-fragment arrays (clu,x0,y0,z0,A,template,t_mid[s],resid,one_flank,n,
    [offset],[snr]).  Returns dict(chunk, chunks, D, links, bundles, link_mask).

    drift      : optional {chunk_id: D_um} to use instead of the (composition-fragile)
                 density cross-correlation -- pass the fiber-intrachunk overlap-backbone
                 drift here for a true per-unit estimate.
    seed_links : optional list of (i,j) index pairs (full-array indices) to union into the
                 bundles before the fingerprint pass -- the overlap-backbone anchors.
    gap        : max chunk skip for the fingerprint pass (2 bridges single-chunk dropouts)."""
    if mask is None:
        mask = fl.MASK_FULL
    y0 = frag["y0"]; A = frag["A"]; logA = np.log(np.clip(A, 1, None))
    chunk = (np.asarray(frag["t_mid"], float) / 60.0 // chunk_min).astype(int)   # t_mid is seconds
    one_flank = frag.get("one_flank", np.zeros(len(y0), int))
    resid = frag.get("resid", np.zeros(len(y0)))
    linkable = ((one_flank == 0) & (y0 > 0) & (y0 < 140)
                & (resid < max_resid) & (frag["n"] >= min_n))
    if min_snr > 0 and "snr" in frag:
        linkable &= np.asarray(frag["snr"]) >= min_snr
    idx = np.flatnonzero(linkable)
    chunks = sorted(int(c) for c in np.unique(chunk[idx]))
    if drift is not None:
        D = {int(c): float(drift.get(int(c), 0.0)) for c in chunks}
    else:
        D = estimate_drift(y0[idx], logA[idx], frag["n"][idx].astype(float), chunk[idx], chunks)
    offs = frag["offset"][idx] if "offset" in frag else None
    # cfiber shape co-gate: compute affine-invariant descriptors over the linkable templates; self-
    # calibrate the veto threshold at LINK TIME from the overlap-backbone same-unit pairs (the cross-
    # chunk analog of intrachunk's within-fragment split-half null) -- NOT inherited from intrachunk,
    # whose null is per-chunk fragment-level.  Descriptors are computed fresh from the merged-unit
    # templates here, so they reflect the post-intrachunk clusters (no carried-forward descriptor).
    cfw = None
    if cfiber_thr is not None or cfiber_q is not None:
        cfw = _cfiber_win(frag["template"].shape[1], _cfiber_peak(frag["template"][idx]))
        if cfiber_thr is None:
            if seed_links:
                Sall = _cfiber_shapes(np.asarray(frag["template"]), cfw)
                d = [float(np.linalg.norm(Sall[int(i)] - Sall[int(j)])) for i, j in seed_links]
                if len(d) >= 5:
                    cfiber_thr = float(np.quantile(d, cfiber_q))
                    _log(f"cfiber co-gate self-calibrated: thr={cfiber_thr:.3f}")
                    _det("from", f"q={cfiber_q} of {len(d):,} backbone same-unit pairs")
                else:
                    _log(f"cfiber co-gate: only {len(d)} backbone pairs (<5), disabled "
                         "(pass --cfiber-thr to force one)")
            else:
                _log("cfiber co-gate: needs backbone pairs (--from-units) or --cfiber-thr, disabled")
    if linkage == "cogated":
        ft_sub = [frag_times[int(i)] for i in idx] if (frag_times is not None and ov_refrac is not None) else None
        raw = cogated_links(frag["x0"][idx], y0[idx], frag["z0"][idx], logA[idx], frag["template"][idx],
                            chunk[idx], chunks, D, mask, cos_thr=cos_thr, pos_thr=pos_thr,
                            off_thr=off_thr, warp_thr=warp_thr, offsets=offs, gap=gap,
                            cfiber_thr=cfiber_thr, cfiber_win=cfw, amp_gate=amp_gate,
                            align_lag=align_lag, align_upsample=align_upsample,
                            primary_amp_frac=primary_amp_frac, tan_thr=tan_thr,
                            tangents=(tangents[idx] if tangents is not None else None),
                            frag_times=ft_sub, ov_refrac=ov_refrac, ov_thr=ov_thr,
                            ov_min_exp=ov_min_exp, ov_censor=ov_censor)
    else:
        if ov_refrac is not None:
            _log("--overlap-refrac-ms only wired for 'cogated' linkage; ignored here")
        raw = _graph_links(linkage, frag, idx, y0, logA, chunk, D, mask, offs)
    nraw = len(raw)
    links = [(int(idx[i]), int(idx[j])) for i, j in raw]
    if seed_links is not None:
        links += [(int(i), int(j)) for i, j in seed_links]
    if chunk_exclusive:
        Tt = frag["template"]
        strength = [masked_cos(Tt[i], Tt[j], mask) for i, j in links]
        for k in range(nraw, len(links)):       # trusted overlap-backbone seeds win first
            strength[k] = np.inf
        if bundle == "varbound":
            P, w = _template_pc_stats(Tt, frag["n"], n_pc=n_pc)
            allow = var_allow
            if allow is None:
                allow = _selfcal_var_allow(links, P, w, logA, var_scale)
            elif var_scale != 1.0:
                allow = float(allow) * float(var_scale)
            bundles, n_blocked = bundles_variance_bounded(len(y0), links, chunk, strength, logA,
                                                          P, w, allow)
            if verbose:
                src = "self-cal" if var_allow is None else "budget"
                _log(f"varbound bundling: var_allow={allow:.4g}  ({src}, scale={var_scale})")
                _det("blocked", f"{n_blocked:,} over-spread merge(s)")
        else:
            bundles = bundles_chunk_exclusive(len(y0), links, chunk, strength)
    else:
        bundles = build_bundles(len(y0), links)
    traj_info = None
    if refine_trajectory:
        try:
            from . import fiber_trajectory as ftj
        except ImportError:
            import fiber_trajectory as ftj
        bundles, traj_info = ftj.refine_bundles(frag, bundles, chunk, chunk_min=chunk_min,
                                                  ext_min=traj_ext_min)
    return dict(chunk=chunk, chunks=chunks, D=D, links=links, bundles=bundles,
                link_mask=linkable, traj_info=traj_info)


def global_clu_map(frag_clu, bundles, src_ids, reserve=(0, 1)):
    """Map source cluster ids -> bundle ids.  Fragments in a multi-member bundle share an
    id; reserve ids (noise/MUA) pass through; non-localized source clusters keep a fresh
    singleton id.  Returns (per-spike new ids, n_clusters)."""
    old2new = {}; nid = max(reserve) + 1
    for b in bundles:
        for fi in b:
            old2new[int(frag_clu[fi])] = nid
        nid += 1
    out = np.empty(len(src_ids), np.int32); extra = {}
    for k, c in enumerate(src_ids):
        c = int(c)
        if c in reserve:
            out[k] = c
        elif c in old2new:
            out[k] = old2new[c]
        else:
            if c not in extra:
                extra[c] = nid; nid += 1
            out[k] = extra[c]
    return out, int(out.max()) + 1


def global_clu_map_units(members, bundles, src_ids, reserve=(0, 1)):
    """Like global_clu_map, but each fragment in a bundle is a per-chunk UNIT carrying a
    `members` list of source cluster ids.  All source clusters in a bundle share one id."""
    old2new = {}; nid = max(reserve) + 1
    for b in bundles:
        for ui in b:
            for c in members[ui]:
                old2new[int(c)] = nid
        nid += 1
    out = np.empty(len(src_ids), np.int32); extra = {}
    for k, c in enumerate(src_ids):
        c = int(c)
        if c in reserve:
            out[k] = c
        elif c in old2new:
            out[k] = old2new[c]
        else:
            if c not in extra:
                extra[c] = nid; nid += 1
            out[k] = extra[c]
    return out, int(out.max()) + 1


def main():
    ap = argparse.ArgumentParser(description="Link per-chunk fragments into tracked units "
                                             "(position fingerprint + A anchor + template co-gate).")
    sy.add_session_args(ap, channels=False, ntotal=False, nsamp=False, nchan=False, sr=False)
    ap.add_argument("--cpos-method", default="stderiv"); ap.add_argument("--cpos-stage", default="refine")
    ap.add_argument("--clu-method", default=None, help="source .clu method (default: mirror --cpos-method)")
    ap.add_argument("--clu-stage", default=None, help="source .clu stage (default: mirror --cpos-stage)")
    ap.add_argument("--chunk-minutes", "--chunk-min", type=float, default=12.0)
    ap.add_argument("--cos-thr", type=float, default=0.975)
    ap.add_argument("--pos-thr", type=float, default=1.5)
    ap.add_argument("--off-thr", type=float, default=1.0, help="inter-channel offset RMS co-gate (samples); <=0 disables")
    ap.add_argument("--amp-gate", type=float, default=0.0,
                    help="absolute log-amplitude gate (natural-log units): veto a cross-chunk link whose two "
                         "fragments differ in log-energy by more than this (0 = off, default). A is treated as "
                         "a drift-stable anchor, so a large energy jump between linked fragments is suspect; the "
                         "cosine/cfiber co-gates are amplitude-invariant and cannot catch it, and logA is "
                         "otherwise only one standardized term in the pos_thr fingerprint. This is an absolute, "
                         "un-pooled cap.")
    ap.add_argument("--align-lag", type=int, default=0,
                    help="sub-sample template re-registration half-window (native samples; 0 = off) applied before "
                         "the cosine gate. The integer mutual_center leaves a fractional-sample residual that drops "
                         "a true same-neuron cross-chunk cosine under threshold; re-registering recovers it (g5: "
                         "+25% of admitted links). Mirrors fiber-intrachunk's --align-lag.")
    ap.add_argument("--align-upsample", type=int, default=1,
                    help="cubic-spline upsampling factor for the --align-lag search (1 = native-rate).")
    ap.add_argument("--primary-amp-frac", type=float, default=0.0,
                    help="restrict the cosine gate to the channels BOTH fragments treat as primary (peak-to-peak "
                         ">= this fraction of the template's own max; 0 = off, full template). On an octrode the "
                         "near-threshold channels carry mostly noise and decorrelate true same-neuron pairs; the "
                         "intersection (g5 median 5 of 8 channels) lifts those links over threshold. ~0.3.")
    ap.add_argument("--tan-thr", type=float, default=None,
                    help="energy-tangent (microfiber) co-gate: require the cosine of the two fragments' energy-"
                         "direction tangents (high-energy minus low-energy template) >= this. A precision guard on "
                         "the recall lifted by --primary-amp-frac; needs a per-fragment 'tangent' array in the "
                         "cpos/units table (emitted by fiber-cpos/-intrachunk). None = off. ~0.5.")
    ap.add_argument("--linkage", choices=["cogated", "spectral"], default="cogated",
                    help="merge method: 'cogated' (default; per-pair mutual-NN position + offset + "
                         "cosine veto stack) or 'spectral' (global graph_link affinity + normalized-"
                         "Laplacian eigengap partition -- transitivity-aware, robust to the per-pair "
                         "gate miscalibration that leaves high-cosine blocks unmerged). EXPERIMENTAL.")
    ap.add_argument("--bundle", choices=["chunkexcl", "varbound"], default="chunkexcl",
                    help="cross-chunk bundling: 'chunkexcl' (default; cosine-ordered union-find with the "
                         "same-chunk-collision guard only) or 'varbound' (energy-seeded, variance-bounded "
                         "agglomeration -- orders high-energy fragments first and refuses a merge that "
                         "spreads a bundle's template variance past a single-neuron envelope, stopping the "
                         "single-linkage chaining where a high-cosine-but-distinct cross edge welds two "
                         "units that never share a chunk).")
    ap.add_argument("--var-allow", type=float, default=None,
                    help="varbound: explicit template PC-variance boundary (default: self-calibrate from "
                         "the high-energy backbone edges at link time).")
    ap.add_argument("--var-scale", type=float, default=1.0,
                    help="varbound: multiply the variance boundary to dial the operating point (>1 merges "
                         "more, <1 splits more); applies to both the self-calibrated and explicit boundary.")
    ap.add_argument("--n-pc", type=int, default=12,
                    help="varbound: number of template principal components defining the variance space.")
    ap.add_argument("--warp-thr", type=float, default=None,
                    help="spatio-temporal WARP continuity co-gate (Omlor-Giese group delay): require the "
                         "cross-channel correlation of two candidates' per-channel group-delay profiles >= this "
                         "to link them. A neuron's warp morphs CONTINUOUSLY with drift (g5: adjacent-chunk "
                         "change ~0.004) while different co-located cells anti-correlate (260x separation), so "
                         "this vetoes false links that share gross shape. None (default) = off; ~0.9 is safe on "
                         "the clean per-chunk unit templates the linker sees.")
    ap.add_argument("--max-gap", type=int, default=2, help="max chunk skip for the fingerprint pass (2 bridges single-chunk dropouts)")
    ap.add_argument("--max-resid", type=float, default=0.08)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--min-snr", type=float, default=0.0, help="gate linkable fragments on waveform SNR (needs snr in the cpos table; 0=off)")
    ap.add_argument("--from-units", default=None, help="link a fiber-intrachunk <...>.units.npz (per-chunk units) instead of raw cpos fragments")
    ap.add_argument("--refine-trajectory", action="store_true",
                    help="post-pass: fit per-bundle depth + PCA-feature trajectories, resolve "
                         "same-chunk-conflict merges, and attach units lying on a bundle's path")
    ap.add_argument("--allow-chunk-clash", action="store_true",
                    help="disable chunk-exclusive bundling (default OFF: a bundle may not hold two "
                         "same-chunk units; the exclusion vetoes provable chained over-merges).")
    ap.add_argument("--traj-ext-min", type=float, default=0.0,
                    help="minutes an attach may extend beyond a bundle's member time span "
                         "(0=interpolation only; ~chunk length allows extrapolation-based extension)")
    ap.add_argument("--cfiber-thr", type=float, default=None,
                    help="cfiber shape co-gate: veto a candidate whose affine-invariant cfiber shape "
                         "distance exceeds this (drift-invariant complement to the cosine gate). Fixed value.")
    ap.add_argument("--cfiber-q", type=float, default=None,
                    help="enable the cfiber co-gate with the threshold self-calibrated at link time to this "
                         "quantile of the overlap-backbone same-unit shape distances (e.g. 0.90; needs --from-units).")
    ap.add_argument("--out-stage", default=None, help="output .clu stage (default: <clu-stage>_linked)")
    ap.add_argument("--gt-clu", default=None, help="ground-truth .clu to score the clustering before vs after linking")
    ap.add_argument("--gt-res", default=None, help=".res for the ground truth (timestamp alignment if it covers a window)")
    ap.add_argument("--overlap-refrac-ms", type=float, default=0.0,
                    help="DEFAULT OFF. >0 enables a power-aware refractory veto on the chunk-OVERLAP region: "
                         "a candidate cross-chunk link is vetoed if, in the time window the two fragments share, "
                         "their spikes coincide at chance level (two neurons) rather than showing a refractory dip "
                         "(one neuron).  Censors zero-lag duplicate detections; abstains (keeps the link) when "
                         "underpowered, so it only ever removes well-supported wrong links.  Needs sr + .res; "
                         "abstains on sparse data (g5).  Only the default 'cogated' linkage is gated.")
    ap.add_argument("--overlap-refrac-thr", type=float, default=0.3,
                    help="coincidence ratio above which the overlap test reads 'two neurons' and vetoes (default 0.3)")
    ap.add_argument("--overlap-censor-ms", type=float, default=0.4,
                    help="zero-lag censor band for the overlap test -- removes the same-spike duplicate detections "
                         "that the overlapping chunks make of one neuron (default 0.4 ms)")
    ap.add_argument("--overlap-min-exp", type=float, default=5.0,
                    help="min expected overlap-window coincidences to have power; below this the test abstains (default 5)")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, require=())
    base = cfg["base"]; elec = a.group
    clu_method = a.clu_method if a.clu_method is not None else a.cpos_method
    clu_stage = a.clu_stage if a.clu_stage is not None else a.cpos_stage
    out_stage = a.out_stage if a.out_stage is not None else (f"{clu_stage}_linked" if clu_stage else "linked")

    _, src = nio.read_clu_at(base, elec, variant=clu_method, tag=clu_stage)

    # overlap-refractory gate (default off): build per-source-id spike trains from .res once
    ov_refrac = None; ov_censor = 0; times_by_id = None
    if a.overlap_refrac_ms and a.overlap_refrac_ms > 0:
        sr = cfg.get("sr")
        if not sr:
            raise SystemExit("[link] --overlap-refrac-ms needs the sampling rate, but none is in the session yaml")
        res = np.asarray(nio.read_res(base, elec))
        if len(res) != len(src):
            raise SystemExit(f"[link] .res ({len(res)}) and source .clu ({len(src)}) lengths differ; "
                             "cannot build per-fragment spike trains for the overlap gate")
        ov_refrac = cg.refrac_samples(a.overlap_refrac_ms, sr)
        ov_censor = cg.refrac_samples(a.overlap_censor_ms, sr)
        o = np.argsort(src, kind="stable"); ss = src[o]; rr = res[o]
        uval, ufirst = np.unique(ss, return_index=True)
        ends = np.r_[ufirst[1:], len(ss)]
        times_by_id = {int(uval[k]): np.sort(rr[ufirst[k]:ends[k]]) for k in range(len(uval))}
    _EMPTY = np.empty(0, np.int64)
    ov_kw = dict(ov_refrac=ov_refrac, ov_thr=a.overlap_refrac_thr, ov_min_exp=a.overlap_min_exp, ov_censor=ov_censor)
    bundle_kw = dict(bundle=a.bundle, var_allow=a.var_allow, var_scale=a.var_scale, n_pc=a.n_pc)

    if a.from_units:
        z = np.load(a.from_units, allow_pickle=True)
        frag = {k: z[k] for k in ("template", "offset", "x0", "y0", "z0", "A", "t_mid", "n")}
        seed = z["backbone"].tolist() if "backbone" in z.files and len(z["backbone"]) else None
        drift = (dict(zip(z["drift_chunks"].tolist(), z["drift_um"].tolist()))
                 if "drift_um" in z.files else None)
        ft = ([np.sort(np.concatenate([times_by_id.get(int(c), _EMPTY) for c in m])) if len(m) else _EMPTY
               for m in z["members"]] if times_by_id is not None else None)
        tangents = z["tangent"] if "tangent" in z.files else None
        R = link_session(frag, chunk_min=a.chunk_minutes, cos_thr=a.cos_thr, pos_thr=a.pos_thr, warp_thr=a.warp_thr,
                         off_thr=a.off_thr, max_resid=a.max_resid, min_n=a.min_n,
                         gap=a.max_gap, drift=drift, seed_links=seed, refine_trajectory=a.refine_trajectory, traj_ext_min=a.traj_ext_min,
                         chunk_exclusive=not a.allow_chunk_clash, cfiber_thr=a.cfiber_thr, cfiber_q=a.cfiber_q, linkage=a.linkage, amp_gate=a.amp_gate,
                         align_lag=a.align_lag, align_upsample=a.align_upsample, primary_amp_frac=a.primary_amp_frac, tan_thr=a.tan_thr, tangents=tangents,
                         frag_times=ft, **ov_kw, **bundle_kw)
        newids, ncl = global_clu_map_units(z["members"], R["bundles"], src)
    else:
        tbl = nio.session_path(base, "cpos", elec, variant=a.cpos_method, tag=a.cpos_stage) + ".clusters.npz"
        z = np.load(tbl)
        if "template" not in z.files:
            raise SystemExit(f"[link] {tbl} has no 'template' -- re-run fiber-cpos (>=0018) to emit templates")
        if "t_mid" not in z.files:
            raise SystemExit(f"[link] {tbl} has no 't_mid' -- re-run fiber-cpos to stamp time")
        frag = {k: z[k] for k in z.files if k != "cols"}
        ft = ([times_by_id.get(int(c), _EMPTY) for c in frag["clu"]] if times_by_id is not None else None)
        tangents = frag.get("tangent")
        R = link_session(frag, chunk_min=a.chunk_minutes, cos_thr=a.cos_thr, pos_thr=a.pos_thr, warp_thr=a.warp_thr,
                         off_thr=a.off_thr, max_resid=a.max_resid, min_n=a.min_n, min_snr=a.min_snr,
                         gap=a.max_gap, refine_trajectory=a.refine_trajectory, traj_ext_min=a.traj_ext_min,
                         chunk_exclusive=not a.allow_chunk_clash, cfiber_thr=a.cfiber_thr, cfiber_q=a.cfiber_q, linkage=a.linkage, amp_gate=a.amp_gate,
                         align_lag=a.align_lag, align_upsample=a.align_upsample, primary_amp_frac=a.primary_amp_frac, tan_thr=a.tan_thr, tangents=tangents,
                         frag_times=ft, **ov_kw, **bundle_kw)
        newids, ncl = global_clu_map(frag["clu"], R["bundles"], src)
    out_path = nio.session_path(base, "clu", elec, variant=clu_method, tag=out_stage)
    nio.write_clu_file(out_path, newids, n_clusters=ncl)

    multi = [b for b in R["bundles"] if len(set(R["chunk"][b])) >= 2]
    Dv = list(R["D"].values())
    _log(f"{int(R['link_mask'].sum()):,} linkable {'units' if a.from_units else 'fragments'} "
         f"over {len(R['chunks'])} chunks")
    _det("links", f"{len(R['links']):,} inter-chunk")
    _det("bundles", f"{len(multi):,} multi-chunk of {len(R['bundles']):,}")
    _det("drift", f"{min(Dv):.0f}..{max(Dv):.0f} um")
    if R.get("traj_info"):
        ti = R["traj_info"]
        _log(f"trajectory refine: conflicts {ti['conflicts_before']} → {ti['conflicts_after']}")
        _det("", f"attached {ti['attached']} · evicted {ti['evicted']} · "
                 f"depth tol {ti['depth_tol']:.1f} · feat tol {ti['feat_tol']:.2f}")
    _log(f"wrote {out_path}   ({ncl:,} units)")

    if a.gt_clu:                                                 # measure whether linking improved agreement
        _, gt = nio.read_clu_file(a.gt_clu)
        res = nio.read_res(base, elec)
        if gt.size == len(src):
            cb, ca, gl = src, newids, gt
        elif a.gt_res:
            gres = nio.read_res_file(a.gt_res)
            cb, gl, _ = fsc.align_by_res(src, res, gt, gres)
            ca, _, _ = fsc.align_by_res(newids, res, gt, gres)
        else:
            _log("--gt-clu length differs from .res; pass --gt-res to align by timestamp")
            return
        sb = fsc.score(cb, gl); sa = fsc.score(ca, gl)
        _log("ground-truth score (before → after linking)")
        _det("ARI", f"{sb['ari']:.4f} → {sa['ari']:.4f}")
        _det("prec", f"{sb['pairwise_precision']:.4f} → {sa['pairwise_precision']:.4f}")
        _det("recall", f"{sb['pairwise_recall']:.4f} → {sa['pairwise_recall']:.4f}")
        _det("GT split", f"{sb['n_gt_split']} → {sa['n_gt_split']}")
        _det("merged", f"{sb['n_cand_merged']} → {sa['n_cand_merged']}")


if __name__ == "__main__":
    main()
