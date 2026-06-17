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

try:
    from . import fiber_lib as fl, fiber_geometry as fg, neuro_io as nio, session_yaml as sy
except ImportError:
    import fiber_lib as fl, fiber_geometry as fg, neuro_io as nio, session_yaml as sy


def masked_cos(ta, tb, mask):
    a = ta[mask].ravel(); b = tb[mask].ravel()
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


def cogated_links(x0, y0, z0, logA, tmpl, chunk, chunks, D, mask, *, cos_thr=0.85,
                  pos_thr=1.5, off_thr=1.0, warp_thr=None, offsets=None, gap=1):
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
    sy_ = y0.std() + 1e-9; sx = x0.std() + 1e-9; sz = z0.std() + 1e-9; sa = logA.std() + 1e-9
    links = []; linked_fwd = set()
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
                    if masked_cos(tc[ai[v]], tc[bi[u]], mask) >= cos_thr and \
                            (off_thr <= 0 or _offset_rms(offsets[ai[v]], offsets[bi[u]]) <= off_thr) and \
                            (warp_thr is None or fg.warp_correlation(gd[ai[v]], gd[bi[u]]) >= warp_thr):
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


def link_session(frag, *, chunk_min=12.0, cos_thr=0.85, pos_thr=1.5, off_thr=1.0, warp_thr=None,
                 max_resid=0.08, min_n=20, min_snr=0.0, mask=None, gap=1,
                 drift=None, seed_links=None, refine_trajectory=False, traj_ext_min=0.0,
                 chunk_exclusive=True):
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
    raw = cogated_links(frag["x0"][idx], y0[idx], frag["z0"][idx], logA[idx], frag["template"][idx],
                        chunk[idx], chunks, D, mask, cos_thr=cos_thr, pos_thr=pos_thr,
                        off_thr=off_thr, warp_thr=warp_thr, offsets=offs, gap=gap)
    nraw = len(raw)
    links = [(int(idx[i]), int(idx[j])) for i, j in raw]
    if seed_links is not None:
        links += [(int(i), int(j)) for i, j in seed_links]
    if chunk_exclusive:
        Tt = frag["template"]
        strength = [masked_cos(Tt[i], Tt[j], mask) for i, j in links]
        for k in range(nraw, len(links)):       # trusted overlap-backbone seeds win first
            strength[k] = np.inf
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
    ap.add_argument("--cos-thr", type=float, default=0.85)
    ap.add_argument("--pos-thr", type=float, default=1.5)
    ap.add_argument("--off-thr", type=float, default=1.0, help="inter-channel offset RMS co-gate (samples); <=0 disables")
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
    ap.add_argument("--out-stage", default=None, help="output .clu stage (default: <clu-stage>_linked)")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, require=())
    base = cfg["base"]; elec = a.group
    clu_method = a.clu_method if a.clu_method is not None else a.cpos_method
    clu_stage = a.clu_stage if a.clu_stage is not None else a.cpos_stage
    out_stage = a.out_stage if a.out_stage is not None else (f"{clu_stage}_linked" if clu_stage else "linked")

    _, src = nio.read_clu_at(base, elec, variant=clu_method, tag=clu_stage)
    if a.from_units:
        z = np.load(a.from_units, allow_pickle=True)
        frag = {k: z[k] for k in ("template", "offset", "x0", "y0", "z0", "A", "t_mid", "n")}
        seed = z["backbone"].tolist() if "backbone" in z.files and len(z["backbone"]) else None
        drift = (dict(zip(z["drift_chunks"].tolist(), z["drift_um"].tolist()))
                 if "drift_um" in z.files else None)
        R = link_session(frag, chunk_min=a.chunk_minutes, cos_thr=a.cos_thr, pos_thr=a.pos_thr, warp_thr=a.warp_thr,
                         off_thr=a.off_thr, max_resid=a.max_resid, min_n=a.min_n,
                         gap=a.max_gap, drift=drift, seed_links=seed, refine_trajectory=a.refine_trajectory, traj_ext_min=a.traj_ext_min,
                         chunk_exclusive=not a.allow_chunk_clash)
        newids, ncl = global_clu_map_units(z["members"], R["bundles"], src)
    else:
        tbl = nio.session_path(base, "cpos", elec, variant=a.cpos_method, tag=a.cpos_stage) + ".clusters.npz"
        z = np.load(tbl)
        if "template" not in z.files:
            raise SystemExit(f"[link] {tbl} has no 'template' -- re-run fiber-cpos (>=0018) to emit templates")
        if "t_mid" not in z.files:
            raise SystemExit(f"[link] {tbl} has no 't_mid' -- re-run fiber-cpos to stamp time")
        frag = {k: z[k] for k in z.files if k != "cols"}
        R = link_session(frag, chunk_min=a.chunk_minutes, cos_thr=a.cos_thr, pos_thr=a.pos_thr, warp_thr=a.warp_thr,
                         off_thr=a.off_thr, max_resid=a.max_resid, min_n=a.min_n, min_snr=a.min_snr,
                         gap=a.max_gap, refine_trajectory=a.refine_trajectory, traj_ext_min=a.traj_ext_min,
                         chunk_exclusive=not a.allow_chunk_clash)
        newids, ncl = global_clu_map(frag["clu"], R["bundles"], src)
    out_path = nio.session_path(base, "clu", elec, variant=clu_method, tag=out_stage)
    nio.write_clu_file(out_path, newids, n_clusters=ncl)

    multi = [b for b in R["bundles"] if len(set(R["chunk"][b])) >= 2]
    Dv = list(R["D"].values())
    print(f"[link] {int(R['link_mask'].sum())} linkable {'units' if a.from_units else 'fragments'} over "
          f"{len(R['chunks'])} chunks -> {len(R['links'])} inter-chunk links, {len(multi)} multi-chunk "
          f"bundles (of {len(R['bundles'])}); drift {min(Dv):.0f}..{max(Dv):.0f}um")
    if R.get("traj_info"):
        ti = R["traj_info"]
        print(f"[link] trajectory refine: conflicts {ti['conflicts_before']}->{ti['conflicts_after']}, "
              f"attached {ti['attached']}, evicted {ti['evicted']} (depth tol {ti['depth_tol']:.1f}, feat tol {ti['feat_tol']:.2f})")
    print(f"[link] wrote {out_path}  ({ncl} units)")


if __name__ == "__main__":
    main()
