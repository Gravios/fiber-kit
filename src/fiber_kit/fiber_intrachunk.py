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
DEFAULT_DEPTH_GATE = 35.0   # um; the energy ladder's depth spread (spatial fiber)
DEFAULT_MIN_N = 12          # fragments below this are too noisy to sign reliably


def _offset_rms(o1, o2):
    m = ~np.isnan(o1) & ~np.isnan(o2)
    return float(np.sqrt(np.nanmean((o1[m] - o2[m]) ** 2))) if m.sum() >= 2 else np.inf


def build_signatures(spkD, clu, t_mid_s, pos, *, chunk_min=12.0, min_n=DEFAULT_MIN_N,
                     reserve=(0, 1), sigma=fg.DEFAULT_SMOOTH_SIGMA):
    """Per-cluster stderiv signature for matching.

    spkD     : (nspk, nsamp, nchan) int16 stderiv waveforms (array or memmap).
    clu      : (nspk,) source cluster id per spike.
    t_mid_s  : (nspk,) spike time in seconds (from .res / sampling rate) — its
               per-cluster mean places the cluster in a chunk.
    pos      : {clu_id: (x0, y0, z0, A)} from the fiber-cpos cluster table.

    Returns a dict of per-cluster arrays keyed by cluster id order in `ids`:
      ids, template (nclu,nsamp,nchan, mutual-centred mean), offset (nclu,nchan),
      x0,y0,z0,A, chunk, t_mid, n.  Clusters in `reserve`, below min_n, or absent
      from `pos` are skipped (they stay singletons downstream)."""
    order = np.argsort(clu, kind="stable")
    cs = clu[order]
    uq, st = np.unique(cs, return_index=True)
    en = np.r_[st[1:], len(cs)]
    ids, T, O, X, Y, Z, A, CH, TM, NS = [], [], [], [], [], [], [], [], [], []
    for k, c in enumerate(uq):
        c = int(c)
        if c in reserve or c not in pos:
            continue
        idx = order[st[k]:en[k]]
        if len(idx) < min_n:
            continue
        al = fg.mutual_center_spikes(fg.denoise(fl.realign(spkD[idx].astype(float)), sigma))
        tmpl = al.mean(0)
        x0, y0, z0, amp = pos[c]
        tm = float(np.mean(t_mid_s[idx]))
        ids.append(c); T.append(tmpl); O.append(fg.interchannel_offsets(tmpl))
        X.append(x0); Y.append(y0); Z.append(z0); A.append(abs(amp))
        CH.append(int(tm / 60.0 // chunk_min)); TM.append(tm); NS.append(len(idx))
    return dict(ids=np.array(ids, int), template=np.array(T, np.float32),
                offset=np.array(O, np.float32), x0=np.array(X), y0=np.array(Y),
                z0=np.array(Z), A=np.array(A), chunk=np.array(CH, int),
                t_mid=np.array(TM), n=np.array(NS, int))


def group_intrachunk(sig, *, cos_thr=DEFAULT_COS_THR, off_thr=DEFAULT_OFF_THR,
                     depth_gate=DEFAULT_DEPTH_GATE):
    """Per-chunk complete-linkage clique on (cosine, offset, depth).  Returns a
    per-cluster integer label (dense, 0-based) — one label per per-chunk unit."""
    T = sig["template"].reshape(len(sig["ids"]), -1)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    off, Y, chunk = sig["offset"], sig["y0"], sig["chunk"]
    label = np.full(len(sig["ids"]), -1, int); nxt = 0
    for ch in np.unique(chunk):
        ix = np.flatnonzero(chunk == ch); n = len(ix)
        C = Tn[ix] @ Tn[ix].T
        edges = []
        for a in range(n):
            for b in range(a + 1, n):
                i, j = ix[a], ix[b]
                if abs(Y[i] - Y[j]) > depth_gate or C[a, b] < cos_thr:
                    continue
                o = _offset_rms(off[i], off[j])
                if o <= off_thr:
                    edges.append((C[a, b] - o, a, b))     # strongest agreement first
        edges.sort(reverse=True)
        par = list(range(n)); mem = {k: [k] for k in range(n)}
        def root(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        es = set((a, b) for _, a, b in edges) | set((b, a) for _, a, b in edges)
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
    chunks = sorted(set(int(c) for c in uC))
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
        lut[c] = c if c in reserve else (max(reserve) + 1 + fresh(("U", unit_of[c])) if c in unit_of
                                         else max(reserve) + 1 + fresh(("S", c)))
    out = np.array([lut[int(c)] for c in src_ids], np.int32)
    return out, int(out.max()) + 1


def main():
    ap = argparse.ArgumentParser(description="Collapse over-split fragments within each "
                                             "chunk into units (stderiv cosine + offset + depth).")
    ap.add_argument("session"); ap.add_argument("group", type=int)
    ap.add_argument("--cpos-method", default="stderiv")
    ap.add_argument("--cpos-stage", default="refine")
    ap.add_argument("--clu-method", default=None); ap.add_argument("--clu-stage", default=None)
    ap.add_argument("--chunk-minutes", type=float, default=12.0)
    ap.add_argument("--cos-thr", type=float, default=DEFAULT_COS_THR)
    ap.add_argument("--off-thr", type=float, default=DEFAULT_OFF_THR)
    ap.add_argument("--depth-gate", type=float, default=DEFAULT_DEPTH_GATE)
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    ap.add_argument("--boundary-minutes", type=float, default=3.0, help="half-window (min) of straddling spikes for the overlap backbone anchor (--emit-units)")
    ap.add_argument("--out-stage", default=None, help="output .clu stage (default: <clu-stage>.intrachunk)")
    ap.add_argument("--emit-units", action="store_true", help="also write a <...>.units.npz unit-signature table for fiber-link")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, require=("nChannels", "samplingRate"))
    base = cfg["base"]; elec = a.group; sr = float(cfg["samplingRate"])
    nsamp = int(cfg.get("nSamples", 32)); nch = int(cfg.get("nChannelsGroup", cfg.get("nChannels")))
    clu_method = a.clu_method if a.clu_method is not None else a.cpos_method
    clu_stage = a.clu_stage if a.clu_stage is not None else a.cpos_stage
    out_stage = a.out_stage if a.out_stage is not None else (f"{clu_stage}.intrachunk" if clu_stage else "intrachunk")

    _, src = nio.read_clu_at(base, elec, variant=clu_method, tag=clu_stage)
    res = nio.read_res(base, elec)
    spkD = nio.open_spkD(base, elec, nsamp, nch)
    tbl = nio.session_path(base, "cpos", elec, variant=a.cpos_method, tag=a.cpos_stage) + ".clusters.npz"
    z = np.load(tbl)
    pos = {int(c): (float(x), float(y), float(zz), float(A))
           for c, x, y, zz, A in zip(z["clu"], z["x0"], z["y0"], z["z0"], z["A"])}

    sig = build_signatures(spkD, src.astype(np.int64), res.astype(float) / sr, pos,
                           chunk_min=a.chunk_minutes, min_n=a.min_n)
    label = group_intrachunk(sig, cos_thr=a.cos_thr, off_thr=a.off_thr, depth_gate=a.depth_gate)
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
