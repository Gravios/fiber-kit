#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  tools/gt_chain_invariant.py — does chaining on INVARIANT channels beat chaining
#  on PRIMARY (amplitude) channels?  Validated against a curated .clu (GT).
#
#  Gravio's curation observation (transcript 2026-07-01-15-14): when merging he
#  trusts the channels that stay INVARIANT across the candidates and discounts the
#  ones that VARY -- "one or two channels should determine the linkages ... for the
#  immediate chunk and its neighbors", while the varying channels reflect ongoing
#  PHYSIOLOGY (most evident in interneurons).  fiber_chan_svd quantifies that
#  (per-channel across-cluster variance).  But piece_interneurons' chain rests on
#  `_pcos` = the PRIMARY (highest-amplitude) channels -- and §6.2 found primary-
#  channel cosine is the WORST co-located discriminator (co-located cells SHARE the
#  loud channels).  Primary != invariant: the loud channel can be the drifting /
#  physiology one; a quieter channel can be the stable identity carrier.  This tests,
#  on GT, whether an INVARIANCE-weighted cosine chains better than the primary one.
#
#  Leave-one-out chaining test (mirrors the real decision -- learn the bundle's
#  stable channels from its members, score the next candidate on them):
#    for each GT unit U with >=3 chunk-nodes, hold out node i:
#      - ref template + per-channel invariance from U's OTHER nodes (fiber_chan_svd)
#      - POSITIVE  = weighted_cos(ref, node_i)          (true continuation of U)
#      - NEGATIVES = weighted_cos(ref, V-node)          (co-located impostor: a
#                    different unit V sharing ref's primary channel)
#    AUC = P(positive > negative), pooled, per channel-weighting scheme:
#       full  | primary(amplitude) | invariant-weighted | invariant-mask
#  If invariant beats primary AND full, chaining should weight by invariance.
#
#  NOTE: computed on whatever .spk resolves.  On stderiv .spk this is stderiv-space
#  invariance; Gravio eyeballs the STANDARD waveform, so the raw .spk.standard makes
#  it match his curation view (localization/amplitude must never use stderiv, §10.2).
#  I/O via neuro_io; invariance via fiber_chan_svd.per_channel_svd.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import numpy as np

try:
    from fiber_kit import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, fiber_chan_svd as cs
except ImportError:
    import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, fiber_chan_svd as cs


def _auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    n1, n2 = pos.size, neg.size
    if n1 == 0 or n2 == 0:
        return float("nan"), n1, n2
    allv = np.concatenate([pos, neg]); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, float); ranks[order] = np.arange(1, allv.size + 1)
    s = allv[order]; i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2)), n1, n2


def wcos(a, b, w):
    """Per-channel-weighted cosine: scale each channel by sqrt(w_c), then cosine."""
    sw = np.sqrt(np.maximum(w, 0.0))[None, :]
    x = (a * sw).ravel(); y = (b * sw).ravel()
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 1e-9 and ny > 1e-9 else np.nan


def weights(ref, var_rel, prim_frac):
    """Four channel-weight schemes over an (nsamp,nch) ref template + per-channel
    invariance var_rel (from fiber_chan_svd: low = invariant)."""
    p2p = np.ptp(ref, axis=0)
    med = float(np.median(var_rel)) + 1e-9
    return {
        "full": np.ones(ref.shape[1]),
        "primary_amp": (p2p >= prim_frac * p2p.max()).astype(float),   # the current _pcos channels
        "invariant_wt": med / (var_rel + med),                          # soft: invariant -> ~1, varying -> <0.5
        "invariant_mask": (var_rel <= np.median(var_rel)).astype(float),  # the more-invariant half
    }


def build_nodes(clu, chunk, spk, units, *, min_chunk_spikes, spk_cap, ref_sample, rng):
    nodes = {}
    for u in units:
        su = (clu == u); cu = chunk[su]; idx_u = np.flatnonzero(su); lst = []
        for c in np.unique(cu):
            idx = idx_u[cu == c]
            if idx.size < min_chunk_spikes:
                continue
            if idx.size > spk_cap:
                idx = rng.choice(idx, spk_cap, replace=False)
            al = fg.mutual_center_spikes(fg.denoise(fl.realign(np.asarray(spk[np.sort(idx)], float))),
                                         ref_sample=ref_sample)
            t = np.median(al, 0)                         # (nsamp,nch), NOT globally normed (per-chan amp matters)
            lst.append(dict(c=int(c), t=t, prim=int(np.argmax(np.ptp(t, axis=0)))))
        if len(lst) >= 3:                                # need >=3 so leave-one-out has >=2 for invariance
            nodes[int(u)] = lst
    return nodes


def main():
    ap = argparse.ArgumentParser(description="Invariant- vs primary-channel chaining discrimination on GT.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--variant", default="stderiv")
    ap.add_argument("--clu-tag", default="")
    ap.add_argument("--nsamp", type=int, default=42)
    ap.add_argument("--nchan", type=int, default=8)
    ap.add_argument("--sr", type=float, default=32552.0)
    ap.add_argument("--chunk-min", type=float, default=12.0)
    ap.add_argument("--min-spikes", type=int, default=500)
    ap.add_argument("--min-chunk-spikes", type=int, default=80)
    ap.add_argument("--spk-cap", type=int, default=400)
    ap.add_argument("--prim-frac", type=float, default=0.30)
    ap.add_argument("--ref-sample", type=int, default=21)
    ap.add_argument("--max-neg-per-query", type=int, default=60)
    ap.add_argument("--reserve", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    res = nio.read_res_file(nio.session_path(a.session, "res", a.group, variant=a.variant))
    _, clu = nio.read_clu_at(a.session, a.group, variant=a.variant, tag=a.clu_tag)
    spk = nio.open_spk_file(nio.session_path(a.session, "spk", a.group, variant=a.variant), a.nsamp, a.nchan)
    assert res.size == clu.size == spk.shape[0], "res/clu/spk length mismatch"
    chunk = (res.astype(np.float64) / a.sr / 60.0 // a.chunk_min).astype(int)
    u, cnt = np.unique(clu, return_counts=True)
    keep = [int(x) for x, n in zip(u, cnt) if n >= a.min_spikes and x not in set(a.reserve)]
    print(f"[gt_chain_invariant] {res.size:,} spikes | space={a.variant} .spk | "
          f"{len(keep)} units >= {a.min_spikes} spikes")

    nodes = build_nodes(clu, chunk, spk, keep, min_chunk_spikes=a.min_chunk_spikes,
                        spk_cap=a.spk_cap, ref_sample=a.ref_sample, rng=rng)
    allnodes = [(uu, nd) for uu, lst in nodes.items() for nd in lst]
    by_prim = {}
    for uu, nd in allnodes:
        by_prim.setdefault(nd["prim"], []).append((uu, nd))
    print(f"[gt_chain_invariant] {len(nodes)} units with >=3 chunk-nodes | "
          f"{len(allnodes)} nodes over {len(by_prim)} primary channels")

    schemes = ["full", "primary_amp", "invariant_wt", "invariant_mask"]
    POS = {k: [] for k in schemes}
    NEG = {k: [] for k in schemes}
    overlap = []                                         # invariant-mask vs primary-mask channel agreement
    nq = 0
    for uu, lst in nodes.items():
        for i in range(len(lst)):
            others = np.array([lst[j]["t"] for j in range(len(lst)) if j != i])
            r = cs.per_channel_svd(others, n_comp=1)
            ref = r["grand"]; var_rel = r["var_rel"]
            W = weights(ref, var_rel, a.prim_frac)
            prim_ch = int(np.argmax(np.ptp(ref, axis=0)))
            overlap.append(float(np.mean((W["invariant_mask"] > 0) == (W["primary_amp"] > 0))))
            negs = [nd for (vv, nd) in by_prim.get(prim_ch, []) if vv != uu]
            if not negs:
                continue
            if len(negs) > a.max_neg_per_query:
                negs = [negs[k] for k in rng.choice(len(negs), a.max_neg_per_query, replace=False)]
            nq += 1
            for k in schemes:
                POS[k].append(wcos(ref, lst[i]["t"], W[k]))
                for nd in negs:
                    NEG[k].append(wcos(ref, nd["t"], W[k]))

    print(f"[gt_chain_invariant] {nq} leave-one-out queries | "
          f"{len(POS['full'])} continuations vs {len(NEG['full'])} co-located impostors")
    print(f"[gt_chain_invariant] invariant-mask vs primary-mask channel agreement: "
          f"{100*np.mean(overlap):.0f}%  (low => invariance picks DIFFERENT channels than amplitude)\n")

    rows = [(k, *_auc(POS[k], NEG[k]), float(np.nanmedian(POS[k])), float(np.nanmedian(NEG[k]))) for k in schemes]
    base = dict((k, au) for k, au, *_ in rows)["full"]
    prim = dict((k, au) for k, au, *_ in rows)["primary_amp"]
    best = max(r[1] for r in rows if np.isfinite(r[1]))
    print(f"{'channel weighting':<18}{'AUC':>8}{'med_cont':>10}{'med_impostor':>14}{'vs full':>10}{'vs primary':>12}")
    print("-" * 74)
    for k, au, _n1, _n2, mp, mn in sorted(rows, key=lambda r: -(r[1] if np.isfinite(r[1]) else -1)):
        star = "  <- best" if np.isfinite(au) and au == best else ""
        print(f"{k:<18}{au:>8.3f}{mp:>10.3f}{mn:>14.3f}{au-base:>+10.3f}{au-prim:>+12.3f}{star}")
    print("\nChaining should weight channels by INVARIANCE only if invariant_* clears BOTH full and "
          "primary_amp (the current _pcos metric).")


if __name__ == "__main__":
    main()
