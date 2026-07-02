#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  tools/gt_discrim.py — co-located same/different discrimination on curated GT.
#
#  Settles the recurring "does any waveform feature earn a place in the linker /
#  intrachunk consolidation?" question against a CURATED .clu (ground truth),
#  removing the circularity of the earlier self-merge proxy (handoff §6.3/§7).
#
#  Framing = the LINKING-over-time problem.  Each curated unit's spikes are split
#  into time chunks; a per-(unit,chunk) median template + per-channel dispersion
#  band is built.  Two pair populations are compared:
#     same-cell  : (U, chunk_i) vs (U, chunk_j)      -- one neuron across time
#                  (this is what a linker must KEEP together under drift)
#     different  : (U, chunk_i) vs (V, chunk_j), U!=V, SAME primary channel
#                  (co-located distinct neurons -- the confusion a linker must
#                   NOT merge; the actual bottleneck, not far-apart pairs)
#  For each feature, AUC = P(same-cell pair scores MORE similar than a co-located
#  different pair).  1.0 = perfectly separable; 0.5 = useless.  A feature "earns a
#  place" only if it beats plain full-template cosine.
#
#  NOTE ON SPACE: features are computed on whatever .spk variant resolves.  On the
#  stderiv (separability) .spk this is a stderiv-space test; the standard-space
#  amplitude law needs the raw .spk.standard (localization must never use stderiv,
#  handoff §10.2).  ci_xcorr's amplitude-variability mechanism is likewise a
#  standard-space signal -- on stderiv it measures stderiv-band consistency.
#
#  Uses neuro_io for all I/O and fiber_geometry for every feature primitive.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import numpy as np

try:
    from fiber_kit import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl
except ImportError:
    import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl

HAS_CI = hasattr(fg, "ci_xcorr_score") and hasattr(fg, "dispersion_profile")


def _auc(pos, neg):
    """P(pos > neg) via the rank-sum (Mann-Whitney) statistic; NaN-safe."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    n1, n2 = pos.size, neg.size
    if n1 == 0 or n2 == 0:
        return float("nan"), n1, n2
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, float); ranks[order] = np.arange(1, allv.size + 1)
    # average tied ranks
    s = allv[order]; i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)), n1, n2


def _offset_rms(oa, ob):
    """NaN-aware RMS of per-channel interchannel-offset differences (>=2 valid)."""
    d = np.asarray(oa, float) - np.asarray(ob, float)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size >= 2 else np.nan


def _primary_mask(ta, tb, amp_frac):
    a = np.ptp(ta, axis=0); b = np.ptp(tb, axis=0)
    return (a >= amp_frac * a.max()) | (b >= amp_frac * b.max())


def build_nodes(clu, chunk, spk, units, *, min_chunk_spikes, spk_cap, sr,
                band, amp_frac, ref_sample, rng):
    """One node per (unit, chunk) with enough spikes: aligned median template
    (unit-norm), dispersion band, primary channel, offsets, group delay."""
    nodes = []
    for u in units:
        su = (clu == u)
        cu = chunk[su]; idx_u = np.flatnonzero(su)
        for c in np.unique(cu):
            idx = idx_u[cu == c]
            if idx.size < min_chunk_spikes:
                continue
            if idx.size > spk_cap:
                idx = rng.choice(idx, spk_cap, replace=False)
            W = np.asarray(spk[np.sort(idx)], float)
            al = fg.mutual_center_spikes(fg.denoise(fl.realign(W)), ref_sample=ref_sample)
            t = np.median(al, 0)
            t = t / (np.linalg.norm(t) + 1e-9)
            disp = fg.dispersion_profile(al, aligned=True) if HAS_CI else None
            nodes.append(dict(
                u=int(u), c=int(c), t=t, disp=disp,
                prim=int(np.argmax(np.ptp(t, axis=0))),
                off=fg.interchannel_offsets(t, amp_frac=amp_frac),
                gd=fg.group_delay_profile(t, sr=sr, band=band, amp_frac=amp_frac),
            ))
    return nodes


def score_pair(a, b, amp_frac):
    ta, tb = a["t"], b["t"]
    m = _primary_mask(ta, tb, amp_frac)
    fa, fb = ta.ravel(), tb.ravel()
    full = float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-9))
    pa, pb = ta[:, m].ravel(), tb[:, m].ravel()
    prim = float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-9)) if m.any() else np.nan
    out = dict(
        full_cos=full,
        primary_cos=prim,
        amp_profile=fg.amp_profile_correlation(ta, tb),
        warp=fg.warp_correlation(a["gd"], b["gd"]),
        neg_offset_rms=-_offset_rms(a["off"], b["off"]),
    )
    if HAS_CI and a["disp"] is not None and b["disp"] is not None:
        out["ci_xcorr"] = fg.ci_xcorr_score(a["disp"], b["disp"], template=0.5 * (ta + tb),
                                            amp_frac=amp_frac)
    return out


def main():
    ap = argparse.ArgumentParser(description="Co-located same/different discrimination on curated GT.")
    ap.add_argument("--session", required=True, help="session base path (…/<base>)")
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--variant", default="stderiv")
    ap.add_argument("--clu-tag", default="", help="curated .clu stage tag (e.g. fiber_session_curated)")
    ap.add_argument("--nsamp", type=int, default=42)
    ap.add_argument("--nchan", type=int, default=8)
    ap.add_argument("--sr", type=float, default=32552.0)
    ap.add_argument("--chunk-min", type=float, default=12.0)
    ap.add_argument("--min-spikes", type=int, default=300, help="min TOTAL spikes for a unit")
    ap.add_argument("--min-chunk-spikes", type=int, default=40, help="min spikes for a (unit,chunk) node")
    ap.add_argument("--spk-cap", type=int, default=400, help="subsample cap per node (speed)")
    ap.add_argument("--amp-frac", type=float, default=0.3)
    ap.add_argument("--band", type=float, nargs=2, default=(300.0, 9000.0))
    ap.add_argument("--ref-sample", type=int, default=21)
    ap.add_argument("--max-diff-pairs", type=int, default=20000)
    ap.add_argument("--reserve", type=int, nargs="*", default=[0, 1], help="labels to exclude (noise/MUA)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    res = nio.read_res_file(nio.session_path(a.session, "res", a.group, variant=a.variant))
    _, clu = nio.read_clu_at(a.session, a.group, variant=a.variant, tag=a.clu_tag)
    spk = nio.open_spk_file(nio.session_path(a.session, "spk", a.group, variant=a.variant),
                            a.nsamp, a.nchan)
    assert res.size == clu.size == spk.shape[0], "res/clu/spk length mismatch"
    chunk = (res.astype(np.float64) / a.sr / 60.0 // a.chunk_min).astype(int)

    u, cnt = np.unique(clu, return_counts=True)
    keep = [int(x) for x, n in zip(u, cnt) if n >= a.min_spikes and x not in set(a.reserve)]
    print(f"[gt_discrim] {res.size:,} spikes | {int(chunk.max()) + 1} chunks of {a.chunk_min:g} min | "
          f"{len(keep)} units >= {a.min_spikes} spikes (of {u.size})")
    print(f"[gt_discrim] space = {a.variant} .spk  (ci_xcorr {'ON' if HAS_CI else 'absent'})")

    nodes = build_nodes(clu, chunk, spk, keep, min_chunk_spikes=a.min_chunk_spikes,
                        spk_cap=a.spk_cap, sr=a.sr, band=tuple(a.band),
                        amp_frac=a.amp_frac, ref_sample=a.ref_sample, rng=rng)
    print(f"[gt_discrim] {len(nodes)} (unit,chunk) nodes")

    # same-cell pairs: nodes of one unit across chunks
    by_u = {}
    for i, nd in enumerate(nodes):
        by_u.setdefault(nd["u"], []).append(i)
    same = [(g[x], g[y]) for g in by_u.values() for x in range(len(g)) for y in range(x + 1, len(g))]

    # co-located different pairs: different units sharing a primary channel
    by_prim = {}
    for i, nd in enumerate(nodes):
        by_prim.setdefault(nd["prim"], []).append(i)
    diff = []
    for g in by_prim.values():
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                if nodes[g[x]]["u"] != nodes[g[y]]["u"]:
                    diff.append((g[x], g[y]))
    rng.shuffle(diff)
    diff = diff[:a.max_diff_pairs]
    print(f"[gt_discrim] {len(same)} same-cell pairs | {len(diff)} co-located different pairs\n")

    S = [score_pair(nodes[i], nodes[j], a.amp_frac) for (i, j) in same]
    D = [score_pair(nodes[i], nodes[j], a.amp_frac) for (i, j) in diff]
    order = ["full_cos", "primary_cos", "amp_profile", "warp", "neg_offset_rms"] + (["ci_xcorr"] if HAS_CI else [])

    def table(title, sub, cond=None):
        s, d = S, D
        if cond is not None:                                   # restrict to LOOK-ALIKE pairs
            s = [r for r in S if r["full_cos"] >= cond]
            d = [r for r in D if r["full_cos"] >= cond]
        print(f"\n{title}  [{len(s)} same, {len(d)} diff]{sub}")
        print(f"{'feature':<16}{'AUC':>8}{'median_same':>14}{'median_diff':>14}{'vs full_cos':>13}")
        print("-" * 65)
        rows = [(k, *_auc([r.get(k, np.nan) for r in s], [r.get(k, np.nan) for r in d]),
                 float(np.nanmedian([r.get(k, np.nan) for r in s])),
                 float(np.nanmedian([r.get(k, np.nan) for r in d]))) for k in order]
        base = dict((k, a_) for k, a_, *_ in rows).get("full_cos", np.nan)
        best = max((r[1] for r in rows if np.isfinite(r[1])), default=np.nan)
        for k, auc, _n1, _n2, ms, md in sorted(rows, key=lambda r: -(r[1] if np.isfinite(r[1]) else -1)):
            tag = "" if k == "full_cos" else f"{auc - base:+.3f}"
            star = "  <- best" if np.isfinite(auc) and auc == best else ""
            print(f"{k:<16}{auc:>8.3f}{ms:>14.3f}{md:>14.3f}{tag:>13}{star}")

    table("UNCONDITIONAL (all co-located different pairs)",
          " -- easy: most co-located pairs are NOT look-alikes")
    for thr in (0.90, 0.95):
        table(f"CONDITIONED full_cos >= {thr} (the genuinely confusable look-alikes -- THE bottleneck)",
              " -- restricted to look-alikes; does ANY feature beat full_cos here?",
              cond=thr)
    print("\nA feature earns a place in the linker only if it clears full_cos in the CONDITIONED "
          "table (the look-alike regime is where the linker actually errs).")


if __name__ == "__main__":
    main()
