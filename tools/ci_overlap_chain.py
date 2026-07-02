#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  tools/ci_overlap_chain.py — chain fiber-session fragments across chunks on the
#  invariant backbone channels, gated by the Omlor-Giese WARP veto.  Validated
#  against the curated .clu.
#
#  Working algorithm (validated on g5 ch33/34, STANDARD .spk):
#    1. per fiber-session fragment (chunk-local, pure to one curated parent) a
#       template band = MEDIAN +/- z*sigma per sample on the invariant channels
#       (median is robust to the contamination co-located cells leak in; +/-1 sigma
#       is tight enough that co-located impostors fall below the overlap).
#    2. window CENTERED on the fragment's RMS-energy centroid, SLID +/- slide samples
#       for the max-overlap lag; per (sample x channel) interval IoU counted when
#       >= iou_thr (0.5).  score = mean IoU.
#    3. link adjacent-chunk fragments by MUTUAL nearest-neighbour above a floor (B is
#       A's best forward AND A is B's best backward -- the bidirectional agreement is
#       the precision gate a one-directional greedy chase lacks).
#    4. WARP VETO on each surviving link (Omlor-Giese, on the FULL footprint): reject
#       unless group-delay coherence (eq.11) AND amplitude-profile corr (eq.10) AND the
#       single-channel incongruity sub-gate all pass.  On the octrode the group-delay
#       term is near-degenerate (~AUC 0.56, units span few channels) so its threshold
#       is relaxed; the amplitude-profile term (~AUC 0.90) is the effective veto.
#    Connected components are the chains.
#
#  g5 ch33/34 result: median+/-1sigma + mutual-NN gives multi-chain purity 0.934; the
#  warp veto lifts it to 0.951 (overall 0.960 -> 0.972), removing improper mergers at a
#  small recall cost.  Top-1 true-next-fragment among ~7 co-located competitors ~0.95.
#
#  Runs on STANDARD .spk (the axis curation/localization use, and group-delay wants raw
#  templates -- handoff §10.2).  Standard spikes are loosely stored-aligned (~1-2 sample
#  jitter); fl.realign re-centres each fragment before templating.  I/O via neuro_io;
#  warp primitives reused from fiber_geometry.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import numpy as np
from collections import defaultdict

try:
    from fiber_kit import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy
except ImportError:
    import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy


def rms_center(mu):
    e = np.sqrt((mu ** 2).mean(1)); s = e.sum()
    return int(round(float((np.arange(mu.shape[0]) * e).sum() / s))) if s > 1e-12 else mu.shape[0] // 2


def build_frag(spk, idx, *, spk_cap, ref_sample, sr, rng):
    if idx.size > spk_cap:
        idx = rng.choice(idx, spk_cap, replace=False)
    w = fg.mutual_center_spikes(fg.denoise(fl.realign(np.asarray(spk[np.sort(idx)], float))), ref_sample=ref_sample)
    med = np.median(w, 0)
    sd = w.std(0, ddof=1) if len(w) > 1 else np.zeros_like(med)
    return dict(med=med, sd=sd, c=rms_center(med), gd=fg.group_delay_profile(med, sr=sr),
                dom=int(np.argmax(np.ptp(med, 0))))


def _win(lo, hi, c, win, ns):
    s, e = c - win, c + win
    return (lo[s:e + 1], hi[s:e + 1]) if (s >= 0 and e < ns) else None


def ci_overlap(A, B, chans, *, z, win, slide, iou_thr):
    """Best-lag median+/-z*sigma band overlap on `chans`. Returns (frac_cells>=thr, mean_iou)."""
    hA = z * A["sd"]; hB = z * B["sd"]; ns = A["med"].shape[0]
    wA = _win(A["med"] - hA, A["med"] + hA, A["c"], win, ns)
    if wA is None:
        return np.nan, np.nan
    aL, aH = wA[0][:, chans], wA[1][:, chans]
    best = None
    for L in range(-slide, slide + 1):
        wB = _win(B["med"] - hB, B["med"] + hB, B["c"] + L, win, ns)
        if wB is None:
            continue
        bL, bH = wB[0][:, chans], wB[1][:, chans]
        inter = np.clip(np.minimum(aH, bH) - np.maximum(aL, bL), 0, None)
        union = np.clip(np.maximum(aH, bH) - np.minimum(aL, bL), 1e-12, None)
        iou = inter / union; miou = float(iou.mean())
        if best is None or miou > best[1]:
            best = (float((iou >= iou_thr).mean()), miou)
    return best if best else (np.nan, np.nan)


def warp_ok(A, B, *, warp_thr, amp_thr, resid_thr):
    """Full Omlor-Giese gate: group-delay coherence (eq.11) AND amplitude profile (eq.10)
    AND single-channel incongruity sub-gate.  Same composition as fiber_link._warp_gate."""
    if warp_thr is not None and fg.warp_correlation(A["gd"], B["gd"]) < warp_thr:
        return False
    if amp_thr is not None and fg.amp_profile_correlation(A["med"], B["med"]) < amp_thr:
        return False
    if resid_thr is not None and fg.warp_channel_incongruity(A["gd"], B["gd"]) > resid_thr:
        return False
    return True


def _auc(pos, neg):
    pos = np.asarray(pos, float); pos = pos[np.isfinite(pos)]
    neg = np.asarray(neg, float); neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort")
    r = np.empty(a.size); r[o] = np.arange(1, a.size + 1)
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def chain(frags, byc, S, *, floor, veto, warp_kw):
    uf = list(range(len(frags)))

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]; x = uf[x]
        return x
    links = 0
    for k in sorted(byc):
        A = byc.get(k, []); Bn = byc.get(k + 1, [])
        for i in A:
            cand = [(S[(i, j)], j) for j in Bn if np.isfinite(S[(i, j)])]
            if not cand:
                continue
            sc, j = max(cand)
            if sc < floor:
                continue
            back = [(S[(i2, j)], i2) for i2 in A if np.isfinite(S[(i2, j)])]
            if not back or max(back)[1] != i:              # mutual-NN
                continue
            if veto and not warp_ok(frags[i], frags[j], **warp_kw):
                continue
            uf[find(i)] = find(j); links += 1
    comp = defaultdict(list)
    for i in range(len(frags)):
        comp[find(i)].append(i)
    return links, [c for c in comp.values()]


def _purity(chains, frags, multi_only=False):
    tot = pur = 0
    for c in chains:
        if multi_only and len(c) < 2:
            continue
        pc = defaultdict(int)
        for i in c:
            pc[frags[i]["parent"]] += frags[i]["size"]
        tot += sum(pc.values()); pur += max(pc.values())
    return pur / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Invariant-channel CI-overlap chaining with the Omlor-Giese warp veto, on GT.")
    ap.add_argument("--session", required=True); ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--variant", default="standard", help="spk axis (standard = curation/localization axis; warp wants raw)")
    ap.add_argument("--frag-tag", default="fiber_session"); ap.add_argument("--gt-tag", default="fiber_session_curated")
    ap.add_argument("--channels", default="33,34", help="invariant backbone channels (global ids)")
    ap.add_argument("--nsamp", type=int, default=42); ap.add_argument("--nchan", type=int, default=8)
    ap.add_argument("--sr", type=float, default=32552.0); ap.add_argument("--chunk-min", type=float, default=12.0)
    ap.add_argument("--z", type=float, default=1.0, help="band half-width in sigma (median +/- z*sigma)")
    ap.add_argument("--win", type=int, default=8); ap.add_argument("--slide", type=int, default=4)
    ap.add_argument("--iou-thr", type=float, default=0.5); ap.add_argument("--floor", type=float, default=0.55)
    ap.add_argument("--warp-thr", type=float, default=0.5, help="group-delay coherence (near-degenerate on octrode -> relaxed)")
    ap.add_argument("--amp-thr", type=float, default=0.85, help="amplitude-profile correlation (the effective veto here)")
    ap.add_argument("--resid-thr", type=float, default=1.0, help="single-channel warp incongruity ceiling (samples)")
    ap.add_argument("--min-frag", type=int, default=40); ap.add_argument("--spk-cap", type=int, default=600)
    ap.add_argument("--ref-sample", type=int, default=21); ap.add_argument("--max-neg", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    warp_kw = dict(warp_thr=a.warp_thr, amp_thr=a.amp_thr, resid_thr=a.resid_thr)

    cfg = sy.resolve_session_params(a.session, a.group); gch = list(cfg["channels"])
    tgt = [int(x) for x in a.channels.split(",")]; chans = [gch.index(t) for t in tgt]
    res = nio.read_res_file(nio.session_path(a.session, "res", a.group, variant="stderiv"))
    _, fs = nio.read_clu_at(a.session, a.group, variant="stderiv", tag=a.frag_tag)
    _, gt = nio.read_clu_at(a.session, a.group, variant="stderiv", tag=a.gt_tag)
    spk = nio.open_spk_file(nio.session_path(a.session, "spk", a.group, variant=a.variant), a.nsamp, a.nchan)
    assert res.size == fs.size == gt.size == spk.shape[0], "res/frag/gt/spk length mismatch"
    chunk = (res.astype(np.float64) / a.sr / 60.0 // a.chunk_min).astype(int)

    frags = []
    for u in np.unique(fs[fs > 1]):
        idx = np.flatnonzero(fs == u)
        if idx.size < a.min_frag:
            continue
        f = build_frag(spk, idx, spk_cap=a.spk_cap, ref_sample=a.ref_sample, sr=a.sr, rng=rng)
        if f["dom"] not in chans:
            continue
        vals, c = np.unique(gt[idx], return_counts=True)
        f.update(parent=int(vals[c.argmax()]), chunk=int(np.round(np.median(chunk[idx]))), size=int(idx.size))
        frags.append(f)
    print(f"[ci_overlap] {a.variant} .spk | ch{tgt} | {len(frags)} fragments, {len({f['parent'] for f in frags})} curated parents")
    if len(frags) < 4:
        raise SystemExit("[ci_overlap] too few target-channel fragments; lower --min-frag")

    byc = defaultdict(list)
    for i, f in enumerate(frags):
        byc[f["chunk"]].append(i)
    S = {}
    for k in sorted(byc):
        for i in byc.get(k, []):
            for j in byc.get(k + 1, []):
                S[(i, j)] = ci_overlap(frags[i], frags[j], chans, z=a.z, win=a.win, slide=a.slide, iou_thr=a.iou_thr)[1]

    # pairwise diagnostic: which signal separates same-parent adjacent from co-located impostor
    good = [(i, j) for k in byc for i in byc[k] for j in byc.get(k + 1, []) if frags[i]["parent"] == frags[j]["parent"]]
    imp = [(frags[i], frags[j]) for i in range(len(frags)) for j in range(i + 1, len(frags)) if frags[i]["parent"] != frags[j]["parent"]]
    if len(imp) > a.max_neg:
        imp = [imp[k] for k in rng.choice(len(imp), a.max_neg, replace=False)]
    gci = [S[(i, j)] for i, j in good]; nci = [ci_overlap(x, y, chans, z=a.z, win=a.win, slide=a.slide, iou_thr=a.iou_thr)[1] for x, y in imp]
    gap = [fg.amp_profile_correlation(frags[i]["med"], frags[j]["med"]) for i, j in good]
    nap = [fg.amp_profile_correlation(x["med"], y["med"]) for x, y in imp]
    print(f"[diagnostic] same-parent adj vs co-located impostor  |  CI-overlap AUC {_auc(gci, nci):.3f}  "
          f"amp-profile(eq10) AUC {_auc(gap, nap):.3f}")

    print("\nmutual-NN chaining, WITHOUT vs WITH the Omlor-Giese warp veto:")
    for veto in (False, True):
        lk, chains = chain(frags, byc, S, floor=a.floor, veto=veto, warp_kw=warp_kw)
        multi = [c for c in chains if len(c) > 1]
        # per-parent completeness (largest-chain spike fraction)
        pp = defaultdict(int); best_in = defaultdict(int)
        for c in chains:
            pc = defaultdict(int)
            for i in c:
                pc[frags[i]["parent"]] += frags[i]["size"]
            for p, n in pc.items():
                pp[p] += n; best_in[p] = max(best_in[p], n)
        comp = float(np.mean([best_in[p] / pp[p] for p in pp]))
        tag = "warp-vetoed" if veto else "raw       "
        print(f"  {tag}: {lk} links, {len(multi)} multi-frag chains | overall purity {_purity(chains, frags):.3f} | "
              f"multi-chain purity {_purity(chains, frags, multi_only=True):.3f} | mean completeness {comp:.3f}")
    print(f"\n(warp veto = group-delay corr>={a.warp_thr} AND amp-profile>={a.amp_thr} AND incongruity<={a.resid_thr}; "
          "group-delay is relaxed because it is near-degenerate on the octrode -- amp-profile is the effective term.)")


if __name__ == "__main__":
    main()
