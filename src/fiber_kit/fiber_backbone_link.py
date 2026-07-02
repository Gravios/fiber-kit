#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_backbone_link.py — link fiber-session fragments across chunks on the
#  INVARIANT BACKBONE channels, gated by the Omlor-Giese warp veto.
#
#  A defrag-family stage: it takes an over-split, chunk-local sort (the output of
#  fiber-session: .clu.<method>.<elec>.fiber_session) and CONCATENATES each
#  neuron's per-chunk fragments into one identity, writing a remapped .clu.  It is
#  the invariant-channel linker validated in tools/ci_overlap_chain.py, promoted to
#  a stage that produces output.
#
#  Match (identity, recall) — MEDIAN +/- z*sigma confidence-band overlap:
#    * per fragment template a per-sample band [median +/- z*sigma] on the backbone
#      channels (median is robust to the contamination co-located cells leak in;
#      +/-1 sigma is tight enough that co-located impostors fall below the overlap;
#      the literal SEM CI goes pencil-thin on big fragments and fails).
#    * backbone channels = the pair's SHARED primary channels (p2p >= prim_frac*max
#      in BOTH) unless pinned with --channels; the band on those channels captures
#      invariance (a stable channel has a tight band that overlaps consistently, a
#      drifting/physiology channel a wide one).
#    * window centred on the fragment's RMS-energy centroid, slid +/- slide samples
#      for the max-overlap lag; per (sample x channel) interval IoU counted at >= iou_thr.
#  Links are MUTUAL nearest-neighbour between adjacent chunks (bidirectional
#  agreement is the precision gate a one-directional chase lacks); connected
#  components are the chains (never transitive union-find of a long drift run).
#
#  Precision veto (reject improper mergers) — full Omlor-Giese warp gate on the
#  FULL footprint (reusing fiber_geometry): group-delay coherence (eq.11) AND
#  amplitude-profile correlation (eq.10) AND the single-channel incongruity sub-gate.
#  On the octrode the group-delay term is near-degenerate (units span few channels)
#  so warp_thr is relaxed; the amplitude-profile term is the effective veto.
#
#  Runs on STANDARD .spk (the axis curation/localization use, and the one group-delay
#  needs -- see the localization note in the handoff).  Standard spikes are loosely
#  stored-aligned (~1-2 sample jitter); each fragment is fl.realign'd before templating.
#
#  g5 ch33/34 (114 fragments, 35 curated parents; standard .spk): mutual-NN
#  multi-chain purity 0.934 -> 0.951 with the warp veto (overall 0.960 -> 0.972),
#  per-parent completeness unchanged at 0.950 -- it removes improper mergers only.
#
#  Knobs read FK_BBLINK_* (CLI > FK_* env > global fiber-kit.yaml > default), so the
#  operating point in fiber-kit-exp.yaml applies on a direct call.  Optional
#  --gt-clu/--gt-res scores purity+completeness against a curated sort (as fiber-defrag).
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np
from collections import defaultdict

try:
    from . import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy
    from . import config as cfgmod
except ImportError:
    import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy
    import config as cfgmod


# ── knob resolution: default <- global fiber-kit.yaml (FK_BBLINK_*) <- FK_* env <- CLI ──
_KNOBS = {
    "FK_BBLINK_Z": ("z", float, 1.0),
    "FK_BBLINK_WIN": ("win", int, 8),
    "FK_BBLINK_SLIDE": ("slide", int, 4),
    "FK_BBLINK_IOU_THR": ("iou_thr", float, 0.5),
    "FK_BBLINK_FLOOR": ("floor", float, 0.55),
    "FK_BBLINK_PRIM_FRAC": ("prim_frac", float, 0.30),
    "FK_BBLINK_WARP_THR": ("warp_thr", float, 0.5),
    "FK_BBLINK_AMP_THR": ("amp_thr", float, 0.85),
    "FK_BBLINK_RESID_THR": ("resid_thr", float, 1.0),
    "FK_BBLINK_MIN_FRAG": ("min_frag", int, 40),
    "FK_BBLINK_MAX_GAP": ("max_gap", int, 1),
    "FK_BBLINK_MIN_SNR_Q": ("min_snr_q", float, 0.0),
}


def _knob_default(name, typ, fallback, gcfg):
    """CLI default = global fiber-kit.yaml (FK_BBLINK_*) if set, else FK_* env, else fallback."""
    v = gcfg.get(name)
    if v in (None, ""):
        v = os.environ.get(name, "")
    if v in (None, ""):
        return fallback
    return typ(v)


def rms_center(mu):
    e = np.sqrt((mu ** 2).mean(1)); s = e.sum()
    return int(round(float((np.arange(mu.shape[0]) * e).sum() / s))) if s > 1e-12 else mu.shape[0] // 2


def build_frag(spk, idx, *, spk_cap, ref_sample, sr, rng):
    if idx.size > spk_cap:
        idx = rng.choice(idx, spk_cap, replace=False)
    w = fg.mutual_center_spikes(fg.denoise(fl.realign(np.asarray(spk[np.sort(idx)], float))), ref_sample=ref_sample)
    med = np.median(w, 0)
    sd = w.std(0, ddof=1) if len(w) > 1 else np.zeros_like(med)
    dom = int(np.argmax(np.ptp(med, 0)))
    snr = float(np.ptp(med[:, dom]) / (sd[:, dom].mean() + 1e-9))    # dominant-channel amplitude / spike-to-spike spread
    return dict(med=med, sd=sd, c=rms_center(med), gd=fg.group_delay_profile(med, sr=sr), dom=dom, snr=snr)


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


def link(frags, byc, *, pinned, prim_frac, z, win, slide, iou_thr, floor, max_gap, veto, warp_kw):
    """Conservative adjacent-chunk (+ one-chunk gap) MUTUAL-NN CI-overlap links with the warp veto.
    Within each chunk boundary the fragments are considered HIGH-SNR FIRST, so the cleanest clusters
    anchor their links before the noisier ones.  (The high-SNR RESTRICTION -- linking only the clean
    backbone this pass, deferring low-SNR/contaminated fragments -- is applied by the caller via the
    SNR floor.)  Union-find over the accepted links; returns a label per fragment index."""
    uf = list(range(len(frags)))

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
                if sc < floor:
                    continue
                back = [(score(i2, j), i2) for i2 in A]
                back = [(s, i2) for s, i2 in back if np.isfinite(s)]
                if not back or max(back)[1] != i:                 # mutual-NN
                    continue
                if veto and not warp_ok(frags[i], frags[j], **warp_kw):
                    continue
                uf[find(i)] = find(j); linked_fwd.add(i)
    return [find(i) for i in range(len(frags))]


def _score_gt(newclu, gt, keepmask):
    """Spike-weighted purity + per-parent completeness of the linked units vs a curated clu."""
    m = keepmask & (gt > 1)
    units = newclu[m]; par = gt[m]
    tot = pur = 0
    for u in np.unique(units):
        p = par[units == u]; vals, cnt = np.unique(p, return_counts=True)
        tot += p.size; pur += cnt.max()
    pp = defaultdict(int); best = defaultdict(int)
    for u in np.unique(units):
        p = par[units == u]
        for v, c in zip(*np.unique(p, return_counts=True)):
            pp[v] += c; best[v] = max(best[v], c)
    comp = float(np.mean([best[v] / pp[v] for v in pp])) if pp else float("nan")
    return (pur / tot if tot else float("nan")), comp


def main():
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(prog="fiber-backbone-link",
                                 description="Link fiber-session fragments across chunks on the invariant backbone "
                                             "(median+/-sigma CI-overlap) with the Omlor-Giese warp veto.")
    ap.add_argument("session"); ap.add_argument("elec", type=int)
    ap.add_argument("--clu-method", default="stderiv", help="fragment .clu feature space (before the group)")
    ap.add_argument("--clu-stage", "--variant", dest="clu_stage", default="fiber_session",
                    help="fragment .clu stage tag (the fiber-session output)")
    ap.add_argument("--in-clu", default=None, help="explicit fragment .clu path (overrides --clu-method/--clu-stage)")
    ap.add_argument("--spk-variant", default="standard", help="waveform axis for templates/warp (standard = curation axis)")
    ap.add_argument("--channels", default=None, help="pin backbone channels (global ids, e.g. 33,34); default = per-pair shared primary")
    ap.add_argument("--out-tag", default="backbone_linked", help="output .clu stage tag (single token)")
    ap.add_argument("--gt-clu", default=None, help="curated .clu to score purity+completeness against")
    ap.add_argument("--gt-res", default=None, help="reserved: .res for the GT (unused when GT shares the session res)")
    ap.add_argument("--spk-cap", type=int, default=600, help="spikes per fragment for the template")
    ap.add_argument("--chunk-min", type=float, default=None, help="chunk length (min); default from <session>.yaml or 12")
    ap.add_argument("--seed", type=int, default=0)
    for name, (dest, typ, fb) in _KNOBS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=_knob_default(name, typ, fb, gcfg),
                        help=f"{name} (default {_knob_default(name, typ, fb, gcfg)})")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cfg = sy.resolve_session_params(a.session, a.elec)
    base = cfg["base"]; elec = a.elec; NS, NC, PK, SR = cfg["nsamp"], cfg["nchan"], cfg["peak"], cfg["sr"]
    gch = list(cfg["channels"])
    pinned = np.array([gch.index(int(x)) for x in a.channels.split(",")]) if a.channels else None
    chunk_min = a.chunk_min if a.chunk_min else float(gcfg.get("FK_SESSION_CHUNK_MIN") or os.environ.get("FK_SESSION_CHUNK_MIN") or 12.0)

    res = nio.read_res_file(nio.session_path(base, "res", elec))
    if a.in_clu:
        _, fs = nio.read_clu_file(a.in_clu, n_spikes=res.size)
    else:
        _, fs = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=res.size)
    spk = nio.open_spk_file(nio.session_path(base, "spk", elec, variant=a.spk_variant), NS, NC)
    assert res.size == fs.size == spk.shape[0], "res/frag/spk length mismatch"
    chunk = (res.astype(np.float64) / SR / 60.0 // chunk_min).astype(int)

    ids = np.unique(fs[fs > 1])
    pinset = set(pinned.tolist()) if pinned is not None else None
    cand = []
    for u in ids:
        idx = np.flatnonzero(fs == u)
        if idx.size < a.min_frag:
            continue
        f = build_frag(spk, idx, spk_cap=a.spk_cap, ref_sample=PK, sr=SR, rng=rng)
        if pinset is not None and f["dom"] not in pinset:      # pinned mode: only fragments on those channels
            continue
        f["fsid"] = int(u); f["chunk"] = int(np.round(np.median(chunk[idx])))
        cand.append(f)
    # START WITH HIGH SNR: link only clusters at/above the SNR floor this pass; low-SNR ones are
    # left as singletons for the contamination/refinement phase.
    snrs = np.array([f["snr"] for f in cand]) if cand else np.array([])
    snr_thr = float(np.quantile(snrs, a.min_snr_q)) if (cand and a.min_snr_q > 0) else -np.inf
    frags = [f for f in cand if f["snr"] >= snr_thr]
    byc = defaultdict(list)
    for i, f in enumerate(frags):
        byc[f["chunk"]].append(i)
    if len(snrs):
        q = np.quantile(snrs, [0.25, 0.5, 0.75])
        print(f"[backbone-link] SNR quartiles {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} | "
              f"floor q={a.min_snr_q} -> SNR>={snr_thr if np.isfinite(snr_thr) else 0:.1f}")
    print(f"[backbone-link] {base} elec {elec} | {a.spk_variant} .spk | {len(frags)}/{len(cand)} clusters linked this pass "
          f"(>= {a.min_frag} spk) over {len(byc)} chunks"
          + (f" | channels pinned {a.channels}" if a.channels else " | per-pair shared-primary channels"))

    labels = link(frags, byc, pinned=pinned, prim_frac=a.prim_frac, z=a.z, win=a.win, slide=a.slide,
                  iou_thr=a.iou_thr, floor=a.floor, max_gap=a.max_gap, veto=True,
                  warp_kw=dict(warp_thr=a.warp_thr, amp_thr=a.amp_thr, resid_thr=a.resid_thr))
    # component -> new contiguous unit id (>=2); fragments below min_frag / unlinked keep their own id
    comp_units = {}
    nxt = 2
    fs_to_unit = {}
    for i, lab in enumerate(labels):
        if lab not in comp_units:
            comp_units[lab] = nxt; nxt += 1
        fs_to_unit[frags[i]["fsid"]] = comp_units[lab]
    # any fiber-session id we did NOT template (too small) keeps a fresh singleton id, preserving the sort
    for u in ids:
        if int(u) not in fs_to_unit:
            fs_to_unit[int(u)] = nxt; nxt += 1
    new = fs.copy().astype(np.int64)
    for u in ids:
        new[fs == u] = fs_to_unit[int(u)]

    n_chains = sum(1 for lab in set(labels) if list(labels).count(lab) > 1)
    nio.write_clu(base, elec, new, variant=a.clu_method, tag=a.out_tag)
    outp = nio.session_path(base, "clu", elec, variant=a.clu_method, tag=a.out_tag)
    print(f"[backbone-link] {len(ids)} fragments -> {len(set(new[new > 1]))} units "
          f"({n_chains} multi-fragment chains) | wrote {os.path.basename(outp)}")

    if a.gt_clu:
        _, gt = (nio.read_clu_file(a.gt_clu, n_spikes=res.size) if os.path.sep in a.gt_clu or a.gt_clu.endswith(".clu")
                 else nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.gt_clu, n_spikes=res.size))
        keep = np.isin(fs, ids)
        pur, comp = _score_gt(new, np.asarray(gt), keep)
        print(f"[backbone-link] vs GT: spike-weighted purity {pur:.3f} | per-parent completeness {comp:.3f}")


if __name__ == "__main__":
    main()
