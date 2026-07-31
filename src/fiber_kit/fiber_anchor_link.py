#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_anchor_link.py — link fiber-session fragments across chunks, SEEDED by
#  the overlap anchors and gated on a null calibrated from the session itself.
#
#  A defrag-family stage, sibling to fiber-backbone-link.  It takes the over-split
#  chunk-local sort (.clu.<method>.<elec>.fiber_session) and concatenates each
#  neuron's per-chunk fragments into one identity, writing .clu + the .clc/.clp
#  hierarchy.  What differs from backbone-link is where the thresholds come from
#  and the order the gates run in:
#
#  SEED — fiber-session's overlap anchors (.fibers `drift_anchor_pairs`, columns
#    (transition, atom_a, atom_b) in .clc labels) are links established from the
#    SAME PHYSICAL SPIKES in the chunk overlap, so they are correct by construction
#    and cost nothing to trust.  They are applied before any scored link, so a
#    scored candidate can never claim an atom an anchor needs.  They also serve as
#    the POSITIVE class for calibration below.
#
#  MATCH — flc.ci_overlap (energy-scaled median+/-z*sigma band overlap on the
#    pair's shared primary channels) times the template cosine.
#
#  CALIBRATION — the threshold is not a constant.  Two fragments in the SAME chunk
#    are different cells by construction (the sorter separated them), so the
#    within-chunk score distribution is a per-session NEGATIVE class; the anchors
#    are the positives.  The floor is set at the requested false-positive rate
#    against that null.  This matters: a fixed floor tuned elsewhere can land at
#    the MEDIAN of a new session's different-cell distribution and gate nothing.
#
#  GATE ORDER — gates are pair properties, so they are applied to the candidate
#    POOL before the mutual-nearest-neighbour selection.  Selecting first and
#    gating after silently discards a cell whose top-scoring partner fails a gate,
#    never considering its valid runner-up.
#
#  RATE GATE — |log2| change in a fragment's RELATIVE firing rate (its share of its
#    chunk's spikes) across the link.  Independent of waveform, and the only gate
#    that catches the dominant failure mode: a small fragment welded onto the tail
#    of a large cell, which passes every shape test because its template is right.
#
#  WARP VETO — applied only when BOTH fragments clear the high-rate cut.  The
#    Omlor-Giese group-delay term needs >=3 channels with finite delay in both
#    profiles; on an octrode, low-rate fragments rarely have them, and
#    fg.warp_correlation returns 0.0 for "unmeasurable", which a threshold reads as
#    "incoherent".  Restricted to high-rate pairs the veto behaves as documented
#    (it removes improper mergers only); applied to all pairs it becomes a
#    near-blanket reject.  warp_thr defaults to OFF for that reason -- set it when
#    the geometry actually supports a group-delay estimate.
#
#  ROUNDS — agglomerative over CELLS, matching at each cell's boundary fragments
#    (nearest in time, so a chain's template is never smeared by pooling), with the
#    permitted chunk gap relaxed as the chains establish.  A cell may never hold two
#    co-temporal fragments.
#
#  Runs on STANDARD .spk (the curation/localization axis, and the one the warp term
#  needs); fragments are fl.realign'd before templating since standard spikes are
#  only loosely stored-aligned.
#
#  Knobs read FK_ALINK_* (CLI > FK_* env > global fiber-kit.yaml > default).
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np
from collections import defaultdict

try:
    from . import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy
    from . import config as cfgmod
    from . import fiber_link_core as flc
except ImportError:
    import neuro_io as nio, fiber_geometry as fg, fiber_lib as fl, session_yaml as sy
    import config as cfgmod
    import fiber_link_core as flc


# ── knob resolution: default <- global fiber-kit.yaml (FK_ALINK_*) <- FK_* env <- CLI ──
_KNOBS = {
    "FK_ALINK_PRIM_FRAC": ("prim_frac", float, 0.15),
    "FK_ALINK_Z": ("z", float, 0.5),
    "FK_ALINK_WIN": ("win", int, 12),
    "FK_ALINK_SLIDE": ("slide", int, 2),
    "FK_ALINK_IOU_THR": ("iou_thr", float, 0.3),
    "FK_ALINK_TARGET_FPR": ("target_fpr", float, 2.0),
    "FK_ALINK_RATE_DEV": ("rate_dev", float, 2.0),
    "FK_ALINK_HI_NSPK": ("hi_nspk", int, 300),
    "FK_ALINK_MAX_GAP": ("max_gap", int, 4),
    "FK_ALINK_SPK_CAP": ("spk_cap", int, 1500),
    "FK_ALINK_WARP_THR": ("warp_thr", float, 0.0),
    "FK_ALINK_AMP_THR": ("amp_thr", float, 0.85),
    "FK_ALINK_MIN_FRAG": ("min_frag", int, 15),
}


def _knob_default(name, typ, fallback, gcfg):
    for src in (os.environ.get(name), (gcfg or {}).get(name)):
        if src is None or str(src).strip() == "":
            continue
        try:
            return typ(src)
        except (TypeError, ValueError):
            pass
    return fallback


def read_anchor_pairs(path):
    """fiber-session's overlap anchors: (transition, atom_a, atom_b) in .clc labels.

    These are links established from the same physical spikes in the chunk overlap --
    the only cross-chunk correspondences in the pipeline that do not depend on waveform
    similarity.  Returns [] when the .fibers predates the payload or linking was off."""
    if not path or not os.path.exists(path):
        return []
    try:
        f = np.load(path, allow_pickle=True)
    except Exception:
        return []
    if "drift_anchor_pairs" not in getattr(f, "files", []):
        return []
    ap = np.asarray(f["drift_anchor_pairs"])
    if ap.ndim != 2 or ap.shape[1] < 3 or ap.shape[0] == 0:
        return []
    return [(int(a), int(b)) for _, a, b in ap[:, :3]]


def build_frag(spk, idx, *, spk_cap, ref_sample, sr, rng):
    """Fragment template: realign -> denoise -> mutual-centre, then median +/- sigma."""
    idx = np.asarray(idx)
    if idx.size > spk_cap:
        idx = rng.choice(idx, spk_cap, replace=False)
    w = fg.mutual_center_spikes(fg.denoise(fl.realign(np.asarray(spk[np.sort(idx)], float))),
                                ref_sample=ref_sample)
    med = np.median(w, 0)
    sd = w.std(0, ddof=1) if len(w) > 1 else np.zeros_like(med)
    dom = int(np.argmax(np.ptp(med, 0)))
    e = np.sqrt((med ** 2).mean(1)); s = e.sum()
    c = int(round(float((np.arange(med.shape[0]) * e).sum() / s))) if s > 1e-12 else med.shape[0] // 2
    return dict(med=med, sd=sd, c=c, gd=fg.group_delay_profile(med, sr=sr), dom=dom,
                snr=float(np.ptp(med[:, dom]) / (sd[:, dom].mean() + 1e-9)),
                cx=fg.waveform_complexity(med))


def atom_chunks_and_rates(clu, res, sr, chunk_min):
    """Per-atom chunk index and RELATIVE firing rate (share of its chunk's spike rate).

    Chunk comes from the fragment's median spike time, so this does not assume the
    caller knows fiber-session's windowing beyond its length."""
    t = np.asarray(res, float) / float(sr) / 60.0
    chunk_of, rel = {}, {}
    ck = np.floor(t / float(chunk_min)).astype(int)
    tot = {}
    for c in np.unique(ck):
        m = ck == c
        span = t[m].max() - t[m].min()
        tot[int(c)] = (m.sum() / span) if span > 1e-9 else np.nan
    for a in np.unique(clu[clu > 0]):
        m = clu == a
        c = int(np.median(ck[m]))
        chunk_of[int(a)] = c
        span = t[m].max() - t[m].min()
        rate = (m.sum() / span) if span > 1e-9 else np.nan
        rel[int(a)] = rate / tot[c] if (c in tot and np.isfinite(tot[c]) and tot[c] > 0) else np.nan
    return chunk_of, rel


def calibrate_floor(score, byc, target_fpr, seeds_score=None):
    """Floor at `target_fpr` percent against the WITHIN-CHUNK null (different cells by
    construction).  Returns (floor, n_null, tpr_on_seeds)."""
    null = []
    for v in byc.values():
        for x in range(len(v)):
            for y in range(x + 1, len(v)):
                s = score(v[x], v[y])
                if np.isfinite(s):
                    null.append(s)
    if len(null) < 20:
        return -np.inf, len(null), np.nan
    floor = float(np.percentile(null, 100.0 - float(target_fpr)))
    tpr = np.nan
    if seeds_score is not None and len(seeds_score):
        ss = np.asarray(seeds_score, float); ss = ss[np.isfinite(ss)]
        if ss.size:
            tpr = float(100.0 * np.mean(ss >= floor))
    return floor, len(null), tpr


def anchor_link(frags, byc, *, seeds, score, floor, rel, nspk, chunk_of,
                rate_dev, hi_nspk, warp_kw, max_gap, verbose=True):
    """Anchor-seeded, null-calibrated, gate-before-select agglomerative linking.

    `seeds` are fragment-index pairs applied before any scored link.  Returns a label
    per fragment index (union-find root)."""
    n = len(frags)
    cell = list(range(n))
    mem = {i: {i} for i in range(n)}

    def occupied(c):
        return {chunk_of[i] for i in mem[c]}

    def merge(i, j):
        a, b = cell[i], cell[j]
        if a == b:
            return False
        if occupied(a) & occupied(b):          # a cell may not hold two co-temporal fragments
            return False
        for k in mem[b]:
            cell[k] = a
        mem[a] |= mem[b]; del mem[b]
        return True

    n_seed = sum(merge(i, j) for i, j in seeds)
    if verbose:
        print(f"[anchor-link] seeded {n_seed}/{len(seeds)} overlap-verified anchor links")

    def gate_ok(i, j, use_warp):
        ri, rj = rel.get(i, np.nan), rel.get(j, np.nan)
        if np.isfinite(ri) and np.isfinite(rj) and ri > 0 and rj > 0:
            if abs(np.log2(ri / rj)) > rate_dev:
                return False
        if use_warp and nspk[i] >= hi_nspk and nspk[j] >= hi_nspk:
            if not flc.warp_ok(frags[i], frags[j], **warp_kw):
                return False
        return True

    def sweep(sel, gaps, use_warp, label):
        total = 0
        for gap in gaps:
            ends = {}
            for c, ms in mem.items():
                k = [i for i in sorted(ms, key=lambda x: chunk_of[x]) if sel(i)]
                if k:
                    ends[c] = (k[0], k[-1])
            keys = list(ends)
            surv = []
            for a in keys:                                   # GATE the pool, then select
                i = ends[a][1]
                for b in keys:
                    if a == b:
                        continue
                    j = ends[b][0]
                    g = chunk_of[j] - chunk_of[i]
                    if g < 1 or g > gap:
                        continue
                    s = score(i, j)
                    if np.isfinite(s) and s >= floor and gate_ok(i, j, use_warp):
                        surv.append((s, i, j, a, b))
            fwd, bwd = {}, {}
            for s, i, j, a, b in sorted(surv, key=lambda t: -t[0]):
                fwd.setdefault(a, (s, i, j, b))
                bwd.setdefault(b, (s, i, j, a))
            acc = 0
            for a, (s, i, j, b) in sorted(fwd.items(), key=lambda kv: -kv[1][0]):
                if b in bwd and bwd[b][3] == a and merge(i, j):   # mutual-NN between cells
                    acc += 1
            total += acc
            if verbose:
                print(f"[anchor-link]   {label} gap {gap}: proposed {len(surv):4d}  accepted {acc:3d}  cells {len(mem)}")
            if acc == 0 and gap == max(gaps):
                break
        return total

    gaps = [g for g in range(1, max_gap + 1) for _ in (0, 1)]
    hi = sweep(lambda i: nspk[i] >= hi_nspk, gaps[:max(2, 2 * min(3, max_gap))], True, "backbone ")
    lo = sweep(lambda i: True, gaps, False, "attach   ")
    if verbose:
        print(f"[anchor-link] {n_seed} seeded + {hi} backbone + {lo} attached = {n_seed + hi + lo} links")
    return np.array([cell[i] for i in range(n)])


def main():
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(prog="fiber-anchor-link",
                                 description="Link fiber-session fragments across chunks, seeded by the overlap "
                                             "anchors and gated on a within-session calibrated null.")
    ap.add_argument("session"); ap.add_argument("elec", type=int)
    ap.add_argument("--clu-method", default="stderiv", help="fragment .clu feature space (before the group)")
    ap.add_argument("--clu-stage", dest="clu_stage", default="fiber_session", help="fragment .clu stage tag")
    ap.add_argument("--in-clu", default=None, help="explicit fragment .clu path (overrides --clu-method/--clu-stage)")
    ap.add_argument("--spk-method", "--spk-variant", dest="spk_variant", default="standard",
                    help="waveform axis for templates/warp (standard = curation axis)")
    ap.add_argument("--fibers", default=None, help="path to the .fibers npz carrying the overlap anchors "
                                                   "(default: derive from --clu-method/--clu-stage)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default="anchor_linked",
                    help="post-fiber stage tag of the output .clu (single token)")
    ap.add_argument("--hierarchy", type=int, default=1, help="1 = also write the .clc/.clp sibling hierarchy")
    ap.add_argument("--promote-noise", type=int, default=0,
                    help="1 = treat cluster 0 as a real cell (give it its own atom + cell) instead of noise")
    ap.add_argument("--chunk-min", type=float, default=None, help="chunk length (min); default from yaml or 12")
    ap.add_argument("--seed", type=int, default=0)
    for name, (dest, typ, fb) in _KNOBS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ,
                        default=_knob_default(name, typ, fb, gcfg),
                        help=f"{name} (default {_knob_default(name, typ, fb, gcfg)})")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cfg = sy.resolve_session_params(a.session, a.elec)
    base, elec = cfg["base"], a.elec
    NS, NC, PK, SR = cfg["nsamp"], cfg["nchan"], cfg["peak"], cfg["sr"]
    chunk_min = a.chunk_min if a.chunk_min else float(
        gcfg.get("FK_SESSION_CHUNK_MIN") or os.environ.get("FK_SESSION_CHUNK_MIN") or 12.0)
    print(f"[anchor-link] knobs (CLI > FK_ALINK_* env > $FK_CONFIG > default): "
          f"prim-frac={a.prim_frac} z={a.z} win={a.win} slide={a.slide} iou-thr={a.iou_thr} "
          f"target-fpr={a.target_fpr}% rate-dev={a.rate_dev} hi-nspk={a.hi_nspk} max-gap={a.max_gap} "
          f"warp-thr={a.warp_thr or 'off'} chunk-min={chunk_min:g}")

    res = nio.read_res(base, elec)
    # The fragment layer is the ATOM layer.  Prefer .clc when the source stage wrote a
    # hierarchy: the overlap anchors index .clc labels, and .clu renumbers (fewer labels,
    # different ids), so reading .clu silently points every anchor at the wrong fragment.
    src = "in-clu"
    if a.in_clu:
        _, frag_clu = nio.read_clu_file(a.in_clu, n_spikes=res.size)
    else:
        clc_in = nio.session_path(base, "clc", elec, variant=a.clu_method, tag=a.clu_stage)
        if os.path.exists(clc_in):
            _, frag_clu = nio.read_clu_file(clc_in, n_spikes=res.size); src = "clc (atom layer)"
        else:
            _, frag_clu = nio.read_clu_file(nio.session_path(base, "clu", elec, variant=a.clu_method,
                                                             tag=a.clu_stage), n_spikes=res.size)
            src = "clu (no .clc sibling -- anchors may not resolve)"
    frag_clu = np.asarray(frag_clu, np.int64)
    print(f"[anchor-link] fragment layer: {src}")
    spk = nio.open_spk_file(nio.session_path(base, "spk", elec, variant=a.spk_variant), NS, NC)

    chunk_of_atom, rel_atom = atom_chunks_and_rates(frag_clu, res, SR, chunk_min)
    atoms = sorted(chunk_of_atom)
    frags, keep, byc = [], [], defaultdict(list)
    for at in atoms:
        idx = np.flatnonzero(frag_clu == at)
        if idx.size < a.min_frag:
            continue
        frags.append(build_frag(spk, idx, spk_cap=a.spk_cap, ref_sample=PK, sr=SR, rng=rng))
        keep.append(at); byc[chunk_of_atom[at]].append(len(frags) - 1)
    row = {at: i for i, at in enumerate(keep)}
    nspk = np.array([int((frag_clu == at).sum()) for at in keep])
    rel = {i: rel_atom[at] for i, at in enumerate(keep)}
    chunk_of = {i: chunk_of_atom[at] for i, at in enumerate(keep)}
    print(f"[anchor-link] {len(frags)} fragments (of {len(atoms)} atoms, min-frag {a.min_frag}) "
          f"over {len(byc)} chunks")

    fib = a.fibers or nio.session_path(base, "fibers", elec, variant=a.clu_method, tag=a.clu_stage)
    if not os.path.exists(fib):
        fib = nio.session_path(base, "fibers", elec, variant=a.clu_method)
    pairs = read_anchor_pairs(fib)
    seeds = [(row[x], row[y]) for x, y in pairs if x in row and y in row]
    print(f"[anchor-link] anchors: {len(pairs)} in {os.path.basename(fib) if os.path.exists(fib) else '(none)'}"
          f" -> {len(seeds)} usable")

    ovk = dict(z=a.z, win=a.win, slide=a.slide, iou_thr=a.iou_thr)
    flat = [f["med"].ravel() for f in frags]
    unit = [v / (np.linalg.norm(v) + 1e-12) for v in flat]
    cache = {}

    def score(i, j):
        k = (i, j)
        if k not in cache:
            o = flc.ci_overlap(frags[i], frags[j], flc.pair_channels(frags[i], frags[j], None, a.prim_frac), **ovk)
            cache[k] = float(o * float(unit[i] @ unit[j])) if np.isfinite(o) else np.nan
        return cache[k]

    floor, n_null, tpr = calibrate_floor(score, byc, a.target_fpr,
                                         [score(i, j) for i, j in seeds])
    print(f"[anchor-link] floor {floor:.4f} = {a.target_fpr}% FPR on {n_null} within-chunk "
          f"(different-cell) pairs | recovers {tpr:.0f}% of the anchors" if np.isfinite(tpr)
          else f"[anchor-link] floor {floor:.4f} from {n_null} within-chunk pairs")

    warp_kw = dict(warp_thr=(a.warp_thr if a.warp_thr > 0 else None), amp_thr=a.amp_thr, resid_thr=0.5)
    lab = anchor_link(frags, byc, seeds=seeds, score=score, floor=floor, rel=rel, nspk=nspk,
                      chunk_of=chunk_of, rate_dev=a.rate_dev, hi_nspk=a.hi_nspk,
                      warp_kw=warp_kw, max_gap=a.max_gap)

    # ── atom -> cell over ALL atoms; unlinked and sub-min-frag atoms keep their own cell ──
    n_atoms = int(frag_clu.max())
    noise_atom = None
    child = frag_clu.copy()
    if a.promote_noise and (frag_clu == 0).any():
        noise_atom = n_atoms + 1; n_atoms = noise_atom
        child[frag_clu == 0] = noise_atom
    groups = defaultdict(list)
    for i, l in enumerate(lab):
        groups[int(l)].append(keep[i])
    cell_of, nxt = {}, 1
    for ms in sorted(groups.values(), key=lambda v: -len(v)):
        for at in ms:
            cell_of[at] = nxt
        nxt += 1
    for at in range(1, n_atoms + 1):
        if at not in cell_of:
            cell_of[at] = nxt; nxt += 1
    parent = np.array([cell_of[at] for at in range(1, n_atoms + 1)], np.int64)
    new = np.zeros(child.size, np.int64)
    m = child > 0
    new[m] = parent[child[m] - 1]

    outp = nio.session_path(base, "clu", elec, variant=a.clu_method, tag=a.out_tag)
    nio.write_clu_file(outp, new)
    n_multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"[anchor-link] {n_atoms} atoms -> {nxt - 1} cells ({n_multi} multi-fragment chains) "
          f"| wrote {os.path.basename(outp)}")
    if a.hierarchy:
        nio.write_clu_file(nio.session_path(base, "clc", elec, variant=a.clu_method, tag=a.out_tag), child)
        nio.write_clu_file(nio.session_path(base, "clp", elec, variant=a.clu_method, tag=a.out_tag),
                           parent, n_clusters=n_atoms)      # header = nChildren, as klusters writes it
        print(f"[anchor-link] hierarchy: {n_atoms} children under {nxt - 1} fibers | wrote .clc + .clp")


if __name__ == "__main__":
    main()
