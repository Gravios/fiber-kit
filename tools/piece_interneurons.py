#!/usr/bin/env python3
"""piece_interneurons.py -- track a cell-type / channel-filtered unit across the session's chunk-local
fiber_session fragments by chaining PRIMARY-channel (invariant) template matches, and plot a report.

At the fiber_session stage each unit is over-clustered per chunk, so one interneuron recurs as
separate fragments in different chunks.  Its high-amplitude channels carry the identity and are stable
step-to-step (the weak channels wander physiologically -- see fiber-chan-svd --within), so a cosine on
the PRIMARY channels links the same cell across adjacent chunks even as slow drift shuffles energy
between channels and the end-to-end waveform morphs past a direct anchor match.  This greedily chains
those local links (forward in time, best match within a time gap, gated on primary cosine + amplitude
ratio) and reports how far a unit pieces together.

NOTE this is a research probe, not a validated linker: the fragments are chunk-local (no temporal
overlap), so there is NO shared-spike / refractory check -- the chain rests entirely on the primary-
channel template match.  Read the per-step cosine in the report as the confidence of each link.

Usage:
    # piece same-channel interneurons across chunks, then chase ONE across the whole session:
    python3 tools/piece_interneurons.py <session> <group> --dom-channels 33,34 \
        --variant stderiv --stage fiber_session [--celltype int] [--min-n 120] \
        [--gap-min 45] [--cos-thr 0.92] [--amp-ratio 2.2] [--out report.png]
    python3 tools/piece_interneurons.py <session> <group> --seed 134 --gap-min 60 --out chase.png
    # hold out a found chain, seed a new anchor from what remains:
    python3 tools/piece_interneurons.py <session> <group> --seed 262 --exclude 134,314,... --gap-min 60
    python3 tools/piece_interneurons.py <session> <group> --seed 134 --spk stderiv   # link in stderiv space
    # auto-pick a correlated seed on ch33 and write the chain back as a parent/child .clu/.clc/.clp:
    python3 tools/piece_interneurons.py <session> <group> --seed-like 134 --seed-on 33 --gap-min 60 --write-clu pieced
"""
import argparse
import os
import numpy as np

try:
    from fiber_kit import fiber_lib as fl, session_yaml as sy, neuro_io as nio, fiber_geometry as fg
except ImportError:
    import fiber_lib as fl, session_yaml as sy, neuro_io as nio, fiber_geometry as fg


def fragment_templates(spk, res, ids, *, min_n, sig_cap, sr, celltype, dom_idx, exclude=None, type_spk=None, seed=0):
    """Per-cluster aligned template + time centroid, filtered to `celltype` with dominant channel in
    `dom_idx`.  `exclude` (a set of clu ids) is held out entirely.  The linking template comes from
    `spk`; cell-typing uses `type_spk` (the STANDARD waveform -- stderiv breaks trough-to-peak
    width) when given, else `spk`.  Returns a time-sorted list."""
    rng = np.random.default_rng(seed)
    exclude = exclude or set()
    tmin = (res - res.min()) / sr / 60.0
    uniq, cnt = np.unique(ids, return_counts=True)
    frags = []
    for u, c in zip(uniq, cnt):
        if u < 2 or c < min_n or int(u) in exclude:
            continue
        idx = np.flatnonzero(ids == u)
        s = idx if len(idx) <= sig_cap else rng.choice(idx, sig_cap, replace=False)
        srt = np.sort(s)
        t = np.median(fl.align_xcorr(np.asarray(spk[srt], float), ref="median", iters=4), axis=0)
        amp = np.ptp(t, axis=0); dom = int(np.argmax(amp))
        if dom not in dom_idx:
            continue
        if celltype:
            tt = (np.median(fl.align_xcorr(np.asarray(type_spk[srt], float), ref="median", iters=4), axis=0)
                  if type_spk is not None else t)
            if fg.classify_celltype(tt, sr) != celltype:
                continue
        frags.append(dict(clu=int(u), n=int(c), t=t, amp=amp, dom=dom, tmid=float(np.median(tmin[idx]))))
    frags.sort(key=lambda f: f["tmid"])
    return frags


def _pcos(a, b, prim_frac):
    """Cosine on the union of the two fragments' primary channels (amp >= prim_frac * max)."""
    p = (a["amp"] >= prim_frac * a["amp"].max()) | (b["amp"] >= prim_frac * b["amp"].max())
    va = a["t"][:, p].ravel(); vb = b["t"][:, p].ravel()
    return float(va @ vb / ((np.linalg.norm(va) + 1e-9) * (np.linalg.norm(vb) + 1e-9)))


def chain(frags, *, gap_min, cos_thr, amp_ratio, prim_frac, min_step_min=2.0):
    """Greedy forward chaining: each fragment links to the best later fragment within `gap_min`,
    primary cosine >= cos_thr and amplitude ratio <= amp_ratio.  Returns chains (lists of indices),
    longest first."""
    nxt = [-1] * len(frags)
    for i, a in enumerate(frags):
        best, bs = -1, cos_thr
        for j in range(i + 1, len(frags)):
            b = frags[j]
            if b["tmid"] - a["tmid"] > gap_min:
                break
            if b["tmid"] <= a["tmid"] + min_step_min:
                continue
            ratio = a["amp"].max() / max(b["amp"].max(), 1e-9); ratio = max(ratio, 1 / ratio)
            cos = _pcos(a, b, prim_frac)
            if cos >= bs and ratio <= amp_ratio:
                bs, best = cos, j
        nxt[i] = best
    succ = {j for j in nxt if j >= 0}
    chains = []
    for i in range(len(frags)):
        if i in succ:
            continue
        ch = [i]; k = nxt[i]
        while k >= 0:
            ch.append(k); k = nxt[k]
        chains.append(ch)
    chains.sort(key=lambda c: -len(c))
    return chains


def chase_from(frags, seed, *, gap_min, cos_thr, amp_ratio, prim_frac):
    """Bidirectional greedy chase of ONE cell from `seed` across ALL channels (drift-following): at each
    end, link to the best union-primary-cosine fragment within gap_min, gated on amplitude ratio.  Unlike
    chain(), this follows the cell wherever its dominant channel drifts.  Returns ordered indices."""
    def step(cur, forward):
        a = frags[cur]; best, bs = -1, cos_thr
        span = range(cur + 1, len(frags)) if forward else range(cur - 1, -1, -1)
        for j in span:
            b = frags[j]
            dt = (b["tmid"] - a["tmid"]) if forward else (a["tmid"] - b["tmid"])
            if dt > gap_min:
                break
            if dt <= 2.0:
                continue
            ratio = a["amp"].max() / max(b["amp"].max(), 1e-9); ratio = max(ratio, 1 / ratio)
            if _pcos(a, b, prim_frac) >= bs and ratio <= amp_ratio:
                bs, best = _pcos(a, b, prim_frac), j
        return best
    used = {seed}
    fwd = [seed]; cur = seed
    while (j := step(cur, True)) >= 0 and j not in used:
        fwd.append(j); used.add(j); cur = j
    bwd = []; cur = seed
    while (j := step(cur, False)) >= 0 and j not in used:
        bwd.append(j); used.add(j); cur = j
    return list(reversed(bwd)) + fwd


def report_figure(track, gch, sr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.array([f["t"] for f in track]); A = np.array([f["amp"] for f in track])
    tm = np.array([f["tmid"] for f in track])
    prim = np.argsort(A.sum(0))[::-1][:2]                       # two most-primary channels overall
    t_ms = np.arange(T.shape[1]) / sr * 1000

    def pcos(a, b):
        p = (A[a] >= 0.3 * A[a].max()) | (A[b] >= 0.3 * A[b].max())
        va = T[a][:, p].ravel(); vb = T[b][:, p].ravel()
        return float(va @ vb / ((np.linalg.norm(va) + 1e-9) * (np.linalg.norm(vb) + 1e-9)))
    anchor = [pcos(0, i) for i in range(len(track))]
    step = [np.nan] + [pcos(i - 1, i) for i in range(1, len(track))]

    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0])
    axh = fig.add_subplot(gs[0, :2])
    im = axh.imshow(A.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[tm[0], tm[-1], gch[0] - 0.5, gch[-1] + 0.5])
    axh.set_yticks(gch); axh.set_ylabel("channel"); axh.set_xlabel("time (min)")
    axh.set_title("per-channel amplitude of the pieced-together unit (footprint drift)")
    fig.colorbar(im, ax=axh, label="p2p amp", fraction=0.05)
    axc = fig.add_subplot(gs[0, 2])
    axc.plot(tm, anchor, "-o", color="#2a9d8f", label="cos->anchor")
    axc.plot(tm, step, "-s", color="#e76f51", label="cos->prev")
    axc.set_ylim(min(0.8, np.nanmin(step) - 0.02), 1.005); axc.set_xlabel("time (min)")
    axc.set_ylabel("primary-channel cosine"); axc.set_title("identity coherence")
    axc.legend(fontsize=7); axc.grid(alpha=0.3)
    for row, ci in ((1, int(prim[0])), (2, int(prim[1]))):
        ax = fig.add_subplot(gs[row, :])
        cols = plt.get_cmap("viridis")(np.linspace(0, 1, len(track)))
        off = 1.05 * max(np.ptp(T[i][:, ci]) for i in range(len(track)))
        for i in range(len(track)):
            ax.plot(t_ms, T[i][:, ci] - i * off, color=cols[i], lw=1.3)
            ax.text(t_ms[-1], -i * off, f" {tm[i]:.0f}m", va="center", fontsize=7, color=cols[i])
        ax.set_yticks([]); ax.set_xlabel("ms")
        ax.set_title(f"waveform on ch{gch[ci]} across the track (top = early, bottom = late)")
    fig.suptitle(f"{track[0]['clu']}..{track[-1]['clu']} pieced across chunks "
                 f"({tm[-1]-tm[0]:.0f} min, {len(track)} fragments)", fontsize=12)
    return fig, anchor, step


def write_chain_hierarchy(base, elec, ids, *, variant, in_tag, out_tag, chain_clus):
    """Group the chain's fragments under a NEW parent fiber and write the .clu/.clc/.clp triple for
    the group: the chain becomes a parent (a top-level cluster in .clu) whose children (.clc) are the
    concatenated pieces (.clp maps each piece -> the parent).  Builds the hierarchy from an existing
    .clc/.clp pair at `in_tag` if present, else lifts it from the flat .clu (each cluster its own
    child).  renumber=False keeps every other cluster's id stable; save() backs up any existing files."""
    try:
        from fiber_kit.fiber_refiberize import FiberHierarchy
        from fiber_kit.fiber_microfiberize import lift_identity
    except ImportError:
        from fiber_refiberize import FiberHierarchy
        from fiber_microfiberize import lift_identity
    try:
        h = FiberHierarchy.load(base, elec, variant=variant, tag=in_tag)
    except (FileNotFoundError, OSError):
        child, parent = lift_identity(np.asarray(ids))
        h = FiberHierarchy(child, parent)
    parent_id = h._fresh_fiber()
    grouped = [int(c) for c in chain_clus if int(c) in h.parent]
    for c in grouped:
        h.move_child(c, parent_id)
    out = h.save(base, elec, variant=variant, tag=out_tag, renumber=False)
    return parent_id, grouped, out


def main():
    ap = argparse.ArgumentParser(prog="piece_interneurons", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sy.add_session_args(ap)
    ap.add_argument("--variant", default="stderiv"); ap.add_argument("--stage", default="fiber_session")
    ap.add_argument("--dom-channels", dest="target", default=None,
                    help="physical channels the unit must be DOMINANT on (e.g. 33,34); default any")
    ap.add_argument("--celltype", choices=["int", "pyr", ""], default="int")
    ap.add_argument("--min-n", type=int, default=120); ap.add_argument("--sig-cap", type=int, default=1500)
    ap.add_argument("--gap-min", type=float, default=45.0); ap.add_argument("--cos-thr", type=float, default=0.92)
    ap.add_argument("--amp-ratio", type=float, default=2.2); ap.add_argument("--prim-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=None,
                    help="chase ONE cell across ALL channels from this seed cluster id (drift-following: "
                         "follows it as its dominant channel drifts); ignores --dom-channels, keeps --celltype")
    ap.add_argument("--seed-like", type=int, default=None,
                    help="auto-pick the seed as the fragment MOST CORRELATED (primary-channel cosine) with "
                         "this reference cluster, then chase it -- 'find another seed like clu X and link it'")
    ap.add_argument("--seed-on", default=None,
                    help="restrict --seed-like's chosen seed to fragments dominant on these physical channels "
                         "(e.g. 33); default: the reference cluster's own dominant channel")
    ap.add_argument("--write-clu", default=None, metavar="TAG",
                    help="after a --seed / --seed-like chase, write the group's .clu/.clc/.clp triple at this "
                         "output stage tag with the chain as a PARENT fiber and the concatenated pieces as its "
                         "children (non-destructive: a new tag; existing files are backed up to .bak)")
    ap.add_argument("--spk", choices=["standard", "stderiv"], default="standard",
                    help="waveform space for the linking templates (default standard = raw; stderiv = the "
                         "clustering feature space).  Cell-typing always uses standard (stderiv breaks width).")
    ap.add_argument("--exclude", default=None,
                    help="comma list of clu ids to HOLD OUT of the fragment pool (e.g. a previously-found "
                         "chain) so a new anchor is linked from the remaining fragments only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    exclude = {int(x) for x in a.exclude.split(',')} if a.exclude else set()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group; nsamp = cfg["nsamp"]; nchan = cfg["nchan"]; sr = cfg["sr"]
    gch = np.array(cfg["channels"], int)
    dom_idx = set(range(len(gch))) if not a.target else {int(np.flatnonzero(gch == int(x))[0])
                                                          for x in a.target.split(",") if (gch == int(x)).any()}
    res = nio.read_res(base, elec)
    if a.spk == "stderiv":
        spk, _ = nio.open_spk(base, elec, nsamp, nchan, prefer=nio.prefer_derived())
        type_spk, _ = nio.open_spk_raw(base, elec, nsamp, nchan)   # standard, for cell-typing only
    else:
        spk, _ = nio.open_spk_raw(base, elec, nsamp, nchan)
        type_spk = None
    _, ids = nio.read_clu_at(base, elec, variant=a.variant, tag=a.stage, n_spikes=len(res))

    if a.seed_like is not None and a.seed is None:             # pick the seed most correlated with a reference
        pool = fragment_templates(spk, res, ids, min_n=a.min_n, sig_cap=a.sig_cap, sr=sr,
                                  celltype=a.celltype or None, dom_idx=set(range(len(gch))),
                                  exclude=exclude, type_spk=type_spk)
        ref = next((f for f in pool if f["clu"] == a.seed_like), None)
        if ref is None:
            raise SystemExit(f"[piece] --seed-like reference clu {a.seed_like} is not in the pool "
                             f"(below --min-n {a.min_n}, wrong --celltype, or excluded)")
        if a.seed_on:
            on = {int(np.flatnonzero(gch == int(x))[0]) for x in a.seed_on.split(",") if (gch == int(x)).any()}
        else:
            on = {ref["dom"]}
        cands = [f for f in pool if f["clu"] != a.seed_like and f["dom"] in on]
        if not cands:
            raise SystemExit(f"[piece] no candidate fragments dominant on {a.seed_on or gch[ref['dom']]} to seed from")
        best = max(cands, key=lambda f: _pcos(ref, f, a.prim_frac))
        a.seed = best["clu"]
        print(f"[piece] --seed-like {a.seed_like} (dom ch{gch[ref['dom']]}) -> chose clu{a.seed} "
              f"(dom ch{gch[best['dom']]}, n={best['n']}, primcos {_pcos(ref, best, a.prim_frac):.3f}) as the seed")

    if a.seed is not None:                                      # chase ONE cell across ALL channels
        frags = fragment_templates(spk, res, ids, min_n=a.min_n, sig_cap=a.sig_cap, sr=sr,
                                   celltype=a.celltype or None, dom_idx=set(range(len(gch))), exclude=exclude, type_spk=type_spk)
        pos = next((k for k, f in enumerate(frags) if f["clu"] == a.seed), None)
        if pos is None:
            raise SystemExit(f"[piece] seed clu {a.seed} not among {len(frags)} {a.celltype} fragments (>= --min-n)")
        order = chase_from(frags, pos, gap_min=a.gap_min, cos_thr=a.cos_thr,
                           amp_ratio=a.amp_ratio, prim_frac=a.prim_frac)
        track = [frags[i] for i in order]
        print(f"[piece] {os.path.basename(base)} elec {elec}: chase from clu {a.seed} across all channels "
              f"({a.celltype}, gap {a.gap_min:.0f} min)")
        print(f"  {'clu':>6} {'t(min)':>7} {'domCh':>5} {'n':>6} {'gap':>5} {'cos->prev':>9}")
        for k, f in enumerate(track):
            if k == 0:
                print(f"  {f['clu']:>6} {f['tmid']:>7.1f} {gch[f['dom']]:>5} {f['n']:>6} {'':>5} {'seed':>9}")
            else:
                p = track[k - 1]
                print(f"  {f['clu']:>6} {f['tmid']:>7.1f} {gch[f['dom']]:>5} {f['n']:>6} "
                      f"{f['tmid']-p['tmid']:>5.0f} {_pcos(p, f, a.prim_frac):>9.3f}")
        span = track[-1]["tmid"] - track[0]["tmid"]
        print(f"  tracked {track[0]['tmid']:.0f} -> {track[-1]['tmid']:.0f} min ({span:.0f} min, {len(track)} fragments, "
              f"~{int(span//18)+1} chunks); dominant channel drifts "
              f"{gch[track[0]['dom']]}..{gch[track[-1]['dom']]}")
        fig, _anchor, _step = report_figure(track, gch, sr)
        out = a.out or f"{base}.piece.{elec}.seed{a.seed}.png"
        fig.savefig(out, dpi=120); print(f"  wrote {out}")
        if a.write_clu:
            chain_clus = [f["clu"] for f in track]
            pid, grouped, paths = write_chain_hierarchy(base, elec, ids, variant=a.variant,
                                                        in_tag=a.stage, out_tag=a.write_clu, chain_clus=chain_clus)
            print(f"  hierarchy: chain -> new parent fiber {pid} with {len(grouped)} children {grouped}")
            print(f"  wrote {os.path.basename(paths['clu'])} + .clc + .clp at tag '{a.write_clu}'")
        return

    frags = fragment_templates(spk, res, ids, min_n=a.min_n, sig_cap=a.sig_cap, sr=sr,
                               celltype=a.celltype or None, dom_idx=dom_idx, exclude=exclude, type_spk=type_spk)
    tgt = [int(gch[i]) for i in sorted(dom_idx)] if a.target else "any"
    print(f"[piece] {os.path.basename(base)} elec {elec}: {len(frags)} {a.celltype} fragments dominant on {tgt}")
    if len(frags) < 2:
        raise SystemExit("[piece] need >= 2 fragments")
    chains = chain(frags, gap_min=a.gap_min, cos_thr=a.cos_thr, amp_ratio=a.amp_ratio, prim_frac=a.prim_frac)
    for ch in chains[:8]:
        seg = " -> ".join(f"clu{frags[i]['clu']}@{frags[i]['tmid']:.0f}m(ch{gch[frags[i]['dom']]})" for i in ch)
        print(f"  [{len(ch)} frags, {frags[ch[-1]]['tmid']-frags[ch[0]]['tmid']:.0f} min] {seg}")
    track = [frags[i] for i in chains[0]]
    if len(track) < 2:
        raise SystemExit("[piece] longest chain has < 2 fragments; loosen --cos-thr / --gap-min")
    fig, anchor, step = report_figure(track, gch, sr)
    out = a.out or f"{base}.piece.{elec}.png"
    fig.savefig(out, dpi=120)
    print(f"  longest track: {len(track)} fragments, {track[-1]['tmid']-track[0]['tmid']:.0f} min "
          f"(~{int((track[-1]['tmid']-track[0]['tmid'])//18)+1} chunks); "
          f"step-cosine min {np.nanmin(step):.3f}, anchor-cosine end {anchor[-1]:.3f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
