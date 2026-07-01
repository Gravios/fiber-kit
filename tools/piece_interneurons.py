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
    python3 tools/piece_interneurons.py <session> <group> --dom-channels 33,34 \
        --variant stderiv --stage fiber_session [--celltype int] [--min-n 120] \
        [--gap-min 45] [--cos-thr 0.92] [--amp-ratio 2.2] [--out report.png]
"""
import argparse
import os
import numpy as np

try:
    from fiber_kit import fiber_lib as fl, session_yaml as sy, neuro_io as nio, fiber_geometry as fg
except ImportError:
    import fiber_lib as fl, session_yaml as sy, neuro_io as nio, fiber_geometry as fg


def fragment_templates(spk, res, ids, *, min_n, sig_cap, sr, celltype, dom_idx, seed=0):
    """Per-cluster aligned template + time centroid, filtered to `celltype` with dominant channel in
    `dom_idx`.  Returns a time-sorted list of fragment dicts."""
    rng = np.random.default_rng(seed)
    tmin = (res - res.min()) / sr / 60.0
    uniq, cnt = np.unique(ids, return_counts=True)
    frags = []
    for u, c in zip(uniq, cnt):
        if u < 2 or c < min_n:
            continue
        idx = np.flatnonzero(ids == u)
        s = idx if len(idx) <= sig_cap else rng.choice(idx, sig_cap, replace=False)
        t = np.median(fl.align_xcorr(np.asarray(spk[np.sort(s)], float), ref="median", iters=4), axis=0)
        amp = np.ptp(t, axis=0); dom = int(np.argmax(amp))
        if dom not in dom_idx:
            continue
        if celltype and fg.classify_celltype(t, sr) != celltype:
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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group; nsamp = cfg["nsamp"]; nchan = cfg["nchan"]; sr = cfg["sr"]
    gch = np.array(cfg["channels"], int)
    dom_idx = set(range(len(gch))) if not a.target else {int(np.flatnonzero(gch == int(x))[0])
                                                          for x in a.target.split(",") if (gch == int(x)).any()}
    res = nio.read_res(base, elec)
    spk, _ = nio.open_spk_raw(base, elec, nsamp, nchan)
    _, ids = nio.read_clu_at(base, elec, variant=a.variant, tag=a.stage, n_spikes=len(res))

    frags = fragment_templates(spk, res, ids, min_n=a.min_n, sig_cap=a.sig_cap, sr=sr,
                               celltype=a.celltype or None, dom_idx=dom_idx)
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
