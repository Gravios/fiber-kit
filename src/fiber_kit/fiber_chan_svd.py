#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  fiber_chan_svd.py  —  per-CHANNEL SVD/PCA of the cluster mean templates.
#
#  Curation aid.  When you merge, some channels look INVARIANT across the
#  candidate clusters (their waveform is the same on every cluster -> that
#  channel is carrying the unit's identity and you trust it) while others VARY
#  slightly (amplitude/shape wander -> that channel is drift/noise, not identity).
#  This quantifies and plots exactly that split.
#
#  Method:
#    1. compute each selected cluster's aligned mean template  T[k] = (nSamp,nCh)
#       (RAW/standard .spk by default -- the waveform you see in Klusters).
#    2. per channel c, stack that channel across clusters  M_c = T[:,:,c]  (K x nSamp),
#       subtract the grand-mean template, and SVD the residual.  The singular
#       values are how much channel c VARIES across the clusters; the right
#       singular vectors are the temporal MODES of that variation.
#    3. rank channels by across-cluster variability (invariant = small) and plot
#       the first --n-comp modes per channel, on a shared magnified scale.
#
#  A channel whose residual is ~flat is invariant (identity-bearing); a channel
#  with a large PC1 that looks like a scaled copy of the template varies mostly in
#  AMPLITUDE (energy-level / drift -- benign to merge); one whose PC1/PC2 change
#  the SHAPE is a different-source signature (be wary of merging).
#
#  Usage:
#    fiber-chan-svd <session> <group> [--in-clu F | --variant stderiv --stage refine]
#        [--clusters 5,12,18] [--spk standard|stderiv] [--n-comp 3] [--normalize]
#        [--min-n 30] [--sig-cap 2000] [--out DIR|PATH] [--tsv PATH]
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np

try:
    from . import fiber_lib as fl
    from . import session_yaml as sy
    from . import neuro_io as nio
except ImportError:                                   # script / flat-layout fallback
    import fiber_lib as fl
    import session_yaml as sy
    import neuro_io as nio


def _need_mpl():
    global plt, mpl
    try:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                               # pragma: no cover
        raise SystemExit("[fiber-chan-svd] needs matplotlib: pip install 'fiber-kit[viz]'")


def _load_clu(base, elec, n_spikes, in_clu, variant, stage):
    """Resolve the requested sort: explicit --in-clu, else a variant/stage-pinned staged
    .clu, else the canonical .clu.  Returns 1-based cluster ids (0 = noise)."""
    if in_clu:
        _, ids = nio.read_clu_file(in_clu, n_spikes=n_spikes)
    elif variant is not None or stage is not None:
        _, ids = nio.read_clu_at(base, elec, variant=variant or "", tag=stage or "", n_spikes=n_spikes)
    else:
        _, ids = nio.read_clu(base, elec, n_spikes=n_spikes, prefer=nio.prefer_canonical())
    return ids.astype(int)


def cluster_templates(spk, ids, clusters, *, sig_cap=2000, normalize=False, seed=0):
    """Aligned mean template per cluster.  Returns (templates (K,nSamp,nCh), kept_ids, counts).
    Each template is the median of up to `sig_cap` xcorr-aligned raw waveforms; with
    `normalize`, each template is divided by its global peak-to-peak so the SVD sees
    SHAPE variation with amplitude (energy-level) drift removed."""
    rng = np.random.default_rng(seed)
    temps, kept, counts = [], [], []
    for c in clusters:
        idx = np.flatnonzero(ids == c)
        if len(idx) == 0:
            continue
        counts.append(len(idx))
        if len(idx) > sig_cap:
            idx = rng.choice(idx, sig_cap, replace=False)
        w = np.asarray(spk[np.sort(idx)], dtype=float)
        t = np.median(fl.align_xcorr(w, ref="median", iters=4), axis=0)   # (nSamp, nCh)
        if normalize:
            t = t / (np.ptp(t) + 1e-9)
        temps.append(t)
        kept.append(int(c))
    return np.asarray(temps), kept, counts


def per_channel_svd(templates, n_comp=3):
    """Per-channel SVD of the across-cluster template residual.  Returns a dict with
    grand-mean template, per-channel components/singular-values/variance-fractions, and
    absolute + template-relative across-cluster variability per channel."""
    K, nsamp, nch = templates.shape
    grand = templates.mean(0)                                   # (nSamp, nCh)
    resid = templates - grand[None]                             # (K, nSamp, nCh)
    comps = np.zeros((nch, n_comp, nsamp))
    svals = np.zeros((nch, n_comp))
    vfrac = np.zeros((nch, n_comp))
    var_abs = np.zeros(nch)                                     # RMS across-cluster deviation
    var_rel = np.zeros(nch)                                     # normalized by template p2p
    for c in range(nch):
        Mc = resid[:, :, c]                                     # (K, nSamp)
        var_abs[c] = float(np.sqrt((Mc ** 2).mean()))
        var_rel[c] = var_abs[c] / (np.ptp(grand[:, c]) + 1e-9)
        # SVD of the residual: U S Vt, rows of Vt are temporal modes of across-cluster variation
        _, s, vt = np.linalg.svd(Mc, full_matrices=False)
        m = min(n_comp, len(s))
        comps[c, :m] = vt[:m]
        svals[c, :m] = s[:m]
        tot = float((s ** 2).sum()) + 1e-12
        vfrac[c, :m] = (s[:m] ** 2) / tot
    return dict(grand=grand, comps=comps, svals=svals, vfrac=vfrac,
                var_abs=var_abs, var_rel=var_rel, K=K)


def figure(res, gch, *, n_comp=3, sr=None, title=""):
    """Three stacked sections: per-channel variability bars, the grand-mean template
    montage, and the per-channel deviation-mode grid (shared magnified y-scale)."""
    grand = res["grand"]; comps = res["comps"]; svals = res["svals"]; vfrac = res["vfrac"]
    nsamp, nch = grand.shape
    ncols = min(4, nch); nrows = int(np.ceil(nch / ncols))
    order = np.argsort(res["var_rel"])                          # invariant (low) -> varying (high)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = (res["var_rel"] - res["var_rel"].min()) / (np.ptp(res["var_rel"]) + 1e-9)

    fig = plt.figure(figsize=(3.4 * ncols, 3.0 + 2.6 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(2 + nrows, ncols, height_ratios=[1.1, 1.4] + [1.0] * nrows)

    # --- Section 1: variability bars (abs + template-relative) --------------
    axb = fig.add_subplot(gs[0, : ncols // 2 or 1])
    xs = np.arange(nch); cols = [cmap(v) for v in norm]
    axb.bar(xs, res["var_abs"], color=cols); axb.set_xticks(xs); axb.set_xticklabels(gch, fontsize=7)
    axb.set_title("across-cluster variability (abs)", fontsize=9); axb.set_ylabel("RMS dev")
    axr = fig.add_subplot(gs[0, ncols // 2 or 1:])
    axr.bar(xs, 100 * res["var_rel"], color=cols); axr.set_xticks(xs); axr.set_xticklabels(gch, fontsize=7)
    axr.set_title("relative to template p2p (%)  -- low = invariant", fontsize=9)
    axr.set_ylabel("% p2p")

    # --- Section 2: grand-mean template montage (channels offset) -----------
    axt = fig.add_subplot(gs[1, :])
    off = 1.15 * np.max(np.ptp(grand, axis=0))
    t = (np.arange(nsamp) / sr * 1000.0) if sr else np.arange(nsamp)
    for c in range(nch):
        axt.plot(t, grand[:, c] - c * off, color=cmap(norm[c]), lw=1.4)
        axt.text(t[0], -c * off, f" ch{gch[c]}", va="center", ha="right", fontsize=7)
    axt.set_title("grand-mean template per channel (colour = variability)", fontsize=9)
    axt.set_yticks([]); axt.set_xlabel("ms" if sr else "sample")

    # --- Section 3: per-channel deviation modes (shared y) ------------------
    scale = svals / np.sqrt(max(res["K"] - 1, 1))               # real deviation magnitude of each mode
    ymax = 1.15 * float(np.max(np.abs(comps * scale[:, :, None]))) + 1e-9
    comp_cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, n_comp))
    for c in range(nch):
        ax = fig.add_subplot(gs[2 + c // ncols, c % ncols])
        for j in range(n_comp):
            ax.plot(t, scale[c, j] * comps[c, j],
                    color=comp_cols[j], lw=1.3,
                    label=f"PC{j+1} {100*vfrac[c,j]:.0f}%")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f"ch{gch[c]}  relvar {100*res['var_rel'][c]:.1f}%", fontsize=8)
        if c == 0:
            ax.legend(fontsize=6, loc="upper right", framealpha=0.6)
        if c % ncols == 0:
            ax.set_ylabel("dev")
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=10)
    return fig, order


def main():
    ap = argparse.ArgumentParser(
        prog="fiber-chan-svd",
        description="Per-channel SVD/PCA of cluster mean templates: which channels are invariant "
                    "vs vary across the clusters (a merge/curation aid).  Reads <session>.yaml.")
    sy.add_session_args(ap)
    ap.add_argument("--in-clu", default=None, help="sort to analyse (default canonical .clu)")
    ap.add_argument("--variant", default=None, help="staged .clu method (e.g. stderiv) instead of --in-clu")
    ap.add_argument("--stage", default=None, help="staged .clu tag (e.g. refine); pair with --variant")
    ap.add_argument("--clusters", default=None,
                    help="comma list of .clu ids to include (default: all ids >= 2 with >= --min-n spikes)")
    ap.add_argument("--spk", choices=["standard", "stderiv"], default="standard",
                    help="waveform space (default standard = the RAW waveform curation sees)")
    ap.add_argument("--n-comp", type=int, default=3, help="components plotted per channel (default 3)")
    ap.add_argument("--min-n", type=int, default=30, help="skip clusters below this many spikes (noisy template)")
    ap.add_argument("--sig-cap", type=int, default=2000, help="spikes sampled per cluster for the template")
    ap.add_argument("--normalize", action="store_true",
                    help="p2p-normalize each template first -> SVD sees SHAPE variation, amplitude drift removed")
    ap.add_argument("--out", default=None, help="output PNG path or directory (default next to the session)")
    ap.add_argument("--tsv", default=None, help="also write the per-channel metric table here")
    a = ap.parse_args()
    _need_mpl()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    nsamp = cfg["nsamp"]; nchan = cfg["nchan"]; sr = cfg["sr"]; gch = list(cfg["channels"])

    res_t = nio.read_res(base, elec)
    n_spikes = len(res_t)
    if a.spk == "stderiv":
        spk, r = nio.open_spk(base, elec, nsamp, nchan, prefer=nio.prefer_derived())
    else:
        spk, r = nio.open_spk_raw(base, elec, nsamp, nchan)
    ids = _load_clu(base, elec, n_spikes, a.in_clu, a.variant, a.stage)
    if len(ids) != spk.shape[0]:
        raise SystemExit(f"[fiber-chan-svd] .clu {len(ids)} vs .spk {spk.shape[0]} spike-count mismatch")

    if a.clusters:
        want = [int(x) for x in a.clusters.split(",")]
    else:
        uniq, cnt = np.unique(ids, return_counts=True)
        want = [int(u) for u, n in zip(uniq, cnt) if u >= 2 and n >= a.min_n]
    if len(want) < 2:
        raise SystemExit(f"[fiber-chan-svd] need >= 2 clusters (got {want}); lower --min-n or pass --clusters")

    templates, kept, counts = cluster_templates(spk, ids, want, sig_cap=a.sig_cap, normalize=a.normalize)
    if len(kept) < 2:
        raise SystemExit("[fiber-chan-svd] fewer than 2 clusters had spikes")
    res = per_channel_svd(templates, n_comp=a.n_comp)

    # text report: channels ranked invariant -> varying
    print(f"[fiber-chan-svd] {os.path.basename(base)} elec {elec} · {len(kept)} clusters "
          f"({a.spk} spk{', p2p-normalized' if a.normalize else ''})")
    print(f"  clusters: {kept}")
    print(f"  {'ch':>4} {'absdev':>8} {'relvar%':>8} {'PC1%':>6} {'PC2%':>6}   (sorted: invariant -> varying)")
    order = np.argsort(res["var_rel"])
    for c in order:
        print(f"  {gch[c]:>4} {res['var_abs'][c]:>8.3f} {100*res['var_rel'][c]:>8.2f} "
              f"{100*res['vfrac'][c,0]:>6.1f} {100*res['vfrac'][c,1]:>6.1f}")
    inv = [gch[c] for c in order[:max(1, len(gch) // 3)]]
    var = [gch[c] for c in order[::-1][:max(1, len(gch) // 3)]]
    print(f"  most INVARIANT (trust for merges): {inv}")
    print(f"  most VARYING   (drift/noise):      {var}")

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("channel\tabs_dev\trel_var\tPC1_frac\tPC2_frac\tPC3_frac\n")
            for c in range(len(gch)):
                fr = "\t".join(f"{res['vfrac'][c,j]:.4f}" for j in range(min(3, a.n_comp)))
                f.write(f"{gch[c]}\t{res['var_abs'][c]:.5f}\t{res['var_rel'][c]:.5f}\t{fr}\n")
        print(f"  wrote {a.tsv}")

    fig, _ = figure(res, gch, n_comp=a.n_comp, sr=sr,
                    title=f"{os.path.basename(base)} elec {elec} — per-channel template SVD "
                          f"({len(kept)} clusters, {a.spk})")
    if a.out and os.path.isdir(a.out):
        out = os.path.join(a.out, f"{os.path.basename(base)}.chansvd.{elec}.png")
    elif a.out:
        out = a.out
    else:
        out = f"{base}.chansvd.{elec}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
