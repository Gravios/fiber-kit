"""fiber-view: visualise fibers.

Three modes, all reading a sort (.clu) + spikes (.spkD) for a group and fitting
each fiber's trajectory d(r) in the whitened feature space:

  templates  per-channel interpolated waveform-template montage -- for each
             selected fiber and channel an image with x=time, y=position along
             the fiber (low->high energy), colour=amplitude; each row is the
             template r*d(r) reconstructed (un-whitened) at that position.  A
             single-cell fiber's rows are vertically uniform; a multi-cell
             footprint visibly morphs down the fiber.
  manifold   the selected local fibers drawn as 3-D curves r*d(r) through a
             PCA(3) projection of the spike cloud -- shows how local fibers fan
             out from the origin and curve.  (PCA(3) is lossy; the % variance
             retained is printed in the axis labels.)
  stats      per-fiber ISI/refractory histogram + the fiber_shape_stats line;
             if a .geom/.geomchunk npz is given (--geom) the geometry time
             series (cone/bend/r_cv across iterations or chunks) is plotted too.

The figure builders take plain arrays so they can be used without session files;
`load_group` does the on-disk loading for the CLI.
"""
import argparse
import os
import numpy as np

from . import fiber_lib as fl
from . import fiber_tracer as ft
from . import fiber_session as fs
from . import neuro_io as nio
from . import session_yaml as sy

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
    _HAVE_MPL = True
except Exception:                                            # pragma: no cover
    _HAVE_MPL = False

try:
    from sklearn.decomposition import PCA
except Exception:                                            # pragma: no cover
    PCA = None


def _need_mpl():
    if not _HAVE_MPL:
        raise SystemExit("fiber-view needs matplotlib: pip install 'fiber-kit[viz]' "
                         "(or pip install matplotlib)")


# ── trajectory / template reconstruction ─────────────────────────────────────
def fiber_curve(waves_idx, W, nmean, mask, npos=80, qlo=0.02, qhi=0.98):
    """Fit one fiber and sample its trajectory.  Returns (X, r, rg, Pw):
    X whitened spikes (n,p), r their radii, rg the npos sampling radii spanning
    the [qlo,qhi] energy range, Pw the whitened trajectory points rg*d(rg)."""
    Wal = fl.realign(waves_idx)
    X = (Wal[:, mask, :].reshape(len(waves_idx), -1) - nmean) @ W
    r = np.linalg.norm(X, axis=1)
    traj = ft.trajectory(X)
    rg = np.linspace(np.quantile(r, qlo), np.quantile(r, qhi), npos)
    Pw = rg[:, None] * ft.predict_many(traj, rg)
    return X, r, rg, Pw


def template_volume(Pw, W, nmean, nch):
    """Un-whiten trajectory points to (position, time, channel) templates."""
    feats = Pw @ np.linalg.pinv(W) + nmean
    nt = feats.shape[1] // nch
    return feats.reshape(len(feats), nt, nch)


def _ids_to_labels(lab, fiber_ids):
    """fiber_ids are .clu ids (1-based, what the user sees); map to 0-based labels
    present in `lab`.  'top:N' -> N largest; 'all' -> every label."""
    present, counts = np.unique(lab[lab >= 0], return_counts=True)
    if isinstance(fiber_ids, str):
        if fiber_ids == "all":
            return list(present[np.argsort(-counts)])
        if fiber_ids.startswith("top:"):
            k = int(fiber_ids.split(":", 1)[1])
            return list(present[np.argsort(-counts)][:k])
        fiber_ids = [int(x) for x in fiber_ids.split(",") if x.strip()]
    out = []
    for fid in fiber_ids:
        l = int(fid) - 1                                     # .clu 1-based -> label
        if l in present:
            out.append(l)
        else:
            print(f"[fiber-view] cluster {fid} not in sort -- skipped")
    return out


# ── figures ─────────────────────────────────────────────────────────────────
def templates_figure(waves, lab, W, nmean, mask, fiber_labels, npos=80,
                     channels=None, cmap="RdBu_r"):
    _need_mpl()
    nch = waves.shape[2]
    chans = list(range(nch)) if channels is None else list(channels)
    vols = {}
    for l in fiber_labels:
        idx = np.flatnonzero(lab == l)
        _, _, _, Pw = fiber_curve(waves[idx], W, nmean, mask, npos=npos)
        vols[l] = (template_volume(Pw, W, nmean, nch),
                   ft.fiber_shape_stats(waves[idx], W, nmean, mask), len(idx))
    vmax = max(np.abs(v[0]).max() for v in vols.values()) * 0.9
    nrow, ncol = len(fiber_labels), len(chans)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(1.9 * ncol, 4.4 * nrow), constrained_layout=True)
    nt = next(iter(vols.values()))[0].shape[1]
    for r0, l in enumerate(fiber_labels):
        T, st, n = vols[l]
        pk = int(np.argmax(np.abs(T).max(axis=(0, 1))))
        for c0, cc in enumerate(chans):
            ax = axes[r0][c0]
            im = ax.imshow(T[:, :, cc], aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                           origin="lower", extent=[0, nt, 0, 100], interpolation="bilinear")
            ax.set_title(f"ch{cc}" + ("  *peak" if cc == pk else ""), fontsize=8,
                         fontweight="bold" if cc == pk else "normal")
            ax.set_xlabel("time (samples)", fontsize=7); ax.tick_params(labelsize=6)
            if c0 == 0:
                ax.set_ylabel(f"fiber {l+1}\n\nposition along fiber\n(low->high energy) %", fontsize=8)
            else:
                ax.set_yticklabels([])
        axes[r0][0].text(0.02, 1.16,
                         f"n={n}  bend={st['traj_bend']:.0f}deg  r_cv={st['r_cv']:.2f}  "
                         f"r_bimod={st['r_bimod']:.2f}  cone={st['cone_med']:.0f}",
                         transform=axes[r0][0].transAxes, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.5, label="waveform amplitude")
    fig.suptitle("fiber templates  —  x=time, y=position along fiber, colour=amplitude; "
                 "each row = interpolated template r*d(r)", fontsize=9)
    return fig


def manifold_figure(waves, lab, W, nmean, mask, fiber_labels, n_spk=400, npos=60,
                    views=((35, 18), (120, 12)), seed=0):
    _need_mpl()
    if PCA is None:                                          # pragma: no cover
        raise SystemExit("fiber-view manifold needs scikit-learn")
    Xs, Ls, curves = [], [], {}
    for l in fiber_labels:
        idx = np.flatnonzero(lab == l)
        X, r, rg, Pw = fiber_curve(waves[idx], W, nmean, mask, npos=npos)
        Xs.append(X); Ls.append(np.full(len(idx), l)); curves[l] = Pw
    Xall = np.vstack(Xs); Lall = np.concatenate(Ls)
    pca = PCA(3).fit(Xall); S = pca.transform(Xall)
    o3 = pca.transform(np.zeros((1, Xall.shape[1])))[0]
    ev = pca.explained_variance_ratio_ * 100.0
    cmap = plt.cm.tab10
    col = {l: cmap(i % 10) for i, l in enumerate(fiber_labels)}
    rng = np.random.RandomState(seed)
    fig = plt.figure(figsize=(8.0 * len(views), 7.5))
    for sp, (az, el) in enumerate(views):
        ax = fig.add_subplot(1, len(views), sp + 1, projection="3d")
        for l in fiber_labels:
            m = np.flatnonzero(Lall == l); m = rng.permutation(m)[:n_spk]
            ax.scatter(S[m, 0], S[m, 1], S[m, 2], s=3, color=col[l], alpha=0.12, linewidths=0)
        for l in fiber_labels:
            Q = pca.transform(curves[l])
            ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], color=col[l], lw=2.6, label=f"fiber {l+1}")
            ax.scatter(*Q[-1], color=col[l], s=28)
        ax.scatter(*o3, color="k", s=40, marker="x"); ax.text(*o3, "  r=0", fontsize=8)
        ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)", fontsize=8)
        ax.set_zlabel(f"PC3 ({ev[2]:.0f}%)", fontsize=8)
        ax.view_init(elev=el, azim=az); ax.tick_params(labelsize=6)
        if sp == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"local fibers as curves r*d(r) through PCA(3)  "
                 f"({ev.sum():.0f}% variance retained; dots=high-energy tips, x=origin)", fontsize=10)
    fig.tight_layout()
    return fig


def stats_figure(waves, lab, res, W, nmean, mask, sr, fiber_labels,
                 floor=None, window_ms=2.0, geom=None):
    _need_mpl()
    n = len(fiber_labels)
    fig, axes = plt.subplots(n, 2, squeeze=False, figsize=(9, 2.4 * n), constrained_layout=True)
    for r0, l in enumerate(fiber_labels):
        idx = np.flatnonzero(lab == l)
        st = ft.fiber_shape_stats(waves[idx], W, nmean, mask)
        # ISI / refractory
        t = np.sort(res[idx]); isi_ms = np.diff(t) / sr * 1000.0
        ax = axes[r0][0]
        ax.hist(isi_ms[isi_ms < 20], bins=80, color="0.4")
        if floor is not None:
            ax.axvline(floor / sr * 1000.0, color="r", lw=1, ls="--")
        ax.axvline(window_ms, color="orange", lw=1, ls=":")
        ax.set_title(f"fiber {l+1}  n={len(idx)}  ISI (ms)", fontsize=8)
        ax.set_xlabel("ISI (ms)", fontsize=7); ax.tick_params(labelsize=6)
        # geometry: time series if a track is present, else the single-shot stat bar
        ax = axes[r0][1]
        track = geom.get(l + 1) if isinstance(geom, dict) else None
        if track is not None:
            xs = [i for i, _ in enumerate(track)]
            for key, c in (("cone_med", "C0"), ("traj_bend", "C1"), ("r_cv", "C2")):
                ys = np.array([s[key] for _, s in track], float)
                ax.plot(xs, ys / (np.nanmax(np.abs(ys)) + 1e-9), c, marker=".", label=key)
            ax.set_title("geometry across track (normalised)", fontsize=8)
            ax.legend(fontsize=6); ax.set_xlabel("snapshot", fontsize=7)
        else:
            keys = ["r_cv", "r_bimod", "cone_med", "traj_bend", "traj_smooth", "resid_mad"]
            ax.bar(range(len(keys)), [st[k] for k in keys], color="0.6")
            ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=40, ha="right", fontsize=6)
            ax.set_title("fiber_shape_stats", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("fiber stats: ISI/refractory + geometry", fontsize=9)
    return fig


# ── bundles: one global fiber across chunks ──────────────────────────────────
# A "bundle" is the set of per-chunk trajectories of one global fiber.  To be
# comparable across chunks (each has its OWN whitener) the curves are stored as
# UN-WHITENED template curves r*d(r) in raw feature space, so drift = the actual
# footprint moving over time.  A selectable table summarises the bundles; picking
# a row plots that bundle (curves + lofted transparent drift sheet).
def make_bundle(gid, curves, counts, times, stats=None):
    """gid: global fiber id; curves: list (per chunk) of (NPOS, nfeat) un-whitened
    template curves; counts: spikes/chunk; times: chunk start (min); stats:
    optional list of per-chunk fiber_shape_stats dicts."""
    return dict(gid=int(gid), curves=[np.asarray(c, float) for c in curves],
                counts=[int(x) for x in counts], times=[float(t) for t in times],
                stats=stats)


def _bundle_frame(curves):
    return PCA(3).fit(np.vstack(curves)) if PCA is not None else None


def projection_basis(curves, ncomp=6):
    """PCA with the top `ncomp` components over a bundle's curves; the display
    projection is a weighted mix of these.  Returns (pca, explained_var_%)."""
    ncomp = min(ncomp, np.vstack(curves).shape[1])
    p = PCA(ncomp).fit(np.vstack(curves))
    return p, p.explained_variance_ratio_ * 100.0


def default_mix(ncomp):
    """K x 3 mixing matrix mapping PC scores to the 3 display axes; identity on
    the first three (PC1->x, PC2->y, PC3->z), zero elsewhere."""
    M = np.zeros((ncomp, 3))
    for i in range(min(3, ncomp)):
        M[i, i] = 1.0
    return M


def apply_mix(pca, curves, M, normalize=True):
    """Project each curve through the PCs then mix to 3-D via M (ncomp x 3).
    Columns are unit-normalised by default so the display axes stay comparable
    in scale as the user dials contributions in and out."""
    Mn = np.asarray(M, float)
    if normalize:
        nrm = np.linalg.norm(Mn, axis=0)
        Mn = Mn / np.where(nrm > 0, nrm, 1.0)
    return [pca.transform(c)[:, :Mn.shape[0]] @ Mn for c in curves]


def bundle_drift_score(bundle, frame=None):
    """Drift = mean pairwise distance between per-chunk high-energy tips / mean
    curve length, in the bundle's common PCA(3) frame.  ~0 = stable rope, larger
    = fanning/translating sheet.  Needs >=2 chunks."""
    cv = bundle["curves"]
    if len(cv) < 2:
        return 0.0
    fp = frame or _bundle_frame(cv)
    Q = [fp.transform(c) for c in cv]
    tips = np.array([q[-1] for q in Q])
    d = np.mean([np.linalg.norm(tips[i] - tips[j])
                 for i in range(len(tips)) for j in range(i + 1, len(tips))])
    L = np.mean([np.linalg.norm(np.diff(q, axis=0), axis=1).sum() for q in Q])
    return float(d / (L + 1e-12))


_BUNDLE_COLS = ("id", "n", "nchunks", "t0", "t1", "drift", "mean_bend", "mean_r_cv")


def bundle_table(bundles):
    """Summary rows for the selectable table -- one per bundle.  Columns:
    id, n (total spikes), nchunks, t0/t1 (min), drift score, mean bend, mean r_cv."""
    rows = []
    for b in bundles:
        st = b.get("stats")
        mb = float(np.mean([s["traj_bend"] for s in st])) if st else float("nan")
        mc = float(np.mean([s["r_cv"] for s in st])) if st else float("nan")
        rows.append(dict(id=b["gid"], n=int(sum(b["counts"])), nchunks=len(b["curves"]),
                         t0=min(b["times"]) if b["times"] else 0.0,
                         t1=max(b["times"]) if b["times"] else 0.0,
                         drift=bundle_drift_score(b), mean_bend=mb, mean_r_cv=mc))
    rows.sort(key=lambda r: -r["drift"])                     # most-drifting first
    return rows


def bundle_figure(bundle, sub=8, alpha=0.28, draw_sheet=True, view=(20, 50),
                  ncomp=6, mix=None):
    """Plot one bundle: per-chunk trajectories coloured by time in the display
    projection, plus (optionally) the transparent drift manifold lofted between
    consecutive chunks along time.  The projection is a mix of the top `ncomp`
    PCs via `mix` (ncomp x 3; default = PC1/PC2/PC3), so the same bundle can be
    re-viewed along different dimensional contributions.  This is what selecting
    a table row renders."""
    _need_mpl()
    cv = bundle["curves"]; nC = len(cv)
    pca, ev6 = projection_basis(cv, ncomp)
    if mix is None:
        mix = default_mix(pca.n_components_)
    Q = apply_mix(pca, cv, mix)
    M = np.stack(Q, 0)                                       # (nC, NPOS, 3)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if draw_sheet and nC >= 2:
        tf = np.linspace(0, nC - 1, (nC - 1) * sub + 1)
        Mf = np.empty((len(tf), M.shape[1], 3))
        for s in range(M.shape[1]):
            for d in range(3):
                Mf[:, s, d] = np.interp(tf, np.arange(nC), M[:, s, d])
        C = cm.viridis(tf / max(nC - 1, 1))[:, None, :].repeat(Mf.shape[1], axis=1); C[..., 3] = alpha
        ax.plot_surface(Mf[:, :, 0], Mf[:, :, 1], Mf[:, :, 2], facecolors=C,
                        rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    for w in range(nC):
        col = cm.viridis(w / max(nC - 1, 1))
        ax.plot(M[w, :, 0], M[w, :, 1], M[w, :, 2], color=col, lw=2.4, zorder=5)
        ax.scatter(*M[w, -1], color=col, s=28, zorder=6)
        ax.scatter(*M[w, 0], color=col, s=12, marker="s", zorder=6)
    ax.set_xlabel("disp-X", fontsize=7); ax.set_ylabel("disp-Y", fontsize=7); ax.set_zlabel("disp-Z", fontsize=7)
    ax.tick_params(labelsize=6); ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(f"bundle {bundle['gid']}  —  {nC} chunks, {sum(bundle['counts'])} spikes, "
                 f"drift {bundle_drift_score(bundle):.2f}", fontsize=9)
    return fig


# ── most-interesting projection tour (guided projection pursuit) ─────────────
def _orthn(M):
    return np.linalg.qr(M)[0][:, :3]


def _frame_dist(P, Q):
    """Chordal subspace distance in [0,1] (0 = same 3-D subspace)."""
    return float(np.sqrt(max(0.0, 3.0 - np.linalg.norm(P.T @ Q, "fro") ** 2)) / np.sqrt(3.0))


def _tour_scatter(scores, blab, clab):
    """Centroid scatter capturing what we want a tour to expose: between-bundle
    separation + between-chunk-within-bundle drift, in the shared PC space."""
    K = scores.shape[1]; m = scores.mean(0); Sb = np.zeros((K, K)); Sd = np.zeros((K, K))
    for b in np.unique(blab):
        Xb = scores[blab == b]; mb = Xb.mean(0); Sb += len(Xb) * np.outer(mb - m, mb - m)
        for c in np.unique(clab[blab == b]):
            Xc = scores[(blab == b) & (clab == c)]; mc = Xc.mean(0)
            Sd += len(Xc) * np.outer(mc - mb, mc - mb)
    return Sb + Sd


def interesting_tour(bundles, ncomp=6, n_keypoints=4, n_random=800, min_sep=0.5, seed=0):
    """Build a guided tour through projection space for the SELECTED bundles.
    Pools their curves into one shared PCA(ncomp) space, scores 3-D projections by
    how much between-bundle + drift structure they expose (tr(P'CP)), and returns
    keypoint frames (K x 3): the default top-3 PCs, the structure-optimal
    projection (top-3 eigvecs of C), then diverse high-scoring random frames.
    Returns (pca, scores, blab, clab, keyframes, C)."""
    if PCA is None:
        raise SystemExit("interesting_tour needs scikit-learn")
    allc = [c for b in bundles for c in b["curves"]]
    K = min(ncomp, np.vstack(allc).shape[1])
    pca = PCA(K).fit(np.vstack(allc))
    sc, bl, cl = [], [], []
    for b in bundles:
        for ci, c in enumerate(b["curves"]):
            s = pca.transform(c); sc.append(s)
            bl += [b["gid"]] * len(s); cl += [b["gid"] * 100000 + ci] * len(s)
    scores = np.vstack(sc); blab = np.array(bl); clab = np.array(cl)
    C = _tour_scatter(scores, blab, clab)
    P0 = np.zeros((K, 3))
    for i in range(3):
        P0[i, i] = 1.0                                       # default = top-3 PCs
    w, V = np.linalg.eigh(C); Popt = V[:, np.argsort(w)[::-1][:3]]
    keys = [P0, _orthn(Popt)]
    rng = np.random.RandomState(seed)
    cand = [_orthn(rng.randn(K, 3)) for _ in range(n_random)]
    order = np.argsort([np.trace(P.T @ C @ P) for P in cand])[::-1]
    for idx in order:
        P = cand[idx]
        if all(_frame_dist(P, k) > min_sep for k in keys):
            keys.append(P)
        if len(keys) >= n_keypoints + 1:
            break
    return pca, scores, blab, clab, keys, C


def _tour_path(keys, steps=24, loop=True):
    """Interpolate keyframes into a smooth frame sequence (blend + re-orthonormalise)."""
    seq = keys + [keys[0]] if loop else keys
    frames = []
    for a, b in zip(seq[:-1], seq[1:]):
        for t in np.linspace(0, 1, steps, endpoint=False):
            frames.append(_orthn((1 - t) * a + t * b))
    frames.append(seq[-1])
    return frames


def render_tour(bundles, out, ncomp=6, n_keypoints=4, steps=24, fps=20,
                spin=0.5, seed=0, dpi=110):
    """Render the most-interesting projection tour of the selected bundles to a
    video (.gif via Pillow, or .mp4 if ffmpeg is available).  Each frame projects
    every bundle's per-chunk curves through the touring projection (camera also
    slowly spins); bundles are coloured distinctly, chunks shaded by time."""
    _need_mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    pca, scores, blab, clab, keys, C = interesting_tour(
        bundles, ncomp=ncomp, n_keypoints=n_keypoints, seed=seed)
    frames = _tour_path(keys, steps=steps)
    curves_pc = [[pca.transform(c) for c in b["curves"]] for b in bundles]  # per bundle, per chunk
    bcols = [cm.tab10(i % 10) for i in range(len(bundles))]
    allpts = scores
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")

    def draw(i):
        P = frames[i]; ax.cla()
        for bi, chunks in enumerate(curves_pc):
            nC = len(chunks)
            for ci, s in enumerate(chunks):
                Y = s @ P; sh = 0.4 + 0.6 * (ci / max(nC - 1, 1))
                col = np.array(bcols[bi]); col[:3] = col[:3] * sh
                ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color=col, lw=2.0)
                ax.scatter(*(s[-1] @ P), color=bcols[bi], s=18)
        G = allpts @ P; pad = 0.05 * (G.max(0) - G.min(0) + 1e-9)
        ax.set_xlim(G[:, 0].min() - pad[0], G[:, 0].max() + pad[0])
        ax.set_ylim(G[:, 1].min() - pad[1], G[:, 1].max() + pad[1])
        ax.set_zlim(G[:, 2].min() - pad[2], G[:, 2].max() + pad[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=18, azim=(spin * i) % 360)
        ax.set_title(f"interesting tour — structure {np.trace(P.T @ C @ P):.1f}", fontsize=9)
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 // fps, blit=False)
    if str(out).lower().endswith(".mp4"):
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(out, writer=FFMpegWriter(fps=fps), dpi=dpi)
        except Exception:
            out = str(out)[:-4] + ".gif"; anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out, keys, C


def _select_bundles(bundles, spec):
    """spec: 'all', 'top:N', or a comma list of bundle ids."""
    if spec in (None, "all"):
        return bundles
    if isinstance(spec, str) and spec.startswith("top:"):
        k = int(spec.split(":", 1)[1])
        return sorted(bundles, key=lambda b: -sum(b["counts"]))[:k]
    ids = {int(x) for x in str(spec).split(",") if x.strip()}
    return [b for b in bundles if b["gid"] in ids]


def tour_main():
    _need_mpl()
    import argparse
    ap = argparse.ArgumentParser(prog="fiber-view-tour",
                                 description="Render the most-interesting projection tour of selected bundles "
                                             "to a video (.gif, or .mp4 if ffmpeg present).")
    ap.add_argument("bundles", help="a .bundles.<group>.npz from fiber-refine --bundles")
    ap.add_argument("--fibers", default="all", help="comma list of bundle ids, 'top:N', or 'all'")
    ap.add_argument("-o", "--out", default=None, help="output .gif/.mp4 (default <bundles>.tour.gif)")
    ap.add_argument("--ncomp", type=int, default=6); ap.add_argument("--keypoints", type=int, default=4)
    ap.add_argument("--steps", type=int, default=24, help="interpolation frames per leg")
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--spin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sel = _select_bundles(load_bundles_npz(a.bundles), a.fibers)
    if len(sel) < 1:
        raise SystemExit("[fiber-view-tour] no bundles selected")
    out = a.out or (a.bundles[:-4] if a.bundles.lower().endswith(".npz") else a.bundles) + ".tour.gif"
    path, keys, C = render_tour(sel, out, ncomp=a.ncomp, n_keypoints=a.keypoints,
                                steps=a.steps, fps=a.fps, spin=a.spin, seed=a.seed)
    print(f"wrote {path}  ({len(sel)} bundles, {len(keys)} keyframes)")


def load_bundles_npz(path):
    """Load a .bundles npz (producer side: refine_chunked).  Expected long-format
    arrays: fiber[N], chunk[N], t_min[N], count[N], curves[N, NPOS, nfeat]
    (un-whitened template curves).  Returns a list of bundles."""
    z = np.load(path, allow_pickle=False)
    fib = z["fiber"]; out = {}
    for i in range(len(fib)):
        g = int(fib[i])
        out.setdefault(g, dict(curves=[], counts=[], times=[]))
        out[g]["curves"].append(z["curves"][i])
        out[g]["counts"].append(int(z["count"][i])); out[g]["times"].append(float(z["t_min"][i]))
    return [make_bundle(g, b["curves"], b["counts"], b["times"]) for g, b in sorted(out.items())]


# ── on-disk loading for the CLI ──────────────────────────────────────────────
def load_group(session, group, in_clu=None, channels=None, ntotal=None, nchan=None,
               nsamp=None, sr=None, dedup=True):
    cfg = sy.resolve_session_params(session, group, channels=channels, ntotal=ntotal,
                                    nchan=nchan, nsamp=nsamp, sr=sr)
    base = cfg["base"]; elec = group
    ntotal = cfg["ntotal"]; nchan = cfg["nchan"]; nsamp = cfg["nsamp"]; sr = cfg["sr"]
    gch = np.array(cfg["channels"], int); mask = fl.MASK_FULL
    res = fs.read_res(base, elec)
    spk, _ = fs.open_spkD(base, elec, nsamp, nchan)
    waves = np.asarray(spk[:], dtype=float)
    if in_clu and os.path.exists(in_clu):
        _, ids = nio.read_clu_file(in_clu, n_spikes=len(res))
    else:
        _, ids = nio.read_clu(base, elec, n_spikes=len(res), prefer=nio.prefer_canonical())
    lab = ids.astype(int) - 1; lab[lab < 0] = -1
    floor, _ = sy.refractory_period_samples(session, group, sr=sr)
    if dedup and floor > 0:
        from .fiber_refine import dedup_spikes
        ptp = np.ptp(waves.reshape(len(waves), -1), axis=1)
        keep = dedup_spikes(res, ptp, int(floor))
        res = res[keep]; waves = waves[keep]; lab = lab[keep]
    filmm = nio.open_signal(f"{base}.fil", ntotal)
    s0 = int(res.min()) - nsamp; s1 = int(res.max()) + nsamp + 1
    W, nmean, _ = fs.fil_chunk_whitener(filmm, gch, s0, s1, res, nsamp, mask)
    return dict(base=base, elec=elec, waves=waves, res=res, lab=lab, W=W, nmean=nmean,
                mask=mask, sr=sr, gch=gch, nchan=nchan, floor=int(floor))


def _load_geom(path):
    """Load a .geom npz into {fiber_id(.clu 1-based): [(iter, stats_dict), ...]}."""
    from .fiber_refine import load_geometry
    g = load_geometry(path)
    keys = g["keys"]; order = g["iter"] if "iter" in g else g["chunk"]
    out = {}
    for i, fid in enumerate(g["fiber"]):
        s = {k: float(g[k][i]) for k in keys}
        out.setdefault(int(fid) + 1, []).append((order[i], s))
    return out


def main():
    ap = argparse.ArgumentParser(prog="fiber-view",
                                 description="Visualise fibers: template montages, a 3-D manifold of "
                                             "local fiber curves, and ISI/geometry panels.")
    sy.add_session_args(ap)
    ap.add_argument("--in-clu", default=None, help="sort to view (default canonical .clu)")
    ap.add_argument("--fibers", default="top:6",
                    help="comma list of .clu cluster ids, 'top:N', or 'all' (default top:6)")
    ap.add_argument("--mode", choices=["templates", "manifold", "stats", "all"], default="all")
    ap.add_argument("--npos", type=int, default=80, help="positions sampled along the fiber")
    ap.add_argument("--geom", default=None, help="a .geom/.geomchunk npz for the stats geometry track")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--out", default=None, help="output path or directory (default next to the session)")
    ap.add_argument("--channels-override", dest="channels_ovr", default=None)
    a = ap.parse_args()
    _need_mpl()

    d = load_group(a.session, a.group, in_clu=a.in_clu, channels=a.channels_ovr,
                   ntotal=a.ntotal, nchan=a.nchan, nsamp=a.nsamp, sr=a.sr, dedup=not a.no_dedup)
    flabels = _ids_to_labels(d["lab"], a.fibers)
    if not flabels:
        raise SystemExit("[fiber-view] no valid fibers selected")
    chans = None if a.channels is None else [int(x) for x in a.channels.split(",")]
    geom = _load_geom(a.geom) if a.geom else None

    base, elec = d["base"], d["elec"]
    outdir = a.out if (a.out and os.path.isdir(a.out)) else None
    def _path(tag):
        if a.out and outdir is None and a.mode != "all":
            return a.out
        stem = os.path.join(outdir, os.path.basename(base)) if outdir else base
        return f"{stem}.fiberview.{elec}.{tag}.png"

    modes = ["templates", "manifold", "stats"] if a.mode == "all" else [a.mode]
    for m in modes:
        if m == "templates":
            fig = templates_figure(d["waves"], d["lab"], d["W"], d["nmean"], d["mask"],
                                   flabels, npos=a.npos, channels=chans)
        elif m == "manifold":
            fig = manifold_figure(d["waves"], d["lab"], d["W"], d["nmean"], d["mask"], flabels)
        else:
            fig = stats_figure(d["waves"], d["lab"], d["res"], d["W"], d["nmean"], d["mask"],
                               d["sr"], flabels, floor=d["floor"], geom=geom)
        p = _path(m); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
