#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  fiber_stats.py — extract per-(chunk,cluster) fiber statistics from an EXISTING
#  sort (e.g. a refined .clu out of fiber-refine), WITHOUT re-clustering.
#
#  fiber-session emits the .fibers table as a side effect of its own clustering;
#  there was no way to get the same table for a sort produced elsewhere.  This
#  reuses fiber_session.fiber_geom + the per-chunk .fil whitener, so every column
#  is identical and the output drops straight into fiber-drift / fiber-view.
#
#    fiber-stats <session> <group> [--variant refine | --in-clu PATH]
#                [--chunk-min 12 --overlap-min 4 | --whole-session]
#                [--method refine] [--min-cluster 20]
#
#  Output: <base>.fibers.<method>.<elec>  (npz; one row per (chunk, cluster),
#  gid = cluster id).  Adds `resid_frame_med`: the median residual energy of a
#  cluster's spikes to its own mean template WITHIN the masked spike frame (raw,
#  whitener-free) — the cluster-tightness signal used for link prioritisation.
# ─────────────────────────────────────────────────────────────────────────────
import argparse, time
import numpy as np

try:
    from . import fiber_session as fs, fiber_lib as fl, neuro_io as nio, session_yaml as sy
except ImportError:
    import fiber_session as fs, fiber_lib as fl, neuro_io as nio, session_yaml as sy


def resid_frame_med(waves, mask):
    """Median over spikes of the residual energy to the cluster's mean template,
    summed over the masked spike-frame samples × channels (raw, un-whitened)."""
    if len(waves) < 2:
        return float("nan")
    al = fl.realign(np.asarray(waves, float))
    tmpl = al.mean(0)
    e = ((al[:, mask, :] - tmpl[mask, :]) ** 2).reshape(len(al), -1).sum(1)
    return float(np.median(e))


# columns fiber_geom does not fill (set by cluster_chunk_fine / post-passes);
# back-filled with neutral defaults so the schema matches fiber-session exactly.
_DEFAULTS = dict(coarse=-1, radius_incl=float("nan"), n_rejected=0, nn_gid=-1)


def _rows_for_window(waves, res_w, clu_w, W, nmean, mask, sr, n_grid,
                     chunk_idx, tmin, min_cluster, core):
    """Per-cluster fiber_geom over one (whitened) window.  `core` is a boolean
    mask into the window selecting the spikes that belong to this chunk's core
    (so overlap spikes feed the whitener but are not double-counted in stats)."""
    rows = []
    for cid in np.unique(clu_w[core][clu_w[core] >= 0]):
        idx = np.flatnonzero(core & (clu_w == cid))
        if len(idx) < min_cluster:
            continue
        g = fs.fiber_geom(waves[idx], res_w[idx], W, nmean, mask, sr, n_grid,
                          chunk_t0=None, chunk_t1=None)
        g = dict(g, **_DEFAULTS)
        g["gid"] = int(cid); g["chunk"] = int(chunk_idx); g["tmin"] = float(tmin)
        g["resid_frame_med"] = resid_frame_med(waves[idx], mask)
        rows.append(g)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Extract per-(chunk,cluster) fiber statistics from an existing "
                    "sort (no re-clustering). Reads <session>.yaml for channels/sr (no probe needed; depth is energy-weighted).")
    sy.add_session_args(ap)
    ap.add_argument("--clu-method", default="stderiv",
                    help="feature space BEFORE the group (standard|stderiv|...); default stderiv")
    ap.add_argument("--variant", "--clu-stage", dest="variant", default="refine",
                    help="fiber STAGE AFTER the group: read <base>.clu.<clu-method>.<elec>.<variant> "
                         "(default: refine; '' = no stage)")
    ap.add_argument("--in-clu", default=None, help="explicit .clu path (overrides --clu-method/--variant)")
    ap.add_argument("--chunk-min", type=float, default=12.0)
    ap.add_argument("--overlap-min", type=float, default=4.0)
    ap.add_argument("--whole-session", action="store_true",
                    help="one row per cluster over the whole session (single whitener)")
    ap.add_argument("--min-cluster", type=int, default=20, help="skip clusters smaller than this")
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--method", default="refine", help="method tag in the .fibers filename")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    ntotal, nchan, nsamp, sr = cfg["ntotal"], cfg["nchan"], cfg["nsamp"], cfg["sr"]
    gch = np.array(cfg["channels"], int); mask = fl.build_masks(cfg["nsamp"], cfg["peak"]).full

    res = fs.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.variant, n_spikes=len(res))
    spk, spkpath = fs.open_spkD(base, elec, nsamp, nchan)
    assert spk.shape[0] == len(res) == len(clu), \
        f".res {len(res)} / .clu {len(clu)} / {spkpath} {spk.shape[0]} mismatch"
    filmm = nio.open_signal(f"{base}.fil", ntotal)
    nclu = len(np.unique(clu[clu >= 0]))
    print(f"loaded {len(res)} spikes, {nclu} clusters ({spkpath})")

    t_min, t_max = int(res.min()), int(res.max())
    if a.whole_session:
        windows = [(0, np.arange(len(res)), np.ones(len(res), bool), 0.0)]
    else:
        chunk_s = a.chunk_min * 60.0 * sr; ov_s = a.overlap_min * 60.0 * sr
        nchunks = int(np.ceil((t_max - t_min) / chunk_s)); windows = []
        for c in range(nchunks):
            lo, hi = t_min + c * chunk_s, t_min + (c + 1) * chunk_s
            ext = np.flatnonzero((res >= lo - ov_s) & (res < hi + ov_s))
            if len(ext) == 0:
                continue
            core = (res[ext] >= lo) & (res[ext] < hi)
            if core.sum() == 0:
                continue
            windows.append((c, ext, core, (lo - t_min) / sr / 60.0))

    rows = []; t0 = time.time()
    for c, ext, core, tmin in windows:
        res_e = res[ext]
        s0 = int(res_e.min()) - nsamp; s1 = int(res_e.max()) + nsamp + 1
        W, nmean, _ = fs.fil_chunk_whitener(filmm, gch, s0, s1, res_e, nsamp, mask)
        waves = np.asarray(spk[ext], dtype=float)
        r = _rows_for_window(waves, res_e, clu[ext], W, nmean, mask, sr, a.n_grid,
                             c, tmin, a.min_cluster, core)
        rows += r
        print(f"[fiber_stats] window {c}: {int(core.sum())} core spikes -> {len(r)} clusters")

    M = len(rows)
    def col(k, dt): return np.array([row[k] for row in rows], dt) if M else np.zeros(0, dt)
    arrs = dict(
        gid=col("gid", int), chunk=col("chunk", int), tmin=col("tmin", np.float32),
        coarse=col("coarse", int), nspk=col("n", int), radius=col("radius", np.float32),
        refrac=col("refrac", np.float32), depth=col("depth", np.float32),
        width_ms=col("width_ms", np.float32), radius_incl=col("radius_incl", np.float32),
        n_rejected=col("n_rejected", int),
        rate=col("rate", np.float32), presence=col("presence", np.float32),
        burst=col("burst", np.float32), isi_cv=col("isi_cv", np.float32), hill_fp=col("hill_fp", np.float32),
        resid_med=col("resid_med", np.float32), resid_mad=col("resid_mad", np.float32),
        resid_frame_med=col("resid_frame_med", np.float32),
        chan_resid_var_mean=col("chan_resid_var_mean", np.float32),
        chan_resid_var_max=col("chan_resid_var_max", np.float32),
        nn_dist=col("nn_dist", np.float32), nn_gid=col("nn_gid", int),
        lratio=col("lratio", np.float32), iso_dist=col("iso_dist", np.float32),
        radius_slope=col("radius_slope", np.float32), depth_slope=col("depth_slope", np.float32),
        dir_drift=col("dir_drift", np.float32),
        adapt_corr=col("adapt_corr", np.float32), adapt_tau=col("adapt_tau", np.float32),
        adapt_snr=col("adapt_snr", np.float32), adapt_meanabsz=col("adapt_meanabsz", np.float32),
        adapt_fracz3=col("adapt_fracz3", np.float32),
    )
    out = a.out or nio.fibers_path(base, a.method, elec)
    with open(out, "wb") as f:
        np.savez_compressed(f, **arrs)
    print(f"wrote {out}  ({M} cluster-instances over {len(windows)} window(s); "
          f"{time.time() - t0:.0f}s; rows sharing gid = that cluster over time)")


if __name__ == "__main__":
    main()
