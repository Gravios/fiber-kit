#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  fiber_cpos.py — per-spike cluster-position sidecar (.cpos) from RAW amplitudes.
#
#  Localizes each cluster's median RAW template (monopole+dipole inverse,
#  fiber_localize.localize_unit) and writes ONE position row per spike (its parent
#  cluster's position), in .res order, plus a per-cluster localization table.
#
#  Because the over-split partitions a neuron by ENERGY level, each fragment is one
#  energy band, so the per-fragment positions sample the bundle's
#  (x0, y0, z0, A)(energy) TRAJECTORY alongside its d(r) fiber.  depth_shift (the
#  energy-stratified axial extent) is carried per fragment as the trajectory summary.
#
#  RAW amplitudes only: localize on the .fil, never the stderiv .spkD — the stderiv
#  transform removes common mode and reweights channels, breaking the
#  amplitude–distance law.  (NB: de-adaptation is deliberately NOT applied —
#  position is scale-invariant, and A's energy dependence is the trajectory signal.)
#
#    fiber-cpos <session> <group> [--clu-variant V] [--out-variant V]
#               [--channels ...] [--ntotal N] [--nsamp 32] [--peak 16]
#
#  Outputs (dotted-variant, .res order):
#    <base>.cpos.<out-variant>.<elec>            int32 nCols header + float32 (Nspk x nCols)
#                                                cols = x0,y0,z0,A,dist,depth_shift,one_flank
#    <base>.cpos.<out-variant>.<elec>.clusters.npz   per-cluster full localization (+CIs, n, resid)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import numpy as np

try:
    from . import fiber_localize as loc, neuro_io as nio, session_yaml as sy
except ImportError:
    import fiber_localize as loc, neuro_io as nio, session_yaml as sy

CPOS_COLS = ("x0", "y0", "z0", "A", "dist", "depth_shift", "one_flank")


def fil_extractor(filmm, col_idx, peak=16, nsamp=32, sample_offset=0):
    """Return extract(spike_samples)->(m, nsamp, nchan) RAW waveforms, dropping any
    spike whose window falls outside the .fil.  `col_idx` selects the group's columns
    in the interleaved .fil; `sample_offset` subtracts the .fil's first absolute
    sample (0 for a full recording, lo_sample for a windowed slice)."""
    pre, post = peak, nsamp - peak
    T = filmm.shape[0]; col_idx = np.asarray(col_idx, int)

    def extract(spike_samples):
        ss = np.asarray(spike_samples, np.int64) - sample_offset
        ok = (ss - pre >= 0) & (ss + post <= T)
        out = np.empty((int(ok.sum()), nsamp, len(col_idx)), np.float32)
        for k, t in enumerate(ss[ok]):
            out[k] = filmm[t - pre:t + post][:, col_idx]
        return out
    return extract


def localize_clusters(extract, res, clu, xy, *, min_spikes=15, dipole=True, nboot=100):
    """Localize each cluster's median raw template.  Returns {clu_id: localize_unit dict (+ n)}."""
    per = {}
    for cid in np.unique(clu[clu >= 0]):
        idx = np.flatnonzero(clu == cid)
        if len(idx) < min_spikes:
            continue
        W = extract(res[idx])
        if len(W) < min_spikes:
            continue
        r = loc.localize_unit(np.asarray(W, float), xy, dipole=dipole, nboot=nboot)
        r["n"] = int(len(W)); per[int(cid)] = r
    return per


def spike_table(nspk, clu, per):
    """Per-spike (Nspk x len(CPOS_COLS)) table; spikes whose cluster was not localized
    (too small / noise / off-.fil) get NaN position and one_flank=1 (treat as degenerate)."""
    T = np.full((nspk, len(CPOS_COLS)), np.nan, np.float32)
    T[:, CPOS_COLS.index("one_flank")] = 1.0
    for cid, r in per.items():
        m = clu == cid
        for j, k in enumerate(CPOS_COLS):
            T[m, j] = float(r[k])
    return T


def write_cpos(path, table):
    """int32 nCols header + float32 (Nspk x nCols), row-major, .res order."""
    table = np.asarray(table, np.float32)
    with open(path, "wb") as f:
        np.array([table.shape[1]], np.int32).tofile(f)
        table.tofile(f)
    return path


def read_cpos(path):
    """Read a .cpos -> (Nspk, nCols) float32.  Columns are fiber_cpos.CPOS_COLS."""
    with open(path, "rb") as f:
        nc = int(np.fromfile(f, np.int32, 1)[0])
        body = np.fromfile(f, np.float32)
    return body.reshape(-1, nc)


def write_cluster_table(path, per):
    keys = ["x0", "y0", "z0", "A", "dist", "depth_shift", "one_flank",
            "resid", "y_lo", "y_hi", "z_lo", "z_hi", "n"]
    cids = sorted(per)
    arrs = {"clu": np.array(cids, int)}
    for k in keys:
        arrs[k] = np.array([per[c].get(k, np.nan) for c in cids], float)
    np.savez(path, cols=np.array(CPOS_COLS), **arrs)
    return path


def cpos_path(base, variant, elec):
    g = str(elec)
    return f"{base}.cpos.{g}" if variant == "" else f"{base}.cpos.{variant}.{g}"


def main():
    ap = argparse.ArgumentParser(
        description="Write a per-spike cluster-position sidecar (.cpos) by localizing each "
                    "cluster's median RAW template (monopole+dipole). Reads <session>.yaml.")
    ap.add_argument("session"); ap.add_argument("group", type=int)
    ap.add_argument("--channels", default=None); ap.add_argument("--ntotal", type=int, default=None)
    ap.add_argument("--nsamp", type=int, default=32); ap.add_argument("--peak", type=int, default=16)
    ap.add_argument("--fil", default=None, help="path to raw .fil (default <base>.fil)")
    ap.add_argument("--fil-offset", type=int, default=0, help="first absolute sample of the .fil (0 for full recording)")
    ap.add_argument("--clu-variant", default="", help="read <base>.clu.<clu-variant>.<elec>")
    ap.add_argument("--out-variant", default="stderiv", help="tag for <base>.cpos.<out-variant>.<elec>")
    ap.add_argument("--min-spikes", type=int, default=15)
    ap.add_argument("--no-dipole", action="store_true")
    ap.add_argument("--probe", nargs="*", default=None, help="probe file(s) for geometry (else from chunk xy via YAML)")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal, nsamp=a.nsamp)
    base = cfg["base"]; elec = a.group; ntotal = cfg["ntotal"]
    channels = np.array(cfg["channels"], int)
    xy = loc.load_geometry(a.probe, channels) if a.probe else np.array(cfg.get("xy"))
    if xy is None:
        raise SystemExit("[cpos] need probe geometry: pass --probe or ensure <session>.yaml carries site xy")

    res = nio.read_res(base, elec)
    prefer = [a.clu_variant, ""] if a.clu_variant else nio.prefer_canonical()
    _, clu = nio.read_clu(base, elec, n_spikes=len(res), prefer=prefer)
    filmm = nio.open_signal(a.fil or f"{base}.fil", ntotal)
    col_idx = channels                                     # .fil column = physical channel id
    extract = fil_extractor(filmm, col_idx, peak=a.peak, nsamp=a.nsamp, sample_offset=a.fil_offset)

    per = localize_clusters(extract, res, clu, xy, min_spikes=a.min_spikes, dipole=not a.no_dipole)
    T = spike_table(len(res), clu, per)
    p = write_cpos(cpos_path(base, a.out_variant, elec), T)
    pc = write_cluster_table(cpos_path(base, a.out_variant, elec) + ".clusters.npz", per)
    nflank = sum(int(r["one_flank"]) for r in per.values())
    print(f"localized {len(per)}/{len(np.unique(clu[clu>=0]))} clusters "
          f"({nflank} edge/one-flank); wrote {p}  and  {pc}")
    print(f"  cpos columns: {CPOS_COLS}")


if __name__ == "__main__":
    main()
