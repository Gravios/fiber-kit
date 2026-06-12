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
#    fiber-cpos <session> <group> [--clu-method M --clu-stage S] [--out-method M --out-stage S]
#               [--channels ...] [--ntotal N] [--nsamp 32] [--peak 16]
#
#  Outputs (dotted-variant, .res order):
#    <base>.cpos.<method>.<elec>.<stage>            int32 nCols header + float32 (Nspk x nCols)
#                                                cols = x0,y0,z0,A,dist,depth_shift,one_flank
#    <base>.cpos.<method>.<elec>.<stage>.clusters.npz   per-cluster full localization (+CIs, n, resid)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import numpy as np

try:
    from . import fiber_localize as loc, fiber_lib as fl, neuro_io as nio, session_yaml as sy
except ImportError:
    import fiber_localize as loc, fiber_lib as fl, neuro_io as nio, session_yaml as sy

CPOS_COLS = ("x0", "y0", "z0", "A", "dist", "depth_shift", "one_flank")


def spk_extractor(spk):
    """Return extract(idx)->(len(idx), nsamp, nchan) RAW waveforms from a pre-extracted
    STANDARD .spk (e.g. nio.open_spk_file on the standard/raw method -- NEVER the stderiv
    .spkD: the stderiv transform breaks the amplitude-distance law).  Preferred source:
    alignment, window and channel-map are fixed once at extraction, no full-recording I/O."""
    spk = np.asarray(spk)

    def extract(idx):
        return np.asarray(spk[np.asarray(idx, int)], np.float32)
    return extract


def fil_extractor(filmm, res, col_idx, peak=16, nsamp=32, sample_offset=0):
    """Return extract(idx)->(m, nsamp, nchan) RAW waveforms windowed from the .fil at the
    spike times res[idx], dropping any spike whose window falls outside the file.  `col_idx`
    selects the group's columns in the interleaved .fil; `sample_offset` subtracts the .fil's
    first absolute sample (0 for a full recording, lo_sample for a windowed slice).  Fallback
    source when no standard .spk is available."""
    pre, post = peak, nsamp - peak
    T = filmm.shape[0]; col_idx = np.asarray(col_idx, int)

    def extract(idx):
        ss = np.asarray(res[np.asarray(idx, int)], np.int64) - sample_offset
        ok = (ss - pre >= 0) & (ss + post <= T)
        out = np.empty((int(ok.sum()), nsamp, len(col_idx)), np.float32)
        for k, t in enumerate(ss[ok]):
            out[k] = filmm[t - pre:t + post][:, col_idx]
        return out
    return extract


def localize_clusters(extract, clu, xy, *, min_spikes=15, dipole=True, nboot=0, amp_method="pc1",
                      templates=True, amp_basis="auto", basis=None):
    """Localize each cluster's median raw template.  `extract(idx)` maps spike INDICES
    (positions in .res order) to raw waveforms.  Returns {clu_id: localize_unit dict (+ n,
    and 'template' = realigned median raw waveform when templates=True -- the shape
    signature used to co-gate / link fragments across chunks).

    `basis` (pre-resolved, e.g. loc.load_pca_basis's .pca.standard eigenvectors) is used as-is.
    Otherwise amp_basis in ('auto','fit') fits ONE group-wide raw basis from `extract`, so the
    positions the depth/offset gates and the linker read are stable at any spike count (no
    per-cluster SVD tail).  amp_basis='none' (or None) reverts to per-cluster SVD."""
    ids = np.unique(clu[clu >= 0])
    rng = np.random.default_rng(0)
    if basis is None and amp_method == "pc1" and amp_basis in ("auto", "fit"):
        class _Extr:                                       # adapt extract(idx) to fit_amp_basis's spk[idx]
            def __getitem__(self, idx): return np.asarray(extract(idx), np.float32)
        by = {int(c): np.flatnonzero(clu == c) for c in ids}
        basis = loc.fit_amp_basis(_Extr(), by, rng)
    per = {}
    for cid in ids:
        idx = np.flatnonzero(clu == cid)
        if len(idx) < min_spikes:
            continue
        W = np.asarray(extract(idx), float)
        if len(W) < min_spikes:
            continue
        r = loc.localize_unit(W, xy, dipole=dipole, nboot=nboot, amp_method=amp_method, basis=basis)
        r["n"] = int(len(W))
        if templates:
            tmpl = np.median(fl.realign(W), 0)
            r["template"] = tmpl.astype(np.float32)
            # waveform SNR: dominant-channel peak-to-peak / baseline noise (raw amplitudes).
            # noise = median across spikes/channels of the pre-peak baseline sample std.
            ptp = float((tmpl.max(0) - tmpl.min(0)).max())
            noise = float(np.median(np.std(W[:, :6, :], axis=1)))
            r["snr"] = ptp / (noise + 1e-6)
        per[int(cid)] = r
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
            "resid", "y_lo", "y_hi", "z_lo", "z_hi", "n", "snr",
            "t_min", "t_mid", "t_max"]            # per-fragment time (s); placed in time for drift/linking
    cids = sorted(per)
    arrs = {"clu": np.array(cids, int)}
    for k in keys:
        arrs[k] = np.array([per[c].get(k, np.nan) for c in cids], float)
    if cids and "template" in per[cids[0]]:
        arrs["template"] = np.stack([per[c]["template"] for c in cids]).astype(np.float32)
    np.savez(path, cols=np.array(CPOS_COLS), **arrs)
    return path


def cpos_path(base, elec, method="", stage=""):
    """<base>.cpos[.<method>].<elec>[.<stage>] -- mirrors the source clu's method+stage so
    e.g. the cpos of clu.stderiv.5.refine is cpos.stderiv.5.refine."""
    return nio.session_path(base, "cpos", elec, variant=method, tag=stage)


def main():
    ap = argparse.ArgumentParser(
        description="Write a per-spike cluster-position sidecar (.cpos) by localizing each "
                    "cluster's median RAW template (monopole+dipole). Reads <session>.yaml.")
    sy.add_session_args(ap, nchan=False, sr=False, nsamp_default=32, peak=True)
    ap.add_argument("--spk", default=None, help="path to a STANDARD/raw .spk (preferred over .fil); never the stderiv .spkD")
    ap.add_argument("--spk-method", default="standard", help="method of the raw .spk to resolve: <base>.spk.<spk-method>.<elec>")
    ap.add_argument("--fil", default=None, help="path to raw .fil (fallback if no standard .spk)")
    ap.add_argument("--fil-offset", type=int, default=0, help="first absolute sample of the .fil (0 for full recording)")
    ap.add_argument("--clu-method", default="stderiv", help="source-clu feature space BEFORE the group (standard|stderiv|...)")
    ap.add_argument("--clu-stage", default="refine", help="source-clu fiber STAGE AFTER the group: read <base>.clu.<clu-method>.<elec>.<clu-stage>")
    ap.add_argument("--in-clu", default=None, help="explicit .clu path (overrides --clu-method/--clu-stage)")
    ap.add_argument("--out-method", default=None, help="cpos method BEFORE the group (default: mirror --clu-method)")
    ap.add_argument("--out-stage", default=None, help="cpos fiber STAGE AFTER the group (default: mirror --clu-stage)")
    ap.add_argument("--min-spikes", type=int, default=15)
    ap.add_argument("--no-dipole", action="store_true")
    ap.add_argument("--no-templates", action="store_true", help="skip per-cluster median templates in the .clusters.npz")
    ap.add_argument("--amp-method", choices=("pc1", "wave", "ptp"), default="pc1",
                    help="per-channel amplitude profile for the position inverse: pc1=rank-1 denoised "
                         "template (default, sharpest footprint + most precise), wave=median-waveform "
                         "ptp, ptp=median per-spike ptp (legacy; ~4-sigma noise floor on far channels "
                         "flattens the footprint).")
    ap.add_argument("--amp-basis", choices=("auto", "pca", "fit", "none"), default="auto",
                    help="amplitude basis the gate-facing positions use: 'pca'=read "
                         ".pca.standard.<elec> (PC1 score per channel = the .fet amplitude); "
                         "'fit'=group basis from .spk; 'auto'=pca if present else fit; 'none'=per-cluster SVD")
    ap.add_argument("--no-amp-basis", action="store_true",
                    help="alias for --amp-basis none (per-cluster SVD)")
    ap.add_argument("--nboot", type=int, default=0,
                    help="bootstrap draws for the depth/distance percentile CIs (z_lo/z_hi/y_lo/y_hi). "
                         "This loop is ~5x the rest of the cost (the dominant runtime); positions "
                         "(x0,y0,z0,A) and the energy-tercile depth-shift do NOT use it. Use --nboot 0 "
                         "for identical positions ~5x faster (analytic sig_y is still written; the "
                         "percentile CIs become NaN).")
    ap.add_argument("--probe", nargs="*", default=None, help="probe file(s) for geometry (else from chunk xy via YAML)")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal, nsamp=a.nsamp)
    base = cfg["base"]; elec = a.group; ntotal = cfg["ntotal"]
    channels = np.array(cfg["channels"], int)
    probe = a.probe or cfg.get("probe")            # CLI --probe wins; else the path(s) named in <session>.yaml
    if not probe:
        raise SystemExit("[cpos] no probe geometry: <session>.yaml names no probe file "
                         "(probeFile/probe/...) and no --probe was given")
    xy = loc.load_geometry(probe, channels)

    res = nio.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=len(res))

    # Prefer a pre-extracted STANDARD/raw .spk (method-pinned, never stderiv): alignment,
    # window and channel-map are fixed at extraction.  Fall back to windowing the .fil.
    spk_path = a.spk
    if spk_path is None:
        for cand in (nio.session_path(base, "spk", elec, variant=a.spk_method), f"{base}.spk.{elec}"):
            if os.path.exists(cand):
                spk_path = cand; break
    if spk_path is not None:
        if spk_path.endswith((f".spkD.{elec}", f".spk.stderiv.{elec}", f".spk.D.{elec}")):
            raise SystemExit(f"[cpos] refusing stderiv waveforms for localization: {spk_path} "
                             f"(use a standard/raw .spk; stderiv breaks the amplitude-distance law)")
        spk = nio.open_spk_file(spk_path, a.nsamp, len(channels))
        extract = spk_extractor(spk); src = spk_path
    else:
        filmm = nio.open_signal(a.fil or f"{base}.fil", ntotal)
        extract = fil_extractor(filmm, res, channels, peak=a.peak, nsamp=a.nsamp, sample_offset=a.fil_offset)
        src = a.fil or f"{base}.fil"

    amp_basis = "none" if a.no_amp_basis else a.amp_basis
    pca_basis = None
    if a.amp_method == "pc1" and amp_basis in ("pca", "auto"):   # prefer the on-disk .pca.standard
        try:
            pca_basis = loc.load_pca_basis(base, elec)
            print(f"[cpos] amplitude basis: on-disk {pca_basis['_path']} (PC1 score per channel)")
        except FileNotFoundError:
            if amp_basis == "pca":
                raise
    per = localize_clusters(extract, clu, xy, min_spikes=a.min_spikes, dipole=not a.no_dipole,
                            nboot=a.nboot, amp_method=a.amp_method, templates=not a.no_templates,
                            amp_basis=amp_basis, basis=pca_basis)
    sr = cfg.get("sr") or 32552.0                          # stamp each fragment's time (s) for drift/linking
    for cid, r in per.items():
        t = res[clu == cid]
        r["t_min"] = float(t.min() / sr); r["t_max"] = float(t.max() / sr)
        r["t_mid"] = float(np.median(t) / sr)
    T = spike_table(len(res), clu, per)
    out_method = a.out_method if a.out_method is not None else a.clu_method
    out_stage = a.out_stage if a.out_stage is not None else a.clu_stage
    cp = cpos_path(base, elec, method=out_method, stage=out_stage)
    p = write_cpos(cp, T)
    pc = write_cluster_table(cp + ".clusters.npz", per)
    nflank = sum(int(r["one_flank"]) for r in per.values())
    print(f"localized {len(per)}/{len(np.unique(clu[clu>=0]))} clusters from {src} "
          f"({nflank} edge/one-flank); wrote {p}  and  {pc}")
    print(f"  cpos columns: {CPOS_COLS}")


if __name__ == "__main__":
    main()
