#!/usr/bin/env python3
"""interneuron_drift.py -- estimate probe drift over the session from a tracked interneuron.

A high-SNR interneuron chained across chunks (piece_interneurons) is a moving fiducial: as tissue
drifts relative to the octrode its footprint slides across the sites.  Reading the energy-weighted
centroid of that footprint on the real channel geometry at each chunk gives the drift trajectory --
a DISTANCE (mostly depth, y, the long probe axis) plus a small lateral/ANGULAR shift (x, the site
stagger).  The footprint's principal-axis orientation is reported too.

Chain-building reuses piece_interneurons (chase_from, primary-channel cosine, drift-following); the
geometry is the probe's true site positions (fiber_localize.load_geometry).  Estimating from several
interneurons and taking the median gives a coherent whole-array drift curve (--seed accepts a list).

Usage:
    python3 tools/interneuron_drift.py <session> <group> --seed 134[,262,...] --gap-min 60 \
        [--celltype int] [--weight-pow 2] [--cos-thr 0.92] [--tsv drift.tsv] [--out drift.png]
"""
import argparse
import os
import sys
import numpy as np

try:
    from fiber_kit import session_yaml as sy, neuro_io as nio, fiber_localize as loc
except ImportError:
    import session_yaml as sy, neuro_io as nio, fiber_localize as loc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piece_interneurons as pi                      # fragment_templates, chase_from


def footprint_centroid(amp, pos, pow_=2.0):
    """Energy-weighted centroid (x,y) of a per-channel amplitude footprint on site positions `pos`,
    plus the orientation (deg) of the footprint's principal axis about that centroid."""
    w = np.asarray(amp, float) ** pow_
    w = w / (w.sum() + 1e-12)
    c = w @ pos
    d = pos - c
    cov = (w[:, None] * d).T @ d
    _, evec = np.linalg.eigh(cov)
    ang = float(np.degrees(np.arctan2(evec[1, -1], evec[0, -1])))
    return c, ang


def drift_track(track, pos, *, pow_=2.0):
    """Per-fragment centroid drift relative to the chain's first fragment.  Returns a structured list
    of dicts: t, x, y, dx, dy, dist, drift_angle (deg, atan2(dx,dy)), axis_angle, dom."""
    rows = []
    for f in track:
        c, ax = footprint_centroid(f["amp"], pos, pow_)
        rows.append(dict(clu=f["clu"], t=f["tmid"], x=float(c[0]), y=float(c[1]), axis=ax, dom=f["dom"]))
    x0, y0 = rows[0]["x"], rows[0]["y"]
    for r in rows:
        r["dx"] = r["x"] - x0; r["dy"] = r["y"] - y0
        r["dist"] = float(np.hypot(r["dx"], r["dy"]))
        r["drift_angle"] = float(np.degrees(np.arctan2(r["dx"], r["dy"])))
    return rows


def build_chain(spk, res, ids, *, seed, sr, ngch, celltype, min_n, sig_cap, gap_min, cos_thr,
                amp_ratio, prim_frac, type_spk=None):
    frags = pi.fragment_templates(spk, res, ids, min_n=min_n, sig_cap=sig_cap, sr=sr,
                                  celltype=celltype or None, dom_idx=set(range(ngch)), type_spk=type_spk)
    pos = next((k for k, f in enumerate(frags) if f["clu"] == seed), None)
    if pos is None:
        raise SystemExit(f"[drift] seed clu {seed} not among {len(frags)} fragments (>= --min-n / --celltype)")
    order = pi.chase_from(frags, pos, gap_min=gap_min, cos_thr=cos_thr, amp_ratio=amp_ratio, prim_frac=prim_frac)
    return [frags[i] for i in order]


def figure(tracks, seeds, pos, gch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axd, axp, axa) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    cols = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(tracks), 1)))
    for k, (sd, rows) in enumerate(zip(seeds, tracks)):
        t = [r["t"] for r in rows]
        axd.plot(t, [r["y"] for r in rows], "-o", ms=3, color=cols[k], label=f"clu{sd}")
        axa.plot(t, [r["drift_angle"] for r in rows], "-o", ms=3, color=cols[k])
    # median coherent depth drift across cells (interpolated on a common grid)
    if len(tracks) > 1:
        tg = np.linspace(min(r["t"] for rows in tracks for r in rows),
                         max(r["t"] for rows in tracks for r in rows), 60)
        Y = np.array([np.interp(tg, [r["t"] for r in rows], [r["y"] for r in rows]) for rows in tracks])
        axd.plot(tg, np.median(Y, 0), "k--", lw=2, label="median")
    axd.set_xlabel("time (min)"); axd.set_ylabel("depth centroid y (um)"); axd.set_title("depth drift")
    axd.legend(fontsize=7); axd.grid(alpha=0.3)
    axp.scatter(pos[:, 0], pos[:, 1], s=80, facecolors="none", edgecolors="gray")
    for c, (x, y) in zip(gch, pos):
        axp.text(x, y, f" {c}", fontsize=6, color="gray", va="center")
    for k, rows in enumerate(tracks):
        sc = axp.scatter([r["x"] for r in rows], [r["y"] for r in rows], c=[r["t"] for r in rows],
                         cmap="viridis", s=25)
    axp.set_xlabel("x (um)"); axp.set_ylabel("y (um)"); axp.set_title("footprint centroid path on the octrode")
    axp.set_aspect("equal"); fig.colorbar(sc, ax=axp, label="time (min)", fraction=0.046)
    axa.set_xlabel("time (min)"); axa.set_ylabel("drift direction (deg from +y)"); axa.set_title("angular shift")
    axa.grid(alpha=0.3)
    return fig


def main():
    ap = argparse.ArgumentParser(prog="interneuron_drift", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sy.add_session_args(ap)
    ap.add_argument("--variant", default="stderiv"); ap.add_argument("--stage", default="fiber_session")
    ap.add_argument("--seed", required=True, help="tracked interneuron seed clu id(s), comma-separated")
    ap.add_argument("--celltype", choices=["int", "pyr", ""], default="int")
    ap.add_argument("--weight-pow", type=float, default=2.0, help="footprint centroid weighting exponent (2=energy)")
    ap.add_argument("--min-n", type=int, default=120); ap.add_argument("--sig-cap", type=int, default=1500)
    ap.add_argument("--gap-min", type=float, default=60.0); ap.add_argument("--cos-thr", type=float, default=0.92)
    ap.add_argument("--amp-ratio", type=float, default=2.2); ap.add_argument("--prim-frac", type=float, default=0.3)
    ap.add_argument("--tsv", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group; nsamp = cfg["nsamp"]; nchan = cfg["nchan"]; sr = cfg["sr"]
    gch = np.array(cfg["channels"], int)
    pos = np.asarray(loc.load_geometry(cfg["probe"], cfg["channels"]), float)
    res = nio.read_res(base, elec)
    spk, _ = nio.open_spk_raw(base, elec, nsamp, nchan)
    _, ids = nio.read_clu_at(base, elec, variant=a.variant, tag=a.stage, n_spikes=len(res))

    seeds = [int(x) for x in str(a.seed).split(",")]
    tracks = []
    print(f"[drift] {os.path.basename(base)} elec {elec}: octrode y-span {np.ptp(pos[:,1]):.0f} um, "
          f"x-stagger {np.ptp(pos[:,0]):.0f} um; {len(seeds)} interneuron(s)")
    for sd in seeds:
        track = build_chain(spk, res, ids, seed=sd, sr=sr, ngch=len(gch), celltype=a.celltype,
                            min_n=a.min_n, sig_cap=a.sig_cap, gap_min=a.gap_min, cos_thr=a.cos_thr,
                            amp_ratio=a.amp_ratio, prim_frac=a.prim_frac)
        rows = drift_track(track, pos, pow_=a.weight_pow); tracks.append(rows)
        dy = rows[-1]["y"] - rows[0]["y"]; span = rows[-1]["t"] - rows[0]["t"]
        print(f"  clu{sd}: {len(rows)} points, {rows[0]['t']:.0f}->{rows[-1]['t']:.0f} min | "
              f"depth {rows[0]['y']:.0f}->{rows[-1]['y']:.0f} um (Δ{dy:+.0f}, {dy/max(span,1e-9):+.3f} um/min), "
              f"lateral excursion {np.ptp([r['x'] for r in rows]):.1f} um, "
              f"|drift| max {max(r['dist'] for r in rows):.1f} um")

    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("seed\tclu\tt_min\tx_um\ty_um\tdx_um\tdy_um\tdist_um\tdrift_angle_deg\taxis_angle_deg\tdom_ch\n")
            for sd, rows in zip(seeds, tracks):
                for r in rows:
                    fh.write(f"{sd}\t{r['clu']}\t{r['t']:.2f}\t{r['x']:.2f}\t{r['y']:.2f}\t{r['dx']:.2f}\t"
                             f"{r['dy']:.2f}\t{r['dist']:.2f}\t{r['drift_angle']:.1f}\t{r['axis']:.1f}\t{gch[r['dom']]}\n")
        print(f"  wrote {a.tsv}")

    fig = figure(tracks, seeds, pos, gch)
    out = a.out or f"{base}.drift.{elec}.png"
    fig.savefig(out, dpi=120); print(f"  wrote {out}")


if __name__ == "__main__":
    main()
