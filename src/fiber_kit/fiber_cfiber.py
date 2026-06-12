#!/usr/bin/env python3
"""fiber_cfiber.py -- complex-fiber feature + drift/identity validation.

Each spike is embedded as a complex LOOP over its window:

    z(t) = sum_c W[t,c] * exp(i*theta_c)        (channels on a ring; theta_c uniform or from geometry)

From the loop we read two coordinates:
  * SHAPE  -- affine-invariant Fourier descriptors of the centred loop (translation+rotation+scale
              invariant).  Hypothesised drift-stable IDENTITY coordinate.
  * AFFINE -- rotation = arg(F1), scale = |F1|, centre = mean_t z.  Hypothesised DRIFT coordinate
              (depth drift rotates/scales/translates the loop); scale is the amplitude r.

Decisive run (CLI): point it at a curated full-session .clu.  For each unit it bins spikes by time
and reports whether the SHAPE descriptor stays flat across time (identity) while the AFFINE
rotation/scale move (drift).  shape_flatness ~ 1 (across-bin variation no larger than within-bin
noise) together with a large rotation_drift confirms the coordinate on real ground truth.

  fiber-cfiber <session> <group> --clu-method stderiv --clu-stage <curated-stage> [--bins 8] [--fig f.png]
"""
import argparse
import numpy as np

try:
    from . import neuro_io as nio, fiber_lib as fl, session_yaml as sy
except ImportError:
    import neuro_io as nio, fiber_lib as fl, session_yaml as sy


# ── primitives (importable as a feature) ─────────────────────────────────────
def channel_angles(C, chpos=None):
    """Per-channel angle theta_c.  Uniform ring by default; from probe geometry (chpos (C,2)) if given.
    A linear probe (collinear sites) is degenerate on a ring, so it falls back to the uniform ring."""
    if chpos is not None:
        xy = np.asarray(chpos, float)
        if xy.shape[0] == C and xy.ndim == 2:
            c = xy - xy.mean(0)
            ev = np.linalg.eigvalsh(c.T @ c)               # 2D spread; collinear sites -> ev[0] ~ 0
            if ev[1] > 0 and ev[0] / ev[1] > 0.05:         # genuinely 2D, not a linear shank
                return np.arctan2(c[:, 1], c[:, 0])
    return 2.0 * np.pi * np.arange(C) / C


def complex_loop(waves, theta, win=None):
    """waves (n,T,C) real -> z (n,Tw) complex loop.  win = index/slice of the time window (default all)."""
    W = waves if win is None else waves[:, win, :]
    return (W * np.exp(1j * np.asarray(theta, float))[None, None, :]).sum(2)


def shape_descriptor(z, modes=(2, 3, 4, -1, -2, -3)):
    """Affine-invariant (translation+rotation+scale) Fourier descriptors of the loop.
    Returns (shape (n, 2*len(modes)), scale=|F1| (n,), rotation=arg(F1) (n,), centre (n,) complex)."""
    centre = z.mean(1)
    F = np.fft.fft(z - centre[:, None], axis=1)
    f1 = F[:, 1]
    norm = np.where(np.abs(f1) < 1e-12, 1.0, f1)
    D = F / norm[:, None]                                  # D[:,1] == 1 -> rotation+scale removed
    Tw = z.shape[1]
    cols = []
    for k in modes:
        c = D[:, k % Tw]; cols += [c.real, c.imag]
    return np.nan_to_num(np.c_[cols].T), np.abs(f1), np.angle(f1), centre


def complex_fiber(waves, theta, win=None, modes=(2, 3, 4, -1, -2, -3)):
    """Convenience: waves -> (z loop, shape, scale, rotation, centre)."""
    z = complex_loop(waves, theta, win)
    shape, scale, rot, centre = shape_descriptor(z, modes)
    return z, shape, scale, rot, centre


# ── decisive-run CLI ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Complex-fiber drift/identity test: per curated unit, bin by time and report whether "
                    "the affine-invariant loop SHAPE stays flat (identity) while the AFFINE rotation/scale "
                    "drift.  Uses RAW (.spk.standard) amplitudes.")
    sy.add_session_args(ap, nchan=False, sr=False, nsamp_default=None, peak=True)
    ap.add_argument("--clu-method", default="stderiv", help="source .clu feature space before the group")
    ap.add_argument("--clu-stage", default="refine_linked", help="source .clu stage after the group")
    ap.add_argument("--in-clu", default=None, help="explicit curated .clu path (overrides method/stage)")
    ap.add_argument("--spk-method", default="standard", help="raw .spk variant to read (.spk.<m>.N)")
    ap.add_argument("--bins", type=int, default=8, help="time bins across the session")
    ap.add_argument("--min-spikes", type=int, default=200)
    ap.add_argument("--min-bins", type=int, default=5, help="require a unit populate >= this many time bins")
    ap.add_argument("--win-pre", type=int, default=10); ap.add_argument("--win-post", type=int, default=12)
    ap.add_argument("--no-align", dest="align", action="store_false", default=True,
                    help="skip per-unit realign before building the loop")
    ap.add_argument("--modes", default="2,3,4,-1,-2,-3", help="comma list of Fourier modes for the shape")
    ap.add_argument("--out", default=None, help="write per-unit metrics .tsv (default <base>.cfiber.<elec>.tsv)")
    ap.add_argument("--fig", default=None, help="write a shape_flatness vs rotation_drift summary figure")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal, nsamp=a.nsamp)
    base, elec, channels = cfg["base"], cfg["group"], cfg["channels"]
    nsamp = cfg["nsamp"]; peak = cfg["peak"] if cfg.get("peak") is not None else a.peak
    if nsamp is None:
        raise SystemExit("[cfiber] nSamples not in <session>.yaml; pass --nsamp")
    C = len(channels)
    res = nio.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=len(res))
    spk, r = nio.open_spk(base, elec, nsamp, C, prefer=[a.spk_method])
    n = min(len(res), len(clu), spk.shape[0]); res, clu = res[:n], clu[:n]
    lo = max(0, peak - a.win_pre); hi = min(nsamp, peak + a.win_post); win = slice(lo, hi)
    theta = channel_angles(C)                              # uniform ring (correct for a linear shank;
    #                                                        2D-probe geometry can be wired via --geometry later)
    modes = tuple(int(x) for x in a.modes.split(","))
    edges = np.linspace(res.min(), res.max(), a.bins + 1)
    print(f"[cfiber] {r.path}: {n} spikes, window[{lo}:{hi}], {C} ch, align={a.align}, {a.bins} time bins")

    rows = []
    for u in np.unique(clu):
        if u == 0:
            continue
        idx = np.flatnonzero(clu == u)
        if len(idx) < a.min_spikes:
            continue
        bi = np.clip(np.searchsorted(edges, res[idx]) - 1, 0, a.bins - 1)
        if (np.bincount(bi, minlength=a.bins) >= 15).sum() < a.min_bins:
            continue
        W = np.asarray(spk[idx], float)
        if a.align:
            W = fl.realign(W)
        _, shape, scale, rot, _ = complex_fiber(W, theta, win, modes)
        # per-bin means + within-bin spread
        bshape = []; brot = []; bscale = []; within = []
        for b in range(a.bins):
            mb = bi == b
            if mb.sum() < 15:
                continue
            bshape.append(shape[mb].mean(0)); brot.append(np.angle(np.exp(1j * rot[mb]).mean()))
            bscale.append(scale[mb].mean()); within.append(np.linalg.norm(shape[mb] - shape[mb].mean(0), axis=1).mean())
        bshape = np.array(bshape)
        across = np.linalg.norm(bshape - bshape.mean(0), axis=1).mean()        # across-bin shape variation
        flat = across / (np.mean(within) + 1e-9)                               # < ~1 => shape is drift-flat
        rot_drift = np.degrees(np.ptp(np.unwrap(np.array(brot))))              # affine rotation drift
        scale_cv = float(np.std(bscale) / (np.mean(bscale) + 1e-9))
        rows.append((int(u), len(idx), len(bshape), float(flat), float(rot_drift), scale_cv))

    rows.sort(key=lambda x: -x[4])
    out = a.out or f"{base}.cfiber.{elec}.tsv"
    with open(out, "w") as f:
        f.write("unit\tn\tnbins\tshape_flatness\trot_drift_deg\tscale_cv\n")
        for rrow in rows:
            f.write("\t".join(str(x) for x in rrow) + "\n")
    print(f"{'unit':>6}{'n':>7}{'bins':>5}{'shape_flat':>12}{'rot_drift_deg':>15}{'scale_cv':>10}")
    for u, nn, nb, fl_, rd, sc in rows[:25]:
        flag = "  <- drift+flat" if (rd > 30 and fl_ < 1.2) else ""
        print(f"{u:>6}{nn:>7}{nb:>5}{fl_:>12.2f}{rd:>15.0f}{sc:>10.2f}{flag}")
    if rows:
        nd = sum(1 for r in rows if r[4] > 30 and r[3] < 1.2)
        print(f"\n[cfiber] {len(rows)} units; {nd} show drift (rot>30deg) with flat shape (<1.2x within-bin)")
    print(f"[cfiber] wrote {out}")

    if a.fig and rows:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        F = np.array([(r[4], r[3]) for r in rows])         # (rot_drift_deg, shape_flatness) per unit
        plt.figure(figsize=(7, 5.5))
        plt.scatter(F[:, 0], F[:, 1], s=18, alpha=0.7)
        plt.axhline(1.0, c="k", lw=.5, ls="--"); plt.axhline(1.2, c="grey", lw=.4, ls=":")
        plt.xlabel("affine rotation drift (deg)  ->  more drift"); plt.ylabel("shape flatness (across/within)  ->  lower = identity stable")
        plt.title("complex fiber: drift (x) vs shape stability (y)\nlower-right = drifts but keeps shape")
        plt.tight_layout(); plt.savefig(a.fig, dpi=110); print(f"[cfiber] wrote {a.fig}")


if __name__ == "__main__":
    main()
