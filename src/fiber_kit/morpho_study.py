#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_study.py — the experiments this modelling stack exists to run.
#
#  Four sub-commands:
#
#    variance  cells x biophysics x rotation x position -> footprints, and a
#              variance decomposition saying how much of the observed waveform
#              spread each factor actually accounts for.  The point is NOT the
#              headline "morphologies differ" (they obviously do) but the
#              RATIO: if position outweighs morphology, then a sorter's job is
#              geometry, not cell typing, and a template library indexed by
#              morphology class would be a category error.
#
#    bap       back-propagation amplitude/latency vs path distance, swept over
#              the dendritic A-current, which is the experimentally observed
#              axis along which real CA1 cells differ most.
#
#    input     the laminar afferent map (morpho_input) for a given morphology.
#
#    state     the analysis that ties the two halves together: drive the cell
#              through one pathway at a time and measure how much the recorded
#              footprint moves, in the same cosine units the linker's veto uses,
#              against a drift displacement of known size for calibration.
#
#  Every sub-command writes an .npz next to its report so a claim can be
#  re-checked without re-running the simulation.
# ════════════════════════════════════════════════════════════════════════════
import argparse, os, sys, time
import numpy as np

try:
    from . import morpho_geom as mg, morpho_cable as mc, morpho_eap as me
    from . import morpho_input as mi, morpho_archetype as ma
except ImportError:
    import morpho_geom as mg, morpho_cable as mc, morpho_eap as me
    import morpho_input as mi, morpho_archetype as ma


# ── shared plumbing ─────────────────────────────────────────────────────────
def _load_cell(spec, d_lambda=0.1, max_comp=2500):
    """spec is a morphology file path, or 'archetype:<kind>[:k=v,...]'."""
    if spec.startswith("archetype:"):
        parts = spec.split(":")
        kind = parts[1]
        kw = {}
        if len(parts) > 2 and parts[2]:
            for item in parts[2].split(","):
                k, v = item.split("=")
                kw[k] = float(v)
        c = ma.build(kind, d_lambda=d_lambda, **kw)
        c.name = kind
        return c
    c = mg.compartmentalize(mg.load(spec), d_lambda=d_lambda, max_comp=max_comp)
    c = mg.orient(c)
    c.name = os.path.splitext(os.path.basename(spec))[0]
    return c


def _sites(args):
    if args.probe:
        xy = me.load_probe(args.probe.split(","), [int(v) for v in args.channels.split(",")])
    else:
        xy = me.staggered_octrode(n=args.nchan)
    return xy


def _spike(cell_cmp, bio, dt, t_stop, amp):
    cell = mc.Cell(cell_cmp, bio)
    return cell, mc.simulate(cell, dt=dt, t_stop=t_stop, stim_amp=amp)


def _footprint(im, cmp_, xy, soma_y, lateral, dt, args, rot=0.0):
    """Place the probe, cut a .spk-convention window.  Returns (wave, metrics)
    or (None, None) if the spike did not fit in the window."""
    c = mg.rotate_z(cmp_, rot) if rot else cmp_
    s = me.sites_3d(xy - np.array([0.0, soma_y]), z=lateral)
    out = me.waveform(im, c, s, dt, sr=args.sr, nsamp=args.nsamp, peak=args.peak,
                      sigma=args.sigma)
    if out["wave"] is None:
        return None, None
    return out["wave"], me.metrics(out["wave"], xy, sr=args.sr)


def _ss(vectors, labels):
    """Between-group sum of squares for one factor, on stacked unit vectors."""
    v = np.asarray(vectors, float)
    gm = v.mean(0)
    ss = 0.0
    for lab in np.unique(labels):
        m = labels == lab
        ss += m.sum() * float(((v[m].mean(0) - gm) ** 2).sum())
    return ss


def _report_variance(waves, factors, out, names):
    """Variance decomposition on unit-normalized footprints.

    A balanced factorial design is sampled, so each factor's between-level sum
    of squares is directly comparable and they sum to no more than the total.
    Reported as percent of total SS; the remainder is interaction plus anything
    the design does not cross.
    """
    V = np.stack([me.normalize(w).ravel() for w in waves])
    tot = float(((V - V.mean(0)) ** 2).sum())
    lines = []
    for k in names:
        lines.append((k, 100.0 * _ss(V, np.asarray(factors[k])) / max(tot, 1e-30)))
    return V, tot, lines


# ── variance ────────────────────────────────────────────────────────────────
def cmd_variance(args):
    xy = _sites(args)
    depths = [float(v) for v in args.depths.split(",")]
    laterals = [float(v) for v in args.laterals.split(",")]
    rots = [float(v) for v in args.rotations.split(",")]
    kas = [float(v) for v in args.ka_scales.split(",")]
    cells = args.cells.split(",")

    waves, rows = [], []
    fac = dict(cell=[], ka=[], rot=[], depth=[], lateral=[], position=[])
    t0 = time.time()
    for spec in cells:
        c = _load_cell(spec, args.d_lambda, args.max_comp)
        for ka in kas:
            bio = mc.Biophys(ka_scale=ka, gna=args.gna, gkdr=args.gkdr)
            _, res = _spike(c, bio, args.dt, args.t_stop, args.stim)
            vmax = float(res["v"][:, 0].max())
            if vmax < args.spike_thresh:
                print(f"[variance] {c.name} ka={ka}: no spike (Vmax {vmax:.1f} mV) — skipped",
                      file=sys.stderr)
                continue
            for rot in rots:
                for dy in depths:
                    for lat in laterals:
                        w, m = _footprint(res["im"], c, xy, dy, lat, args.dt, args, rot)
                        if w is None:
                            continue
                        waves.append(w); fac["cell"].append(c.name); fac["ka"].append(ka)
                        fac["rot"].append(rot); fac["depth"].append(dy)
                        fac["lateral"].append(lat); fac["position"].append(f"{dy}/{lat}")
                        rows.append((c.name, ka, rot, dy, lat, m["amp"], m["width_ms"],
                                     m["decay_um"], m["spread_um"], m["peak_chan"], vmax))
    if not waves:
        raise SystemExit("[variance] no footprints produced")
    W = np.stack(waves)
    V, tot, lines = _report_variance(waves, fac, args.out,
                                     ["cell", "position", "rot", "ka"])

    cellv = np.asarray(fac["cell"])
    within, between = [], []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            d = 1.0 - float(V[i] @ V[j])
            (within if cellv[i] == cellv[j] else between).append(d)
    within = np.asarray(within); between = np.asarray(between)

    print(f"\n=== waveform variance ({len(W)} footprints, {time.time()-t0:.0f}s) ===")
    print(f"design: {len(set(cellv))} morphologies x {len(kas)} ka x {len(rots)} rot "
          f"x {len(depths)*len(laterals)} positions")
    print("\nshape variance (percent of total SS on unit-normalized footprints)")
    for k, pct in lines:
        print(f"  {k:<10s} {pct:6.1f}%")
    print(f"  {'residual':<10s} {100.0 - sum(p for _, p in lines):6.1f}%   "
          "(interactions)")
    print("\ncosine distance 1-cos(a,b)")
    print(f"  same morphology, different nuisance : median {np.median(within):.3f} "
          f"[{np.percentile(within,5):.3f}, {np.percentile(within,95):.3f}]  n={len(within)}")
    print(f"  different morphology                : median {np.median(between):.3f} "
          f"[{np.percentile(between,5):.3f}, {np.percentile(between,95):.3f}]  n={len(between)}")
    ov = float((between < np.percentile(within, 95)).mean())
    print(f"  different-morphology pairs below the within-morphology 95th pct: {100*ov:.1f}%")

    print("\nper-cell footprint summary (median over nuisance factors)")
    print(f"  {'cell':<20s} {'amp uV':>8s} {'width ms':>9s} {'decay um':>9s} {'spread um':>10s}")
    arr = np.array([r[5:9] for r in rows], float)
    for nm in dict.fromkeys(cellv):
        m = cellv == nm
        q = np.nanmedian(arr[m], axis=0)
        print(f"  {nm:<20s} {q[0]:8.1f} {q[1]:9.3f} {q[2]:9.0f} {q[3]:10.0f}")

    if args.out:
        np.savez_compressed(args.out, waves=W, cell=cellv, ka=np.asarray(fac["ka"]),
                            rot=np.asarray(fac["rot"]), depth=np.asarray(fac["depth"]),
                            lateral=np.asarray(fac["lateral"]), xy=xy,
                            within=within, between=between)
        print(f"\nwrote {args.out}")


# ── bap ─────────────────────────────────────────────────────────────────────
def cmd_bap(args):
    kas = [float(v) for v in args.ka_scales.split(",")]
    print(f"{'cell':<20s} {'ka':>5s} {'Vsoma':>7s} " +
          "".join(f"{lo:>4.0f}-{hi:<4.0f}" for lo, hi in
                  [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 600)]))
    store = {}
    for spec in args.cells.split(","):
        c = _load_cell(spec, args.d_lambda, args.max_comp)
        for ka in kas:
            bio = mc.Biophys(ka_scale=ka, gna=args.gna, gkdr=args.gkdr)
            cell, res = _spike(c, bio, args.dt, args.t_stop, args.stim)
            b = mc.bap_profile(res, c)
            den = c.type != mg.AXON
            row = []
            for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 600)]:
                m = den & (b["dist"] >= lo) & (b["dist"] < hi)
                row.append(float(np.median(b["amp"][m])) if m.sum() else np.nan)
            print(f"{c.name:<20s} {ka:5.2f} {b['soma_peak']:7.1f} " +
                  "".join(f"{v:9.1f}" for v in row))
            store[f"{c.name}|{ka}"] = np.stack([b["dist"], b["amp"], b["tpeak"]])
    print("\ncolumns are median bAP amplitude (mV above local baseline) by path distance (um)")
    if args.out:
        np.savez_compressed(args.out, **store)
        print(f"wrote {args.out}")


# ── input ───────────────────────────────────────────────────────────────────
def cmd_input(args):
    pw = (mi.parse_bezaire(args.bezaire, post=args.post) if args.bezaire
          else mi.load_table(post=args.post))
    if not pw:
        raise SystemExit(f"[input] no pathways for post={args.post!r}")
    c = _load_cell(args.cells.split(",")[0], args.d_lambda, args.max_comp)
    prof = mi.laminar_profile(pw, c)
    names = [n for n, _, _ in mi.LAYERS]
    print(f"afferent topology onto {args.post} mapped onto {c.name}")
    print("  layers (um, soma centre = 0): " +
          ", ".join(f"{n} [{lo:.0f},{hi:.0f})" for n, lo, hi in mi.LAYERS))
    print(f"\n{'pre':<18s}{'region':<15s}{'E/I':>4s}{'nsyn':>8s}{'g nS':>9s}" +
          "".join(f"{n:>8s}" for n in names) + f"{'outside':>9s}")
    for (pre, reg), d in sorted(prof.items(), key=lambda kv: -kv[1]["total"]):
        r = d["counts"]
        print(f"{pre:<18s}{reg:<15s}{'E' if d['excitatory'] else 'I':>4s}"
              f"{d['total']:8.0f}{d['gmax_nS']:9.1f}" +
              "".join(f"{r[n]:8.0f}" for n in names) + f"{r['outside']:9.0f}")
    out_frac = sum(d["counts"]["outside"] for d in prof.values()) / \
        max(sum(d["total"] for d in prof.values()), 1e-9)
    print(f"\n{100*out_frac:.0f}% of synapses fall outside the layer stack — these are "
          "compartments\nbeyond the modelled 450 um of CA1, i.e. the reconstruction is "
          "taller than the\nsource model's laminar coordinate.  Treat that fraction as "
          "unassigned, not absent.")
    if args.out:
        np.savez_compressed(args.out, **{f"{p}|{r}": np.array(
            [d["counts"][n] for n in names] + [d["counts"]["outside"], d["total"], d["gmax_nS"]])
            for (p, r), d in prof.items()})
        print(f"wrote {args.out}")


# ── state ───────────────────────────────────────────────────────────────────
def cmd_state(args):
    """Pathway-driven dendritic depolarization vs. drift, in cosine units."""
    xy = _sites(args)
    pw = mi.load_table(post=args.post)
    c = _load_cell(args.cells.split(",")[0], args.d_lambda, args.max_comp)
    bio = mc.Biophys(ka_scale=args.ka, gna=args.gna, gkdr=args.gkdr)
    nt = int(round(args.t_stop / args.dt))
    dy, lat = args.depth, args.lateral

    cell = mc.Cell(c, bio)
    res0 = mc.simulate(cell, dt=args.dt, t_stop=args.t_stop, stim_amp=args.stim)
    w0, m0 = _footprint(res0["im"], c, xy, dy, lat, args.dt, args)
    if w0 is None:
        raise SystemExit("[state] baseline spike did not fit the extraction window")

    print(f"baseline: {c.name}  amp {m0['amp']:.1f} uV  width {m0['width_ms']:.3f} ms  "
          f"peak ch{m0['peak_chan']}  Vsoma {res0['v'][:,0].max():.1f} mV")
    print(f"\n{'condition':<34s}{'amp uV':>8s}{'d-amp %':>9s}{'1-cos':>9s}{'dVdend mV':>11s}")

    rows = []
    for p in pw:
        if p.pre not in args.pathways.split(","):
            continue
        act = {p.pre: np.array([args.syn_time])}
        drive = mi.Drive([p], c, args.dt, nt, act, active_fraction=args.active_fraction,
                         rng=np.random.default_rng(args.seed))
        if len(drive) == 0:
            continue
        cell = mc.Cell(c, bio)
        res = mc.simulate(cell, dt=args.dt, t_stop=args.t_stop, stim_amp=args.stim,
                          drive=drive)
        w, m = _footprint(res["im"], c, xy, dy, lat, args.dt, args)
        if w is None:
            print(f"{p.pre+'/'+p.region:<34s}   spike left the extraction window")
            continue
        elig = mi.eligible(p, c)
        k = int(res["v"][:, 0].argmax())
        dv = float(np.median(res["v"][k, elig] - res0["v"][k, elig])) if elig.any() else 0.0
        d = 1.0 - me.cosine(w0, w)
        print(f"{p.pre+'/'+p.region:<34s}{m['amp']:8.1f}"
              f"{100*(m['amp']-m0['amp'])/m0['amp']:9.1f}{d:9.4f}{dv:11.2f}")
        rows.append((p.pre, p.region, m["amp"], d, dv))

    print(f"\n{'drift calibration':<34s}{'amp uV':>8s}{'d-amp %':>9s}{'1-cos':>9s}")
    for um in [float(v) for v in args.drift.split(",")]:
        w, m = _footprint(res0["im"], c, xy, dy + um, lat, args.dt, args)
        if w is None:
            continue
        d = 1.0 - me.cosine(w0, w)
        print(f"{'probe moved %+.0f um'%um:<34s}{m['amp']:8.1f}"
              f"{100*(m['amp']-m0['amp'])/m0['amp']:9.1f}{d:9.4f}")
        rows.append((f"drift{um:+.0f}", "", m["amp"], d, 0.0))

    print("\nRead this as: how many microns of drift would a sorter have to attribute to a\n"
          "change that was actually synaptic state.  A pathway whose 1-cos matches that of\n"
          "a several-micron displacement is a nuisance direction a drift model can absorb\n"
          "wrongly; one far below the smallest drift step is not worth modelling.")
    if args.out:
        np.savez_compressed(args.out, base=w0, xy=xy,
                            label=np.array([f"{a}|{b}" for a, b, _, _, _ in rows]),
                            amp=np.array([r[2] for r in rows]),
                            cosd=np.array([r[3] for r in rows]),
                            dvdend=np.array([r[4] for r in rows]))
        print(f"wrote {args.out}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def _common(p):
    p.add_argument("--cells", default="archetype:pyramidal",
                   help="comma-separated morphology paths and/or archetype:<kind>[:k=v,..]")
    p.add_argument("--d-lambda", type=float, default=0.1, dest="d_lambda")
    p.add_argument("--max-comp", type=int, default=2500, dest="max_comp")
    p.add_argument("--dt", type=float, default=0.01, help="ms")
    p.add_argument("--t-stop", type=float, default=14.0, dest="t_stop", help="ms")
    p.add_argument("--stim", type=float, default=6.0, help="somatic pulse, nA")
    p.add_argument("--gna", type=float, default=0.025)
    p.add_argument("--gkdr", type=float, default=0.01)
    p.add_argument("--out", default=None, help="write results .npz here")
    return p


def _probe(p):
    p.add_argument("--probe", default=None, help="NeuroSuite .probe file(s), comma-separated")
    p.add_argument("--channels", default=None, help="global channel ids for the group")
    p.add_argument("--nchan", type=int, default=8, help="fallback octrode site count")
    p.add_argument("--sr", type=float, default=32552.0)
    p.add_argument("--nsamp", type=int, default=42)
    p.add_argument("--peak", type=int, default=21)
    p.add_argument("--sigma", type=float, default=me.SIGMA_DEFAULT, help="S/m")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-morpho",
        description="Biophysical modelling of spike-waveform variance across "
                    "hippocampal neuron morphologies.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = _probe(_common(sub.add_parser("variance", help="waveform variance decomposition")))
    v.add_argument("--depths", default="0,40,80",
                   help="soma depth relative to site 0, um")
    v.add_argument("--laterals", default="20,40,70", help="soma distance from shank plane, um")
    v.add_argument("--rotations", default="0,90", help="rotation about the depth axis, deg")
    v.add_argument("--ka-scales", default="0.5,1.0,2.0", dest="ka_scales")
    v.add_argument("--spike-thresh", type=float, default=-10.0, dest="spike_thresh")
    v.set_defaults(func=cmd_variance)

    b = _common(sub.add_parser("bap", help="back-propagation profile"))
    b.add_argument("--ka-scales", default="0.25,1.0,4.0", dest="ka_scales")
    b.set_defaults(func=cmd_bap)

    i = _common(sub.add_parser("input", help="laminar afferent map"))
    i.add_argument("--post", default="pyramidalcell")
    i.add_argument("--bezaire", default=None,
                   help="regenerate from a checkout's datasets/ instead of the shipped table")
    i.set_defaults(func=cmd_input)

    s = _probe(_common(sub.add_parser("state", help="pathway drive vs drift, in cosine units")))
    s.add_argument("--post", default="pyramidalcell")
    s.add_argument("--pathways", default="ca3cell,eccell,olmcell,pvbasketcell")
    s.add_argument("--ka", type=float, default=1.0)
    s.add_argument("--depth", type=float, default=40.0)
    s.add_argument("--lateral", type=float, default=30.0)
    s.add_argument("--syn-time", type=float, default=1.5, dest="syn_time",
                   help="ms; synaptic input precedes the somatic spike")
    s.add_argument("--active-fraction", type=float, default=0.1, dest="active_fraction",
                   help="fraction of a pathway's synapses co-active in one event")
    s.add_argument("--drift", default="2,5,10,20", help="calibration displacements, um")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_state)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
