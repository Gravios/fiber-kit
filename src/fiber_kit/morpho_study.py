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
    from . import morpho_envelope as mv
    from . import morpho_chan_ca1 as mca
except ImportError:
    import morpho_geom as mg, morpho_cable as mc, morpho_eap as me
    import morpho_input as mi, morpho_archetype as ma
    import morpho_envelope as mv
    import morpho_chan_ca1 as mca


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


# ── envelope ────────────────────────────────────────────────────────────────
def cmd_envelope(args):
    """Build the physiologically admissible waveform-change envelope."""
    xy = _sites(args)
    pats = args.patterns.split(",")
    poss = [tuple(float(x) for x in p.split("/")) for p in args.positions.split(",")]
    ars = [float(v) for v in args.na_ar.split(",")]
    plats = [float(v) for v in args.plateaus.split(",")]

    groups, nsim = {}, 0
    t0 = time.time()
    for spec in args.cells.split(","):
        c = _load_cell(spec, args.d_lambda, args.max_comp)
        for ar in ars:
            for plat in plats:
                for pat in pats:
                    times = mv.train_times(pat)
                    cell = mc.Cell(c, mc.Biophys(na_ar=ar, gna=args.gna, gkdr=args.gkdr))
                    _, im, v = mv.simulate_train(cell, times, dt=args.dt,
                                                 stim_amp=args.stim, plateau=plat)
                    nsim += 1
                    for dy, lat in poss:
                        sites = me.sites_3d(xy - np.array([0.0, dy]), z=lat)
                        W, _ = mv.train_footprints(im, c, sites, times, args.dt, sr=args.sr,
                                                   nsamp=args.nsamp, peak=args.peak,
                                                   sigma=args.sigma, v=v,
                                                   v_thresh=args.v_thresh,
                                                   detect_uv=args.detect)
                        if len(W):
                            groups.setdefault((c.name, ar, plat, dy, lat), []).append(W)

    R, D, AF, nsp = [], [], [], 0
    for key, Ws in groups.items():
        # Pairs are formed WITHIN one cell at one electrode position, pooling
        # firing states.  Pairing across positions would fold the geometry
        # variance measured by `variance` back into a gate that must not
        # contain it: the gate's job is to say what one cell at one electrode
        # can do.
        W = np.concatenate(Ws)
        if len(W) < 3:
            continue
        nsp += len(W)
        r, d = mv.pairwise(W)
        R.append(r); D.append(d)
        af, _ = mv.along_curve_fraction(W)
        AF.append(af)
    if not R:
        raise SystemExit("[envelope] no usable spikes — lower --detect or --stim")
    R = np.concatenate(R); D = np.concatenate(D); AF = np.asarray(AF, float)
    af = float(np.nanmedian(AF))
    env = mv.build_envelope(R, D, q=args.q, nbin=args.nbin, along_frac=af,
                            meta=dict(cells=args.cells, detect_uv=str(args.detect),
                                      na_ar=args.na_ar, plateaus=args.plateaus,
                                      patterns=args.patterns, q=str(args.q)))

    print(f"=== physiological envelope ({nsim} trains, {nsp} detected spikes, "
          f"{len(R)} within-cell pairs, {time.time()-t0:.0f}s) ===")
    print(f"detection threshold      {args.detect:.0f} uV")
    print(f"amplitude ratio          median {np.median(R):.2f}  p95 {np.percentile(R,95):.2f}  "
          f"max {R.max():.2f}")
    print(f"variance along d(r)      {af:.3f}   (fraction of direction variance the "
          f"energy curve explains)")
    print(f"\n{'amplitude ratio':<22s}{'max admissible 1-cos':>22s}{'pairs':>8s}")
    for k in range(len(env.cos_thr)):
        lo = env.edges[k]
        hi = env.edges[k + 1] if k + 1 < len(env.edges) else np.inf
        n = int(((R >= lo) & (R < hi)).sum())
        rng = f"{lo:.2f} - {hi:.2f}" if np.isfinite(hi) else f">= {lo:.2f}"
        print(f"{rng:<22s}{env.cos_thr[k]:22.4f}{n:8d}")
    print(f"\nratio above {env.ratio_max:.2f} is not reachable by one cell at this "
          "detection threshold:\nany merge requiring it is rejected outright.")
    if args.out:
        env.save(args.out)
        print(f"wrote {args.out}")
    return env


def cmd_gate(args):
    """Apply an envelope to candidate merge pairs."""
    env = mv.Envelope.load(args.envelope)
    z = np.load(args.templates, allow_pickle=True)
    key = args.key or ("templates" if "templates" in z else z.files[0])
    T = np.asarray(z[key], float)
    if T.ndim != 3:
        raise SystemExit(f"[gate] {key} must be (nunit, nsamp, nchan), got {T.shape}")
    lab = [str(x) for x in z[args.labels]] if args.labels and args.labels in z.files \
        else [str(i) for i in range(len(T))]
    print(f"envelope: {env}")
    print(f"{'a':>6s}{'b':>6s}{'ratio':>8s}{'1-cos':>9s}{'allowed':>9s}  verdict")
    nok = 0
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            ok, rho, d, thr = env.admissible(T[i], T[j])
            nok += ok
            why = "" if ok else ("amplitude ratio unreachable" if rho > env.ratio_max
                                 else "shape change exceeds physiology")
            print(f"{lab[i]:>6s}{lab[j]:>6s}{rho:8.3f}{d:9.4f}{thr:9.4f}  "
                  f"{'ADMISSIBLE' if ok else 'REJECT'} {why}")
    tot = len(T) * (len(T) - 1) // 2
    print(f"\n{nok}/{tot} candidate pairs are physiologically admissible")
    print("Admissible means only that ONE CELL COULD produce both; it is a necessary\n"
          "condition for a merge, never a sufficient one.  Refractory, drift and\n"
          "feature-space evidence still apply.")


# ── shape ───────────────────────────────────────────────────────────────────
def cmd_shape(args):
    """Shape-only variance at MATCHED amplitude — the intra-chunk over-merge bound.

    Within one chunk the electrode does not move and the cell does not change
    type, so the only thing that varies is the cell's own state.  The quantity
    that governs whether a within-chunk merge is safe is therefore how much the
    SHAPE can change with amplitude held fixed: an amplitude-ratio-indexed
    envelope is the wrong tool here, because two fragments of the same size are
    exactly the case it says nothing about.

    Pairs are taken within one cell at one probe position with amplitude ratio
    below --match, pooling firing states and synaptic states.  Between-type
    distances are computed at the SAME positions, so the two numbers are
    directly comparable and the gap between them is the whole discriminative
    budget a cosine threshold has to live in.
    """
    xy = _sites(args)
    poss = [tuple(float(x) for x in p.split("/")) for p in args.positions.split(",")]
    pats = args.patterns.split(",")
    kinds = args.types.split(",")
    pw = {p.pre: p for p in mi.load_table(post="pyramidalcell")}
    drives = [None] + [(k, f) for k in args.drive_pathways.split(",") if k
                       for f in [float(x) for x in args.drive_fractions.split(",")]]

    per_type, rows = {}, []
    t0 = time.time()
    for spec in args.cells.split(","):
        c = _load_cell(spec, args.d_lambda, args.max_comp)
        for kind in kinds:
            bank = {}
            for pat in pats:
                times = mv.train_times(pat)
                nt = int(round((max(times) + 6.0) / args.dt))
                for dspec in drives:
                    drive = None
                    if dspec is not None:
                        pre, frac = dspec
                        if pre not in pw:
                            continue
                        drive = mi.Drive([pw[pre]], c, args.dt, nt,
                                         {pre: np.array([max(times[0] - 3.0, 0.5)])},
                                         active_fraction=frac,
                                         rng=np.random.default_rng(0))
                        if len(drive) == 0:
                            drive = None
                    cell = mc.Cell(c, mca.biophys(kind, na_ar=args.na_ar))
                    _, im, v = mv.simulate_train(cell, times, dt=args.dt,
                                                 stim_amp=args.stim, drive=drive)
                    for dy, lat in poss:
                        sites = me.sites_3d(xy - np.array([0.0, dy]), z=lat)
                        W, _ = mv.train_footprints(im, c, sites, times, args.dt, sr=args.sr,
                                                   nsamp=args.nsamp, peak=args.peak,
                                                   sigma=args.sigma, v=v,
                                                   v_thresh=args.v_thresh,
                                                   detect_uv=args.detect)
                        if len(W):
                            bank.setdefault((dy, lat), []).append(W)
            D = []
            for key, Ws in bank.items():
                W = np.concatenate(Ws)
                if len(W) < 2:
                    continue
                r, d = mv.pairwise(W)
                D.append(d[r <= args.match])
            D = np.concatenate(D) if D else np.zeros(0)
            if len(D):
                per_type[kind] = per_type.get(kind, {})
                per_type[kind][c.name] = D
                mm = {k: np.concatenate(v_) for k, v_ in bank.items()}
                rows.append((c.name, kind, len(D), float(np.median(D)),
                             float(np.percentile(D, 99)), float(D.max()),
                             mm))

    if not rows:
        raise SystemExit("[shape] no matched-amplitude pairs — lower --detect or --stim")
    print(f"=== shape variance at matched amplitude (ratio <= {args.match:.2f}, "
          f"{time.time()-t0:.0f}s) ===")
    print(f"detection {args.detect:.0f} uV | states: {len(pats)} ISI patterns x "
          f"{len(drives)} synaptic conditions\n")
    print(f"{'morphology':<16s}{'cell type':<14s}{'pairs':>7s}{'median':>10s}"
          f"{'p99':>10s}{'max':>10s}")
    for nm, kind, n, med, p99, mx, _ in rows:
        flag = " *" if kind in mca.INCOMPLETE else ""
        print(f"{nm:<16s}{kind + flag:<14s}{n:>7d}{med:>10.4f}{p99:>10.4f}{mx:>10.4f}")

    allD = np.concatenate([r[3 - 3] if False else per_type[r[1]][r[0]] for r in rows])
    floor = float(np.percentile(allD, 99))
    print(f"\nphysiological SHAPE floor (99th pct over everything): 1-cos = {floor:.4f}")

    # between-type distances at the same positions
    banks = {}
    for nm, kind, *_rest in rows:
        banks[(nm, kind)] = _rest[-1]
    keys = sorted(banks)
    cross = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i][0] != keys[j][0] or keys[i][1] == keys[j][1]:
                continue
            for pos in set(banks[keys[i]]) & set(banks[keys[j]]):
                A, B = banks[keys[i]][pos], banks[keys[j]][pos]
                da, db = mv._dirs(A), mv._dirs(B)
                cross.append(1.0 - (da @ db.T).ravel())
    if cross:
        cross = np.concatenate(cross)
        print(f"between cell TYPES, same morphology and position: median "
              f"{np.median(cross):.4f}  5th pct {np.percentile(cross, 5):.4f}")
        print(f"separation ratio (type median / within-cell p99): "
              f"{np.median(cross)/max(floor,1e-9):.1f}x")

    thr = 1.0 - args.threshold
    print(f"\nAgainst an operating cosine threshold of {args.threshold:.2f} "
          f"(1-cos = {thr:.3f}):")
    print(f"  physiology needs at most {floor:.4f}; the remaining {thr - floor:+.4f} "
          f"is budget for\n  template-estimation noise, NOT for physiology.  If the "
          "sort's templates are\n  estimated from enough spikes that their own cosine "
          "error is below that,\n  the threshold is looser than anything the cell can "
          "justify and is a\n  candidate cause of intra-chunk over-merging.")
    if any(k in mca.INCOMPLETE for _, k, *_ in rows):
        print("\n* these types omit Ca / KCa (and where listed HCN, KvM): the AHP is "
              "outside\n  the 1.3 ms window, but Ca accumulation across a burst is a "
              "real shape-variance\n  source that is MISSING here, so the floor above "
              "is a LOWER bound.")
    if args.out:
        np.savez_compressed(args.out, floor=floor,
                            **{f"{nm}|{kind}": per_type[kind][nm] for nm, kind, *_ in rows})
        print(f"\nwrote {args.out}")


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

    e = _probe(_common(sub.add_parser("envelope",
                                      help="build the physiological merge envelope")))
    e.add_argument("--patterns",
                   default="single,burst4_4,burst4_6,burst3_10,tonic_50_5,recover_4_150",
                   help="ISI patterns to pool")
    e.add_argument("--positions", default="20/25,40/30,60/45",
                   help="probe placements as depth/lateral in um")
    e.add_argument("--na-ar", default="1.0,0.5", dest="na_ar",
                   help="Na slow-inactivation floors to pool")
    e.add_argument("--plateaus", default="0.0,1.0",
                   help="burst plateau currents (nA); a stand-in for dendritic Ca")
    e.add_argument("--detect", type=float, default=50.0,
                   help="detection threshold in uV — this BOUNDS the envelope")
    e.add_argument("--v-thresh", type=float, default=0.0, dest="v_thresh",
                   help="somatic mV a stimulus must reach to count as a spike")
    e.add_argument("--q", type=float, default=0.99, help="envelope quantile")
    e.add_argument("--nbin", type=int, default=8)
    e.set_defaults(func=cmd_envelope, t_stop=None)

    g = sub.add_parser("gate", help="apply an envelope to candidate merge pairs")
    g.add_argument("--envelope", required=True, help="npz written by `envelope`")
    g.add_argument("--templates", required=True,
                   help="npz holding a (nunit, nsamp, nchan) template array")
    g.add_argument("--key", default=None, help="array name inside --templates")
    g.add_argument("--labels", default=None, help="array of unit labels inside --templates")
    g.set_defaults(func=cmd_gate)

    sh = _probe(_common(sub.add_parser("shape",
                                       help="shape-only variance at matched amplitude "
                                            "(the intra-chunk over-merge bound)")))
    sh.add_argument("--types", default="pyramidal,pvbasket,bistratified,ngf",
                    help="CA1 cell types from morpho_chan_ca1.CA1_TYPES")
    sh.add_argument("--patterns", default="single,burst4_4,burst4_6,burst3_10,tonic_50_5")
    sh.add_argument("--positions", default="20/25,40/30,60/45")
    sh.add_argument("--drive-pathways", default="ca3cell,eccell", dest="drive_pathways",
                    help="afferent pathways to vary dendritic state with ('' for none)")
    sh.add_argument("--drive-fractions", default="0.01,0.02", dest="drive_fractions")
    sh.add_argument("--match", type=float, default=1.05,
                    help="max amplitude ratio counted as 'matched amplitude'")
    sh.add_argument("--threshold", type=float, default=0.90,
                    help="the sort's operating cosine threshold, for comparison")
    sh.add_argument("--na-ar", type=float, default=0.5, dest="na_ar")
    sh.add_argument("--detect", type=float, default=50.0)
    sh.add_argument("--v-thresh", type=float, default=0.0, dest="v_thresh")
    sh.set_defaults(func=cmd_shape)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
