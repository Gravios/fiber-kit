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
    from . import morpho_validate as mvd
    from . import morpho_features as mf
    from . import morpho_localize as mlz
    from . import neuro_io as nio
    from . import morpho_fetch as mfe
except ImportError:
    import morpho_geom as mg, morpho_cable as mc, morpho_eap as me
    import morpho_input as mi, morpho_archetype as ma
    import morpho_envelope as mv
    import morpho_chan_ca1 as mca
    import morpho_validate as mvd
    import morpho_features as mf
    import morpho_localize as mlz
    import neuro_io as nio
    import morpho_fetch as mfe


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
    if getattr(args, "probe", None):
        ch = [int(v) for v in args.channels.split(",")] if args.channels else None
        if ch is None:
            raise SystemExit("--probe requires --channels (site ids are never inferred "
                             "from index arithmetic; that is what the probe file is for)")
        try:
            xy = me.group_geometry(args.probe.split(",")[0], ch)
        except Exception:
            xy = me.load_probe(args.probe.split(","), ch)
        # Re-reference to the group's own site 0.  A .probe file gives absolute
        # coordinates on the array -- group 5 of a Buzsaki64L sits at x ~ 1000 um
        # -- and the cell is simulated at the origin, so using them unshifted puts
        # the electrode a millimetre away and every spike falls below detection.
        # Within-group geometry is the only part that matters for one group.
        return np.asarray(xy, float) - np.asarray(xy, float)[0]
    return me.staggered_octrode(n=args.nchan)


def _post(args, W):
    """Push simulated footprints into the recorded feature space, if asked.

    A cosine threshold calibrated on raw footprints is NOT the threshold that
    applies to stderiv features: the transform is linear but not orthogonal, so
    it does not preserve angles.  Any comparison against a session whose .spk and
    .fet are stderiv has to happen after this.
    """
    spec = getattr(args, "sdiff_pairs", None)
    if not spec:
        return W
    sets = me.parse_sdiff_sets(spec)
    return me.stderiv(np.asarray(W, float), sets, drop_last=bool(getattr(args, "drop_last", False)))


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
    w = _post(args, out["wave"])
    return w, me.metrics(w, xy[:w.shape[1]], sr=args.sr)


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
                            bank.setdefault((dy, lat), []).append(_post(args, W))
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


# ── validate ────────────────────────────────────────────────────────────────
def cmd_validate(args):
    """Confront the model with a curated sort."""
    s = mvd.Sort(args.base, args.group, args.nsamp, args.nchan,
                 variant=args.variant, tag=args.tag)
    sr = args.sr
    sizes = s.sizes()
    print(f"group {args.group}: {len(s.res)} spikes, {len(sizes)} clusters")
    for k, (path, var) in s.provenance.items():
        print(f"  {k:<4s}{os.path.basename(path):<62s}[{var}]")

    main = args.main
    if main is None:
        main = max(sizes, key=lambda k: sizes[k])
        print(f"\nmain cluster not given; using the largest, {main} "
              f"({sizes[main]} spikes)")
    frags = ([int(v) for v in args.fragments.split(",")] if args.fragments else
             [k for k in sorted(sizes) if k != main
              and abs(k - main) <= args.neighbours and sizes[k] >= args.min_spikes])
    Tm = s.template(main)
    tb = mvd.time_budget(s, main, args.nblock)
    budget = float(np.nanmax(tb))
    nf_main = mvd.split_half_noise(s, main)
    ref0 = mvd.refractory(s, [main], sr, args.ref_ms)

    print(f"\nmain {main}: n={sizes[main]}  p2p={mvd.p2p(Tm):.0f}  "
          f"split-half {nf_main:.4f}  ISI<{args.ref_ms:.0f}ms {100*ref0:.3f}%")
    print(f"time budget across {args.nblock} blocks of {main}: max 1-cos "
          f"{budget:.4f}  (the within-unit variation this sort already accepts)")

    print(f"\n{'clu':>6s}{'n':>7s}{'1-cos':>9s}{'noise':>8s}{'d/noise':>8s}"
          f"{'vs budget':>10s}{'lat%':>7s}{'chance%':>8s}{'merge ISI%':>11s}"
          f"{'t5-t95 min':>12s}")
    rows = []
    for k in frags:
        if sizes.get(k, 0) < 8:
            continue
        T = s.template(k)
        d = mvd.cos_dist(T, Tm)
        nf = mvd.split_half_noise(s, k)
        lat, ch = mvd.latency_enrichment(s, main, k, sr, args.window_ms)
        rf = mvd.refractory(s, [main, k], sr, args.ref_ms)
        t = s.minutes(k, sr)
        lo, hi = np.percentile(t, [5, 95])
        print(f"{k:>6d}{sizes[k]:>7d}{d:>9.4f}{nf:>8.4f}{d/max(nf,1e-9):>8.1f}"
              f"{d/max(budget,1e-9):>10.1f}{100*lat:>7.1f}{100*ch:>8.1f}"
              f"{100*rf:>11.3f}{f'{lo:.0f}-{hi:.0f}':>12s}")
        rows.append((k, sizes[k], d, nf, lat, ch, rf, lo, hi))

    if args.recovery:
        ks = [main] + [r[0] for r in rows]
        try:
            rec, base, niso = mvd.recovery_curve(s, ks, sr)
        except ValueError as e:
            print(f"\nrecovery: {e}")
        else:
            print(f"\nrecovery over {main} + {len(rows)} fragments "
                  f"(normalised to {niso} spikes with ISI > 200 ms, p2p {base:.0f})")
            print(f"{'preceding ISI':>16s}{'n':>8s}{'amp ratio':>11s}{'1-cos':>9s}")
            for lo, hi, n, r, dd in rec:
                lab = f"{lo:.0f}-{hi:.0f} ms" if np.isfinite(hi) else f">{lo:.0f} ms"
                print(f"{lab:>16s}{n:>8d}{r:>11.3f}{dd:>9.4f}")
            print("\nIf amp ratio is flat, adaptation is not moving this unit's "
                  "amplitude and\na gate built on predicted amplitude ratio does not "
                  "apply to it.  Compare the\n1-cos column with the fragment "
                  "distances above: firing state has to account\nfor them, or "
                  "something else does.")

    print("\nNo verdict is printed on purpose.  The refractory column has almost no "
          "power at\na few hundred spikes; latency enrichment is shared by a burst "
          "continuation and by\na synaptically driven partner; and a fragment inside "
          "its own split-half noise has\nnot been shown to deviate at all.  These are "
          "inputs to a decision, not one.")
    if args.out:
        np.savez_compressed(args.out, main=main, budget=budget,
                            clu=np.array([r[0] for r in rows]),
                            n=np.array([r[1] for r in rows]),
                            cosd=np.array([r[2] for r in rows]),
                            noise=np.array([r[3] for r in rows]),
                            lat=np.array([r[4] for r in rows]),
                            chance=np.array([r[5] for r in rows]),
                            refrac=np.array([r[6] for r in rows]),
                            t5=np.array([r[7] for r in rows]),
                            t95=np.array([r[8] for r in rows]), tb=tb)
        print(f"\nwrote {args.out}")


# ── span ────────────────────────────────────────────────────────────────────
def cmd_span(args):
    """Model-predicted within-cell span, per cell type, in the session's features.

    Reported as radius / |template|, which is dimensionless.  That is not
    fastidiousness: the model produces microvolts and the recording produces
    ADU, so an absolute comparison would be reporting the acquisition gain.  The
    ratio is also the quantity that should be constant across cells if the span
    is a fixed fraction of amplitude, and constant in ABSOLUTE terms if it is
    not -- so publishing the ratio lets the reader check which.
    """
    pca = mf.load_pca(args.pca)
    xy = _sites(args)
    sets = me.parse_sdiff_sets(args.sdiff_pairs) if args.sdiff_pairs else None
    pats = args.patterns.split(",")
    pw = {p.pre: p for p in mi.load_table(post="pyramidalcell")}
    drives = [None] + [(k, f) for k in args.drive_pathways.split(",") if k
                       for f in [float(x) for x in args.drive_fractions.split(",")]]
    print(f"{'cell':<20s}{'type':<14s}{'states':>7s}{'radius':>9s}{'|templ|':>9s}"
          f"{'r/|t|':>8s}{'rel':>7s}  per-channel var fraction")
    base = None
    for spec in args.cells.split(","):
        # Split the CA1 type off the RIGHT: a spec is "<morphology>[:<type>]" and
        # the morphology may itself contain colons ("template:/path/x.hoc"), so
        # partitioning from the left eats the path.  Only strip the suffix when it
        # actually names a type, or a plain path with no type would be truncated.
        name, kind = spec, "pyramidal"
        head, sep, tail = spec.rpartition(":")
        if sep and tail in mca.CA1_TYPES:
            name, kind = head, tail
        c = _load_cell_laminar(name, args)
        W = []
        for pat in pats:
            times = mv.train_times(pat)
            nt = int(round((max(times) + 6.0) / args.dt))
            for d in drives:
                drive = None
                if d is not None and d[0] in pw:
                    drive = mi.Drive([pw[d[0]]], c, args.dt, nt,
                                     {d[0]: np.array([max(times[0] - 3.0, 0.5)])},
                                     active_fraction=d[1],
                                     rng=np.random.default_rng(args.seed))
                    if len(drive) == 0:
                        drive = None
                cell = mc.Cell(c, mca.biophys(kind))
                _, im, v = mv.simulate_train(cell, times, dt=args.dt,
                                             stim_amp=args.stim, drive=drive)
                sites = me.sites_3d(xy - np.array([0.0, args.depth]), z=args.lateral)
                Wk, _ = mv.train_footprints(im, c, sites, times, args.dt, sr=args.sr,
                                            nsamp=args.nsamp, peak=args.peak,
                                            sigma=args.sigma, v=v, v_thresh=0.0)
                if len(Wk):
                    W.append(Wk)
        if not W:
            print(f"{os.path.basename(name):<20s}{kind:<14s}   no spikes survived extraction")
            continue
        W = np.concatenate(W)
        F = mf.to_features(W, pca, sdiff_sets=sets)
        mu = F.mean(0)
        R = float(np.sqrt(((F - mu) ** 2).sum(1).mean()))
        q = R / max(float(np.linalg.norm(mu)), 1e-30)
        if base is None:
            base = q
        vc = ((F - mu) ** 2).mean(0).reshape(pca.nch, pca.ncomp).sum(1)
        vc = vc / max(vc.sum(), 1e-30)
        print(f"{os.path.basename(name):<20s}{kind:<14s}{len(W):>7d}{R:>9.1f}"
              f"{np.linalg.norm(mu):>9.0f}{q:>8.3f}{q/base:>7.2f}  {np.round(vc, 3)}")
    print("\nThis is the PHYSIOLOGICAL span only — the model has no recording noise,\n"
          "so it is a lower bound on a cluster's observed radius, and the per-channel\n"
          "fractions are the discriminator: physiological variance CONCENTRATES on the\n"
          "channels carrying the state-dependent current, while additive noise spreads\n"
          "uniformly.  Compare against `fiber-morpho validate` on a curated unit.")


def _load_cell_laminar(name, args):
    """Load a morphology, without re-orienting a cable template.

    A cable template's layout is already laminar (morpho_geom places it from the
    section names), so orient()'s axis search would stand a symmetric cell
    upright.  Reconstructions carry no laminar convention and do need it.
    """
    if name.startswith("template:"):
        secs = mg.load_cable_template(name.split(":", 1)[1])
        return mg.orient(mg.compartmentalize(secs, d_lambda=args.d_lambda),
                         axis=(0.0, 1.0, 0.0))
    return _load_cell(name, args.d_lambda, args.max_comp)


# ── localize ────────────────────────────────────────────────────────────────
def cmd_localize(args):
    """Fit a position to every cluster and atom; write a sidecar.

    The position table is built once and CACHED, because it depends only on the
    probe geometry and the morphology set -- not on the sort.  A rerun after
    re-clustering reuses it, which is what makes this cheap enough to sit in a
    pipeline rather than be run by hand once.
    """
    import glob
    xy = me.group_geometry(args.probe, [int(v) for v in args.channels.split(",")])
    if args.table and os.path.exists(args.table):
        tab = mlz.PositionTable.load(args.table)
        print(f"table: {len(tab)} positions (cached, {args.table})")
    else:
        # A directory, a glob, or a comma-separated list.  --max-morph used to
        # take the first N ALPHABETICALLY, which silently selected an unrelated
        # cell type and produced RMSE 0.22 instead of 0.01 with no error --
        # the caller must be able to say WHICH morphologies, not how many.
        # Each comma-separated entry is resolved INDEPENDENTLY as a directory,
        # a glob, or a file.  The first version treated every comma entry as a
        # file path, so `--morphologies a/,b/` produced two directory paths that
        # passed the existence check and only failed later, as two entries with
        # an empty basename and no biophysics rule.
        paths = []
        for entry in [e.strip() for e in args.morphologies.split(",") if e.strip()]:
            if os.path.isdir(entry):
                got = sorted(glob.glob(os.path.join(entry, "*.swc")))
                if not got:
                    raise SystemExit(f"[localize] no .swc files under {entry}")
            elif any(ch in entry for ch in "*?["):
                got = sorted(glob.glob(entry))
                if not got:
                    raise SystemExit(f"[localize] glob matched nothing: {entry}")
            elif os.path.exists(entry):
                got = [entry]
            else:
                raise SystemExit(f"[localize] no such morphology, directory or glob: {entry}")
            paths.extend(got)
        seen, uniq = set(), []
        for q in paths:
            r = os.path.realpath(q)
            if r not in seen:
                seen.add(r); uniq.append(q)
        if len(uniq) != len(paths):
            print(f"  [note] {len(paths)-len(uniq)} duplicate path(s) dropped")
        paths = uniq
        if not paths:
            raise SystemExit("[localize] --morphologies resolved to nothing")
        if args.max_morph:
            paths = paths[:args.max_morph]
        if not paths:
            raise SystemExit(f"[localize] no .swc under {args.morphologies}")
        # Per-morphology biophysics.  A single --kind across a mixed set runs
        # pyramidal reconstructions with interneuron conductances, which yields
        # a plausible-looking waveform that is simply wrong and which nothing
        # downstream can flag.  The manifest already records cell_type, so the
        # preset is read from it rather than asserted on the command line.
        kinds = {q: args.kind for q in paths}
        mans = list(args.manifest or [])
        if mans:
            # Repeatable, because a morphology set fetched per cell class has one
            # manifest per directory.  Later manifests win on a name collision.
            km, unk = {}, None
            for mp in mans:
                if not os.path.exists(mp):
                    raise SystemExit(f"[localize] no such manifest: {mp}")
                k1, u1 = mfe.kinds_from_manifest(
                    mp, paths, default=(None if args.strict_kind else args.kind))
                km.update(k1)
                unk = u1 if unk is None else [x for x in unk if x[0] in {y[0] for y in u1}]
            unk = unk or []
            kinds.update(km)
            if unk:
                print(f"  [warn] {len(unk)} morphologies have no biophysics rule; "
                      + ("refusing (--strict-kind)" if args.strict_kind
                         else f"using --kind {args.kind}"))
                for n_, t_ in unk[:5]:
                    print(f"    {n_ or '<unnamed>'}: {t_[:60]}")
                if args.strict_kind:
                    raise SystemExit("[localize] refusing to simulate with a guessed preset")
            rows_m = {}
            for mp in mans:
                rows_m.update({r["neuron_name"]: r.get("cell_type", "")
                               for r in mfe.read_manifest(mp)})
            approx = [(os.path.basename(q), mfe.approximated_as(rows_m.get(
                os.path.basename(q).replace(".CNG.swc", "").replace(".swc", ""), "")))
                      for q in paths]
            approx = [(n_, a_) for n_, a_ in approx if a_]
            if approx:
                print(f"  [note] {len(approx)} morphologies use an APPROXIMATE preset "
                      "(no exact match in CA1_TYPES): "
                      + ", ".join(sorted({a_ for _, a_ in approx})))
            import collections as _c
            print("  biophysics: " + ", ".join(
                f"{k}={v}" for k, v in sorted(_c.Counter(kinds.values()).items())))
        t0 = time.time()
        tab = mlz.build_table(paths, xy,
                              rotations=[float(v) for v in args.rotations.split(",")],
                              depths=np.arange(args.depth_min, args.depth_max + 1e-9, args.step),
                              laterals=np.arange(args.lat_min, args.lat_max + 1e-9, args.step),
                              kinds=kinds,
                              progress=lambda k, n, lab: print(f"  [{k}/{n}] {lab}"))
        print(f"table: {len(tab)} positions from {len(paths)} morphologies "
              f"in {time.time()-t0:.0f}s")
        if args.table:
            tab.save(args.table); print(f"  cached to {args.table}")

    # Localisation MUST use the raw waveform.  The channel difference removes
    # the common mode, which is precisely the amplitude-distance relationship a
    # position fit reads -- the same reason read_pca refuses to fall back to
    # stderiv.  Sort() resolves .spk by the clustering variant, so the raw file
    # is opened explicitly here rather than inherited.
    s = mvd.Sort(args.base, args.group, args.nsamp, args.nchan,
                 variant=args.variant, tag=args.tag)
    rs = nio.resolve_any(args.base, "spk", args.group, preferred=args.spk_variant)
    if not rs.found:
        raise SystemExit(f"[localize] no .spk.{args.spk_variant} for group {args.group}; "
                         f"localisation cannot use the transformed waveform")
    if rs.variant != args.spk_variant:
        print(f"  [warn] asked for .spk.{args.spk_variant}, resolved {rs.variant}")
    s.spk = nio.open_spk_file(rs.path, args.nsamp, args.nchan)
    print(f"  waveforms: {os.path.basename(rs.path)} [{rs.variant}]")
    sr = args.sr
    clu = s.clu
    clc = None
    if args.atoms:
        cpath = nio.session_path(args.base, "clc", args.group, args.variant, args.tag)
        if os.path.exists(cpath):
            clc = np.fromfile(cpath, dtype=np.int32)[1:]
            if len(clc) != len(clu):
                print(f"  [warn] .clc has {len(clc)} entries for {len(clu)} spikes; "
                      "ignoring atoms"); clc = None
        else:
            print(f"  [warn] no .clc at {cpath}; cluster-level only")

    tmin = s.res.astype(np.float64) / sr / 60.0
    rows = []
    sizes = s.sizes()
    for k in sorted(sizes):
        if sizes[k] < args.min_spikes:
            continue
        i = s.idx(k)
        w = np.asarray(s.spk[i], np.float64)
        f = tab.fit(mlz.profile(w))
        fl = mlz.split_half_floor(tab, w, n_rep=args.n_rep)
        rows.append(dict(cluster=k, atom=-1, chunk=-1, n=len(i), **f, floor=fl))
        if clc is None:
            continue
        for a in np.unique(clc[i]):
            j = np.flatnonzero(clc == a)
            if len(j) < args.min_spikes_atom:
                continue
            wa = np.asarray(s.spk[j], np.float64)
            fa = tab.fit(mlz.profile(wa))
            rows.append(dict(cluster=k, atom=int(a),
                             chunk=int(np.median(tmin[j]) // args.chunk_min),
                             n=len(j), **fa,
                             floor=mlz.split_half_floor(tab, wa, n_rep=args.n_rep)))

    cols = ["cluster", "atom", "chunk", "n", "morphology", "rot", "depth",
            "lateral", "rmse", "floor"]
    # NOT `.pos`: in a hippocampus session that extension already means the
    # animal's tracked position, and a file called <session>.pos sitting beside
    # <session>.whl would be read as behaviour by anyone scanning the directory.
    # `fk-cpos` is cluster position, in the type slot the convention expects, so
    # session_path builds it and resolve_any can find it.
    out = args.out or nio.session_path(args.base, "fk-cpos", args.group,
                                       variant=args.variant, tag=args.tag)
    with open(out, "w") as fh:
        fh.write("# positions fitted by fiber-morpho localize\n")
        fh.write(f"# probe={args.probe} channels={args.channels} "
                 f"table={len(tab)} morphologies\n")
        fh.write("# atom=-1 is the whole cluster; floor is that population's own "
                 "split-half resolution in um\n")
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(
                ("%.4g" % r[c] if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    nc = sum(1 for r in rows if r["atom"] < 0)
    print(f"\nwrote {out}: {nc} clusters, {len(rows)-nc} atoms")
    med = np.median([r["rmse"] for r in rows if r["atom"] < 0])
    mfl = np.median([r["floor"] for r in rows if r["atom"] < 0 and np.isfinite(r["floor"])])
    print(f"  median cluster fit RMSE {med:.4f} ; median resolution {mfl:.1f} um")

    if not args.split_scan or clc is None:
        return 0
    print("\nwithin-chunk atom splits (two positions in one time window):")
    hit = 0
    for k in sorted(sizes):
        per = {}
        for r in rows:
            if r["cluster"] == k and r["atom"] >= 0:
                per.setdefault(r["chunk"], []).append(r)
        for ch, v in sorted(per.items()):
            if len(v) < 2:
                continue
            v = sorted(v, key=lambda r: -r["n"])[:2]
            sep = float(np.hypot(v[0]["depth"] - v[1]["depth"],
                                 v[0]["lateral"] - v[1]["lateral"]))
            fl = max([r["floor"] for r in v if np.isfinite(r["floor"])] or [0.0])
            thr = max(3 * fl, args.split_floor)
            if max(v[0]["rmse"], v[1]["rmse"]) > args.max_rmse:
                continue
            if sep > thr:
                hit += 1
                print("  clu %-6d chunk %-4d atoms %d/%d  sep %5.1f um  thr %4.1f"
                      % (k, ch, v[0]["atom"], v[1]["atom"], sep, thr))
    print(f"  {hit} chunk(s) flagged — those clusters hold more than one cell")
    return 0


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
    p.add_argument("--sdiff-pairs", default=None, dest="sdiff_pairs",
                   help="session sdiffPairs; applies the stderiv transform so simulated "
                        "footprints live in the recorded feature space")
    p.add_argument("--drop-last", type=int, default=0, dest="drop_last",
                   help="also drop the last channel, as SDIFF_PASS does at the PCA stage")
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

    va = sub.add_parser("validate", help="confront the model with a curated sort")
    va.add_argument("--base", required=True, help="session base path (no extension)")
    va.add_argument("--group", type=int, required=True)
    va.add_argument("--variant", default="", help="method token, e.g. stderiv.C5.D34")
    va.add_argument("--tag", default="", help="fiber stage, e.g. anchor_linked")
    va.add_argument("--nsamp", type=int, default=42)
    va.add_argument("--nchan", type=int, default=8)
    va.add_argument("--sr", type=float, default=32552.0)
    va.add_argument("--main", type=int, default=None,
                    help="the accepted cluster; default is the largest")
    va.add_argument("--fragments", default=None,
                    help="explicit candidate ids; default is neighbours by id")
    va.add_argument("--neighbours", type=int, default=16,
                    help="id distance from --main to sweep when --fragments is absent")
    va.add_argument("--min-spikes", type=int, default=8, dest="min_spikes")
    va.add_argument("--nblock", type=int, default=6, help="time blocks for the budget")
    va.add_argument("--window-ms", type=float, default=10.0, dest="window_ms")
    va.add_argument("--ref-ms", type=float, default=2.0, dest="ref_ms")
    va.add_argument("--recovery", type=int, default=1)
    va.add_argument("--out", default=None)
    va.set_defaults(func=cmd_validate)

    sp = _probe(_common(sub.add_parser("span",
                                       help="model-predicted within-cell span per cell type")))
    sp.add_argument("--pca", required=True, help="the session's PCAE .pca basis")
    sp.add_argument("--patterns", default="single,burst4_4,burst4_6,burst3_10,tonic_50_5")
    sp.add_argument("--drive-pathways", default="ca3cell,eccell", dest="drive_pathways")
    sp.add_argument("--drive-fractions", default="0.01,0.02", dest="drive_fractions")
    sp.add_argument("--depth", type=float, default=40.0)
    sp.add_argument("--lateral", type=float, default=30.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_span)

    lz = sub.add_parser("localize", help="fit positions to clusters/atoms; write a sidecar")
    lz.add_argument("--base", required=True); lz.add_argument("--group", type=int, required=True)
    lz.add_argument("--variant", default=""); lz.add_argument("--tag", default="")
    lz.add_argument("--probe", required=True); lz.add_argument("--channels", required=True)
    lz.add_argument("--morphologies", default=None,
                    help="directory, glob, or comma-separated list of .swc files")
    lz.add_argument("--table", default=None, help="cache the position table here (reused)")
    lz.add_argument("--kind", default="pvbasket",
                    help="fallback preset; --manifest overrides it per morphology")
    lz.add_argument("--manifest", action="append", default=None,
                    help="morphologies.tsv; repeatable, one per morphology directory. "
                         "Biophysics is inferred from its cell_type column")
    lz.add_argument("--strict-kind", action="store_true", dest="strict_kind",
                    help="refuse rather than fall back to --kind for unmapped cells")
    lz.add_argument("--spk-variant", default="standard", dest="spk_variant",
                    help="waveform variant to localise on; MUST be untransformed")
    lz.add_argument("--rotations", default="0,90,180,270")
    lz.add_argument("--depth-min", type=float, default=0.0, dest="depth_min")
    lz.add_argument("--depth-max", type=float, default=200.0, dest="depth_max")
    lz.add_argument("--lat-min", type=float, default=2.5, dest="lat_min")
    lz.add_argument("--lat-max", type=float, default=70.0, dest="lat_max")
    lz.add_argument("--step", type=float, default=2.5)
    lz.add_argument("--max-morph", type=int, default=0, dest="max_morph",
                    help="cap the count AFTER selection; it does not choose which")
    lz.add_argument("--nsamp", type=int, default=42); lz.add_argument("--nchan", type=int, default=8)
    lz.add_argument("--sr", type=float, default=32552.0)
    lz.add_argument("--min-spikes", type=int, default=150, dest="min_spikes")
    lz.add_argument("--min-spikes-atom", type=int, default=120, dest="min_spikes_atom")
    lz.add_argument("--n-rep", type=int, default=4, dest="n_rep")
    lz.add_argument("--chunk-min", type=float, default=18.0, dest="chunk_min")
    lz.add_argument("--atoms", type=int, default=1)
    lz.add_argument("--split-scan", type=int, default=1, dest="split_scan")
    lz.add_argument("--split-floor", type=float, default=7.5, dest="split_floor")
    lz.add_argument("--max-rmse", type=float, default=0.06, dest="max_rmse",
                    help="skip the split test where the model does not fit; a large\nseparation between two bad fits means nothing")
    lz.add_argument("--out", default=None,
                    help="default <base>.fk-cpos.<variant>.<group>[.<tag>]")
    lz.set_defaults(func=cmd_localize)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
