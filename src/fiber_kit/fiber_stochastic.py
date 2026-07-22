#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_stochastic.py — ENSEMBLE / consensus fibering by resampling (diagnostic).
#
#  PROPOSAL under test (not yet a pipeline stage):
#    Instead of clustering each chunk once, draw K random SUBSAMPLES of the chunk
#    (fraction `frac`), fiber each draw with the ORDINARY pipeline clusterer, then
#    find the fibers that RECUR across draws — the consensus set — and (optionally)
#    PEEL them: freeze the recurring fibers, remove their spikes, and re-run the
#    ensemble on the residual to expose the next tier.  Finally, assign every spike
#    to the frozen consensus set.
#
#  This module does NOT change the production path.  It reuses the real per-chunk
#  worker (_init_chunk_worker / _process_chunk from fiber_session — the SAME .fil
#  whitener + cluster_chunk_fine + fiber_geom), so what it measures is the real
#  clusterer's behaviour under resampling, not a stand-in.
#
#  WHY a diagnostic first (see the sandbox analysis that motivated this):
#    - Dominance is chunk-dependent: peeling helps only where a few fibers carry
#      the chunk (top-5 ~44% in a dominant chunk vs ~16-20% in flat ones).
#    - Starvation is the risk: a 50% draw pushes 50-70% of fibers below min_group.
#      So `frac` defaults HIGH (0.8) and is the first thing to sweep.
#    - Templates are stable from a fraction of spikes (half-vs-full corr ~0.985 at
#      60-90 spk), so any instability lives in the clusterer's split/merge
#      DECISIONS — which is exactly what this harness is built to observe.
#
#  OUTPUT (so the fiber space itself can be studied, not just the summary):
#    <base>.fiberens.<elec>.npz  — every fiber from every draw + full geometry +
#    consensus assignment + per-instance stability, near-miss, and peel-round
#    provenance.  Schema mirrors the production .fibers.npz row-store, with extra
#    columns (draw, frac, peel_round, consensus_gid, match_corr, match_corr2,
#    recovery_freq, ...) documented in _dump_ensemble below.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import numpy as np

try:
    from . import neuro_io as nio
    from . import fiber_lib as fl
    from . import fiber_session as fsess
except ImportError:
    import neuro_io as nio
    import fiber_lib as fl
    import fiber_session as fsess

_POOL_CFG = None                     # stashed chunk-worker cfg so parallel draws can init pool workers


# ── template distance: the label-free way to say "same fiber across two draws" ──
def _template_vec(geom):
    """Flatten a fiber's template to a unit vector for correlation matching.  Label
    ids are arbitrary per draw, so fibers are matched by SHAPE, not by id."""
    t = np.asarray(geom["template"], float).ravel()
    if t.size == 0 or not np.all(np.isfinite(t)):
        return np.zeros(t.size)                     # degenerate template -> matches nothing (corr 0)
    t = t - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else np.zeros_like(t)


def _draw_indices(n, frac, rng):
    """One resample of a chunk: a random `frac` subset WITHOUT replacement (bootstrap
    with replacement collapses duplicate spikes onto identical features and biases the
    density clusterer, so we subsample)."""
    k = max(2, int(round(frac * n)))
    return np.sort(rng.choice(n, size=k, replace=False))


# ── consensus: connected components of the "same fiber" graph over all draw fibers ──
def _consensus(all_geoms, match_thr, link="average"):
    """all_geoms: list over draws of lists of geom dicts.  Returns, for each (draw,
    local index), a consensus id, plus per-consensus recovery frequency and the
    per-instance best/second-best match into OTHER draws.

    match_corr  = best template correlation into a fiber from any OTHER draw.
    match_corr2 = best correlation into a fiber that ends up in a DIFFERENT consensus
                  component -- i.e. the nearest rival, the real merge-proneness signal.

    link: how instances are grouped into consensus fibers from the pairwise template
          correlation.  'single' is transitive union-find (A~B, B~C => A,B,C together
          even if A and C are anticorrelated) -- it CHAINS distinct co-located sub-modes
          (e.g. a narrow and a broad spike bridged by intermediate-width instances) into
          one component, undercounting real units.  'average' (default) and 'complete'
          are agglomerative: two groups merge only if their AVERAGE (resp. MINIMUM)
          cross-correlation exceeds match_thr, so a chain through intermediates cannot
          weld anticorrelated modes.  On g5 chunk 16 single-linkage welded ~2x as many
          real sub-modes into each 'stable' fiber as average-linkage keeps apart."""
    inst = [(d, i, _template_vec(g)) for d, gs in enumerate(all_geoms) for i, g in enumerate(gs)]
    m = len(inst)
    ndraw = len(all_geoms)
    if m == 0:
        return inst, np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0)

    V = np.stack([v for _, _, v in inst])          # (m, D) unit templates
    draw_id = np.array([d for d, _, _ in inst])
    C = V @ V.T                                     # one matmul, not m^2 Python dots
    same_draw = draw_id[:, None] == draw_id[None, :]
    np.fill_diagonal(same_draw, True)
    C_other = np.where(same_draw, -np.inf, C)       # mask self and same-draw pairs

    if link == "single":
        # transitive union-find over pairs above threshold (original behaviour; chains sub-modes)
        parent = list(range(m))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ii, jj = np.where(np.triu(C_other >= match_thr, k=1))
        for a, b in zip(ii.tolist(), jj.tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        comp = np.array([find(x) for x in range(m)])
    else:
        # agglomerative average/complete linkage: merge the closest pair of GROUPS while their
        # linkage score is >= match_thr.  Groups never merge two instances from the same draw
        # (that pairing is -inf).  O(m^2) memory for the score matrix; the merge loop re-scans the
        # full matrix with argmax each step, so it is O(m^3) time -- measured ~1.5s at the realistic
        # worst case (m~2600 instances per chunk-round, nothing merging), which is not a bottleneck
        # (one _consensus call per chunk), so it is left simple rather than heap/cache-optimised.
        groups = [[i] for i in range(m)]
        # group-vs-group linkage matrix, seeded from the instance matrix
        G = C_other.copy()
        np.fill_diagonal(G, -np.inf)
        # for average linkage we track sum and count to recompute merged rows in O(m)
        cnt = np.ones(m)
        Ssum = np.where(np.isfinite(C_other), C_other, np.nan)  # same-draw stays nan -> excluded
        active = np.ones(m, bool)
        while True:
            # best mergeable pair among active groups
            Gm = np.where(active[:, None] & active[None, :], G, -np.inf)
            a, b = np.unravel_index(np.argmax(Gm), Gm.shape)
            if not np.isfinite(Gm[a, b]) or Gm[a, b] < match_thr:
                break
            # merge b into a
            if link == "complete":
                newrow = np.minimum(np.where(np.isfinite(G[a]), G[a], np.inf),
                                    np.where(np.isfinite(G[b]), G[b], np.inf))
                newrow = np.where(np.isfinite(newrow), newrow, -np.inf)
            else:  # average: weighted mean of the two groups' summed cross-corrs
                sa = Ssum[a] * cnt[a]; sb = Ssum[b] * cnt[b]
                with np.errstate(invalid="ignore"):
                    newrow = (np.nan_to_num(sa) + np.nan_to_num(sb)) / (cnt[a] + cnt[b])
                nanmask = np.isnan(Ssum[a]) & np.isnan(Ssum[b])
                newrow[nanmask] = np.nan
                Ssum[a] = newrow
                G[a] = np.where(np.isnan(newrow), -np.inf, newrow)
            if link == "complete":
                G[a] = newrow
            groups[a] = groups[a] + groups[b]
            cnt[a] += cnt[b]
            active[b] = False
            G[:, a] = G[a]; np.fill_diagonal(G, -np.inf)
        comp = np.empty(m, int)
        for gi, rows in enumerate(g for g, ok in zip(groups, active) if ok):
            for r in rows:
                comp[r] = gi

    draws_of = {}
    for x in range(m):
        draws_of.setdefault(comp[x], set()).add(int(draw_id[x]))
    freq = {c: len(ds) / ndraw for c, ds in draws_of.items()}
    order = sorted(freq, key=lambda c: -freq[c])
    remap = {c: i for i, c in enumerate(order)}
    cons_gid = np.array([remap[comp[x]] for x in range(m)], int)
    recov = np.array([freq[comp[x]] for x in range(m)])

    best = np.where(np.isfinite(C_other.max(1)), C_other.max(1), 0.0)
    # nearest rival: best corr into an instance of a DIFFERENT component
    diff_comp = comp[:, None] != comp[None, :]
    C_rival = np.where(same_draw | ~diff_comp, -np.inf, C)
    mx = C_rival.max(1)
    best2 = np.where(np.isfinite(mx), mx, 0.0)
    return inst, cons_gid, recov, best, best2


def _consensus_templates(inst, cons_gid, all_geoms):
    """Median template per consensus fiber, over all its instances (for assignment)."""
    C = cons_gid.max() + 1 if len(cons_gid) else 0
    out = []
    for c in range(C):
        members = [all_geoms[inst[x][0]][inst[x][1]]["template"] for x in np.flatnonzero(cons_gid == c)]
        out.append(np.median(np.stack(members), 0))
    return out


# ── the ensemble over one chunk (K draws), optionally over a residual index set ──
def _draw_worker(task):
    """Pool worker: run the real per-chunk clusterer on one pre-drawn subsample.  task is
    (c, ext_d, res_d, di) where di is the ext-relative index array the draw owns.  _process_chunk
    reads the module _CTX set by _init_chunk_worker (the pool initializer), exactly as in
    fiber_session's chunk pool, so this is the same validated code path.  Returns (geoms, di_members)
    where di_members[k] is the drawn spikes owned by the k-th fiber -- computed here so the parent does
    no per-draw work."""
    import fiber_kit.fiber_session as fsess
    c, ext_d, res_d, di = task
    _, _, lab, geoms, _, _ = fsess._process_chunk((c, ext_d, res_d))
    members = [di[np.flatnonzero(lab == gk)]
               for gk in sorted(set(int(x) for x in lab if x >= 0))]
    return geoms, members


def _ensemble_chunk(c, ext, res_ext, sub, ctx_process, ndraw, frac, rng, jobs=1, seedseq=None):
    """Run the real per-chunk worker on K subsamples of `sub` (indices into ext).
    Returns a list (per draw) of geom-lists.  Each draw reuses _process_chunk, so the
    real whitener + cluster_chunk_fine + fiber_geom run on the drawn spikes.

    The draws are independent, so with jobs>1 they run on a ProcessPoolExecutor (the SAME worker
    fiber_session's chunk pool uses).  Subsampling stays in the PARENT with per-draw seeds spawned from
    seedseq, so the draw set is deterministic regardless of jobs (a spawned SeedSequence per draw, not
    the shared mutable rng), and only the drawn indices cross to workers."""
    # spawn one reproducible seed per draw so serial and parallel give identical draws
    seeds = (seedseq.spawn(ndraw) if seedseq is not None
             else [None] * ndraw)
    tasks = []
    for k in range(ndraw):
        r = np.random.default_rng(seeds[k]) if seeds[k] is not None else rng
        di = sub[_draw_indices(len(sub), frac, r)]
        tasks.append((c, ext[di], res_ext[di], di))

    all_geoms = []
    if jobs <= 1:
        for c_, ext_d, res_d, di in tasks:
            geoms, members = _draw_worker((c_, ext_d, res_d, di))
            for g, mem in zip(geoms, members):
                g["_draw_members"] = mem
            all_geoms.append(geoms)
    else:
        # workers inherit _CTX via the pool initializer (set in run_stochastic); reuse the same one.
        from concurrent.futures import ProcessPoolExecutor
        cfg = _POOL_CFG
        nworkers = min(jobs, ndraw)
        with ProcessPoolExecutor(max_workers=nworkers,
                                 initializer=fsess._init_pool_worker, initargs=(cfg,)) as ex:
            for geoms, members in ex.map(_draw_worker, tasks):
                for g, mem in zip(geoms, members):
                    g["_draw_members"] = mem
                all_geoms.append(geoms)
    return all_geoms


def run_stochastic(a):
    """Entry point.  Resolves session params and configures the clusterer EXACTLY as
    fiber_session.main() does (same resolve_session_params + build_cf + build_masks),
    then runs the ensemble/peel diagnostic and dumps the full fiber space.  Touches no
    production output file."""
    try:
        from . import session_yaml as sy
        from . import fiber_pca as _fpca
    except ImportError:
        import session_yaml as sy
        import fiber_pca as _fpca

    cfg_s = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                      nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    a.base = cfg_s["base"]; a.elec = a.group
    a.ntotal = cfg_s["ntotal"]; a.nchan = cfg_s["nchan"]; a.nsamp = cfg_s["nsamp"]; a.sr = cfg_s["sr"]
    gch = np.array(cfg_s["channels"], int)
    mask = fl.build_masks(cfg_s["nsamp"], cfg_s["peak"]).full

    meth = "none" if a.no_fine else a.fine_method
    cluster_basis = None if a.no_cluster_basis else _fpca.read_cluster_basis(a.base, a.elec, a.method)
    cf = fsess.build_cf(a, meth, cluster_basis)          # identical clusterer config to production

    cfg = dict(base=a.base, elec=a.elec, fil=f"{a.base}.fil", ntotal=a.ntotal,
               nsamp=a.nsamp, nchan=a.nchan, sr=a.sr, min_group=a.min_group,
               gch=gch, mask=mask, cf=cf, gpu=a.gpu, no_whiten=getattr(a, "no_whiten", False))
    global _POOL_CFG
    _POOL_CFG = cfg                                   # so parallel-draw pool workers re-init the same ctx
    fsess._init_chunk_worker(cfg)
    process = fsess._process_chunk

    res = nio.read_res(a.base, a.elec)
    t_min = float(res.min())
    chunk_s = a.chunk_min * 60.0 * a.sr; ov_s = a.overlap_min * 60.0 * a.sr
    nchunks = int(np.ceil((res.max() - t_min) / chunk_s))
    rng = np.random.default_rng(a.stochastic_seed)
    ensemble_seedseq = np.random.SeedSequence(a.stochastic_seed)   # spawns deterministic per-draw seeds
    ejobs = int(getattr(a, "stochastic_jobs", 1))

    rows = []                      # every fiber instance, all draws, all peel rounds
    peel_log = []                  # per (chunk, round): frozen count, residual size, remaining
    # per-spike votes (ext-absolute res index -> Counter over consensus fibers / sub-modes), for the
    # optional clu/clc/clp triplet.  A spike is in ~frac of the draws and may land in different fibers
    # across them; the final label is the majority vote.
    vote_fiber = {}                # res_index -> {consensus_gid: count}
    vote_submode = {}              # res_index -> {(consensus_gid, submode): count}
    chunks = a.stochastic_chunks if a.stochastic_chunks else list(range(nchunks))

    import sys as _sys, time as _time
    def _plog(msg):                # progress -> stderr, flushed, so a long run is not silent
        print(msg, file=_sys.stderr, flush=True)
    _t0 = _time.time(); _done = 0
    _plog(f"fiber_stochastic: {len(chunks)} chunk(s), {a.stochastic_draws} draw(s)/round"
          + (f" x{ejobs} parallel" if ejobs > 1 else "")
          + f", up to {a.stochastic_peel_rounds} peel round(s), frac {a.stochastic_frac:g}, "
          f"link {a.stochastic_link}")

    for ci, c in enumerate(chunks):
        _tc = _time.time()
        lo_s = t_min + c * chunk_s; hi_s = t_min + (c + 1) * chunk_s
        ext = np.flatnonzero((res >= lo_s - ov_s) & (res < hi_s + ov_s))
        if len(ext) < 2 * a.min_group:
            _plog(f"  [{ci + 1:>3}/{len(chunks)}] chunk {c:>3}: {len(ext):>6} ext spikes — skipped (< {2 * a.min_group})")
            continue
        res_ext = res[ext]
        residual = np.arange(len(ext))       # ext-relative indices still "in play"
        frozen_templates = []

        for rnd in range(a.stochastic_peel_rounds + 1):
            if len(residual) < 2 * a.min_group:
                break
            all_geoms = _ensemble_chunk(c, ext, res_ext, residual, process,
                                        a.stochastic_draws, a.stochastic_frac, rng,
                                        jobs=ejobs, seedseq=ensemble_seedseq)
            if not any(all_geoms):
                break
            inst, cons_gid, recov, best, best2 = _consensus(all_geoms, a.stochastic_match_corr, link=a.stochastic_link)

            # tally per-spike votes: each instance owns some ext-relative spikes (_draw_members) and
            # belongs to a consensus fiber; a per-draw sub-mode id distinguishes branches within a fiber
            # (instances of the same fiber in the same draw -- rare -- get distinct sub-mode ids).
            # cons_gid restarts at 0 in every chunk (consensus is per-chunk, no cross-chunk matching),
            # so the vote keys are namespaced by chunk -- otherwise chunk-A fiber N and chunk-B fiber N,
            # which are different cells, would collide onto the same .clu cluster.
            if a.stochastic_write_clu:
                submode_ctr = {}
                for x, (dd, ii, _) in enumerate(inst):
                    cg = (c, int(cons_gid[x]))                 # (chunk, per-chunk consensus id)
                    sm = submode_ctr.get((dd, cg), 0); submode_ctr[(dd, cg)] = sm + 1
                    mem = all_geoms[dd][ii].get("_draw_members")
                    if mem is None:
                        continue
                    for e in mem.tolist():
                        ri = int(ext[e])                       # ext-relative -> absolute res index
                        vote_fiber.setdefault(ri, {})[cg] = vote_fiber.setdefault(ri, {}).get(cg, 0) + 1
                        key = (cg, sm)
                        vote_submode.setdefault(ri, {})[key] = vote_submode.setdefault(ri, {}).get(key, 0) + 1

            # record every instance (this is the fiber-space dump)
            for x, (d, i, _) in enumerate(inst):
                g = all_geoms[d][i]
                rows.append(dict(g, _chunk=c, _draw=d, _frac=a.stochastic_frac,
                                 _peel_round=rnd, _consensus_gid=int(cons_gid[x]),
                                 _recovery_freq=float(recov[x]), _match_corr=float(best[x]),
                                 _match_corr2=float(best2[x])))

            # consensus fibers stable enough to freeze this round
            stable = [cg for cg in range(cons_gid.max() + 1 if len(cons_gid) else 0)
                      if recov[np.flatnonzero(cons_gid == cg)][0] >= a.stochastic_stable_freq]
            if not stable or not a.stochastic_peel_rounds:
                peel_log.append(dict(chunk=c, round=rnd, frozen=len(stable),
                                     residual_in=len(residual), remaining=len(residual)))
                break

            # peel: remove the union of stable fibers' spikes (majority vote across their instances)
            cons_t = _consensus_templates(inst, cons_gid, all_geoms)
            frozen_templates.extend(cons_t[cg] for cg in stable)
            owned = np.zeros(len(ext), bool)
            for cg in stable:
                for x in np.flatnonzero(cons_gid == cg):
                    mem = all_geoms[inst[x][0]][inst[x][1]].get("_draw_members")
                    if mem is not None:
                        owned[mem] = True
            new_residual = residual[~owned[residual]]
            peel_log.append(dict(chunk=c, round=rnd, frozen=len(stable),
                                 residual_in=len(residual), remaining=len(new_residual)))
            if a.stochastic_verbose:
                _plog(f"        round {rnd}: froze {len(stable):>3} stable fiber(s), "
                      f"residual {len(residual):>6} -> {len(new_residual):>6}")
            if len(new_residual) == len(residual):
                break
            residual = new_residual

        _done += 1
        _rows_c = sum(r['_chunk'] == c for r in rows)
        _npeel = len([p for p in peel_log if p['chunk'] == c])
        _dt = _time.time() - _tc; _el = _time.time() - _t0
        _eta = _el / _done * (len(chunks) - _done)
        _plog(f"  [{ci + 1:>3}/{len(chunks)}] chunk {c:>3}: {len(ext):>6} ext | "
              f"{_rows_c:>5} fiber instances | {_npeel} peel round(s) | "
              f"{_dt:5.1f}s (elapsed {_el / 60:4.1f}m, eta {_eta / 60:4.1f}m)")

    _plog(f"fiber_stochastic: all {_done} chunk(s) done in {(_time.time() - _t0) / 60:.1f}m; "
          f"{len(rows):,} fiber instances total")
    _dump_ensemble(a, rows, peel_log, mask, gch)
    if a.stochastic_write_clu:
        _write_cluster_triplet(a, res.size, vote_fiber, vote_submode)


def _write_cluster_triplet(a, n_spikes, vote_fiber, vote_submode):
    """Resolve the per-spike votes to a Klusters clu/clc/clp triplet so the consensus fibers can be
    inspected in Klusters.  .clc = each spike's majority SUB-MODE (the atom/leaf layer, the branch);
    .clp = sub-mode -> consensus-fiber parent map; .clu = the per-spike fiber, DERIVED from each
    sub-mode's parent.  Spikes never assigned to any fiber (not drawn into a stable mode) go to noise.

    Built via the shared FiberHierarchy writer (the same one fiber_session / intrachunk / link / refine
    use) rather than hand-writing the three files: it derives .clu from (child, parent) so the fiber
    layer is guaranteed consistent with the sub-mode layer, compacts fiber ids (renumber, 0/1 reserved
    for noise/artefact), and warns on orphaned children.  The sub-mode is the child; its consensus fiber
    is the parent."""
    from .fiber_refiberize import FiberHierarchy

    # majority sub-mode per spike -> the child (atom) layer; child ids are 1-based, 0 = noise
    sub_of = {ri: max(ctr, key=ctr.get) for ri, ctr in vote_submode.items()}   # ri -> ((chunk,cg), sm)
    subkeys = sorted(set(sub_of.values()))
    child_id = {key: k + 1 for k, key in enumerate(subkeys)}
    child = np.zeros(n_spikes, np.int64)
    for ri, key in sub_of.items():
        child[ri] = child_id[key]

    # each sub-mode's consensus fiber -> the child->parent map; parent fiber ids are 2-based (0/1 noise)
    fibkeys = sorted({key[0] for key in subkeys})                              # distinct (chunk,cg)
    fib_id = {cg: k + 2 for k, cg in enumerate(fibkeys)}
    parent = {child_id[key]: fib_id[key[0]] for key in subkeys}               # child -> fiber

    tag = a.stochastic_clu_tag
    paths = FiberHierarchy(child, parent).save(a.base, a.elec, variant="stderiv", tag=tag,
                                               renumber=True, backup=False)
    nfib = len(fibkeys)
    nassigned = int((child > 0).sum())
    print(f"fiber_stochastic: wrote consensus triplet .clu/.clc/.clp (tag '{tag}') — "
          f"{nfib} fibers, {len(subkeys)} sub-modes, {nassigned}/{n_spikes} spikes assigned "
          f"({100*nassigned/max(n_spikes,1):.0f}%; rest -> noise) [{paths['clu']}]")


# ── serialization: the fiber-space file (mirrors production .fibers.npz + extras) ──
def _dump_ensemble(a, rows, peel_log, mask, gch):
    out = f"{a.base}.fiberens.{a.elec}.npz"
    M = len(rows)
    p = len(mask) * a.nchan

    def col(k, dt, default=np.nan):
        return np.array([r.get(k, default) for r in rows], dt) if M else np.zeros(0, dt)

    arrs = dict(
        # ── provenance: which draw / peel round / consensus each instance belongs to ──
        chunk=col("_chunk", int, -1), draw=col("_draw", int, -1),
        frac=col("_frac", np.float32), peel_round=col("_peel_round", int, 0),
        consensus_gid=col("_consensus_gid", int, -1),
        # ── stability observables (the point of the exercise) ──
        recovery_freq=col("_recovery_freq", np.float32),   # fraction of draws this consensus fiber appears in
        match_corr=col("_match_corr", np.float32),         # best template corr into another draw
        match_corr2=col("_match_corr2", np.float32),       # 2nd-best into a DIFFERENT consensus (merge-proneness)
        # ── geometry / size (same columns as production .fibers.npz) ──
        nspk=col("n", int), radius=col("radius", np.float32),
        depth=col("depth", np.float32), width_ms=col("width_ms", np.float32),
        rate=col("rate", np.float32), presence=col("presence", np.float32),
        refrac=col("refrac", np.float32), burst=col("burst", np.float32),
        isi_cv=col("isi_cv", np.float32), hill_fp=col("hill_fp", np.float32),
        resid_med=col("resid_med", np.float32), resid_mad=col("resid_mad", np.float32),
        chan_resid_var_mean=col("chan_resid_var_mean", np.float32),
        chan_resid_var_max=col("chan_resid_var_max", np.float32),
        radius_slope=col("radius_slope", np.float32), depth_slope=col("depth_slope", np.float32),
        dir_drift=col("dir_drift", np.float32),
        adapt_corr=col("adapt_corr", np.float32), adapt_tau=col("adapt_tau", np.float32),
        adapt_snr=col("adapt_snr", np.float32), adapt_meanabsz=col("adapt_meanabsz", np.float32),
        adapt_fracz3=col("adapt_fracz3", np.float32),
        # ── the shapes themselves, so the fiber space can be embedded/clustered offline ──
        template=np.stack([np.asarray(r["template"], np.float32) for r in rows]) if M
                 else np.zeros((0, a.nsamp, a.nchan), np.float32),
        grid=np.stack([np.asarray(r["grid"], np.float32) for r in rows]) if M
             else np.zeros((0, a.n_grid), np.float32),
        dir=np.stack([np.asarray(r["dir"], np.float32) for r in rows]) if M
            else np.zeros((0, a.n_grid, p), np.float32),
        # ── peel trajectory (one row per chunk*round): does freezing the strong fibers
        #    stabilize the remainder, or just remove the easy spikes?  Separate row-store
        #    from the per-instance columns above (hence the peellog_ prefix). ──
        peellog_chunk=np.array([q["chunk"] for q in peel_log], int),
        peellog_round=np.array([q["round"] for q in peel_log], int),
        peellog_frozen=np.array([q["frozen"] for q in peel_log], int),
        peellog_residual_in=np.array([q["residual_in"] for q in peel_log], int),
        peellog_remaining=np.array([q["remaining"] for q in peel_log], int),
        # ── meta ──
        meta_elec=a.elec, meta_channels=np.asarray(gch), meta_sr=a.sr, meta_mask=np.asarray(mask),
        meta_nsamp=a.nsamp, meta_nchan=a.nchan, meta_n_grid=a.n_grid, meta_p=p,
        meta_draws=a.stochastic_draws, meta_frac=a.stochastic_frac,
        meta_match_corr=a.stochastic_match_corr, meta_link=a.stochastic_link, meta_stable_freq=a.stochastic_stable_freq,
        meta_peel_rounds=a.stochastic_peel_rounds, meta_seed=a.stochastic_seed, meta_jobs=int(getattr(a,'stochastic_jobs',1)))

    import sys as _sys, time as _time
    print(f"fiber_stochastic: compressing + writing {out} ({M:,} instances)...",
          file=_sys.stderr, flush=True)
    _tw = _time.time()
    with open(out, "wb") as f:
        np.savez_compressed(f, **arrs)
    print(f"fiber_stochastic: wrote {out}  ({_time.time() - _tw:.1f}s)", file=_sys.stderr, flush=True)
    nconsensus = int(col("_consensus_gid", int, -1).max() + 1) if M else 0
    print(f"  {M:,} fiber instances over {a.stochastic_draws} draws x {len(set(r['_chunk'] for r in rows))} chunks "
          f"(frac={a.stochastic_frac}) -> {nconsensus} consensus fibers")
    if M:
        rf = col("_recovery_freq", np.float32)
        print(f"  recovery: >={a.stochastic_stable_freq}: {(rf>=a.stochastic_stable_freq).sum()} instances | "
              f"median match_corr2 (merge-proneness) = {np.median(col('_match_corr2', np.float32)):.3f}")


def add_arguments(ap):
    """Register --stochastic-* flags on an existing parser (shared with fiber_session)."""
    g = ap.add_argument_group("stochastic ensemble / consensus fibering (diagnostic; default off)")
    g.add_argument("--stochastic", action="store_true",
                   help="run the ensemble/consensus fibering diagnostic instead of the normal single pass")
    g.add_argument("--stochastic-draws", type=int, default=20,
                   help="number of resampled draws per chunk (per peel round)")
    g.add_argument("--stochastic-frac", type=float, default=0.8,
                   help="subsample fraction per draw.  HIGH by default: a low fraction starves most "
                        "fibers below min_group (a 0.5 draw drops 50-70%% of fibers on this data)")
    g.add_argument("--stochastic-link", choices=("average", "complete", "single"), default="average",
                   help="how draw-fibers are grouped into consensus fibers from pairwise template correlation. "
                        "'single' (transitive union-find) CHAINS anticorrelated sub-modes through intermediate "
                        "shapes into one component -- it undercounts real units; 'average' (default) and "
                        "'complete' are agglomerative and will not weld a chain of distinct co-located cells.")
    g.add_argument("--stochastic-match-corr", type=float, default=0.95,
                   help="template correlation above which two fibers from different draws are the SAME "
                        "consensus fiber")
    g.add_argument("--stochastic-stable-freq", type=float, default=0.6,
                   help="a consensus fiber is 'stable' (and gets frozen when peeling) if it appears in "
                        "at least this fraction of draws")
    g.add_argument("--stochastic-peel-rounds", type=int, default=0,
                   help="0 = single ensemble pass (no peeling).  N>0 = freeze stable fibers, remove their "
                        "spikes, and re-run the ensemble on the residual, up to N times")
    g.add_argument("--stochastic-chunks", type=int, nargs="*", default=None,
                   help="restrict to these chunk indices (default: all) — useful for a quick look")
    g.add_argument("--stochastic-seed", type=int, default=0, help="RNG seed for the draws")
    g.add_argument("--stochastic-jobs", type=int, default=1,
                   help="parallelise the (independent) resampling draws across this many worker processes "
                        "(the same per-chunk worker fiber_session uses).  Draws are deterministic regardless "
                        "of job count (per-draw seeds spawned from --stochastic-seed).  1 = serial.")

    g.add_argument("--stochastic-write-clu", action="store_true",
                   help="also write a Klusters clu/clc/clp triplet from the per-spike majority vote, so the "
                        "consensus fibers (.clu) and their sub-modes/branches (.clc) can be inspected in Klusters")
    g.add_argument("--stochastic-clu-tag", default="fiber_stochastic",
                   help="tag for the written triplet: <base>.clu.stderiv.<elec>.<tag> etc")

    g.add_argument("--stochastic-verbose", action="store_true", help="per-chunk progress")
    return ap


def main():
    """Console-script entry (fiber-stochastic).  Reads <session>.yaml; CLI flags override.
    Reuses the real per-chunk clusterer, so results reflect the production path."""
    try:
        from . import session_yaml as sy
    except ImportError:
        import session_yaml as sy
    ap = argparse.ArgumentParser(description="ensemble/consensus fibering diagnostic (fiber_stochastic). "
                                             "Reads <session>.yaml; CLI flags override. Reuses the real "
                                             "per-chunk clusterer, so results reflect the production path.")
    sy.add_session_args(ap)            # same session/group/channels resolution as fiber_session
    fsess.add_core_arguments(ap)       # identical clusterer knobs
    add_arguments(ap)                  # the --stochastic-* diagnostic knobs
    a = ap.parse_args()
    run_stochastic(a)


if __name__ == "__main__":
    main()
