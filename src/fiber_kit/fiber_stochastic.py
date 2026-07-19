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
def _consensus(all_geoms, match_thr):
    """all_geoms: list over draws of lists of geom dicts.  Returns, for each (draw,
    local index), a consensus id, plus per-consensus recovery frequency and the
    per-instance best/second-best match into OTHER draws.

    match_corr  = best template correlation into a fiber from any OTHER draw.
    match_corr2 = best correlation into a fiber that ends up in a DIFFERENT consensus
                  component -- i.e. the nearest rival, the real merge-proneness signal.
                  (Second-best overall is usually just another fragment of the SAME
                  fiber and says nothing about merge risk, so it is computed against the
                  final component labels, not by raw rank.)"""
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

    # union-find over pairs above threshold (vectorized edge list)
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
def _ensemble_chunk(c, ext, res_ext, sub, ctx_process, ndraw, frac, rng):
    """Run the real per-chunk worker on K subsamples of `sub` (indices into ext).
    Returns a list (per draw) of geom-lists.  Each draw reuses _process_chunk, so the
    real whitener + cluster_chunk_fine + fiber_geom run on the drawn spikes."""
    all_geoms = []
    for _ in range(ndraw):
        di = sub[_draw_indices(len(sub), frac, rng)]
        ext_d = ext[di]; res_d = res_ext[di]
        _, _, lab, geoms, _, _ = ctx_process((c, ext_d, res_d))
        # tag each geom with which drawn spikes it owns (ext-relative) for later peel/assign
        for g, gk in zip(geoms, sorted(set(int(x) for x in lab if x >= 0))):
            g["_draw_members"] = di[np.flatnonzero(lab == gk)]
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
    fsess._init_chunk_worker(cfg)
    process = fsess._process_chunk

    res = nio.read_res(a.base, a.elec)
    t_min = float(res.min())
    chunk_s = a.chunk_min * 60.0 * a.sr; ov_s = a.overlap_min * 60.0 * a.sr
    nchunks = int(np.ceil((res.max() - t_min) / chunk_s))
    rng = np.random.default_rng(a.stochastic_seed)

    rows = []                      # every fiber instance, all draws, all peel rounds
    peel_log = []                  # per (chunk, round): frozen count, residual size, remaining
    chunks = a.stochastic_chunks if a.stochastic_chunks else list(range(nchunks))

    for c in chunks:
        lo_s = t_min + c * chunk_s; hi_s = t_min + (c + 1) * chunk_s
        ext = np.flatnonzero((res >= lo_s - ov_s) & (res < hi_s + ov_s))
        if len(ext) < 2 * a.min_group:
            continue
        res_ext = res[ext]
        residual = np.arange(len(ext))       # ext-relative indices still "in play"
        frozen_templates = []

        for rnd in range(a.stochastic_peel_rounds + 1):
            if len(residual) < 2 * a.min_group:
                break
            all_geoms = _ensemble_chunk(c, ext, res_ext, residual, process,
                                        a.stochastic_draws, a.stochastic_frac, rng)
            if not any(all_geoms):
                break
            inst, cons_gid, recov, best, best2 = _consensus(all_geoms, a.stochastic_match_corr)

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
            if len(new_residual) == len(residual):
                break
            residual = new_residual

        if a.stochastic_verbose:
            print(f"  chunk {c:>3}: {len(ext):>6} ext | "
                  f"{sum(r['_chunk']==c for r in rows):>5} fiber instances over draws | "
                  f"{len([p for p in peel_log if p['chunk']==c])} peel round(s)")

    _dump_ensemble(a, rows, peel_log, mask, gch)


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
        meta_match_corr=a.stochastic_match_corr, meta_stable_freq=a.stochastic_stable_freq,
        meta_peel_rounds=a.stochastic_peel_rounds, meta_seed=a.stochastic_seed)

    with open(out, "wb") as f:
        np.savez_compressed(f, **arrs)
    nconsensus = int(col("_consensus_gid", int, -1).max() + 1) if M else 0
    print(f"fiber_stochastic: wrote {out}")
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
