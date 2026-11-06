#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_psame.py -- fiber-psame-merge: merge clusters by a calibrated
#  probability that two clusters are ONE CELL, from history-conditioned cell models.
#
#  Each cluster gets a cell model of its spikes in a global PCA space of its .spk:
#      y_i ~ N( a_i T + (a_i - <a>) S , diag var )
#  where a_i, the spike's amplitude on the template T, is PREDICTED from the cluster's
#  own spikes BEFORE t_i (an EWMA adaptation state at the best tau, and a burst term
#  exp(-dt_prev / 8 ms)), and S is the cluster's shape change per unit of amplitude.
#  For a candidate pair the evidence is a BIC comparison of one such model of the union
#  against two:
#      dbic = [2 (LL_union - LL_A - LL_B) + k ln n] / n            (higher = one cell)
#  so fragments of one adapting/bursting cell -- whose spikes ONE adaptation and burst
#  law predicts -- score as one cell, and a co-shaped cell without that law does not.
#  P(same) = logistic(bias + w_rc * roll-cos + w_dbic * dbic).
#
#  LIMIT: a cell cut by a hard AMPLITUDE threshold.  Each piece is then narrower than the
#  cell along the noisy amplitude axis (selection), so the union always fits worse and
#  dbic < 0 (test/test_psame.py documents the synthetic numbers).  Sorted fragments are
#  rarely pure amplitude cuts -- on g6 dbic separated real same-cell pairs at AUC
#  0.79-0.98 -- but a sort that bands by energy (fiber-session eband) produces exactly
#  that case; leave those merges to the stages built for it.
#
#  Merging is SEQUENTIAL WITH REFIT: the most probable candidate pair merges, the
#  merged cluster's model is refitted and re-scored against its neighbours, repeat
#  while P >= theta.  A member-level CANNOT-LINK stops snowballing: two groups may only
#  merge if no pair of their ORIGINAL clusters was scored below --cannot-link at the
#  start (without it one growing, ever broader cluster absorbs its neighbours: on g6 a
#  single group swallowed five cells).
#
#  Calibration (the shipped coefficients): g6 sirotaA-jg-000005-20120316, the
#  exp_chunk + fs_clean sorts (2,026 GT-labelled candidate pairs, roll-cos >= 0.75,
#  8 nearest), stderiv_C5 .spk.  Held-out sorts: AUC 0.958 vs 0.878 for roll-cos alone;
#  P >= 0.97 -> 100 % same cell, 0.90-0.97 -> 97 %.  End to end on the linked sort
#  (fix_linked -> GT, ARI / purity / completeness):
#      fiber-xcorr-merge (exp plan)             0.577 / 0.758 / 0.598
#      fiber-psame-merge theta 0.5, cl 0.3      0.60-0.67 / 0.72 / 0.70-0.77 (two runs)
#      merging by the GT itself (ceiling)       0.692 / 0.759 / 0.764
#  i.e. it recovers most of the merge ceiling, at a purity cost (merges between
#  clusters that were already mixed).  The coefficients are SESSION-CALIBRATED: on a
#  different probe / filter / .spk variant re-fit them (--w-rc/--w-dbic/--bias).
#
#  --geo 1 adds the shared-geometry test of fiber_srcgeom (one 6-source chain at one
#  place for both raw templates) as a third term; it needs the STANDARD .spk and the
#  probe geometry.  On g6 it raises pair AUC (0.958 -> 0.962; held-out sort 0.871 ->
#  0.895) and purity at strict theta, but not the end-to-end ARI.  Off by default.
#
#  A pure .clu relabel (.res/.spk untouched); the .clc/.clp hierarchy is carried.
#  Writes <base>.psame.<group>.<out-stage>.tsv: every accepted merge with P and terms.
#  Knobs FK_PSAME_* (CLI > env > $FK_CONFIG > default).
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy, config as cfgmod
except ImportError:
    import neuro_io as nio, session_yaml as sy, config as cfgmod

_LP = "▸ fiber-psame-merge"
TAUS = np.array([0.03, 0.1, 0.3, 0.6, 1.2])          # s; adaptation time constants searched
BURST_TAU = 0.008                                      # s; burst term exp(-dt_prev / BURST_TAU)
N_PC = 24

_KNOBS = {
    "FK_PSAME_THETA": ("theta", float, 0.5),
    "FK_PSAME_CANNOT_LINK": ("cannot_link", float, 0.3),
    "FK_PSAME_KNN": ("knn", int, 8),
    "FK_PSAME_COS_FLOOR": ("cos_floor", float, 0.75),
    "FK_PSAME_MIN_N": ("min_n", int, 50),
    "FK_PSAME_SHIFT": ("shift", int, 4),
    "FK_PSAME_W_RC": ("w_rc", float, 22.2143),
    "FK_PSAME_W_DBIC": ("w_dbic", float, 0.6144),
    "FK_PSAME_BIAS": ("bias", float, -17.773),
    "FK_PSAME_GEO": ("geo", int, 0),
    "FK_PSAME_GEO_W_RC": ("geo_w_rc", float, 21.5792),
    "FK_PSAME_GEO_W_DBIC": ("geo_w_dbic", float, 0.5606),
    "FK_PSAME_GEO_W_Z": ("geo_w_z", float, -0.4395),
    "FK_PSAME_GEO_BIAS": ("geo_bias", float, -17.0015),
}


# ── history-conditioned cell model ──────────────────────────────────────────
def ewma_state(t, tau, tq=None, horizon=10.0):
    """EWMA adaptation state sum_{t_j < q} exp(-(q - t_j)/tau) of the sorted train t at query times tq
    (default: t itself, i.e. from each spike's PREDECESSORS).  Terms older than horizon*tau are dropped
    (< 5e-5 each); vectorised over queries, one pass per predecessor rank."""
    t = np.asarray(t, float); tq = t if tq is None else np.asarray(tq, float)
    j = np.searchsorted(t, tq, side="left") - 1                     # last spike strictly before
    lo = np.searchsorted(t, tq - horizon * tau, side="left")
    out = np.zeros(len(tq))
    K = int((j - lo + 1).max()) if len(tq) else 0
    for k in range(max(K, 0)):
        jj = j - k
        ok = jj >= lo
        if not ok.any():
            break
        out[ok] += np.exp(-(tq[ok] - t[jj[ok]]) / tau)
    return out


def prev_interval(t, tq=None):
    t = np.asarray(t, float); tq = t if tq is None else np.asarray(tq, float)
    j = np.searchsorted(t, tq, side="left") - 1
    return np.where(j >= 0, tq - t[np.clip(j, 0, None)], np.inf)


class CellModel:
    """y ~ N(a_hat T + (a_hat - <a>) S, diag var), a_hat from the cell's own preceding spikes."""

    def __init__(self, Y, t):
        o = np.argsort(t, kind="stable")
        self.t = np.asarray(t, float)[o]; Y = np.asarray(Y, float)[o]
        self.T = Y.mean(0); nT = float(self.T @ self.T) or 1e-12
        a = Y @ self.T / nT; self.abar = float(a.mean())
        burst = np.exp(-prev_interval(self.t) / BURST_TAU)
        best = None
        for tau in TAUS:
            M = np.c_[np.ones(len(a)), ewma_state(self.t, tau), burst]
            b = np.linalg.lstsq(M, a, rcond=None)[0]
            sse = float(((a - M @ b) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, tau, b)
        _, self.tau, self.b = best
        R = Y - a[:, None] * self.T[None]; da = a - self.abar
        self.S = (da[:, None] * R).sum(0) / max(float(da @ da), 1e-12)
        self.var = (Y - self._mu(self.t)).var(0) + 1e-6
        self.Y = Y

    def _mu(self, tq):
        ah = (self.b[0] + self.b[1] * ewma_state(self.t, self.tau, tq)
              + self.b[2] * np.exp(-prev_interval(self.t, tq) / BURST_TAU))
        return ah[:, None] * self.T[None] + (ah - self.abar)[:, None] * self.S[None]

    def loglik(self):
        """Total log-likelihood of the model's own spikes."""
        return float(-0.5 * (((self.Y - self._mu(self.t)) ** 2) / self.var + np.log(2 * np.pi * self.var)).sum())


def dbic(mA, mB):
    """[2 (LL_union - LL_A - LL_B) + k ln n] / n: per-spike BIC evidence that A and B are one cell."""
    U = CellModel(np.concatenate([mA.Y, mB.Y]), np.concatenate([mA.t, mB.t]))
    n = len(U.t); k = 3 * U.Y.shape[1] + 4
    return float((2 * (U.loglik() - mA.loglik() - mB.loglik()) + k * np.log(n)) / n)


def rollcos(a, b, shift=4):
    """Max over circular shifts of the cosine of two mean waveforms (channel means removed)."""
    a = a - a.mean(0); na = np.sqrt((a * a).sum()); best = -1.0
    for s in range(-shift, shift + 1):
        bb = np.roll(b, s, 0); bb = bb - bb.mean(0)
        best = max(best, float((a * bb).sum() / max(na * np.sqrt((bb * bb).sum()), 1e-12)))
    return best


def pca_basis(spk, n=N_PC, nfit=60000, rng=None):
    rng = rng or np.random.default_rng(0)
    N = spk.shape[0]
    s = np.sort(rng.choice(N, min(nfit, N), replace=False))
    X = np.asarray(spk[s], np.float32).reshape(len(s), -1)
    mu = X.mean(0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:n].T
    return mu, V.astype(np.float32)


def project(spk, idx, mu, V):
    # NOT centred on the global mean: the cell model a T + (a - <a>) S has no offset term, so a cell
    # x = a c must stay proportional to its template (centring adds -mu to every spike, which the
    # model would then scale with the amplitude)
    return np.asarray(spk[np.sort(idx)], np.float32).reshape(len(idx), -1) @ V


# ── the merge ───────────────────────────────────────────────────────────────
def psame_merge(spk, times, clu, *, theta=0.5, cannot_link=0.3, knn=8, cos_floor=0.75, min_n=50, shift=4,
                w_rc=22.2143, w_dbic=0.6144, bias=-17.773, geo_term=None, w_geo=0.0, cap=None, rng=None,
                log=print):
    """Sequential P(same) merge.  spk: (N, nsamp, nchan) memmap of the scoring variant; times (s) per spike;
    clu per spike.  geo_term: optional callable (members_a, members_b) -> shared-geometry z of the two groups' raw templates.  Returns (new clu,
    list of accepted merges as dicts)."""
    rng = rng or np.random.default_rng(0)
    clu = np.asarray(clu).astype(np.int64).copy()
    mu, V = pca_basis(spk, rng=rng)
    ids = [int(u) for u, c in zip(*np.unique(clu[clu > 1], return_counts=True)) if c >= min_n]
    if len(ids) < 2:
        return clu, []
    model, mean = {}, {}

    def refresh(u):
        idx = np.flatnonzero(clu == u)
        sub = np.sort(rng.choice(idx, min(1000, idx.size), replace=False))
        mean[u] = np.asarray(spk[sub], float).mean(0)
        use = idx if cap is None or idx.size <= cap else np.sort(rng.choice(idx, cap, replace=False))
        model[u] = CellModel(project(spk, use, mu, V), times[use])

    for u in ids:
        refresh(u)

    def neighbours(u):
        sc = sorted(((rollcos(mean[u], mean[v], shift), v) for v in ids if v != u), reverse=True)[:knn]
        return [v for r, v in sc if r >= cos_floor]

    def score(a, b):
        rc = rollcos(mean[a], mean[b], shift)
        d = dbic(model[a], model[b])
        z = geo_term(members[a], members[b]) if geo_term is not None else 0.0
        p = 1.0 / (1.0 + np.exp(-(bias + w_rc * rc + w_dbic * d + w_geo * z)))
        return float(p), dict(rc=rc, dbic=d, geo_z=z)

    members = {u: {u} for u in ids}
    cand = {}
    for u in ids:
        for v in neighbours(u):
            k = (min(u, v), max(u, v))
            if k not in cand:
                cand[k] = score(*k)
    p0 = {k: v[0] for k, v in cand.items()}

    def allowed(a, b):
        return all(p0.get((min(x, y), max(x, y)), 1.0) >= cannot_link for x in members[a] for y in members[b])

    merges = []
    log(f"{_LP} · {len(ids)} clusters >= {min_n} spikes, {len(cand)} candidate pairs "
        f"(roll-cos >= {cos_floor}, {knn} nearest)")
    while cand:
        (a, b), (p, terms) = max(cand.items(), key=lambda kv: kv[1][0])
        if p < theta:
            break
        if not allowed(a, b):
            del cand[(a, b)]; continue
        clu[clu == b] = a; ids.remove(b); members[a] |= members.pop(b)
        merges.append(dict(into=a, merged=b, p=p, **terms))
        del mean[b], model[b]; refresh(a)
        cand = {k: v for k, v in cand.items() if a not in k and b not in k}
        for v in neighbours(a):
            k = (min(a, v), max(a, v)); cand[k] = score(*k)
    return clu, merges


def _knob_default(name, typ, fb, gcfg):
    return cfgmod.knob_value(name, typ, fb, gcfg)


def main():
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(prog="fiber-psame-merge",
                                 description="Merge clusters by a calibrated P(same cell) from history-conditioned "
                                             "cell models (sequential, with refit and a member cannot-link).")
    ap.add_argument("session"); ap.add_argument("elec", type=int)
    ap.add_argument("--clu-method", default="stderiv")
    ap.add_argument("--clu-stage", default="xcorr_merged", help="input .clu stage tag")
    ap.add_argument("--in-clu", default=None, help="explicit input .clu path (overrides --clu-method/--clu-stage)")
    ap.add_argument("--spk-method", "--spk-variant", dest="spk_variant", default=None,
                    help="waveform variant the cell models are fitted on (default: the WAVEFORM token of "
                         "--clu-method, e.g. stderiv_C5_D34 -> stderiv_C5; the shipped coefficients were "
                         "calibrated on stderiv_C5)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default="psame_merged")
    ap.add_argument("--cap", type=int, default=None, help="spikes per cluster model (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    for name, (dest, typ, fb) in _KNOBS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ, default=_knob_default(name, typ, fb, gcfg),
                        help=f"{name} (default {_knob_default(name, typ, fb, gcfg)})")
    a = ap.parse_args()
    cfg = sy.resolve_session_params(a.session, a.elec)
    base = cfg["base"]; sr = cfg["sr"]; ns = cfg["nsamp"]; nc = cfg["nchan"]
    res = nio.read_res(base, a.elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=res.size)
    else:
        _, clu = nio.read_clu_at(base, a.elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=res.size)
    # the .spk of a _D feature-space token is its WAVEFORM token (stderiv_C5_D34 -> .spk.stderiv_C5): there is
    # no _D waveform set, so the clu method itself names a file that does not exist
    var = a.spk_variant or nio.waveform_token(a.clu_method)
    spk = nio.open_spk_file(nio.session_path(base, "spk", a.elec, variant=var), ns, nc)
    if not (res.size == clu.size == spk.shape[0]):
        raise SystemExit(f"{_LP} · res/clu/spk length mismatch ({res.size}/{clu.size}/{spk.shape[0]})")
    rng = np.random.default_rng(a.seed)
    w_rc, w_dbic, bias, w_geo = a.w_rc, a.w_dbic, a.bias, 0.0
    geo_term = None
    if a.geo:
        try:
            from . import fiber_srcgeom as sg, fiber_localize as loc
        except ImportError:
            import fiber_srcgeom as sg, fiber_localize as loc
        if not cfg.get("probe"):
            raise SystemExit(f"{_LP} · --geo 1 needs the probe geometry: name the probe file in <session>.yaml")
        S = sg.sites3(loc.load_geometry(cfg["probe"], cfg["channels"]))
        raw = nio.open_spk_file(nio.session_path(base, "spk", a.elec, variant="standard"), ns, nc)
        fits = {}

        def _fit(mem):
            key = frozenset(mem)
            if key not in fits:
                idx = np.flatnonzero(np.isin(clu, list(mem)))     # the INPUT labels: a group = its members' spikes
                idx = np.sort(rng.choice(idx, min(1500, idx.size), replace=False))
                X = np.asarray(raw[idx], float); x = X.mean(0)
                p = rng.permutation(len(X)); h = len(p) // 2
                nz = float(((X[p[:h]].mean(0) - X[p[h:]].mean(0)) ** 2).sum() / 4 / max((x ** 2).sum(), 1e-30))
                fits[key] = (x, sg.fit(x, S), nz)
            return fits[key]

        def _geo_term(ma, mb):
            xa, fa, na = _fit(ma); xb, fb, nb = _fit(mb)
            z, _, _ = sg.shared_geometry_z(xa, xb, S, na, nb, fa, fb)
            return z
        geo_term = _geo_term
        w_rc, w_dbic, bias, w_geo = a.geo_w_rc, a.geo_w_dbic, a.geo_bias, a.geo_w_z
    print(f"{_LP} · {base} elec {a.elec} | {var} .spk | theta {a.theta} cannot-link {a.cannot_link}"
          + (" | + shared geometry" if a.geo else ""))
    new, merges = psame_merge(spk, res / float(sr), clu, theta=a.theta, cannot_link=a.cannot_link, knn=a.knn,
                              cos_floor=a.cos_floor, min_n=a.min_n, shift=a.shift, w_rc=w_rc, w_dbic=w_dbic,
                              bias=bias, geo_term=geo_term, w_geo=w_geo, cap=a.cap, rng=rng)
    nio.write_clu(base, a.elec, new, variant=a.clu_method, tag=a.out_tag)
    n_in = len(np.unique(clu[clu > 1])); n_out = len(np.unique(new[new > 1]))
    print(f"{_LP} · {len(merges)} merges ({n_in} -> {n_out} clusters); wrote "
          f"{os.path.basename(nio.session_path(base, 'clu', a.elec, variant=a.clu_method, tag=a.out_tag))}")
    log_path = f"{base}.psame.{a.elec}.{a.out_tag}.tsv"
    with open(log_path, "w") as fh:
        fh.write("into\tmerged\tp\trc\tdbic\tgeo_z\n")
        for m in merges:
            fh.write(f"{m['into']}\t{m['merged']}\t{m['p']:.4f}\t{m['rc']:.4f}\t{m['dbic']:.4f}\t{m['geo_z']:.3f}\n")
    print(f"{_LP} · wrote {os.path.basename(log_path)}")
    # carry the .clc/.clp hierarchy: a merge only re-parents children
    try:
        from .fiber_refiberize import FiberHierarchy as _FH
    except ImportError:
        from fiber_refiberize import FiberHierarchy as _FH
    try:
        h = _FH.load(base, a.elec, variant=a.clu_method, tag=a.clu_stage)
    except (FileNotFoundError, OSError):
        print(f"{_LP} · no .clc/.clp at .{a.clu_stage} - wrote .clu only"); return
    remap = {}
    for m in merges:
        remap[m["merged"]] = m["into"]

    def root(x):
        while x in remap:
            x = remap[x]
        return x
    newpar = {c: int(root(int(f))) for c, f in h.parent.items()}
    _FH(np.asarray(h.child, np.int64), newpar).save(base, a.elec, variant=a.clu_method, tag=a.out_tag,
                                                     renumber=False, backup=False)
    print(f"{_LP} · hierarchy carried: {len(newpar):,} children -> {len(set(newpar.values())):,} fibers")


if __name__ == "__main__":
    main()
