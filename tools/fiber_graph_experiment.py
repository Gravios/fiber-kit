#!/usr/bin/env python3
"""fiber_graph_experiment.py -- typed-graph fiber consolidation prototype + scoring.

Nodes = over-split fragments carrying (shape descriptor, depth, amplitude, chunk, t, variance).
Edges are TYPED, with different cost functions:
  * intrachunk (same chunk): near-pure IDENTITY claim; weight = shape distance.  Contracted FIRST.
  * temporal (adjacent chunks): identity + RELOCATION; weight = shape distance + drift-DISAGREEMENT,
    where the shared rigid drift D(c) is estimated (EM) from the consolidated fibers and a merge
    must AGREE with it (observed Δdepth ≈ ensemble-predicted Δdepth) -- the swap-killer.

Consolidation is MUTUAL-NEAREST-NEIGHBOUR agglomeration with feature RECOMPUTE (each fiber finds its
best partner; only mutual-best pairs merge; the merged mean is recomputed before the next round) --
which does NOT chain the way threshold->union-find (single linkage) does.

Pipeline: intrachunk MNN -> estimate D(c) -> drift-scored temporal MNN (re-estimate D as fibers grow).
Scoring (--gt-clu): pairwise F1, #fibers vs ground truth, id-mixing, over-split.
Ablations: --no-drift (temporal=shape only), --union-find (threshold shape dist -> components).
RAW .spk.standard amplitudes for depth/amp; affine-invariant shape descriptor for identity.
"""
import argparse
import numpy as np

try:
    from fiber_kit import neuro_io as nio, fiber_lib as fl, session_yaml as sy, fiber_cfiber as cf
    from fiber_kit.fiber_drift import decentralized_drift
except ImportError:
    import neuro_io as nio, fiber_lib as fl, session_yaml as sy, fiber_cfiber as cf
    from fiber_drift import decentralized_drift


class FiberStore:
    """Per-fiber running mean shape + per-chunk amp-weighted depth + membership."""
    def __init__(self, nodes):
        self.zsh = {u: nodes[u]["zshape"].astype(float).copy() for u in nodes}
        self.zpr = {u: nodes[u]["zprof"].astype(float).copy() for u in nodes}
        self.n   = {u: 1 for u in nodes}
        self.cd  = {u: {nodes[u]["chunk"]: [nodes[u]["depth"], nodes[u]["amp"]]} for u in nodes}
        self.mem = {u: [u] for u in nodes}
        self.nspk = {u: nodes[u]["n"] for u in nodes}
    def roots(self):
        return list(self.zsh)
    def combine(self, ra, rb):
        na, nb = self.n[ra], self.n[rb]; w = na / (na + nb)
        self.zsh[ra] = w * self.zsh[ra] + (1.0 - w) * self.zsh[rb]
        self.zpr[ra] = w * self.zpr[ra] + (1.0 - w) * self.zpr[rb]
        for c, (d, a) in self.cd[rb].items():
            if c in self.cd[ra]:
                d0, a0 = self.cd[ra][c]; A = a0 + a
                self.cd[ra][c] = [(d0 * a0 + d * a) / A, A]
            else:
                self.cd[ra][c] = [d, a]
        self.n[ra] = na + nb
        self.mem[ra] += self.mem[rb]
        self.nspk[ra] += self.nspk[rb]
        del self.zsh[rb]; del self.zpr[rb]; del self.cd[rb]; del self.n[rb]; del self.mem[rb]; del self.nspk[rb]
    def shape_cost(self, ra, rb):
        return float(np.linalg.norm(self.zsh[ra] - self.zsh[rb]))
    def prof_cost(self, ra, rb):
        return float(np.linalg.norm(self.zpr[ra] - self.zpr[rb]))
    def mean_chunk(self, r):
        cs = np.array(list(self.cd[r])); ws = np.array([self.cd[r][c][1] for c in self.cd[r]])
        return float((cs * ws).sum() / (ws.sum() + 1e-9))
    def mean_depth(self, r):
        ds = np.array([self.cd[r][c][0] for c in self.cd[r]]); ws = np.array([self.cd[r][c][1] for c in self.cd[r]])
        return float((ds * ws).sum() / (ws.sum() + 1e-9))


def mnn_agglomerate(store, adj, cost_fn, thr, anchor_min=0, max_rounds=200):
    """Mutual-nearest-neighbour agglomeration: chaining-resistant (no transitive threshold cascade)."""
    for _ in range(max_rounds):
        roots = store.roots(); best = {}
        for r in roots:
            bc, bp = thr, None
            for x in adj.get(r, ()):
                if x == r:
                    continue
                c = cost_fn(store, r, x)
                if c < bc:
                    bc, bp = c, x
            if bp is not None:
                best[r] = bp
        merged, any_merge = set(), False
        for r in list(best):
            if r in merged:
                continue
            p = best[r]
            if p in merged or best.get(p) != r:        # mutual best only
                continue
            if max(store.nspk[r], store.nspk[p]) < anchor_min:
                continue   # two low-confidence fibers never merge directly (attach to anchors only)
            store.combine(r, p)
            adj[r] = (adj.get(r, set()) | adj.get(p, set())) - {r, p}
            for x in list(adj.get(p, ())):
                if x in adj:
                    adj[x].discard(p); adj[x].add(r)
            adj.pop(p, None)
            merged.add(r); merged.add(p); any_merge = True
        if not any_merge:
            break
    return store


def remap_adj(frag_adj, store):
    """Project the original fragment-level adjacency onto the CURRENT fiber roots."""
    frag2root = {f: r for r in store.mem for f in store.mem[r]}
    radj = {}
    for u, nbrs in frag_adj.items():
        ru = frag2root.get(u)
        if ru is None:
            continue
        for v in nbrs:
            rv = frag2root.get(v)
            if rv is None or rv == ru:
                continue
            radj.setdefault(ru, set()).add(rv)
    return radj


# ── node features ────────────────────────────────────────────────────────────
def build_nodes(spk, res, clu, theta, win, chunk_samples, min_spikes, cap, align, rng):
    nodes = {}; C = spk.shape[2]; chan = np.arange(C)
    for u in np.unique(clu):
        if u == 0:
            continue
        idx = np.flatnonzero(clu == u)
        if len(idx) < min_spikes:
            continue
        take = np.sort(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
        W = np.asarray(spk[take], float)
        if align:
            W = fl.realign(W)
        mw = np.median(W, axis=0)
        z = cf.complex_loop(mw[None], theta, win)
        shape, _, _, _ = cf.shape_descriptor(z)
        E = np.ptp(mw[win], axis=0); tot = E.sum() + 1e-9
        pk = int(E.argmax()); tr = mw[:, pk]
        tmin = int(tr.argmin()); tpk = tmin + int(tr[tmin:].argmax()) if tmin < len(tr) - 1 else tmin
        width = float(tpk - tmin)
        nodes[int(u)] = dict(shape=shape[0], prof=E.astype(float), depth=float((chan * E).sum() / tot), amp=float(E.max()),
                             chunk=int(res[take].mean() // chunk_samples), t=float(res[take].mean()),
                             var=float(np.median(((W - mw) ** 2).sum((1, 2)))), n=len(idx), width=width)
    return nodes


def standardize(nodes):
    S = np.array([nodes[u]["shape"] for u in nodes]); mu, sd = S.mean(0), S.std(0) + 1e-9
    P = np.array([nodes[u]["prof"] for u in nodes]); pmu, psd = P.mean(0), P.std(0) + 1e-9
    for u in nodes:
        nodes[u]["zshape"] = (nodes[u]["shape"] - mu) / sd
        nodes[u]["zprof"] = (nodes[u]["prof"] - pmu) / psd


def candidate_adjacency(nodes, depth_gate):
    ids = list(nodes); intra, temporal = {}, {}
    for i in range(len(ids)):
        a = nodes[ids[i]]
        for j in range(i + 1, len(ids)):
            b = nodes[ids[j]]
            if abs(a["depth"] - b["depth"]) > depth_gate:
                continue
            dc = abs(a["chunk"] - b["chunk"])
            tgt = intra if dc == 0 else (temporal if dc == 1 else None)
            if tgt is None:
                continue
            tgt.setdefault(ids[i], set()).add(ids[j]); tgt.setdefault(ids[j], set()).add(ids[i])
    return intra, temporal


# ── drift estimation + temporal cost ─────────────────────────────────────────
def estimate_drift(store, chunks):
    obs = {r: {c: store.cd[r][c][0] for c in store.cd[r]} for r in store.roots() if len(store.cd[r]) >= 2}
    if len(obs) < 3:
        return np.zeros(len(chunks))
    D, _b, _r, _bd = decentralized_drift(obs, chunks)
    return D


def make_temporal_cost(chunks, D, lam, use_drift):
    cidx = {c: i for i, c in enumerate(chunks)}
    carr = np.array(chunks, float)
    def Dinterp(mc):
        return float(np.interp(mc, carr, D))
    def cost(store, ra, rb):
        sd = store.shape_cost(ra, rb)
        if not use_drift:
            return sd
        ca, cb = store.mean_chunk(ra), store.mean_chunk(rb)
        observed = store.mean_depth(rb) - store.mean_depth(ra)
        predicted = Dinterp(cb) - Dinterp(ca)
        return sd + lam * abs(observed - predicted)
    return cost


# ── baseline + scoring ───────────────────────────────────────────────────────
def union_find_baseline(nodes, intra, temporal, thr):
    parent = {u: u for u in nodes}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def zs(u):
        return nodes[u]["zshape"]
    seen = set()
    for adj in (intra, temporal):
        for u, nb in adj.items():
            for v in nb:
                key = (min(u, v), max(u, v))
                if key in seen:
                    continue
                seen.add(key)
                if float(np.linalg.norm(zs(u) - zs(v))) < thr:
                    parent[find(u)] = find(v)
    comp = {}
    for u in nodes:
        comp.setdefault(find(u), []).append(u)
    return comp


def score_vs_gt(members, nodes, gt):
    frag2comp = {}
    for ci, mem in enumerate(members):
        for u in mem:
            frag2comp[u] = ci
    ids = [u for u in nodes if u in gt]
    tp = fp = fn = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            sp = frag2comp[a] == frag2comp[b]; sg = gt[a] == gt[b]
            if sp and sg:
                tp += 1
            elif sp and not sg:
                fp += 1
            elif not sp and sg:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    comps = {}
    for u in ids:
        comps.setdefault(frag2comp[u], set()).add(gt[u])
    id_mixing = sum(1 for s in comps.values() if len(s) > 1)
    units = {}
    for u in ids:
        units.setdefault(gt[u], set()).add(frag2comp[u])
    over_split = sum(1 for s in units.values() if len(s) > 1)
    return dict(prec=prec, rec=rec, f1=f1, nfib=len(comps), ngt=len(units),
                id_mixing=id_mixing, over_split=over_split)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sy.add_session_args(ap, nchan=False, sr=False, nsamp_default=None, peak=True)
    ap.add_argument("--clu-method", default="stderiv"); ap.add_argument("--clu-stage", default="refine")
    ap.add_argument("--spk-method", default="standard")
    ap.add_argument("--chunk-min", type=float, default=12.0)
    ap.add_argument("--min-spikes", type=int, default=50); ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--no-align", dest="align", action="store_false", default=True)
    ap.add_argument("--win-pre", type=int, default=10); ap.add_argument("--win-post", type=int, default=12)
    ap.add_argument("--intra-thr", type=float, default=0.0, help="0 -> percentile (--thr-pct) of anchor intra cost")
    ap.add_argument("--temporal-thr", type=float, default=0.0, help="0 -> percentile (--thr-pct) of anchor temporal shape dist")
    ap.add_argument("--thr-pct", type=float, default=10.0, help="percentile for auto shape_thr")
    ap.add_argument("--anchor-min", type=int, default=50, help="min spikes for a fragment to be a merge anchor")
    ap.add_argument("--depth-gate", type=float, default=2.0)
    ap.add_argument("--intra-beta", type=float, default=1.0, help="weight of amplitude-profile in intrachunk cost")
    ap.add_argument("--drift-lambda", type=float, default=1.0); ap.add_argument("--em-iters", type=int, default=4)
    ap.add_argument("--no-drift", dest="use_drift", action="store_false", default=True)
    ap.add_argument("--union-find", action="store_true")
    ap.add_argument("--gt-clu", default=None); ap.add_argument("--pitch", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal, nsamp=a.nsamp)
    base, elec, channels = cfg["base"], cfg["group"], cfg["channels"]
    nsamp = cfg["nsamp"]; peak = cfg["peak"] if cfg.get("peak") is not None else a.peak
    sr = cfg["sr"]; C = len(channels)
    res = nio.read_res(base, elec)
    _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.clu_stage, n_spikes=len(res))
    spk, r = nio.open_spk(base, elec, nsamp, C, prefer=[a.spk_method])
    n = min(len(res), len(clu), spk.shape[0]); res, clu = res[:n], clu[:n]
    lo = max(0, peak - a.win_pre); hi = min(nsamp, peak + a.win_post); win = slice(lo, hi)
    theta = cf.channel_angles(C); chunk_samples = a.chunk_min * 60.0 * sr
    print(f"[graph] {r.path}: {n} spikes, {C} ch, win[{lo}:{hi}], chunk={a.chunk_min}min")

    nodes = build_nodes(spk, res, clu, theta, win, chunk_samples, a.min_spikes, a.cap, a.align, rng)
    if len(nodes) < 2:
        raise SystemExit("[graph] <2 fragments meet --min-spikes")
    standardize(nodes)
    chunks = sorted({nd["chunk"] for nd in nodes.values()})
    intra, temporal = candidate_adjacency(nodes, a.depth_gate)
    ni = sum(len(v) for v in intra.values()) // 2; nt = sum(len(v) for v in temporal.values()) // 2
    print(f"[graph] {len(nodes)} fragments / {len(chunks)} chunks; {ni} intrachunk + {nt} temporal edges")

    anchors = {u for u in nodes if nodes[u]["n"] >= a.anchor_min}
    A = anchors
    di = [float(np.linalg.norm(nodes[u]["zshape"] - nodes[v]["zshape"])) +
          a.intra_beta * float(np.linalg.norm(nodes[u]["zprof"] - nodes[v]["zprof"]))
          for u, nb in intra.items() for v in nb if u < v and u in A and v in A]
    dt = [float(np.linalg.norm(nodes[u]["zshape"] - nodes[v]["zshape"]))
          for u, nb in temporal.items() for v in nb if u < v and u in A and v in A]
    intra_thr = a.intra_thr if a.intra_thr > 0 else (float(np.percentile(di, a.thr_pct)) if di else 1.0)
    temporal_thr = a.temporal_thr if a.temporal_thr > 0 else (float(np.percentile(dt, a.thr_pct)) if dt else 1.0)
    print(f"[graph] {len(anchors)} anchors (>= {a.anchor_min} spk); "
          f"intra_thr={intra_thr:.3f} (shape+{a.intra_beta}*prof), temporal_thr={temporal_thr:.3f} (shape)")

    if a.union_find:
        comp = union_find_baseline(nodes, intra, temporal, temporal_thr)
        members = list(comp.values()); D = None; label = "union-find baseline"
    else:
        store = FiberStore(nodes)
        intra_cost = lambda s, x, y: s.shape_cost(x, y) + a.intra_beta * s.prof_cost(x, y)
        mnn_agglomerate(store, remap_adj(intra, store), intra_cost, intra_thr, a.anchor_min)
        n_intra = len(store.roots())
        D = np.zeros(len(chunks))
        for _ in range(max(1, a.em_iters)):
            D = estimate_drift(store, chunks) if a.use_drift else np.zeros(len(chunks))
            cost = make_temporal_cost(chunks, D, a.drift_lambda, a.use_drift)
            before = len(store.roots())
            mnn_agglomerate(store, remap_adj(temporal, store), cost, temporal_thr, a.anchor_min)
            if len(store.roots()) == before:
                break
        members = list(store.mem.values())
        label = "typed graph" + ("" if a.use_drift else " (no-drift ablation)")
        print(f"[graph] intrachunk MNN -> {n_intra} fibers; after temporal -> {len(members)}")

    dur_s = (res.max() - res.min()) / sr
    us_per_smpl = 1e6 / sr
    rows = []
    for m in members:
        nspk = sum(nodes[u]["n"] for u in m)
        wid = np.median([nodes[u]["width"] for u in m]) * us_per_smpl
        rows.append((len(m), nspk, nspk / dur_s, wid))
    rows.sort(key=lambda x: -x[1])
    print(f"[graph] {label}: {len(members)} fibers. Top by spike count "
          f"(frags, spikes, rate_Hz, width_us; narrow width ~ interneuron):")
    for fr, nspk, rate, wid in rows[:10]:
        tag = "  <- narrow-spiking (interneuron-like)" if wid < 350 else ("  <- high-rate" if rate > 5 else "")
        print(f"        frags={fr:5d}  spikes={nspk:7d}  rate={rate:6.1f}Hz  width={wid:5.0f}us{tag}")
    if D is not None and a.use_drift and len(D):
        print(f"[graph] drift D(c) range = {D.min():.2f}..{D.max():.2f} ch "
              f"({(D.max()-D.min())*a.pitch:.1f} um @ {a.pitch} um pitch)")

    if a.gt_clu:
        _, gtc = nio.read_clu_file(a.gt_clu, n_spikes=n); gtc = gtc[:n]; gt = {}
        for u in nodes:
            vals = gtc[np.flatnonzero(clu == u)]; vals = vals[vals > 0]
            if len(vals):
                gt[u] = int(np.bincount(vals).argmax())
        sc = score_vs_gt(members, nodes, gt)
        print(f"\n[score] vs {sc['ngt']} gt units: F1={sc['f1']:.3f} (P={sc['prec']:.3f} R={sc['rec']:.3f})  "
              f"fibers={sc['nfib']}  id_mixing={sc['id_mixing']}  over_split={sc['over_split']}")
    else:
        print("[graph] no --gt-clu: consolidation only (pass a curated .clu to score)")


if __name__ == "__main__":
    main()
