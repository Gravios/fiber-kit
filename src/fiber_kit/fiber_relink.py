# ════════════════════════════════════════════════════════════════════════════
#  fiber_relink.py — geometry-aware re-bundling/re-linking of a .fibers run.
#
#  Operates on an existing <base>.fibers.<method>.<elec>.npz (no re-run needed).
#  Two stages, both STRICTLY ADDITIVE (they only merge, never split, so the
#  original anchor links are preserved):
#
#   1. within-chunk bundling — merge same-neuron fibers inside one chunk by
#      direction-profile + template agreement (concentrates over-split fragments).
#   2. cross-chunk matching by EVOLVING GEOMETRY — match fibers in nearby chunks
#      with a noise-aware blended distance (template-dominant when sparse, where
#      the direction profile is the sharp discriminator only when well sampled),
#      mutual-nearest-neighbour + uniqueness margin, plus a one-chunk gap bridge
#      so a momentarily-undetected unit is tracked across the gap.
#
#  Gates were calibrated on sirotaA-jg-000005-20120312 group 5: true anchor links
#  have consecutive template-dist 95pct=0.049, |Δdepth| 95pct=0.087, direction
#  profile 95pct=0.179; matchability scales with spike count (nspk>=1k -> 0.030
#  consecutive, nspk<300 -> ~0.19, geometry too noisy to link alone).
#
#  Outputs a remapped <base>.clu (old gid -> merged unit) and a per-unit drift
#  report TSV. Sparse single-chunk fibers that geometry cannot link safely are
#  left as their own units (those need the in-pipeline overlap anchors / CCG).
# ════════════════════════════════════════════════════════════════════════════
import argparse
import numpy as np
from collections import defaultdict
try:
    from . import neuro_io as nio
    from . import fiber_ccg as cg
except ImportError:
    import neuro_io as nio
    import fiber_ccg as cg


# ── geometry distances on the stored per-fiber summaries ────────────────────
def _interp_dir(grid, dvec, r):
    j = int(np.clip(np.searchsorted(grid, r) - 1, 0, len(grid) - 2))
    w = (r - grid[j]) / (grid[j + 1] - grid[j] + 1e-12)
    v = dvec[j] + w * (dvec[j + 1] - dvec[j])
    return v / (np.linalg.norm(v) + 1e-12)


def profile_dist(gi, di, gj, dj, n=7):
    """Mean (1-cos) of two direction profiles d(r) over their overlapping energy
    range. Returns a large value (9.0) when the ranges don't overlap."""
    lo = max(gi[0], gj[0]); hi = min(gi[-1], gj[-1])
    if hi - lo < 1e-6:
        return 9.0
    rs = np.linspace(lo, hi, n)
    return float(np.mean([1.0 - _interp_dir(gi, di, r) @ _interp_dir(gj, dj, r) for r in rs]))


def tmpl_dist(ti, tj):
    a = ti.ravel() - ti.mean(); b = tj.ravel() - tj.mean()
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── union-find ──────────────────────────────────────────────────────────────
class _UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ── stage 1: within-chunk bundling ──────────────────────────────────────────
def bundle_within_chunk(uf, F, prof_thr=0.10, tcorr_min=0.96, veto=None):
    """Merge over-split fragments of one neuron inside a chunk.  Uses MUTUAL-BEST
    pairing (not single-linkage connected components) so it cannot chain unrelated
    fibers into a blob: a fiber bundles only with the one partner that is also its
    own best match, and only when both the direction profile and the template agree
    tightly.  Iterated a few rounds so a neuron split into >2 fragments can still
    coalesce, while each merge stays a vetted pairwise decision."""
    grid, dirA, tmpl, _, chunk = F['grid'], F['dir'], F['template'], F['nspk'], F['chunk']
    by = defaultdict(list)
    for r in range(len(chunk)):
        if F['gid'][r] >= 0:
            by[int(chunk[r])].append(r)
    n_merged = 0
    for _ in range(3):                                   # let >2-way splits coalesce in rounds
        changed = False
        for rows in by.values():
            reps = sorted({uf.find(r) for r in rows})    # current bundle reps in this chunk
            if len(reps) < 2:
                continue
            geo = {r: (uf.find(r)) for r in reps}        # rep row -> itself
            D = np.full((len(reps), len(reps)), 9.0)
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    a, b = reps[i], reps[j]
                    if (1.0 - tmpl_dist(tmpl[a], tmpl[b])) > tcorr_min and \
                            profile_dist(grid[a], dirA[a], grid[b], dirA[b]) < prof_thr:
                        D[i, j] = D[j, i] = profile_dist(grid[a], dirA[a], grid[b], dirA[b])
            for i in range(len(reps)):
                j = int(np.argmin(D[i]))
                if D[i, j] >= 9.0:
                    continue
                if int(np.argmin(D[j])) != i:            # mutual-best only
                    continue
                if uf.find(reps[i]) != uf.find(reps[j]):
                    if veto is not None and veto(reps[i], reps[j]):   # refractory CCG veto (powered, no dip = 2 cells)
                        continue
                    uf.union(reps[i], reps[j]); n_merged += 1; changed = True
        if not changed:
            break
    return n_merged


# ── stage 2: cross-chunk geometry matching ──────────────────────────────────
def link_across_chunks(uf, F, prof_gate=0.18, tdist_gate=0.055, depth_gate=0.12,
                       q_half=300.0, thr=0.14, margin=0.7, max_gap=2, veto=None):
    """Cross-chunk matching by evolving geometry as ONE-TO-ONE FORWARD CHAINING:
    each fiber may match at most one successor and one predecessor, so units are
    simple chains (one fiber per chunk) that track smooth drift end-to-end — never
    tangled blobs.  Distance is noise-aware (template-dominant when sparse, where
    the direction profile is the sharp discriminator only when well sampled);
    consecutive chunks are matched first, then a one-chunk gap is bridged for
    fibers still unmatched forward.  depth gate scales with the chunk distance."""
    grid, dirA, tmpl, nspk, depth, chunk = (F['grid'], F['dir'], F['template'],
                                            F['nspk'], F['depth'], F['chunk'])
    by = defaultdict(list)
    for r in range(len(chunk)):
        if F['gid'][r] >= 0:
            by[int(chunk[r])].append(r)
    chunks = sorted(by)
    # representative per (chunk, seeded-component): chain on the richest member so
    # already-linked tracks and bundles extend as one node per chunk.
    rep = {}
    for c in chunks:
        for r in by[c]:
            key = (c, uf.find(r))
            if key not in rep or nspk[r] > nspk[rep[key]]:
                rep[key] = r
    nodes = defaultdict(list)                        # chunk -> representative rows
    for (c, _), r in rep.items():
        nodes[c].append(r)
    # live chunk occupancy per component: two tracks that share a chunk are two
    # distinct neurons (a neuron cannot appear twice in one chunk), so they must
    # never be chained together.
    rchunks = defaultdict(set)
    for r in range(len(chunk)):
        if F['gid'][r] >= 0:
            rchunks[uf.find(r)].add(int(chunk[r]))

    def cost(a, b, ddep_gate):
        td = tmpl_dist(tmpl[a], tmpl[b])
        if td > tdist_gate or abs(depth[a] - depth[b]) > ddep_gate:
            return 9.0
        pd = profile_dist(grid[a], dirA[a], grid[b], dirA[b])
        q = min(nspk[a], nspk[b]); wp = q / (q + q_half)   # 0 sparse (trust template) .. 1 dense (trust profile)
        if pd > prof_gate and q >= q_half:                 # well-sampled fibers must also agree on profile
            return 9.0
        return wp * pd + (1.0 - wp) * td

    matched_fwd, matched_bwd = {}, {}               # one-to-one constraint (by representative row)
    n_links = 0
    for gap in (1, 2):
        if gap > max_gap:
            break
        ddep_gate = depth_gate * gap
        for k in range(len(chunks) - gap):
            cA, cB = chunks[k], chunks[k + gap]
            A = [a for a in nodes.get(cA, []) if a not in matched_fwd]
            B = [b for b in nodes.get(cB, []) if b not in matched_bwd]
            if not A or not B:
                continue
            C = np.array([[cost(a, b, ddep_gate) for b in B] for a in A])
            for ia in range(len(A)):
                ib = int(np.argmin(C[ia])); best = C[ia, ib]
                if best >= thr:
                    continue
                sec = np.sort(C[ia])[1] if C.shape[1] > 1 else 9.0
                if best > margin * sec and best > 0.05:        # ambiguous: skip
                    continue
                if int(np.argmin(C[:, ib])) != ia:             # not reciprocal-best
                    continue
                a, b = A[ia], B[ib]
                if a in matched_fwd or b in matched_bwd:       # one-to-one: already taken
                    continue
                ra, rb = uf.find(a), uf.find(b)
                if ra != rb:
                    if rchunks[ra] & rchunks[rb]:              # temporally overlapping -> distinct neurons
                        continue
                    if veto is not None and veto(a, b):        # refractory CCG veto on the link
                        continue
                    merged = rchunks[ra] | rchunks[rb]
                    uf.union(a, b); n_links += 1
                    rchunks[uf.find(a)] = merged
                matched_fwd[a] = b; matched_bwd[b] = a
    return n_links


# ── driver ───────────────────────────────────────────────────────────────────
def relink(npz_path, prof_thr=0.10, tcorr_min=0.96, prof_gate=0.18, tdist_gate=0.055,
           depth_gate=0.12, q_half=300.0, thr=0.14, margin=0.7, max_gap=2,
           consec_guard=0.08, e2e_guard=0.35, clu_path=None, res=None, refrac=0,
           refrac_thr=0.3, refrac_min_exp=5.0, refrac_censor=0, verbose=True):
    """Returns (row2unit, oldgid2unit, units, report_rows). Seeds the union-find
    with the original gid grouping (preserving every original link), then adds
    within-chunk bundles and geometry links.

    If refrac>0 (samples) with clu_path+res given, a curation-INDEPENDENT
    refractory cross-correlogram veto guards every bundle/link decision: a
    proposed merge whose two spike trains coincide at chance level on their
    temporal overlap (two distinct neurons, no refractory dip) is BLOCKED, so a
    geometry/template match alone cannot fuse two co-active cells.  The test is
    power-aware (overlap_refractory_gate) and ABSTAINS where rates are too low,
    so it never vetoes blindly -- it only ever removes false linkages."""
    z = np.load(npz_path, allow_pickle=True)
    F = {k: z[k] for k in ('gid', 'chunk', 'grid', 'dir', 'template', 'nspk', 'depth', 'radius')}
    G = len(F['gid'])
    uf = _UF(G)
    # seed: preserve original cross-chunk links
    first = {}
    for r in range(G):
        g = int(F['gid'][r])
        if g < 0:
            continue
        if g in first:
            uf.union(first[g], r)
        else:
            first[g] = r
    n_seed = G - len({uf.find(r) for r in range(G) if F['gid'][r] >= 0})

    # ── optional refractory veto (curation-independent false-linkage guard) ──
    n_veto = [0]
    if refrac and refrac > 0 and clu_path is not None and res is not None:
        _, clu_ids = nio.read_clu_file(clu_path)
        clu_ids = np.asarray(clu_ids).ravel()
        res = np.asarray(res).ravel()
        rowgid = F['gid'].astype(int)
        trains = {}
        for g in np.unique(rowgid[rowgid >= 0]):
            t = np.sort(res[clu_ids == g + 1].astype(np.int64))   # clu id = gid+1; 0 = noise
            if t.size:
                trains[int(g)] = t

        def _comp_times(root):
            gids = {int(rowgid[r]) for r in range(G) if rowgid[r] >= 0 and uf.find(r) == root}
            ts = [trains[g] for g in gids if g in trains]
            return np.sort(np.concatenate(ts)) if ts else np.empty(0, np.int64)

        def veto(a, b):
            ta = _comp_times(uf.find(a)); tb = _comp_times(uf.find(b))
            if ta.size == 0 or tb.size == 0:
                return False
            g = cg.overlap_refractory_gate(ta, tb, refrac, thr=refrac_thr,
                                           min_exp=refrac_min_exp, censor=refrac_censor)
            blocked = (g["verdict"] == "veto")
            if blocked:
                n_veto[0] += 1
            return blocked
    else:
        veto = None

    n_bundle = bundle_within_chunk(uf, F, prof_thr, tcorr_min, veto=veto)
    n_link = link_across_chunks(uf, F, prof_gate, tdist_gate, depth_gate, q_half, thr, margin, max_gap, veto=veto)

    # contiguous unit ids
    roots = {}
    row2unit = np.full(G, -1, int)
    for r in range(G):
        if F['gid'][r] < 0:
            continue
        rt = uf.find(r); roots.setdefault(rt, len(roots)); row2unit[r] = roots[rt]
    n_units = len(roots)
    oldgid2unit = {}
    for r in range(G):
        if F['gid'][r] >= 0:
            oldgid2unit[int(F['gid'][r])] = int(row2unit[r])

    # per-unit drift report
    members = defaultdict(list)
    for r in range(G):
        if row2unit[r] >= 0:
            members[int(row2unit[r])].append(r)
    rep = []
    for u, rs in members.items():
        rs = sorted(rs, key=lambda r: F['chunk'][r])
        cs = [int(F['chunk'][r]) for r in rs]
        steps = [(a, b) for a, b in zip(rs[:-1], rs[1:]) if F['chunk'][b] > F['chunk'][a]]
        max_step = max((tmpl_dist(F['template'][a], F['template'][b]) for a, b in steps), default=0.0)
        e2e_t = tmpl_dist(F['template'][rs[0]], F['template'][rs[-1]]) if len(rs) > 1 else 0.0
        e2e_d = abs(float(F['depth'][rs[0]] - F['depth'][rs[-1]])) if len(rs) > 1 else 0.0
        rep.append(dict(unit=u, n_oldgid=len({int(F['gid'][r]) for r in rs}),
                        chunk0=cs[0], chunk1=cs[-1], n_chunks=len(set(cs)),
                        n_fibers=len(rs), spikes=int(sum(int(F['nspk'][r]) for r in rs)),
                        e2e_tmpl=round(e2e_t, 4), e2e_depth=round(e2e_d, 4),
                        max_consec_step=round(max_step, 4),
                        suspect=int(max_step > consec_guard or e2e_t > e2e_guard)))
    rep.sort(key=lambda d: -d['n_chunks'])
    if verbose:
        nmc = sum(1 for d in rep if d['n_chunks'] >= 2)
        print(f"[relink] {G} fibers | seed links={n_seed} + bundles={n_bundle} + geometry links={n_link}"
              + (f" | refractory vetoes={n_veto[0]}" if veto is not None else ""))
        print(f"[relink] units: {len(first)} (original) -> {n_units}   multi-chunk units={nmc}   "
              f"suspect(step>{consec_guard})={sum(d['suspect'] for d in rep)}")
    return row2unit, oldgid2unit, n_units, rep


def _load_res(path):
    """Per-spike sample times from a .res: ascii (one int per line) or binary int64/int32.
    Picks the interpretation that yields a non-negative, non-decreasing array."""
    def _ok(a):
        return a.size > 0 and a.min() >= 0 and bool(np.all(np.diff(a) >= 0))
    try:
        a = np.loadtxt(path, dtype=np.int64).ravel()
        if _ok(a):
            return a
    except Exception:
        pass
    for dt in (np.int64, np.int32):
        a = np.fromfile(path, dtype=dt)
        if _ok(a):
            return a
    raise ValueError(f"could not parse .res times from {path} (tried ascii, int64, int32)")


def rewrite_clu(clu_in, clu_out, oldgid2unit):
    """Remap an existing binary .clu (int32 nClusters header + ids; id=gid+1,
    0=noise) onto merged unit ids, renumbered contiguously from 1."""
    _, ids = nio.read_clu_file(clu_in)                  # per-spike (gid+1); 0 = noise
    # map old id -> new id (noise stays 0); units renumbered 1..K in first-seen order
    nxt = 1; out = np.empty_like(ids)
    seen = {}
    for i, v in enumerate(ids):
        v = int(v)
        if v == 0:
            out[i] = 0; continue
        u = oldgid2unit.get(v - 1, None)
        if u is None:                               # gid absent from .fibers: keep distinct
            key = ('g', v)
        else:
            key = ('u', u)
        if key not in seen:
            seen[key] = nxt; nxt += 1
        out[i] = seen[key]
    n_units = nxt - 1
    nio.write_clu_file(clu_out, out.astype(np.int32), n_clusters=n_units + 1)
    return n_units, len(ids)


def write_report(rep, path):
    cols = ['unit', 'n_oldgid', 'chunk0', 'chunk1', 'n_chunks', 'n_fibers', 'spikes',
            'e2e_tmpl', 'e2e_depth', 'max_consec_step', 'suspect']
    with open(path, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for d in rep:
            f.write('\t'.join(str(d[c]) for c in cols) + '\n')


def main():
    ap = argparse.ArgumentParser(
        description="Geometry-aware re-bundling/re-linking of a .fibers run (no re-run needed).")
    ap.add_argument("fibers", help="path to <base>.fibers.<method>.<elec>.npz")
    ap.add_argument("--clu", default=None, help="existing .clu to remap (gid+1; 0=noise)")
    ap.add_argument("--out", default=None, help="output .clu (default <clu>_relinked)")
    ap.add_argument("--report", default=None, help="per-unit drift report TSV")
    ap.add_argument("--prof-thr", type=float, default=0.10);  ap.add_argument("--tcorr-min", type=float, default=0.96)
    ap.add_argument("--prof-gate", type=float, default=0.18);  ap.add_argument("--tdist-gate", type=float, default=0.055)
    ap.add_argument("--depth-gate", type=float, default=0.12); ap.add_argument("--q-half", type=float, default=300.0)
    ap.add_argument("--thr", type=float, default=0.14);        ap.add_argument("--margin", type=float, default=0.7)
    ap.add_argument("--max-gap", type=int, default=2);         ap.add_argument("--consec-guard", type=float, default=0.08)
    ap.add_argument("--e2e-guard", type=float, default=0.35)
    ap.add_argument("--refrac-ms", type=float, default=0.0,
                    help="DEFAULT OFF. >0 enables a curation-independent refractory cross-correlogram "
                         "veto on every bundle/link: a merge whose two trains coincide at chance level on "
                         "their temporal overlap (two neurons, no refractory dip) is blocked. Needs --res. "
                         "Power-aware: ABSTAINS at low firing rates, only ever removes false linkages.")
    ap.add_argument("--res", default=None, help="per-spike .res (sample times) aligned to --clu; required for --refrac-ms")
    ap.add_argument("--sr", type=float, default=None, help="sample rate (Hz) for --refrac-ms (e.g. 32552)")
    ap.add_argument("--refrac-thr", type=float, default=0.3, help="coincidence ratio above which the overlap is 'two neurons' (default 0.3)")
    ap.add_argument("--refrac-min-exp", type=float, default=5.0, help="min expected coincidences for the test to be powered (default 5)")
    ap.add_argument("--refrac-censor-ms", type=float, default=0.0, help="censor window (ms) to drop duplicate detections of the same spike (default 0)")
    a = ap.parse_args()
    res = refrac = censor = None
    if a.refrac_ms and a.refrac_ms > 0:
        if not (a.clu and a.res and a.sr):
            ap.error("--refrac-ms requires --clu, --res and --sr")
        res = _load_res(a.res)
        refrac = cg.refrac_samples(a.refrac_ms, a.sr)
        censor = cg.refrac_samples(a.refrac_censor_ms, a.sr)
    row2unit, oldgid2unit, n_units, rep = relink(
        a.fibers, a.prof_thr, a.tcorr_min, a.prof_gate, a.tdist_gate, a.depth_gate,
        a.q_half, a.thr, a.margin, a.max_gap, a.consec_guard, a.e2e_guard,
        clu_path=a.clu, res=res, refrac=(refrac or 0), refrac_thr=a.refrac_thr,
        refrac_min_exp=a.refrac_min_exp, refrac_censor=(censor or 0))
    if a.report:
        write_report(rep, a.report); print(f"[relink] report -> {a.report}")
    if a.clu:
        out = a.out or (a.clu + "_relinked")
        k, n = rewrite_clu(a.clu, out, oldgid2unit)
        print(f"[relink] {n} spikes remapped -> {k} units -> {out}")


if __name__ == "__main__":
    main()
