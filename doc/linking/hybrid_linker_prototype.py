"""Hybrid cross-chunk linker PROTOTYPE (standalone validation, not packaged).

Backbone: overlap-anchor mutual-majority (the trusted, drift-proof link).
Fallback: for fibers overlap can't link (too sparse), bridge global tracks across
chunk gaps using a DRIFT-PREDICTED depth (coherent Δz(t) estimated from the
well-linked units) GATED by a signature match, and allowed to REFUSE.
"""
import numpy as np
rng = np.random.RandomState(1)

# ── synthetic coherent-drift session ─────────────────────────────────────────
NCH, CHUNK, OVL, D = 8, 4.0, 1.0, 12          # chunks, min/chunk, overlap min, sig dims
def dz(tmin):                                  # coherent probe drift Δz(t), µm (smooth)
    x = tmin / (NCH * CHUNK)
    return 30.0 * x + 4.0 * np.sin(2.5 * x * np.pi)
def sig(): 
    v = rng.randn(D); return v / np.linalg.norm(v)

# units: (z0 µm, signature, rate Hz, chunk_start, chunk_end_inclusive)
units = {}
for i in range(6):                             # 6 high-rate, link cleanly by overlap
    units[i] = dict(z0=80 + 60 * i, s=sig(), rate=rng.uniform(4, 8), c0=0, c1=NCH - 1)
units[100] = dict(z0=110, s=sig(), rate=0.045, c0=0, c1=NCH - 1)  # SPARSE: <8 anchors/overlap -> fails
xz = 200.0
xsig = sig()
yb = sig(); ysig = yb - (yb @ xsig) * xsig; ysig = ysig / np.linalg.norm(ysig)   # Y waveform distinct from X
print(f'cos(X,Y signature) = {float(xsig @ ysig):+.3f}   (different neuron -> distinct waveform)\n')
units[200] = dict(z0=xz, s=xsig, rate=3.0, t0=0.0,  t1=13.0)            # X: vanishes before the 3|4 overlap
units[300] = dict(z0=xz, s=ysig, rate=3.0, t0=17.0, t1=NCH * CHUNK)     # Y: appears after it, SAME depth path
GT = len(units)

# emit spikes (time, true unit, depth, signature)
spk_t, spk_u, spk_z, spk_s = [], [], [], []
for u, p in units.items():
    t0 = p.get("t0", p.get("c0", 0) * CHUNK); t1 = p.get("t1", (p.get("c1", NCH - 1) + 1) * CHUNK)
    n = rng.poisson(p["rate"] * (t1 - t0) * 60)
    ts = np.sort(rng.uniform(t0, t1, n))
    for t in ts:
        spk_t.append(t); spk_u.append(u)
        spk_z.append(p["z0"] + dz(t) + rng.randn() * 2.0)
        spk_s.append(p["s"] + rng.randn(D) * 0.15)
spk_t = np.array(spk_t); spk_u = np.array(spk_u); spk_z = np.array(spk_z); spk_s = np.array(spk_s)
order = np.argsort(spk_t); spk_t, spk_u, spk_z, spk_s = spk_t[order], spk_u[order], spk_z[order], spk_s[order]

# per-chunk fibers with SCRAMBLED local ids (as within-chunk clustering would give)
chunk_fibers = {}                # (chunk, localid) -> dict(spk idx, depth, sig, true)
fib_of = {}                      # (chunk, true unit) -> (chunk, localid)
for c in range(NCH):
    lo, hi = c * CHUNK - OVL, (c + 1) * CHUNK + OVL
    m = (spk_t >= max(lo, 0)) & (spk_t < hi)
    idx = np.where(m)[0]
    perm = rng.permutation(np.unique(spk_u[idx]))           # scramble label numbering
    for lid, u in enumerate(perm):
        sel = idx[spk_u[idx] == u]
        chunk_fibers[(c, lid)] = dict(idx=sel, depth=np.median(spk_z[sel]),
                                      sig=spk_s[sel].mean(0) / np.linalg.norm(spk_s[sel].mean(0)),
                                      true=int(u), tmid=(c + 0.5) * CHUNK)
        fib_of[(c, u)] = (c, lid)

# ── overlap-anchor backbone (mutual-majority over shared physical spikes) ─────
MIN_ANCHOR = 8
parent = {k: k for k in chunk_fibers}
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b): parent[find(a)] = find(b)

for c in range(NCH - 1):
    lo, hi = (c + 1) * CHUNK - OVL, (c + 1) * CHUNK + OVL     # overlap region between c and c+1
    ov = np.where((spk_t >= lo) & (spk_t < hi))[0]
    for u in np.unique(spk_u[ov]):
        nshare = int((spk_u[ov] == u).sum())
        if nshare >= MIN_ANCHOR and (c, u) in fib_of and (c + 1, u) in fib_of:
            union(fib_of[(c, u)], fib_of[(c + 1, u)])        # same physical spikes => same fiber

def globals_from_uf():
    g = {}
    for k in chunk_fibers: g.setdefault(find(k), []).append(k)
    return list(g.values())

def track(members):                                          # ordered (chunk, depth, sig)
    ms = sorted(members, key=lambda k: k[0])
    return [(k[0], chunk_fibers[k]["depth"], chunk_fibers[k]["sig"]) for k in ms]

# drift Δz per chunk step, estimated from overlap-linked MULTI-chunk globals (the well-sampled ones)
step = np.zeros(NCH)
for g in globals_from_uf():
    tr = track(g)
    if len(tr) >= 2:
        for (ca, za, _), (cb, zb, _) in zip(tr[:-1], tr[1:]):
            if cb == ca + 1: step[cb] += zb - za
cnt = np.zeros(NCH)
for g in globals_from_uf():
    tr = track(g)
    for (ca, _, _), (cb, _, _) in zip(tr[:-1], tr[1:]):
        if cb == ca + 1: cnt[cb] += 1
dstep = np.where(cnt > 0, step / np.maximum(cnt, 1), 0.0)     # per-chunk median-ish drift step

# ── signature-gated, drift-predicted continuity fallback ──────────────────────
DEPTH_GATE, SIG_THR, MAXGAP = 14.0, 0.6, 2
def continuity_merge(use_sig):
    for k in chunk_fibers: parent[k] = k                      # rebuild from overlap backbone
    for c in range(NCH - 1):
        lo, hi = (c + 1) * CHUNK - OVL, (c + 1) * CHUNK + OVL
        ov = np.where((spk_t >= lo) & (spk_t < hi))[0]
        for u in np.unique(spk_u[ov]):
            if int((spk_u[ov] == u).sum()) >= MIN_ANCHOR and (c, u) in fib_of and (c + 1, u) in fib_of:
                union(fib_of[(c, u)], fib_of[(c + 1, u)])
    bridges = []                                              # (ze, cb, zb, ce, correct)
    ends, starts = {}, {}                                     # per root: endpoint (chunk, depth, sig, true)
    for g in globals_from_uf():
        ms = sorted(g, key=lambda k: k[0]); r = find(g[0])
        fe, fb = chunk_fibers[ms[-1]], chunk_fibers[ms[0]]
        ends[r]   = (ms[-1][0], fe["depth"], fe["sig"], fe["true"])
        starts[r] = (ms[0][0],  fb["depth"], fb["sig"], fb["true"])
    for rb in sorted(starts, key=lambda r: starts[r][0]):     # process by start chunk
        cb, zb, sb, tb = starts[rb]
        best = None
        for re in list(ends):
            if find(re) == find(rb): continue
            ce, ze, se, te = ends[re]
            gap = cb - ce
            if not (1 <= gap <= MAXGAP): continue
            pred = ze + sum(dstep[ce + 1:cb + 1])             # drift-predicted depth at cb
            scos = float(se @ sb)
            if abs(pred - zb) <= DEPTH_GATE * gap and ((scos >= SIG_THR) or not use_sig):
                score = abs(pred - zb) - 5 * scos
                if best is None or score < best[0]: best = (score, re, ce, ze, te)
        if best is not None:
            _, re, ce, ze, te = best
            bridges.append((ze, cb, zb, ce, te == tb))
            union(rb, re)
    return globals_from_uf(), bridges


def report(glist, tag):
    mixing = 0; comp = {}
    for g in glist:
        trues = [chunk_fibers[k]["true"] for k in g]
        if len(set(trues)) > 1: mixing += 1
        for t in set(trues): comp.setdefault(t, set()).add(id(g))
    sparse_pieces = sum(1 for g in glist if all(chunk_fibers[k]["true"] == 100 for k in g))
    xy_together = any({200, 300}.issubset({chunk_fibers[k]["true"] for k in g}) for g in glist)
    print(f"{tag:28s} globals={len(glist):3d}  id-mixing={mixing}  "
          f"sparse#100 in {sparse_pieces} piece(s)  X&Y merged={xy_together}")
    return len(glist), mixing, xy_together

print(f"ground-truth units: {GT}   (6 high-rate, 1 sparse #100, X#200 vanishes@3, Y#300 appears@4 same depth-path)\n")
ov = report(globals_from_uf(), "overlap-only (backbone)")
g_hg, br_hg = continuity_merge(True);  hg = report(g_hg, "hybrid (signature-gated)")
g_ng, br_ng = continuity_merge(False); ng = report(g_ng, "hybrid (NO signature gate)")
print()
print("PASS" if (hg[0] == GT and hg[1] == 0 and not hg[2] and ng[2]) else "CHECK",
      "— sig-gated recovers GT with 0 id-mixing & keeps X/Y apart; no-gate wrongly merges X&Y")
np.save("/tmp/proto_state.npy", dict(dstep=dstep), allow_pickle=True)


# ── figure: depth vs chunk, bridges drawn ───────────────────────────────────
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
uids = sorted(units); col = {u: cm.tab10(i % 10) for i, u in enumerate(uids)}
def panel(ax, glist, bridges, title):
    for k, f in chunk_fibers.items():
        ax.scatter(k[0], f["depth"], color=col[f["true"]], s=26, zorder=3)
    for g in glist:                                  # final-global tracks (overlap + continuity)
        tr = sorted(g, key=lambda k: k[0])
        xs = [k[0] for k in tr]; ys = [chunk_fibers[k]["depth"] for k in tr]
        ax.plot(xs, ys, color="0.6", lw=1.0, zorder=2)
    for (ze, cb, zb, ce, ok) in bridges:             # continuity bridges
        ax.plot([ce, cb], [ze, zb], color=("#1a9850" if ok else "#d73027"),
                lw=2.6, ls=("-" if ok else "--"), zorder=4)
    ax.set_title(title, fontsize=10); ax.set_xlabel("chunk"); ax.set_ylabel("depth (µm)")
    ax.invert_yaxis()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
panel(axes[0], g_hg, br_hg, f"signature-gated  →  {len(g_hg)} globals, {hg[1]} id-mixing  (= ground truth {GT})")
panel(axes[1], g_ng, br_ng, f"NO signature gate  →  {len(g_ng)} globals, {ng[1]} id-mixing  (X↔Y mis-linked)")
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([0],[0],color="#1a9850",lw=2.6,label="continuity link (correct)"),
                        Line2D([0],[0],color="#d73027",lw=2.6,ls="--",label="continuity link (WRONG: X↔Y)"),
                        Line2D([0],[0],color="0.6",lw=1.0,label="global track")],
               fontsize=8, loc="lower right")
fig.suptitle("Hybrid cross-chunk linker — overlap backbone + drift-predicted, signature-gated continuity\n"
             "colour = true unit; sparse unit #100 (~0.045 Hz) fails overlap and is recovered by continuity; "
             "X vanishes @ chunk 3, Y appears @ chunk 4 on the SAME drift-predicted depth path",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/home/claude/hybrid_linker_prototype.png", dpi=120, bbox_inches="tight")
print("\nwrote hybrid_linker_prototype.png")
