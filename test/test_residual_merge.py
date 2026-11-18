"""fiber-residual-merge: the reverse residual strip as a confidence-ordered agglomeration.

Synthetic raw .spk, four channels.  Units:
  A   a cell at site 0;  in three fragments (2, 3, 4)
  B   another cell at the SAME site, a different shape (its template cosine with A is high);  5, 6
  C   A's shape at the NEIGHBOURING site (same time course, the footprint moved one channel);  7
  A'  A at 0.65 x amplitude -- same footprint, same shape;  8

  * the fragments of A fuse, the fragments of B fuse, and nothing else does: B is kept out by the
    on-channel term, C by the off-channel term;
  * A' is kept apart from A: the residual is gain-FREE, but it is normalised by the spike's own
    energy, so a population at 0.65 x sits at ~2.4 x A's noise residual -- outside A's tails;
  * the cleaner cluster's tails decide: a fragment that is half A and half B is not merged into A,
    though under ITS OWN (inflated) tails it would accept all of A;
  * the refractory veto: two fragments of one shape whose trains coincide far above a refractory
    cell's are not merged;
  * --shift: a fragment stored 2 samples late merges only when the pair may be compared at its best lag;
  * re-templating: the result does not depend on the order the fragments are listed in;
  * the contract fiber-xcorr-merge's downstream expects (mapping, counts) and the .tsv rows;
  * the CLI on a small session: .clu, .clc/.clp carried, the .rsm table, a bad --merge-align refused.
"""
import os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_residual_merge as rm   # noqa: E402

fails = 0
def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


T, C, PK = 32, 4, 16
_t = np.arange(T)
SHAPE_A = -np.exp(-0.5 * ((_t - PK) / 1.4) ** 2) + 0.30 * np.exp(-0.5 * ((_t - PK - 5) / 2.5) ** 2)
SHAPE_B = -np.exp(-0.5 * ((_t - PK) / 2.6) ** 2) + 0.55 * np.exp(-0.5 * ((_t - PK - 8) / 3.0) ** 2)
FOOT0 = np.array([1.0, 0.6, 0.3, 0.1])
FOOT1 = np.array([0.55, 1.0, 0.6, 0.25])
TA = SHAPE_A[:, None] * FOOT0[None, :] * 1500
TB = SHAPE_B[:, None] * FOOT0[None, :] * 1500
TC = SHAPE_A[:, None] * FOOT1[None, :] * 1500
NOISE = 60.0


def draw(rng, tm, n, gain_sd=0.08):
    """n spikes of template tm: a per-spike gain (a cell's amplitude is not constant) plus white noise."""
    g = 1.0 + gain_sd * rng.standard_normal(n)
    return g[:, None, None] * tm[None] + rng.normal(0, NOISE, (n, T, C))


def build(rng, plan):
    """plan: [(label, [(template, n), ...])] -> (spk, ids, unit tag per spike, res)."""
    W, ids, unit = [], [], []
    for lab, parts in plan:
        for name, tm, n in parts:
            W.append(draw(rng, tm, n)); ids += [lab] * n; unit += [name] * n
    spk = np.clip(np.rint(np.concatenate(W)), -32768, 32767).astype(np.int16)
    res = np.sort(rng.choice(np.arange(10 ** 8), len(ids), replace=False))
    perm = rng.permutation(len(ids))
    return spk[perm], np.array(ids)[perm], np.array(unit)[perm], res


def run(spk, ids, res, labs=None, **kw):
    labs = sorted({int(x) for x in ids}) if labs is None else labs
    idx0 = [np.flatnonzero(ids == u) for u in labs]; times0 = [np.sort(res[i]) for i in idx0]
    st = {}
    args = dict(acc_thr=0.7, cos=0.5, shift=0, refrac=0, cap=1000, ref_sample=PK, rng=np.random.default_rng(0))
    args.update(kw)
    mp, nm, nv = rm.agglomerate(spk, labs, idx0, times0, float(res.max()), stats=st, **args)
    out = ids.copy()
    for u in labs:
        out[ids == u] = mp[u]
    return out, mp, nm, nv, st


def groups(out, ids):
    """The partition of the input labels the merge produced, as a sorted list of sorted tuples."""
    g = {}
    for lab in np.unique(ids):
        g.setdefault(int(out[ids == lab][0]), []).append(int(lab))
    return sorted(tuple(sorted(v)) for v in g.values())


def main():
    rng = np.random.default_rng(7)
    plan = [(2, [("A", TA, 400)]), (3, [("A", TA, 300)]), (4, [("A", TA, 250)]),
            (5, [("B", TB, 350)]), (6, [("B", TB, 300)]),
            (7, [("C", TC, 350)]),
            (8, [("A'", 0.65 * TA, 350)])]
    spk, ids, unit, res = build(rng, plan)
    cosAB = float((TA * TB).sum() / np.linalg.norm(TA) / np.linalg.norm(TB))
    cosAC = float((TA * TC).sum() / np.linalg.norm(TA) / np.linalg.norm(TC))
    check(cosAB > 0.5 and cosAC > 0.5, f"the fixture: B and C are candidates for A by cosine ({cosAB:.2f}, {cosAC:.2f})")

    out, mp, nm, nv, st = run(spk, ids, res)
    g = groups(out, ids)
    check(g == [(2, 3, 4), (5, 6), (7,), (8,)], f"A's fragments fuse, B's fuse, C and A' stand alone ({g})")
    check(nm == 3 and nv == 0, f"three unions, no veto ({nm}, {nv})")
    check(set(mp) == {2, 3, 4, 5, 6, 7, 8} and all(mp[mp[u]] == mp[u] for u in mp),
          "the mapping covers every input id and points at group roots (fiber-xcorr-merge's contract)")
    merged = [p for p in st["pairs"] if p[-1] == "merge"]
    check(len(merged) == 3 and all(p[2] >= 0.7 for p in merged),
          f"every union is logged with its acceptance ({[round(p[2], 2) for p in merged]})")
    lines = rm.table_lines(st["pairs"])
    check(lines[0].split("\t") == ["a", "b", "acceptance", "cos", "lag", "n_a", "n_b", "verdict"]
          and len(lines) == len(st["pairs"]) + 1, "the table: a header and one row per pair")

    # which term keeps B and C out: the acceptance of each by A, per direction's worst
    ea, eb, ec, eh = (rm.entry(np.asarray(spk[ids == u], float)) for u in (2, 5, 7, 8))
    acc = {n: rm.acceptance(ea, e) for n, e in (("B", eb), ("C", ec), ("A'", eh))}
    check(all(v < 0.3 for v in acc.values()),
          "B (same site, other shape), C (same shape, next site) and A' (0.65 x) are far below the threshold "
          + str({k: round(v, 2) for k, v in acc.items()}))
    import fiber_kit.fiber_consolidate as fc
    tb = fc.resid_terms(np.asarray(spk[ids == 5], float), ea["T"])
    tc = fc.resid_terms(np.asarray(spk[ids == 7], float), ea["T"])
    check(np.median(tb["onr"][tb["ok"]]) > ea["thr"][0] and np.median(tc["offr"][tc["ok"]]) > ea["thr"][1],
          "B is outside A's ON-channel tail and C outside A's OFF-channel tail")

    # order independence (re-templating, keyed samples)
    perm_labs = [6, 3, 8, 2, 7, 5, 4]
    out2, *_ = run(spk, ids, res, labs=perm_labs)
    check(groups(out2, ids) == g, "the partition does not depend on the order the clusters are listed in")

    # the cleaner cluster's tails
    spk2, ids2, unit2, res2 = build(np.random.default_rng(11),
                                    [(2, [("A", TA, 400)]), (3, [("A", TA, 200), ("B", TB, 200)])])
    e2 = rm.entry(np.asarray(spk2[ids2 == 2], float)); e3 = rm.entry(np.asarray(spk2[ids2 == 3], float))
    own = fc.resid_terms(e2["W"], e3["T"])
    own_acc = float((own["ok"] & (own["onr"] <= e3["thr"][0]) & (own["offr"] <= e3["thr"][1])).mean())
    out3, *_ = run(spk2, ids2, res2)
    check(e3["thr"][0] > 2 * e2["thr"][0], f"a half-A half-B fragment's own on-channel tail is inflated "
                                           f"({e3['thr'][0]:.3f} vs the clean fragment's {e2['thr'][0]:.3f})")
    check(groups(out3, ids2) == [(2,), (3,)],
          f"so it is judged under the CLEAN fragment's tails and not merged (under its own it would accept "
          f"{100 * own_acc:.0f} % of A)")

    # refractory veto
    r = np.random.default_rng(3)
    spk4, ids4, _, _ = build(r, [(2, [("A", TA, 600)]), (3, [("A", TA, 600)])])
    dur = 30000 * 60                                        # one minute at 30 kHz: ~10 Hz each
    t_a = np.sort(r.choice(np.arange(dur), 600, replace=False))
    t_b = np.clip(t_a + r.integers(-20, 21, 600), 0, dur - 1)    # locked to A within 0.7 ms: not one refractory cell
    res4 = np.empty(1200, np.int64); res4[ids4 == 2] = t_a; res4[ids4 == 3] = np.sort(t_b)
    idx0 = [np.flatnonzero(ids4 == u) for u in (2, 3)]
    st4 = {}
    mp4, nm4, nv4 = rm.agglomerate(spk4, [2, 3], idx0, [np.sort(res4[i]) for i in idx0], float(dur), refrac=60,
                                   refrac_thr=0.3, refrac_min_exp=5.0, ref_sample=PK, stats=st4)
    mp5, nm5, _ = rm.agglomerate(spk4, [2, 3], idx0, [np.sort(res4[i]) for i in idx0], float(dur), refrac=0,
                                 ref_sample=PK)
    check(nm5 == 1, "two fragments of one shape merge on the residual alone")
    check(nm4 == 0 and nv4 == 1 and st4["pairs"][0][-1] == "refractory-veto",
          "and are vetoed when their trains coincide inside the refractory window; the table says so")

    # --shift
    spk6, ids6, _, res6 = build(np.random.default_rng(5), [(2, [("A", TA, 400)]), (3, [("A", np.roll(TA, 2, axis=0), 400)])])
    g0 = groups(run(spk6, ids6, res6)[0], ids6)
    out7, _, _, _, st7 = run(spk6, ids6, res6, shift=3)
    check(g0 == [(2,), (3,)], "a fragment stored 2 samples late is not merged at the stored alignment")
    check(groups(out7, ids6) == [(2, 3)] and abs(st7["pairs"][0][4]) == 2,
          f"--shift 3 compares the pair at its best lag ({st7['pairs'][0][4]}) and merges it")

    # a cluster with no usable template stays as it is, and is counted
    spk8, ids8, _, res8 = build(np.random.default_rng(9), [(2, [("A", TA, 300)]), (3, [("A", TA, 300)])])
    spk8 = spk8.copy(); spk8[ids8 == 3] = 0
    out8, _, nm8, _, st8 = run(spk8, ids8, res8)
    check(nm8 == 0 and st8["unjudged"] == 1, "a flat cluster is left alone and reported as unjudged")

    # the CLI
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "ses")
        n = spk.shape[0]
        from fiber_kit import neuro_io as nio
        from fiber_kit.fiber_refiberize import FiberHierarchy
        nio.write_res(base, 1, res)
        spk.tofile(base + ".spk.standard.1")
        nio.write_clu(base, 1, ids, variant="stderiv", tag="lnk")
        FiberHierarchy(ids.astype(np.int64), {int(u): int(u) for u in np.unique(ids)}).save(   # one child per fiber
            base, 1, variant="stderiv", tag="lnk", renumber=False, backup=False)
        with open(base + ".yaml", "w") as fh:
            fh.write("acquisitionSystem: {nBits: 16, nChannels: 4, samplingRate: 30000, voltageRange: 20, amplification: 1000, offset: 0}\n"
                     "fieldPotentials: {lfpSamplingRate: 1250}\n"
                     "spikeDetection:\n  channelGroups:\n"
                     f"    - channels: [0, 1, 2, 3]\n      nSamples: {T}\n      peakSampleIndex: {PK}\n      nFeatures: 3\n")
        env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
        for k in [k for k in env if k.startswith("FK_")]:
            del env[k]
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_residual_merge", base, "1", "--clu-stage", "lnk",
                            "--out-stage", "rsm", "--emit-drift", "0"], capture_output=True, text=True, env=env, cwd=d)
        ok = r.returncode == 0
        check(ok, "the CLI runs on a small session" + ("" if ok else "\n" + (r.stdout + r.stderr)[-1500:]))
        if ok:
            _, new = nio.read_clu_at(base, 1, variant="stderiv", tag="rsm", n_spikes=n)
            check(groups(new, ids) == g, f"and writes the same partition as the function ({groups(new, ids)})")
            check(os.path.exists(base + ".clc.stderiv.1.rsm") and os.path.exists(base + ".clp.stderiv.1.rsm"),
                  "the .clc/.clp hierarchy is carried to the out-stage")
            h = FiberHierarchy.load(base, 1, variant="stderiv", tag="rsm")
            check({int(c): int(p) for c, p in h.parent.items()} == {int(u): int(mp[int(u)]) for u in np.unique(ids)},
                  "each child's parent went through the merge's mapping")
            with open(base + ".rsm.1.rsm.tsv") as fh:
                rows = fh.read().strip().split("\n")
            check(rows[0].startswith("a\tb\tacceptance") and len(rows) == 4, f"the .rsm table: header + 3 unions ({len(rows) - 1} rows)")
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_residual_merge", "nosuch", "1", "--merge-align", "bogus"],
                           capture_output=True, text=True, env=env, cwd=d)
        check(r.returncode != 0 and "--merge-align must be stored, peak or klusters" in (r.stdout + r.stderr),
              "a bad --merge-align is refused before anything is read")

    print(f"test_residual_merge: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
