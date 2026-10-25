"""Over-split before false merge; cluster 0 only for derivative outliers.

  * fiber_session._complete_link: A~B and B~C do not weld A to C unless A~C is itself a pair (union-find
    on the raw edges did); a triangle still merges; the best pair goes first;
  * fiber_session._merge_vetoes: the refractory veto removes a pair whose trains coincide at chance level,
    and keeps one with a refractory gap (power-aware);
  * fiber-xcorr-merge's knn split: a bucket the knn assigns to a reference cluster folds into it only if
    the pair passes the merge's own tests -- a bucket assigned to a dissimilar neighbour becomes a new
    cluster instead (counted in knn_fold_vetoes); assigned to a similar one it still folds;
  * fiber-session on a session whose span is an exact multiple of the chunk length: the last spike is in
    a fiber, not cluster 0; spikes of a chunk too small to sort are an unsorted fiber, not cluster 0.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import fiber_session as fs          # noqa: E402
from fiber_kit import fiber_xcorr_merge as fx      # noqa: E402
from fiber_kit import neuro_io as nio              # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def complete_link():
    print("complete linkage")
    sc = {(0, 1): 0.995, (1, 2): 0.993}
    out = fs._complete_link(list(sc), sc)
    check(out == [(0, 1)], f"A~B, B~C: only the best pair joins, C stays apart ({out})")
    sc[(0, 2)] = 0.991
    out = fs._complete_link(list(sc), sc)
    check(len(out) == 2 and {x for e in out for x in e} == {0, 1, 2}, f"a triangle merges all three ({out})")
    sc = {(0, 1): 0.991, (1, 2): 0.999}
    out = fs._complete_link(list(sc), sc)
    check(out == [(1, 2)], f"the best pair goes first ({out})")
    check(fs._complete_link([], {}) == [], "no edges, no merges")


def vetoes():
    print("refractory veto")
    rng = np.random.default_rng(1)
    sr = 20000
    dur = 600 * sr
    a = np.sort(rng.choice(dur, 3000, replace=False))
    b = np.sort(rng.choice(dur, 3000, replace=False))                # independent: chance coincidences
    c = np.sort(rng.choice(np.arange(0, dur, 200), 3000, replace=False))   # the same cell as d: never within 5 ms
    d = c + 100
    res = np.concatenate([a, b, c, d])
    fine = np.repeat([0, 1, 2, 3], 3000)
    kw = dict(refrac_ms=2.0, refrac_thr=0.3, refrac_min_exp=5.0, refrac_censor_ms=0.0)
    out = fs._merge_vetoes([(0, 1), (2, 3)], fine, None, None, res, sr, **kw)
    check((0, 1) not in out, "two independent trains are vetoed")
    check((2, 3) in out, "a pair with a refractory gap is kept")
    check(fs._merge_vetoes([(0, 1)], fine, None, None, res, sr) == [(0, 1)], "no veto configured, no veto")


def knn_fold():
    print("xcorr-merge knn fold")
    rng = np.random.default_rng(0)
    T, C = 32, 4
    t = np.arange(T)
    s1 = (-3000 * np.exp(-0.5 * ((t - 15) / 2.0) ** 2))[:, None] * np.array([1.0, 0.8, 0.5, 0.2])
    s2 = (-3000 * np.exp(-0.5 * ((t - 15) / 2.0) ** 2))[:, None] * np.array([0.1, 0.3, 0.9, 1.0])
    s2 = s2 + (2000 * np.exp(-0.5 * ((t - 22) / 2.0) ** 2))[:, None] * np.array([0.0, 0.2, 0.6, 1.0])
    n = 120
    spk = np.concatenate([s1 + rng.normal(0, 40, (n, T, C)), s1 + rng.normal(0, 40, (n, T, C)),
                          s2 + rng.normal(0, 40, (n, T, C))]).astype(np.int16)
    res = np.sort(rng.choice(10 ** 7, 3 * n, replace=False)).astype(np.int64)
    idx0 = [np.arange(0, n), np.arange(n, 2 * n), np.arange(2 * n, 3 * n)]
    times0 = [np.sort(res[i]) for i in idx0]

    def run(target):
        real = fx._fc.knn_pass

        def forced(get, lab, rows, **kw):                # every source spike to reference `target`
            new = np.asarray(lab).copy()
            new[lab == 0] = target
            return new, None, None, None
        fx._fc.knn_pass = forced
        st = {}
        try:
            fx.agglomerate(spk, [2, 3, 4], idx0, times0, 10 ** 7, cos_thr=0.9, shift=2, refrac=0, refrac_thr=0.3,
                           refrac_min_exp=5.0, censor=0, cap=500, ref_sample=15, rng=np.random.default_rng(0),
                           rvu_thr=1e-9, rvu_knn=1, res=res, knn_refs=8, min_n=15, stats=st)
        finally:
            fx._fc.knn_pass = real
        return st
    def home(st, rows):                                  # the output cluster holding most of `rows`
        best = max(st["members"].items(), key=lambda kv: np.intersect1d(kv[1], rows).size)
        return best[1]
    st = run(2)                                          # refs = [partner, the s2 cluster]: 2 -> s2
    with_z = np.intersect1d(home(st, idx0[2]), idx0[0]).size
    check(st.get("knn_fold_vetoes", 0) >= 1 and with_z == 0,
          f"a bucket sent to the dissimilar cluster does not fold into it (vetoes {st.get('knn_fold_vetoes')}, "
          f"moved into it {with_z})")
    st = run(1)                                          # 1 -> the partner, same shape
    h = home(st, idx0[0])
    check(np.intersect1d(h, idx0[0]).size == n and np.intersect1d(h, idx0[1]).size == n,
          "a bucket sent to a same-shape partner still folds into it")
    check(fs._chunk_hi(1, 0, 100.0, 2) == np.inf and fs._chunk_hi(0, 0, 100.0, 2) == 100.0,
          "the last chunk's core has no end (a spike on t_min + nchunks*chunk is in it)")


def session_leftovers():
    print("fiber-session: nothing unplaced goes to cluster 0")
    d = tempfile.mkdtemp(prefix="fk_rules_")
    try:
        rng = np.random.default_rng(3)
        sr, T, C = 20000, 32, 3
        t = np.arange(T)
        sh = [(-3000 * np.exp(-0.5 * ((t - 15) / 2.0) ** 2))[:, None] * np.array(w)
              for w in ([1.0, 0.6, 0.2], [0.2, 0.6, 1.0])]
        chunk = int(0.5 * 60 * sr)
        n = 900
        res = np.sort(rng.choice(np.arange(1, chunk), n, replace=False))
        res = np.concatenate([[0], res, chunk + np.sort(rng.choice(np.arange(1, chunk), 30, replace=False)),
                              [2 * chunk]]).astype(np.int64)     # span = exactly 2 chunks; chunk 1 too small
        lab = rng.integers(0, 2, res.size)
        spk = np.stack([sh[k] + rng.normal(0, 60, (T, C)) for k in lab]).astype(np.int16)
        b = os.path.join(d, "S")
        nio.write_res_file(f"{b}.res.1", res)
        nio.write_spk_file(f"{b}.spk.stderiv.1", spk)
        nio.write_spk_file(f"{b}.spk.standard.1", spk)
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env.update(FK_CONFIG=os.devnull, PYTHONPATH=os.path.join(ROOT, "src"))
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_session", b, "1", "--method", "stderiv",
                            "--channels", "0,1,2", "--ntotal", "3", "--nchan", "3", "--nsamp", str(T), "--sr",
                            str(sr), "--no-link", "--chunk-min", "0.5", "--overlap-min", "0", "--min-group", "40",
                            "--clu-stage", "fs", "--realign", "off", "--whiten-source", "spk"],
                           capture_output=True, text=True, env=env, timeout=1200)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            check(False, "fiber-session ran: " + out[-1500:])
            return
        _, clu = nio.read_clu_file(f"{b}.clu.stderiv.1.fs")
        clu = np.asarray(clu)
        check(clu[-1] >= 2, f"the last spike (on the chunk edge) is in a fiber ({clu[-1]})")
        small = slice(n + 1, n + 31)
        check((clu[small] >= 2).all() and np.unique(clu[small]).size == 1,
              f"the 30 spikes of the too-small chunk are one unsorted fiber ({np.unique(clu[small])})")
        check(int((clu == 0).sum()) == 0, f"no spike in cluster 0 ({int((clu == 0).sum())})")
        check("unsorted fiber" in out, "the log names the unsorted fiber")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    complete_link()
    vetoes()
    knn_fold()
    session_leftovers()
    print(f"test_merge_rules: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
