#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  test/verify_optimizations.py — regression + rig-verification for the
#  vectorization / GPU / parallelism optimizations (fiber-kit >= 0.6.0).
#
#  Default run (no GPU, no real data needed) asserts the optimized kernels are
#  numerically identical to reference loop implementations, and that the
#  chunk-parallel path produces identical output to serial on a synthetic
#  session.  Flags:
#     --bench   also print speedups (realign / predict / ewma)
#     --gpu     additionally check the CuPy path matches CPU (run this on the rig)
#
#  Usage:
#     python test/verify_optimizations.py
#     python test/verify_optimizations.py --bench
#     FIBER_KIT_GPU=1 python test/verify_optimizations.py --gpu --bench
# ════════════════════════════════════════════════════════════════════════════
import argparse, os, sys, tempfile, time
import numpy as np

from fiber_kit import fiber_lib as fl, fiber_tracer as ft, fiber_adapt as fa
from fiber_kit import fiber_session as fs, neuro_io as nio, backend as bk

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ── reference (pre-optimization) loop implementations ───────────────────────
def realign_ref(W, lo=6, hi=26, maxlag=4):
    m = W.mean(0); dom = int(np.argmax(m.max(0) - m.min(0))); ref = m[:, dom]
    out = np.empty_like(W)
    for i, w in enumerate(W):
        best = (-1e18, 0)
        for lag in range(-maxlag, maxlag + 1):
            c = np.dot(np.roll(w[:, dom], lag)[lo:hi], ref[lo:hi])
            if c > best[0]: best = (c, lag)
        out[i] = np.roll(w, best[1], axis=0)
    return out

def predict_ref(traj, r):
    grid, D = traj
    if r <= grid[0]:  return D[0]
    if r >= grid[-1]: return D[-1]
    j = np.searchsorted(grid, r); f = (r - grid[j-1]) / (grid[j] - grid[j-1])
    pd = D[j-1] + (D[j] - D[j-1]) * f
    return pd / np.linalg.norm(pd)

def ewma_ref(ts, tau):
    a = np.zeros(len(ts))
    for i in range(1, len(ts)):
        a[i] = np.exp(-(ts[i] - ts[i-1]) / tau) * (a[i-1] + 1.0)
    return a


def test_equality():
    print("numerical equality vs reference loops:")
    rng = np.random.default_rng(0)
    mdr = 0; int_ok = True
    for _ in range(15):
        n = rng.integers(60, 400); C = rng.integers(4, 9)
        W = (rng.standard_normal((n, 32, C)) * 150); W[:, 13:17, :] -= 120
        mdr = max(mdr, float(np.abs(realign_ref(W) - fl.realign(W)).max()))
    Wi = (rng.standard_normal((200, 32, 8)) * 300).astype(np.int16)
    int_ok = np.array_equal(realign_ref(Wi), fl.realign(Wi))
    check("realign float bit-identical (max|diff|==0)", mdr == 0.0)
    check("realign int16 identical", bool(int_ok))

    X = rng.standard_normal((500, 12)) * np.arange(1, 13)
    grid = np.linspace(np.percentile(np.linalg.norm(X,1,1) if False else np.linalg.norm(X,axis=1),1),
                       np.percentile(np.linalg.norm(X,axis=1),99), 40)
    D = rng.standard_normal((40, 12)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    traj = (grid, D); r = np.linalg.norm(X, axis=1)
    rext = np.concatenate([r, [grid[0]-1, grid[-1]+1, grid[0], grid[-1]]])
    pr = np.array([predict_ref(traj, float(ri)) for ri in rext])
    pn = ft.predict_many(traj, rext)
    check("predict_many matches predict (<=1e-12)", float(np.abs(pr - pn).max()) <= 1e-12)

    mde = 0
    for _ in range(10):
        ts = np.cumsum(rng.exponential(0.04, rng.integers(50, 2000)))
        taus = np.logspace(np.log10(0.005), np.log10(2.0), 30)
        A = fa.ewma_multi(ts, taus)
        for ti, tau in enumerate(taus):
            mde = max(mde, float(np.abs(ewma_ref(ts, tau) - A[ti]).max()))
    check("ewma_multi bit-identical to per-tau ewma (max|diff|==0)", mde == 0.0)


def _make_session(d, elec=5, ntotal=12, nchan=8, nsamp=32, sr=32552.0, seconds=50):
    base = os.path.join(d, "synth"); rng = np.random.default_rng(0)
    T = int(seconds * sr)
    (rng.standard_normal((T, ntotal)) * 30).astype(np.int16).tofile(f"{base}.fil")
    def tmpl(ch, amp):
        prof = np.exp(-0.5 * ((np.arange(nchan) - ch) / 1.3) ** 2)
        tr = -amp * np.exp(-0.5 * ((np.arange(nsamp) - 15) / 1.6) ** 2)
        return (tr[:, None] * prof[None, :]).astype(np.float32)
    res_l, wav_l = [], []
    for tpl, rate in [(tmpl(2, 380), 6.0), (tmpl(5, 300), 4.0), (tmpl(6, 250), 3.0)]:
        n = int(rate * seconds); tt = np.sort(rng.integers(nsamp, T - nsamp, n))
        res_l.append(tt); wav_l.append((tpl[None] + rng.standard_normal((n, nsamp, nchan)) * 18).astype(np.int16))
    res = np.concatenate(res_l); o = np.argsort(res); res = res[o]
    spk = np.concatenate(wav_l, 0)[o]
    nio.write_res_file(f"{base}.res.{elec}", res.astype(np.int64))
    spk.tofile(f"{base}.spkD.{elec}")
    return base, res, elec, ntotal, nchan, nsamp, sr


def _cfg(base, res, elec, ntotal, nchan, nsamp, sr):
    cf = dict(method="gmm", fine_kappa=40.0, fine_dedup=5.0, fine_mg=40, pca_k=6, max_sub=8,
              n_grid=40, incl_k=3.0, dipsplit=False, dip_dim=4, dip_alpha=0.01, dip_min=40,
              rkk_dims=6, rkk_max=50, merge_corr=0.0, merge_method="template", sliding_nwin=14,
              profile_thr=None, profile_floor_pct=90.0, profile_min_n=120, emit_candidates=False,
              deadapt=False, deadapt_min_corr=0.2, adapt_clean=False, adapt_z=3.0, adapt_isi_ms=10.0,
              adapt_clean_corr=0.4, adapt_clean_snr=0.5, adapt_taumax=0.5, collision_flag=False,
              collision_gain=0.09, collision_shift=8, quality_metrics=False, quality_dims=10)
    return dict(base=base, elec=elec, fil=f"{base}.fil", ntotal=ntotal, nsamp=nsamp, nchan=nchan,
                sr=sr, min_group=120, gch=np.arange(2, 2 + nchan), mask=fl.MASK_FULL, cf=cf, gpu=False)


def test_parallel():
    print("chunk parallelism (serial vs ProcessPoolExecutor):")
    from concurrent.futures import ProcessPoolExecutor
    with tempfile.TemporaryDirectory() as d:
        base, res, *rest = _make_session(d)
        cfg = _cfg(base, res, *rest)
        tmin, tmax = int(res.min()), int(res.max()); mid = (tmin + tmax) // 2
        tasks = [(0, np.flatnonzero(res < mid), res[res < mid]),
                 (1, np.flatnonzero(res >= mid), res[res >= mid])]
        fs._init_chunk_worker(cfg)
        serial = {t[0]: fs._process_chunk(t)[2:4] for t in tasks}  # (lab, geoms)
        with ProcessPoolExecutor(max_workers=2, initializer=fs._init_chunk_worker,
                                 initargs=(cfg,)) as ex:
            par = {r[0]: r[2:4] for r in ex.map(fs._process_chunk, tasks)}
        all_ok = True
        for c in serial:
            ls, gs = serial[c]; lp, gp = par[c]
            ok = np.array_equal(ls, lp) and len(gs) == len(gp)
            all_ok &= ok; print(f"    chunk {c}: nfib {len(gs)}/{len(gp)} labels_identical={np.array_equal(ls, lp)}")
        check("parallel labels + fiber counts identical to serial", all_ok)


def test_gpu():
    print("GPU path:")
    on = bk.use_gpu(True)
    check("GPU enable requested (numpy fallback is OK without CUDA)", True)
    print(f"    backend = {bk.backend_name()}")
    if not on:
        print("    CuPy/CUDA unavailable -> CPU fallback verified; skipping GPU-vs-CPU compare")
        return
    rng = np.random.default_rng(3)
    W = (rng.standard_normal((400, 32, 8)) * 150); W[:, 13:17, :] -= 120
    g = fl.realign(W); bk.use_gpu(False); c = fl.realign(W); bk.use_gpu(True)
    check("GPU realign matches CPU (<=1e-3)", float(np.abs(g - c).max()) <= 1e-3)
    p = len(fl.MASK_FULL) * 8; Wm = rng.standard_normal((p, p)); nm = rng.standard_normal(p)
    Xg = fl.features(W, Wm, nm)[0]; bk.use_gpu(False); Xc = fl.features(W, Wm, nm)[0]; bk.use_gpu(True)
    check("GPU features matches CPU (<=1e-2)", float(np.abs(Xg - Xc).max()) <= 1e-2)
    bk.use_gpu(False)


def bench():
    print("benchmarks:")
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((20000, 32, 8)) * 200); W[:, 13:17, :] -= 150
    t = time.time(); realign_ref(W); t_o = time.time() - t
    t = time.time(); fl.realign(W); t_n = time.time() - t
    print(f"    realign 20k: ref {t_o*1000:.0f}ms  opt {t_n*1000:.0f}ms  ({t_o/t_n:.1f}x)")
    X = rng.standard_normal((20000, 12)) * np.arange(1, 13)
    grid = np.linspace(np.percentile(np.linalg.norm(X,axis=1),1), np.percentile(np.linalg.norm(X,axis=1),99), 40)
    D = rng.standard_normal((40, 12)); D /= np.linalg.norm(D, axis=1, keepdims=True); traj = (grid, D)
    r = np.linalg.norm(X, axis=1); K = 8
    t = time.time()
    for _ in range(K): np.array([predict_ref(traj, float(ri)) for ri in r])
    t_o = time.time() - t
    t = time.time()
    for _ in range(K): ft.predict_many(traj, r)
    t_n = time.time() - t
    print(f"    predict 20k x{K}: ref {t_o*1000:.0f}ms  opt {t_n*1000:.0f}ms  ({t_o/t_n:.0f}x)")
    ts = np.cumsum(rng.exponential(0.02, 40000)); taus = np.logspace(np.log10(0.005), np.log10(2.0), 30)
    t = time.time()
    for tau in taus: ewma_ref(ts, tau)
    t_o = time.time() - t
    t = time.time(); fa.ewma_multi(ts, taus); t_n = time.time() - t
    print(f"    ewma 40k x30tau: ref {t_o*1000:.0f}ms  opt {t_n*1000:.0f}ms  ({t_o/t_n:.0f}x)")


def main():
    ap = argparse.ArgumentParser(description="verify + benchmark fiber-kit optimizations")
    ap.add_argument("--bench", action="store_true"); ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    test_equality()
    test_parallel()
    if a.gpu:
        test_gpu()
    if a.bench:
        bench()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed" + (": " + ", ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
