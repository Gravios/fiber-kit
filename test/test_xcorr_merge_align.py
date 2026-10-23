"""fiber-xcorr-merge agglomeration, with each --merge-align.

Synthetic standard .spk: unit A in two fragments (its spikes individually jittered by up to
+-3 samples), unit B with a different footprint, sparse independent spike times.
  * peak and klusters alignment both merge A's fragments and keep B apart at cos-thr 0.96;
  * _klusters_align recovers the jitter: the template of the Klusters-aligned sample has a
    deeper trough than the unaligned mean;
  * an unknown --merge-align is refused.
"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_xcorr_merge as xm   # noqa: E402

fails = 0
def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def main():
    rng = np.random.default_rng(4)
    T, C, PK = 32, 4, 16
    t = np.arange(T)
    def unit(foot):
        return -np.exp(-0.5 * ((t - (PK - 1)) / 1.4) ** 2)[:, None] * np.asarray(foot)[None, :] * 1500
    A, B = unit([1.0, 0.6, 0.3, 0.1]), unit([0.1, 0.3, 0.6, 1.0])
    W, ids = [], []
    for lab, tm, n in ((2, A, 300), (3, A, 300), (4, B, 300)):
        for _ in range(n):
            W.append(np.roll(tm, int(rng.integers(-3, 4)), axis=0) + rng.normal(0, 60, (T, C))); ids.append(lab)
    spk = np.clip(np.rint(np.array(W)), -32768, 32767).astype(np.int16); ids = np.array(ids)
    res = np.sort(rng.choice(np.arange(10**7), ids.size, replace=False))
    labs = [2, 3, 4]; idx0 = [np.flatnonzero(ids == u) for u in labs]; times0 = [np.sort(res[i]) for i in idx0]
    for align in ("peak", "klusters"):
        mp, nm, nv = xm.agglomerate(spk, labs, idx0, times0, float(res.max()), cos_thr=0.96, shift=4, refrac=0,
                                    refrac_thr=0.3, refrac_min_exp=5.0, censor=0, cap=300, ref_sample=PK,
                                    rng=np.random.default_rng(0), band_thr=0.5, align=align)
        check(mp[2] == mp[3] and mp[4] != mp[2], f"{align}: A's two fragments merged, B kept apart ({mp})")
    sub = spk[idx0[0]]
    al = xm._klusters_align(sub, PK)
    d_al, d_raw = -al.mean(0)[:, 0].min(), -sub.astype(float).mean(0)[:, 0].min()
    check(d_al > 1.2 * d_raw, f"Klusters alignment sharpens the jittered template (trough {d_al:.0f} vs {d_raw:.0f})")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_xcorr_merge", "nosuch", "1", "--merge-align", "bogus"],
                       capture_output=True, text=True, env=env)
    check(r.returncode != 0 and "--merge-align must be peak or klusters" in (r.stdout + r.stderr),
          "an unknown --merge-align is refused")
    print(f"\ntest_xcorr_merge_align: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
