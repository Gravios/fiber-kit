"""fiber-cpos: bounded memory on large clusters, one grouping instead of per-cluster scans.

  * spike_table (vectorised) equals the per-cluster loop it replaced, including unlocalized clusters,
    negative ids and gaps;
  * localize_clusters never asks extract() for more than max_spikes spikes of a cluster, still reports
    the cluster's full count as n, and with max_spikes=0 loads every spike (the old behaviour);
  * the capped template of a 6,000-spike synthetic cluster is within noise of the full one.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_cpos as cp   # noqa: E402

fails = 0
def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def naive_table(nspk, clu, per):
    T = np.full((nspk, len(cp.CPOS_COLS)), np.nan, np.float32)
    T[:, cp.CPOS_COLS.index("one_flank")] = 1.0
    for cid, r in per.items():
        m = clu == cid
        for j, k in enumerate(cp.CPOS_COLS):
            T[m, j] = float(r[k])
    return T


def main():
    rng = np.random.default_rng(0)
    clu = rng.choice([-1, 0, 2, 3, 7, 40], 5000)
    per = {c: {k: float(rng.normal()) for k in cp.CPOS_COLS} for c in (0, 3, 40)}   # 2, 7 unlocalized
    a, b = cp.spike_table(clu.size, clu, per), naive_table(clu.size, clu, per)
    check(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b)),
          "spike_table equals the per-cluster loop")

    T, C = 32, 8
    t = np.arange(T)
    xy = np.stack([np.zeros(C), np.arange(C) * 20.0], 1)
    tmpl = -np.exp(-0.5 * ((t - 15) / 1.5) ** 2)[:, None] * np.exp(-0.5 * ((np.arange(C) - 3) / 1.5) ** 2)[None, :] * 800
    n_big, n_small = 6000, 300
    clu2 = np.r_[np.full(n_big, 2), np.full(n_small, 3)]
    W = tmpl[None] + rng.normal(0, 30, (clu2.size, T, C))
    asked = {}
    def extract(idx):
        idx = np.asarray(idx); c = int(clu2[idx[0]])
        asked[c] = max(asked.get(c, 0), idx.size)
        return W[idx].astype(np.float32)
    per_cap = cp.localize_clusters(extract, clu2, xy, max_spikes=1000, amp_basis="none")
    check(asked[2] <= 1000 and asked[3] == n_small, f"extract asked for at most the cap ({asked[2]} of {n_big}; small cluster whole)")
    check(per_cap[2]["n"] == n_big and per_cap[3]["n"] == n_small, "n reports each cluster's full spike count")
    asked.clear()
    per_all = cp.localize_clusters(extract, clu2, xy, max_spikes=0, amp_basis="none")
    check(asked[2] == n_big, "max_spikes=0 loads every spike (the previous behaviour)")
    d = np.abs(per_cap[2]["template"] - per_all[2]["template"]).max()
    check(d < 15, f"capped template within noise of the full one (max |diff| {d:.1f}, noise sd 30)")
    dz = abs(per_cap[2]["y0"] - per_all[2]["y0"])
    check(dz < 1.0, f"capped position within 1 um of the full one ({dz:.3f})")
    print(f"\ntest_cpos_cap: {'FAIL' if fails else 'PASS'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
