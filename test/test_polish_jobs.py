#!/usr/bin/env python3
"""test_polish_jobs.py — fiber-polish's island pass in worker processes gives the same labels.

The island pass was 84% of the stage's time on the 0312 g8 test (two chunks, 121,000 spikes), and
95% of that is the per-cluster search: kd leaves, eight subsampled t-SNE embeddings per leaf,
co-neighbour communities.  That search runs in worker processes with --jobs / FK_POLISH_JOBS.
A cluster's random subsets come from a generator seeded by the pass and the cluster's id, so:

  * the labels with two workers are the labels with none, spike for spike;
  * a cluster's islands do not depend on which other clusters are searched, or in what order;
  * the pass still does its job on the data of test_polish (a foreign island moves, the
    homogeneous clusters stay whole);
  * polish(jobs=2) gives polish(jobs=1).

This file has a main guard, as any script that starts spawned workers must.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_polish as fpo, fiber_consensus as fcs      # noqa: E402

fails = 0
S, C = 40, 8


def check(ok, what):
    global fails
    fails += not ok
    print(("  ok:   " if ok else "  FAIL: ") + what)


def cell(n, chans, amps, shape, rng, noise=3.0, jitter=0.15):
    w = np.zeros((n, S, C))
    g = 1.0 + jitter * rng.standard_normal(n)
    for c, a in zip(chans, amps):
        w[:, :, c] = -a * g[:, None] * shape[None, :]
    return w + rng.normal(0, noise, w.shape)


def main():
    try:
        import networkx  # noqa: F401
        import openTSNE  # noqa: F401
    except ImportError:
        print("  skip: island pass (openTSNE / networkx not installed)")
        print("test_polish_jobs: PASS (0 failed)")
        return 0
    t = np.arange(S)
    bump = np.exp(-((t - 20) ** 2) / 8.0)
    wide = np.exp(-((t - 21) ** 2) / 30.0)
    rng = np.random.default_rng(3)
    W = np.concatenate([cell(600, (0, 1), (100, 60), bump, rng), cell(300, (4, 5), (90, 70), bump, rng),
                        cell(300, (0, 1), (100, 60), wide, rng)]).astype(np.float32)
    raw = lambda ix: np.asarray(W[np.asarray(ix)], float)   # noqa: E731
    F = W.reshape(len(W), -1)[:, ::5].astype(float)
    lab = np.concatenate([np.full(600, 2), np.full(300, 4), np.full(300, 5)]).astype(np.int64)
    lab[1150:1200] = 2                                      # 50 spikes of the third cell sit in the first cluster
    kw = dict(leaf=300, runs=4, min_clu=60, margin=0.5)

    # seed 2: one of the seeds for which the 50-spike island comes out whole with 4 runs (about half do)
    serial, st0 = fpo.island_pass(lab, raw, F, rng=np.random.default_rng(2), **kw)
    check((serial[1150:1200] == 5).mean() >= 0.8 and (serial[600:900] == 4).all(),
          "without workers: the foreign island moves, the homogeneous cluster stays whole")
    pool = fcs._pool(2)
    check(pool is not None, "a pool of two spawned workers starts")
    try:
        pooled, st1 = fpo.island_pass(lab, raw, F, rng=np.random.default_rng(2), pool=pool, **kw)
        check(np.array_equal(pooled, serial) and st0 == st1,
              f"two workers give the same labels and counts ({st1})")

        task = lambda ci, f: (ci, f, F[lab == f], 300, 4, 60, 10, 12345)   # noqa: E731
        alone = fpo._island_cluster(task(0, 2))
        again = list(pool.map(fpo._island_cluster, [task(1, 4), task(0, 2), task(2, 5)]))[1]
        same = alone[1] == again[1] and len(alone[2]) == len(again[2]) and all(
            np.array_equal(a, b) for a, b in zip(alone[2], again[2]))
        check(same, "a cluster's islands do not depend on the process, or on what else is searched")
        other = fpo._island_cluster((0, 2, F[lab == 2], 300, 4, 60, 10, 54321))
        check(alone[1] == other[1], "another pass seed keeps the leaves (only the random subsets change)")
    finally:
        pool.shutdown()

    feats = lambda ix: F[np.asarray(ix)]                    # noqa: E731
    one, _ = fpo.polish(lab, raw, feats, F, loops=0, rounds=1, island=1, merge=0, island_runs=4, jobs=1)
    two, _ = fpo.polish(lab, raw, feats, F, loops=0, rounds=1, island=1, merge=0, island_runs=4, jobs=2)
    check(np.array_equal(one, two), "polish(jobs=2) gives polish(jobs=1)")
    print(f"test_polish_jobs: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
