#!/usr/bin/env python3
"""A cluster's template sample depends on the cluster alone (fiber_lib.subsample / content_rng), not on the
run's shared random stream -- the fix for the merge-stage flip of g6 0316 (34+32+33 fused by fiber-xcorr-merge
from inputs spike-identical to a run that kept them apart, because two more clusters upstream shifted the
shared stream and every later cluster drew a different 300-spike sample)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fiber_kit import fiber_lib as fl, fiber_xcorr_merge as xm, fiber_anchor_link as al   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


idx = np.arange(1000, 5000, 3)                       # a 1334-spike cluster
rng = np.random.default_rng(0)
a = fl.subsample(idx, 300)
rng.choice(10**6, 777)                               # unrelated draws on the shared stream
b = fl.subsample(idx, 300, rng)                      # keyed by default: rng ignored
check(np.array_equal(a, b) and a.size == 300 and np.all(np.diff(a) > 0), "keyed: the same 300 spikes whatever was drawn before (sorted)")
check(np.array_equal(fl.subsample(idx[::-1], 300), a), "keyed: the sample is a function of the SET (order of idx irrelevant)")
check(not np.array_equal(fl.subsample(idx[1:], 300), a), "keyed: a different cluster draws a different sample")
check(np.array_equal(fl.subsample(idx[:200], 300), idx[:200]), "cap >= size: all spikes, sorted")
r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
u1 = fl.subsample(idx, 300, r1, keyed=False); r2.choice(10**6, 5); u2 = fl.subsample(idx, 300, r2, keyed=False)
check(not np.array_equal(u1, u2), "keyed=0 (pre-0.67.4): the sample moves with the shared stream")
check(not np.array_equal(fl.subsample(idx, 300, salt=1), a), "salt separates the draws one stage takes of one cluster")

# fiber-xcorr-merge._tmpl: same cluster, templates identical across runs that drew differently before it
NS, NC, N = 32, 8, 6000
g = np.random.default_rng(3)
t = -np.exp(-((np.arange(NS) - 12) / 2.5) ** 2) * 600
spk = (g.normal(0, 40, (N, NS, NC)) + t[None, :, None] * np.linspace(1, 0.2, NC)[None, None, :]).astype(np.int16)
cl = np.flatnonzero(np.arange(N) % 2 == 0)           # a 3000-spike cluster
ra, rb = np.random.default_rng(0), np.random.default_rng(0)
Ta = xm._tmpl(spk, cl, cap=300, ref_sample=12, rng=ra)[0]
rb.choice(N, 4321)                                   # "two more clusters upstream"
Tb = xm._tmpl(spk, cl, cap=300, ref_sample=12, rng=rb)[0]
check(np.array_equal(Ta, Tb), "xcm template of one cluster identical across runs with different upstream draws (keyed)")
xm._SAMPLE_KEYED = False
ra, rb = np.random.default_rng(0), np.random.default_rng(0); rb.choice(N, 4321)
Ta = xm._tmpl(spk, cl, cap=300, ref_sample=12, rng=ra)[0]; Tb = xm._tmpl(spk, cl, cap=300, ref_sample=12, rng=rb)[0]
check(not np.array_equal(Ta, Tb), "FK_XCM_SAMPLE_KEYED 0 reproduces the old stream-dependent template")
xm._SAMPLE_KEYED = True
fa = al.build_frag(spk, cl, spk_cap=1500, ref_sample=12, sr=20000.0, rng=np.random.default_rng(0))
rb = np.random.default_rng(0); rb.choice(N, 99)
fb = al.build_frag(spk, cl, spk_cap=1500, ref_sample=12, sr=20000.0, rng=rb)
same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(fa, fb) if isinstance(x, np.ndarray))
check(same, "anchor-link fragment template identical across runs with different upstream draws (keyed)")
print(f"test_sample_keyed: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
sys.exit(1 if fails else 0)
