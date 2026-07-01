#!/usr/bin/env python3
# test_chan_svd.py — per-channel template SVD (fiber_chan_svd).  Build synthetic cluster
# templates with KNOWN per-channel behaviour and check the decomposition recovers it:
#   ch 0,1  invariant (identical across clusters)          -> smallest relvar
#   ch 2,3  amplitude-only variation (scaled template)     -> high relvar, PC1 ~ template shape (>85%)
#   ch 4,5  shape variation (independent extra mode)        -> high relvar, PC1 fraction lower (spread)
#   ch 6,7  flat/noise                                      -> tiny
# Invariant channels must rank below the varying ones, and the amplitude channel's PC1 must be
# both large-fraction AND shaped like the template (an energy-level merge signature), while the
# shape channel's variation must NOT match the template.  Also render the figure headless.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np

try:
    from fiber_kit import fiber_chan_svd as cs
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_chan_svd as cs

NSAMP, NCH, K = 42, 8, 8
fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    print(("  ok:   " if ok else "  FAIL: ") + what)
    if not ok:
        fails += 1


def _wave(width, t0=15):
    t = np.arange(NSAMP)
    return -np.exp(-((t - t0) ** 2) / 4.0) + 0.6 * np.exp(-((t - (t0 + width)) ** 2) / 6.0)


def build():
    rng = np.random.default_rng(0)
    base = _wave(6)                              # shared template shape
    alt = _wave(16)                              # a DIFFERENT shape (for the shape-varying channel)
    T = np.zeros((K, NSAMP, NCH))
    for k in range(K):
        s = 1.0 + 0.6 * (k - K / 2) / K          # per-cluster amplitude factor
        T[k, :, 0] = base                        # invariant
        T[k, :, 1] = base
        T[k, :, 2] = s * base                    # amplitude-only variation (scaled template)
        T[k, :, 3] = s * base
        T[k, :, 4] = base + 0.5 * (k - K / 2) / K * alt   # shape variation (extra independent mode)
        T[k, :, 5] = base + 0.5 * (k - K / 2) / K * alt
        T[k, :, 6] = 0.0                         # flat
        T[k, :, 7] = 0.0
    T += 0.01 * rng.standard_normal(T.shape)
    return T


def main():
    T = build()
    res = cs.per_channel_svd(T, n_comp=3)
    rv = res["var_rel"]
    order = list(np.argsort(rv))

    inv = {0, 1}
    varying = {2, 3, 4, 5}
    # every invariant channel ranks below every varying channel
    check(max(order.index(c) for c in inv) < min(order.index(c) for c in varying),
          "invariant channels (0,1) rank below all varying channels (2-5)")
    check(rv[0] < rv[2] and rv[1] < rv[3], "invariant relvar < amplitude-channel relvar")
    check(rv[0] < rv[4] and rv[1] < rv[5], "invariant relvar < shape-channel relvar")

    # amplitude channel: PC1 dominates AND looks like the template (energy-level signature)
    grand = res["grand"]
    tmpl2 = grand[:, 2] / (np.linalg.norm(grand[:, 2]) + 1e-9)
    pc1_2 = res["comps"][2, 0] / (np.linalg.norm(res["comps"][2, 0]) + 1e-9)
    check(res["vfrac"][2, 0] > 0.85, f"amplitude channel PC1 fraction > 85% (got {100*res['vfrac'][2,0]:.0f}%)")
    check(abs(float(tmpl2 @ pc1_2)) > 0.9, "amplitude channel PC1 ~ template shape (|cos| > 0.9)")

    # shape channel: its dominant mode is NOT the template (it's the injected alt shape)
    tmpl4 = grand[:, 4] / (np.linalg.norm(grand[:, 4]) + 1e-9)
    pc1_4 = res["comps"][4, 0] / (np.linalg.norm(res["comps"][4, 0]) + 1e-9)
    check(abs(float(tmpl4 @ pc1_4)) < abs(float(tmpl2 @ pc1_2)),
          "shape channel PC1 diverges from its template more than the amplitude channel does")

    # figure renders headless
    cs._need_mpl()
    fig, _ = cs.figure(res, list(range(NCH)), n_comp=3, sr=32552.0, title="test")
    out = os.path.join(HERE, "_chan_svd_test.png")
    fig.savefig(out, dpi=80)
    cs.plt.close(fig)
    check(os.path.getsize(out) > 2000, "figure rendered to a non-trivial PNG")
    os.remove(out)

    # --within: split ONE synthetic cluster into time-ordered sub-templates.  The dominant channel is
    # invariant; the weak channels carry a structured, time-varying (physiological-like) component.  The
    # per-channel SVD over the sub-templates must rank the dominant channel most invariant.
    rng = np.random.default_rng(1)
    N = 2400; t = np.arange(NSAMP)
    base = -np.exp(-((t - 15) ** 2) / 4.0) + 0.6 * np.exp(-((t - 21) ** 2) / 6.0)
    amp = np.array([1.0, 0.8, 0.5, 0.4, 0.3, 0.25, 0.15, 0.12])   # dominant = ch 0
    res_t = np.sort(rng.integers(0, 3_000_000, size=N)).astype(np.int64)
    phase = np.linspace(0, 1, N)                                   # slow time-ordered drift of the weak channels
    spk = np.zeros((N, NSAMP, NCH))
    for k in range(N):
        for c in range(NCH):
            wob = 0.0 if c < 2 else 0.4 * amp[c] * np.sin(2 * np.pi * phase[k])   # structured on weak channels only
            spk[k, :, c] = amp[c] * base + wob * base
    spk += 0.01 * rng.standard_normal(spk.shape)
    sub, span = cs.cluster_subtemplates(spk, res_t, np.arange(N), bins=12)
    check(len(sub) >= 8, f"cluster_subtemplates produced {len(sub)} sub-templates")
    rw = cs.per_channel_svd(sub, n_comp=3)
    dom = int(np.argmax(np.ptp(sub.mean(0), axis=0)))
    check(rw["var_rel"][dom] < rw["var_rel"][6] and rw["var_rel"][dom] < rw["var_rel"][7],
          "within-unit: dominant channel is more invariant than the weak channels")
    check(rw["var_rel"][0] < rw["var_rel"][2], "within-unit: relvar rises as channel amplitude falls")

    print(f"\n{ran - fails}/{ran} checks passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
