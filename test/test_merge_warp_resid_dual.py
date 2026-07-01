#!/usr/bin/env python3
# test_merge_warp_resid_dual.py — the cell-type-aware (dual) warp-incongruity SUB-gate in
# fiber_refine.merge_back.  Two clusters that ARE merge candidates (high similarity) are given a
# controlled ~1.2-sample single-channel group-delay incongruity, then run once as INTERNEURONS
# (narrow raw waveform) and once as PYRAMIDAL (wide).  At that identical incongruity the dual gate
# must VETO the interneuron merge (tight --merge-warp-resid-thr-int) and ADMIT the pyramidal one
# (loose --merge-warp-resid-thr-pyr).  With the dual thresholds unset it must reproduce the prior
# behaviour (both merge).  This locks the cell-type branching and the raw-template classification.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np

try:
    from fiber_kit import fiber_refine as fr, fiber_geometry as fg
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "src", "fiber_kit"))
    import fiber_refine as fr
    import fiber_geometry as fg

SR = 32552.0
NSAMP = 42
NCH = 8
INT_THR = 0.7
PYR_THR = 1.3

fails = 0
ran = 0


def check(ok, what):
    global fails, ran
    ran += 1
    if not ok:
        fails += 1
        print(f"  FAIL: {what}")
    else:
        print(f"  ok:   {what}")


def _spike(width, dom, delay, shift_ch=None, shift=0.0):
    """A propagating octrode footprint: biphasic (trough->peak) on each channel, amplitude peaked at
    `dom`, per-channel group delay `delay[c]` (+ optional extra `shift` samples on one channel).
    `width` samples trough->peak sets the cell type (narrow=int, wide=pyr)."""
    t = np.arange(NSAMP)
    T = np.zeros((NSAMP, NCH))
    for c in range(NCH):
        d = delay[c] + (shift if c == shift_ch else 0.0)
        tr = 15.0 + d
        pk = tr + width
        T[:, c] = np.exp(-abs(c - dom) / 3.0) * (
            -np.exp(-((t - tr) ** 2) / 4.0) + 0.6 * np.exp(-((t - pk) ** 2) / 6.0))
    return T


def _run(celltype, dual):
    rng = np.random.default_rng(42)                       # identical noise every call -> identical incong
    delay = np.array([-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    wid = 5 if celltype == "int" else 16                  # ~0.15 ms (int) vs ~0.49 ms (pyr) at 32552 Hz
    dom = 4
    Ta = _spike(wid, dom, delay)
    Tb = _spike(wid, dom, delay, shift_ch=5, shift=1.2)   # one STRONG channel shifted -> incongruity ~1.2
    wav = np.concatenate([Ta[None] + 0.02 * rng.standard_normal((60, NSAMP, NCH)),
                          Tb[None] + 0.02 * rng.standard_normal((60, NSAMP, NCH))], 0)
    raw = wav.copy()                                      # raw drives classify_celltype (width -> type)
    lab = np.r_[np.zeros(60, int), np.ones(60, int)]
    res = np.sort(rng.integers(0, 20_000_000, size=120)).astype(float)   # spread -> refractory gate inert
    ctx = fr.Ctx(W=None, nmean=None, mask=np.ones(NCH, bool), sr=SR,
                 floor=16, window=int(0.002 * SR), basis=None)
    kw = dict(budget=100.0, min_sim=0.80, warp_thr=None, warp_resid_thr=None,
              raw_waves=raw, sr=SR, verbose=False)
    if dual:
        kw.update(warp_resid_thr_int=INT_THR, warp_resid_thr_pyr=PYR_THR)
    out = fr.merge_back(lab.copy(), wav, res, ctx, **kw)
    merged = len(np.unique(out[out >= 0])) == 1
    cls = fg.classify_celltype(np.median(raw[:60], 0), SR)
    inc = fg.warp_channel_incongruity(fg.group_delay_profile(np.median(wav[:60], 0)),
                                      fg.group_delay_profile(np.median(wav[60:], 0)))
    return cls, inc, merged


def main():
    # cell-typing sanity: the narrow footprint classifies int, the wide one pyr
    ci, inci, mi_off = _run("int", dual=False)
    cp, incp, mp_off = _run("pyr", dual=False)
    check(ci == "int", f"narrow footprint classified 'int' (got {ci})")
    check(cp == "pyr", f"wide footprint classified 'pyr' (got {cp})")

    # the constructed incongruity straddles the two thresholds for BOTH types (fair comparison)
    check(INT_THR < inci and inci < PYR_THR, f"int-pair incongruity {inci:.2f} in (int_thr,pyr_thr)")
    check(INT_THR < incp and incp < PYR_THR, f"pyr-pair incongruity {incp:.2f} in (int_thr,pyr_thr)")

    # dual OFF: both pairs merge (the sub-gate is not applied) -- backward compatible
    check(mi_off is True, "dual off: interneuron pair merges (no cell-type gate)")
    check(mp_off is True, "dual off: pyramidal  pair merges (no cell-type gate)")

    # dual ON: the SAME incongruity vetoes the interneuron merge but admits the pyramidal one
    _, _, mi_on = _run("int", dual=True)
    _, _, mp_on = _run("pyr", dual=True)
    check(mi_on is False, f"dual on: interneuron pair VETOED (incong {inci:.2f} > int_thr {INT_THR})")
    check(mp_on is True, f"dual on: pyramidal  pair ADMITTED (incong {incp:.2f} < pyr_thr {PYR_THR})")

    print(f"\n{ran - fails}/{ran} checks passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
