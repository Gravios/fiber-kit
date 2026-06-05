# ════════════════════════════════════════════════════════════════════════════
#  fiber_realign.py — per-spike fiber-template offsets and .res time correction.
#
#  Each spike's waveform is aligned to the template of the unit it belongs to by
#  multichannel cross-correlation (integer lag within +-max_shift), refined to
#  sub-sample resolution by a parabola through the correlation peak.  The per-spike
#  offset (in samples; the "fiber template offset") is saved, and the .res spike
#  times are corrected by the rounded integer part:  res_corrected = res + round(off).
#
#  The template is recomputed from the aligned spikes and the alignment repeated a
#  couple of rounds so it converges; units with too few spikes are left untouched
#  (their template is too noisy to anchor an alignment).
#
#  This both removes residual detection jitter and — importantly after re-linking —
#  forces every spike of a merged unit onto ONE canonical template, so a unit built
#  from fibers detected against different per-chunk references gets a single
#  consistent spike-time convention.  Where the extractor already peak-aligned the
#  spikes, the offsets are correctly small (mostly 0, +-1); the tool still writes
#  them and the corrected .res so the convention is explicit and reproducible.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np

try:
    from . import fiber_session as fs
except ImportError:
    import fiber_session as fs
try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio


def _read_clu(path):
    nclu, ids = nio.read_clu_file(path)
    return nclu, ids                                    # nClusters, per-spike ids


def template_offsets(spk, labels, max_shift=5, iters=2, min_n=20,
                     subsample=True, noise_label=0):
    """Compute per-spike sub-sample offsets aligning each spike to its unit's
    multichannel template.

    spk     : (N, T, C) float/int waveforms (sample-major, peak near T//2)
    labels  : (N,) per-spike unit id (noise_label is left at offset 0)
    returns : off (N,) float32 sub-sample offset, ioff (N,) int32 rounded offset.
    res_corrected = res + ioff.
    """
    spk = np.asarray(spk, np.float32)
    N, T, C = spk.shape
    ms = int(max_shift)
    off = np.zeros(N, np.float32)
    core = slice(ms, T - ms)
    Tcore = T - 2 * ms
    lags = np.arange(-ms, ms + 1)

    by = {}
    for i, l in enumerate(labels):
        by.setdefault(int(l), []).append(i)

    for u, rows in by.items():
        if u == noise_label or len(rows) < min_n:
            continue
        idx = np.asarray(rows)
        W = spk[idx]                                     # (n,T,C)
        cur = np.zeros(len(idx), int)                    # current integer lag
        corr = None
        for _ in range(max(1, iters)):
            # template from currently-aligned spikes (robust median), core region.
            # Vectorized gather of each spike's lag-shifted core (was a per-spike
            # Python loop); identical result, O(n) interpreter calls removed.
            gidx = np.arange(Tcore)[None, :] + (ms + cur)[:, None]   # (n, Tcore)
            al = np.take_along_axis(W, gidx[:, :, None], axis=1).astype(np.float32)
            templ = np.median(al, axis=0)                # (Tcore,C)
            tc = templ - templ.mean(axis=0, keepdims=True)
            # correlation at every lag (n, nLags)
            corr = np.empty((len(idx), len(lags)), np.float32)
            for k, L in enumerate(lags):
                seg = W[:, ms + L:T - ms + L, :]
                seg = seg - seg.mean(axis=1, keepdims=True)
                corr[:, k] = np.einsum('ntc,tc->n', seg, tc)
            cur = lags[np.argmax(corr, axis=1)]
        # sub-sample refinement: parabola through (k-1,k,k+1) of the correlation
        kbest = np.argmax(corr, axis=1)
        frac = np.zeros(len(idx), np.float32)
        if subsample:
            ok = (kbest > 0) & (kbest < len(lags) - 1)
            a = corr[np.arange(len(idx)), np.clip(kbest - 1, 0, len(lags) - 1)]
            b = corr[np.arange(len(idx)), kbest]
            c = corr[np.arange(len(idx)), np.clip(kbest + 1, 0, len(lags) - 1)]
            den = (a - 2 * b + c)
            good = ok & (np.abs(den) > 1e-6)
            frac[good] = 0.5 * (a[good] - c[good]) / den[good]
            frac = np.clip(frac, -0.5, 0.5)
        off[idx] = lags[kbest] + frac
    ioff = np.rint(off).astype(np.int32)
    return off, ioff


def realign(base, elec, nsamp, nch, clu_path=None, max_shift=5, iters=2,
            min_n=20, verbose=True):
    """Read <base>.spkD/.spk, .res, .clu; return (res, off, ioff, res_corrected)."""
    spk, spk_path = fs.open_spkD(base, elec, nsamp, nch)
    res = fs.read_res(base, elec)
    nclu, labels = _read_clu(clu_path or f"{base}.clu.{elec}")
    n = min(len(res), len(labels), spk.shape[0])
    if not (len(res) == len(labels) == spk.shape[0]):
        if verbose:
            print(f"[realign] WARNING length mismatch res={len(res)} clu={len(labels)} "
                  f"spk={spk.shape[0]} -> using first {n}")
    res, labels, spk = res[:n], labels[:n], spk[:n]
    off, ioff = template_offsets(spk, labels, max_shift, iters, min_n)
    res_corr = res + ioff.astype(np.int64)
    if verbose:
        nz = np.count_nonzero(ioff)
        print(f"[realign] {spk_path}: {n} spikes, {nclu - 1} units")
        print(f"[realign] offsets: nonzero={nz} ({100 * nz / n:.1f}%)  "
              f"mean={off.mean():+.3f}  std={off.std():.3f}  "
              f"|off|>=1: {100 * np.mean(np.abs(ioff) >= 1):.1f}%  "
              f"max|off|={np.abs(ioff).max()} samp")
    return res, off, ioff, res_corr


def write_outputs(base, elec, off, res_corr, out_res=None, out_off=None):
    out_res = out_res or f"{base}.res.{elec}.realigned"
    out_off = out_off or f"{base}.offsets.{elec}.npy"
    nio.write_res_file(out_res, res_corr)               # binary LE int64, same as .res
    np.save(out_off, off.astype(np.float32))            # sub-sample per-spike offsets
    return out_res, out_off


def main():
    ap = argparse.ArgumentParser(
        description="Per-spike fiber-template offsets + corrected .res spike times.")
    ap.add_argument("base", help="session base path (<base>.spkD.<elec>/.res.<elec>/.clu.<elec>)")
    ap.add_argument("elec", type=int, help="electrode/spike group")
    ap.add_argument("--nsamp", type=int, required=True, help="samples per spike waveform")
    ap.add_argument("--nch", type=int, required=True, help="channels in the group")
    ap.add_argument("--clu", default=None, help="cluster file (default <base>.clu.<elec>; pass the relinked one)")
    ap.add_argument("--max-shift", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--out-res", default=None)
    ap.add_argument("--out-off", default=None)
    a = ap.parse_args()
    res, off, ioff, res_corr = realign(a.base, a.elec, a.nsamp, a.nch, a.clu,
                                       a.max_shift, a.iters, a.min_n)
    orr, off_path = write_outputs(a.base, a.elec, off, res_corr, a.out_res, a.out_off)
    print(f"[realign] wrote {orr}  and  {off_path}")


if __name__ == "__main__":
    main()
