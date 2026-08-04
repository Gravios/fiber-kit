#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_features.py — put a simulated waveform into the SESSION'S feature
#  space, not a plausible-looking one.
#
#  Everything this repo measured before now used cosine distance on waveforms.
#  The sort does not cluster on waveforms.  It clusters on the .fet file, and
#  that file is neither the waveform nor a straightforward PCA of it:
#
#    1. the .spk on disk is ALREADY transformed -- the custom channel-difference
#       (SDIFF_CUSTOM_CAR, Method 8) plus the temporal first-difference are
#       applied at extraction, so a consumer must not apply them again;
#    2. the per-channel basis stores FOUR vectors that are not four principal
#       components.  Three of them are the SAME vector at lags -3, 0, +3, with
#       the shift baked in by zero-padding inside the 31-sample window; the
#       fourth is a second component at lag 0.
#
#  Consequence, and it is the reason this module exists: three of every four
#  dimensions per channel are one filter at three time offsets, so the space
#  encodes TIMING explicitly and a plain cosine in it is dominated by
#  sub-sample jitter rather than by shape.  Measured on g5, the 32 nominal
#  dimensions have a participation ratio of 6.1.  Any threshold calibrated on
#  waveform cosine is calibrated in the wrong geometry.
#
#  Verified against the reference session: projecting the recorded .spk through
#  loadPca + project() reproduces the on-disk .fet to |corr| = 1.0000 on all 32
#  columns.  That round-trip is the only reason to trust anything below.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

# neurosuite::core::Method, from libneurosuite-core/pca_projection.hpp.  Mirrored
# rather than re-derived; pca_method_vectors.tsv is the cross-repo contract that
# keeps the two in step.
METHOD = {0: "standard", 1: "sdiff_S1", 2: "sdiff_S2", 3: "sdiff_S3",
          4: "stderiv_S1", 5: "stderiv_S2", 6: "stderiv_S3",
          7: "stderiv_C4", 8: "stderiv_C5"}
TEMPORAL_DIFF = {4, 5, 6, 7, 8}      # hasTemporalDiff()
PCAE_MAGIC = 0x50434145


class PcaBasis:
    """A loaded PCAE basis, plus what the header says was used to train it."""

    __slots__ = ("version", "nch", "data2use", "ncomp", "rec_shift", "centered",
                 "method", "n_input", "means", "evec", "path")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def method_tag(self):
        return METHOD.get(self.method, f"method{self.method}")

    @property
    def temporal_diff(self):
        return self.method in TEMPORAL_DIFF

    def lag_structure(self, tol=0.999, max_lag=8):
        """Recover which components are lag-shifted copies of which.

        Returns [(component, source component, lag), ...] with source == self and
        lag 0 for a component that is nobody's shift.  Detected rather than
        assumed: the lag triple is a session convention baked into the stored
        vectors, and a future session could bake in a different one.  Reading it
        off the basis means the feature space is described by the file rather
        than by a comment that can go stale.
        """
        out = []
        for ch in range(self.nch):
            e = self.evec[ch]
            rows = []
            for k in range(self.ncomp):
                src, lag, best = k, 0, tol
                for j in range(k):
                    for L in range(-max_lag, max_lag + 1):
                        c = abs(_shift_corr(e[j], e[k], L))
                        if c > best:
                            src, lag, best = j, L, c
                rows.append((k, src, lag))
            out.append(rows)
        return out

    def __repr__(self):
        return (f"<PcaBasis {self.method_tag} nCh={self.nch} data2use={self.data2use} "
                f"nComp={self.ncomp} recShift={self.rec_shift} "
                f"centered={bool(self.centered)}>")


def _shift_corr(a, b, lag):
    if lag > 0:
        x, y = a[lag:], b[:-lag]
    elif lag < 0:
        x, y = a[:lag], b[-lag:]
    else:
        x, y = a, b
    d = np.sqrt((x @ x) * (y @ y))
    return 0.0 if d <= 0 else float(x @ y / d)


def load_pca(path):
    """Read a PCAE .pca basis.

    Body is BLOCK-WISE -- all per-channel means, then all per-channel
    eigenvectors -- which is what process_pca writes and what klusters and
    KlustaKwik read.  An interleaved reader parses the header fine and returns
    silently wrong vectors, which is a bug this codebase has already had once.
    """
    raw = open(path, "rb").read()
    h = np.frombuffer(raw[:8], dtype=np.int32)
    if int(h[0]) != PCAE_MAGIC:
        raise ValueError(f"{path}: not a PCAE file (magic {int(h[0]):#x})")
    version = int(h[1])
    core = np.frombuffer(raw[8:28], dtype=np.int32)
    nch, d2u, ncomp, rec, cen = (int(v) for v in core)
    off = 28
    method, n_input = 0, nch
    if version >= 2:
        ext = np.frombuffer(raw[off:off + 8], dtype=np.int32)
        method, n_input = int(ext[0]), int(ext[1])
        off += 8
    if not (0 <= method <= 8):
        raise ValueError(f"{path}: invalid method {method}")
    nm = nch * d2u * 8
    ne = nch * d2u * ncomp * 8
    if len(raw) < off + nm + ne:
        raise ValueError(f"{path}: short file for nCh={nch} data2use={d2u} nComp={ncomp}")
    means = np.frombuffer(raw[off:off + nm], dtype=np.float64).reshape(nch, d2u).copy()
    evec = np.frombuffer(raw[off + nm:off + nm + ne],
                         dtype=np.float64).reshape(nch, ncomp, d2u).copy()
    return PcaBasis(version=version, nch=nch, data2use=d2u, ncomp=ncomp,
                    rec_shift=rec, centered=bool(cen), method=method,
                    n_input=n_input or nch, means=means, evec=evec, path=path)


def project(waves, pca, shift=0):
    """(nspike, nsamp, nchan) already-transformed waveforms -> (nspike, nCh*nComp).

    Channel-major, matching the .fet column order that process_pca writes and
    that featuremask.h maps back to channels.  `shift` moves the extraction
    window, which is how a realignment is applied without rewriting the .spk.

    The input must ALREADY carry whatever transform the basis was trained
    against.  Passing a raw waveform to a stderiv basis produces numbers, not an
    error, and they are meaningless -- so callers coming from the model must go
    through session_transform() first.
    """
    w = np.asarray(waves, np.float64)
    if w.ndim == 2:
        w = w[None]
    n, nsamp, nchan = w.shape
    if nchan < pca.nch:
        raise ValueError(f"basis needs {pca.nch} channels, waveform has {nchan}")
    s = pca.rec_shift + int(shift)
    if s < 0 or s + pca.data2use > nsamp:
        raise ValueError(f"window [{s},{s+pca.data2use}) outside {nsamp} samples")
    out = np.empty((n, pca.nch, pca.ncomp))
    for ch in range(pca.nch):
        x = w[:, s:s + pca.data2use, ch]
        if pca.centered:
            x = x - pca.means[ch]
        out[:, ch, :] = x @ pca.evec[ch].T
    return out.reshape(n, pca.nch * pca.ncomp)


# ── model -> session space ──────────────────────────────────────────────────
def session_transform(waves, sdiff_sets=None, temporal_diff=True, drop_last=False):
    """Apply what the extractor applies, so a simulated waveform matches the .spk.

    Order matters and is the extractor's: the channel difference FIRST, then the
    temporal first-difference.  Reversing them is not equivalent once the
    reference set spans channels with different time courses.

    The temporal difference is taken as x[t] - x[t-1] with the first sample held,
    so the output keeps the input's sample count and the peak index stays where
    nSamples/peakSampleIndex say it is.  Dropping a sample instead would shift
    every downstream window by one.
    """
    try:
        from . import morpho_eap as me
    except ImportError:
        import morpho_eap as me
    w = np.asarray(waves, np.float64)
    if sdiff_sets is not None:
        w = me.stderiv(w, sdiff_sets, drop_last=drop_last)
    if temporal_diff:
        d = np.empty_like(w)
        d[..., 1:, :] = np.diff(w, axis=-2)
        d[..., 0, :] = d[..., 1, :]
        w = d
    return w


def to_features(waves, pca, sdiff_sets=None, shift=0, quantize=False):
    """Simulated raw footprints -> the session's .fet columns, in one call."""
    w = session_transform(waves, sdiff_sets, temporal_diff=pca.temporal_diff)
    f = project(w, pca, shift=shift)
    return np.rint(f).astype(np.int64) if quantize else f


# ── realignment ─────────────────────────────────────────────────────────────
def align_shifts(waves, template=None, max_shift=3, chan=None):
    """Integer sample shift per spike that best matches `template`.

    Cross-correlation against the cluster mean on its dominant channel -- the
    same criterion klusters' realignSpikes uses.  Restricted to integer shifts
    on purpose: the feature space already carries three lags of PC1, so
    sub-sample timing is representable IN the features, and interpolating the
    waveform to chase it would filter the signal before the basis sees it.
    """
    w = np.asarray(waves, np.float64)
    if template is None:
        template = w.mean(0)
    t = np.asarray(template, np.float64)
    if chan is None:
        chan = int(np.argmax(t.max(0) - t.min(0)))
    tv = t[:, chan]
    tv = tv - tv.mean()
    best = np.zeros(len(w), int)
    score = np.full(len(w), -np.inf)
    for s in range(-max_shift, max_shift + 1):
        x = np.roll(w[:, :, chan], -s, axis=1)
        if s > 0:
            x[:, -s:] = w[:, -1:, chan]
        elif s < 0:
            x[:, :-s] = w[:, :1, chan]
        x = x - x.mean(1, keepdims=True)
        c = x @ tv / (np.linalg.norm(x, axis=1) * np.linalg.norm(tv) + 1e-30)
        m = c > score
        score[m] = c[m]; best[m] = s
    return best, score


def realign_features(waves, pca, shifts):
    """Project each spike through its own realigned window.

    Grouped by shift so the projection stays a matmul per (channel, shift)
    rather than a Python loop over spikes; on a 400k-spike group the difference
    is minutes against hours.
    """
    w = np.asarray(waves, np.float64)
    out = np.empty((len(w), pca.nch * pca.ncomp))
    for s in np.unique(shifts):
        m = shifts == s
        out[m] = project(w[m], pca, shift=int(s))
    return out
