# ════════════════════════════════════════════════════════════════════════════
#  fiber_pca.py — standalone neurosuite-3 .pca/.pcaD basis: I/O, fit, project,
#  and the Stage-2 PCA-projection-energy canonical shift.
#
#  Makes fiber-kit self-contained for the Klusters-style realign + reproject that
#  process_pca / process_alignspikes_pca (ndmanager-plugins) provide in C++, so the
#  package does not depend on the state of that pipeline.  Pairs with
#  fiber_realign.template_offsets, which already implements Stage 1 (per-spike
#  cross-correlation alignment to the cluster template, iters=2, max_shift=5).
#
#  ── .pca.N / .pcaD.N binary layout (authoritative: process_pca.cpp writer) ──
#    int32 x5 header :  nCh, data2use, nComp, centered, recShift
#    nCh   x  data2use            double  : per-channel mean window  (ALL channels)
#    nCh   x  nComp x data2use    double  : eigenvectors, col-major  (ALL channels)
#  Means are written UNCONDITIONALLY by process_pca (the `centered` flag governs
#  whether projection subtracts the mean, NOT whether the means are stored).  NB the
#  process_alignspikes_pca *reader* instead reads means/evec interleaved per channel
#  and gates the mean read on `centered`; that disagrees with this writer and would
#  mis-parse a real centered=0 file -- see read_pcad's note.  This module follows the
#  WRITER (the file actually on disk), validated by a byte-exact round-trip.
#
#  The projection window is samples [recShift : recShift + data2use] of each spike;
#  recShift is stored in the header (so the window start is not guessed).
# ════════════════════════════════════════════════════════════════════════════
import argparse
import struct
import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy
except ImportError:
    import neuro_io as nio, session_yaml as sy

HEADER_FMT = "<5i"          # nCh, data2use, nComp, centered, recShift
HEADER_SIZE = 20


def read_pcad(path):
    """Read a neurosuite-3 .pca/.pcaD basis (process_pca writer layout).

    Returns dict(nCh, data2use, nComp, centered, recShift,
                 means  (nCh, data2use),
                 evec   (nCh, nComp, data2use)  -- evec[ch,k] is eigenvector k).
    Means are always read (process_pca writes them unconditionally); a file written
    by a tool that omits them when centered=0 would be shorter and is detected here."""
    raw = open(path, "rb").read()
    nCh, data2use, nComp, centered, recShift = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
    body = np.frombuffer(raw, "<f8", offset=HEADER_SIZE)
    n_means = nCh * data2use
    n_evec = nCh * nComp * data2use
    if body.size == n_means + n_evec:
        means = body[:n_means].reshape(nCh, data2use).copy()
        evec = body[n_means:].reshape(nCh, nComp, data2use).copy()
    elif body.size == n_evec:                 # means omitted (some writers, centered=0)
        means = np.zeros((nCh, data2use))
        evec = body.reshape(nCh, nComp, data2use).copy()
    else:
        raise ValueError(f"{path}: {body.size} doubles != means {n_means} + evec {n_evec}")
    return dict(nCh=nCh, data2use=data2use, nComp=nComp, centered=int(centered),
                recShift=recShift, means=means, evec=evec)


def write_pcad(path, means, evec, recShift, centered=0):
    """Write a .pca/.pcaD byte-compatible with process_pca (means-then-evec, col-major
    eigenvectors).  means (nCh,data2use); evec (nCh,nComp,data2use)."""
    means = np.ascontiguousarray(means, np.float64)
    evec = np.ascontiguousarray(evec, np.float64)
    nCh, data2use = means.shape
    nComp = evec.shape[1]
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, nCh, data2use, nComp, int(centered), int(recShift)))
        f.write(means.tobytes())                          # nCh x data2use
        f.write(evec.tobytes())                           # nCh x (nComp x data2use), col-major per ch
    return path


def extract_windows(spk, recShift, data2use):
    """(N, nsamp, nCh) waveforms -> (N, data2use, nCh) PCA window [recShift:recShift+data2use]."""
    return np.asarray(spk, np.float64)[:, recShift:recShift + data2use, :]


def fit_basis(windows, nComp=3, centered=True):
    """Per-channel temporal PCA (the process_pca fit).  windows (N, data2use, nCh).
    Returns means (nCh,data2use), evec (nCh,nComp,data2use).  Eigenvectors come from the
    mean-subtracted covariance regardless of `centered`; `centered` is only recorded in
    the file and governs projection-time mean subtraction."""
    N, data2use, nCh = windows.shape
    means = np.zeros((nCh, data2use)); evec = np.zeros((nCh, nComp, data2use))
    for ch in range(nCh):
        X = windows[:, :, ch]
        mu = X.mean(0); Xc = X - mu
        C = (Xc.T @ Xc) / max(N - 1, 1)
        w, V = np.linalg.eigh(C)
        V = V[:, ::-1][:, :nComp]                         # top nComp eigenvectors (cols)
        # sign convention: largest-magnitude entry positive (reproducible across runs)
        for k in range(nComp):
            if V[np.argmax(np.abs(V[:, k])), k] < 0:
                V[:, k] = -V[:, k]
        means[ch] = mu; evec[ch] = V.T
    return means, evec


def project(windows, basis):
    """Project windows (N, data2use, nCh) onto the basis -> features (N, nCh*nComp),
    channel-major then component (the .fet column order of process_pca).  Subtracts the
    per-channel mean iff basis['centered']."""
    means, evec, centered = basis["means"], basis["evec"], basis["centered"]
    nCh, nComp, _ = evec.shape
    out = np.empty((len(windows), nCh * nComp))
    col = 0
    for ch in range(nCh):
        X = windows[:, :, ch]
        if centered:
            X = X - means[ch]
        out[:, col:col + nComp] = X @ evec[ch].T          # (N, nComp)
        col += nComp
    return out


def projection_energy(mean_window, basis):
    """Stage-2 metric E = sum_ch sum_k <evec_ch,k, mean_window_ch (- mu if centered)>^2.
    mean_window (data2use, nCh).  Higher E => the mean lies more strongly along the
    cluster's principal axes (process_alignspikes_pca's pcaProjectionEnergy)."""
    means, evec, centered = basis["means"], basis["evec"], basis["centered"]
    nCh = min(evec.shape[0], mean_window.shape[1])
    e = 0.0
    for ch in range(nCh):
        x = mean_window[:, ch] - (means[ch] if centered else 0.0)
        s = evec[ch] @ x                                  # (nComp,)
        e += float(s @ s)
    return e


def stage2_shift(cluster_spikes, basis, *, max_global=4):
    """Rigid whole-cluster shift maximizing PCA-projection energy of the cluster mean.

    cluster_spikes (n, nsamp, nCh).  Returns the integer shift s* in [-max_global, max_global]
    that, applied to every spike, brings the cluster mean's window onto its PC-energy maximum
    (Stage 2 of process_alignspikes_pca).  Uses circular shift of the extracted window -- exact
    when re-extraction from .fil is unavailable; for the typical small s* the window edges are
    ~0 so the two agree."""
    r, d = basis["recShift"], basis["data2use"]
    best_s, best_e = 0, -1.0
    mean_full = np.asarray(cluster_spikes, np.float64).mean(0)   # (nsamp, nCh)
    for s in range(-max_global, max_global + 1):
        win = np.roll(mean_full, s, axis=0)[r:r + d, :]
        e = projection_energy(win, basis)
        if e > best_e:
            best_e, best_s = e, s
    return best_s


def _shift_spikes(spk, shift_per_spike):
    """Circular-shift each spike along the sample axis (a positive shift moves the waveform
    later).  Standalone substitute for re-extracting from .fil at the corrected timestamp;
    exact for the small shifts realign produces (window edges ~0)."""
    out = np.array(spk, np.float64)
    for s in np.unique(shift_per_spike):
        if s:
            m = shift_per_spike == s
            out[m] = np.roll(out[m], int(s), axis=1)
    return out


def realign_pca(spk, clu, res, basis, *, max_shift=5, iters=2, min_n=20, max_global=4):
    """Full standalone Klusters-style realign on in-memory arrays.

    Stage 1 — per-spike xcorr alignment to the cluster template (fiber_realign.template_offsets).
    Stage 2 — per-cluster rigid shift to the PCA-projection-energy maximum (this module).
    Then reproject the realigned windows onto `basis` to refresh features.

    spk (N,nsamp,nCh), clu (N,), res (N,).  Returns dict(res, spk, fet, ioff, s2) with the
    corrected times, realigned waveforms, refreshed .fet, and the two shift components.
    To align a waveform to its template we roll by -ioff (the inverse of the measured lag);
    the .res convention res+ioff matches fiber_realign."""
    try:
        from . import fiber_realign as fr
    except ImportError:
        import fiber_realign as fr
    spk = np.asarray(spk, np.float64)
    off, ioff = fr.template_offsets(spk, clu, max_shift=max_shift, iters=iters, min_n=min_n)
    spk1 = _shift_spikes(spk, -ioff)                       # canonicalize waveform to template
    s2 = np.zeros(len(clu), np.int32)
    by = {}
    for i, l in enumerate(clu):
        by.setdefault(int(l), []).append(i)
    for u, rows in by.items():
        if u < 2 or len(rows) < min_n:
            continue
        idx = np.asarray(rows)
        s = stage2_shift(spk1[idx], basis, max_global=max_global)
        if s:
            s2[idx] = s
    spk2 = _shift_spikes(spk1, -s2)
    win = extract_windows(spk2, basis["recShift"], basis["data2use"])
    fet = project(win, basis)
    return dict(res=res + ioff.astype(np.int64) + s2.astype(np.int64),
                spk=spk2, fet=fet, ioff=ioff, s2=s2)


def main():
    ap = argparse.ArgumentParser(description="neurosuite-3 .pca/.pcaD basis: fit a per-channel "
                                             "PCA basis from .spk+.clu and write the binary, or inspect one.")
    sy.add_session_args(ap, channels=False, ntotal=False, nsamp=False, nchan=False, sr=False)
    ap.add_argument("--info", default=None, help="print the header of an existing .pcaD and exit")
    ap.add_argument("--clu-method", default="stderiv"); ap.add_argument("--clu-stage", default="refine")
    ap.add_argument("--data2use", type=int, default=None, help="PCA window length (default: nSamples)")
    ap.add_argument("--rec-shift", type=int, default=None, help="window start sample (default: peak - data2use//2)")
    ap.add_argument("--ncomp", type=int, default=3)
    ap.add_argument("--centered", action="store_true", help="store centered flag (projection subtracts the mean)")
    ap.add_argument("--stderiv", action="store_true", help="fit on .spkD (stderiv) instead of .spk")
    ap.add_argument("--out", default=None, help="output .pcaD path (default <base>.pca<D>.<group>)")
    a = ap.parse_args()

    if a.info:
        b = read_pcad(a.info)
        print(f"[pca] {a.info}: nCh={b['nCh']} data2use={b['data2use']} nComp={b['nComp']} "
              f"centered={b['centered']} recShift={b['recShift']}  "
              f"(evec per-ch orthonormal: {np.allclose(b['evec'][0] @ b['evec'][0].T, np.eye(b['nComp']), atol=1e-6)})")
        return

    cfg = sy.resolve_session_params(a.session, a.group, require=("nchan", "nsamp"))
    base, elec, nsamp, nch = cfg["base"], cfg["group"], cfg["nsamp"], cfg["nchan"]
    data2use = a.data2use or nsamp
    rec_shift = a.rec_shift if a.rec_shift is not None else max(0, nsamp // 2 - data2use // 2)
    spk = (nio.open_spkD(base, elec, nsamp, nch) if a.stderiv
           else nio.open_spk_raw(base, elec, nsamp, nch)[0])
    win = extract_windows(spk[:], rec_shift, data2use)
    means, evec = fit_basis(win, nComp=a.ncomp, centered=a.centered)
    out = a.out or nio.session_path(base, "pcaD" if a.stderiv else "pca", elec)
    write_pcad(out, means, evec, rec_shift, centered=int(a.centered))
    print(f"[pca] fit {len(win)} spikes, {means.shape[0]}ch x {a.ncomp} comp, data2use={data2use}, "
          f"recShift={rec_shift} -> wrote {out}")


if __name__ == "__main__":
    main()
