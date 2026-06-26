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
#  ── .pca.N / .pcaD.N binary layout — PCAE (mirror of libneurosuite-core) ──
#    int32 header   :  magic=0x50434145 ("PCAE"), version,
#                      nCh, data2use, nComp, recShift, centered,
#                      [v2+: method, nInputChannels]
#    nCh   x  data2use            double  : per-channel mean window  (ALL channels)
#    nCh   x  nComp x data2use    double  : eigenvectors, col-major  (ALL channels)
#  Block-wise (all means, then all eigenvectors) and means written UNCONDITIONALLY
#  (the `centered` flag governs whether projection subtracts the mean, NOT whether
#  the means are stored).  This module reads+writes the SAME PCAE format as
#  neurosuite::core::{loadPca,writePca}; the Method enum below mirrors
#  neurosuite::core::Method exactly (cross-repo contract).
#
#  The projection window is samples [recShift : recShift + data2use] of each spike;
#  recShift is stored in the header (so the window start is not guessed).
# ════════════════════════════════════════════════════════════════════════════
import argparse
import enum
import struct
import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy
except ImportError:
    import neuro_io as nio, session_yaml as sy


# ── PCAE format + transform method (mirror of libneurosuite-core/pca_projection.hpp) ──
PCAE_MAGIC   = 0x50434145          # "PCAE"
PCAE_VERSION = 2                   # v2 adds method + nInputChannels


class Method(enum.IntEnum):
    """Transform a basis was trained against.  Integer values MUST match
    neurosuite::core::Method exactly — a basis written by either repo is read by
    the other.  Flattened (no separate sdiffOrder), so illegal combinations are
    unrepresentable."""
    STANDARD          = 0          # raw waveform, no transform
    SDIFF_FIRST       = 1          # spatial derivative only
    SDIFF_LAPLACIAN   = 2
    SDIFF_ALLPAIRS    = 3
    STDERIV_FIRST     = 4          # spatial derivative + temporal first-difference
    STDERIV_LAPLACIAN = 5
    STDERIV_ALLPAIRS  = 6          # canonical


def method_tag(m):
    """Chain-of-custody tag ('standard' / 'sdiff' / 'stderiv') for a Method."""
    m = Method(m)
    if m in (Method.SDIFF_FIRST, Method.SDIFF_LAPLACIAN, Method.SDIFF_ALLPAIRS):
        return "sdiff"
    if m in (Method.STDERIV_FIRST, Method.STDERIV_LAPLACIAN, Method.STDERIV_ALLPAIRS):
        return "stderiv"
    return "standard"


def has_temporal_diff(m):
    """True if the method applies the temporal first-difference (stderiv vs sdiff/raw)."""
    return Method(m) in (Method.STDERIV_FIRST, Method.STDERIV_LAPLACIAN,
                         Method.STDERIV_ALLPAIRS)

def read_pcad(path):
    """Read a neurosuite-3 PCAE .pca/.pcaD basis.

    Returns dict(nCh, data2use, nComp, centered, recShift, method, nInputChannels,
                 means  (nCh, data2use),
                 evec   (nCh, nComp, data2use)  -- evec[ch,k] is eigenvector k).
    Header field order matches neurosuite::core (recShift BEFORE centered).  Means
    are always present (block-wise body).  v1 files (no transform descriptor) load
    as method=STANDARD, nInputChannels=0."""
    raw = open(path, "rb").read()
    if len(raw) < 8:
        raise ValueError(f"{path}: too short for a PCAE header")
    magic, version = struct.unpack("<2i", raw[:8])
    if magic != PCAE_MAGIC:
        raise ValueError(f"{path}: not a PCAE file (magic={magic:#x}); regenerate as PCAE")
    if version not in (1, 2):
        raise ValueError(f"{path}: unsupported PCAE version {version}")
    nCh, data2use, nComp, recShift, centered = struct.unpack("<5i", raw[8:28])
    if version >= 2:
        method, nInputChannels = struct.unpack("<2i", raw[28:36])
        off = 36
    else:
        method, nInputChannels, off = int(Method.STANDARD), 0, 28
    body = np.frombuffer(raw, "<f8", offset=off)
    n_means = nCh * data2use
    n_evec = nCh * nComp * data2use
    if body.size != n_means + n_evec:
        raise ValueError(f"{path}: {body.size} doubles != means {n_means} + evec {n_evec}")
    means = body[:n_means].reshape(nCh, data2use).copy()
    evec = body[n_means:].reshape(nCh, nComp, data2use).copy()
    return dict(nCh=nCh, data2use=data2use, nComp=nComp, centered=int(centered),
                recShift=recShift, method=int(method), nInputChannels=int(nInputChannels),
                means=means, evec=evec)


def read_pca(base, elec, prefer=None):
    """Resolve and read <base>.pca[.<variant>].<elec> (RAW/standard first by default), returning
    the read_pcad dict tagged _pca/_path for amplitude-basis use.  Centralizes .pca resolve+read
    here next to read_pcad (neuro_io owns res/clu/fet/spk; fiber_pca owns .pca/.pcaD — and since
    fiber_pca imports neuro_io, this is the only side that can hold both).  RAW/standard only by
    default — never the stderiv .pcaD.  Raises FileNotFoundError if absent."""
    r = nio.resolve_input(base, "pca", elec, prefer or nio.prefer_standard())
    if not r.found:
        raise FileNotFoundError(f"no .pca for {base} elec {elec}")
    b = read_pcad(r.path)
    b["_pca"] = True                                      # tag for fiber_localize._profile dispatch
    b["_path"] = r.path
    return b


def write_pcad(path, means, evec, recShift, centered=0,
               method=Method.STANDARD, n_input_channels=0):
    """Write a PCAE v2 .pca/.pcaD (block-wise means-then-evec, col-major
    eigenvectors), byte-compatible with neurosuite::core::writePca.
    means (nCh,data2use); evec (nCh,nComp,data2use).  `method` is the transform the
    basis was trained against; `n_input_channels` is the raw channel count the
    transform consumes (0 ⇒ == nCh, i.e. no channel drop)."""
    means = np.ascontiguousarray(means, np.float64)
    evec = np.ascontiguousarray(evec, np.float64)
    nCh, data2use = means.shape
    nComp = evec.shape[1]
    nin = int(n_input_channels) if int(n_input_channels) > 0 else nCh
    with open(path, "wb") as f:
        # header order matches core: magic, version, nCh, data2use, nComp,
        # recShift, centered, method, nInputChannels
        f.write(struct.pack("<9i", PCAE_MAGIC, PCAE_VERSION, nCh, data2use, nComp,
                            int(recShift), int(centered), int(method), nin))
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


def read_cluster_basis(base, elec, method="standard"):
    """Resolve and read the GLOBAL per-channel PCA basis used for clustering features
    (the ndm_pca / process_pca basis).  `method`: 'standard' (raw .pca) or 'stderiv'
    (.pca.stderiv).  Returns the basis dict (means, evec, recShift, data2use, centered)
    or None if absent, so callers can fall back to a per-call local SVD.  nComponents and
    the window come from the basis header (= the session.yaml `nFeatures` the basis was fit
    with), so regenerating the basis at a different nFeatures (2 -> 4) or with --varimax
    propagates into clustering without any code change here."""
    prefer = nio.prefer_standard() if method in (None, "standard") else [method]
    try:
        r = nio.resolve_input(base, "pca", elec, prefer)
    except Exception:
        return None
    if not getattr(r, "found", False):
        return None
    b = read_pcad(r.path)
    b["_path"] = r.path
    return b


def cluster_features(spk, basis, *, realign=True, dims=None):
    """Project (optionally realigned) waveforms onto the GLOBAL ndm_pca basis, returning
    (N, nCh*nComp) cluster features in the .fet column order.  This is the shared-basis
    replacement for a per-call local SVD: one basis across every chunk/run, so features are
    comparable and a basis change (nFeatures, varimax) propagates.  The stderiv all-pairwise
    spatial derivative makes the trailing channel linearly dependent, so a basis with exactly
    one fewer channel than the input drops the trailing channel (process_pca_stderiv's
    reduction).  Returns None on an unresolvable channel-count mismatch so the caller falls
    back to local_features().

    `dims`: if given and < nCh*nComp, reduce the projected features to their top-`dims` SVD
    scores (mean-centred), exactly as local_features() reduces the raw masked waveforms.  The
    channel-major projection is nCh*nComp wide (14 at nFeatures=2, 28 at nFeatures=4); feeding
    that straight into a full-covariance CEM (klustakwik) or GMM costs O(D^2) params and needs
    >= D+2 points per cluster, which silently over-merges.  Reducing to the caller's target
    dims keeps the drift-stable global basis as the feature SPACE while handing the splitter the
    dimensionality it is tuned for.  None (default) keeps the full projection."""
    try:
        from . import fiber_lib as fl
    except ImportError:
        import fiber_lib as fl
    w = np.asarray(spk, np.float64)
    if realign:
        w = fl.realign(w)
    nb = basis["evec"].shape[0]
    if w.shape[2] != nb:
        if w.shape[2] == nb + 1:
            w = w[:, :, :nb]                              # stderiv: drop trailing dependent channel
        else:
            return None
    F = project(extract_windows(w, int(basis["recShift"]), int(basis["data2use"])), basis)
    if dims is not None and 0 < int(dims) < F.shape[1]:
        Fc = F - F.mean(0)
        U, S, _ = np.linalg.svd(Fc, full_matrices=False)
        F = U[:, :int(dims)] * S[:int(dims)]             # top-`dims` PCs of the global-basis features
    return F


def local_features(spk, dims=12, *, mask=None, realign=True):
    """Legacy per-call local SVD of the masked window -> top `dims` scores (the historical
    klustakwik feature path).  The basis is refit on every call, so features are NOT
    comparable across chunks/runs; used only as the fallback when no global basis exists."""
    try:
        from . import fiber_lib as fl
    except ImportError:
        import fiber_lib as fl
    w = np.asarray(spk, np.float64)
    if realign:
        w = fl.realign(w)
    if mask is not None:
        w = w[:, mask, :]
    M = w.reshape(len(w), -1)
    M = M - M.mean(0)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :dims] * S[:dims]


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
              f"centered={b['centered']} recShift={b['recShift']} "
              f"method={Method(b['method']).name}({method_tag(b['method'])}) "
              f"nInputChannels={b['nInputChannels']}  "
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
    # Tag the transform.  This fit runs on .spkD (already stderiv-transformed by the
    # extractor) without dropping a channel, so nInputChannels == nch (no drop).  The
    # spatial order defaults to the canonical ALLPAIRS; a basis built from a .spkD made
    # with a different order should carry the matching StderivLaplacian/First instead
    # (follow-up: read sdiffOrder from the session YAML, as the klusters nudge path does).
    method = Method.STDERIV_ALLPAIRS if a.stderiv else Method.STANDARD
    write_pcad(out, means, evec, rec_shift, centered=int(a.centered),
               method=method, n_input_channels=nch)
    print(f"[pca] fit {len(win)} spikes, {means.shape[0]}ch x {a.ncomp} comp, data2use={data2use}, "
          f"recShift={rec_shift} method={method.name} -> wrote {out}")


if __name__ == "__main__":
    main()
