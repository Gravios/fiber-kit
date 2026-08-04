#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  morpho_envelope.py — how much may one neuron's waveform change, and along
#  which direction, before a merge stops being physiologically possible?
#
#  This is the module the rest of the modelling stack exists to feed.  A merge
#  gate calibrated on the recording is circular: the threshold is fitted to the
#  same sort whose errors it is supposed to catch, so it inherits that sort's
#  over-splitting and its false merges alike.  A gate calibrated on a forward
#  model is not -- it says what a SINGLE NEURON CAN DO, independent of any
#  clustering, and a pair of fragments that would require more change than that
#  cannot be the same cell no matter how convincing the feature-space overlap.
#
#  The envelope is a FUNCTION OF AMPLITUDE RATIO, not a scalar.  Two fragments
#  at the same energy are allowed almost no shape difference; two fragments
#  differing 2x in energy -- one from a burst's first spike, one from its
#  fourth -- are allowed a great deal, because Na availability really does
#  reshape the spike over that range.  A single cosine threshold has to be set
#  loose enough for the second case and is then useless for the first, which is
#  the specific way a scalar gate fails.
#
#  Two quantities come out of it, and they answer different questions:
#
#    cos_env(rho)   the maximum cosine distance a single cell can show between
#                   two of its own spikes whose energies differ by rho.  This is
#                   the admissibility test.
#
#    along_frac     the fraction of that variance that lies ALONG the cell's own
#                   d(r) curve rather than orthogonal to it.  This is a test of
#                   fiber-kit's central premise -- if physiological modulation
#                   moved waveforms orthogonally to d(r), the fiber abstraction
#                   would be measuring the wrong thing.
#
#  What is deliberately NOT in the envelope: electrode drift, noise, and
#  overlapping spikes.  Those are real causes of waveform difference but they
#  are not the CELL changing, and folding them in would make the gate unable to
#  distinguish "this pair needs 8 um of drift to reconcile" from "this pair
#  needs a physiologically impossible spike".  Combine them downstream, where
#  the drift budget is known.
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

try:
    from . import morpho_eap as me
except ImportError:
    import morpho_eap as me


# ── spike trains ────────────────────────────────────────────────────────────
def burst_times(n=4, isi=5.0, t0=5.0):
    """A complex-spike burst: n spikes at a fixed short ISI."""
    return [t0 + k * isi for k in range(n)]


def train_times(pattern, t0=5.0):
    """Named ISI patterns spanning the physiological range a CA1 cell shows.

    'burst4_5' etc. are complex spikes; 'tonic_N' are steady trains; 'recover'
    probes the slow-inactivation recovery limb, which is where the largest
    within-cell amplitude ratio lives (a burst's last spike vs an isolated spike
    seconds later).
    """
    kind = pattern.split("_")
    if kind[0] == "single":
        return [t0]
    if kind[0].startswith("burst"):
        n = int(kind[0][5:]); isi = float(kind[1])
        return burst_times(n, isi, t0)
    if kind[0] == "tonic":
        hz = float(kind[1]); n = int(kind[2]) if len(kind) > 2 else 6
        return [t0 + k * 1000.0 / hz for k in range(n)]
    if kind[0] == "recover":
        isi = float(kind[1]) if len(kind) > 1 else 4.0
        gap = float(kind[2]) if len(kind) > 2 else 300.0
        return burst_times(4, isi, t0) + [t0 + 3 * isi + gap]
    raise ValueError(f"unknown ISI pattern {pattern!r}")


def simulate_train(cell, times, dt=0.01, stim_amp=10.0, stim_dur=0.8, tail=6.0,
                   drive=None, plateau=0.0, plateau_pad=2.0):
    """Integrate a whole train in ONE run and return (t, im, v).

    One run, not one per spike: the entire point is that spike k depends on
    spikes 1..k-1 through Na availability, so simulating them independently
    would erase the effect being measured.

    plateau (nA) is a sustained somatic current spanning the burst.  It is a
    STAND-IN for the dendritic calcium plateau that underlies a real CA1 complex
    spike, not a mechanism: this model has no calcium channels, and without some
    substitute the soma repolarizes fully between spikes, Na inactivation barely
    accumulates, and the model reports a burst amplitude decrement several times
    smaller than the recorded one.  Because a merge gate built on an
    UNDER-estimated envelope rejects legitimate merges, that error runs in the
    dangerous direction for the over-splitting problem this is meant to help
    with -- so the substitute is provided, labelled, and swept rather than
    quietly omitted.  Any envelope built with plateau > 0 must carry the value
    used, which build_envelope's meta does.
    """
    t_stop = max(times) + tail
    nt = int(round(t_stop / dt))
    istim = np.zeros(cell.n)
    im = np.zeros((nt, cell.n), np.float32)
    v = np.zeros((nt, cell.n), np.float32)
    p_lo, p_hi = (min(times) - plateau_pad, max(times) + plateau_pad)
    for k in range(nt):
        t = k * dt
        istim[:] = 0.0
        if plateau and p_lo <= t < p_hi:
            istim[0] = plateau
        if any(t0 <= t < t0 + stim_dur for t0 in times):
            istim[0] = stim_amp
        vv, ii = cell.step(dt, istim, drive.at(k) if drive is not None else None)
        im[k] = ii; v[k] = vv
    return np.arange(nt) * dt, im, v


def train_footprints(im, cmp_, sites, times, dt, sr=32552.0, nsamp=42, peak=21,
                     sigma=me.SIGMA_DEFAULT, win_ms=3.0, v=None, v_thresh=0.0,
                     detect_uv=0.0):
    """Per-spike footprints from one train.

    detect_uv drops spikes whose peak-channel amplitude falls below a detection
    threshold, and it is the parameter that actually BOUNDS the envelope: the
    largest amplitude ratio a merge can legitimately span is not set by the cell
    but by (largest spike) / (detection threshold), because a spike below
    threshold is never detected and so never joins a cluster to be merged.  An
    envelope built with detect_uv = 0 licenses ratios no sorter can ever
    encounter, and inherits the wild shapes of near-threshold events.

    v / v_thresh drop stimulus events that did NOT produce a somatic spike.
    This is not tidying: late in a fast burst the cell enters depolarization
    block and the stimulus yields a subthreshold hump.  A real sorter never sees
    those -- they are not detected -- so letting them into the envelope would
    license merges that no DETECTED pair of spikes could ever require, which is
    the precise way a physiological gate becomes permissive by accident.

    The filter and resample are applied to the WHOLE trace before windowing, not
    per spike, so the band-pass sees the true spike-to-spike context.  Cutting
    first would let each spike's window edge ring differently and manufacture
    shape variance that no cell produced.
    """
    K = me.transfer_matrix(cmp_, sites, sigma=sigma)
    vf = me.bandpass(me.extracellular(im, K), dt)
    vr, tr = me.resample(vf, dt, sr)
    amp = vr.max(0) - vr.min(0)
    ch = int(np.argmax(amp))
    out, keep = [], []
    for t0 in times:
        # Locate the trough inside the spike's own interval, then cut from the
        # FULL trace around that absolute index.  Cutting a sub-segment first and
        # searching inside it fails for exactly the common case: the trough sits
        # ~0.6 ms after stimulus onset, closer to the segment start than `peak`,
        # so every spike would be silently dropped.  Cutting from the full trace
        # also means the pre-trough samples contain the previous spike's tail
        # during a burst -- which is correct, since a real .spk window does too.
        a = int(np.searchsorted(tr, t0))
        b = int(np.searchsorted(tr, t0 + win_ms))
        if b <= a:
            continue
        if v is not None:
            i0, i1 = int(t0 / dt), int(min((t0 + win_ms) / dt, len(v)))
            if i1 <= i0 or float(v[i0:i1, 0].max()) < v_thresh:
                continue
        k = a + int(np.argmin(vr[a:b, ch]))
        lo, hi = k - peak, k - peak + nsamp
        if lo < 0 or hi > len(vr):
            continue
        w = vr[lo:hi]
        if detect_uv > 0.0 and float((w.max(0) - w.min(0)).max()) < detect_uv:
            continue
        out.append(w); keep.append(t0)
    return (np.stack(out) if out else np.zeros((0, nsamp, len(sites)))), np.asarray(keep)


# ── envelope ────────────────────────────────────────────────────────────────
def _energy(W):
    return np.linalg.norm(W.reshape(len(W), -1), axis=1)


def _dirs(W):
    X = W.reshape(len(W), -1).astype(float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-30)


def pairwise(W):
    """(amplitude ratio >= 1, cosine distance) for every within-cell spike pair."""
    r = _energy(W); D = _dirs(W)
    C = np.clip(D @ D.T, -1.0, 1.0)
    iu = np.triu_indices(len(W), 1)
    hi = np.maximum(r[iu[0]], r[iu[1]]); lo = np.minimum(r[iu[0]], r[iu[1]])
    return hi / np.maximum(lo, 1e-30), 1.0 - C[iu]


def along_curve_fraction(W, nq=5):
    """Fraction of the direction variance that the d(r) curve accounts for.

    Bin spikes by energy, take the mean direction per bin (that IS the d(r)
    curve, the same construction fiber_geometry.fiber_curve uses), and compare
    the between-bin scatter with the total.  A high value means physiological
    modulation slides a unit along one smooth curve -- the fiber premise; a low
    value would mean it scatters, and the premise would be wrong.
    """
    if len(W) < nq + 1:
        return np.nan, np.nan
    r = _energy(W); D = _dirs(W)
    edges = np.quantile(r, np.linspace(0, 1, nq + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    gm = D.mean(0)
    tot = float(((D - gm) ** 2).sum())
    between = 0.0
    for k in range(nq):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if m.sum():
            between += m.sum() * float(((D[m].mean(0) - gm) ** 2).sum())
    return (between / max(tot, 1e-30)), tot


class Envelope:
    """Admissible within-cell waveform change, as a function of amplitude ratio.

    Stored as a monotone step function over log amplitude-ratio bins: the
    quantile `q` of cosine distance observed in the model for that ratio,
    made non-decreasing in ratio (a wider ratio can never license LESS change --
    the raw bin estimates are noisy at the sparse high-ratio end, and a
    non-monotone gate would reject a pair while accepting a more extreme one).
    """

    __slots__ = ("edges", "cos_thr", "q", "ratio_max", "along_frac", "meta")

    def __init__(self, edges, cos_thr, q, ratio_max, along_frac, meta=None):
        self.edges = np.asarray(edges, float); self.cos_thr = np.asarray(cos_thr, float)
        self.q = float(q); self.ratio_max = float(ratio_max)
        self.along_frac = float(along_frac); self.meta = meta or {}

    def allowed(self, ratio):
        """Max admissible cosine distance at this amplitude ratio (>= 1)."""
        rho = np.maximum(np.asarray(ratio, float), 1.0)
        k = np.clip(np.searchsorted(self.edges, rho, side="right") - 1,
                    0, len(self.cos_thr) - 1)
        return self.cos_thr[k]

    def admissible(self, temp_a, temp_b):
        """Could one neuron have produced both templates?

        Returns (ok, ratio, observed cosine distance, allowed).  Two ways to
        fail, reported separately because they mean different things: the
        amplitude ratio exceeds anything the model's cell reaches (ratio_max),
        or the SHAPE differs by more than that ratio licenses.
        """
        a = np.asarray(temp_a, float).ravel(); b = np.asarray(temp_b, float).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        rho = float(max(na, nb) / max(min(na, nb), 1e-30))
        d = 1.0 - float((a / max(na, 1e-30)) @ (b / max(nb, 1e-30)))
        thr = float(self.allowed(rho))
        return bool(rho <= self.ratio_max and d <= thr), rho, d, thr

    def save(self, path):
        np.savez_compressed(path, edges=self.edges, cos_thr=self.cos_thr,
                            q=self.q, ratio_max=self.ratio_max,
                            along_frac=self.along_frac,
                            meta=np.array(sorted(self.meta.items()), dtype=object))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        return Envelope(z["edges"], z["cos_thr"], float(z["q"]), float(z["ratio_max"]),
                        float(z["along_frac"]),
                        dict(z["meta"].tolist()) if "meta" in z else {})

    def __repr__(self):
        return (f"<Envelope q={self.q:.3f} ratio<={self.ratio_max:.2f} "
                f"cos {self.cos_thr.min():.4f}..{self.cos_thr.max():.4f} "
                f"along={self.along_frac:.3f}>")


def build_envelope(ratios, cosd, q=0.99, nbin=8, ratio_cap=None, along_frac=np.nan,
                   meta=None):
    """Pool (ratio, cosine distance) samples into a monotone Envelope."""
    ratios = np.asarray(ratios, float); cosd = np.asarray(cosd, float)
    m = np.isfinite(ratios) & np.isfinite(cosd) & (ratios >= 1.0)
    ratios, cosd = ratios[m], cosd[m]
    if not len(ratios):
        raise ValueError("no samples to build an envelope from")
    hi = float(ratio_cap if ratio_cap is not None else ratios.max())
    edges = np.unique(np.concatenate([[1.0],
                                      np.geomspace(max(ratios.min(), 1.0 + 1e-6),
                                                   max(hi, 1.0 + 1e-5), nbin)]))
    thr = np.empty(len(edges))
    for k in range(len(edges)):
        lo = edges[k]
        up = edges[k + 1] if k + 1 < len(edges) else np.inf
        sel = (ratios >= lo) & (ratios < up)
        thr[k] = float(np.quantile(cosd[sel], q)) if sel.sum() >= 5 else np.nan
    # fill empty bins forward, then enforce non-decreasing
    last = 0.0
    for k in range(len(thr)):
        if np.isnan(thr[k]):
            thr[k] = last
        last = thr[k] = max(thr[k], last)
    return Envelope(edges, thr, q, hi, along_frac, meta)
