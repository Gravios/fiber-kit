#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  fiber_deconv.py  —  detection-anchored sub-sample template deconvolution
#
#  fiber_collision decomposes a collision inside ONE 42-sample snippet, against
#  whole-sample rolled templates, in the snippet whitener's space.  Its
#  decomposition to a specific fiber pair is separability-bound (~15-19 %) and
#  kept advisory.  Three things limit it, and this module removes each:
#
#    * the snippet truncates the partner -- a spike 10 samples off the
#      detection loses a quarter of its support at the window edge.  Here the
#      pursuit runs on the CONTINUOUS signal over a segment that covers every
#      detection and its neighbours, so each component is fitted whole;
#    * whole-sample shifts leave up to half a sample of misfit, which at CA1
#      SNR is the same size as the difference between look-alike units.  Each
#      template is interpolated at R sub-sample phases (Wobble / DARTsort);
#    * a free amplitude lets a template absorb a neighbour's energy along a
#      near-collinear direction (the reason decompose_pinned exists).  Each
#      unit's amplitude carries a Gaussian prior centred on 1 with the unit's
#      own measured spread, and is bounded.
#
#  Noise model.  The pursuit is a maximum-likelihood fit under a Gaussian noise
#  model, so inner products must be taken after whitening the continuous
#  signal.  The snippet whitener (fiber_lib) mixes samples inside a fixed
#  window and cannot be convolved.  Instead a vector autoregression of order p
#  is fitted to off-spike stretches of the transformed signal:
#        e_t = x_t - sum_i A_i x_{t-i},   w_t = Sigma_e^{-1/2} e_t
#  This is a causal, time-invariant filter: applied to the signal and to every
#  template alike, a sum of placed templates plus noise stays a sum, and the
#  noise becomes white in time and across channels.
#
#  Gain of placing atom k at offset s with amplitude a (whitened units):
#        G = 2 a c - a^2 n - lam_k (a - 1)^2,   c = <y_s, A_k>,  n = |A_k|^2,
#        a* = (c + lam_k) / (n + lam_k),   lam_k = 1 / sd_a(k)^2
#  which is twice the log-likelihood gain (Wobble's amplitude-scaled
#  objective).  Placements are greedy, largest gain first; after each one the
#  amplitudes of all placements in the segment are refitted jointly (the
#  orthogonal matching pursuit step: otherwise the first of two overlapping
#  spikes absorbs part of its partner) and the scores are corrected through the
#  precomputed atom cross-correlations, so nothing is re-convolved.  A unit
#  cannot be placed twice within its refractory window.  The pursuit stops when
#  the best gain falls below a threshold calibrated on off-spike noise windows
#  of the same chunk (the largest first-step gain pure noise produces).
#
#  Over-split input.  fiber-session's fibers are over-split by design, so
#  several carry one cell's template.  Agreement, the label margin and the
#  collision partner are judged up to a pairwise template EQUIVALENCE
#  (template_similarity): naming a sibling fragment is not a disagreement.
#
#  Detection anchoring.  Only segments around detected spikes are pursued:
#  every collision has a detected component, so this covers collisions at a
#  fraction of the cost of the whole record, and it keeps fiber-session's
#  output aligned to the .res it was given.  Placements that match no
#  detection are reported (unanchored) but never written into a .clu.
#
#  Domain.  Templates are estimated from the same transformed continuous
#  signal the pursuit runs on (allpairs stderiv of the .fil by default:
#  fiber_lib.fil_to_spkD_space, the space the chunk whitener uses), so they are
#  consistent with it whatever variant the .spk was extracted in.
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
from collections import namedtuple
from sklearn.covariance import LedoitWolf

try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl


NoiseModel = namedtuple("NoiseModel", "A W order")
Bank = namedtuple("Bank", "atoms unit frac norm2 lam amin amax keys lead G order nsamp peak R")


# ── transformed continuous signal ───────────────────────────────────────────
def memmap_reader(mm, channels):
    """read_raw(a, b) over a (nSamples, nTotal) memmap restricted to `channels`.
    Rows outside the recording read as zeros, so every request returns exactly
    b - a rows (segments at the record edges keep their geometry)."""
    ch = np.asarray(channels, np.int64)
    N = mm.shape[0]
    contiguous = ch.size > 0 and bool((np.diff(ch) == 1).all())
    lo, hi = int(ch[0]), int(ch[-1]) + 1

    def read_raw(a, b):
        a, b = int(a), int(b)
        out = np.zeros((max(0, b - a), ch.size), np.float64)
        a0, b0 = max(0, a), min(N, b)
        if b0 > a0:
            blk = mm[a0:b0, lo:hi] if contiguous else mm[a0:b0][:, ch]
            out[a0 - a:b0 - a] = blk
        return out
    return read_raw



def read_transformed(read_raw, a, b, transform=None):
    """Transformed signal over [a, b) as (b-a, C) float64.  Reads one extra raw
    sample before `a` so the temporal first difference at `a` is exact (the
    transform zeroes its first row)."""
    tf = transform or fl.fil_to_spkD_space
    a0 = max(0, int(a) - 1)
    raw = np.asarray(read_raw(a0, int(b)), np.float64)
    x = tf(raw)
    return x[(int(a) - a0):]


# ── noise model: VAR(p) + spatial whitening ─────────────────────────────────
def _off_spike_starts(s0, s1, spike_abs, seg_len, guard, n, rng):
    sp = np.sort(np.asarray(spike_abs, np.int64))
    out = []; tries = 0
    hi = int(s1) - seg_len
    if hi <= int(s0):
        return out
    while len(out) < n and tries < 60 * n:
        tries += 1
        st = int(rng.integers(int(s0), hi))
        i = np.searchsorted(sp, st - guard, "left")
        j = np.searchsorted(sp, st + seg_len + guard, "right")
        if j == i:
            out.append(st)
    return out


def fit_noise_model(read_raw, s0, s1, spike_abs, *, order=4, n_seg=300, seg_len=192,
                    guard=48, seed=0, transform=None):
    """Fit e_t = x_t - sum_i A_i x_{t-i} and W = Sigma_e^{-1/2} on off-spike
    stretches of the transformed signal inside [s0, s1).  Returns NoiseModel."""
    rng = np.random.default_rng(seed)
    starts = _off_spike_starts(s0, s1, spike_abs, seg_len, guard, n_seg, rng)
    if len(starts) < 20:
        raise RuntimeError(f"fit_noise_model: only {len(starts)} off-spike stretches in "
                           f"[{s0}, {s1}) (need 20)")
    p = int(order)
    Y, X = [], []
    for st in starts:
        x = read_transformed(read_raw, st, st + seg_len, transform)
        x = x - x.mean(0)
        Y.append(x[p:])
        X.append(np.hstack([x[p - i:len(x) - i] for i in range(1, p + 1)]) if p else
                 np.zeros((len(x), 0)))
    Y = np.vstack(Y); X = np.vstack(X)
    C = Y.shape[1]
    if p:
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)          # (p*C, C)
        A = np.stack([coef[i * C:(i + 1) * C].T for i in range(p)])   # A_i: (C, C), x_{t-i} @ A_i.T
        E = Y - X @ coef
    else:
        A = np.zeros((0, C, C)); E = Y
    # The allpairs transform sums to zero across channels, so the residual has
    # rank C-1: whiten the signal subspace and zero the null direction (a
    # floored eigenvalue would amplify rounding noise there instead).  Signal
    # and templates have no component along it, so nothing is lost.
    S = np.cov(E.T) if E.shape[0] > 50 * C else LedoitWolf().fit(E).covariance_
    ev, V = np.linalg.eigh(S)
    inv = np.where(ev > 1e-6 * ev.max(), 1.0 / np.sqrt(np.maximum(ev, 1e-300)), 0.0)
    W = V @ np.diag(inv) @ V.T
    return NoiseModel(A=A, W=W, order=p)


def whiten(x, nm):
    """Apply the noise model's filter to x (T, C).  Samples before x[0] are
    taken as zero, so the result has the same length T (use this for templates
    and for signal blocks that carry `order` samples of history; drop the first
    `order` rows of a signal block)."""
    x = np.asarray(x, np.float64)
    e = x.copy()
    for i in range(1, nm.order + 1):
        e[i:] -= x[:-i] @ nm.A[i - 1].T
    return e @ nm.W.T


# ── templates ───────────────────────────────────────────────────────────────
def _align_to_median(S, maxlag=3, iters=3):
    """Integer-align snippets (n, L, C) to their median by cross-correlation on
    the full multichannel window.  Returns aligned copy (edges zero-filled)."""
    S = np.asarray(S, np.float64)
    out = S.copy()
    for _ in range(iters):
        ref = np.median(out, 0)
        best = np.zeros(len(S), int); bv = np.full(len(S), -np.inf)
        for d in range(-maxlag, maxlag + 1):
            sh = np.roll(S, d, axis=1)
            v = np.einsum("ntc,tc->n", sh, ref)
            m = v > bv
            best[m] = d; bv[m] = v[m]
        new = np.empty_like(S)
        for d in np.unique(best):
            ix = best == d
            r = np.roll(S[ix], d, axis=1)
            if d > 0:
                r[:, :d] = 0
            elif d < 0:
                r[:, d:] = 0
            new[ix] = r
        if np.allclose(new, out):
            break
        out = new
    return out


def unit_templates(read_raw, res_abs, labels, nsamp, peak, *, min_n=30, max_n=400,
                   seed=0, transform=None, exclude=()):
    """Mean transformed-continuous-signal template per label >= 0 (skipping
    `exclude`).  Snippets are [res - peak, res - peak + nsamp), aligned to the
    unit median.  Returns {label: (nsamp, C)}."""
    rng = np.random.default_rng(seed)
    res_abs = np.asarray(res_abs, np.int64); labels = np.asarray(labels)
    T = {}
    for k in np.unique(labels[labels >= 0]):
        if int(k) in exclude:
            continue
        idx = np.flatnonzero(labels == k)
        if len(idx) < min_n:
            continue
        if len(idx) > max_n:
            idx = np.sort(rng.choice(idx, max_n, replace=False))
        S = np.stack([read_transformed(read_raw, r - peak, r - peak + nsamp, transform)
                      for r in res_abs[idx]])
        T[int(k)] = _align_to_median(S).mean(0)
    return T


def _frac_shift(buf, frac):
    """Delay buf (L, C) by `frac` samples by Fourier interpolation (circular;
    the caller pads with zeros so nothing wraps)."""
    if frac == 0:
        return buf.copy()
    L = buf.shape[0]
    f = np.fft.rfftfreq(L)
    ph = np.exp(-2j * np.pi * f * frac)[:, None]
    return np.fft.irfft(np.fft.rfft(buf, axis=0) * ph, n=L, axis=0)


def _taper(nsamp, edge=3):
    w = np.ones(nsamp)
    if edge > 0 and nsamp > 2 * edge:
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge + 2)[1:-1])
        w[:edge] = r; w[-edge:] = r[::-1]
    return w


def build_bank(templates, nm, nsamp, peak, *, R=4, amp_sd=None, amp_sd_floor=0.08,
               amp_bounds=(0.25, 3.0)):
    """Whitened atom dictionary: every template at R sub-sample phases.

    Raw template sample 0 sits at buffer index `lead` = order + 1, the buffer is
    nsamp + 2*order + 2 long (order samples of zero history in front for the
    filter, order samples behind for the filter's tail, one each side for the
    phase shift).  An atom placed at segment offset s therefore puts the
    template's detection sample (index `peak`) at  s + lead + peak + frac.

    amp_sd: {label: robust sd of the per-spike amplitude} (default 0.25 each).
    Returns Bank.  G is a lazy cache of atom cross-correlations: gcol(bank, j)
    gives G[:, j, d + La - 1] = sum_u <A_i[u], A_j[u + d]> for every atom i, and
    is computed only for atoms the pursuit actually places (a full KR x KR x
    (2La - 1) table is ~170 MB at 115 fibers x 4 phases)."""
    keys = sorted(templates)
    if not keys:
        raise ValueError("build_bank: no templates")
    p = nm.order
    lead = p + 1
    La = nsamp + 2 * p + 2
    tap = _taper(nsamp)[:, None]
    atoms, unit, frac = [], [], []
    for k in keys:
        buf = np.zeros((La, templates[k].shape[1]))
        buf[lead:lead + nsamp] = templates[k] * tap
        for j in range(R):
            atoms.append(whiten(_frac_shift(buf, j / float(R)), nm))
            unit.append(k); frac.append(j / float(R))
    atoms = np.stack(atoms)                                    # (KR, La, C)
    norm2 = (atoms ** 2).sum((1, 2))
    sd = {k: max(amp_sd_floor, float((amp_sd or {}).get(k, 0.25))) for k in keys}
    lam = np.array([1.0 / sd[u] ** 2 for u in unit])
    amin = np.array([max(amp_bounds[0], 1.0 - 5.0 * sd[u]) for u in unit])
    amax = np.array([min(amp_bounds[1], 1.0 + 5.0 * sd[u]) for u in unit])
    return Bank(atoms=atoms, unit=np.asarray(unit), frac=np.asarray(frac), norm2=norm2,
                lam=lam, amin=amin, amax=amax, keys=keys, lead=lead, G={}, order=p,
                nsamp=nsamp, peak=peak, R=R)


def gcol(bank, j):
    """(KR, 2La-1) cross-correlations of every atom with atom j (cached)."""
    col = bank.G.get(j)
    if col is None:
        A = bank.atoms
        La = A.shape[1]
        col = np.empty((len(A), 2 * La - 1))
        for d in range(-(La - 1), La):
            lo, hi = max(0, -d), La - max(0, d)
            col[:, d + La - 1] = np.tensordot(A[:, lo:hi], A[j, lo + d:hi + d], axes=([1, 2], [0, 1]))
        bank.G[j] = col
    return col


def amplitude_spread(read_raw, res_abs, labels, bank, nm, *, max_n=300, seed=0,
                     transform=None):
    """Robust sd of each unit's per-spike amplitude: project each spike's
    whitened window on its unit's phase-0 atom (a = <x, A>/|A|^2).  The prior
    width the pursuit uses.  Returns {label: sd}."""
    rng = np.random.default_rng(seed)
    out = {}
    p, lead, La = bank.order, bank.lead, bank.atoms.shape[1]
    for k in bank.keys:
        i0 = int(np.flatnonzero((bank.unit == k) & (bank.frac == 0))[0])
        idx = np.flatnonzero(np.asarray(labels) == k)
        if len(idx) < 10:
            continue
        if len(idx) > max_n:
            idx = rng.choice(idx, max_n, replace=False)
        A0 = bank.atoms[i0].ravel()
        a = []
        for r in np.asarray(res_abs)[idx]:
            s = int(r) - bank.peak - lead                      # atom start in absolute samples
            x = read_transformed(read_raw, s - 3 - p, s + La + 3, transform)
            y = whiten(x, nm)[p:]                              # offsets s-3 .. s+3
            win = np.lib.stride_tricks.sliding_window_view(y, (La, y.shape[1]))[:, 0]
            c = win.reshape(len(win), -1) @ A0                 # detection jitter: best of +/-3
            a.append(float(c.max()) / bank.norm2[i0])
        a = np.asarray(a)
        med = np.median(a)
        if med > 0:
            out[k] = float(1.4826 * np.median(np.abs(a - med)) / med)
    return out


def template_similarity(templates, nm, counts, maxlag=4):
    """Pairwise similarity of templates: the whitened cosine, maximised over
    +/-maxlag samples and corrected for template noise.  Returns (keys, S).

    Why not families.  fiber-session is over-split by design, so several
    fragments carry one cell's template and the pursuit may name any of them.
    Grouping them transitively (linkage) fails both ways on real over-splits:
    a fragment that is mostly one cell but carries collisions or a sub-sample
    offset sits far from its siblings and splits the family, while look-alike
    cells chain together.  A PAIRWISE equivalence (S >= cut) asks only the
    question that matters for each decision -- is the template the pursuit
    chose distinguishable from this one?

    Corrections.  A template averaged over n spikes carries whitened noise of
    variance 1/n per dimension (rank(W) per sample), inflating |T|^2 by d/n and
    deflating the cosine between two fragments of a small unit; the norms are
    corrected (the cross term has no noise bias).  Fragments can be centred a
    few samples apart (nudge / realign splits), hence the lag maximum."""
    keys = sorted(templates)
    n = len(keys)
    pad = int(maxlag)
    V = []
    for k in keys:
        t = templates[k]
        V.append(whiten(np.vstack([np.zeros((pad, t.shape[1])), t, np.zeros((pad, t.shape[1]))]), nm))
    V = np.stack(V)                                               # (n, L, C)
    d = templates[keys[0]].shape[0] * int(np.linalg.matrix_rank(nm.W))
    cnt = np.maximum(np.array([float(counts.get(k, 1)) for k in keys]), 1.0)
    e2 = (V * V).sum((1, 2))
    nrm = np.sqrt(np.maximum(e2 - d / cnt, 1e-6 * e2))
    S = np.full((n, n), -np.inf)
    L = V.shape[1]
    for lag in range(-pad, pad + 1):
        lo, hi = max(0, lag), L + min(0, lag)
        X = np.tensordot(V[:, lo:hi], V[:, lo - lag:hi - lag], axes=([1, 2], [1, 2]))
        S = np.maximum(S, X / np.outer(nrm, nrm))
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 1.0)
    return keys, S


# ── pursuit ─────────────────────────────────────────────────────────────────
def _gains(c, bank):
    n = bank.norm2[:, None]; lam = bank.lam[:, None]
    a = np.clip((c + lam) / (n + lam), bank.amin[:, None], bank.amax[:, None])
    return 2 * a * c - a * a * n - lam * (a - 1) ** 2, a


def _scores(y, bank):
    La = bank.atoms.shape[1]
    S = y.shape[0] - La + 1
    if S <= 0:
        return np.zeros((len(bank.atoms), 0))
    win = np.lib.stride_tricks.sliding_window_view(y, (La, y.shape[1]))[:, 0]   # (S, La, C)
    return bank.atoms.reshape(len(bank.atoms), -1) @ win.reshape(S, -1).T          # (KR, S)


def _refit(c0, bank, at, st):
    """Joint amplitudes of the placed atoms (orthogonal matching pursuit step):
    minimise |y - sum a_k A_k|^2 + sum lam_k (a_k - 1)^2 over the placements
    (at[k], st[k]); the Gram matrix comes from the precomputed G.  Without it
    the first of two overlapping spikes absorbs part of its partner."""
    La = bank.atoms.shape[1]
    k = len(at)
    M = np.zeros((k, k))
    for p_ in range(k):
        for q in range(k):
            d = st[p_] - st[q]
            if abs(d) < La:
                M[p_, q] = gcol(bank, at[q])[at[p_], d + La - 1]
    lam = bank.lam[at]
    b = c0[at, st] + lam
    a = np.linalg.solve(M + np.diag(lam), b)
    return np.clip(a, bank.amin[at], bank.amax[at])


def pursue(y, bank, thr, refrac, max_iter=64):
    """Greedy matching pursuit on one whitened segment y (L, C), with a joint
    amplitude refit after every placement.  Returns (placements, residual) with
    placements a list of (atom, offset, amp, gain); gain is the gain at the
    moment of selection, amp the final jointly refitted amplitude."""
    La = bank.atoms.shape[1]
    c0 = _scores(y, bank)
    if c0.shape[1] == 0:
        return [], y.copy()
    KR, S = c0.shape
    blocked = np.zeros((KR, S), bool)
    c = c0.copy()
    at, st, gains = [], [], []
    amps = np.zeros(0)
    for _ in range(max_iter):
        g, a = _gains(c, bank)
        g[blocked] = -np.inf
        flat = int(np.argmax(g))
        i, s = divmod(flat, S)
        if not np.isfinite(g[i, s]) or g[i, s] < thr:
            break
        at.append(i); st.append(s); gains.append(float(g[i, s]))
        amps = _refit(c0, bank, at, st)
        c = c0.copy()                       # c[j, t] = c0 - sum_k a_k <A_j at t, A_k at s_k>
        for ik, sk, ak in zip(at, st, amps):
            lo, hi = max(0, sk - La + 1), min(S, sk + La)
            c[:, lo:hi] -= ak * gcol(bank, ik)[:, (lo - sk + La - 1):(hi - sk + La - 1)]
        blocked[bank.unit == bank.unit[i], max(0, s - refrac):min(S, s + refrac + 1)] = True
    r = y.copy()
    for ik, sk, ak in zip(at, st, amps):
        r[sk:sk + La] -= ak * bank.atoms[ik]
    return [(ik, sk, float(ak), gk) for ik, sk, ak, gk in zip(at, st, amps, gains)], r


def null_threshold(read_raw, s0, s1, spike_abs, bank, nm, *, n=400, seg_len=None,
                   q=0.995, floor=16.0, guard=48, seed=1, transform=None):
    """Largest first-step gain on off-spike windows of the chunk: the pursuit's
    stopping threshold.  Returns (thr, null_gains)."""
    rng = np.random.default_rng(seed)
    La = bank.atoms.shape[1]
    L = int(seg_len or (2 * La + bank.nsamp))
    starts = _off_spike_starts(s0, s1, spike_abs, L + bank.order, guard, n, rng)
    gm = []
    for st in starts:
        x = read_transformed(read_raw, st, st + L + bank.order, transform)
        y = whiten(x, nm)[bank.order:]
        g, _ = _gains(_scores(y, bank), bank)
        gm.append(float(g.max()) if g.size else 0.0)
    gm = np.asarray(gm)
    thr = max(float(floor), float(np.quantile(gm, q))) if gm.size else float(floor)
    return thr, gm


def segments(res_abs, nsamp, peak, pad, lead, La):
    """Union of [r - peak - pad - lead, r - peak + nsamp + pad + La) over
    detections, as sorted (a, b, first_index, last_index_exclusive) over the
    sorted detections."""
    r = np.asarray(res_abs, np.int64)
    order = np.argsort(r, kind="stable")
    rs = r[order]
    lo = rs - peak - pad - lead
    hi = rs - peak + nsamp + pad + La
    segs = []
    a, b, i0 = int(lo[0]), int(hi[0]), 0
    for i in range(1, len(rs)):
        if lo[i] <= b:
            b = max(b, int(hi[i]))
        else:
            segs.append((a, b, i0, i)); a, b, i0 = int(lo[i]), int(hi[i]), i
    segs.append((a, b, i0, len(rs)))
    return segs, order


Result = namedtuple("Result", "primary amp dt gain margin resid partner partner_dt "
                              "placements thr keys equiv")


def _unit_margin(r, s, i, amp, bank, tol, same=None):
    """Confidence of placement (atom i, offset s, amplitude amp): put the
    component back into the residual r and refit every atom alone within
    +/-tol samples of s.  Returns the best gain of i's unit (or any unit
    equivalent to it: `same`, a set of labels) minus the best gain of any
    NON-equivalent unit -- twice the log-likelihood ratio between the two
    distinguishable explanations of this one event."""
    La = bank.atoms.shape[1]
    lo = max(0, s - tol); hi = min(r.shape[0] - La + 1, s + tol + 1)
    y = r[lo:hi + La - 1].copy()
    y[s - lo:s - lo + La] += amp * bank.atoms[i]
    g, _ = _gains(_scores(y, bank), bank)
    best = g.max(1)
    u = bank.unit[i]
    eq = np.isin(bank.unit, list(same)) if same else (bank.unit == u)
    own = best[eq].max()
    oth = best[~eq]
    return float(own - (oth.max() if oth.size else -np.inf))


def deconvolve(read_raw, res_abs, labels, nsamp, peak, sr, *, s0=None, s1=None, R=4,
               order=4, refrac_ms=1.0, pad=None, tol=4, null_q=0.995, thr_floor=16.0,
               min_n=30, exclude=(), transform=None, seed=0, nm=None, bank=None,
               equiv_cos=0.93):
    """Deconvolve the detections `res_abs` (absolute samples) of one chunk.

    labels : clustering of those detections (local ids, -1 = noise); units with
             >= min_n spikes and not in `exclude` provide templates.
    equiv_cos: templates at lag-tolerant, noise-corrected whitened cosine >= this
             are EQUIVALENT (template_similarity): indistinguishable to the pursuit,
             so naming one for the other is not a disagreement, and the margin is
             against the best non-equivalent template.  >1 = only identity.
    Returns Result with per-detection arrays (aligned to res_abs):
      primary  label of the placement anchored on the detection, or -1
      amp/dt/gain  its amplitude, time correction (samples, float), gain
      margin   its gain minus the best gain of any other unit refitted alone on
               the same event (2 x log-likelihood ratio; confidence of the label)
      resid    whitened residual energy per sample-channel in the detection's
               window after the whole segment's pursuit (~1 for a clean fit)
      partner/partner_dt  the strongest placement of ANOTHER unit within half a
               window that is not itself anchored on a detection (the collision
               partner), or -1
    and `placements` (absolute peak time, unit, amp, gain, anchored_index or -1).
    """
    res_abs = np.asarray(res_abs, np.int64)
    labels = np.asarray(labels)
    if s0 is None:
        s0 = int(res_abs.min()) - 4 * nsamp
    if s1 is None:
        s1 = int(res_abs.max()) + 4 * nsamp
    if nm is None:
        nm = fit_noise_model(read_raw, s0, s1, res_abs, order=order, seed=seed, transform=transform)
    if bank is None:
        T = unit_templates(read_raw, res_abs, labels, nsamp, peak, min_n=min_n, seed=seed,
                           transform=transform, exclude=exclude)
        if not T:
            raise ValueError("deconvolve: no unit with enough spikes for a template")
        bank0 = build_bank(T, nm, nsamp, peak, R=1)
        sd = amplitude_spread(read_raw, res_abs, labels, bank0, nm, seed=seed, transform=transform)
        bank = build_bank(T, nm, nsamp, peak, R=R, amp_sd=sd)
        counts = {k: int((labels == k).sum()) for k in T}
        keys, S = template_similarity(T, nm, counts)
        equiv = {k: {keys[j] for j in np.flatnonzero(S[i] >= equiv_cos)} for i, k in enumerate(keys)}
    else:
        equiv = {k: {k} for k in bank.keys}
    thr, _ = null_threshold(read_raw, s0, s1, res_abs, bank, nm, q=null_q, floor=thr_floor,
                            seed=seed + 1, transform=transform)
    pad = int(nsamp // 2 if pad is None else pad)
    refr = fl_refrac(refrac_ms, sr)
    La = bank.atoms.shape[1]
    lead, p = bank.lead, bank.order
    n = len(res_abs)
    primary = np.full(n, -1)
    amp = np.full(n, np.nan); dt = np.full(n, np.nan)
    gain = np.full(n, np.nan); resid = np.full(n, np.nan); margin = np.full(n, np.nan)
    partner = np.full(n, -1); partner_dt = np.full(n, np.nan)
    rows = []
    segs, order_ix = segments(res_abs, nsamp, peak, pad, lead, La)
    for a, b, i0, i1 in segs:
        x = read_transformed(read_raw, a - p, b, transform)
        y = whiten(x, nm)[p:]
        det = order_ix[i0:i1]
        pl, rsd = pursue(y, bank, thr, refr, max_iter=4 * len(det) + 4)
        if not pl:
            for d in det:
                w0 = int(res_abs[d]) - peak - a
                seg = rsd[max(0, w0):w0 + nsamp]
                resid[d] = float((seg ** 2).mean()) if seg.size else np.nan
            continue
        t_peak = np.array([a + s + lead + peak + bank.frac[i] for i, s, _, _ in pl])
        u_pl = np.array([bank.unit[i] for i, _, _, _ in pl])
        g_pl = np.array([g for _, _, _, g in pl]); a_pl = np.array([am for _, _, am, _ in pl])
        anchored = np.full(len(pl), -1)
        # anchor: each detection takes its highest-gain placement within tol (greedy by gain)
        for j in np.argsort(-g_pl):
            dd = det[(np.abs(res_abs[det] - t_peak[j]) <= tol) & (primary[det] < 0)]
            if dd.size:
                d = int(dd[np.argmin(np.abs(res_abs[dd] - t_peak[j]))])
                primary[d] = u_pl[j]; amp[d] = a_pl[j]; gain[d] = g_pl[j]
                dt[d] = t_peak[j] - res_abs[d]; anchored[j] = d
                ia, sa, am, _ = pl[j]
                margin[d] = _unit_margin(rsd, sa, ia, am, bank, tol, equiv[u_pl[j]])
        free = np.flatnonzero(anchored < 0)
        for d in det:
            w0 = int(res_abs[d]) - peak - a
            seg = rsd[max(0, w0):w0 + nsamp]
            resid[d] = float((seg ** 2).mean()) if seg.size else np.nan
            if free.size:
                own = equiv.get(int(primary[d]), set()) if primary[d] >= 0 else set()
                near = free[(np.abs(t_peak[free] - res_abs[d]) <= nsamp / 2.0)
                            & ~np.isin(u_pl[free], list(own))]
                if near.size:
                    j = near[np.argmax(g_pl[near])]
                    partner[d] = u_pl[j]; partner_dt[d] = t_peak[j] - res_abs[d]
        for j in range(len(pl)):
            rows.append((float(t_peak[j]), int(u_pl[j]), float(a_pl[j]), float(g_pl[j]),
                         int(anchored[j])))
    plc = np.array(rows, dtype=[("t", "f8"), ("unit", "i8"), ("amp", "f8"), ("gain", "f8"),
                                ("anchor", "i8")])
    return Result(primary=primary, amp=amp, dt=dt, gain=gain, margin=margin,
                  resid=resid, partner=partner, partner_dt=partner_dt, placements=plc, thr=thr,
                  keys=list(bank.keys), equiv=equiv)


def fl_refrac(refrac_ms, sr):
    return int(round(float(refrac_ms) * float(sr) / 1000.0))


def calibrate_margin(margin, agree, target=0.999, floor=10.0, min_n=50, slack=0.002):
    """Smallest margin m >= floor at which the deconvolved label agrees with
    the clustering as well as it ever does.

    Among the CLEAN detections (clustered and placed), agreement rises with
    the margin and then plateaus.  The plateau is not 1: the clustering is not
    ground truth, and where the pursuit disagrees with it at high margin it is
    usually the clustering that is wrong (deconv_agree tracks purity).  So the
    required agreement is min(target, plateau - slack), the plateau being the
    agreement over the more confident half; m is the lowest margin whose top
    set reaches it on >= min_n spikes.  Returns inf when none does (then
    nothing is reassigned)."""
    m = np.asarray(margin, float); ok = np.asarray(agree, bool)
    keep = np.isfinite(m)
    m, ok = m[keep], ok[keep]
    if m.size < min_n:
        return float("inf")
    o = np.argsort(-m)
    m, ok = m[o], ok[o]
    plateau = float(ok[:max(min_n, m.size // 2)].mean())
    need = min(float(target), plateau - float(slack))
    frac = np.cumsum(ok) / np.arange(1, m.size + 1)       # agreement among the top-i by margin
    good = (frac >= need) & (np.arange(1, m.size + 1) >= min_n) & (m >= floor)
    if not good.any():
        return float("inf")
    return float(m[np.flatnonzero(good).max()])
