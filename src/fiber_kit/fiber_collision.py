#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  fiber_collision.py  —  collision detection + two-template decomposition
#
#  A collision is two near-simultaneous spikes from different units summed in
#  the recording.  The stderiv transform (spatial all-pairs diff + temporal
#  first-diff) is LINEAR, so a collision is also a sum in .spkD space and in the
#  whitened masked-feature space (whitening W is linear) — decomposition is a
#  linear problem there.
#
#  Detection (validated prior session):
#    * residual to the best single fiber template = the inclusion-radius
#      quantity (AUC ~0.96).  Collisions already fail inclusion -> noise cluster.
#    * second-template gain from matching pursuit triages the noise cluster into
#      recoverable collisions vs junk (AUC 0.79 vs clean, 0.98 vs junk).
#
#  Decomposition (this module):  greedy matching pursuit over a rolled-template
#  dictionary.  Convention (KlustaKwik):
#    * u1 is FIXED at tau=0 — the spike was detected at its own peak and is the
#      reference; only u2 is circularly shifted over a small window.
#    * self-pairs excluded — one neuron cannot fire twice inside the spike window
#      (refractory), so u2's fiber != u1's fiber.
#  Returns (k1, a1, 0), (k2, a2, tau2), gain.
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
try:
    from . import fiber_lib as fl
except ImportError:
    import fiber_lib as fl


def build_templates(waves, labels, mask=fl.MASK_FULL, min_n=40):
    """Mean realigned .spkD waveform per fiber label (>=0).  Returns
    {k: template(nsamp,nch)} — the dictionary atoms in waveform space."""
    T = {}
    for k in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == k)
        if len(idx) < min_n:
            continue
        T[int(k)] = fl.realign(waves[idx]).mean(0)
    return T


def whiten_atoms(templates, W, nmean, mask=fl.MASK_FULL, shifts=range(-8, 9)):
    """Roll each template over `shifts` (sample axis) then mask+whiten, giving a
    (K, S, p) dictionary of whitened atoms + bookkeeping.  Rolling BEFORE the
    mask/whiten is required: the shift is a sample-space time shift, and the mask
    + whitener mix samples, so it cannot be applied in feature space."""
    keys = sorted(templates)
    shifts = list(shifts)
    s0 = shifts.index(0)
    K, S = len(keys), len(shifts)
    nsamp = next(iter(templates.values())).shape[0]
    M = np.zeros((K, S, len(mask) * templates[keys[0]].shape[1]))
    for ki, k in enumerate(keys):
        for si, s in enumerate(shifts):
            rolled = np.roll(templates[k], s, axis=0)
            M[ki, si] = (rolled[mask, :].reshape(-1) - nmean) @ W
    norm2 = (M ** 2).sum(-1) + 1e-12
    return dict(M=M, norm2=norm2, keys=keys, shifts=shifts, s0=s0)


def _feat(w, W, nmean, mask):
    return (w[mask, :].reshape(-1) - nmean) @ W            # w already peak-aligned


def decompose_batch(waves, dic, W, nmean, mask=fl.MASK_FULL, exclude_self=True, u1_shift=0):
    """Vectorized two-step matching pursuit over MANY already-peak-aligned spikes
    at once.  `waves` is (n, nsamp, nch) (e.g. the noise-cluster waveforms — the
    .spkD extraction is peak-aligned, and realign of a single spike is the
    identity, so no per-spike realign is needed).  Returns a dict of length-n
    arrays (k1,a1,tau1,k2,a2,tau2,gain,r1n,r2n,xn) identical to looping
    decompose() per spike; ~Nx fewer interpreter calls."""
    M, norm2, keys, shifts, s0 = dic['M'], dic['norm2'], dic['keys'], dic['shifts'], dic['s0']
    K, S, p = M.shape
    keys = np.asarray(keys)
    waves = np.asarray(waves, float)
    n = len(waves)
    X = (waves[:, mask, :].reshape(n, -1) - nmean) @ W          # (n, p) whitened features
    xn = np.sqrt((X * X).sum(1))
    u1s = np.array([si for si, s in enumerate(shifts) if abs(s) <= u1_shift] or [s0])

    # ── u1: best fiber over the small tau window ──
    Mu = M[:, u1s, :]                                            # (K, |u1s|, p)
    proj_u1 = np.einsum('ksp,np->nks', Mu, X)                    # (n, K, |u1s|)
    red0 = proj_u1 ** 2 / norm2[:, u1s][None]                   # (n, K, |u1s|)
    flat0 = red0.reshape(n, -1).argmax(1)
    k1i = flat0 // len(u1s); j1 = flat0 % len(u1s); s1 = u1s[j1]
    rows = np.arange(n)
    atom1 = M[k1i, s1]                                          # (n, p)
    nrm1 = norm2[k1i, s1]                                       # (n,)
    a1 = (atom1 * X).sum(1) / nrm1
    r1 = X - a1[:, None] * atom1
    r1n2 = (r1 * r1).sum(1)

    # ── u2: best OTHER fiber over all shifts ──
    Mf = M.reshape(K * S, p)
    proj = r1 @ Mf.T                                           # (n, K*S)
    red = (proj ** 2 / norm2.reshape(-1)[None]).reshape(n, K, S)
    if exclude_self:
        red[rows, k1i, :] = -np.inf
    flat = red.reshape(n, -1).argmax(1)
    k2i = flat // S; s2 = flat % S
    atom2 = M[k2i, s2]; nrm2 = norm2[k2i, s2]
    a2 = (atom2 * r1).sum(1) / nrm2
    r2 = r1 - a2[:, None] * atom2
    r2n2 = (r2 * r2).sum(1)
    gain = 1.0 - r2n2 / (r1n2 + 1e-12)

    shifts_arr = np.asarray(shifts)
    return dict(k1=keys[k1i], a1=a1, tau1=shifts_arr[s1],
                k2=keys[k2i], a2=a2, tau2=shifts_arr[s2], gain=gain,
                r1n=np.sqrt(r1n2), r2n=np.sqrt(r2n2), xn=xn)


def decompose(w, dic, W, nmean, mask=fl.MASK_FULL, exclude_self=True, u1_shift=0):
    """Two-step matching pursuit on one (already realigned) waveform w.
    Step 1: best fiber for u1 (searches +/- u1_shift samples, since realign of a
    collision pulls u1 off tau=0 in ~half of cases; u1_shift=0 keeps the strict
    convention).  Step 2: best OTHER fiber over all shifts (u2).  Returns
    dict(k1,a1,tau1,k2,a2,tau2,gain,r1,r2)."""
    M, norm2, keys, shifts, s0 = dic['M'], dic['norm2'], dic['keys'], dic['shifts'], dic['s0']
    x = _feat(w, W, nmean, mask)
    # ── u1: search a small shift window around tau=0 ──
    u1s = [si for si, s in enumerate(shifts) if abs(s) <= u1_shift] or [s0]
    red0 = (M[:, u1s, :] @ x) ** 2 / norm2[:, u1s]         # (K, |u1s|)
    k1i, j1 = np.unravel_index(int(np.argmax(red0)), red0.shape)
    s1 = u1s[j1]
    a1 = (M[k1i, s1] @ x) / norm2[k1i, s1]
    r1 = x - a1 * M[k1i, s1]
    r1n2 = float(r1 @ r1)
    # ── u2: any other fiber, any shift ──
    proj = M.reshape(len(keys) * len(shifts), -1) @ r1
    red = (proj ** 2 / norm2.reshape(-1)).reshape(len(keys), len(shifts))
    if exclude_self:
        red[k1i, :] = -np.inf
    flat = int(np.argmax(red)); k2i, s2 = divmod(flat, len(shifts))
    a2 = (M[k2i, s2] @ r1) / norm2[k2i, s2]
    r2 = r1 - a2 * M[k2i, s2]
    r2n2 = float(r2 @ r2)
    gain = 1.0 - r2n2 / (r1n2 + 1e-12)
    return dict(k1=keys[k1i], a1=float(a1), tau1=int(shifts[s1]),
                k2=keys[k2i], a2=float(a2), tau2=int(shifts[s2]), gain=float(gain),
                r1n=float(np.sqrt(r1n2)), r2n=float(np.sqrt(r2n2)),
                xn=float(np.sqrt(x @ x)))


def decompose_pinned(w, dic, W, nmean, u1_key, u1_energy, mask=fl.MASK_FULL, u1_shift=1):
    """Decompose with u1 KNOWN (the fiber the spike was sorted to) and its
    amplitude PINNED so the removed u1 component has whitened energy `u1_energy`
    — supply g(a) from the unit's adaptation history.  This stops the free fit
    from absorbing u2's energy along u1's (near-collinear) direction.  Then find
    u2 over the other fibers.  Returns dict(k1,a1,tau1,k2,a2,tau2,gain,...)."""
    M, norm2, keys, shifts = dic['M'], dic['norm2'], dic['keys'], dic['shifts']
    x = _feat(w, W, nmean, mask)
    ki = keys.index(u1_key)
    u1s = [si for si, s in enumerate(shifts) if abs(s) <= u1_shift] or [shifts.index(0)]
    proj = M[ki, u1s, :] @ x
    j = int(np.argmax(np.abs(proj))); s1 = u1s[j]
    a1 = float(np.sign(proj[j])) * u1_energy / np.sqrt(norm2[ki, s1])   # pin |a1|*||atom|| = energy
    r1 = x - a1 * M[ki, s1]; r1n2 = float(r1 @ r1)
    P = M.reshape(len(keys) * len(shifts), -1) @ r1
    red = (P ** 2 / norm2.reshape(-1)).reshape(len(keys), len(shifts))
    red[ki, :] = -np.inf
    flat = int(np.argmax(red)); k2i, s2 = divmod(flat, len(shifts))
    a2 = (M[k2i, s2] @ r1) / norm2[k2i, s2]
    r2 = r1 - a2 * M[k2i, s2]; r2n2 = float(r2 @ r2)
    return dict(k1=u1_key, a1=float(a1), tau1=int(shifts[s1]),
                k2=keys[k2i], a2=float(a2), tau2=int(shifts[s2]),
                gain=float(1.0 - r2n2 / (r1n2 + 1e-12)),
                r1n=float(np.sqrt(r1n2)), r2n=float(np.sqrt(r2n2)))


def detect_residual(w, dic, W, nmean, mask=fl.MASK_FULL):
    """Whitened residual to the single best fiber (tau=0) — the inclusion-radius
    collision detector.  Large => not explained by one fiber."""
    M, norm2, s0 = dic['M'], dic['norm2'], dic['s0']
    x = _feat(w, W, nmean, mask)
    proj0 = M[:, s0, :] @ x
    k1i = int(np.argmax(proj0 ** 2 / norm2[:, s0]))
    a1 = proj0[k1i] / norm2[k1i, s0]
    r = x - a1 * M[k1i, s0]
    return float(np.sqrt(r @ r))
