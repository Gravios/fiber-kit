"""tools/fil_noise_states.py on synthetic sessions with known noise states.

  * TWO-STATE file (120 s, 20 kHz, 8-channel group of a 10-channel file): band-limited noise alternating
    every 8-20 s between a quiet state and an active one (x1.6 level, stronger common mode, sub-threshold
    'MUA' bumps, noise amplitude modulated by an 8-Hz theta that the LFP carries), detected spikes on top
    (masked by the tool), and ripple events in the quiet state (LFP 180-Hz bursts, .fil noise x2 for +-30
    ms).  The tool must find: most of the level's variation real; two states with the right occupancy,
    level ratio, common share and tail ordering and dwell times (windows that straddle a switch may form a
    small extra state, which must be marked transient); theta-phase modulation of the noise; the noise rise
    at ripples; the whitener mismatch between the states.
  * STATIONARY control (same quiet noise throughout, no theta, no ripples): no real structure, one state,
    no phase modulation.
  * the stderiv_C5 space (sdiffPairs from the yaml) runs and still separates the two states.
"""
import os
import sys
import tempfile

import numpy as np
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import fil_noise_states as fns    # noqa: E402

fails = 0
SR, LFP_SR, NTOT, DUR = 20000, 1250, 10, 120
PAIRS = "0-2+4,1-3+5,2-4+0+6,3-5+1+7,4-6+2+7,5-7+3+6,6-7+3+5,7-5+6"


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def make_session(d, name, two_state, rng):
    T = DUR * SR
    t = np.arange(T) / SR
    # state sequence: alternating segments of 8-20 s, quiet first
    state = np.zeros(T, np.int8)
    edges = [0]
    while edges[-1] < T:
        edges.append(edges[-1] + int(rng.uniform(8, 20) * SR))
    if two_state:
        for i in range(1, len(edges) - 1, 2):
            state[edges[i]:min(edges[i + 1], T)] = 1
    sos = butter(3, (300, 5000), "band", fs=SR, output="sos")
    ind = sosfilt(sos, rng.normal(0, 1, (T, 8)), axis=0); ind /= ind.std(0)
    cm = sosfilt(sos, rng.normal(0, 1, T)); cm /= cm.std()
    s_ind = np.where(state == 1, 40.0, 30.0); s_cm = np.where(state == 1, 35.0, 12.0)
    phi = 2 * np.pi * 8.0 * t + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    mod = np.where(state == 1, 1.0 + 0.3 * np.cos(phi), 1.0)
    x = (s_ind[:, None] * ind + (s_cm * cm)[:, None]) * mod[:, None]
    # sub-threshold bumps in the active state
    if two_state:
        act = np.flatnonzero(state == 1)
        nb_ = int(500 * act.size / SR)
        at = rng.choice(act[(act > 20) & (act < T - 20)], nb_, replace=False)
        bump = -np.diff(np.exp(-0.5 * (np.arange(-8, 9) / 2.5) ** 2), prepend=0) * 60 / 0.25
        ch = rng.integers(0, 8, nb_)
        for k, o in enumerate(range(-8, 9)):
            np.add.at(x, (at + o, ch), bump[k])
    # ripples in the quiet state: .fil noise x2 for +-30 ms
    rip = np.array([], np.int64)
    if two_state:
        q = np.flatnonzero((state == 0) & (t > 1) & (t < DUR - 1))
        rip = np.sort(rng.choice(q, 20, replace=False))
        rip = rip[np.r_[True, np.diff(rip) > SR]]
        for r in rip:
            x[r - int(0.03 * SR):r + int(0.03 * SR)] *= 2.0
    # detected spikes (masked by the tool)
    sp = np.sort(rng.choice(np.arange(SR, T - SR), int(25 * DUR), replace=False))
    tmpl = -400 * np.exp(-0.5 * (np.arange(-15, 17) / 3.0) ** 2)
    for o in range(-15, 17):
        x[sp + o, :4] += tmpl[o + 15] * np.array([1.0, 0.8, 0.5, 0.3])
    fil = np.zeros((T, NTOT), np.int16)
    fil[:, :8] = np.clip(np.round(x), -32767, 32767)
    fil[:, 8:] = np.round(rng.normal(0, 30, (T, 2)))
    base = os.path.join(d, name)
    fil.tofile(base + ".fil")
    sp.astype("<i8").tofile(base + ".res.1")
    # LFP: theta (phase phi) in the active state, 2-Hz delta otherwise, ripple bursts
    tl = np.arange(DUR * LFP_SR) / LFP_SR
    st_l = state[(tl * SR).astype(np.int64)]
    phl = 2 * np.pi * 8.0 * tl + 0.5 * np.sin(2 * np.pi * 0.3 * tl)
    lfp = np.where(st_l == 1, 400 * np.cos(phl), 300 * np.cos(2 * np.pi * 2.0 * tl)) + rng.normal(0, 20, tl.size)
    for r in rip:
        tr = r / SR
        lfp += 300 * np.exp(-0.5 * ((tl - tr) / 0.015) ** 2) * np.cos(2 * np.pi * 180 * (tl - tr))
    np.repeat(np.round(lfp).astype(np.int16)[:, None], NTOT, 1).tofile(base + ".lfp")
    with open(base + ".yaml", "w") as fh:
        fh.write(f"acquisitionSystem: {{nBits: 16, nChannels: {NTOT}, samplingRate: {SR}}}\n"
                 f"fieldPotentials: {{lfpSamplingRate: {LFP_SR}}}\n"
                 "spikeDetection:\n  channelGroups:\n    - channels: [0, 1, 2, 3, 4, 5, 6, 7]\n"
                 f"      nSamples: 32\n      peakSampleIndex: 15\n      sdiffPairs: '{PAIRS}'\n")
    return base, state


def dominant(lab, lev):
    """The two most occupied states, (lower level, higher level)."""
    ks, n = np.unique(lab[lab >= 0], return_counts=True)
    two = ks[np.argsort(-n)[:2]]
    return tuple(sorted(two, key=lambda k: np.nanmedian(lev[lab == k])))


def run(base, out, *extra):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(base))
    try:
        rc = fns.main([os.path.basename(base), "1", "--out", out, "--max-states", "3", *extra])
    finally:
        os.chdir(cwd)
    return rc, np.load(out + ".npz")


def main():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        base, state = make_session(d, "two", True, rng)
        rc, Z = run(base, os.path.join(d, "two_std"))
        check(rc == 0, "two-state file: the tool runs")
        lev, e, o, lab = Z["level"], Z["lev_even"], Z["lev_odd"], Z["state"]
        m = np.isfinite(lev) & np.isfinite(e) & np.isfinite(o)
        real = 1 - np.var(e[m] - o[m]) / 4 / np.var(lev[m])
        check(real > 0.8, f"most of the level's window-to-window variation is real ({real:.0%})")
        K = int(np.argmin(Z["bic"])) + 1
        q, act = dominant(lab, lev)
        cover = np.mean(np.isin(lab[lab >= 0], (q, act)))
        check(K in (2, 3) and cover >= 0.75,
              f"two dominant states by BIC (K = {K}; the two largest hold {cover:.0%} of the windows)")
        tr = Z["transient"]
        check(all(tr[k] for k in range(K) if k not in (q, act)), "any further state is marked transient")
        win_state = state[(Z["t"] * SR).astype(np.int64) + SR // 2]
        sel = np.isin(lab, (q, act))
        agree = np.mean((lab[sel] == act) == (win_state[sel] == 1))
        check(agree > 0.9, f"windows of the two states match the built states ({agree:.0%})")
        ratio = 10 ** (np.nanmedian(lev[lab == act]) - np.nanmedian(lev[lab == q]))
        check(1.4 < ratio < 1.9, f"level ratio of the states x{ratio:.2f} (built x1.64)")
        cm_q, cm_a = np.nanmedian(Z["cm"][lab == q]), np.nanmedian(Z["cm"][lab == act])
        check(abs(cm_q - 0.14) < 0.05 and abs(cm_a - 0.43) < 0.07 and
              np.nanmedian(Z["tails"][lab == act]) > np.nanmedian(Z["tails"][lab == q]),
              f"common share {cm_q:.2f} / {cm_a:.2f} (built 0.14 / 0.43); the active state has heavier tails")
        runs = []
        for k in (q, act):
            idx = np.flatnonzero(lab == k)
            runs.append(float(np.median(np.diff(np.r_[0, np.flatnonzero(np.diff(idx) != 1) + 1, idx.size]))))
        check(all(6 <= r <= 22 for r in runs), f"dwell medians {runs} s (built 8-20 s)")
        pm = Z["phase_mod"]
        check(pm.size == 8 and pm.max() / pm.min() > 1.5, f"theta-phase modulation of the noise x{pm.max() / pm.min():.2f}")
        peri = Z["peri"]
        pk = int(np.argmax(peri)) - peri.size // 2
        check(peri.size > 0 and peri.max() > 1.5 and abs(pk) <= 4,
              f"noise rises at ripples: x{peri.max():.2f} at {pk * 10:+d} ms")
        names = list(Z["mismatch_names"]); eig = Z["mismatch_eig"]
        f0 = np.exp(np.log(eig[names.index(f"state {q}")]).mean())
        f1 = np.exp(np.log(eig[names.index(f"state {act}")]).mean())
        check(1.6 < f1 / f0 < 2.8, f"whitener mismatch: the active state's noise x{f1 / f0:.2f} the quiet's "
                                   "(geometric mean over the 120 dims: x1.78 independent, x2.7 in the common ones)")

        rc, Z5 = run(base, os.path.join(d, "two_c5"), "--variant", "stderiv_C5", "--no-lfp")
        lab5 = Z5["state"]
        q5, a5 = dominant(lab5, Z5["level"])
        sel5 = np.isin(lab5, (q5, a5))
        agree5 = np.mean((lab5[sel5] == a5) == (win_state[sel5] == 1))
        check(rc == 0 and agree5 > 0.9 and np.mean(sel5[lab5 >= 0]) >= 0.75,
              f"stderiv_C5 space (sdiffPairs): the two dominant states match the built ones ({agree5:.0%})")

        base0, _ = make_session(d, "flat", False, rng)
        rc, Z0 = run(base0, os.path.join(d, "flat_std"))
        lev, e, o = Z0["level"], Z0["lev_even"], Z0["lev_odd"]
        m = np.isfinite(lev) & np.isfinite(e) & np.isfinite(o)
        real0 = max(0.0, 1 - np.var(e[m] - o[m]) / 4 / np.var(lev[m]))
        K0 = int(np.argmin(Z0["bic"])) + 1
        check(rc == 0 and real0 < 0.3, f"stationary control: real share {real0:.0%}")
        check(K0 == 1, f"stationary control: one state (K = {K0})")
        pm0 = Z0["phase_mod"]
        check(pm0.size == 0 or pm0.max() / pm0.min() < 1.15,
              f"stationary control: no phase modulation (x{(pm0.max() / pm0.min()) if pm0.size else 1:.2f})")
    print(f"\ntest_fil_noise_states: {'PASS' if fails == 0 else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
