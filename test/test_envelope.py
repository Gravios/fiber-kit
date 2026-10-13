"""fiber_envelope: the collision-robust envelope and near-collision detection on a synthetic
recording with KNOWN templates, amplitude/jitter spreads, a detector dead time and an unsorted
background unit -- so the true per-unit sigma(t) and every true near collision are known.

Checks:
  * the plain std (dispersion_profile-like) is inflated by near collisions; the robust envelope
    recovers the true sigma(t) (noise + amplitude + jitter) within tolerance;
  * flagged spikes are the near-collided ones (recall / precision);
  * explained partners have the right identity and offset, including hidden (dead-time) ones;
  * the pair separation d' keeps a same-unit split near 0 and separates distinct units;
  * the CLI end to end on session-named files (.res/.clu + an int16 signal) writes .env/.nco/.eap.
"""
import os
import sys
import tempfile

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fiber_kit import fiber_envelope as fe     # noqa: E402
from fiber_kit import neuro_io as nio          # noqa: E402

FS, C, PRE, POST, MARGIN, MAXLAG = 32552, 8, 21, 21, 32, 4
SD_A, SD_J = 0.08, 0.35            # amplitude-scale sd, jitter sd (samples)


def _shape(t, gains, delays, w1=2.2, w2=6.0, r=0.35):
    """Analytic multichannel spike at continuous time t (samples, (n,) array) -> (n, C)."""
    tt = t[:, None] - delays[None, :]
    return gains[None, :] * (-np.exp(-0.5 * (tt / w1) ** 2) + r * np.exp(-0.5 * ((tt - 7) / w2) ** 2))


UNITS = {   # id: (rate Hz, gains (C,), delays (C,))
    2: (25.0, np.array([900, 700, 400, 200, 100, 60, 40, 30.]), np.array([0, .5, 1, 1.5, 2, 2, 2, 2.])),
    3: (35.0, np.array([60, 150, 400, 800, 650, 300, 120, 50.]), np.array([2, 1.5, 1, 0, 0, .5, 1, 1.5])),
    4: (20.0, np.array([40, 60, 100, 200, 450, 750, 850, 500.]), np.array([2, 2, 1.5, 1, .5, 0, 0, .5])),
}
BG = (60.0, np.array([120, 160, 200, 220, 200, 160, 120, 90.]), np.zeros(C))   # unsorted background


def simulate(dur_s=60.0, seed=1, dead=20):
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    sos = butter(3, [800, 7000], "bandpass", fs=FS, output="sos")
    sig = sosfiltfilt(sos, rng.normal(0, 1, (n, C)), axis=0)
    sig *= 60.0 / sig.std(0)
    events = []                              # (time float, unit, amp)
    for u, (rate, _g, _d) in list(UNITS.items()) + [(1, BG)]:
        t = np.cumsum(rng.exponential(FS / rate, size=int(rate * dur_s * 1.5)))
        t = t[(t > 200) & (t < n - 200)]
        t = t[np.r_[True, np.diff(t) > 0.002 * FS]]          # refractory
        for x in t:
            events.append((x + rng.normal(0, SD_J), u, 1.0 + rng.normal(0, SD_A)))
    events.sort()
    k = np.arange(-40, 60)
    for x, u, a in events:
        g, d = (UNITS[u][1], UNITS[u][2]) if u > 1 else (BG[1], BG[2])
        i0 = int(np.floor(x))
        sig[i0 + k] += a * _shape(i0 + k - x, g, d)
    ev = np.array([(int(round(x)), u) for x, u, _ in events])
    # detector: sorted units only, with a dead time (a spike within `dead` of the previous detection is lost)
    det, last = [], -10 ** 9
    for t, u in ev:
        if u > 1 and t - last > dead:
            det.append((t, u)); last = t
    det = np.array(det)
    return sig.astype(np.float32), det[:, 0], det[:, 1], ev


def _true_sigma(u, noise_var):
    g, d = UNITS[u][1], UNITS[u][2]
    t = np.arange(-PRE, POST).astype(float)
    T = _shape(t, g, d); dT = np.gradient(T, axis=0)
    return np.sqrt(noise_var[None] + SD_A ** 2 * T ** 2 + dT ** 2 / 12.0)   # res rounds the jittered time: residual offset ~ U(-.5,.5)


_CACHE = {}


def _run():
    if "r" not in _CACHE:
        sig, res, clu, ev = simulate()
        envs, table, noise = fe.run(sig, res, clu, pre=PRE, post=POST, margin=MARGIN, maxlag=MAXLAG,
                                    log=lambda s: None)
        _CACHE["r"] = (sig, res, clu, ev, envs, table, noise)
    return _CACHE["r"]


def _near(res, clu, ev, win=PRE + 11, min_amp=0.2, background=False):
    """True near-collision mask per detected spike: another SORTED unit's event within +-win
    samples (the core window plus the partner's ~11-sample rebound tail that reaches into it) whose
    peak exceeds min_amp of the detected unit's peak.  The unsorted background unit is excluded by
    default: it is never detected, so its spikes are in the off-spike noise windows too -- it IS
    part of the noise floor by construction, not a contamination the envelope should remove."""
    peak = {u: UNITS[u][1].max() for u in UNITS}; peak[1] = BG[1].max()
    out = np.zeros(res.size, bool); part = np.full(res.size, -9); off = np.zeros(res.size, int)
    et = ev[:, 0]
    for i, (t, u) in enumerate(zip(res, clu)):
        j0, j1 = np.searchsorted(et, t - win), np.searchsorted(et, t + win, "right")
        for j in range(j0, j1):
            t2, u2 = ev[j]
            if (u2 == u and abs(t2 - t) <= 1) or (u2 == 1 and not background):
                continue
            if peak[u2] >= min_amp * peak[u]:
                out[i] = True; part[i] = u2; off[i] = t2 - t
    return out, part, off


def test_robust_sigma_recovers_truth_naive_is_inflated():
    sig, res, clu, ev, envs, table, noise = _run()
    for u, env in envs.items():
        ts = _true_sigma(u, noise.var)
        rn, rr = (env.sigma_naive / ts) ** 2, (env.sigma / ts) ** 2      # variance ratios, all (t, c)
        # the plain std is inflated, most where the unit's own signal is small (the contamination
        # lands at random offsets, so it is spread thin but dominates low-variance samples)
        assert rn.mean() > rr.mean() + 0.07 and np.quantile(rn, 0.95) > 1.3, (u, rn.mean(), rr.mean())
        assert 0.9 < rr.mean() < 1.1 and np.quantile(rr, 0.95) < 1.3, (u, rr.mean(), np.quantile(rr, 0.95))


def test_flags_match_true_near_collisions():
    sig, res, clu, ev, envs, table, noise = _run()
    near, _, _ = _near(res, clu, ev)
    near_any, _, _ = _near(res, clu, ev, background=True)          # incl. the unsorted background
    f = np.zeros(res.size, bool); f[table["idx"]] = table["flagged"].astype(bool)
    rec = (f & near).sum() / max(near.sum(), 1)
    fpr = (f & ~near_any).sum() / max((~near_any).sum(), 1)       # flags on spikes with NO second event
    assert near.mean() > 0.05                          # the simulation does contain them
    assert rec > 0.6, rec
    assert fpr < 0.02, fpr                             # ~ the 1 % null quantile


def test_partners_identity_and_offset():
    sig, res, clu, ev, envs, table, noise = _run()
    near, part, off = _near(res, clu, ev)
    e = table["explained"].astype(bool)
    i = table["idx"][e]
    sorted_partner = part[i] > 1                       # the background unit is not in the dictionary
    ok = (table["partner"][e] == part[i]) & (np.abs(table["shift"][e] - off[i]) <= 1)
    assert e.sum() > 50
    assert ok[sorted_partner].mean() > 0.85, ok[sorted_partner].mean()
    # hidden partners (inside the dead time, no .res row) are recovered too
    hid = sorted_partner & (np.abs(off[i]) <= 20)
    assert hid.sum() > 10 and ok[hid].mean() > 0.8, (hid.sum(), ok[hid].mean())


def test_separation_same_vs_different():
    sig, res, clu, ev, envs, table, noise = _run()
    lo, hi = -PRE - MARGIN - MAXLAG, POST + MARGIN + MAXLAG
    core = slice(MARGIN + MAXLAG, MARGIN + MAXLAG + PRE + POST)
    idx = np.flatnonzero(clu == 2)
    halves = [fe.fit_envelope(fe.extract(sig, res[h], lo, hi), core, noise.var, noise.windows,
                              maxlag=MAXLAG) for h in (idx[::2], idx[1::2])]
    same, _ = fe.separation(*halves)
    diff = min(fe.separation(envs[a], envs[b])[0] for a in envs for b in envs if a < b)
    assert same < 2.0, same
    assert diff > 4 * max(same, 1.0), (diff, same)


PAIRS_G6 = "0-2+4,1-3+5,2-4+0+6,3-5+1+7,4-6+2+7,5-7+3+6,6-7+3+5,7-5+6"   # a real 8-ch order-5 pattern


def test_space_transform_tokens():
    from fiber_kit import fiber_stderiv as fst
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 200, (5, 43, C)).round()
    assert fe.space_transform("standard") is None and fe.space_transform("fil") is None
    np.testing.assert_array_equal(fe.space_transform("stderiv")(raw), fst.apply_stderiv_transform(raw))
    t = fe.space_transform("stderiv_C5_D34", PAIRS_G6)          # _D suffix ignored
    np.testing.assert_array_equal(t(raw), fst.apply_stderiv_transform(raw, sets=fst.parse_sdiff_sets(PAIRS_G6)))
    for bad in (("stderiv_C4", PAIRS_G6), ("stderiv_C5", None), ("pca", None)):
        try:
            fe.space_transform(*bad)
        except ValueError:
            continue
        raise AssertionError(f"space_transform{bad} should refuse")


def test_custom_c5_space():
    """The same checks in the session's custom order-5 stderiv space (the fiber feature space).
    The spatial + temporal derivative suppresses the slow rebound tail and the shared footprint, so
    only partners whose trough lands inside the window stay visible (|offset| < 16: ~64 % flagged,
    vs ~6 % at 22-32) -- and, consistently, the plain std is barely inflated in this space."""
    sig, res, clu, ev, *_ = _run()
    envs, table, noise = fe.run(sig, res, clu, pre=PRE, post=POST, margin=MARGIN, maxlag=MAXLAG,
                                transform=fe.space_transform("stderiv_C5", PAIRS_G6), log=lambda s: None)
    near, part, off = _near(res, clu, ev)
    near_any, _, _ = _near(res, clu, ev, background=True)
    f = np.zeros(res.size, bool); f[table["idx"]] = table["flagged"].astype(bool)
    close = near & (np.abs(off) < 16)
    assert f[close].mean() > 0.5, f[close].mean()
    assert (f & ~near_any).sum() / (~near_any).sum() < 0.02
    e = table["explained"].astype(bool); i = table["idx"][e]; sp = part[i] > 1
    ok = (table["partner"][e] == part[i]) & (np.abs(table["shift"][e] - off[i]) <= 1)
    assert e.sum() > 50 and ok[sp].mean() > 0.85, (e.sum(), ok[sp].mean())


def test_channel_ratio_amplitude_scaled_and_merge_sensitive():
    """channel_ratio: observed / (noise + var_a T^2 + var_b T'^2) per channel -- ~1 for a single unit
    whatever its amplitude (the three units differ ~2x in peak), and well above 1 when 10 % of
    another unit is merged in.  'kept' sees the merge; 'clean' is blind to it (the foreign spikes
    fail the bump test and are excluded); 'all' is inflated by near collisions even when single."""
    sig, res, clu, ev, envs, table, noise = _run()
    single = {u: e.channel_ratio().max() for u, e in envs.items()}
    assert all(0.9 < x < 1.2 for x in single.values()), single
    assert all(e.channel_ratio("all").max() > max(single.values()) + 0.1 for e in envs.values())
    rng = np.random.default_rng(1)
    c = clu.copy(); iu = np.flatnonzero(clu == 2); iv = np.flatnonzero(clu == 3)
    c[rng.choice(iv, int(0.1 / 0.9 * iu.size), replace=False)] = 2
    E, _, _ = fe.run(sig, res, c, pre=PRE, post=POST, margin=MARGIN, maxlag=MAXLAG, log=lambda s: None)
    assert E[2].channel_ratio().max() > 1.5, E[2].channel_ratio().max()
    assert E[2].channel_ratio("clean").max() < 1.2


DISTAL = np.array([0, 0, 0, 0, 0, -150, -250, -300.])   # positive-going


def simulate_adapting(dur_s=90.0, seed=3, tau=0.3, k=0.6, widen=0.5):
    """Unit 2 adapts: amplitude 1/(1 + k*EWMA_tau(prior spikes)), and a smaller spike also gains a
    distal-channel component widen*(1-amp)*DISTAL -- an amplitude-dependent SHAPE change on the far
    channels (where the state-dependent part of real CA1 footprints sits), not a rescaling.
    Unit 2 fires in bursts (3-5 spikes at 4-8 ms) so the adaptation state spans a wide range."""
    from fiber_kit import fiber_adapt as fad
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    sos = butter(3, [800, 7000], "bandpass", fs=FS, output="sos")
    sig = sosfiltfilt(sos, rng.normal(0, 1, (n, C)), axis=0); sig *= 60.0 / sig.std(0)
    ev = []
    for u, (rate, _g, _d) in UNITS.items():
        t0 = np.cumsum(rng.exponential(FS / (rate / (4.0 if u == 2 else 1.0)), size=int(rate * dur_s)))
        ts = []
        for x in t0:
            ts.append(x)
            if u == 2:
                for _ in range(rng.integers(2, 5)):
                    ts.append(ts[-1] + rng.uniform(0.004, 0.008) * FS)
        ts = np.array([x for x in ts if 300 < x < n - 300])
        amp = np.ones(ts.size)
        if u == 2:
            amp = 1.0 / (1.0 + k * fad.ewma(ts / FS, tau))
        ev += [(x + rng.normal(0, 0.2), u, a * (1 + rng.normal(0, 0.03))) for x, a in zip(ts, amp)]
    ev.sort()
    kk = np.arange(-40, 60)
    for x, u, a in ev:
        g, d = UNITS[u][1], UNITS[u][2]
        i0 = int(np.floor(x))
        sig[i0 + kk] += a * _shape(i0 + kk - x, g, d)
        if u == 2:      # the adapted (smaller) spike also carries a distal-channel component
            sig[i0 + kk] += widen * (1 - min(a, 1.0)) * _shape(i0 + kk - x, DISTAL, d, w1=3.0, r=0.0)
    det, last = [], -10 ** 9
    for x, u, _ in ev:
        t = int(round(x))
        if t - last > 20:
            det.append((t, u)); last = t
    det = np.array(det)
    return sig.astype(np.float32), det[:, 0], det[:, 1]


_CACHE_A = {}


def _run_adapting():
    if "r" not in _CACHE_A:
        sig, res, clu = simulate_adapting()
        envs, table, noise = fe.run(sig, res, clu, pre=PRE, post=POST, margin=MARGIN, maxlag=MAXLAG,
                                    sr=FS, log=lambda s: None)
        _CACHE_A["r"] = (sig, res, clu, envs, table, noise)
    return _CACHE_A["r"]


def test_cell_layer_adaptation_and_amplitude_dependent_shape():
    """The cell layer recovers the adaptation law (tau, R^2) of the adapting unit, finds a substantial
    amplitude-dependent shape term for it, and its amplitude-scaled ratio drops towards 1 once
    that term is allowed (the plain amplitude model reads it as excess spread)."""
    sig, res, clu, envs, table, noise = _run_adapting()
    e = envs[2]
    assert 0.12 <= e.adapt["tau"] <= 0.75 and e.adapt["r2"] > 0.3, e.adapt
    assert all(envs[u].adapt["r2"] < 0.1 for u in (3, 4)), [envs[u].adapt for u in (3, 4)]
    assert e.shape_gain() > 2 * max(envs[3].shape_gain(), envs[4].shape_gain())
    plain = (e.var_clean.sum(0) / (e.var_noise[None] + e.var_a * e.template ** 2
                                    + e.var_b * e.dtemplate ** 2).sum(0)).max()
    cell = e.channel_ratio("clean").max()
    assert cell < plain - 0.02 and cell < 1.05, (plain, cell)


def test_merge_units_rejoins_energy_band_fragments():
    """fiber_session-style fragments -- the adapting unit cut at its amplitude median (an energy-band
    split), another unit cut at random -- are re-joined by merge_units at theta 7, nothing true is
    vetoed, and one adaptation law explains the re-joined energy bands."""
    sig, res, clu, envs, table, noise = _run_adapting()
    lab = clu.copy()
    i2 = np.flatnonzero(clu == 2); lo_amp = envs[2].a < np.median(envs[2].a)
    lab[i2[lo_amp]] = 12
    i3 = np.flatnonzero(clu == 3); lab[i3[::2]] = 13
    E, _, nz = fe.run(sig, res, lab, pre=PRE, post=POST, margin=MARGIN, maxlag=MAXLAG, sr=FS,
                      partners=False, log=lambda s: None)
    new, rows = fe.merge_units(sig, res, lab, E, nz, theta=7.0, sr=FS, log=lambda s: None)
    for a_, b_ in ((2, 12), (3, 13)):
        assert np.unique(new[(lab == a_) | (lab == b_)]).size == 1, (a_, b_)
    assert np.unique(new[new > 1]).size == 3
    same = [r for r in rows if {r["ua"], r["ub"]} in ({2, 12}, {3, 13})]
    assert same and not any(r["vetoed"] for r in same)
    # the inflation of a same-unit union stays at ~0 (the gate it feeds is validated on real look-alike
    # pairs: 31 of 32 vetoed g6 pairs were different cells; synthetic units are too unlike for it)
    assert all(r["inflation"] < 0.03 for r in same), [r["inflation"] for r in same]
    # one adaptation law over the re-joined energy bands explains their amplitudes
    pr = [r for r in rows if {r["ua"], r["ub"]} == {2, 12}][0]
    assert pr["adapt_joint"] > 0.3, pr


def test_type_features_and_conflict():
    t = np.arange(-21, 21, dtype=float)
    narrow = np.zeros((42, 2)); narrow[:, 0] = -np.exp(-0.5 * (t / 2) ** 2) + 0.4 * np.exp(-0.5 * ((t - 6) / 3) ** 2)
    broad = np.zeros((42, 2)); broad[:, 0] = -np.exp(-0.5 * (t / 3) ** 2) + 0.3 * np.exp(-0.5 * ((t - 18) / 4) ** 2)
    burst_times = np.sort(np.r_[np.arange(0, 60, 1.0), np.arange(0, 60, 1.0) + 0.005])
    fn = fe.type_features(narrow, np.arange(0, 60, 0.05), FS)
    fb = fe.type_features(broad, burst_times, FS)
    assert fn["narrow"] and not fb["narrow"] and fb["width_ms"] > 0.45, (fn, fb)
    assert fb["burst"] > 0.45 and fn["burst"] == 0.0 and abs(fn["rate_hz"] - 20) < 0.5
    veto = fe.cell_type_conflict({1: fn, 2: fb, 3: fb})
    assert veto(1, 2) and not veto(2, 3)


def test_probe_mirror_shanks_y():
    import os as _os
    from fiber_kit import fiber_localize as flz
    g = [[0, 0], [-11, 20], [11, 40], [-11, 60], [200, 0], [189, 20], [211, 40], [189, 60]]
    m = flz.mirror_shanks_y(g, 4)
    np.testing.assert_array_equal(m[:, 0], np.asarray(g)[:, 0])           # stagger kept
    np.testing.assert_array_equal(m[:4, 1], [60, 40, 20, 0])
    np.testing.assert_array_equal(m[4:, 1], [60, 40, 20, 0])
    with tempfile.TemporaryDirectory() as d:
        p = _os.path.join(d, "x.probe")
        with open(p, "w") as f:
            f.write("probeFile:\n  sites:\n    count_per_shank: 4\n    geometry:\n"
                    + "".join(f"      - [{x}, {y}]\n" for x, y in g))
        np.testing.assert_array_equal(flz.load_geometry(p, [4, 5, 6, 7], flip_y=[0])[:, 1], [60, 40, 20, 0])
        old = _os.environ.get("FIBER_PROBE_FLIP_Y")
        try:
            _os.environ["FIBER_PROBE_FLIP_Y"] = "0"
            np.testing.assert_array_equal(flz.load_geometry(p, [0, 1])[:, 1], [60, 40])
            _os.environ["FIBER_PROBE_FLIP_Y"] = ""
            np.testing.assert_array_equal(flz.load_geometry(p, [0, 1])[:, 1], [0, 20])
        finally:
            if old is None:
                _os.environ.pop("FIBER_PROBE_FLIP_Y", None)
            else:
                _os.environ["FIBER_PROBE_FLIP_Y"] = old


def test_cli_end_to_end():
    sig, res, clu, ev, envs, table, noise = _run()
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "sim")
        nio.write_res(b, 6, res, variant="stderiv")
        nio.write_clu(b, 6, clu, variant="stderiv_C5_D34")
        sigp = b + ".fil"
        np.clip(np.rint(sig), -32768, 32767).astype("<i2").tofile(sigp)
        fe.main([b, "6", "--signal", sigp, "--total-channels", str(C), "--channels", "0-7",
                 "--space", "fil", "--out-tag", "env", "--eap", "--merge", "7"])
        _, merged = nio.read_clu_at(b, 6, variant="stderiv_C5_D34", tag="envmerge")
        assert merged.size == res.size and np.unique(merged[merged > 1]).size == 3   # 3 distinct units stay apart
        assert os.path.exists(b + ".envmerge.6.env")
        assert os.path.exists(b + ".env.6.env") and os.path.exists(b + ".nco.6.env")
        with open(b + ".nco.6.env") as fh:
            rows = fh.read().splitlines()
        assert len(rows) - 1 == res.size
        e = nio.read_eap(b, 6, tag="")             # annotates the analysed clu stage
        assert e["ok"] and (e["cells"] != nio.EAP_ABSENT).any()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
