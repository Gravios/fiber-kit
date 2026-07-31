#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  fiber_flag_artifacts.py — flag acquisition artifacts BEFORE clustering, from
#  the unwhitened temporal derivative.  Emits an exclusion mask in .clu form.
#
#  WHY THIS AND NOT AN OUTLIER RULE.  The obvious test -- a spike far from its
#  cluster's centroid -- cannot run as preprocessing, because "far from the unit it
#  was assigned to" is a property of a spike-and-assignment PAIR and does not exist
#  before clustering.  Measured on sirotaA-jg-000005-20120312 g4: the spikes at
#  per-cluster |z|>10 sit at a GLOBAL robust z of only 6.8, and a cluster-free
#  global rule either flags 33k spikes to catch 1.5k, or at |z|>10 recovers 3% of
#  them.  The features here are per-spike intrinsics instead -- they need only the
#  spike window, so they run on the raw sort.
#
#  THE DISCRIMINANT.  An acquisition glitch is high-frequency and shape-incoherent;
#  a spike is smooth and repeatable.  The unwhitened first difference separates them
#  where the raw waveform does not:
#
#      hf_frac = RMS(dx/dt) / RMS(x)        on the channel with the largest |dx|
#
#  On that session, hf_frac > 0.90 flags 30 spikes with split-half template cosine
#  0.060 -- they share NO waveform, the definitive artifact signature -- and ISI
#  co-occurrence 12,000x Poisson.  Every population that is actually a neuron scores
#  0.97-1.00 on the same test.  A representative flagged trace, sign-flipping nearly
#  every sample and then landing on exact zeros before resuming:
#
#      184 -127 -214  159  -58 -400  266  518 -881    0    0  258  463  184  -29
#
#  DO NOT gate on amplitude.  The flagged events are SMALL -- median peak 0.34x a
#  normal spike -- so every amplitude-driven rule misses them and instead removes the
#  high-amplitude tail of real units (peak>8000 on that session: split-half 0.974,
#  zero refractory violations -- a clean cell, not an artifact).
#
#  SCOPE, HONESTLY.  30 spikes is 0.005% of a session; this will not visibly change a
#  sort.  Its value is being correct and cheap, not being a fix.  Where an acquisition
#  fault is frequent, most of its damage sits in windows milder than this catches, and
#  only a scan of the .dat/.fil for the actual corrupt intervals will find those --
#  42-sample spike windows sample too little of the recording to substitute.
#
#  Output is a per-spike .clu-format mask (0 = keep, 1 = artifact) aligned to the .res,
#  consumed by `fiber-session --exclude-clu`.  Knobs read FK_FLAG_*.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import numpy as np

try:
    from . import neuro_io as nio, session_yaml as sy
    from . import config as cfgmod
except ImportError:
    import neuro_io as nio, session_yaml as sy
    import config as cfgmod

_LP = "\u25b8 fiber-flag-artifacts"
def _log(m=""): print(f"{_LP} \u00b7 {m}" if m else _LP)
def _det(k, v, w=12): print(f"{' ' * (len(_LP) + 3)}{k:<{w}} {v}")

_KNOBS = {
    "FK_FLAG_HF_FRAC": ("hf_frac", float, 0.90),
    "FK_FLAG_D_CREST": ("d_crest", float, 0.0),
    "FK_FLAG_SAT_ADU": ("sat_adu", int, 32000),
    "FK_FLAG_BATCH": ("batch", int, 50000),
}


def _knob_default(name, typ, fallback, gcfg):
    for src in (os.environ.get(name), (gcfg or {}).get(name)):      # env BEFORE yaml, per the documented order
        if src is None or str(src).strip() == "":
            continue
        try:
            return typ(src)
        except (TypeError, ValueError):
            pass
    return fallback


def spike_features(spk, n, nsamp, nchan, batch=50000):
    """Per-spike intrinsics from the unwhitened temporal derivative.

    Returns hf_frac (RMS(dx)/RMS(x)), d_crest (peak(dx)/RMS(dx)) and peak(|x|), each
    taken on the channel carrying the largest |dx| -- the channel the glitch is on,
    which is not generally the channel carrying the spike."""
    hf = np.zeros(n, np.float32); dc = np.zeros(n, np.float32); pk = np.zeros(n, np.float32)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        X = np.asarray(spk[s:e], np.float32)
        D = np.diff(X, axis=1)
        Dc = D - D.mean(1, keepdims=True)
        A = X - X.mean(1, keepdims=True)
        drms = np.sqrt((Dc ** 2).mean(1)); dpk = np.abs(Dc).max(1)
        rms = np.sqrt((A ** 2).mean(1))
        j = dpk.argmax(1); i = np.arange(e - s)
        hf[s:e] = drms[i, j] / np.maximum(rms[i, j], 1e-6)
        dc[s:e] = dpk[i, j] / np.maximum(drms[i, j], 1e-6)
        pk[s:e] = np.abs(A).max((1, 2))
    return hf, dc, pk


def main():
    gcfg = cfgmod.load_global_config()
    ap = argparse.ArgumentParser(prog="fiber-flag-artifacts",
                                 description="Flag acquisition artifacts from the unwhitened derivative and "
                                             "emit a per-spike exclusion mask for fiber-session --exclude-clu.")
    sy.add_session_args(ap, positional=False, nchan=False, sr=False)
    ap.add_argument("args", nargs="*", metavar="[session] group",
                    help="1-based spike group, optionally preceded by the session basename/folder; "
                         "session defaults to $FK_SESS, else the basename of $FK_DIR (default $PWD)")
    ap.add_argument("--spk-method", "--spk-variant", dest="spk_variant", default="standard",
                    help="waveform axis (standard = the unwhitened curation axis these features assume)")
    ap.add_argument("--out-stage", "--out-tag", dest="out_tag", default="artifact",
                    help="stage tag of the emitted mask .clu")
    ap.add_argument("--dry-run", action="store_true", help="report the flagged count, write nothing")
    for name, (dest, typ, fb) in _KNOBS.items():
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest, type=typ,
                        default=_knob_default(name, typ, fb, gcfg),
                        help=f"{name} (default {_knob_default(name, typ, fb, gcfg)})")
    a = ap.parse_args()

    if os.environ.get("FK_DIR"):
        os.chdir(os.environ["FK_DIR"])
    if len(a.args) == 2:
        a.session, grp = a.args
    elif len(a.args) == 1:
        a.session = os.environ.get("FK_SESS") or os.path.basename(os.path.abspath(os.getcwd()))
        grp = a.args[0]
    else:
        ap.error("expected '<session> <group>' or just '<group>'")
    try:
        elec = int(grp)
    except ValueError:
        ap.error(f"group must be an integer, got {grp!r}")

    cfg = sy.resolve_session_params(a.session, elec, channels=a.channels, ntotal=a.ntotal, nsamp=a.nsamp)
    base = cfg["base"]; NS, NC = cfg["nsamp"], cfg["nchan"]
    res = nio.read_res(base, elec); n = len(res)
    spkp = nio.session_path(base, "spk", elec, variant=a.spk_variant)
    spk = nio.open_spk_file(spkp, NS, NC)
    assert spk.shape[0] == n, f".res {n} vs {os.path.basename(spkp)} {spk.shape[0]}"
    _log(f"group {elec} \u00b7 {n:,} spikes \u00b7 {os.path.basename(spkp)}")

    hf, dc, pk = spike_features(spk, n, NS, NC, batch=a.batch)
    flag = hf > a.hf_frac
    if a.d_crest > 0:
        flag &= dc > a.d_crest
    if a.sat_adu > 0:
        flag |= pk >= a.sat_adu                       # saturation is unambiguous, gate it regardless
    _det("hf_frac", f"median {np.median(hf):.3f}  p99 {np.percentile(hf, 99):.3f}  max {hf.max():.3f}")
    _det("thresholds", f"hf_frac > {a.hf_frac}" + (f" AND d_crest > {a.d_crest}" if a.d_crest > 0 else "")
         + (f"; OR |peak| >= {a.sat_adu} (saturation)" if a.sat_adu > 0 else ""))
    _det("flagged", f"{int(flag.sum()):,} spikes ({100 * flag.mean():.4f}%)")
    if flag.any():
        _det("their peak", f"median {np.median(pk[flag]):.0f} ADU vs {np.median(pk):.0f} for all spikes "
                           f"({np.median(pk[flag]) / max(np.median(pk), 1e-9):.2f}x)")
    if a.dry_run:
        _det("dry-run", "nothing written")
        return
    out = nio.session_path(base, "clu", elec, variant=a.spk_variant, tag=a.out_tag)
    nio.write_clu_file(out, flag.astype(np.int32), n_clusters=2)
    _det("wrote", f"{os.path.basename(out)}   (0 = keep, 1 = artifact; pass to fiber-session --exclude-clu)")


if __name__ == "__main__":
    main()
