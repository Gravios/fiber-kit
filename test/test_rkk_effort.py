"""The rkk effort knobs: FK_SESSION_RKK_STARTS (restarts per starting K), FK_SESSION_RKK_MIN (a starting-K
sweep up to --rkk-max) and FK_SESSION_RKK_REALIGN_ITERS (the --rkk-realign-iters default).

The CEM itself always runs to convergence, so the knobs are checked where they act: what _rkk_labels
asks klustakwik for (n_starts, min_clusters), that a sweep / restarts never score worse than the single
start they include (the best score wins), and that the CLI default follows the knob."""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_session as fs   # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += not ok


def main():
    rng = np.random.default_rng(0)
    F = np.concatenate([rng.normal(m, 0.6, (120, 4)) for m in ((0, 0, 0, 0), (5, 0, 0, 0), (0, 5, 0, 0))])
    calls = []
    real = fs._rkk

    def spy(X, **kw):
        calls.append(dict(kw))
        return real(X, **kw)

    fs._rkk = spy
    try:
        fs._RKK_STARTS, fs._RKK_MIN = 1, 0
        lab0 = fs._rkk_labels(F, 12, 10)
        check(calls[-1]["n_starts"] == 1 and calls[-1]["min_clusters"] == 12 and calls[-1]["max_clusters"] == 12,
              "defaults: one start at --rkk-max, no sweep (as before)")
        fs._RKK_STARTS, fs._RKK_MIN = 3, 4
        lab1 = fs._rkk_labels(F, 12, 10)
        check(calls[-1]["n_starts"] == 3 and calls[-1]["min_clusters"] == 4 and calls[-1]["max_clusters"] == 12,
              "RKK_STARTS 3 / RKK_MIN 4 -> klustakwik(n_starts=3, min_clusters=4, max_clusters=12)")
        fs._RKK_STARTS, fs._RKK_MIN = 1, 40
        fs._rkk_labels(F, 12, 10)
        check(calls[-1]["min_clusters"] == 12, "RKK_MIN above --rkk-max is clamped to it (no inverted sweep)")
        k0 = len(np.unique(lab0[lab0 >= 0])); k1 = len(np.unique(lab1[lab1 >= 0]))
        check(k0 >= 3 and k1 >= 3, f"both settings find the 3 planted clusters ({k0}, {k1})")
    finally:
        fs._rkk = real
        fs._RKK_STARTS, fs._RKK_MIN = 1, 0

    # the sweep + restarts keep the best score: never worse than the single start they include
    import importlib
    kk = importlib.import_module("fiber_kit.klustakwik")
    X = F
    single = kk._KK((X - X.min(0)) / (X.max(0) - X.min(0)), 12, 0.0, 1, rng=np.random.default_rng((1, 12, 0))).cem()
    best = np.inf
    for k0 in kk._sweep(4, 12):
        for st in range(3):
            o = kk._KK((X - X.min(0)) / (X.max(0) - X.min(0)), k0, 0.0, 1, rng=np.random.default_rng((1, k0, st))).cem()
            best = min(best, o)
    check(best <= single + 1e-9, f"the sweep's best score ({best:.4g}) is never worse than the single start ({single:.4g})")

    env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
    env["FK_SESSION_RKK_REALIGN_ITERS"] = "5"
    r = subprocess.run([sys.executable, "-c",
                        "import argparse, sys; from fiber_kit import fiber_session as fs; ap = argparse.ArgumentParser(); "
                        "fs.add_core_arguments(ap); a = ap.parse_args([]); print(a.rkk_realign_iters)"],
                       capture_output=True, text=True, env=env)
    check(r.stdout.strip().endswith("5"), f"--rkk-realign-iters defaults to FK_SESSION_RKK_REALIGN_ITERS ({r.stdout.strip()[-3:]!r})")
    print(f"test_rkk_effort: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
