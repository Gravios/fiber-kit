"""fiber_localize: the one-flank (edge) refit starts from a feasible point at any amplitude scale.

A near-flat cluster (peak amplitude below 1/30 ADU) at a terminal channel sent fiber-cpos's
cpos-clean step down with "ValueError: Initial guess is outside of provided bounds" (the edge refit
started its source amplitude at a.max()*30, below its lower bound 1), killing the stage and the trial.
  * _fit_edge on such a profile returns finite parameters inside the bounds;
  * localize_unit on a whole near-silent cluster peaking at the array end returns finite numbers;
  * a normal-amplitude one-flank profile fits exactly as before (the start point is unchanged there).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_localize as loc   # noqa: E402

fails = 0
def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


def main():
    C = 8
    xy = np.stack([np.zeros(C), np.arange(C) * 20.0], 1)
    # a one-flank profile peaking at the last channel, at two scales
    d = np.sqrt((0 - xy[:, 0]) ** 2 + (150.0 - xy[:, 1]) ** 2 + 30.0 ** 2)
    for scale, what in ((300.0 * 30, "normal amplitude"), (0.02 * 30, "near-flat (peak < 1/30)")):
        a = scale / d
        try:
            p, sig, rel = loc._fit_edge(a, xy, dipole=True)
            ok = np.all(np.isfinite(p[:4])) and 3.0 <= p[2] <= 300.0 and 120.0 <= p[1] <= 160.0
            check(ok, f"_fit_edge, {what}: finite, inside the bounds (y0 {p[1]:.1f}, r {p[2]:.1f})")
        except ValueError as e:
            check(False, f"_fit_edge, {what}: {e}")
    # the same profile through localize_unit, from waveforms
    rng = np.random.default_rng(0)
    t = np.arange(32)
    shape = -np.exp(-0.5 * ((t - 10) / 1.5) ** 2)
    prof = 0.01 / d * d.min()                                    # peak-to-peak ~0.01 on the last channel
    W = shape[None, :, None] * prof[None, None, :] + rng.normal(0, 1e-4, (50, 32, C))
    try:
        r = loc.localize_unit(W, xy, dipole=True)
        check(all(np.isfinite(r[k]) for k in ("x0", "y0", "z0", "A")), "localize_unit on a near-silent edge cluster: finite")
    except ValueError as e:
        check(False, f"localize_unit on a near-silent edge cluster: {e}")
    print("OK (0 failures)" if not fails else f"FAILED ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
