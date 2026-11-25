"""The rkk realign loop's lag half-width and the fine split's dimensionality are knobs.

  * FK_SESSION_RKK_REALIGN_MAXLAG: was a literal 6 at both alignment sites (the rkk realign loop and
    Block N); the default is still 6 and the environment moves it;
  * FK_SESSION_RKK_DIMS: --rkk-dims had no knob (default 6), so a plan could not set it.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += 0 if ok else 1


def probe(env):
    code = ("import sys, argparse; sys.path.insert(0, %r)\n"
            "from fiber_kit import fiber_session as fs\n"
            "ap = argparse.ArgumentParser(); fs.add_core_arguments(ap)\n"
            "print(fs._RKK_REALIGN_MAXLAG, ap.get_default('rkk_dims'))\n" % SRC)
    e = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
    e.update(env)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e)
    return r.stdout.split() if r.returncode == 0 else [r.stderr[-300:]]


check(probe({}) == ["6", "6"], "defaults: half-width 6 (the former literal), rkk-dims 6")
check(probe({"FK_SESSION_RKK_REALIGN_MAXLAG": "12", "FK_SESSION_RKK_DIMS": "9"}) == ["12", "9"],
      "FK_SESSION_RKK_REALIGN_MAXLAG and FK_SESSION_RKK_DIMS move them")
src = open(os.path.join(SRC, "fiber_kit", "fiber_session.py")).read()
check("maxlag=6" not in src.replace("maxlag=6)", "", 0).split("def _rkk_realign")[1].split("\ndef ")[0],
      "the rkk realign loop no longer carries a literal half-width")
print("test_rkk_realign_knobs:", "PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
