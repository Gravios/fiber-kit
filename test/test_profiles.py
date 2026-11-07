#!/usr/bin/env python3
# test_profiles.py -- the shipped config profiles, `fk list config`, `fk set config`, `fk completion`.
#
#   * fiber-kit-best.yaml lists exactly fiber-kit-exp.yaml's knobs, once each, and differs from it in
#     exactly the exp-search winner (5 knobs) and the plan (psame-chunk dropped, link reads fs_clean,
#     psame carries theta 0.9 / chunk-aware 1); every step of its plan has a unique name;
#   * a profile is named by short name, file name (with/without .yaml) or a path; an unknown one fails;
#   * `fk set config --profile best` writes ./fiber-kit.yaml, refuses to overwrite without --force,
#     `fk list config` marks it (exact, then edited); `--names` lists the short names;
#   * `fk completion bash` / `zsh` is valid bash and completes the subcommands, the profiles and the steps.
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fiber_kit import init_config as ic, plan_steps as ps   # noqa: E402
from fiber_kit.config import yaml_load                      # noqa: E402

PKG = os.path.join(ROOT, "src", "fiber_kit")
fails = 0


def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what)
    fails += not ok


def load(name):
    with open(os.path.join(PKG, name)) as fh:
        return yaml_load(fh)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def fk(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    return subprocess.run([sys.executable, "-c", "from fiber_kit.plan_steps import main; import sys; "
                           "sys.exit(main(sys.argv[1:]))", *args], cwd=cwd, env=env, capture_output=True, text=True)


def main():
    exp, best = load("fiber-kit-exp.yaml"), load("fiber-kit-best.yaml")
    ke, kb = {k for k in exp if k.startswith("FK_")}, {k for k in best if k.startswith("FK_")}
    check(ke == kb, f"best lists exactly exp's knobs ({len(kb)}; differ: {sorted(ke ^ kb)})")
    with open(os.path.join(PKG, "fiber-kit-best.yaml")) as fh:
        names = re.findall(r"^(FK_[A-Z0-9_]+):", fh.read(), re.M)
    check(len(names) == len(set(names)), "every knob appears once in best")
    diff = {k: (exp[k], best[k]) for k in ke if exp[k] != best[k]}
    want = {"FK_SESSION_CFIBER_Q": (0.9, 0.85), "FK_SESSION_INCL_ASSIGN": (0, 1), "FK_SESSION_DIP_ALPHA": (0.02, 0.03),
            "FK_CONSENSUS_THETA_CELL": (7.0, 5.0), "FK_XCM_COS_THR": (0.96, 0.94)}
    check(diff == want, f"best differs from exp in exactly the winner's 5 knobs ({diff})")
    pe = [s["name"] for s in exp["pipeline"]]; pb = [s["name"] for s in best["pipeline"]]
    check(pb == [n for n in pe if n != "psame-chunk"], f"best's plan is exp's without psame-chunk ({pb})")
    link = next(s for s in best["pipeline"] if s["name"] == "link")
    psame = next(s for s in best["pipeline"] if s["name"] == "psame")
    check(link["in"] == "fs_clean", "link reads fs_clean (the dropped step's input)")
    check(psame.get("params") == {"theta": 0.9, "chunk-aware": 1}, f"psame carries the tuned params ({psame.get('params')})")
    try:
        steps = ps.load_steps(os.path.join(PKG, "fiber-kit-best.yaml"))
        check(all(s["name"] for s in steps) and len({s["name"] for s in steps}) == len(steps), "best's plan names every step uniquely")
    except ps.PlanError as e:
        check(False, f"best's plan loads: {e}")

    # names
    for arg, want_short in [("best", "best"), ("best.yaml", "best"), ("fiber-kit-best.yaml", "best"), ("fiber-kit-exp", "exp"),
                            ("fiber-kit.yaml", "base"), ("base", "base")]:
        short, fn, text = ic.profile_text(arg)
        check(short == want_short and text.startswith("# ===="), f"profile_text({arg!r}) -> {short}")
    try:
        ic.profile_text("nope"); check(False, "unknown profile is refused")
    except ValueError as e:
        check("base, exp, best" in str(e), f"unknown profile is refused, naming the profiles: {e}")
    check(ic.describe(ic.profile_text("best")[2]).startswith("TUNED"), "describe() reads the tagline")

    # fk set / list config in a scratch folder
    tmp = tempfile.mkdtemp(prefix="fk-profiles-")
    try:
        r = fk("list", "config", cwd=tmp)
        check(r.returncode == 0 and "not present" in r.stdout and " best " in r.stdout, "fk list config: no ./fiber-kit.yaml yet")
        r = fk("list", "config", "--names", cwd=tmp)
        check(r.stdout.split() == ["base", "exp", "best"], f"fk list config --names ({r.stdout.split()})")
        r = fk("set", "config", "--profile", "best.yaml", cwd=tmp)
        same = os.path.exists(os.path.join(tmp, "fiber-kit.yaml")) and \
            read(os.path.join(tmp, "fiber-kit.yaml")) == read(os.path.join(PKG, "fiber-kit-best.yaml"))
        check(r.returncode == 0 and same, "fk set config --profile best.yaml writes ./fiber-kit.yaml = the best profile")
        r = fk("set", "config", "--profile", "exp", cwd=tmp)
        check(r.returncode == 1 and "already exists (the best profile)" in r.stderr, "a second set refuses to overwrite")
        r = fk("list", "config", cwd=tmp)
        check("*best" in r.stdout and "the best profile" in r.stdout, "fk list config marks the exact profile")
        with open(os.path.join(tmp, "fiber-kit.yaml"), "a") as fh:
            fh.write("FK_CHUNKS: \"1-3\"\n")
        r = fk("list", "config", cwd=tmp)
        check("~best" in r.stdout and "edited copy" in r.stdout, "an edited copy is marked ~")
        r = fk("set", "config", "--profile", "exp", "--force", cwd=tmp)
        check(r.returncode == 0 and ic.current_profile(os.path.join(tmp, "fiber-kit.yaml")) == ("exp", "exact"), "--force overwrites")
        r = fk("set", "config", "--profile", "nope", cwd=tmp)
        check(r.returncode == 2 and "no profile 'nope'" in r.stderr, "fk set config refuses an unknown profile")
        r = fk("set", "nothing", "--profile", "exp", cwd=tmp)
        check(r.returncode == 2 and "knows `config` only" in r.stderr, "fk set refuses anything but config")
        r = fk("list", cwd=tmp)      # the exp plan now sits in ./fiber-kit.yaml
        check(r.returncode == 0 and "psame-chunk" in r.stdout, "fk list reads the plan of the written profile")
        # completion
        for shell in ("bash", "zsh"):
            r = fk("completion", shell, cwd=tmp)
            ok = r.returncode == 0 and "complete -F _fk_complete fk" in r.stdout
            if shell == "bash" and shutil.which("bash"):
                ok = ok and subprocess.run(["bash", "-n"], input=r.stdout, capture_output=True, text=True).returncode == 0
            check(ok, f"fk completion {shell} is a complete script" + (" (bash -n ok)" if shell == "bash" else ""))
        if shutil.which("bash"):
            script = fk("completion", "bash", cwd=tmp).stdout
            # a stand-in `fk` on PATH for the script's own calls
            with open(os.path.join(tmp, "fk"), "w") as fh:
                fh.write("#!/bin/bash\nexec " + sys.executable + " -c 'from fiber_kit.plan_steps import main; import sys; "
                         "sys.exit(main(sys.argv[1:]))' \"$@\"\n")
            os.chmod(os.path.join(tmp, "fk"), 0o755)
            drive = script + '''
t() { COMP_WORDS=("$@"); COMP_CWORD=$(( ${#COMP_WORDS[@]} - 1 )); COMP_LINE="$*"; COMP_POINT=${#COMP_LINE}
      COMPREPLY=(); _fk_complete; echo "${COMPREPLY[*]}"; }
t fk ""; t fk set ""; t fk set config --profile ""; t fk run 6 --from ""; t fk list ""
'''
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ.get("PATH", ""), PYTHONPATH=os.path.join(ROOT, "src"))
            r = subprocess.run(["bash", "-c", drive], cwd=tmp, env=env, capture_output=True, text=True)
            lines = r.stdout.splitlines()
            check(len(lines) == 5 and lines[0].split() == ["list", "run", "set", "subsession", "completion"], f"completes the subcommands ({lines[:1]})")
            check(len(lines) == 5 and lines[1] == "config", "completes `fk set` -> config")
            check(len(lines) == 5 and lines[2].split()[:3] == ["base", "exp", "best"], f"completes --profile with the profiles ({lines[2:3]})")
            check(len(lines) == 5 and "psame-chunk" in lines[3].split() and "realign-final" in lines[3].split(),
                  "completes --from with the plan's step names")
            check(len(lines) == 5 and lines[4].split()[0] == "config" and "session" in lines[4].split(), "completes `fk list` with config + steps")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_profiles: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
