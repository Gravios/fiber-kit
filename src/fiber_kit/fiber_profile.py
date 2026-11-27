#!/usr/bin/env python3
"""fiber-profile -- sample a fiber-kit command (or a running job) and summarise where the time goes.

    fiber-profile record [--out DIR] [--name NAME] [--rate HZ] [--native] [--gil] [--idle] -- CMD ...
    fiber-profile record --pid PID --duration SEC [--out DIR] [--name NAME] ...
    fiber-profile summary RUN.raw [RUN2.raw ...] [--top N] [--json]

RECORD runs CMD under py-spy (a sampling profiler: no code change, a few percent overhead) with
--subprocesses, so the ProcessPoolExecutor workers of fiber-session / fiber-refine /
fiber-stochastic are sampled too.  It writes, under --out (default ./fk-profile):
  <name>.raw   py-spy's collapsed stacks, one line per distinct stack with its sample count.  Open it in
               https://www.speedscope.app (drag the file in) for an interactive flamegraph.
  <name>.json  wall time, the CPU time and peak RSS of the command and everything it started,
               the command line, the sampling rate, the fiber-kit and Python versions.
  <name>.txt   the summary below, also printed.
With --pid it attaches to a job that is already running for --duration seconds instead (Linux
usually needs root or kernel.yama.ptrace_scope=0 for that; profiling a command you start does not).

SUMMARY reads a .raw (and the .json beside it) and reports
  * the run: wall time, CPU time, cores used on average, peak RSS;
  * the processes: the parent and its workers, seconds sampled in each, the load balance of the
    workers (an idle worker while others still run is time the pool loses);
  * what the sampled time was spent on: importing modules, waiting (on the pool, a queue, a lock),
    moving data between processes (pickling), and computing;
  * the computing by package (fiber-kit itself, numpy, scipy, sklearn, ...) and by function, and the
    fiber-kit functions that include the most time (themselves plus everything they call).

What a sample is.  py-spy looks at every Python thread `rate` times per second and records its stack;
a thread it judges idle is left out unless --idle.  So `samples / rate` is the Python thread-seconds
the job was busy.  BLAS and OpenMP threads are not Python threads: their CPU time is in the .json's
CPU figure but in no sample.  A run whose CPU time is far above its sampled seconds is spending that
difference in native threads (e.g. a multi-threaded BLAS in the parent).  --native adds the C/C++/
Cython frames below the Python ones; it costs more and needs symbols to be readable.

In a plan run, set FK_PROFILE=<dir> and fiber-pipeline profiles every step it runs this way, one
.raw/.json/.txt per step (FK_PROFILE_ARGS adds record flags, e.g. "--native --rate 200").
"""
import argparse
import datetime as _dt
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter

_FRAME = re.compile(r'^(?P<func>.*) \((?P<loc>[^()]*)\)$')
_PYLOC = re.compile(r'^(?P<path>.+):(?P<line>\d+)$')

# packages reported on their own; anything else is "stdlib" or "other"
PACKAGES = ("fiber_kit", "numpy", "scipy", "sklearn", "openTSNE", "umap", "pynndescent", "numba",
            "networkx", "skimage", "matplotlib", "PySide6", "pyqtgraph", "holoviews", "bokeh",
            "gudhi", "diptest", "yaml", "cupy", "pandas", "joblib", "threadpoolctl")
_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"<frozen importlib"}

# a leaf frame in one of these is the thread waiting, not working
_WAIT_FILES = ("threading.py", "selectors.py", "queue.py", "multiprocessing/synchronize.py",
               "multiprocessing/popen_fork.py", "multiprocessing/popen_forkserver.py",
               "multiprocessing/popen_spawn_posix.py", "multiprocessing/pool.py",
               "concurrent/futures/_base.py", "concurrent/futures/thread.py", "subprocess.py")
_IPC_FILES = ("pickle.py", "copyreg.py", "multiprocessing/reduction.py", "multiprocessing/queues.py")
ACTIVITIES = ("compute", "waiting", "pickle/IPC", "import")


# ── parsing ──────────────────────────────────────────────────────────────────────────────────────
# A process frame quotes its command line, which can itself hold ';' and '"'.  It ends at the '";'
# that is followed by another process frame or by an ordinary frame ("func (loc)").
_PROC_AT = re.compile(r'process (\d+):"(.*?)"(?:;|$)(?=process \d+:"|[^;"]* \([^()]*\)(?:;|$)|$)')
SHIM = "fiber-profile-exec"            # $0 of the shell that runs the command and keeps its exit status


def _split_stack(stack):
    """-> ([(pid, cmd), ...], [frame strings]) for one py-spy stack."""
    chain, pos = [], 0
    while stack.startswith("process ", pos):
        m = _PROC_AT.match(stack, pos)
        if not m:
            break
        chain.append((int(m.group(1)), m.group(2)))
        pos = m.end()
    rest = stack[pos:]
    return chain, [f for f in rest.split(";") if f] if rest else []


def parse_frame(f):
    """-> dict(kind='py'|'native'|'thread', func, path, line, lib)."""
    m = _FRAME.match(f)
    if not m:
        return {"kind": "native", "func": f, "path": "", "line": 0, "lib": ""}
    func, loc = m.group("func"), m.group("loc")
    if func == "thread" and loc.startswith("0x"):
        return {"kind": "thread", "func": f, "path": "", "line": 0, "lib": ""}
    p = _PYLOC.match(loc)
    if p and (".py" in p.group("path") or p.group("path").startswith("<")):
        return {"kind": "py", "func": func, "path": p.group("path"), "line": int(p.group("line")), "lib": ""}
    return {"kind": "native", "func": func, "path": "", "line": 0, "lib": loc}


def package_of(fr):
    """The package a frame belongs to: one of PACKAGES, 'stdlib', 'native' or 'other'."""
    if fr["kind"] == "native":
        return "native"
    path = fr["path"]
    if path.startswith("<frozen"):
        return "stdlib"
    parts = path.replace("\\", "/").split("/")
    for part in parts:                                   # site-packages/x/... or src/fiber_kit/...
        if part in PACKAGES:
            return part
    top = parts[0][:-3] if parts[0].endswith(".py") else parts[0]
    return "stdlib" if top in _STDLIB else "other"


def activity_of(frames):
    """compute | waiting | pickle/IPC | import, for one sampled stack (Python frames, root first)."""
    py = [f for f in frames if f["kind"] == "py"]
    if any(f["path"].startswith("<frozen importlib") or f["path"].startswith("importlib/") for f in py):
        return "import"
    if any(f["path"].endswith(_IPC_FILES) for f in py):
        return "pickle/IPC"
    if py:
        leaf = py[-1]
        if leaf["path"].endswith("multiprocessing/connection.py"):
            return "pickle/IPC" if "send" in leaf["func"] else "waiting"
        if leaf["path"].endswith(_WAIT_FILES):
            return "waiting"
    return "compute"


def parse_raw(path, shim=None):
    """-> list of (process chain [(pid, cmd), ...], frames [dict], count).

    shim: True when `fiber-profile record` ran the command under its exit-status shell (the .json
    says so); None guesses from the command line, which py-spy sometimes reads as empty."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            stack, _, cnt = line.rpartition(" ")
            try:
                n = int(cnt)
            except ValueError:
                continue
            chain, fr = _split_stack(stack)
            if chain and (shim or (shim is None and (SHIM in chain[0][1] or not chain[0][1]))):
                # the exit-status shell is not the job: drop it.  py-spy read the command's own
                # command line before the shell exec'd it, so it carries the shell's: relabel it.
                chain = [(pid, "(the profiled command)" if SHIM in cmd else cmd) for pid, cmd in chain[1:]]
            rows.append((chain, [parse_frame(f) for f in fr], n))
    return rows


# ── summary ──────────────────────────────────────────────────────────────────────────────────────
def summarise(rows, rate, meta=None, top=15):
    """The numbers the text report prints, as a dict (also what --json writes)."""
    meta = meta or {}
    rate = float(rate or meta.get("rate") or 100)
    total = sum(n for _, _, n in rows)
    root = None
    for chain, _, _ in rows:
        if chain:
            root = chain[0][0]; break
    per_proc = Counter(); proc_cmd = {}
    act = {"parent": Counter(), "workers": Counter()}
    by_pkg = Counter(); by_func = Counter(); fk_incl = Counter(); native_lib = Counter()
    for chain, frames, n in rows:
        pid, cmd = chain[-1] if chain else (root, "")
        per_proc[pid] += n; proc_cmd.setdefault(pid, cmd)
        role = "parent" if (not chain or len(chain) == 1) else "workers"
        a = activity_of(frames)
        act[role][a] += n
        if a != "compute":
            continue
        code = [f for f in frames if f["kind"] in ("py", "native")]
        if not code:
            continue
        leaf = code[-1]
        by_pkg[package_of(leaf)] += n
        if leaf["kind"] == "native":
            native_lib[leaf["lib"] or "?"] += n
            pyleaf = next((f for f in reversed(code) if f["kind"] == "py"), None)
            key = f'{leaf["func"]} [{leaf["lib"]}]' + (f' <- {pyleaf["func"]} ({pyleaf["path"]}:{pyleaf["line"]})'
                                                       if pyleaf else "")
        else:
            key = f'{leaf["func"]} ({leaf["path"]}:{leaf["line"]})'
        by_func[key] += n
        seen = set()
        for f in code:
            if f["kind"] == "py" and package_of(f) == "fiber_kit":
                k = f'{f["func"]} ({f["path"]})'
                if k not in seen:
                    seen.add(k); fk_incl[k] += n

    sec = lambda n: n / rate                                          # noqa: E731
    workers = {p: c for p, c in per_proc.items() if p != root}
    wsec = sorted(sec(c) for c in workers.values())
    wall = meta.get("wall_s")
    busy = sum(sec(act[r]["compute"] + act[r]["pickle/IPC"] + act[r]["import"]) for r in act)
    out = {
        "command": meta.get("command"), "rate_hz": rate, "samples": total,
        "sampled_thread_s": sec(total), "wall_s": wall,
        "cpu_s": meta.get("cpu_s"), "cpu_user_s": meta.get("cpu_user_s"), "cpu_sys_s": meta.get("cpu_sys_s"),
        "peak_rss_mib": meta.get("peak_rss_mib"), "cores": meta.get("cores"),
        "cpu_cores_avg": (meta["cpu_s"] / wall) if wall and meta.get("cpu_s") is not None else None,
        "busy_threads_avg": (busy / wall) if wall else None,
        "parent": {"pid": root, "cmd": proc_cmd.get(root, ""), "s": sec(per_proc.get(root, 0))},
        "workers": {"n": len(workers), "s_total": sum(wsec),
                    "s_min": wsec[0] if wsec else None, "s_median": wsec[len(wsec) // 2] if wsec else None,
                    "s_max": wsec[-1] if wsec else None,
                    "balance": (sum(wsec) / (len(wsec) * wsec[-1])) if wsec and wsec[-1] else None},
        "activity_s": {r: {a: sec(act[r][a]) for a in ACTIVITIES} for r in act},
        "compute_by_package_s": {k: sec(v) for k, v in by_pkg.most_common()},
        "native_libs_s": {k: sec(v) for k, v in native_lib.most_common(top)},
        "top_self_s": [(k, sec(v)) for k, v in by_func.most_common(top)],
        "top_fiber_kit_inclusive_s": [(k, sec(v)) for k, v in fk_incl.most_common(top)],
    }
    return out


def _dur(s):
    if s is None:
        return "?"
    s = float(s)
    if s >= 3600:
        return f"{int(s // 3600)}h {int(s % 3600 // 60):02d}m {int(s % 60):02d}s"
    if s >= 60:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{s:.2f}s" if s < 10 else f"{s:.1f}s"


def _pct(x, tot):
    return f"{100.0 * x / tot:5.1f}%" if tot else "    -"


def format_summary(s, top=15):
    L = []
    w = s["wall_s"]
    L.append(f"command   : {' '.join(s['command']) if s.get('command') else '(attached / unknown)'}")
    L.append(f"wall      : {_dur(w)}")
    if s.get("cpu_s") is not None:
        L.append(f"cpu       : {_dur(s['cpu_s'])}  (user {_dur(s['cpu_user_s'])}, sys {_dur(s['cpu_sys_s'])})"
                 + (f"  = {s['cpu_cores_avg']:.2f} cores on average of {s.get('cores') or '?'}"
                    if s.get("cpu_cores_avg") is not None else ""))
    if s.get("peak_rss_mib") is not None:
        L.append(f"peak RSS  : {s['peak_rss_mib']:.0f} MiB (largest single process)")
    L.append(f"sampled   : {s['samples']:,} samples at {s['rate_hz']:g} Hz = {_dur(s['sampled_thread_s'])} "
             "of busy Python thread time"
             + (f"; {s['busy_threads_avg']:.2f} threads busy on average" if s.get("busy_threads_avg") else ""))
    if s.get("cpu_s") and s["sampled_thread_s"] and s["cpu_s"] > 1.5 * s["sampled_thread_s"]:
        L.append(f"            note: CPU time is {s['cpu_s'] / s['sampled_thread_s']:.1f}x the sampled time -- "
                 "the rest is native threads (BLAS/OpenMP) or work py-spy judged idle")
    L.append("")
    p, wk = s["parent"], s["workers"]
    L.append("processes")
    L.append(f"  parent  pid {p['pid']}: {_dur(p['s'])} sampled")
    if wk["n"]:
        L.append(f"  workers {wk['n']}: {_dur(wk['s_total'])} sampled in all; per worker min {_dur(wk['s_min'])}, "
                 f"median {_dur(wk['s_median'])}, max {_dur(wk['s_max'])}; balance {wk['balance']:.2f} "
                 "(1.00 = every worker as busy as the busiest)")
    else:
        L.append("  workers 0 (the command ran in one process)")
    L.append("")
    L.append(f"{'activity':12s} {'parent':>12s} {'':>6s} {'workers':>12s} {'':>6s}")
    pa, wa = s["activity_s"]["parent"], s["activity_s"]["workers"]
    pt, wt = sum(pa.values()), sum(wa.values())
    for a in ACTIVITIES:
        L.append(f"  {a:10s} {_dur(pa[a]):>12s} {_pct(pa[a], pt)} {_dur(wa[a]):>12s} {_pct(wa[a], wt)}")
    L.append("")
    comp = s["compute_by_package_s"]; ct = sum(comp.values())
    L.append("computing, by package of the innermost frame")
    for k, v in list(comp.items())[:top]:
        L.append(f"  {k:14s} {_dur(v):>12s} {_pct(v, ct)}")
    if s["native_libs_s"]:
        L.append("  native, by library: " + ", ".join(f"{k} {_dur(v)}" for k, v in s["native_libs_s"].items()))
    L.append("")
    L.append(f"top {top} functions by own time (computing only)")
    for k, v in s["top_self_s"]:
        L.append(f"  {_dur(v):>10s} {_pct(v, ct)}  {k}")
    L.append("")
    L.append(f"top {top} fiber-kit functions including what they call (computing only)")
    for k, v in s["top_fiber_kit_inclusive_s"]:
        L.append(f"  {_dur(v):>10s} {_pct(v, ct)}  {k}")
    return "\n".join(L) + "\n"


def _meta_for(raw):
    j = os.path.splitext(raw)[0] + ".json"
    if os.path.exists(j):
        with open(j, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ── record ───────────────────────────────────────────────────────────────────────────────────────
def _pyspy():
    exe = shutil.which("py-spy")
    if exe:
        return exe
    cand = os.path.join(os.path.dirname(sys.executable), "py-spy")
    return cand if os.path.exists(cand) else None


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "run"


def record(a):
    exe = _pyspy()
    if not exe:
        print("fiber-profile: py-spy not found (pip install py-spy)", file=sys.stderr)
        return 2
    cmd = list(a.cmd)
    if not cmd and not a.pid:
        print("fiber-profile record: give a command after --, or --pid", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(a.out, f"{_safe_name(a.name or (os.path.basename(cmd[0]) if cmd else f'pid{a.pid}'))}-{stamp}")
    raw = base + ".raw"
    spy = [exe, "record", "--format", "raw", "--rate", str(a.rate), "--output", raw, "--subprocesses"]
    if a.native:
        spy.append("--native")
    if a.gil:
        spy.append("--gil")
    if a.idle:
        spy.append("--idle")
    if a.nonblocking:
        spy.append("--nonblocking")
    if a.pid:
        spy += ["--pid", str(a.pid)] + (["--duration", str(a.duration)] if a.duration else [])
    else:
        # py-spy exits 0 whatever the command did, and fiber-pipeline (set -e) must still stop on a
        # failed step: a shell between them keeps the command's exit status.  It is not Python, so
        # py-spy samples nothing in it; the summary drops it from the process chain.
        status = base + ".exit"
        spy += ["--", "/bin/sh", "-c", 'st=$1; shift; "$@"; echo $? > "$st"', SHIM, status] + cmd
    t0 = time.time()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    spy_rc = subprocess.call(spy)
    wall = time.time() - t0
    rc = spy_rc
    if not a.pid:
        try:
            with open(status) as fh:
                rc = int(fh.read().strip())
            os.remove(status)
        except (OSError, ValueError):
            rc = spy_rc or 1                  # the shell never wrote it: killed, or py-spy failed
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta = {
        "command": cmd or None, "pid": a.pid, "rate": a.rate, "native": a.native, "gil": a.gil,
        "idle": a.idle, "shim": not a.pid, "started": _dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "wall_s": wall, "exit": rc, "py_spy_exit": spy_rc, "cores": os.cpu_count(),
        "python": sys.version.split()[0], "fiber_kit": _version(), "host": os.uname().nodename,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("FK_", "OMP_", "OPENBLAS_", "MKL_")) and k != "FK_PROFILE_ARGS"},
    }
    if not a.pid:                     # the CPU of py-spy itself is in here too: small at 100 Hz
        meta.update(cpu_user_s=r1.ru_utime - r0.ru_utime, cpu_sys_s=r1.ru_stime - r0.ru_stime,
                    peak_rss_mib=r1.ru_maxrss / 1024.0)
        meta["cpu_s"] = meta["cpu_user_s"] + meta["cpu_sys_s"]
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    if not os.path.exists(raw) or os.path.getsize(raw) == 0:
        print(f"fiber-profile: py-spy wrote no samples (command exit {rc}, py-spy exit {spy_rc}).  If it reported a permission "
              "error: profiling a command you start needs ptrace on it -- in a container add "
              "CAP_SYS_PTRACE; attaching with --pid usually needs root.", file=sys.stderr)
        return rc or 1
    s = summarise(parse_raw(raw, meta["shim"]), a.rate, meta, top=a.top)
    txt = format_summary(s, top=a.top)
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(txt)
    print("\n== fiber-profile ==\n" + txt + f"\nwritten: {raw} (+ .json, .txt); "
          "open the .raw in https://www.speedscope.app for a flamegraph"
          + (f"\nthe command exited {rc}" if rc else ""))
    return rc


def _version():
    try:
        from . import __version__
        return __version__
    except ImportError:
        try:
            from fiber_kit import __version__
            return __version__
        except ImportError:
            return None


def summary(a):
    for raw in a.raw:
        meta = _meta_for(raw)
        s = summarise(parse_raw(raw, meta.get("shim")), a.rate or meta.get("rate"), meta, top=a.top)
        if a.json:
            print(json.dumps(s, indent=1))
        else:
            if len(a.raw) > 1:
                print(f"== {raw} ==")
            print(format_summary(s, top=a.top))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fiber-profile",
        description="Sample a fiber-kit command (or a running job) with py-spy, workers included, and "
                    "summarise where the time goes.")
    ap.add_argument("cmd", choices=("record", "summary"), help="record a run, or summarise a .raw")
    ap.add_argument("raw", nargs="*", metavar="RAW",
                    help="summary: one or more .raw files (record: the command goes after --)")
    ap.add_argument("--out", default="fk-profile", help="record: output directory (default ./fk-profile)")
    ap.add_argument("--name", default=None, help="record: file name stem (default the command's name); "
                                                 "a time stamp is appended")
    ap.add_argument("--rate", type=int, default=100, help="samples per second (default 100)")
    ap.add_argument("--native", action="store_true", help="record: include C/C++/Cython frames")
    ap.add_argument("--gil", action="store_true", help="record: only samples holding the GIL")
    ap.add_argument("--idle", action="store_true", help="record: include idle threads")
    ap.add_argument("--nonblocking", action="store_true",
                    help="record: do not pause the process to sample (less overhead, less exact)")
    ap.add_argument("--pid", type=int, default=None, help="record: attach to a running process")
    ap.add_argument("--duration", type=int, default=None, help="record --pid: seconds to sample")
    ap.add_argument("--top", type=int, default=15, help="rows per table (default 15)")
    ap.add_argument("--json", action="store_true", help="summary: print the numbers as JSON")
    argv = list(sys.argv[1:] if argv is None else argv)
    tail = []
    if "--" in argv:                          # everything after -- is the command to profile
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if a.cmd == "record":
        if a.raw:
            ap.error("record: put the command after --  (fiber-profile record [options] -- CMD ...)")
        a.cmd = tail
        return record(a)
    if tail or not a.raw:
        ap.error("summary needs at least one .raw file")
    missing = [x for x in a.raw if not os.path.exists(x)]
    if missing:
        ap.error("no such file: " + ", ".join(missing))
    a.rate = a.rate if "--rate" in argv else None
    return summary(a)


if __name__ == "__main__":
    sys.exit(main())
