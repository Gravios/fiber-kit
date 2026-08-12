#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  run_blocks.py — run test_morpho.py one BLOCK at a time, in parallel.
#
#  The suite is a single script of numbered blocks that has grown past the point
#  where it finishes in one process on a modest machine.  When it stops
#  finishing, the practical response is to verify a subset and ship -- which is
#  how several defects reached a user in this project: a console script that was
#  never registered, a resolver that rejected a valid session, a worker pool that
#  crashed on launch.  Each would have been caught by a suite that ran.
#
#  So this runner does three things the monolith cannot:
#
#    ISOLATION   each block runs in its own process, so one hang or segfault
#                costs that block and not the run.
#    TIMEOUTS    a block that exceeds --timeout is reported as TIMEOUT rather
#                than silently killing everything after it.
#    PARALLELISM blocks are independent, so on a many-core machine the wall time
#                is the slowest block rather than their sum.
#
#  It does NOT modify test_morpho.py.  Blocks are extracted by their `# ── N.`
#  markers and recombined with the file's shared header, so the suite stays
#  runnable directly with `python3 test/test_morpho.py` and there is no second
#  source of truth about what the tests are.
#
#  The cost of that choice is that every block re-runs the header.  That is
#  deliberate: a header cheap enough to repeat is also a header that cannot
#  leak state between blocks, and cross-block state is what makes a monolithic
#  suite fail in ways that depend on execution order.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUITE = os.path.join(HERE, "test_morpho.py")
MARKER = re.compile(r"^# ── (\d+)\.\s*(.*?)\s*─*\s*$", re.M)
SUMMARY = 'print(f"\\n{ran - fails}/{ran} checks passed")'
COUNT = re.compile(r"^(\d+)/(\d+) checks passed", re.M)


def split_blocks(path):
    """(header, [(number, title, source)]) for a suite of `# ── N.` blocks.

    Refuses a suite with no markers rather than running it as one block: that
    would silently reproduce the monolith this exists to break up, and the
    caller would see a green run with none of the isolation they asked for.
    """
    src = open(path).read()
    marks = list(MARKER.finditer(src))
    if not marks:
        raise SystemExit(f"[rig] no `# ── N. title ──` blocks found in {path}")
    try:
        end = src.rindex(SUMMARY)
    except ValueError:
        raise SystemExit(f"[rig] {path} has no final summary line to anchor on; "
                         f"expected {SUMMARY!r}")
    header = src[:marks[0].start()]
    blocks = []
    for i, m in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else end
        blocks.append((int(m.group(1)), m.group(2), src[m.start():stop]))
    return header, blocks


IMPORT = re.compile(r"^\s*(?:import\s+\S+(?:\s+as\s+\w+)?|from\s+\S+\s+import\s+[^\n(]+)$",
                    re.M)


def hoist_imports(blocks):
    """Every import that appears in ANY block, as a preamble for EVERY block.

    Blocks turned out not to be independent: several used names imported by an
    earlier block (`_tf`, `_ms`, `mlz`), which works only because the monolith
    runs them in order.  That is exactly the hidden coupling isolation is meant
    to expose, and the fix belongs here rather than in 35 edits to the suite --
    imports are idempotent, so replaying them all costs nothing and removes the
    ordering dependency entirely.

    Each is wrapped in try/except so an import that is itself conditional (a
    module only present in some environments) does not break unrelated blocks.
    Only import statements are hoisted; a block that depends on a VALUE another
    block computed will still fail, and should -- that is a real defect in the
    suite, not something a runner should hide.
    """
    seen, out = set(), []
    for _, _, body in blocks:
        for m in IMPORT.finditer(body):
            line = m.group(0).strip()
            if line not in seen:
                seen.add(line)
                out.append(f"try:\n    {line}\nexcept Exception:\n    pass")
    return "\n".join(out) + "\n" if out else ""


def run_one(header, num, title, body, timeout, cwd, keep_dir=None, preamble=""):
    """Run one block in its own process.  Returns a result dict."""
    fd, path = tempfile.mkstemp(prefix=f"blk{num:02d}_", suffix=".py",
                                dir=keep_dir)
    with os.fdopen(fd, "w") as fh:
        fh.write(header + preamble + body + "\n" + SUMMARY + "\n")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, path], cwd=cwd, timeout=timeout,
                           capture_output=True, text=True)
        out, rc = p.stdout + p.stderr, p.returncode
        status = "ok" if rc == 0 else "ERROR"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        status, rc = "TIMEOUT", -1
    finally:
        if keep_dir is None:
            os.unlink(path)
    dt = time.time() - t0
    m = COUNT.search(out)
    passed, total = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    # A block whose process succeeded but reported fewer passes than checks has
    # failing assertions; one that produced no count at all died before the
    # summary, which is a different fault and must not read as "0 failures".
    if status == "ok" and not m:
        status = "NO SUMMARY"
    if status == "ok" and passed != total:
        status = "FAIL"
    fails = [ln for ln in out.splitlines() if "FAIL" in ln]
    return dict(num=num, title=title, status=status, passed=passed, total=total,
                seconds=dt, rc=rc, fail_lines=fails, output=out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="run_blocks",
        description="Run a numbered test suite one block per process, in parallel.")
    ap.add_argument("--suite", default=DEFAULT_SUITE)
    ap.add_argument("--block", action="append", type=int, default=None,
                    help="run only this block; repeatable")
    ap.add_argument("--from-block", type=int, default=None, dest="from_block")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel blocks; 0 = all cores, 1 = serial")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="seconds per block before it is reported TIMEOUT")
    ap.add_argument("--list", action="store_true", help="list blocks and exit")
    ap.add_argument("--verbose", action="store_true",
                    help="print each block's full output")
    ap.add_argument("--keep", default=None,
                    help="write the generated per-block scripts here and keep them")
    a = ap.parse_args(argv)

    header, all_blocks = split_blocks(a.suite)
    # Hoist from EVERY block, not just the selected ones: an import that lives in
    # a block the caller did not select is still needed by one that they did, and
    # computing this after filtering made `--block 31` fail on a name defined in
    # block 6.  Selection must not change what a block sees.
    preamble = hoist_imports(all_blocks)
    blocks = list(all_blocks)
    if a.block:
        want = set(a.block)
        blocks = [b for b in blocks if b[0] in want]
        missing = want - {b[0] for b in blocks}
        if missing:
            raise SystemExit(f"[rig] no such block(s): {sorted(missing)}")
    if a.from_block is not None:
        blocks = [b for b in blocks if b[0] >= a.from_block]
    if a.list:
        for n, t, body in blocks:
            print(f"  {n:3d}  {t}  ({len(body.splitlines())} lines)")
        return 0
    if not blocks:
        raise SystemExit("[rig] no blocks selected")

    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    jobs = a.jobs or (os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(blocks)))
    cwd = os.path.dirname(os.path.abspath(a.suite)) or "."
    print(f"[rig] {len(blocks)} block(s), {jobs} parallel, "
          f"{a.timeout:.0f}s timeout each")

    t0 = time.time()
    # Threads, not processes: each task is a subprocess.run that releases the
    # GIL while it waits, so the pool only needs to supervise, and a thread pool
    # avoids re-importing this module in every worker.
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(
            lambda b: run_one(header, b[0], b[1], b[2], a.timeout, cwd, a.keep,
                              preamble),
            blocks))
    wall = time.time() - t0

    results.sort(key=lambda r: r["num"])
    print(f"\n{'blk':>4} {'status':<10} {'checks':>9} {'time':>8}  title")
    bad = []
    for r in results:
        checks = f"{r['passed']}/{r['total']}" if r["total"] else "-"
        print(f"{r['num']:>4} {r['status']:<10} {checks:>9} {r['seconds']:>7.1f}s  "
              f"{r['title'][:44]}")
        if r["status"] != "ok":
            bad.append(r)

    ok = sum(1 for r in results if r["status"] == "ok")
    tot = sum(r["total"] for r in results)
    psd = sum(r["passed"] for r in results)
    slow = max(results, key=lambda r: r["seconds"])
    print(f"\n{ok}/{len(results)} blocks ok, {psd}/{tot} checks passed, "
          f"{wall:.0f}s wall (slowest: block {slow['num']} at {slow['seconds']:.0f}s)")

    for r in bad:
        print(f"\n─── block {r['num']} [{r['status']}] {r['title']}")
        if a.verbose or r["status"] in ("ERROR", "NO SUMMARY", "TIMEOUT"):
            tail = r["output"].strip().splitlines()[-25:]
            print("\n".join("    " + ln for ln in tail))
        else:
            for ln in r["fail_lines"][:10]:
                print("    " + ln.strip())
    if a.verbose:
        for r in results:
            if r["status"] == "ok":
                print(f"\n─── block {r['num']} {r['title']}")
                print(r["output"].rstrip())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
