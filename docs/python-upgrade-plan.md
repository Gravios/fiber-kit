# Python version upgrade plan (3.12 → 3.15)

A **design plan**, written 2026-11-27. Nothing here is built and nothing is scheduled. The review
found too little gain for the effort to move now; this file holds what it found, so the move can be
made quickly once one of the triggers in § 5 is met.

**Target: Python 3.15**, skipping 3.14 as a stop. 3.15 (released 2026-10-09) has the two changes that
matter most to fiber-kit: lazy imports for the `fiber-*` start-up, and a sampling profiler that
attaches to a running job. Everything 3.14 added comes with it. 3.14 stays in the CI matrix only
as an intermediate check (step 3).

Where fiber-kit stands today:

| | value |
|---|---|
| `requires-python` (pyproject.toml) | `>=3.9` |
| README badge / requirements line | `python ≥3.9` |
| CI (`.github/workflows/ci.yml`) | one job, Python 3.12, ubuntu-24.04 |

## 1. What 3.15 (and 3.14) offer fiber-kit

Ranked by what it would change for fiber-kit, not by how much it changes Python. None of it speeds
up the numpy/BLAS work, which is where the run time goes.

### Most relevant (3.15)

**Lazy imports (PEP 810).** Each `fiber-*` command imports numpy, scipy, sklearn and often
matplotlib/Qt before it parses its arguments, so `fiber-* -h`, a mistyped flag and every short
stage pay the full import cost.

- `lazy import x` / `lazy from x import y` defers the load to the first use of the name.
- It can be measured without editing code: `-X lazy_imports=all` or `PYTHON_LAZY_IMPORTS=all`
  makes every import lazy; `none` turns it off.
- `__lazy_modules__ = ["matplotlib", "sklearn", ...]` at the top of a module makes the plain
  `import` statements for those names lazy. Older Pythons ignore the attribute, so this needs
  **no floor change** and no version check (the `lazy` keyword itself is a `SyntaxError` before 3.15).
- Restrictions: `lazy` is allowed at module scope only, not inside a function, class body or
  `try`/`except`. fiber-kit's many function-local imports are already deferred and need nothing.
  The `try: from . import x` / `except ImportError: import x` fallback at the top of most stage
  modules cannot use the keyword; check whether `__lazy_modules__` covers imports in a `try` block.
- Risks: an import error now surfaces at first use, not at start-up. A stage would then fail
  partway through a run on a missing optional dependency, so keep the `diptest`, `cupy` and Qt
  checks eager. Any module that registers something when imported (stage lists, Qt plugins,
  matplotlib backends) must stay eager, or the registration silently never happens.

**Tachyon sampling profiler (`profiling.sampling`, PEP 799).** It profiles a running process with
near-zero overhead, with no code change or restart.
Most of this is available before any upgrade: `fiber-profile` (py-spy, fiber-kit 0.71.22) records
and attaches on 3.12/3.13, follows the pool workers and has `--gil`; see [profiling.md](profiling.md).
What Tachyon adds is being in the standard library, the line heatmap and higher rates; it is no
reason to move on its own.

- `attach <PID>`: profile a `fiber-session` or `fiber-pipeline` that is already running.
- `dump`: one snapshot of every thread's stack; the first thing to run on a job that looks stuck.
- `--mode gil`: time spent holding the GIL. This is the measurement that decides whether the
  free-threaded build (step 7) can pay off at all; `--mode cpu` separates compute from waiting on I/O.
- `-a` (all threads), for the thread pools in `fiber_refine` and `morpho_localize`.
- Output as `--flamegraph` / `--heatmap` (self-contained HTML), `--collapsed` (speedscope), `--pstats`.
- Not yet checked: whether it follows `ProcessPoolExecutor` children. If not, attach to each
  worker PID.

**Clearer worker deaths.** When a `ProcessPoolExecutor` child dies abruptly, the traceback now gives
the child's PID and exit code. An OOM kill (exit -9) in a chunk pool can then be told apart from a
crash without reading `dmesg`.

**Stable ABI for free-threaded builds (`abi3t`, PEP 803/793/820).** One wheel then serves every
free-threaded Python, instead of a `cp315t` wheel per version. It doesn't change fiber-kit, which is
pure Python, but it makes free-threaded wheels for gudhi, openTSNE, diptest, etc. cheaper to ship.
This is the thing to watch for step 7.

### Useful once fiber-kit can require 3.14 or later

| feature (since) | where it applies in fiber-kit |
|---|---|
| `Executor.map(fn, it, buffersize=N)` (3.14) | `fiber_session.py` (`_process_chunk`), `fiber_refine.py` (`_refine_one_chunk`), `fiber_stochastic.py` (`_draw_worker`) all call `ex.map` over every task at once; `buffersize` limits how many are in flight, which bounds the parent's memory on long sessions. |
| `ProcessPoolExecutor.terminate_workers()` / `kill_workers()` (3.14) | the same three pools: shut down cleanly on Ctrl-C or on a failed chunk, with no orphaned workers. |
| `frozendict` (3.15) | the `cfg` handed to pool initializers: immutable, so a worker cannot change shared configuration, and hashable (with hashable values), so it can be a cache key. It is not a `dict` subclass: check code that tests `isinstance(cfg, dict)`. |
| `sentinel` (3.15) | "no value given" defaults that survive pickling to pool workers (pickles when importable by name). |
| `heapq.heappush_max`, `heappop_max`, `heapify_max` (3.14) | the merge heaps in `fiber_polish`, `fiber_residual_merge` and `fiber_intrachunk` push `(-score, a, b, ...)`. **Not a drop-in swap:** a max-heap on `(score, a, b)` also reverses the tie-break on the ids, so merge order (and output) can change on tied scores. Only with a test that pins the order. |
| `compression.zstd` (3.14) | an option for compressed caches/intermediates with no new dependency. Not needed today. |

### Useful just from running on it (no floor change)

- argparse "did you mean" suggestions: on by default in 3.15 (`suggest_on_error`), opt-in in 3.14,
  for all ~30 `fiber-*` entry points.
- `python -m pdb -p <PID>` (3.14): attach a debugger to a running job.
- Frame pointers on by default (3.15, PEP 831): cleaner `perf` profiles through the C/BLAS layers.
- Better error messages; a slightly faster interpreter and JIT. Expect no measurable speed-up for
  fiber-kit.

### Changes to check (not benefits)

- **UTF-8 is the default encoding** (3.15, PEP 686): `open()` with no `encoding=` uses UTF-8 whatever
  the locale. Harmless for the ASCII `.clu`/`.res`/`.fet`/YAML files, and it removes a locale
  dependency. `PYTHONUTF8=0` reverts it.
- **Garbage collector:** 3.15 goes back to the 3.13 generational GC (3.14.0–3.14.4's incremental GC
  was reverted), so memory behaviour on 3.15 is closer to 3.12 than 3.14's was.
- **Removals:** the full "Removed" and "Deprecated" sections of What's New in 3.15 have to be read
  before step 3. They were not reviewed in full for this plan; `__cached__` is no longer set (use
  `__spec__.cached`).
- **The process-pool start method** changed in 3.14 and still applies: § 2.

### Not useful for fiber-kit (recorded so it is not re-evaluated)

- **Sub-interpreters** (`concurrent.interpreters`, `InterpreterPoolExecutor`): numpy does not
  support multiple interpreters.
- **JIT, tail-call interpreter**: gains on pure-Python code only.
- **Template strings, deferred annotations, unpacking in comprehensions, new typing features**:
  no use in this code.

### Longer term: the free-threaded build

The only change that could alter fiber-kit's design. The chunk pools could become thread pools that
share the memmaps, `cfg` and the waveform arrays, with no pickling and no per-worker numpy import.
Two things block it:

1. **Dependencies.** Every package in `dependencies` needs free-threaded wheels. numpy, scipy and
   scikit-learn have them; PySide6, gudhi, openTSNE, diptest and umap-learn (through numba) are
   the ones to check. `abi3t` (above) should make these more likely.
2. **Payoff.** The heavy work is already in BLAS/numpy and releases the GIL, so the gain is start-up
   time and memory, not throughput. Measure first: Tachyon `--mode gil` on a real run shows how much
   time the GIL is actually held, and `fiber-bench` must show the gain before anything is rewritten.

## 2. The one behaviour change to handle first

**On Linux, since Python 3.14, the default start method for `multiprocessing` and
`ProcessPoolExecutor` is `forkserver`, not `fork`.** fiber-kit would change behaviour on 3.14 or
3.15 without a line of its code changing. Three pools set no start method:

| pool | file:line (0.71.21) | initializer |
|---|---|---|
| chunk clustering | `src/fiber_kit/fiber_session.py:2766` | `_init_pool_worker(cfg)` |
| chunk refinement | `src/fiber_kit/fiber_refine.py:1109` | `_init_refine_pool_worker(cfg)` |
| stochastic draws | `src/fiber_kit/fiber_stochastic.py:272` | `fsess._init_pool_worker(cfg)` |

The pools that already choose a method are unaffected: `morpho_localize.py` (`fork`, `spawn`
fallback), `fiber_manifold.py` (`fork`), `fiber_consensus.py` (`spawn`).

What changes under `forkserver`:

- each worker imports numpy/scipy/fiber_kit itself instead of inheriting them (start-up cost at
  64 workers; see the `morpho_localize` comment on `spawn`). Lazy imports (§ 1) reduce this cost;
- `initargs=(cfg,)` and each task are pickled, so everything in `cfg` must pickle;
- worker functions must be importable by module path. Functions defined in `__main__` of a
  `tools/` script, or code run from stdin, fail;
- module-level state set in the parent after import is **not** seen by the worker unless it comes
  through the initializer. `fiber_stochastic._POOL_CFG` is set at run time but handed over as
  `initargs`, so it is safe; any new global must follow the same pattern.

The upside: `forkserver` removes the "BLAS/openTSNE threads inherited at fork" hazard that pushed
`fiber_consensus` to `spawn`.

**Decision to take at step 1 of § 3:** pass `mp_context` to all three pools explicitly, so the start
method does not depend on the Python version. Either keep `fork` (no change in behaviour; the BLAS
pin already runs in each initializer), or move to `forkserver` everywhere on purpose and test it on
3.12 first. Recommendation: explicit `fork` now, with a `FK_MP_START` knob (default `fork`) so
`forkserver` can be benchmarked without a code change.

## 3. Steps

Each step is a separate change with its own changelog entry; each can stop there. Steps 1, 2 and 4
are worth doing on 3.12 and need no new Python.

1. **Pin the start method** (§ 2). Pure fiber-kit change; works on every supported Python.
   Test: `test/run_all.py` with `FK_MP_START=fork` and `=forkserver`; one `fiber-session --jobs 8`
   run on the reference session compared byte-for-byte with the current output.
2. **Make the stated floor true.** `requires-python = ">=3.9"` is no longer honest: 3.9 and 3.10 are
   end-of-life (3.10.22 was the last 3.10 release), and the current numpy/scipy/networkx no longer
   install on them, so pip picks years-old versions. Raise to the oldest version CI actually tests
   (`>=3.11` or `>=3.12`), and update the README badge and requirements line to match.
3. **Add 3.15 (and 3.14) to CI next to 3.12** (a matrix; the new versions allowed to fail at first).
   This is where missing wheels show up: gudhi, openTSNE, diptest, umap-learn/numba, PySide6,
   pyqtgraph, and on the GPU side `cupy-cuda12x` / `cuml-cu12`. numba is usually the last to support a
   new Python (≥ 0.63 for 3.14; check for 3.15). If 3.15 is blocked but 3.14 passes, 3.14 can be
   the interim target. numpy 1.x has no wheels for either; the `numpy>=1.21` floor is fine but those
   users get numpy 2.x, which the tests must pass with. Read the 3.15 removals (§ 1) here.
4. **Lazy imports, measured first.** On 3.15, time `fiber-session -h`, `fiber-spkc -h` and one short
   stage with `PYTHON_LAZY_IMPORTS=all` against `normal`. If the gain is real, add
   `__lazy_modules__` to the entry modules for the heavy, side-effect-free packages (sklearn,
   matplotlib, holoviews/bokeh, openTSNE, umap, gudhi, networkx, skimage). It works on 3.12 as a no-op,
   so it can ship before the floor moves. Keep eager anything that registers on import or that
   must fail early. Test: `test/run_all.py` under both `PYTHON_LAZY_IMPORTS` values.
5. **Run the workstations on 3.15** for a few sessions (no floor change). Compare `fiber-bench` scores
   and wall time against 3.12 on the reference session, and profile one long `fiber-session` to record where the time goes
   (`fiber-profile`, with and without `--gil`; Tachyon once on 3.15).
6. **Raise the floor to 3.15** only to use the § 1 APIs and the `lazy` keyword without version
   checks. Until then the APIs can be guarded:
   ```python
   kw = {"buffersize": 2 * nworkers} if sys.version_info >= (3, 14) else {}
   for result in ex.map(_process_chunk, tasks, **kw):
   ```
7. **(Separate project) Free-threaded evaluation.** Only after every dependency ships free-threaded
   wheels (`cp315t` or `abi3t`), and only if the step 5 GIL profile shows time to win: a
   `fiber-session` thread-pool branch behind a knob, benchmarked against the process pool for wall
   time and peak RSS. Adopt only on a clear win.

## 4. Cross-repo notes

- neurosuite-3 is C++/Qt; its Python parts (ndmanager-plugins, scripts) should follow the same
  floor so one environment serves both. `test_shared_contract` runs in fiber-kit's CI, so a CI
  matrix covers it.
- `cupy-cuda12x` (`gpu` extra) and `cuml-cu12` (off-PyPI): check their 3.15 wheels before step 5 on
  the GPU workstations.

## 5. When to revisit

Revisit this plan when any of the following happens; otherwise review it once a year, after each
October CPython release (next: 3.16, October 2027):

- every package in `dependencies` has 3.15 wheels (→ steps 3–5);
- a dependency drops 3.12 (look first at numpy, scipy, PySide6, numba);
- the workstations' OS image moves to 3.14 or later;
- CLI start-up time becomes a nuisance (→ step 4; the measurement needs a 3.15 interpreter only);
- a long run needs profiling or hangs (→ a 3.15 interpreter for Tachyon `attach`/`dump`);
- memory from eagerly submitted chunk tasks becomes a problem (→ guard `buffersize`, step 6);
- every dependency has free-threaded wheels (→ step 7);
- **3.12 approaches end-of-life (October 2028)**: the move must be done by then.
