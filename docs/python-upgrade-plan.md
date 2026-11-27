# Python version upgrade plan (3.12 → 3.14 and later)

A **design plan**, written 2026-11-27. Nothing here is built and nothing is scheduled. The 3.14
review found too little gain for the effort to move now; this file holds what that review found,
so the move can be made quickly once one of the triggers in § 5 is met.

Where fiber-kit stands today:

| | value |
|---|---|
| `requires-python` (pyproject.toml) | `>=3.9` |
| README badge / requirements line | `python ≥3.9` |
| CI (`.github/workflows/ci.yml`) | one job, Python 3.12, ubuntu-24.04 |

## 1. What a newer Python offers fiber-kit

Ranked by what it would change for fiber-kit, not by how much it changes Python.

### Useful once fiber-kit can require it (3.14)

| feature | where it applies in fiber-kit |
|---|---|
| `Executor.map(fn, it, buffersize=N)` | `fiber_session.py` (`_process_chunk`), `fiber_refine.py` (`_refine_one_chunk`), `fiber_stochastic.py` (`_draw_worker`) all call `ex.map` over every task at once; `buffersize` limits how many are in flight, which bounds the parent's memory on long sessions. |
| `ProcessPoolExecutor.terminate_workers()` / `kill_workers()` | the same three pools: shut down cleanly on Ctrl-C or on a failed chunk, with no orphaned workers. |
| `heapq.heappush_max`, `heappop_max`, `heapify_max` | the merge heaps in `fiber_polish`, `fiber_residual_merge` and `fiber_intrachunk` push `(-score, a, b, ...)`. **Not a drop-in swap:** a max-heap on `(score, a, b)` also reverses the tie-break on the ids, so merge order (and output) can change on tied scores. Only with a test that pins the order. |
| argparse `suggest_on_error=True`, `color=True` | the ~30 `fiber-*` entry points: "did you mean" on mistyped flags. |
| `compression.zstd` in the standard library | an option for compressed caches/intermediates with no new dependency. Not needed today. |

### Useful just from running on it (no floor change)

- `python -m pdb -p <PID>`: attach a debugger to a running `fiber-session` / `fiber-pipeline`
  job, for a multi-hour run that looks stuck.
- Better error messages; a slightly faster interpreter. The hot loops are numpy/BLAS, so expect
  no measurable speed-up.

### Not useful for fiber-kit (recorded so it is not re-evaluated)

- **Sub-interpreters** (`concurrent.interpreters`, `InterpreterPoolExecutor`): numpy does not
  support multiple interpreters.
- **Experimental JIT, tail-call interpreter**: a few percent on pure-Python code only.
- **Template strings, deferred annotations**: no use in this code.

### Longer term: the free-threaded build (`3.14t` and later)

The only change that could alter fiber-kit's design. The chunk pools could become thread pools that
share the memmaps, `cfg` and the waveform arrays, with no pickling and no per-worker numpy import.
Two things block it:

1. **Dependencies.** Every package in `dependencies` needs free-threaded wheels. numpy, scipy and
   scikit-learn have them; PySide6, gudhi, openTSNE, diptest and umap-learn (through numba) are
   the ones to check.
2. **Payoff.** The heavy work is already in BLAS/numpy and releases the GIL, so the gain is startup
   time and memory, not throughput. It has to be measured with `fiber-bench` before anything is
   rewritten.

### Python 3.15 (released 2026-10-09): worth a look at the next review

- **Lazy imports.** Each `fiber-*` command imports numpy, scipy, sklearn and often matplotlib/Qt at
  start-up; `fiber-* -h` and short stages pay for that every call. This may be a larger practical
  gain than anything in 3.14. Evaluate at the next review (§ 5).
- Frame pointers on by default: better `perf` profiles of the native parts of the pipeline.

## 2. The one behaviour change to handle first

**On Linux, Python 3.14 changes the default start method for `multiprocessing` and
`ProcessPoolExecutor` from `fork` to `forkserver`.** fiber-kit would change behaviour without a
line of its code changing. Three pools set no start method:

| pool | file:line (0.71.21) | initializer |
|---|---|---|
| chunk clustering | `src/fiber_kit/fiber_session.py:2766` | `_init_pool_worker(cfg)` |
| chunk refinement | `src/fiber_kit/fiber_refine.py:1109` | `_init_refine_pool_worker(cfg)` |
| stochastic draws | `src/fiber_kit/fiber_stochastic.py:272` | `fsess._init_pool_worker(cfg)` |

The pools that already choose a method are unaffected: `morpho_localize.py` (`fork`, `spawn`
fallback), `fiber_manifold.py` (`fork`), `fiber_consensus.py` (`spawn`).

What changes under `forkserver`:

- each worker imports numpy/scipy/fiber_kit itself instead of inheriting them (start-up cost at
  64 workers; see the `morpho_localize` comment on `spawn`);
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

Each step is a separate change with its own changelog entry; each can stop there.

1. **Pin the start method** (§ 2). Pure fiber-kit change; works on every supported Python.
   Test: `test/run_all.py` with `FK_MP_START=fork` and `=forkserver`; one `fiber-session --jobs 8`
   run on the reference session compared byte-for-byte with the current output.
2. **Make the stated floor true.** `requires-python = ">=3.9"` is no longer honest: 3.9 and 3.10 are
   end-of-life (3.10.22 was the last 3.10 release), and the current numpy/scipy/networkx no longer
   install on them, so pip picks years-old versions. Raise to the oldest version CI actually tests
   (`>=3.11` or `>=3.12`), and update the README badge and requirements line to match.
3. **Add 3.14 to CI next to 3.12** (a matrix; 3.14 allowed to fail at first). This is where missing
   wheels show up: gudhi, openTSNE, diptest, umap-learn/numba (numba ≥ 0.63 is the first with 3.14),
   PySide6, pyqtgraph. numpy 1.x has no 3.14 wheels; the `numpy>=1.21` floor is fine but 3.14 users
   get numpy 2.x, which the tests must pass with.
4. **Run the workstations on 3.14** for a few sessions (no floor change). Compare `fiber-bench` scores
   and wall time against 3.12 on the reference session.
5. **Raise the floor to 3.14** only to use the § 1 APIs without version checks. Until then each can
   be guarded:
   ```python
   kw = {"buffersize": 2 * nworkers} if sys.version_info >= (3, 14) else {}
   for result in ex.map(_process_chunk, tasks, **kw):
   ```
6. **(Separate project) Free-threaded evaluation.** Only after every dependency ships `cp31Xt` wheels:
   a `fiber-session` thread-pool branch behind a knob, benchmarked against the process pool for wall
   time and peak RSS. Adopt only on a clear win.

## 4. Cross-repo notes

- neurosuite-3 is C++/Qt; its Python parts (ndmanager-plugins, scripts) should follow the same
  floor so one environment serves both. `test_shared_contract` runs in fiber-kit's CI, so a CI
  matrix covers it.
- `cupy-cuda12x` (`gpu` extra) and `cuml-cu12` (off-PyPI): check their 3.14 wheels before step 4 on
  the GPU workstations.

## 5. When to revisit

Revisit this plan when any of the following happens; otherwise review it once a year, after each
October CPython release:

- a dependency drops 3.12 (look first at numpy, scipy, PySide6, numba);
- the workstations' OS image moves to 3.14 or later;
- memory from eagerly submitted chunk tasks becomes a problem (→ step 5 for `buffersize`, or guard it);
- every dependency has free-threaded wheels (→ step 6);
- CLI start-up time becomes a nuisance (→ evaluate 3.15 lazy imports);
- **3.12 approaches end-of-life (October 2028)**: the move must be done by then.
