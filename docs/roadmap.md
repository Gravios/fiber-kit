# fiber-kit roadmap

Planned work that is not yet scheduled. Each item points to its design document; an item moves to
the [CHANGELOG](../CHANGELOG.md) when it ships. Items are not in priority order.

| item | status | plan |
|---|---|---|
| High-density probes (Neuropixels): tiles, spatial-profile assignment, drift field | design, nothing built | [neuropixels-plan.md](neuropixels-plan.md) |
| Python version upgrade, target 3.15 | deferred, 2026-11-27 | [python-upgrade-plan.md](python-upgrade-plan.md) |

## Python version upgrades

**Status:** deferred (2026-11-27). **Target: Python 3.15**; 3.14 only as an interim stop if a
dependency blocks 3.15. The review found no speed gain for fiber-kit (the work is in numpy/BLAS). The
gains are faster `fiber-*` start-up (lazy imports), a profiler that attaches to a running job, and
small pool/CLI APIs. None of that justifies the move yet.

**Order when picked up** (each step stands alone; details in the plan):

1. Pass `mp_context` explicitly to the three process pools (`fiber_session`, `fiber_refine`,
   `fiber_stochastic`). Since 3.14 the Linux default is `forkserver`, not `fork`; this step makes
   fiber-kit's behaviour independent of the Python version. **Do this before anyone runs fiber-kit
   on 3.14 or 3.15.**
2. Make `requires-python` true: raise it from `>=3.9` (end-of-life, and no longer installable with
   current dependencies) to the oldest tested version.
3. Add 3.15 and 3.14 to CI next to 3.12, to find missing wheels (numba first).
4. Measure lazy imports on 3.15 (`PYTHON_LAZY_IMPORTS=all`); if worth it, add `__lazy_modules__` to
   the entry modules. It is a no-op on older Pythons, so it can ship before the floor moves.
5. Run the workstations on 3.15; compare with `fiber-bench`; profile one long run with Tachyon
   (`--mode gil` decides step 7).
6. Raise the floor to 3.15 to use the `lazy` keyword, `Executor.map(buffersize=)`,
   `terminate_workers()`, `frozendict` and `sentinel` without version checks.
7. Later, a separate project: evaluate the free-threaded build (thread pools sharing the memmaps),
   once every dependency ships free-threaded wheels and step 5 shows time spent holding the GIL.

**Revisit when:** every dependency has 3.15 wheels; a dependency drops 3.12; the workstations move
to a newer Python; CLI start-up or a long run that needs profiling becomes a problem; or at the
yearly review after each October CPython release (next: 3.16, October 2027). **Hard deadline:**
before Python 3.12 reaches end-of-life in October 2028.
