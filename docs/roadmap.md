# fiber-kit roadmap

Planned work that is not yet scheduled. Each item points to its design document; an item moves to
the [CHANGELOG](../CHANGELOG.md) when it ships. Items are not in priority order.

| item | status | plan |
|---|---|---|
| High-density probes (Neuropixels): tiles, spatial-profile assignment, drift field | design, nothing built | [neuropixels-plan.md](neuropixels-plan.md) |
| Python version upgrades (3.14 and later) | deferred, 2026-11-27 | [python-upgrade-plan.md](python-upgrade-plan.md) |

## Python version upgrades

**Status:** deferred. The 3.14 review (2026-11-27) found no speed gain for fiber-kit (the work is in
numpy/BLAS) and only small API improvements, which don't justify the move yet.

**Order when picked up** (each step stands alone; details in the plan):

1. Pass `mp_context` explicitly to the three process pools (`fiber_session`, `fiber_refine`,
   `fiber_stochastic`). Python 3.14 changes the Linux default from `fork` to `forkserver`; this
   step makes fiber-kit's behaviour independent of the Python version. **Do this before anyone runs
   fiber-kit on 3.14.**
2. Make `requires-python` true: raise it from `>=3.9` (end-of-life, and no longer installable with
   current dependencies) to the oldest tested version.
3. Add Python 3.14 to CI next to 3.12, to find missing wheels.
4. Run the workstations on 3.14; compare with `fiber-bench`.
5. Raise the floor to 3.14 to use `Executor.map(buffersize=)`, `terminate_workers()`,
   `heapq.*_max` and argparse suggestions.
6. Later, a separate project: evaluate the free-threaded build (thread pools sharing the memmaps),
   once every dependency ships free-threaded wheels.

**Revisit when:** a dependency drops 3.12; the workstations move to a newer Python; every dependency
has free-threaded wheels; or at the yearly review after each October CPython release (next: look at
3.15 lazy imports for CLI start-up). **Hard deadline:** before Python 3.12 reaches end-of-life in
October 2028.
