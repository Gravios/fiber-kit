# Profiling fiber-kit (`fiber-profile`)

`fiber-profile` samples a fiber-kit command with [py-spy](https://github.com/benfred/py-spy) and
writes a summary of where the time went. It needs no code change, works on the Python fiber-kit runs on
today (3.12, 3.13), and follows the process-pool workers of `fiber-session`, `fiber-refine` and
`fiber-stochastic`, where most of the time is spent. py-spy installs with fiber-kit (a binary wheel).

## Recording

```bash
# one stage
fiber-profile record --out prof -- fiber-session SESS 8 --jobs 8 ...

# every step of a plan run: one profile per command the runner starts
FK_PROFILE=prof fiber-pipeline 8 --plan plans/01-baseline.yaml
FK_PROFILE=prof FK_PROFILE_ARGS="--rate 200" fiber-pipeline 8 fiber-session

# a job that is already running (Linux usually needs root for this, see Permissions)
sudo fiber-profile record --out prof --pid 123456 --duration 120
```

Each recording writes three files under `--out`, named `<name>-<YYYYmmdd-HHMMSS>`:

| file | contents |
|---|---|
| `.raw` | py-spy's collapsed stacks, one line per distinct stack with its sample count. Drag it into <https://www.speedscope.app> for an interactive flamegraph ("Left Heavy" view). |
| `.json` | wall time; CPU time (user, sys) and peak RSS of the command and everything it started; command line, sampling rate, host, Python and fiber-kit versions, and the `FK_*`, `OMP_*`, `OPENBLAS_*`, `MKL_*` variables of the run |
| `.txt` | the summary, as printed at the end of the run |

In a plan run the name is `<command>_<output tag>`, e.g. `fiber-session_fc_session-20261127-221500`.

`fiber-profile summary prof/*.raw` prints the summaries again; `--json` prints the numbers instead,
for comparing runs in a script.

## Reading the summary

```
wall      : 3.48s
cpu       : 5.04s  (user 4.48s, sys 0.56s)  = 1.45 cores on average of 2
peak RSS  : 204 MiB (largest single process)
sampled   : 499 samples at 100 Hz = 4.99s of busy Python thread time; 1.41 threads busy on average

processes
  parent  pid 402: 1.64s sampled
  workers 2: 3.35s sampled in all; per worker min 1.53s, median 1.82s, max 1.82s; balance 0.92

activity           parent             workers
  compute           0.52s  31.7%        3.34s  99.7%
  waiting           0.08s   4.9%        0.00s   0.0%
  pickle/IPC        0.04s   2.4%        0.00s   0.0%
  import            1.00s  61.0%        0.01s   0.3%
...
```

(A synthetic two-worker `fiber-session` run, for the shape of the report; the numbers mean nothing.)

- **cores on average**: CPU time / wall time. Well below `--jobs` means the pool is often idle,
  because of serial work in the parent, uneven chunks or I/O.
- **threads busy on average**: the same from the samples, Python threads only.
- **CPU time far above the sampled time** (the report says so): the difference ran in native
  threads py-spy does not sample. Usually that is a multi-threaded BLAS in the parent; the pool
  workers pin BLAS to one thread.
- **balance**: worker seconds / (workers × the busiest worker's seconds). 1.00 means every worker was as
  busy as the busiest. Low balance means a few long chunks keep the pool open while the others wait.
- **activity**, by the frames of each sample:
  - *import*: a frame in `importlib`. Start-up cost. This is what 3.15's lazy imports would cut.
  - *pickle/IPC*: in `pickle`, `copyreg`, `multiprocessing.reduction`/`queues` or sending on a
    connection. Moving tasks and results between processes.
  - *waiting*: the innermost Python frame is in `threading`, `selectors`, `queue`,
    `concurrent.futures` or a connection receive: the parent waiting for the pool, a lock, a queue.
  - *compute*: everything else.
- **computing, by package**: the package of the innermost frame. *fiber_kit* is time in fiber-kit's
  own Python lines; *numpy*/*scipy*/*sklearn* is time inside those libraries (most of it in their
  compiled code, which py-spy attributes to the Python call that entered it unless `--native`).
- **top functions by own time**: by line, so a hot loop shows up as its line.
- **top fiber-kit functions including what they call**: the fiber-kit call tree folded per function.
  Read it for which stage of the algorithm costs the time, e.g. `cluster_chunk_fine` → `_rkk_labels` →
  `klustakwik`.

## Options worth knowing

| option | when |
|---|---|
| `--rate HZ` | 100 by default. Long runs are fine at 100; raise it for runs under a minute. |
| `--native` | to see inside numpy/BLAS/Cython (openTSNE) frames. More overhead; symbol names depend on how the libraries were built. |
| `--gil` | only samples that hold the GIL. Record the same command with and without it: (sampled seconds with `--gil`) / (without) is the share of busy time that holds the GIL. Low means threads would parallelise it as well as processes do (the question for the free-threaded build, see [python-upgrade-plan.md](python-upgrade-plan.md)). |
| `--idle` | include threads py-spy judges idle (e.g. the parent blocked on the pool) to see waiting in full. |
| `--nonblocking` | do not pause the process for each sample: less overhead, slightly torn stacks. |

## Permissions

py-spy reads the target's memory with `ptrace`.

- **A command `fiber-profile` starts** is py-spy's own child, which Ubuntu's default
  (`kernel.yama.ptrace_scope = 1`) allows. No root needed. In a container, add `CAP_SYS_PTRACE`.
- **`--pid`** attaches to a process that is not a child: run it with `sudo`, or set
  `sudo sysctl kernel.yama.ptrace_scope=0` until the next boot.

If py-spy writes no samples, `fiber-profile` says so and gives this hint.

## Suggested first measurements

1. `fiber-session` on the reference session at the production `--jobs`, default rate: shows how
   much of the wall time the pool is busy (cores on average, balance) and where the workers spend it.
2. The same with `--gil`: the GIL share (above).
3. A whole plan with `FK_PROFILE=prof`: which steps dominate, and their import overhead
   (`fiber-profile summary prof/*.raw | grep -E '^==|wall|import'`).
4. A short stage (`fiber-spkc`, `fiber-ccg`) and `fiber-session -h`: start-up cost alone, the baseline
   for lazy imports.
