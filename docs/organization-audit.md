# fiber-kit source — legibility and organization audit (2026-11-25)

**Scope.** `src/fiber_kit/` (100 modules, 47,479 lines), the runner `scripts/fiber-pipeline`
(1,382 lines of bash), the four configuration profiles, `test/` and `tools/`. The same questions
as the Klusters audit of 2026-11-24: can the code be read, and can a thing be found.

**Method.** Measured, then read where the numbers pointed. Function metrics with `lizard`
(`tools/audit/function_metrics.py`); unused functions and repeated bodies by a textual pass
(`tools/audit/dead_and_dupes_textual.py`); knobs, option names, entry points and comments by
counts (`tools/audit/surface_counts.py`). State: 0.71.10.

**Nothing was changed.** Each finding ends in what a fix would be.

**Limits.** The unused-function list is textual and was not triaged one by one. Counts by regular
expression are approximate. Whether an algorithm is right was not asked here (see
`docs/methods.md` and the measurements in the changelog); neither was speed.

## Summary

| # | Finding | Evidence | Weight |
|---|---|---|---|
| 1 | A few functions are programs of their own | 13 functions over 200 lines hold 13% of all function lines; one takes 96 parameters | high |
| 2 | A knob reaches its stage by one of two routes | 107 knobs go through the runner as flags, 336 are read by Python, 17 by both | high |
| 3 | A stage is listed in five places, and the lists disagree | `fiber-plan-lint` rejects three of the four shipped profiles | high |
| 4 | `fk run` takes the runner from `PATH` before the checkout | with an older install present, a checkout runs the old runner | high |
| 5 | Several names for one option | input stage: `--clu-stage` in 18 modules, `--in-clu` in 12 | medium |
| 6 | The profiles repeat each other | four files, 6,504 lines; 9 in 10 comment lines are copies | medium |
| 7 | 28 modules are named by no test | among them the three linkers' shared code | medium |
| 8 | Where things are | 100 modules in one directory, three subsystems in one package | medium |
| 9 | Small debts | swallowed exceptions, six unregistered environment knobs, long lines | low |

What is in good order: no `TODO` or `FIXME` anywhere; no function body occurs twice; 29
functions (287 lines) are named nowhere else, against 107 in Klusters; every registered knob is
read somewhere; comments are 8% of lines and docstrings 13%, and only 39 comment lines in the
Python tell history; `pyflakes` gives 11 lines for `src`, `tools` and `test` together; 101 test
files run on every push.

## 1. A few functions are programs of their own

1,752 functions; median 11 lines, 90th percentile 48.

| Lines | Complexity | Function |
|---|---|---|
| 671 | 190 | `fiber_session.main` |
| 503 | 192 | `fiber_session.cluster_chunk_fine` — 96 parameters |
| 428 | 78 | `fiber_intrachunk.main` |
| 415 | 33 | `fiber_refine.main` |
| 402 | 115 | `fiber_realign.main` |
| 332 | 107 | `neuro_io.apply_spike_keep` |
| 313 | 64 | `fiber_anchor_link.main` |
| 313 | 56 | `fiber_link.main` |

- The 65 `main` functions hold 17% of all function lines; seven are over 200 lines. In those the
  algorithm's outer loop lives in the command-line function, so it cannot be called or tested
  without a session on disk.
- 57 functions take more than 12 parameters: `cluster_chunk_fine` 96, `fiber_refine.refine` 65,
  `fiber_link.link_session` 47, `fiber_polish.polish` 37. Each new knob adds a parameter, a
  keyword at the call and an `argparse` line.
- `fiber_pipeline_editor._build_qt` (686 lines) and `fiber_config_editor._build_window` (526) are
  one closure each: complexity 1 by the metric because everything is a nested function.

**Fix.** One settings object per stage, built once from the parsed arguments and passed down
(`config.StageConfig` exists; only `IntrachunkConfig` uses it). `main` parses, builds the
settings and calls a `run(settings, session)` that can be imported.

## 2. A knob reaches its stage by one of two routes

460 knobs are registered in `fiber-kit-exp.yaml`.

- **Through the runner.** `scripts/fiber-pipeline` reads the configuration into 118 shell
  variables and turns them into flags (`--fine-algo "$SESSION_FINE_METHOD"`). 107 registered
  knobs are named only there: most of `FK_SESSION_*`, `FK_REFINE_*`, `FK_LINK_*`, `FK_REALIGN_*`,
  `FK_PEEL_*`.
- **By the stage.** The stage reads `FK_*` itself from the configuration and the environment: 336 knobs,
  among them all of `fiber-polish`, `fiber-xcorr-merge`, `fiber-psame-merge` and `fiber-consensus`.
- 17 knobs are named on both sides.

For those 107 knobs a stage run by hand does not see the configuration: `fiber-session <base> 8`
takes its `argparse` defaults for them, not the profile's values. For the others it does. Nothing
on the command line says which kind a knob is. There are also 59 reads of `os.environ` outside
`config.py` (17 in `fiber_session.py`).

**Fix.** Every stage reads its own knobs; the runner passes only the session, the group, the
tags and the step's `params`. The runner's stage functions then shrink to one line each
(`fiber_polish`, `fiber_clean` and `fiber_unmix` already are).

*Done in 0.71.14, the stage side:* `config.RUNNER_KNOBS` and `config.apply_profile_knobs`; nine
stages read the knobs the runner passes, and `test_runner_knobs` holds a run by hand equal to a
run through the runner under every shipped profile.  The runner still passes the flags, so its
stage functions are as long as before.  Found on the way: `FK_SESSION_DIPSPLIT: 0` did nothing
(0.71.13), and with no profile 23 settings of four stages differ between the two routes because
the stages' own defaults are an older configuration than the runner's.

## 3. A stage is listed in five places, and the lists disagree

A plannable stage is named in: the runner (a function, two dispatch tables, the known-stage set,
the default-output table, the shield table, the `--list` text: a median of 10 lines per stage, 56
for `fiber-session`); the plan editor's `CATALOG` and `STAGE_MODULES`; `pyproject.toml`;
`tools/gen_stages_doc.py`; the plan template.

| List | Stages |
|---|---|
| console entry points | 65 |
| sections in `docs/stages.md` | 55 |
| the runner can plan | 22 |
| the plan editor and `fiber-plan-lint` know | 18 |

- The editor does not know `fiber-flag-artifacts`, `fiber-flag-collisions`, `fiber-manifold` and
  `fiber-peel`. **Run:** `fiber-plan-lint` on `fiber-kit-exp.yaml`, `fiber-kit-best.yaml` and
  `fiber-kit-klusters.yaml` stops at step 0 with "missing or unknown 'stage'"; only
  `fiber-kit.yaml` passes. The profiles the runner executes cannot be linted or opened in the
  editor.
- No section in `docs/stages.md`: `fiber-lowmix`, `fiber-decollide`, `fiber-flag-artifacts`,
  `fiber-template`, `fiber-unmix-children`, `fiber-morpho`, `fiber-morpho-fetch`,
  `fiber-config-edit`, `fiber-plan-lint`, `fiber-view-tour`. `gen_stages_doc.py --check` passes:
  it checks that each section matches its parser, not that each entry point has a section.
  *Done after 0.71.12:* the ten have a section, and `--check` fails on a command without one.
- `fiber_chunks` and `fiber_subsession` have a `main` and no entry point (reached through `fk`
  and the runner).

**Fix.** One table in Python (stage → module, input and output option, default output tag,
whether it is shielded on a chunk run, one line of summary). The runner asks it
(`plan_steps` already hands the runner its plan rows), and the editor, the linter and the
documentation generator read it. A test that every entry point is in it or on a short list of
tools.

*Done in 0.71.12, in part:* `fiber_kit/stages.py` (name, module, group, summary); `--list`, the
plan parser, the editor and the linter read it; `test_stage_table` compares the runner's arrays
and dispatch tables with it.  The per-stage option names and default tags are still in the
runner's functions.

## 4. `fk run` takes the runner from `PATH` before the checkout

`plan_steps._cmd_run` uses `shutil.which("fiber-pipeline")` and falls back to the checkout's
`scripts/fiber-pipeline` only when nothing is on `PATH`. **Run:** on a machine with an editable
install of 0.67.9 left on `PATH`, `fk run` from a 0.71.10 checkout ran the 0.67.9 runner, which
does not know `fiber-flag-collisions`. `test_profiles.py` fails there on "fk run --config
<file> runs that file's plan" and passes with `scripts/` first on `PATH`. The runner prints no
version, so the two cannot be told apart from the log.

**Fix.** Prefer the runner beside the package that is running; have the runner print its version
and refuse a Python package of another version.

*Done in 0.71.11:* `plan_steps.runner_path()`, the runner's version check and hand-over,
`test_runner_version`.

## 5. Several names for one option

642 distinct long options over 1,163 `add_argument` calls.

| Meaning | Names (modules) |
|---|---|
| the clustering read | `--clu-stage` (18), `--in-clu` (12), `--stage` (3), `--in-stage` (1) |
| the clustering written | `--out-stage` (19), `--out` (18), `--out-tag` (9), `--tag` (2) |
| the method token | `--clu-method` (21), `--spk-method` (12), `--method` (6), `--variant` (3) |

In `fiber-session` `--clu-stage` names the output (its input is `--in-stage`); in `fiber-polish`,
`fiber-clean`, `fiber-xcorr-merge` and `fiber-unmix` it names the input, and the output is
`--out-tag` in three of them and `--out-stage` in the fourth. The runner hides the difference (a plan says `in:` and `out:`), which is why it carries a function
per stage.

**Fix.** `--in`, `--out`, `--method` everywhere, the present names kept as aliases for a
release; the table of finding 3 then needs no per-stage option names.

## 6. The profiles repeat each other

| File | Lines | Knobs | Comment lines also in `exp` |
|---|---|---|---|
| `fiber-kit-exp.yaml` | 1,898 | 460 | — |
| `fiber-kit-best.yaml` (generated) | 1,807 | 460 | 1,144 of 1,252 |
| `fiber-kit-klusters.yaml` (generated) | 1,775 | 460 | 1,139 of 1,225 |
| `fiber-kit.yaml` (by hand) | 1,024 | 263 | 627 of 709 |

- A knob's explanation is in up to four files and in `docs/config.md`. Two of the files are
  generated; `fiber-kit.yaml` is kept by hand and lists 263 of the 460 knobs.
- The comments are the reference manual and the lab notebook at once: 54 comment lines of
  `fiber-kit-exp.yaml` tell what a value used to be or which version changed it; lines run to
  258 columns.
- A session's own `fiber-kit.yaml` is a copy of one of these at the version it was copied, with
  the plan of that day (the header of the file audited for 0312 still described a plan it no
  longer held).

**Fix.** One registry (name, type, default, range, one paragraph) from which the profiles and
`docs/config.md` are generated; a profile is then the values that differ from the defaults plus
its plan. The history of a value belongs in the changelog and in `docs/bench.md`.

## 7. 28 modules are named by no test

`fiber_link`, `fiber_link_core`, `fiber_backbone_link`, `fiber_merge`, `fiber_split`,
`fiber_refit`, `fiber_reject`, `fiber_score`, `fiber_stats`, `fiber_qc`, `fiber_contam`,
`fiber_calibrate`, `fiber_cfiber`, `fiber_branch`, `fiber_defrag`, `fiber_drift`,
`fiber_position`, `fiber_methods`, `fiber_microfiberize`, `fiber_refiber`, `fiber_provenance`,
`fiber_stochastic`, `fiber_trajectory`, `fiber_view`, `fiber_view_gui`, `graph_link`,
`laplacian_link`, `raw_vs_stderiv`.

- Textual: a module exercised only through another one is on the list wrongly. Coverage was not
  measured.
- The tests are scripts with their own `check()`; `run_all.py` runs each as a process and
  reports per file. A failure names the file and the sentence, not the line.
- Finding 4 is a test that depends on what is installed on the machine.

**Fix.** Measure coverage once (`coverage run test/run_all.py`) and decide per module: test it,
or move it to the list of finding 8. `run_all.py` should put `scripts/` first on `PATH`.

*Measured at 0.71.12* (`tools/audit/coverage_run.sh`): the suite executes 62% of the statements of
`src/fiber_kit` (17,450 of 28,106); 38 of 101 modules are under 50%.

| Covered | Modules |
|---|---|
| 0% | `fiber_provenance`, `fiber_trajectory`, `graph_link` |
| under 20% | `laplacian_link`, `fiber_methods`, `fiber_collision`, `fiber_view_gui`, `fiber_stochastic`, `fiber_merge`, `fiber_link`, `fiber_contam`, `fiber_view`, `fiber_score`, `fiber_defrag`, `fiber_drift`, `fiber_qc` |
| 20 to 35% | `raw_vs_stderiv`, `fiber_cfiber`, `morpho_study`, `fiber_position`, `fiber_relink`, `fiber_reject`, `ndm_fiber_kit`, `fiber_stats`, `fiber_refit`, `fiber_pipeline_editor`, `fiber_calibrate`, `fiber_branch`, `fiber_refiber`, `fiber_refine`, `fiber_microfiberize`, `fiber_intrachunk` |

- In most of these the covered part is the import and the argument parser (`test_stage_startup`
  starts every stage); the algorithm is not run.
- Three stages of the canonical order are among them: `fiber-refine` 30%, `fiber-intrachunk` 35%,
  `fiber-link` 16%. The plans in use (session, polish, clean, xcorr-merge, unmix) are at 64%, 86%,
  89%, 72% and 91%.
- The textual list above was close: 26 of its 28 modules are under 50% (`fiber_backbone_link` is
  at 60%, `fiber_link_core` at 51%).

## 8. Where things are

- **One directory, three subsystems.** 74 `fiber_*` modules (the sorter), 13 `morpho_*` (5,925
  lines of biophysical modelling with its own 2,020-line manual), and the Qt tools
  (`fiber_view*`, the two editors: 3,274 lines). Importing the sorter does not need the other
  two.
- **Which modules are the pipeline.** 22 stages can be planned; 41 other entry points are
  diagnostics, older linkers, studies and tools, in the same directory under the same prefix.
  Nothing marks `fiber_link`, `fiber_refiber`, `fiber_microfiberize`, `fiber_defrag` as outside
  the present plans.
- **`tools/`** holds the generators CI needs (`gen_stages_doc.py`, `gen_config_doc.py`,
  `gen_best_profile.py`) beside 15 study scripts.
- **`neuro_io.py`** (1,971 lines) is the file formats, the session paths, the method-token rules
  and the spike-keep sweep.
- 23 functions (492 lines) are used only by tests or tools, most of them in `morpho_validate`.

**Fix.** Sub-packages `fiber_kit.morpho`, `fiber_kit.gui`, `fiber_kit.legacy` (entry points
unchanged); `tools/research/`; `neuro_io` split into formats, paths and tokens.

## 9. Small debts

- 11 `except Exception` followed by `pass` (4 in `fiber_realign.py`, 4 in `fiber_session.py`), 7
  of them without a word on why; 65 `except Exception` in all. *Corrected the same day:* this
  said 38; that count took every `except <anything>: pass`.
- Environment knobs outside the registry: `FIBER_ALIGN`, `FIBER_ALIGN_CHANNELS`,
  `FIBER_SUBSAMPLE`, `FIBER_PROBE_FLIP_Y`, `FIBER_KIT_GPU`, `FK_SPK_VARIANT`; the profiles still
  explain the retired `FK_REFINE_DEDUP_STRICT`. *Done after 0.71.12:* `docs/pipeline.md` lists the
  environment variables that are not knobs.
- 448 `print` calls and no logging level: a stage cannot be made quiet or verbose.
- 343 lines over 120 columns, 26 over 160.
- 47 function names are defined in more than one module with different bodies
  (`read_manifest` and `write_manifest` three times each, `_taper` twice).
- 29 functions named nowhere else: `fiber_view.bundle_figure` (36 lines),
  `fiber_pca.realign_pca` (34), `morpho_envelope.measure_recovery` and `fit_recovery`,
  `fiber_geometry.temporal_offset` and `curl_feature`, `neuro_io.session_rates`.

## Order of work

1. Finding 4 (one function) and the four missing stages of finding 3 (one table): both make a
   shipped command fail today.
2. The stage table of finding 3, with the test that keeps it whole.
3. Finding 2 for `fiber-session`, `fiber-refine`, `fiber-link`, `fiber-realign`, one stage per
   patch, each with a test that a direct run and a run through the runner give the same command.
4. Finding 1 for `fiber_session` once its knobs are a settings object.
5. Findings 5 and 6, which change what a user types and reads, together in one release.
6. Findings 7 to 9 as they are met.

## State of the findings

| Finding | State |
|---|---|
| 1 Large functions | open. `cluster_chunk_fine` and the `main` functions are as they were |
| 2 Two routes of a knob | the stage side done (0.71.14); the runner still builds the flags. 23 defaults differ with no profile: to decide |
| 3 Stage lists | one table, tested (0.71.12); every command documented. The runner's per-stage functions remain |
| 4 Runner from `PATH` | done (0.71.11) |
| 5 Option names | open: the names are a choice (`--in` / `--out` / `--method`, or the present ones) |
| 6 Profiles | open: the layout of the profiles is a choice |
| 7 Untested modules | measured (62%); no test added |
| 8 Where things are | open: sub-packages change import paths |
| 9 Small debts | lint, the environment variables and the exception count done; 20 unused functions removed (0.71.15) |
