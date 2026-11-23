# Changelog

All notable changes to **fiber-kit**. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic-ish
`0.MINOR.PATCH` versions (each minor adds a tool or a self-contained capability).

## [0.70.0] — the sparse .spkc: collision rows with the partner subtracted
- **`.spkc`** (`neuro_io.read_spkc` / `write_spkc`, `SPKC_VERSION` 1; `contract.yaml` `contract: 3`):
  the corrected records of the FEW spike rows whose `.eap` row names a partner unit, as a row index
  plus those rows' records.  The `.spk` stays the raw cut of the recording.  The format is
  neurosuite-3's (`doc/ndmanager-plugins/formats/spkc.md`); Klusters builds the same rows in memory
  and serves them in place of the `.spk`'s.
- **`fiber_spkc`** and the **`fiber-spkc`** command: the numpy mirror of neurosuite-3's
  `core/decollide_fit.hpp`.  A partner's template is fitted into the record within a few samples of
  the offset the `.eap` holds, jointly with the row's own template and to a fraction of a sample,
  and subtracted when its amplitude is plausible and it removes at least 5 % of the residual.  The
  amplitudes are not stored: the fit is redone each time and returns the offset it found
  (`--write-offsets` puts it in the `.eap`).
- **`fiber-unmix`**: a collision goes INTO the unit on its detection when that label is confident
  (`--collision-as unit`, the new default; `cluster` is the 0.68 behaviour), and the stage's `.spkc`
  is written for the standard and the method `.spk` (`--spkc 1`).
- A dedup of a group moves its `.spkc` files aside as stale (they name rows).  Not covered by a test.
- Parity, measured 2026-11-23 outside the test suite: on 2,000 random cases `fit_partner` and
  `decollide_fit::fitPartner` made the same accept decision and found the same offset, the shift
  within 2e-10 sample, and the corrected int16 records were identical; on a synthetic session with
  40 planted collisions `fiber-spkc` wrote the `.spkc` Klusters wrote, row for row and record for
  record.  `test/test_spkc.py`: 60 planted collisions among 900 rows are exactly the rows corrected
  (error against the record before the partner under 0.1 % of the raw row's), rows naming a partner
  that is not there are left alone, the fitted offsets are the planted ones.
- Synthetic data only: no real session has been through it.

## [0.69.0] — fiber-clean: the core cut and the quality value as a stage
- **New stage `fiber-clean`.**  One pass over a flat `.clu`: the core cut on every cluster of at least
  `--cc-min` spikes (RAW `.spk.standard`), then the quality value of what is left (method `.spk`);
  cut-off spikes, clusters at or above `--q-dirty`, clusters left under `--min-keep` and the `--junk`
  clusters go to the noise cluster.  Cluster 0, the noise and reserved clusters are untouched.  Writes
  the stage's flat `.clu` and `<base>.clean.<method>.<g>.<stage>.tsv`.  Plannable, not in `all`.  With
  `fiber-unmix` it closes the loop clean -> unmix -> sort what is left (docs/stages.md).
- **`fiber_corecut`**: a numpy port of neurosuite-3's `corecut.h` (Klusters' Split Off Spurious Spikes)
  and of `waveformquality.h` (the waveform view's quality dot).  Against the header on
  sirotaA-jg-000005-20120312 group 8: the same keep mask, spike for spike, on 40 clusters of 200-5,000
  spikes, 8 of 6,000-22,000 and (ordering a subset of 2,000) 4 of 6,400-10,400; `fiber_clean.clean`
  relabels 643 clusters (279,964 spikes) exactly as the analysis script that drives the header.  No
  cluster above 40,000 spikes was compared.  The port was 3 x faster than the header's serial loop on
  the large clusters (BLAS).
- It trades yield for purity (same group, fs_polish, time-local purity against the curated file):
  purity 93.0 -> 95.8 %, clusters >= 90 % pure 57 -> 66 %, curated spikes held 77.1 -> 53.7 %.
- Run through `fiber-pipeline` on that group (fs_polish, `junk: 2769`, 4 minutes): 456 clusters kept,
  the same labels as the analysis script's except that the 3 artefact spikes stay in cluster 0.

## [0.68.0] — fiber-unmix: the noise cluster explained on the .fil
- **New stage `fiber-unmix`.**  After a cleaning pass the noise cluster is mostly spikes of units the
  sort already holds, alone or summed (sirotaA-jg-000005-20120312 group 8: about 7 in 10 fit one kept
  unit, 1 in 10 is a detectable sum of two; re-sorting that noise gave units 84 % pure against 96 % for
  the kept ones -- analysis/fs-polish-clean-resort).  The stage takes the kept units as a MASK and as
  templates, and pursues only the noise detections, on the continuous `.fil`, with `fiber_deconv`, per
  block of the recording.  Each gets a kind: `one` (a kept unit), `collision` (two kept units in one
  waveform, the second detected or hidden), `shadow` (a second detection of a neighbour's flank),
  `partial` (a kept unit and a remainder) or `unexplained`.  The first three leave the noise cluster
  (`--one-as cluster|units|assign`); the last two stay for the next sort.  Writes the stage's flat
  `.clu`, `<base>.unmix.<method>.<g>.<stage>.tsv` and the `.eap`/`.tcl` membership (kept spikes: own
  class at offset 0; explained detections: the classes in them).  Nothing is subtracted from a file.
  Plannable (`{stage: fiber-unmix, in: ..., out: ...}`), not in `all`; skips reserved clusters itself.
- **Templates reach beyond the `.spk` window** (`--extend`, default 8 samples each side).  With the
  window's own length a spike's tail stays in the signal as if it were somebody else's: on the test
  recording 37 % of the second detections on a flank were left unexplained at `--extend 0`, 4 % at 8.
- `fiber_deconv.build_model` (noise model + atom bank + equivalence classes, what `deconvolve` built
  inline) and `deconvolve(..., equiv=, thr=)`, so templates can come from other detections than the
  ones pursued.  `deconvolve`'s results are unchanged (test_deconv, test_session_deconv).
- `fiber_decollide.write_eap_membership(..., own=(rows, units))` marks spikes with their own class at
  offset 0, and a manifest row with a negative `k1`/`k2` writes no cell for that slot.
- Limits: only DETECTED events are examined -- a spike under a kept spike that never crossed the
  threshold still needs a subtraction from the recording; two spikes a few samples apart are mostly
  read as `one`; the `.eap` is spikes x classes bytes (`--eap resolved|off` for less).
- Checked on a synthetic 150 s recording with planted cases (test/test_unmix.py: one 95 %, collision
  94 % with both units right, shadow 96 %, unit + unknown cell 100 % kept in the noise, a cell without
  a template 100 % kept in the noise, detections on bare noise 0 %).  NOT run on a real session: the
  machine it was written on has no `.fil`.

## [0.67.37] — dedup: super-sampled spaces, the EAP table, Klusters working copies
- **`fiber-refine --dedup` no longer breaks a super-sampled space.**  `neuro_io.apply_spike_keep` read
  every `.spk` of the group at the session's `nSamples`.  A `<base>.spk.<token>_U4.<g>` (Klusters'
  super-sample mode, `ndm_extractspikes --supersample`) holds 4 x that per channel, so it came out at
  4 x the spike count, was judged stale and moved to `.stalebkp` -- while the same space's `.fet`,
  `.clu` and tick `.res` were subset.  Each waveform file is now read at its token's geometry
  (`perspike_row_samples`) and subset row for row.
- **The EAP membership table is subset with its stage.**  `<base>.eap.<g>[.<stage>]` has one row per
  spike and was not in the sweep: after a dedup every row past the first dropped spike named another
  spike.  `read_eap_file` / `write_eap_file` take a path.
- **A Klusters working copy refuses the dedup.**  With any `<file>.pending` of the group present the
  dedup raises before writing: Klusters has the session open (or crashed with it), and its copies were
  being treated as a stage named `pending` -- subset or quarantined under it.
- **Files of a type the sweep does not know are named.**  `neuro_io.PER_SPIKE_TYPES` and
  `NOT_PER_SPIKE_TYPES` list what a group's files are.  A binary group file in neither gets a warning
  on stderr after the dedup (it may be per-spike, and is then stale); `--dedup-stale error` refuses
  before writing.  Text sidecars and backups are not asked about.
- Found by running the dedup on a 100-spike group with `standard` and `standard_U4` files
  (claude/pipeline-audit-2026-11-20.md, H1).  Not run on a real session.
- Tests: `test/test_dedup_perspike.py` (16 checks; on 0.67.36 it fails at the first: the `_U4` `.spk` is quarantined).

## [0.67.36] — fiber-config-edit: a form editor for fiber-kit.yaml
- **New command `fiber-config-edit [fiber-kit.yaml]`** (Qt >= 6.8, the `[editor]` extra; opens `$FK_CONFIG`,
  else `./fiber-kit.yaml`).  The file's own banner sections are drop-downs; each `FK_*` knob is a row typed
  from the lead of its comment -- `0 | 0/1` and `0 = off, 1 = on` are switches, `auto | nst | fil | spk` a
  choice list, the rest a text field (numbers right-aligned, outlined when outside the documented `lo-hi`).
  A knob that is off is greyed, and so is every `FK_X_*` under an off `FK_X`, with the gate named beside it.
  A value that differs from the documented default shows that default and a button back to it; an edited
  value shows what the file had.  The embedded plan's steps are rows with a tick (commented in or out).
  Pointing at a knob, a section heading or a note between knobs shows its whole comment in a side pane.
  Filter by name / value / help text, "Edited only", "Hide off", a review dialog with the diff.
- **Lossless.**  The file is never re-serialised: only the value token of an edited line is rewritten (the
  padding absorbs the width so the comment column stays), and a toggled step gains or loses its `# `.  All
  five shipped configs round-trip byte for byte; a save is YAML-checked first and written atomically.
- The typing is a heuristic read of the comments, not a schema: on fiber-kit-exp.yaml it finds 45 switches,
  29 choice lists, 349 numbers and 29 text knobs; 33 numbers have no documented default, and the off/gating
  rule (name prefix) marks 5 knobs as gated.  A knob it types wrongly is still editable as text-in-a-list;
  the stage's own check of the value is unchanged.  `fiber-config-edit --check FILE` prints the reading.
- Structural plan edits (add, reorder, rewire) stay in `fiber-plan-edit`.
- Tests: `test/test_config_editor.py` (49 checks; the window is built off-screen when PySide6 is present).

## [0.67.35] — fiber-residual-merge tuned; a threshold sweep from one run (--replay)
- **Tuned defaults: `FK_RSM_ACC_THR` 0.7 -> 0.675, `FK_RSM_SHIFT` 0 -> 2, `FK_RSM_STRIP_K_ON` 3 -> 6.**
  g6 of 0316, 80 min sorted without cross-chunk linking (1,961 fibers), curated `.clu` (contaminated:
  read differences).  35 configurations searched on minutes 0-54, the choice checked on 54-80 and run
  through the command on the whole session:

  | whole session | clusters | acc | purity | good | complete | mixed | time |
  |---|---|---|---|---|---|---|---|
  | input | 1961 | 0.119 | 73.4 | 59.6 | 8.3 | 19 | — |
  | fiber-xcorr-merge, tuned profile (knn split off) | 1388 | 0.176 | 73.2 | 65.2 | 24.9 | 18 | 337 s |
  | fiber-xcorr-merge, plan 10 | 837 | 0.317 | 69.1 | 66.1 | 43.9 | 13 | 66 s |
  | fiber-residual-merge, 0.67.34 defaults | 1562 | 0.187 | 73.0 | 62.2 | 21.4 | 21 | 240 s |
  | **fiber-residual-merge, tuned** | 1345 | 0.241 | 72.1 | 62.5 | 27.8 | 16 | 407 s |

  | acc / purity / mixed | 0-54 min (tuned on) | 54-80 min (held out) |
  |---|---|---|
  | input | 0.124 / 77.0 / 12 | 0.165 / 61.8 / 7 |
  | fiber-xcorr-merge, tuned profile | 0.176 / 77.0 / 12 | 0.199 / 61.3 / 3 |
  | fiber-residual-merge, 0.67.34 defaults | 0.205 / 76.7 / 11 | 0.205 / 61.3 / 10 |
  | fiber-residual-merge, tuned | 0.266 / 75.9 / 9 | 0.244 / 60.7 / 11 |

  `acc` is fiber-tune's `all_acc`.  What tuning bought: accuracy and completeness, for about a point of
  purity; the gain holds on the held-out minutes (0.205 -> 0.244).  What it did not buy: against the
  tuned fiber-xcorr-merge it is more accurate and more complete but a point less pure, with less `good`,
  and on the held-out minutes it leaves 11 mixed clusters to that stage's 3.  It still replaces
  fiber-xcorr-merge in no shipped plan; the exp plan keeps it as a branch.
- **The selection rule is not fiber-tune's objective.**  `all_acc - 0.005 * merges` rises all the way
  into a collapsed sort: the best objective of the whole search (0.317) is a run that fuses 1,169
  fibers into 40 clusters and counts 4 "merges" -- a blob of ten units has no second unit at 20 %, so
  the penalty goes DOWN as the sort gets worse.  (With the 0.67.34 values, acceptance 0.2 takes 745
  fibers to 37 clusters at 0.290 and 3 merges.)  The rule used instead: the highest
  `all_acc` whose spike-weighted purity stays within one point of the INPUT sort's (over-split before
  false merges).  Worth knowing for any fiber-tune search that can reach a collapse.
- What the search found, per knob (all at equal purity unless said; details in section [I]):
  `SHIFT` 2 alone 0.231 -> 0.252; `STRIP_K_ON` 6 (with `K_OFF` 4) 0.250; both 0.265.  The on-channel
  term OFF scored a little higher (0.255; 0.271 with `MERGE_ALIGN` klusters) and was not taken -- it is
  what keeps a different cell at the same site out, and this curation cannot price that.  The
  off-channel term off: per-unit precision 0.58 -> 0.51; it carries the purity.  `MERGE_ALIGN` klusters:
  helps alone (0.241), not on top of the two (0.260).  `MIN_N` 25: more `good` (67 -> 71-75), no more
  accuracy, twice the time.  `K` 2.5, `MERGE_ALIGN` peak: worse.  `COS` 0.3 / 0.7, `WIN` 16, no
  refractory veto, `K_OFF` 4: no difference worth a default.
- **`--replay <table>`.**  A run at threshold t is exactly a run at any lower threshold truncated at
  the first pair popped below t (the heap pops in descending acceptance; a lower threshold only adds
  pairs below the others).  The `.rsm` table is that pop log, now with the run's settings in its first
  line and six decimals of acceptance; `--replay` reproduces the run at any `--acc-thr` at or above the
  table's without scoring a pair, and refuses a table written at a HIGHER threshold than
  the one asked for, with other settings, or for other clusters.  This is what made the search affordable: every
  configuration gave its whole `ACC_THR` curve from one agglomeration.
- `test_residual_merge.py`: the tuned defaults keep the fixture's partition; replay == direct run at
  five thresholds; the `--replay` CLI and its refusals (30 checks).

## [0.67.34] — fiber-residual-merge: the reverse residual strip in fiber-xcorr-merge's flow
- **New stage `fiber-residual-merge`** (`fiber_residual_merge.py`, `FK_RSM_*`, section [I] of the config).
  fiber-xcorr-merge's flow, step for step -- clusters >= `MIN_N` on the standard `.spk`, a
  confidence-ordered agglomeration (best pair first, re-template after every union), the refractory
  CCG veto, `--chunks`, the `.clc`/`.clp` hierarchy carried, the drift table emitted, `--gt-clu` -- with
  a different SCORE.  Instead of the roll cosine of two median waveforms: the share of each cluster's
  SPIKES that land inside the other cluster's residual-strip tails (the gain-free on- / off-channel
  residual of 0.67.31), the minimum of the two directions, under the CLEANER cluster's tails.  The
  rules are the ones measured for fiber-polish's merge (0.67.33): cleaner tails, greedy with
  re-templating, `ACC_THR` 0.7.  Two conservative pre-filters keep it affordable: template cosine >
  `COS`, and each cluster's MEDIAN must itself pass the other's strip.  Writes
  `<base>.rsm.<group>.<out>.tsv`, every pair at or above the threshold with its verdict.
- **Not a win yet.**  Measured 2026-11-19 on g6 of 0316, five 18-min chunks sorted without cross-chunk
  linking (1,961 fibers), against the curated `.clu` (itself contaminated; read differences):

  | run | unions | clusters | purity | good | complete | mixed | time |
  |---|---|---|---|---|---|---|---|
  | input | — | 1961 | 73.4 | 59.6 | 8.3 | 19 | — |
  | fiber-xcorr-merge, tuned profile (knn split off) | 573 | 1388 | 73.2 | 65.2 | 24.9 | 18 | 337 s |
  | fiber-xcorr-merge, plan 10 (cos 0.96, min-n 15) | 1124 | 837 | 69.1 | 66.1 | 43.9 | 13 | 66 s |
  | **fiber-residual-merge, defaults** | 399 | 1562 | 73.0 | 62.2 | 21.4 | 21 | 240 s |
  | `--min-n 15` | 798 | 1163 | 72.2 | 68.5 | 19.7 | 23 | 709 s |
  | `--acc-thr 0.8` / `0.9` | 149 / 1 | 1812 / — | 73.6 | 60.8 | 15.6 | 19 | 153 s |
  | `--shift 2` | 428 | 1533 | 72.8 | 62.4 | 22.4 | 21 | — |
  | `--merge-align klusters` | 596 | 1365 | 72.7 | 63.1 | 22.2 | 22 | — |

  At its defaults it is the conservative of the two: the same purity as the tuned fiber-xcorr-merge,
  fewer unions, less `good`, less completeness.  `--min-n 15` has the most `good` of any run and pays a
  point of purity and 3x the time.  `ACC_THR` is read against a cluster's acceptance of itself, ~0.85-0.9
  at k 3 / 3, so 0.9 merges nothing.  One sort, one curation, nothing searched; the input is the
  unlinked session sort, not the linker output the exp plan hands a merge.
- **So it does not replace fiber-xcorr-merge in any shipped plan.**  `fiber-kit-exp.yaml` runs it as a
  BRANCH: `merge-resid` (exp_linked -> exp_rsm) beside `merge`, and `psame-resid` (exp_rsm ->
  exp_rsm_psame) beside `psame`, so one run gives both lines from one linker output.  The swap is one
  line (in the plan's notes).  `fiber-kit-best.yaml` / `-klusters.yaml` do not inherit the branch.
  `plans/11-session-residual.yaml` is plan 10 with the stage swapped, carrying the table above.
- What the synthetic test pins (`test_residual_merge.py`, 22 checks): fragments of one cell fuse; a
  different shape at the same site is kept out by the on-channel term and the same shape at the next
  site by the off-channel term; a population at 0.65x amplitude is kept apart (the residual is
  gain-free but normalised by the spike's own energy, so it sits at ~2.4x the noise residual); a
  half-and-half mixture is judged under the clean fragment's tails and not merged, though under its own
  it would accept 97 % of the clean one; the refractory veto; `--shift`; order independence; the CLI
  with its hierarchy and table.  The chunk-run contract is in `test_chunks.py`.
- `fiber_xcorr_merge.carry_hierarchy` / `emit_drift`: the two blocks of that stage's `main()` lifted
  into functions, unchanged, so both merges share them.
- Wired into `fiber-pipeline` (an alternative-linker stage, native `--chunks`), the plan editor,
  `fiber_bench`'s knob prefixes and `fiber_chunks.SUPPORTED_STAGES`.

## [0.67.33] — fiber-polish: strip + pooled rkk, small-leaf t-SNE islands, reverse-strip merge
- **New stage `fiber-polish`** (`fiber_polish.py`, `FK_POLISH_*`, section [H] of the config). Four
  operations on a finished `.clu`, all scored on the RAW windows at their stored alignment:
  - *loop* × `LOOPS`: strip every cluster ≥ `STRIP_MIN` with the gain-free on/off residual (0.67.31),
    pool the parked spikes with every cluster < `POOL_MAX`, and re-cluster the pool with a
    full-dimensional rkk (every column of the session's feature space). The pool's remainder goes
    to a residue cluster with a real id — nothing the stage parks ends in cluster 0 or 1.
  - *islands*: each cluster is cut into ≤ `LEAF`-spike leaves by median bisection along its first two
    PCs; each leaf gets a resampled t-SNE consensus; every community but the leaf's largest is an
    island, and moves only when its median fits another cluster's median `ISLAND_MARGIN` better than
    its own parent's (gain-only residual, the parent's median taken without it).
  - *merge*: the residual strip in reverse. A pair merges when at least `MERGE_THR` of each cluster
    survives the other's strip. Greedy with re-templating after every union (no union-find chaining),
    under the cleaner cluster's tails.
- Measured 2026-11-18 against the curators' labels (`good` = kept spikes in a ≥50-spike cluster whose
  dominant curated label is their own; `complete` = share in the largest cluster of their unit):

  | sort | clusters | purity | good | complete |
  |---|---|---|---|---|
  | 0312 g6 chunk 19, standard | 178 → 80 | 63.6 → 73.8 | 62.7 → 78.7 | 33.8 → 42.6 |
  | 0312 g6 chunk 19, hybrid | 110 → 63 | 64.1 → 72.8 | 63.5 → 77.9 | 71.8 → 56.4 |
  | 0312 g6 t 24–36, hybrid | 110 → 71 | 68.1 → 70.7 | 67.8 → 73.7 | 51.3 → 55.5 |
  | 0316 g6, hybrid, 5 unlinked chunks | 750 → 322 | 73.4 → 71.9 | 59.6 → 77.1 | 8.3 → 25.2 |

  The CLI reproduces the chunk-19 hybrid row stage for stage. **Untuned**: every default is the value
  those runs used. What the table does not hide: the island pass buys purity and costs completeness
  (4–15 points a pass) — on a sort that starts complete, the chain is a trade; the loop cost 6.6
  points of purity on the multi-chunk 0316 run and none on single-chunk windows (cause untested);
  and the curated labels are themselves contaminated.
- **`MERGE_THR` does not transfer from clean pairs.** 0.3 separated 29 clean fibers at AUC 0.99
  (template cosine 0.97) and collapsed a whole sort to 9 clusters: a contaminated cluster's own tails
  are inflated, so it accepts everything, and union-find chained the acceptances. Hence the cleaner
  cluster's tails, the re-templating, and a default of 0.7.
- Two fiber-session knobs for what the route needed: `FK_SESSION_RKK_REALIGN_MAXLAG` (the lag
  half-width of the rkk realign loop and Block N — a literal 6 at both sites) and
  `FK_SESSION_RKK_DIMS` (`--rkk-dims` had no knob). Measured with `RKK_REALIGN_ITERS` 5 / 12 / 9 on
  chunk 19: 248 → 325 fibers, `good` 62.7 → 62.1 %, 345 s → 597 s — finer splitting, not more correct
  spikes; it pays only with a merge after it.
- **Config.** `fiber-kit-exp.yaml` runs the route: `RKK_REALIGN_ITERS` 5, `MAXLAG` 12, `RKK_DIMS` 9,
  and the plan gains a BRANCH at its end — `polish` (fiber_session → fs_polish) and `rdr-polish` — so
  one run of the exp plan yields both `exp_psame` and `fs_polish` to compare. The branch is last on
  purpose: every other step runs exactly as before. `fiber-kit-best.yaml` / `-klusters.yaml` do NOT
  inherit any of it (`BRANCH_DROP_STEPS`, `HELD` in `gen_best_profile.py`): the profile was confirmed
  without the branch and at 2 / 6 / 6. `fiber-tune`'s default `score_at` now follows the plan's main
  line, so it is still `psame`.
- **Repairs to 0.67.31 / 0.67.32**, which shipped with the repo's own checks failing: the ten
  `FK_CONS_RESID_*` knobs were in no config template (now in section [6]); `docs/stages.md` and
  `plans/TEMPLATE-all-stages.yaml` were stale; `__init__.__version__` was still 0.67.30.
- Wired into `fiber-pipeline` (post-sort stage, shielded), the plan editor, `fiber_bench`'s knob
  prefixes and `fiber_chunks.SHIELD_STAGES`; cluster 0 is never touched, so a shielded chunk stays
  shielded. Writes a flat `.clu` — the `.clc`/`.clp` hierarchy is not rebuilt. The island pass needs
  the `[consensus]` extra. 21 + 3 tests.

## [0.67.32] — realigning before the residual strip spends the signal it needs
- Gravio, on the previous entry's claim that the residual-covariance mixture test failed because a
  clean unit's residual is dominated by alignment jitter: *"There should be no alignment jitter if
  the spikes are realigned before stripping."* He is right that the pipeline realigns —
  `consolidate` does one `fl.realign` per chunk and hands that frame to every pass — and neither of
  0.67.31's two measurements did it. Both redone, in three frames, on the same 22 clusters.
- **The jitter diagnosis was correct and does not rescue the test.** `rho(r1, contamination)` goes
  **−0.361 → +0.103** (chunk realign) **→ +0.134** (per-cluster), and `r1` itself collapses on the
  clean clusters exactly as the mechanism predicts (cluster 4: 0.507 → 0.184; 110: 0.474 → 0.181;
  12: 0.427 → 0.201). So jitter *was* what made clean units look structured. But +0.10 is not a
  detector, and cluster 229 — 0.0 % contamination — still carries the highest `r1` in its window
  (0.658) while cluster 3 at 53 % sits at 0.242. The blind spot stands; the reasoning for it was
  right and the conclusion unchanged.
- **Realigning does not help the strip and slightly hurts it**, which was not expected:

  | `align` | clusters | parked | precision | recall | keepers lost |
  |---|---|---|---|---|---|
  | `chunk` (the frame every pass gets) | 22 | 10504 | 0.43 | 0.27 | 11.0 % |
  | `cluster` (realigned to its own median) | 22 | 10319 | 0.43 | 0.26 | 10.8 % |
  | `stored` (chunk realign bypassed) | 22 | 10148 | **0.49** | 0.29 | 9.5 % |

  The reason is the one the template and the threshold already ran into: a contaminated cluster's
  own median is a **polluted alignment reference**, so aligning to it pulls the contaminant into
  the cell's frame and shrinks the residual that would have exposed it. The cost lands exactly
  where the strip was working — cluster 3 drops 0.80 → 0.63 precision, 11 drops 0.61 → 0.31, 184
  drops 0.27 → 0.08. A timing offset against the template is *information* about contamination,
  and a self-derived realign spends it.
- `FK_CONS_RESID_ALIGN` = `chunk` (default) / `cluster` / `stored`, validated like `--cons-mode`.
  **The default stays `chunk`** because of a confound that decides it: `stored` in that table is
  not an unaligned session, it is *this* session's curator-committed per-cluster realign, done
  after splitting — a good frame no fresh pipeline run has. The table shows a well-seeded alignment
  beating a self-derived one; it does **not** show that skipping the realign helps a raw session,
  which is not measured. `stored` is the lever for a `.spk` that already carries a curated realign.
- The template now comes from the same array the spikes are scored from, so frame and template can
  never disagree — unlike `strip`'s `tpl_align`, which sharpens the template alone (harmless for
  its calibrated quantities, but here it would inflate every residual it is subtracted from).
- 5 tests, including the mechanism in miniature: 30 spikes rolled 4 samples late read as residual
  at `stored` and are realigned away under `chunk`. 85 checks / 0 failed.

## [0.67.31] — the gain-free residual strip in fiber-consolidate, and an IQR gate that measured backwards
- `resid_pass` + `FK_CONS_RESID_*`: the pipeline twin of Klusters' "Strip Residual Outliers"
  (ns3 patches 1094/1095). `shed` scores a gain-blind distance under a `|T|` point kernel, so
  amplitude counts as distance and the off-template channels — where a spatially displaced
  intruder lands — carry almost no weight. This fits a per-spike gain on the template's own
  channels, takes `r = x − a·T`, and splits the residual's power by `u_c = E_c/max_c E_c`:
  `onr = Σ u_c q_c /(a²Eon)`, `offr = Σ (1−u_c) q_c /(a²Eon)`, each with its OWN robust tail and
  the flagged sets **unioned** — their sum scores 0.869 against 0.887 for the on-channel term
  alone, so summing dilutes whichever term carries the signal. A sibling of `shed`, not a
  replacement: `shed` stays calibrated to `FK_CONS_SHED_K`. Independent of `--cons-mode`, so it
  composes with strip/knn/shed or runs alone (`mode=off` with a tail set is a valid request, in
  `fiber-consolidate` and in `fiber-session`).
- **The IQR gate was asked for as a ceiling and measured as a mistake.** Requested: gate on the
  waveform IQR so the strip "only runs on relatively clean clusters" — the automated reading of
  the GUI finding that it works while the cluster is well defined. Measured over 22 clusters in
  two verified 12-min windows of `jg05-20120312` group 6 (t 24–36 and t 228–240 min, 192 min
  apart on purpose), scored against the curator's own `.clu`:

  | gate | clusters | parked | precision | recall | keepers lost |
  |---|---|---|---|---|---|
  | none | 22 | 10148 | 0.49 | 0.29 | 9.5 % |
  | `giqr ≤ 0.25` | 12 | 6119 | **0.27** | 0.15 | 9.2 % |
  | `giqr ≤ 0.20` | 11 | 6083 | **0.27** | 0.15 | 9.2 % |

  The ceiling halves precision, and structurally, not through a bad threshold: it rejects the
  clusters the strip is informative on (cluster 104 at 57 % contamination runs at precision 0.94
  / recall 0.85; cluster 2 at 78 % at 0.99) and admits the ones the curator left clean, where
  there is nothing to remove and the strip parks 6–14 % of good spikes at precision 0.00 —
  cluster 4 parks 1,354 of 9,673 and loses 14.0 % of keepers. Corrected for each cluster's own
  base rate the strip is most useful in the middle band, 15–40 % contamination (lift 1.6–3.0),
  and purely costly at 0 %. So the gate that matches the data is a **floor**. Both `iqr_max` and
  `iqr_min` ship, both default to **0 (off)**, because the clusters the curator left at 0.0 % may
  be clean or may be untouched and that confound decides how hard a floor is justified.
- The gate reuses `giqr`, the own-gain IQR `pure_iqr` already gates the claim on, because it was
  the best of six label-free predictors of the curator's contamination (Spearman ρ **0.743**;
  shape IQR 0.600, median `onr` 0.589, IQR of `onr` 0.666, median `offr` 0.680, no-gain fraction
  0.643). No new statistic was added.
- **One blind spot no candidate covered**, stated because it is the dangerous one: a near-even
  mixture of two cells of similar amplitude *and* similar shape. Cluster 3 at 53 % contamination
  scores `giqr` 0.188, inside the band of clusters at 0.0 % — and 53 % is past the per-point
  median's breakdown point, where the template is no longer the dominant cell's. A
  residual-covariance test was tried for it and **refuted** (ρ −0.361: a *clean* unit's residual
  is dominated by alignment jitter, so one dominant direction marks cleanliness, not mixture).
  The protection for that case is a seeded template (ns3 1095), not a gate.
- Per-cluster log lines carry the sited tails, which is the only way to see whether the
  thresholds are near the data; `FK_CONS_RESID_QUIET=1` silences them. 15 tests: both terms
  separately, each disabled on its own, the gain divided out (a half-amplitude spike of the same
  cell is not an outlier), the gate in both directions, independence from `--cons-mode`, the
  exclude list, `resid_min`, and the taper.

## [0.67.30] — the time-column guard failed on every realigned session
- Found by running 1088 on real data for the first time (`kern_pipe/ws_hyb`, a 30-min g6
  subsession with both `.fet` spaces staged). `looks_like_time_column` returned **False** for
  `sub.fet.stderiv_C5_D34.6` and `sub.fet.standard.6` even though the last column of each is
  `np.array_equal` to the `.res`.
- Cause: 0.67.27's guard required `np.all(np.diff(c) >= 0)`. A committed realign reorders spikes —
  `res_corr = res + ioff`, integer shifts of up to 10 samples a pass — so neighbours cross. On this
  file: **5 backward steps in 113,515 gaps (0.004 %), worst −14 samples, 121 exact duplicates**.
  Five spikes out of 113,516 were enough to make the guard reject a real time column.
- The effect was the opposite of the guard's purpose: `--fet-drop-time` silently did nothing on
  exactly the files it exists for, so `--fet-hybrid` would have stacked 33 + 25 = 58 columns
  carrying **two** time ramps — the configuration 1088 added the flag to prevent.
- `looks_like_time_column(values, monotone=0.9)` now takes a FRACTION of non-decreasing steps. A
  PCA component sits near 0.5 on this measure and a time column at 0.99996, so the separation is
  not delicate; `monotone=1.0` restores the old behaviour. On the real files the blocks now come
  out 33 → 32 and 25 → 24 as intended. 1 test, with realign-sized inversions and a duplicate.
- The lesson is the one the session keeps teaching: this passed seven tests and two generator
  gates, and was wrong on the first real file it saw, because every test fixture built its time
  column with `np.arange`.

## [0.67.29] — the residual terms reach the merge table
- `<base>.psame.<g>.<tag>.tsv` carried `into merged p rc dbic geo_z how` and stopped there, so the
  scale residual was computed, logged to stderr on a veto, and then thrown away. With 0.67.28 that
  table is the diagnostic: **the floor of each accepted merge is how a session's
  `FK_PSAME_SRESID_REF_MAX` gets chosen**, and the excess is what the score term actually charged.
  Four columns added: `sresid`, `sresid_ref`, `sresid_exc`, `sresid_pen`.
- `merge_table_rows(merges)` / `MERGE_TABLE_COLUMNS` pulled out of `main()` so the formatting is
  testable at all — it never was. A term the run did not compute (NaN, e.g. the floor with
  `sresid_ref` off) writes as an **empty cell**, not the string `nan`, so the column reads as
  missing in a spreadsheet rather than as a value. 4 tests.

## [0.67.28] — the scale residual as a score term, over the pair's OWN noise floor
Splitting a multi-unit cluster puts every child in the regime where the merge guards stop working.
Measured on the shipped defaults:

| guard | stops protecting below | why |
|---|---|---|
| scale-residual veto | `scale_min_n` 100 spikes — and `FK_PSAME_SCALE_VETO` is `0` in both profiles | a small fragment's mean is not a cell's template (AUC 0.943 → 0.764) |
| refractory cannot-link | **~1,101 spikes per cluster** at a 12-min chunk | `c_exp = 2.475e-6 · n_a · n_b`, and `refractory_gate` abstains below `min_exp` 3 |
| ΔBIC | never off, but biased **toward** merging as n falls | `dbic = [2ΔLL + k·ln n]/n` with `w_dbic` +0.6144: the complexity term is positive and grows as `ln n / n` |

A 757-spike cluster is already below the refractory threshold (`c_exp` 1.42) before any split; four
~190-spike children give `c_exp` 0.09. The ΔBIC bias moves from +0.15 to +0.48 logit (k=28) over the
same split — a free push toward merging, applied to exactly the units just separated.

- `split_half_residual(W)` — a cluster's spikes in two **interleaved** halves (so its own drift lands
  in both, not in the reference), each averaged, one fitted to the other as a scaled copy. That is the
  residual template NOISE alone produces at that spike count and SNR.
- `residual_reference(sr0_lo, m_lo, m_hi, sc, k)` = `k · sr0_lo/4 · (1 + sc²·m_lo/m_hi)`, from noise
  falling as 1/m. Validated against same-cell pairs: 2–5 % optimistic above 200 averaged spikes, ~17 %
  at 60–100. `m_*` are the spikes actually averaged (psame caps at 1000), not the cluster sizes.
- `FK_PSAME_W_SRESID` makes the residual a **score term**: the logit loses
  `w · max(0, sresid − reference)`. The veto could not do this — refusing the top pair leaves the loop
  to take the next-best, so it **reroutes** a merge instead of preventing it, and it has a cliff at
  `scale_min_n`. Only an EXCESS costs; a residual at or below the floor says nothing and must not read
  as evidence *for* one cell either.
- **Measured** (`analysis/sresid-reference/`, synthetic 32×8 template, per-spike noise sd 15): the
  different-cell excess beats the same-cell excess by 500–25,000× above 100 spikes, **74× at 50 spikes
  and 13× at 20** — the regime where `scale_min_n` switches the veto off entirely. At 20 spikes the
  same-cell cross residual reaches 0.189, so the fixed `scale_thr` 0.20 is about to start refusing
  correct merges; the adaptive floor keeps tracking.
- `FK_PSAME_SRESID_REF_MAX` refuses a merge whose noise floor alone exceeds it: at that spike count
  the shape test cannot tell the templates apart, and over-merging is the irreversible direction.
  Unlike the shape veto it does **not** reroute — the floor belongs to the dimmer cluster, so a
  cluster refused against one partner is refused against all. It also catches a *contaminated*
  cluster, whose own halves disagree and whose floor is therefore high.
- All off by default (`w_sresid` 0.0, `sresid_ref_max` 0.0): the logit is unchanged bit for bit, and
  `test_psame.py` asserts the merge list is identical. Known limitation carried over from 0.67.24 and
  now asserted in the tests: a look-alike cell that really *is* a scaled copy of its partner slips
  both the veto and the term — only `dbic` separates that one. 16 tests.

## [0.67.27] — only drop a column that IS the time, and say so when one is embedded
- Follow-up to 0.67.26, from the obvious question: *wasn't the time feature turned off?* It is — everywhere
  else. KlustaKwik's `useFeatures` default masks the last column off, which is **why** the time goes
  last (`ns3_sort_bench/adapters/fiber_kit.py::_compute_features`: "the last column is the spike sample
  index (time feature) so KK's `useFeatures` default of masking that one off works as expected"). In
  fiber-kit it is off more thoroughly still: `klustakwik.py` and `fiber_session` never read a `.fet` at
  all, they build features from the waveforms via `fiber_pca.cluster_features`. Of every `.fet` reader in
  `src/fiber_kit/`, only `fiber_consensus` uses one as a feature matrix (`fiber_subsession` slices rows to
  copy a subsession); so fiber-consensus was the one place without the mask.
- **Chunking does not mitigate it.** `consensus_neighbourhood` z-scores `features[idx]` where `idx` is the
  CHUNK's rows, so the ramp is renormalised to unit variance inside every chunk exactly like every PCA
  column. A shorter chunk narrows what the ramp spans, not what it costs: 1/d of the embedding, whatever
  `FK_CONSENSUS_CHUNK_MIN` is.
- `looks_like_time_column` — non-decreasing over the whole file — gates the drop. Not every `.fet` has one:
  `fiber_session`, `fiber_realign` and `fiber_manifold` write the time last, but `fiber_decollide`'s `.fet`
  is `cluster_features` alone, and blanket-dropping its last column would have eaten a real PCA component.
  `--fet-drop-time 1` is now safe to leave on across a whole session.
- The run log now states what happened to each block's last column, and says so loudly when one is being
  embedded: *"time column KEPT in the embedding -- KlustaKwik's useFeatures masks this one off;
  --fet-drop-time 1 does the same here"*. The default is unchanged (`auto` = drop only under
  `--fet-hybrid`), so no number already measured moves — but it is now an informed default rather than a
  silent one. 2 tests (9 on the feature path).

## [0.67.26] — fiber-consensus can embed TWO feature spaces at once (`--fet-hybrid`)
- The curator's case: a large multi-unit cluster in the 0312 g6 sort whose children separate into clean,
  well-spaced islands under t-SNE **in the standard space** — while the sort, and therefore fiber-consensus,
  clusters in `stderiv_C5_D34`. The consensus runs embed `--fet-variant`'s `.fet` and nothing else, so a
  separation that lives in the other space is invisible to them no matter how many runs they average.
- `--fet-hybrid 1` stacks a second `.fet` beside the first: `--hybrid-fet-variant` (default `standard`,
  falling back to the canonical `<base>.fet.<elec>` a Klusters session from ndm_pca holds),
  `--hybrid-scale`, `--hybrid-pcs`. This is `FK_SESSION_FEAT_HYBRID` one stage later and at the `.fet`
  level instead of the waveform's — the session knob was adopted ON (`PCS 3`, `SCALE 2.0`) in the g6
  profile, so the same argument applies to the stage that decides what a cell IS.
- `stack_blocks` divides the column counts out before weighting: after `consensus_neighbourhood`'s
  per-column z-score a block of n columns carries variance n, so a 24-column space would outweigh an
  8-column one 3:1 — a statement about how many components ndm_pca wrote, not about the spikes. Block i
  is weighted `scale_i * sqrt(n_0/n_i)`, so `--hybrid-scale 1` means "both spaces count the same".
  The weight is applied **after** the z-score, inside `consensus_neighbourhood` (`col_weight=`), because
  the z-score would otherwise flatten it away.
- **Observation, not a fix:** a Klusters `.fet`'s LAST column is the spike TIME (`data.cpp` requires the
  timestamp last; `fiber_session` and `fiber_realign` write it there). fiber-consensus passed `fet.values`
  whole into the embedding, so every consensus run up to 0.67.25 z-scored a monotonic ramp into the t-SNE
  as a drift coordinate worth 1/d of the variance. `--fet-drop-time` defaults to `auto` = drop only under
  `--fet-hybrid` (two blocks would carry two ramps); the single-space path is unchanged and still carries
  it, so nothing already measured moves. Whether dropping it helps on its own is **unmeasured** — it wants
  its own A/B before the default changes.
- Off by default; `--fet-hybrid 0` embeds exactly what 0.67.25 embedded, column for column, with
  `col_weight=None`. Three spaces are now one flag apart on the same neighbourhood: the clustering variant,
  `--fet-variant standard` (which already existed), and the two stacked. 7 tests.

## [0.67.25] — stamp each trial with its RESOLVED config: `overrides` alone never identified one
- `overrides` records only the delta from the profile, so when patch 1084 moved the realign gate, `COS 0.6`
  and `NEIGH_PASSES 2` into `fiber-kit-best.yaml`, one override dict started meaning two different
  configurations — and summary.md printed them as one row type and ranked them against each other:

  | override dict | results |
  |---|---|
  | `{}` | 0.3671 (0.67.9), 0.3817 (0.67.10), 0.3863 (0.67.12), 0.3863 (0.67.13) |
  | `{FEAT_HYBRID 1, HYBRID_PCS 3, HYBRID_SCALE 2.0}` | 0.3851, 0.3850, 0.3850, 0.3892 (0.67.10 → .15) |
  | `… + NEIGH_PASSES 1` | 0.3944 (0.67.21), 0.4180 (0.67.24) |

  `{}` — "the base profile untouched" — is the worst case. **No measurement was wrong**: fiber-bench's step
  hash distinguished every one of these and nothing was wrongly cached or conflated. Only the label was.
- `Tuner.cfg_of(cfg_run)` returns `(cfg_id, resolved surface)`. `cfg_id` is a blake2b digest of every `FK_*`
  value except `fiber_bench._PERF_KNOBS` (throughput knobs cannot change a result, so they must not change
  the identity); the surface is `Tuner.knob_surface()` — `start` plus every stage's `FK_*` knobs — resolved,
  which is the human-readable "which knob moved". All six knobs 1084 moved are on that surface.
- Both record constructors stamp `cfg_id` and `cfg` on every trial. trials.tsv gains `cfg_id`, `fiber_kit`
  and `cfg` (the resolved surface as JSON); summary.md gains `cfg` (the digest, 8 chars) and `fk` columns.
- summary.md now **warns** when an override string in the displayed table covers more than one
  configuration, listing each `cfg_id/version`. Rows written before this release have no `cfg_id`, so they
  are grouped by `fiber_kit` version instead — which is enough to flag all three cases above. They cannot be
  backfilled: reconstructing a historical resolved config would need the profile as it was then.
- Nothing that runs is changed: this is the record and the tables only. No knob added, no step hash touched,
  so no cached artifact is invalidated.

## [0.67.24] — psame: a scale-residual veto on over-merges (prototype, off by default)
- New `fiber_psame.scale_residual(Ta, Tb, shift)`: roll A's template onto B's, fit ONE amplitude scalar,
  return the residual energy relative to B's.  Distance attenuates a waveform without changing its shape,
  so one cell's energy bands differ by a scalar while two cells at matched amplitude do not.
- `psame_merge(..., scale_veto=False, scale_thr=0.20, scale_min_n=100)` refuses a merge the P(same) score
  has already accepted when the dimmer cluster's template is not a scaled copy of the brighter one.  Every
  merge record now carries `sresid`, so the statistic is visible even with the veto off.  Knobs
  `FK_PSAME_SCALE_VETO` (0), `FK_PSAME_SCALE_THR` (0.20), `FK_PSAME_SCALE_MIN_N` (100), and the CLI flags
  they generate.  **Off by default — it changes merge decisions.**
- MEASURED on g6 (`analysis/scale-residual-veto-g6-0316/`), extending the 0312 band result
  (`analysis/adaptation-energy-band-residuals-g6-0312/`) to the tuning session and then to the pipeline's
  own clusters:

  | pair set | same unit 10/50/90 % | different 10/50/90 % | AUC |
  |---|---|---|---|
  | 0316 truth templates, own energy quartiles vs other cells | 0.002 / 0.011 / 0.054 | 0.076 / 0.197 / 0.329 | **0.968** |
  | pipeline pre-psame clusters, purity >= 0.85, n >= 100 | 0.013 / 0.046 / 0.137 | 0.093 / 0.215 / 0.323 | **0.943** |
  | pipeline clusters, n >= 30, any purity | 0.018 / 0.071 / 0.278 | 0.080 / 0.201 / 0.323 | 0.764 |

  `theta 0.20` keeps 93 % of correct merges and refuses 71 % of over-merges; `0.15` keeps 90 % and
  refuses 90 %; `0.33` keeps 100 % and refuses 8 %.
- Two design questions the measurement settled against the obvious choice.  A channel mask HURTS (AUC
  0.928 at 0.25 x peak, 0.914 at 0.50, against 0.943 for the full template).  And projecting out the
  cell's amplitude-shape direction `S` — CellModel's adaptation term, and the natural way to tell an
  adapted tail from a more distant cell — hurts too (0.928 -> 0.913 at the same mask): adaptation here is
  a <= 10 % amplitude effect (amp(ISI<10 ms)/amp(ISI>100 ms) 0.90–1.01 across six GT cells) and its
  direction is nearly shared between the look-alike pairs (|cos| 0.95 for units 5 and 8).  Scale
  invariance is the whole statistic; the adaptation signature contributes nothing.
- The guard matters more than the threshold: the AUC falls 0.943 -> 0.764 once small or contaminated
  fragments are included, because such a mean is not a single cell's template.  Hence `scale_min_n`.
- Known blind spot, asserted in the tests: two genuinely different cells whose templates really are
  scaled copies of one another pass the veto.  On 0316 those are the ~10 % it misses — pairs at roll-cos
  >= 0.99 with residuals of 0.004–0.016 and amplitude ratios up to 4.9 (truth 10 vs 9, 10 vs 4, 4 vs 21).
  Shape cannot separate them; that is the de-collision/timing problem, where the earlier timing evidence
  also failed (AUC 0.40–0.60).
- Adding three `FK_PSAME_*` knobs changes `stage_params("fiber-psame-merge")`, so the cached `exp_psame`
  artifacts rehash and that one step re-runs (1 of 10) on the next bench run; the upstream steps are
  untouched.

## [0.67.23] — regenerate plans/TEMPLATE-all-stages.yaml (it had lagged four patches of new flags)
- `tools/gen_stages_doc.py --check --template` had been failing since 0.67.11: the template documents the
  argument parsers, and four patches added fiber-session / psame flags without regenerating it —
  `hybrid-gain-fit` and `hybrid-deriv-basis` (0.67.11), `align-pairs` (0.67.12), `neigh-passes` /
  `neigh-k` / `neigh-cos` / `neigh-seed` (0.67.13), and `jobs` (reworded) with `job-mem-gib` (0.67.17 /
  0.67.19).  Documentation only: the flags themselves have worked since those patches, and `docs/stages.md`
  was never out of date.  `test/run_all.py` goes back to green.

## [0.67.22] — the round-2 winner is adopted: the realign gate, the hybrid space and Block N
- **The first search winner here to survive a confirm.**  The 0.67.21 search (chunks 2-7, the realign
  inside the search) ranked winner > PASSES 1 > +RKK_MIN 20 > plain at 0.3777 / 0.3757 / 0.3752 / 0.3500,
  and `fiber-tune confirm --chunks all` reproduced that order exactly at **0.4253 / 0.4180 / 0.4068 /
  0.3724**.  Every earlier round inverted at full scale; the difference is that the search now runs the
  realign steps it is judged on (`search_realign: true`, 0.67.21).
- `tools/gen_best_profile.py`: six knobs added to `WINNER`, and `fiber-kit-best.yaml` /
  `fiber-kit-klusters.yaml` regenerated.  Against the round-1 profile with the hybrid on, at full scale:

  | | obj | acc | int | pyr | ARI | comp | purity | frag80 | merges | pyr-pyr | clusters |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | round 1 + hybrid | 0.3944 | 0.434 | 0.220 | 0.441 | 0.270 | 0.541 | **0.838** | 4.516 | 8 | 2 | 471 |
  | **round 2 (adopted)** | **0.4253** | **0.450** | **0.485** | **0.449** | **0.423** | **0.651** | 0.816 | **3.600** | **5** | 2 | 500 |

  `FK_REALIGN_MIN_SCORE` 0.10 -> 0.70, `FK_SESSION_FEAT_HYBRID` 0 -> 1, `_HYBRID_SCALE` 1.0 -> 2.0,
  `_HYBRID_PCS` 2 -> 3, `FK_SESSION_NEIGH_COS` 0.7 -> 0.6, `FK_SESSION_NEIGH_PASSES` 0 -> 2.  Purity is
  the only axis that gives ground.
- The gate is the round's largest single move (+0.028 at search scale) and it is the *label-conditioned*
  realign it tames: at 0.10 a mis-assigned spike is rolled onto the template of whatever cluster holds
  it, writing the error into the window and the `.res`
  (`claude/realign-label-conditioning-2026-11-14.md`).
- `NEIGH_PASSES 2` over 1 was a 0.0073 call on the objective, decided by the per-type columns rather than
  the merge count: **pyramidal-pyramidal merges 5 -> 2** and frag80 4.433 -> 3.600 at equal pyramidal
  accuracy (0.451 vs 0.449).  PASSES 1 keeps more interneuron accuracy (0.527 vs 0.485) and a tidier
  partition (ARI 0.455, completeness 0.679), so a run that cares about the broad high-rate units should
  set it back.
- The hybrid's case is settled at full scale and it is the opposite of the 2026-11-14 three-way: with the
  gate and Block N tuned, plain reaches **pyramidal 0.396** against the hybrid's **0.449**, while winning
  interneuron accuracy (0.603 vs 0.485), ARI (0.522 vs 0.423) and completeness (0.716 vs 0.651).  Plain
  is the choice for interneurons and the wrong one for low-amplitude broad pyramidals.
- `RKK_MIN 20` is rejected again at full scale (0.4068): it cuts 500 clusters to 372 and takes
  interneuron accuracy 0.485 -> 0.342 with it.
- `plans/tune/hybrid-search.yaml`: `start: {}` -- the winner is the shipped profile now, so the start row
  *is* the profile and every stage still has something to try (the gate sees 0.10 / 0.40 against 0.70,
  neigh sees COS 0.7 / 0.8 and PASSES 1 / 0, the control sees the hybrid off).
- `fiber-kit-exp.yaml`: the `FK_REALIGN_MIN_SCORE` comment no longer claims 0.10 is "used here" and now
  records the label-conditioning measurement and which units a 0.70 gate protects.

## [0.67.21] — tune spec: the search now runs the realign it is judged on, and the alignment gate is searched
- **The search and the confirm were not running the same pipeline.**  `search-base.yaml` is
  `flag > session > clean > cpos-clean > link > merge > psame` with no `fiber-realign` step, and
  `search_realign: false` also forces `FK_SESSION_REALIGN` off; `confirm-base.yaml` interleaves
  `realign-1/-2/-3` and turns the end-of-run commit back on.  Every knob was therefore tuned on a
  pipeline that never moves the data and then judged on one that re-cuts it between every stage.
  `plans/tune/hybrid-search.yaml` sets `search_realign: true`.  No code change — the facility was
  already there and its docstring already carried the consequence (one 6-min 0316 chunk at identical
  parameters: search ARI 0.675 / acc 0.376, workspace run ARI 0.503 / acc 0.417).
- Measured on 0316 g6 (138,319 spikes, 42x8 windows, the profile's realign settings;
  `analysis/realign-influence-g6-0316/`): the realign is label-conditioned — each spike is rolled onto
  the template of whatever cluster currently holds it — so it is a function of the labelling under test.
  89.2 % of spikes get a nonzero committed shift (mean -0.85, sd 1.86, max |21|); the fragments of one
  truth unit end up with mean shifts a median 1.83 samples apart (25/30 units >= 1 sample), so the merge
  stages must undo an offset the realign introduced; of 23,094 spikes seated in a cluster another unit
  dominates, 52 % are committed to a different time than their own unit would give (mean |d| 1.03
  samples); and it is not idempotent at fixed labels (a second pass still moves 3.6 %).
- Ruled out by the same measurement, and recorded so they are not re-chased: cumulative interpolation
  loss (the commit is integer — `roll_spikes` or a re-extraction at the corrected sample, never a
  resample), the chunk restriction (a chunks 2-3 realign agrees with the all-chunk one on 94.7 % of the
  shared spikes, median |d| 0, and no cluster falls below `min_n`), and `PARK_DUPS` attrition (7 spikes).
- New `realign` tuning stage, searched FIRST: `FK_REALIGN_MIN_SCORE [0.10, 0.70, 0.40]` and
  `FK_REALIGN_MAX_SHIFT ["", 4]`.  The profile's 0.10 is the curator's Klusters setting, not Klusters'
  factory 0.70, so the gate is effectively open.  A 0.70 gate would freeze 61.3 % of foreign-seated
  spikes against 34.4 % of own-unit ones, and it is selective for the low-amplitude broad units —
  89.3 % of unit 2's foreign-seated spikes (amp 1782, half-width 9, 0.06 Hz) score below 0.70, as do
  88.0 % of unit 34's and 96.5 % of unit 6's, while the high-amplitude units' foreign spikes mostly
  clear it (unit 13 9.5 %, unit 10 12.4 %).  Both knobs are INERT unless `search_realign: true`: the
  search plan has no realign step, and fiber-session's own rkk cluster<->realign loop uses
  `fl.align_xcorr` (ref=median, iters 6, maxlag 6) and reads none of them — the FEAT_HYBRID /
  NEIGH_PASSES stage-order trap a third time, which is why the two changes arrive together.

## [0.67.20] — fiber-bench: performance knobs out of the step hash; fiber-tune: `confirm --extra`
- `stage_params` drops a `_PERF_KNOBS` set (`FK_SESSION_JOBS`, `FK_SESSION_JOB_MEM_GIB`) from the step hash:
  without this, 0.67.17/0.67.19 adding those keys to the profiles would have invalidated every cached
  artifact and forced a full re-sort.  The older `*_JOBS` / `*_WORKERS` keys are deliberately not listed --
  they are already inside existing hashes, and removing them would invalidate the same artifacts this set
  exists to protect.
- `fiber-tune confirm --extra JSON` (repeatable) confirms configurations the search did not rank highest, and
  `--top 0` runs exactly the start plus those.  A re-baseline of several configurations under one code version
  needs this: ranking by the search objective cannot reach a row that the search scored badly (the plain
  control) or scored as a no-op (`DIP_ALPHA 0.02`, whose effect only appears across many chunks).
- `plans/tune/hybrid-search.yaml`: `start:` no longer carries `RKK_MIN 35` / `XCM_COS_THR 0.96` -- the full-plan
  confirm rejected them (0.3701 / acc 0.420 / ARI 0.280 / 10 merges against the start's 0.3892 / 0.429 / 0.378
  / 8) -- and `chunks` widens from `2-3` to `2-7`, because three search winners in a row have now reversed on
  the full plan: on two chunks anchor-link, xcorr-merge and psame have almost no cross-chunk work to get wrong.

## [0.67.19] — fiber-session: `FK_SESSION_JOB_MEM_GIB` bounds `--jobs 0`, because memory is the real limit
- Measured while testing 0.67.17: a chunk worker holds 2.5-3.4 GiB of anon RSS for a 50k-spike 12-min g6
  chunk with the hybrid stack (its chunk's windows, their realigned copies and the per-cluster work in
  flight), and TWO of them OOM-killed a 7 GiB cgroup.  0.67.17's "~1-2 GiB per worker" was an estimate from
  array sizes and it was too low; one worker per chunk on a 25-chunk session wants ~100 GiB, not ~40.
- `--jobs 0` is now additionally bounded by `MemAvailable // FK_SESSION_JOB_MEM_GIB` (default 4.0 GiB; 0
  disables the cap), read from this process's cgroup v2 limit when there is one, else `/proc/meminfo`, and the
  log says when the memory cap bit.  An explicit `--jobs N` is never reduced: whoever names a number owns the
  memory.  Scale the budget with the chunk's spike count, not the core count.
- `test_session_jobs.py` covers the memory cap, the 0 = no-cap case, unknown memory, and the one-worker floor.

## [0.67.18] — fiber-consensus: the pool is capped by the task count and reused across a neighbourhood's bursts
- `_pool(workers, ntasks)` never starts more workers than there are tasks.  Before, `workers 0` meant
  `cpu_count()` for every burst, so on a 64-core box the watershed's 5 maps started 64 processes to keep 5
  busy and the 40 consensus runs started 24 workers that got no work at all.
- One pool per `consensus_neighbourhood` call, shared by the runs and the watershed maps, instead of one
  created and torn down per burst: a spawned worker costs ~1-1.5 s of interpreter start and imports, and that
  was paid twice per neighbourhood.  `tsne_runs` / `watershed_categories` take an optional `pool=` and shut
  down only a pool they created themselves.
- CORRECTION to 0.67.14's note: there is NO numba JIT penalty on the first embedding in a process.  openTSNE's
  gradient code is compiled Cython; measured, three successive fits in one process took 19.5 / 17.2 / 18.2 s.
  The per-worker cost is the ~1-1.5 s of imports, not ~20 s of compilation.
- Occupancy note for a many-core box: a burst cannot use more cores than it has tasks, so with
  `FK_CONSENSUS_WORKERS 0` the run burst tops out at `FK_CONSENSUS_RUNS` cores (40 of 64) and the watershed at
  `FK_CONSENSUS_WS_RUNS` (5 of 64).  Raising both to the core count costs no wall clock -- 64 runs on 64
  workers take as long as 40 -- and makes the consensus and the basins more stable.  Neighbourhoods are still
  processed one at a time, and the serial sections between the bursts (envelope fit, Louvain, the quarantine
  rules) are single-threaded: cross-neighbourhood parallelism is the remaining structural win.
- `test_consensus.py`: the knob-coverage test allows `FK_CONSENSUS_TSNE_BACKEND` (a `choices=` flag, so it
  cannot live in the scalar `_KNOBS` table); it had been failing in script mode since 0.67.15.

## [0.67.17] — fiber-session: `FK_SESSION_JOBS`, and `--jobs 0` = one worker per chunk
- `--jobs` now resolves from `FK_SESSION_JOBS` (env > global config > 1), so the chunk pool is reachable from
  a profile instead of only from a hand-typed command line -- nothing in fiber-bench or fiber-tune ever passed
  `-j`, so every pipeline and tune sort has been serial.  `0` means one worker per chunk, bounded by
  `sched_getaffinity` (a cgroup or taskset quota, not `cpu_count`); `_resolve_jobs` is a module-level helper
  and never returns more than one worker per chunk, since chunks are the unit of parallelism.
- Why it matters on a big box: with `--jobs 1` the only parallelism left is BLAS inside one chunk, on matrices
  (a chunk's windows, 6-12-dim features, the per-cluster CEM) far too small to repay a wide fork/join barrier,
  and NumPy's bundled OpenBLAS has a build-time thread ceiling -- commonly 64 -- that no fiber-kit setting
  raises (`openblas_get_config()` prints `MAX_THREADS`).  Pool workers pin BLAS to one thread each, so N
  workers use N threads rather than N*N.  Budget ~1-2 GiB resident per worker (a 70k-spike 18-min g6 chunk is
  0.22 GiB per float64 window array, several in flight; the `.spk`/`.res` memmaps are shared page cache).
- The log line now names the worker count, says when `--jobs` was capped by the chunk count, and states that
  BLAS is pinned per worker.  `FK_SESSION_JOBS: 1` in the four profiles, so no cached sort changes.
- CAVEAT, documented with the knob: the serial path leaves BLAS threaded while the pool pins it, so `> 1` can
  differ from `1` in the last bits of a reduction.  It is a speed knob, not a tuning one, and it is not part
  of fiber-bench's step hash.  `test_session_jobs.py`.

## [0.67.16] — hybrid-search spec: Block N on by default in `start:`, its PASSES last; the 2026-11-14 findings
- The 2026-11-14 search on 0316 g6 chunks 2-3 (45 trials, 2 passes, core truth) put `FK_SESSION_NEIGH_PASSES`
  FIRST in the `neigh` stage; 1 and 2 both lost to the start, so coordinate ascent kept 0 and then searched
  `NEIGH_K` and `NEIGH_COS` with Block N OFF -- five trials, five no-ops.  The stage now mirrors the hybrid
  stage: `start:` carries `NEIGH_PASSES: 1` and the stage tries K / COS / the retry first, PASSES last.
- `start:` also carries the run's winners, `FK_SESSION_RKK_MIN: 35` and `FK_XCM_COS_THR: 0.96` (best 0.3889 vs
  the start's 0.3818; acc 0.419, merges 6, 175 clusters), and the `session` stage gains `FK_SESSION_COARSE_DR`
  [0.09, 0.06].
- Recorded in the spec header: `RKK_PTS_PER_START` is a no-op on 0316 chunks 2-3 (CHUNK_MIN 18, so no group is
  small enough for the CEM's start to die out); `FK_PSAME_ALIGN_PAIRS` 0 == 1 at search scale but lifts the
  full plan's plain sort (int_acc 0.363 -> 0.451, ARI 0.345 -> 0.400, merges 9 -> 7); `COARSE_DR` 0.06 trades
  all_acc 0.415 -> 0.390 and pyr_acc 0.419 -> 0.387 for int_acc 0.278 -> 0.504, ARI 0.433 -> 0.521 and
  completeness 0.622 -> 0.714 -- the largest effect in the table, which this objective rejects.

## [0.67.15] — fiber-consensus: cuML (GPU) t-SNE backend, `FK_CONSENSUS_TSNE_BACKEND`
- `auto` (default) uses RAPIDS cuML's t-SNE when `cuml` and a CUDA device import, else openTSNE; `opentsne`
  / `cuml` force one.  Parameters mapped one to one (FFT gradients, random init, early exaggeration 12 for
  TSNE_EE_ITER iterations, TSNE_ITER main iterations, the perplexity cap); the runs and the watershed maps go
  serially on the one device instead of the worker pool.  The backend is logged at start.  NOT measured here
  (no GPU on the dev box): compare one neighbourhood's communities with the CPU's before trusting it (two
  CPU schedules agree at ARI ~0.73; UMAP, tried as a cheaper embedding, agrees at 0.54-0.58 and loses 0.04-
  0.05 ARI against the curated units, so it is not a substitute -- `analysis/consensus-tsne-schedule-g6-0312/`).

## [0.67.14] — fiber-consensus: t-SNE schedule knobs, short schedule in the profiles (1.8x faster)
- `FK_CONSENSUS_TSNE_EE_ITER` / `FK_CONSENSUS_TSNE_ITER`: the early-exaggeration and main-phase iteration
  counts of every t-SNE run (consensus runs and watershed maps); code default 250 / 500 (openTSNE's), the
  profiles set 125 / 250.  Measured on a 9,655-spike g6 neighbourhood (0312, units 12/27/49, 32-dim .fet, 40
  runs at 70 %, 1 thread): a run is 25 s of which 90 % is gradient iterations (affinities 2 s, so precomputing
  the high-dim kNN once was not the lever; PCA-12 input saved 15 % but moved the embedding, kNN overlap 0.29);
  125 / 250 takes 13 s and gives the same consensus -- Louvain communities at ARI 0.73 to the full schedule's,
  which is where two full schedules land against each other (Barnes-Hut vs FFT gradients 0.74, the first 20
  of 40 runs 0.71), ARI to the curated units 0.40 vs 0.41.  The first openTSNE call in a process is a ~20 s
  numba compile, paid once per spawned worker.  `analysis/consensus-tsne-schedule-g6-0312/`.

## [0.67.13] — fiber-session: Block N (neighbourhood re-cluster), the rkk die-out retry; hybrid-search spec
- **Block N** (`FK_SESSION_NEIGH_PASSES` 0-2, `_K` 3, `_COS` 0.7, `_SEED`; default off): after the resplit
  block, each cluster in random order draws 1..K untouched neighbours (template cosine >= COS), the union is
  realigned per cluster and featurised on the clustering basis (hybrid block included), and KlustaKwik runs
  ONE seeded CEM from the current labels (`klustakwik.klustakwik_refine`, which returns the penalised score
  before and after on the same normalised data).  The new partition is adopted only when KK's own score
  improved; spikes KK sends to its noise cluster keep their old label (the pass moves, splits and merges, it
  never unsorts); new clusters inherit the id of the old cluster holding most of them.  Per-cluster rkk can
  only split, so a boundary the coarse pass drew through one unit was never revisited.  0312 g6 chunks 0-1
  (hybrid scale 2 / 3 PCs, Hamming pre-aligned, units >= 300 spikes): acc 0.582 -> 0.600 (1 pass) -> 0.609
  (2), int_acc 0.509 -> 0.568, ARI 0.564 -> 0.654, completeness 0.738 -> 0.808, core precision 0.720 ->
  0.751, merges 4.5 -> 4.5, no unit worse by > 0.01; the big interneuron 0.67 -> 0.85 from two fragments to
  one.  `test_neigh_recluster.py`.
- **rkk die-out retry** (`FK_SESSION_RKK_PTS_PER_START`, default 0; 21 to try): the CEM starts at --rkk-max
  (50) clusters whatever the group size, and the M-step deletes every cluster with n <= dims -- at N <= ~120
  spikes that is all of them in the first step, and the whole group lands in the reject cluster (0312 g6:
  single curated units at 25-100 spikes -> 0 clusters, 100 % reserve, at any min group; a 12-min chunk holds
  144 spikes of a 0.2 Hz unit).  With the knob > 0, a first start that returns no cluster is retried with K
  capped at N // knob (>= 2): a 60-100-spike unit then keeps 65-95 % of itself in one cluster; groups the
  first start clusters are untouched (identical labels).  A general cap was measured to send 30-60 % of
  150-250-spike units to the reject cluster, hence a retry.  `test_rkk_small_group.py`;
  `analysis/rkk-min-spikes-g6-0312/` has the full count sweep (pairs at cosine <= 0.42 separate at 150-250
  spikes per unit; cosine >= 0.8 never, at any count).
- `plans/tune/hybrid-search.yaml`: `start:` is the clean run's winner (hybrid on, scale 2, 3 PCs); a `neigh`
  stage (NEIGH_PASSES / K / COS, RKK_PTS_PER_START) is searched first; `FK_PSAME_ALIGN_PAIRS [1, 0]` joins
  the psame stage; the hybrid stage is the on/off control, last.
- fiber-tune (0.67.10 follow-up): `Tuner._inputs` defaults to None so a Tuner built without `__init__`
  (test_gt_review.py) skips the inputs check instead of failing.

## [0.67.12] — fiber-psame-merge: dbic on pair-aligned windows (`FK_PSAME_ALIGN_PAIRS`, on)
- The ΔBIC evidence is now scored with B's windows rolled by the shift that maximises the pair's roll-cos
  (`rollcos_shift`, `project(..., roll=)`), so it is about shape, as rc already was, and not about where each
  cluster's realign put its peak.  Why: in the full plan an in-place per-cluster realign (`realign-3`) runs
  before psame and leaves two fragments of one unit at an offset to each other; on raw windows the union
  model reads that offset as a shape difference and calls them two cells.  On g6 0316 (2026-11-13) the same
  hybrid xcm sort scored 0.397 through the plan's psame (after realign-3) and 0.440 on un-realigned windows
  (units 9/10/14/15 and the big interneuron merged or not), and the plain sort flipped other units
  (21, 28 vs 9, 14, 15) for a net 0.406 vs 0.399 -- the search-time hybrid margin (+0.03) was being
  undone by this, not by the features.  `FK_PSAME_ALIGN_PAIRS 0` restores the raw-window dbic.
  `test_psame.py`: an offset fragment collapses dbic on raw windows (0.31 -> -3.0) and merges again
  pair-aligned.

## [0.67.11] — fiber-session: energy-aware hybrid knobs (measured: not an improvement)
- `FK_SESSION_HYBRID_GAIN_FIT` (all | low | high): the energy half of the sample the block-equalising gain is
  fit over.  Over `all` the top energy quartile (4x the per-spike feature variance of the bottom one) sets the
  gain and the low three quartiles load 0.45 on the raw block at scale 1 (nominal 0.50; 0312 g6).
  `FK_SESSION_HYBRID_DERIV_BASIS` (session | high): the derivative block's basis refit per channel on the
  high-energy half (`fiber_pca.refit_basis`, `energy_half`) -- the low half's derivative variance is nearly
  flat across components (0.10/0.13/0.10 of it on the session basis, a third in all) where the standard
  space keeps 0.58.  Both default off.  Measured on 0312 g6 chunks 0-1 (Hamming pre-aligned, hybrid scale 2 /
  3 PCs, units >= 300 spikes): base 0.582, deriv basis high 0.572, gain fit low 0.579, both 0.577 -- every
  well-sorted unit identical, the fragmented small ones a little worse; the imbalance is too small to matter and
  the clusters that move have the least signal.  `analysis/pca-low-vs-high-energy-g6-0312/loading.py`,
  `test_hybrid_energy.py`.

## [0.67.10] — fiber-tune: confirm and the summary rank the latest run, trials carry an inputs id
- Every trial record is stamped with `inputs`, a digest of fiber-bench's input fingerprint (the session
  files the run reads); `_current` requires it to match the files on disk now, as it already required the
  truth id.  `confirm` picks the start and the `--top` best from the LATEST search run only (from its last
  `start` record on) and from current rows; the summary's search table does the same and says how many
  earlier rows it leaves out; `rescore` skips rows sorted from other files.  Why: on g6 0316 the stale first
  run's rows (0.416, sorted from the pre-realign .spk) outranked the clean run's (0.376) in one trials.tsv
  and `confirm --top 3 --chunks all` spent 4 x 44 min on them -- plain-config rows (`FK_SESSION_FEAT_HYBRID`
  0) whose SCALE/PCS knobs were no-ops, so four identical runs (acc 0.412, objective 0.367 on the full plan,
  which is at least the plain control).  `test_tune.py` covers the inputs stamp and the latest-run selection.

## [0.67.9] — fiber-realign: Hamming window, on by default (+-16 samples)
- `--kernel hamming` (0.54 + 0.46 cos over +-width; Klusters 0972), and the window is now the DEFAULT:
  `FK_REALIGN_KERNEL hamming`, `FK_REALIGN_KERNEL_WIDTH 16` (samples of the recording, 0.49 ms at 32,552 Hz).
  Measured on the 0312 g6 sub-session (113,516 spikes, 21 curated units >= 200 spikes, the commit judged on
  its re-cut raw windows): in the standard space the trough-position SD per unit 0.80 -> 0.78 samples at
  +-16 (0.76 at +-10, 0.74 with the template window), on the 12,235 spikes with a second trough in their
  window 1.08 -> 1.05 (1.01 / 0.98), central residual 197.0 -> 196.3, 81-90 % of units tighter; in the
  stderiv_C5 space no measurable change (the derivative already localises the energy).  `none` restores
  the pre-0.67.9 kernel.  `test_realign_kernel.py` covers the window.

## [0.67.8] — fiber-tune: score each truth unit on its core (`truth: core:`)
- `truth_core` / `Truth(..., core=)`: per unit, the spikes within `k` robust SDs of the median residual RMS
  from the unit's realigned median template (on the session's raw windows) are the core; the rest are
  matched to no truth at all.  A curated interneuron holding other cells' spikes no longer caps the
  accuracy of a sort that cleans it, nor charges a merge to the cluster that isolates its contamination.
  Spec `truth: core: {k: 3.0, min_n: 50, variant: standard}`; the truth id carries it, so `rescore`
  re-scores recorded trials without re-sorting; per-unit cores logged.  `test_truth_core.py`; docs/bench.md.

## [0.67.7] — fiber-realign: Klusters' correlation window (--kernel)
- `klusters_offsets(kernel=, kernel_width=)`, `realign_kernel_weights`: the window Klusters' realign gained in
  neurosuite-3 patch 0971 (`--kernel none|gauss|hann|template`, `--kernel-width`), ported step for step.
  Under a window the score is the template's correlation with the part of the spike the window admits,
  normalised by that part's energy at each lag, so a collision in the tail or lead-in cannot steer the lag;
  `none` is the classic kernel, bit-identical to before.  CLI `--kernel` / `--kernel-width`, knobs
  `FK_REALIGN_KERNEL` (none) / `FK_REALIGN_KERNEL_WIDTH` (0 = half the peak position), in the profiles and
  so in fiber-session's end-of-run commit and `fiber-tune confirm`.  `test_realign_kernel.py` reproduces the
  C++ test vector (two collided spikes: whole window loses them, Gaussian +-4 and template align both).

## [0.67.6] — fiber-realign: a commit leaves super-sampled (_U) spaces alone
- `fiber-realign --variants all` (what fiber-session's end-of-run commit and `fiber-tune confirm` run)
  listed the `_U<factor>` super-sampled spaces among the waveform variants to refresh and derived each as
  its base-rate family: a 52-sample window written under the `_U4` name (a corrupt space, factor x
  nSamples ticks expected), its tick `.res` and phase column never touched.  Reachable the moment a session
  has a Klusters- or ndm-built `_U` space beside the spaces the pipeline commits (g6 0316, 2026-11-12).
  `supersampled_variants_on_disk` / `_drop_supersampled`: `_U` tokens are skipped with a log line naming
  the file and how to rebuild it (Klusters' Sync Feature Spaces, `ndm_extractspikes --supersample`); a
  realign asked for IN a `_U` variant is refused with the base-rate token named.  `test_realign_supersampled.py`.

## [0.67.5] — fiber-bench: the session files a run reads are in its step hashes
- `fiber_bench.input_fingerprint(base, elec)`: size + mtime of every `<base>.<spk|pca|nst|res>.<method>.<elec>`
  without a hash tag (stage outputs and `.fet` excluded) plus `<base>.xml` / `.yaml`, folded into every step's
  hash by `build_run(..., inputs=)`; `fiber-bench run` and `fiber-tune` pass it, `--no-input-hash` leaves it
  out.  Why: on g6 0316 (2026-11-12) the `.spk` files were realigned between two `fiber-tune` runs; the second
  reused the first's cached sorts (made from the old windows) and re-ran the merge stages on the new ones --
  its "start" rows mixed two spike files and its fresh re-sorts disagreed with its cached ones (0.4026 vs
  0.3487 for the same config), and nothing in the table was comparable.  A rewritten input now gives every
  step a new tag.  `fiber-tune`'s confirm workspace copies with `cp -p`, so the planned hash is the one the
  workspace run makes.  `test/test_bench_inputs.py`; `docs/bench.md`.

## [0.67.4] — merge-stage templates keyed by the cluster, not by the run's random stream
- `fiber-xcorr-merge`, `fiber-anchor-link`, `fiber-psame-merge`: the `SPK_CAP` sample a cluster's template
  (anchor-link: fragment template; psame: model cap, mean and geometry fit) is built from was drawn from one
  `default_rng(seed)` shared by the whole run, so which spikes a cluster got depended on how many draws came
  before it -- on every other cluster of the partition.  Traced on g6 0316 chunks 2-3 (the hybrid-search
  trials `93098823` / `a579931f`): from spike-identical linked clusters, one run's xcm fused units 34+32+33
  into a 2,740-spike cluster and 19+14+8 into another, the other kept them apart -- 0.036 of the tune
  objective (18 x `min_gain`), attributed to the hybrid feature space until the chains were compared stage
  by stage.  Now `fiber_lib.content_rng(idx)` / `fiber_lib.subsample(idx, cap)` seed the draw from the
  cluster's own sorted spike set: a template is a function of the cluster alone.  Knobs
  `FK_XCM_SAMPLE_KEYED`, `FK_ALINK_SAMPLE_KEYED`, `FK_PSAME_SAMPLE_KEYED` (1; 0 = the old shared stream),
  listed in the profiles so fiber-bench's stage hashes change and cached link/merge outputs are rebuilt
  while the sorts are reused.  `test/test_sample_keyed.py`.
- Every fiber-tune table scored at or after `exp_linked` so far carries merge-stage noise of this size
  between re-sorting trials; `fiber-tune search` in an existing tune directory re-runs the link/merge
  steps of every recorded trial (cheap: ~1-2 min per trial) and rescores.

## [0.67.3] — hybrid-search: start with the hybrid on, try off last
- `plans/tune/hybrid-search.yaml`: the first 0316 run (chunks 2-3) tried `FK_SESSION_FEAT_HYBRID 1` at the
  defaults first; it did not beat the start, and coordinate ascent then searched `HYBRID_SCALE` /
  `HYBRID_PCS` with the hybrid OFF -- six re-sorts identical to the start (0.4159 to the last digit).
  The spec now `start:`s with the hybrid on and tries `FEAT_HYBRID 0` as the LAST value of the stage,
  against the best scale / pcs / lag found; a rerun in the same `fk-bench/tune/hybrid-search/` reuses the
  two sorts it already has (hash-matched).  `test_gt_review.py` checks the order.

## [0.67.2] — hybrid-search: the rkk stage ordered by what was measured
- `plans/tune/hybrid-search.yaml`: on the three messiest clusters of the 0312 chunk, rkk restarts
  without the starting-K sweep changed nothing, the sweep acted (completeness 0.26 → 0.76 at unchanged
  purity, reserve 244 → 41) and restarts helped on top of it; the rkk stage now searches
  `FK_SESSION_RKK_MIN` first, then `RKK_STARTS` [1, 4, 8], `RKK_REALIGN_ITERS` [2, 4].

## [0.67.1] — rkk effort knobs and the hybrid-search tune spec
- **`FK_SESSION_RKK_STARTS`** (random restarts per starting K, best score wins), **`FK_SESSION_RKK_MIN`**
  (a starting-K sweep from this value up to `--rkk-max`, as the classic KlustaKwik driver does; 0 = one
  start at `--rkk-max`, the behaviour so far) and **`FK_SESSION_RKK_REALIGN_ITERS`** (the
  `--rkk-realign-iters` default): the "let rkk work harder on a coarse fiber" knobs.  The CEM itself
  always runs to convergence, so effort means more of the search around it; `klustakwik.py` had
  `n_starts` and the sweep already, `_rkk_labels` now passes them.  On a 3-cluster toy a sweep 4..12 × 3
  starts scores −755 against the single start's −685 and returns 3 clusters where the single start
  returned 5.  `test/test_rkk_effort.py`.
- **`plans/tune/hybrid-search.yaml`**: a fiber-tune spec on `fiber-kit-best.yaml` (the previous
  fk-tune run's best.yaml folded into the profile) searching the hybrid feature space
  (`FEAT_HYBRID` / `HYBRID_SCALE` / `HYBRID_PCS` / `FEAT_LAG`), the rkk effort and dimensionality
  (`RKK_STARTS` / `RKK_MIN` / `RKK_REALIGN_ITERS` / `session:rkk-dims` / `DIP_DIM`), then DIP_ALPHA /
  INCLUSION_K and the downstream merges; knobs the exp-search found inert are left at the template.
  Checked by `test/test_gt_review.py`; `docs/bench.md`.

## [0.67.0] — hybrid standard × stderiv fine-split features
- **`fiber-session --feat-hybrid` (`FK_SESSION_FEAT_HYBRID`, with `FK_SESSION_HYBRID_PCS`,
  `FK_SESSION_HYBRID_SCALE`)**: the fine-split features become `[stderiv block (lag space under
  `--feat-lag`, else PC1-3) | scale × standard-basis PCs of the paired raw `.spk.standard` window]`.
  Why: the derivative spaces lose broad, low-energy units -- d′(C5)/d′(standard) = 0.93 / 0.61 /
  0.44 / 0.26 at 1× / 1.5× / 2× / 3× the pyramidal width on 20120312 g6, at every amplitude -- and
  the C5 operator is near-singular, so the raw window has to be read, not reconstructed
  (`analysis/derivative-penalty-g6-0312/`).  Plumbing: `fiber_pca.hybrid_stack` appends the raw
  windows as trailing channels of the fine-split arrays and the basis carries `_hyb`
  (`hybrid_basis`, `hybrid_features`, `fit_hybrid_gain`), so the same call sites that take `_lag`
  take the hybrid; `fiber_lib.set_align_channels(n)` (`FIBER_ALIGN_CHANNELS`) makes the aligner's
  xcorr and centroid seed vote on the first `n` channels only while the shift applies to all, so the
  raw block follows the stderiv alignment.  `--emit-fet` names the space `<method>_D<lag><per+pcs>`
  (`stderiv_C5_D36` at lag 3 with 2 PCs); no `.pca` is written for it (two bases), so Klusters shows
  it but cannot refeaturize or realign in it, and a later standalone `fiber-realign` leaves its
  `.fet` stale (the in-run `--realign` commit rewrites it).  Off by default; untuned
  (`claude/hybrid-feature-tuning-plan-2026-11-11.md`).  `test/test_session_hybrid.py`.

## [0.66.8] — one tick `.res` per factor, named after the shared `.res`
- **`neuro_io.supersampled_res(base, group, factor, preferred)`** / **`read_res_ticks`**: the
  group's tick `.res` at a factor is ONE file for every `_U<factor>` space, named after the shared
  base-rate `.res` (`.res.stderiv_S3.6` → `.res.stderiv_S3_U4.6`; `standard_U4` for an untagged
  one) -- neurosuite-3 0962 corrects 0958's per-space naming.  `custody_vectors.tsv` synced (+7
  rows, kind `supersampled_res`; contract hash updated).

## [0.66.7] — a `.res` under a `_U` token is in ticks; the shared `.res` stays base-rate
- **`neuro_io.base_rate_token(variant)`**: the `_U<factor>` dropped, kind and lag kept
  (`stderiv_C5_U4_D34` → `stderiv_C5_D34`).  Klusters (neurosuite-3 0958) now writes a
  super-sampled session's spike times in **ticks** (`sample × factor + phase`) to
  `<base>.res.<token>_U<f>.N`; it is never the shared base-rate `.res`: `resolve()` takes a `_U`
  method's `.res` at the base rate (its `.spk` stays its own), `resolve_any()` tries a `_U`
  preferred as its base-rate token and never scans to a `_U`-tagged copy.  `read_res` of a session
  that carries both therefore still returns acquisition samples, as `fiber-manifold` assumes.
  `custody_vectors.tsv` synced from neurosuite-3 (+19 rows, kind `base_rate_token`; contract hash
  updated); `test_custody_token_conformance` 182 checks.

## [0.66.6] — the manifold plugin follows the space Klusters shows
- **`fiber-kit-manifold.xml`**: `FK_MANIFOLD_SPK` defaults to `%spk`, the context token neurosuite-3
  0951 substitutes with the shown `.spk`'s method token (`standard`, `stderiv_C5`, `standard_U4` in
  super-sample mode), so a refine run from Klusters fits in the space on screen; a named variant
  still overrides it.  `test_klusters_contract.py` +1 (the argv Klusters assembles carries
  `FK_MANIFOLD_SPK=standard_U4` from the context); an unsubstituted `%spk` (an older Klusters) falls
  back to `standard`, `test_manifold.py` +1 (38).

## [0.66.5] — the `_U<factor>` super-sampled token; `fiber-manifold` in a super-sampled space
- **Token grammar** (neurosuite-3 0949): `<family>[_<kind><order>][_U<factor>][_D<lag><dims>]`.
  `_U<factor>` (one digit, 2-9) marks a super-sampled waveform space (Klusters' super-sample mode:
  factor ticks per acquisition sample, the `.spk` factor × nSamples long, the peak at 0-based tick
  (peakSampleIndex − 1) × factor); it belongs to the waveform token.  `neuro_io.parse_variant_token`
  gains `supersample`, `waveform_token` keeps the suffix, `supersample_of` and `waveform_geometry`
  are new; `test/custody_vectors.tsv` carries the canonical table's 19 new rows (contract.yaml hash
  updated), `test_custody_token_conformance.py` runs the `supersample_of` kind.
- **`fiber-manifold --spk-variant standard_U4`**: the geometry follows the token (factor × nSamples
  ticks, the peak scaled), lags are searched in ticks (`FK_MANIFOLD_LAG_MS` × factor) and the lag
  column stays in µs; with no `.nst` of the space, the base space's covariance is super-sampled by
  the same natural cubic spline (`spline_interp_matrix`, `supersample_cov`; C_up = L C Lᵀ plus a
  small diagonal for the rank the interpolation lacks and the int16 rounding) -- a white model
  would have seen fields in the correlated noise of the interpolated ticks (39% false at ×2);
  `--signal` is not used in such a space.  `test_manifold.py` +2 (37).

## [0.66.4] — `fiber-manifold` in any waveform space (`FK_MANIFOLD_SPK`)
- **`FK_MANIFOLD_SPK`** (`--spk-variant`, default `standard`): the `.spk.<variant>.<g>` the fields are
  fitted in -- the raw waveforms, `stderiv`, a whitened space, any variant the session holds (every
  one goes into the Klusters sub-session).  The model is the same in any space (the unit's own
  dominant-channel shape in that space, a spatial decay); the noise covariance comes from the matching
  `.nst.<variant>`, else `--signal`, else white.  A named variant that is absent is refused by name
  (custody's fallback to `standard` no longer applies to an explicit choice).  A `text` parameter of
  the Klusters descriptor, a knob of the `klusters-manifold` profile; `test_manifold.py` +3 (34): the
  planted field is found in the temporal-derivative space as well.

## [0.66.3] — the manifold plugin is a Klusters REFINE
- **`fiber-kit-manifold.xml`** declares `<integration>refine</integration>`: Klusters (neurosuite-3 0948)
  runs it on the atoms of the selected units -- the sub-session's seed `.clu` carries the parent ids and a
  seed `.clc`/`.clp` the atoms -- and reads the final `.clc` back as the new atoms of the same parents,
  so the parents keep their ids (a `recluster-integrate` plugin would have replaced them with new
  ones).  `fiber-manifold` already honours a seed hierarchy at its input tag (the atoms of the units it
  does not examine are kept; the `.clp` maps every child to its parent); `test_manifold.py` +4 (31)
  exercises exactly that contract on a sub-session, `test_klusters_contract.py` +1 (17).
  The wheel's *Recluster ▸ Plugin* ring carries it as **Refine**.

## [0.66.2] — the shipped Klusters descriptors were not well-formed XML
- **Fix**: `fiber-kit-recluster.xml` and `fiber-kit-manifold.xml` carried `--` inside their header
  comment, which XML forbids; Klusters' parser refused both, so neither plugin appeared in its
  *Plugins* menu or *Preferences → Plugins* (neurosuite-3 0947 now says so instead of staying silent).
  The comments use a dash.
- **`test/test_klusters_contract.py`** (16 checks): every shipped descriptor parses, follows the schema
  Klusters reads (`<n>`/`<value>`, known `<type>`, `<choices>` for a choice, defaults within min/max,
  the `<klusters>` block with `%params`/`%final`), every `FK_*` parameter of the manifold descriptor is
  a knob the stage reads and `FK_MANIFOLD_CHANNELS` is the `off|full` choice; and the argv Klusters
  assembles from the INSTALLED descriptor (its `paramArgs` rule, the run dialog holding
  `FK_MANIFOLD_CHANNELS=full`) is accepted by `fk run` and reaches the stage.

## [0.66.1] — `fiber-manifold --channels-set off | full`
- **`FK_MANIFOLD_CHANNELS`** (`--channels-set`, default `off`): the channels the field is fitted on.
  `off` = the unit's off-spike channels (the neighbours' fields, the unit's own template kept out of the
  fit); `full` = every channel of the group, the unit's own included -- the residual on its strong
  channels (shape / amplitude misfit, a collision on its own channels) is fitted with the same model and
  the position spans the whole group.  g6 0312 unit 5: 22.4% at SNR ≥ 4 on `full` against 19.6% on
  `off`, 63% of them coincident, 5% coincident at the dominant channel.  A `choice` parameter of the
  Klusters descriptor; in the `klusters-manifold` profile.  `test_manifold.py` +3 (26).

## [0.66.0] — `fiber-manifold`: a unit's off-spike-channel field manifold, in sliding windows, as children and a feature space
- **`fiber-manifold`** (new stage, `src/fiber_kit/fiber_manifold.py`; section [G] of `fiber-kit-exp.yaml`,
  `FK_MANIFOLD_*`): after a unit's own template is subtracted, the channels it has little power on still
  carry the fields of neighbouring cells that fired within the window and its own far field.  Per spike
  the stage fits one field event, A·g((t−τ)/w)·exp(−|ch−p|/λ), on those channels (noise-whitened matched
  filter from the group's `.nst` or `--signal`'s spike-free snippets; coarse lag × position × decay grid
  then refinement) in **sliding windows** (`FK_MANIFOLD_WINDOW_MIN` 20 every `FK_MANIFOLD_STEP_MIN` 5), each
  window with its own template (the per-sample median of its best-fitting spikes: drift followed, a
  minority's fields kept out of the template).  Writes `.clc`/`.clp` at `<stage>_manifold` (children =
  time bin × class: plain | position bin × before / coincident / after; ≤ `FK_MANIFOLD_MAX_CHILDREN`; the
  `.clu` unchanged -- nothing moves between units), `.fet.manifold.<g>` (snr×100 | lag_µs | position×100 |
  decay×100 | amplitude | window | class | isi_flag | amp_dom | time: a feature space Klusters lists
  under *Space*) and `.mft.<method>.<g>.<tag>.tsv`.  The ISI flag marks, of each pair closer than
  `FK_MANIFOLD_REFRAC_MS`, the member farther from the unit's median waveform.  g6 0312 unit 5 (311k
  spikes, 67 windows): 19% of spikes carry a field at SNR ≥ 4 (noise 0.7%), 119 children, 20 s serial.
- **Multi-CPU and GPU**: units run in parallel (`FK_MANIFOLD_JOBS`, 0 = one process per CPU, fork; the
  BLAS threads split between the workers); `FIBER_KIT_GPU=1` puts the matched-filter products on the GPU
  through `fiber_kit.backend` (CuPy).  Both paths give the serial result bit for bit.
- **fiber-pipeline**: `fiber-manifold` is a POST stage (plannable, shielded in chunk runs; `in: ""` reads
  the base clu and writes tag `manifold`); the post-run summary lists the rdr and manifold outputs.
- **Klusters plugin**: `fk install klusters-plugin` now also installs `fiber-kit-manifold.xml` (a
  `recluster` plugin on the sub-session with the `FK_MANIFOLD_*` knobs as parameters) and the
  **`klusters-manifold` profile** (`fiber-kit-klusters-manifold.yaml`: the one step) it points at.  The
  Klusters side (neurosuite-3 0946) carries the sub-session's `.fet.manifold.<g>` back as a
  session-level feature space.
- **`fk run --FK_<knob> value`** (or `--FK_<knob>=value`): any knob on the command line, exported for the
  run so it wins over the config (how a descriptor's parameters reach the stage); refused outside `fk run`.
- Docs: `docs/pipeline.md` "Refining a unit", `docs/stages.md`, `docs/config.md`, README; tests
  `test_manifold.py` (23 checks: the planted field found, lag and position right, children ≤ N, the
  parallel path identical, the pipeline step, the `klusters-manifold` plan on a sub-session) and
  `test_profiles.py` (+5).

## [0.65.0] — the `klusters` profile: what a Klusters recluster runs, whatever the session
- **`fiber-kit-klusters.yaml`** (profile `klusters`), generated by `tools/gen_best_profile.py` beside
  `best`: the tuned profile made session-agnostic for a recluster SUB-SESSION -- `FK_CHUNKS ""`
  (a selection is reclustered whole), `FK_SESSION_SKIP_EXTANT 0` (never resume), `FK_REALIGN_VARIANT ""`
  (the plugin's `method` decides), `fiber-rdr` dropped from the plan.  `fk list config` / `fk set config
  --profile klusters` / `fk run --profile klusters` know it.
- **`fk install klusters-plugin`** installs it beside the descriptor and points the descriptor's `config`
  parameter at that copy, so Klusters passes it at run time whatever session is open; the session
  folder's `fiber-kit.yaml` is no longer what a recluster runs.  An edited installed profile is kept by
  a later install (`--force` restores the shipped one).  The descriptor's `method` default is `%variant`
  (the open clustering's; neurosuite-3 0941 substitutes it) and `profile` offers `klusters`.
- test_profiles: 43 checks; `gen_best_profile --check` covers both generated profiles.

## [0.64.0] — the Klusters plugin descriptor, `fk run --profile`
- **`fk install klusters-plugin`** copies the shipped Klusters plugin descriptor
  `src/fiber_kit/klusters/fiber-kit-recluster.xml` into Klusters' plugin directory (per-user data
  location, or `$KLUSTERS_PLUGIN_PATH`; `--dir` elsewhere; a differing copy needs `--force`).  It is a
  `recluster` plugin on the sub-session (the Klusters <-> sorter contract: neurosuite-3 PLUGIN_API.md):
  `<command>fk run %group --session %base --final-clu %final %params</command>`, with typed parameters
  `profile` (choice), `config` / `plan` (path), `method` edited in Klusters' Preferences → Plugins.
- **`fk run --profile <name>` / `--config <file>`** run with that config (knobs and plan) instead of
  the session folder's fiber-kit.yaml (`FK_CONFIG` for the run).
- test_profiles: 40 checks.

## [0.63.0] — sub-sessions: a plan on the spikes of chosen clusters (the Klusters recluster contract)
- **`fk subsession export <base> <g> --clu <clu> --clusters ...`** writes a sub-session: the selected spikes
  as a session of their own in the session directory -- row subsets of every `.res`/`.spk`/`.fet` variant
  (time order, the same in every file), a seed `.clu` with their current ids, copies of the parameter files,
  and symlinks to `.fil`/`.dat`/`.nst`/`.pca`/`.eap`/`.tcl` so realign, deconv, consensus and refeaturize read
  the real recording.  `fk subsession clean` removes it with everything a run wrote.
  `fiber_subsession.export()` is the reference implementation of the files Klusters writes for *Recluster
  with fiber-kit* (neurosuite-3 0938).
- **`fiber-pipeline` / `fk run`: `--session <base>`** (the CLI spelling of `FK_DIR`/`FK_SESS`) and
  **`--final-clu <path>`** (copy the run's last `.clu` -- the plan's last `out:` step or the last canonical
  stage -- and its `.clc`/`.clp` to `<path>`; the dry run names it).
- docs/pipeline.md "Reclustering a selection"; test_subsession (23 checks: subsets equal the originals' rows,
  a stage and a plan run on the sub-session and deliver one id per selected spike, refusals, clean).

## [0.62.0] — the `best` config profile, `fk set config` / `fk list config`, shell completion
- **`fiber-kit-best.yaml`**, a third shipped profile: fiber-kit-exp.yaml with the fiber-tune exp-search
  winner on g6 0316 (2026-11-07) applied -- the plan without the per-chunk P(same) step, `psame` at theta
  0.9 chunk-aware, `FK_SESSION_CFIBER_Q` 0.85, `FK_SESSION_INCL_ASSIGN` 1, `FK_SESSION_DIP_ALPHA` 0.03,
  `FK_CONSENSUS_THETA_CELL` 5.0, `FK_XCM_COS_THR` 0.94 -- and its knobs ordered by RELEVANCE: [T1] the
  knobs the search moved (each with the value it replaced), [T2] the ones it tried that were inert, [T3]
  what is set per session (chunks, variant, resume, chunk grid, realign modes), then the stages the plan
  runs in plan order, then the stages it does not.  GENERATED from fiber-kit-exp.yaml by
  `tools/gen_best_profile.py` (every knob once, comments verbatim); `run_all.py` checks it is current, and
  docs/config.md gains a `fiber-kit-best.yaml` column (bold where tuned).  Not yet confirmed with
  `fiber-tune confirm --chunks all`.
- **`fk set config --profile <name>`** writes a profile (`base`, `exp`, `best`; a file name or a path) as
  `./fiber-kit.yaml` in the working directory, refusing to overwrite without `--force`; **`fk list config`**
  lists the profiles (tagline, knob count, plan steps) and which one `./fiber-kit.yaml` is (`*` exact, `~`
  edited).  `fiber-kit-init` gains `--profile` / `--list` and is the same operation.
- **`fk completion bash|zsh`**: tab completion for `fk` -- subcommands, `config`, `--profile` with the
  profile names, and the plan's step names (read from `fk list --names`, honouring `--plan`).
- test_profiles: 33 checks.

## [0.61.0] — clean-core metrics, fiber-tune rescore, fiber-gt-review
For the goal of clean-but-incomplete cores linked across the session, and for refining the curation
they are scored against.
- **Clean-core metrics** (`fiber_bench.core_metrics`; fiber-bench `score` and every fiber-tune trial):
  `<g>_core_prec` (a unit's core = the cluster holding most of its spikes; only other curated cells count as
  contamination, not spikes the curator left in noise), `<g>_noise_share` (those, reported), `<g>_link`
  (share of the unit's sorted spikes in its core: linking across the session), `<g>_cov` (share kept at
  all), `<g>_clean` (precision >= 0.95 and link >= 0.8) and `core_purity`.  An unsorted spike costs only
  coverage, so a cleaner, less complete sort no longer scores worse.  The fiber-tune summary shows them.
- **`fiber-tune rescore`**: scores every recorded trial again from its stored .clu with the current truth
  and metric set; nothing is re-run.  Trials record the metric set (`SCORE_VERSION` s2 in the truth id), so
  a resumed search re-scores older trials instead of lacking the new names.
- **`fiber-gt-review`** (new stage): where a sort and the curation disagree, waveform (raw roll-cosine)
  and refractory (cross-refractory ratio) evidence for which is wrong -- curated units violating their own
  refractory period, sort clusters holding two curated units or curator-noise spikes, curated units split
  over sort clusters -- as a TSV and a review .clu with every listed spike set as a cluster.  On the g6
  0316 curation, unit 21 (10 Hz, 39% of the spikes) shows 549 intervals of (0.6, 1.5] ms against 431
  expected at chance: not one cell.
- **`plans/tune/core-search.yaml`**: a spec for clean, linked cores -- objective `pyr_link - 10*max(0,
  0.95 - pyr_core_prec) + 0.2*pyr_cov - 0.02*merges_pyr_pyr`, three chunks, starting from the exp-search
  winner, without the knobs that search found inert, with the strip / shed / quarantine dials added.
- **Fix:** `fiber_bench.match_gt(units=None)` (keep every truth label) returned no match: `np.asarray(None)`
  is not None.  Caught by the new test before anything used it.
- test_gt_review: 21 checks.

## [0.60.3] — fiber-cpos survives a near-flat edge cluster
- **fiber_localize._fit_edge** (the one-flank refit fiber-cpos runs when a cluster's depth fit leaves the
  array end) started its source amplitude at `a.max()*30` without clipping it into its bounds
  `[1, 1e9]`, as `_fit` does.  A near-flat cluster -- peak amplitude below 1/30 ADU -- at a terminal
  channel raised "ValueError: Initial guess is outside of provided bounds" and killed the stage: on
  g6 0316 the exp plan's cpos-clean step, and with it the fiber-tune trial
  (`FK_SESSION_CFIBER_Q=0.85, FK_SESSION_DIP_ALPHA=0.01`).  The start point is now clipped into the
  bounds (unchanged for any profile that was feasible before), and a single-depth-row geometry gets a
  nonzero axial step instead of empty bounds.
- test_localize_edge: both checks fail on 0.60.2 with the error above.

## [0.60.2] — fiber-tune checks the truth before scoring
- **fiber-tune refuses a truth that is not the session's spike set.**  A spec whose `truth: res:` named
  group 1's `.res` beside group 6's `.clu` ran 60 trials (8 hours) that all scored nothing
  (`units_scored 0`, objective -inf), and coordinate search had nothing to compare.  Now the truth `.clu`
  and `.res` must hold the same number of spikes, and at least 20% of the truth's unit spikes must match
  the session's `.res` within `tol`; otherwise the run stops with the counts before any trial is scored.
- **Trials record their truth** (`truth` in trials.jsonl / trials.tsv).  A resumed search re-scores trials
  recorded under another truth (the sorts are re-used through fiber-bench's tags), and summary.md lists
  only the trials scored against the current one.
- test_tune: 4 new checks (3 fail on 0.60.1).

## [0.60.1] — fiber-kit.yaml runs the P(same) merges
- **The default template's plan** (fiber-kit.yaml, the fiber-link route) gains the exp plan's two
  fiber-psame-merge steps: `psame-chunk` (`per-chunk: 1`, fiber_session -> fk_psc) before intrachunk, and
  `psame` (fk_linked -> fk_psame) after the linker; realign-final re-extracts fk_psame.  `cpos-session`
  now runs on fk_psc, the tag intrachunk reads its positions from.  psame-chunk sits before intrachunk,
  not just before the linker as in exp, because fiber-link reads intrachunk's units.npz.  The template
  gains section [F] (`FK_PSAME_*`, the built-in defaults); the old plan stays commented out.
  This plan has NOT been scored on a ground truth (the exp plan's 0.60.0 numbers are anchor-link's).

## [0.60.0] — fiber-psame-merge per chunk and chunk-aware; the exp plan merges within chunks before linking; fiber-tune searches plans and per-step parameters
- **fiber-psame-merge `--per-chunk 1`** (`FK_PSAME_PER_CHUNK`): only clusters of the same chunk merge (the
  partition fiber-anchor-link uses).  For a plan step before the linker: the cell model has no drift term, and
  inside a chunk there is no drift.
- **fiber-psame-merge `--chunk-aware 1`** (`FK_PSAME_CHUNK_AWARE`, `FK_PSAME_MIN_CHUNK_N`, `FK_PSAME_MAX_GAP`):
  each pair is scored inside the chunks both clusters occupy, else across their two nearest chunks (at most
  MAX_GAP apart).  The merge log gains a `how` column (lifetime / shared / adjacent).
- **The exp plan** (fiber-kit-exp.yaml, "SINCE 0.60.0") gains step `psame-chunk` (fiber-psame-merge
  `per-chunk: 1`, fs_clean -> fs_psc) before `link`, which now reads fs_psc.  Measured on g6 0316 (four 18-min
  chunks, real anchor-link -> xcorr-merge -> psame-merge, GT cells, no oracle; three seeds and the stage CLI):
  ARI 0.617-0.635 -> 0.677-0.685, completeness 0.785-0.793 -> 0.828-0.854, purity 0.779-0.793 -> 0.762-0.770;
  ARI after the linker 0.203 -> 0.631-0.660.  `--chunk-aware 1` on the final step instead is the purest
  (0.800-0.808) at completeness 0.744-0.774; the two together do not add up (0.664 / 0.780 / 0.786).
  Not covered: overlapping chunks, realign / consensus / cpos, deconv.
- **fiber-tune** searches **plans** and **per-step parameters**: spec key `plans:` (named alternatives:
  `base`, a step list, `{file: ...}`, or `{drop: [steps]}` with the dropped step's readers re-wired), the
  pseudo-knob `@plan`, and `<step>:<param>` knobs that set one step's `params:` (fiber-psame-merge runs twice
  in the exp plan).  Each trial carries its plan to fiber-bench as a `pipeline` override, so unchanged steps
  are still re-used; best.yaml carries the winning plan.  `plans/tune/exp-search.yaml` is reorganised in plan
  order: plan, session, consensus, psame-chunk, link, merge, psame.
- fiber-bench prints a plan override as its step names.
- docs: methods.md "P(same cell)" -> Chunks; pipeline.md plan table; bench.md (knob kinds, plans);
  config.md, stages.md and the plan template regenerated.
- Tests: test_psame (per-chunk never merges across chunks and still merges within one; chunk-aware joins a
  cell's time halves across adjacent chunks, keeps a look-alike apart, and refuses with max-gap 0);
  test_tune (plan derivation and re-wiring, step parameters, knob checks, the plan's hash round trip, a
  search over @plan); test_plan_steps and test_bench for the new step.

## [0.59.0] — fiber-tune: a knob search against a ground truth on a subset of chunks
- **New tool `fiber-tune`** (`fiber_tune`): `search` runs a coordinate search over the knob groups of a spec
  file, each trial a `fiber-bench run` of the plan up to `score_at` on the spec's chunks, scored on a
  ground truth (fiber-bench's truth metrics plus ARI, purity, completeness) with a user objective;
  steps whose parameters did not change are re-used through fiber-bench's hashed tags, and a search
  resumes from `trials.jsonl`.  `confirm` re-runs the best configurations with the full plan, in-place
  realign steps included, each in a private workspace with its own `.res`/`.spk`/`.fet` copies.  Writes
  `fk-bench/tune/<name>/` with `best.yaml` (the base config with the winning overrides).
  `search_realign: true` runs every search trial that way.  Measured on g6 0316 (one 6-min chunk, the
  same parameters): search ARI 0.675 / accuracy 0.376, workspace run ARI 0.503 / accuracy 0.417 -- the
  realign changes what the later stages see, so the search ranks and `confirm` decides.
- **Example spec** `plans/tune/exp-search.yaml`: the exp plan's session, consensus, link, xcorr-merge and
  psame knobs (about 44 trials per pass, 18 of which re-sort the session).
- **Fix (fiber-bench): an annotating step keeps its input's tag.**  The exp plan's `cpos-clean` step
  (`fiber-cpos`, `in: fs_clean, out: fs_clean`) writes a .cpos table, never a .clu, but bench gave its
  `out:` a new hashed tag, so the next step (anchor-link, `in: fs_clean`) looked for a labelling nobody
  wrote and `fiber-bench run` of the exp plan stopped there.  A fiber-cpos step with `out:` equal to its
  `in:` now keeps the input's tag (its table is named for the labelling it describes) and its done-check
  is the .cpos table.  test_bench checks both.
- **plans/TEMPLATE-all-stages.yaml regenerated:** 0.58.2 changed fiber-psame-merge's `--spk-method` help
  without regenerating the template, so `gen_stages_doc --check --template` (in run_all) failed from 0.58.2.
- docs/bench.md section "fiber-tune", docs/stages.md entry, README tool row.  test/test_tune.py: the
  objective parser, the search plan, the override/hash round trip, the coordinate search on a known
  objective, the workspace layout and the scoring.

## [0.58.3] — fiber-kit-exp.yaml: narrower coarse bands, and the session gates strip instead of forcing
- **Embedded plan's knobs:** `FK_SESSION_COARSE_DR` 0.15 -> 0.09, `FK_SESSION_INCL_ASSIGN` 1 -> 0,
  `FK_SESSION_ASSIGN_UNSORTED` 1 -> 0.  The plan steps are unchanged.
- **Measured** on sirotaA-jg-000005-20120316 group 6, four 18-min chunks (0-72 min), real anchor-link ->
  xcorr-merge -> psame-merge, GT cells only: ARI 0.587 -> 0.635, purity 0.778 -> 0.793, completeness 0.785
  both, 2.3% of spikes left unsorted.  Not covered: overlapping chunks, realign / consensus / cpos, deconv,
  and the band and the strip separately.  Header note "SINCE 0.58.3" in the yaml; docs/config.md regenerated.

## [0.58.2] — fiber-psame-merge finds the .spk of a _D feature-space method
- **Fix:** with a `_D` feature-space `--clu-method` (e.g. `stderiv_C5_D34`, as the pipeline passes) and no
  `--spk-method`, fiber-psame-merge opened `.spk.stderiv_C5_D34.<elec>`, which never exists (there is no `_D`
  waveform set), and stopped with FileNotFoundError.  The default is now the method's waveform token
  (`nio.waveform_token`: stderiv_C5_D34 -> stderiv_C5), as fiber-realign resolves it.  An explicit
  `--spk-method` / plan `spk-variant` is used as given.  Found running xcorr-merge -> psame-merge on g6 0316.

## [0.58.1] — fiber-kit-exp.yaml carries the lab's production plan, with the P(same) merge
- **Embedded plan** replaced by the one that worked on the lab's sessions: flag -> session -> realign ->
  consensus (quarantine-only) -> realign -> cpos -> anchor-link (units) -> realign -> xcorr-merge ->
  **psame** (fiber-psame-merge, new in 0.58.0) -> realign-final -> rdr, both last steps on exp_psame.
- **Knob values** from that config: FK_SESSION_CHUNK_MIN 18, FK_SESSION_OVERLAP_MIN 8, FK_SESSION_INCLUSION_K
  3.0, FK_REALIGN_VARIANT stderiv_C6_D34.  The knob set is unchanged (every FK_* of 0.58.0, FK_PSAME_* included).
- **Kept as comments:** the plan without the psame step, and the 0.52.0 plan.
- **Note in section [F]:** psame's weights were calibrated on stderiv_C5; with stderiv_C6_D34 check the
  .psame tsv on a known session, or pin `spk-variant: stderiv_C5` on the step.
- docs/pipeline.md plan table and docs/config.md follow.  No GT score for the plan as a whole.

## [0.58.0] — fiber-psame-merge: merge by a calibrated P(same cell) from history-conditioned cell models
- **New stage `fiber-psame-merge`** (`fiber_psame`; opt-in, ALT stage, plan step commented in
  fiber-kit-exp.yaml section [F]; knobs `FK_PSAME_*`).  Each cluster gets a cell model whose expected amplitude
  comes from its own preceding spikes (EWMA adaptation state, best of 30 ms - 1.2 s, plus a burst term) and
  whose shape follows the amplitude.  A pair's evidence is the BIC of one model of the union against two;
  P(same) = logistic(bias + w_rc roll-cos + w_dbic dbic), calibrated on g6 0316 (2,026 GT pairs).
- **Merging** is sequential with refit, while P >= theta (0.5), with a member-level cannot-link (0.3) that
  stops one growing cluster from absorbing its neighbours.  Writes the `.clu`, carries `.clc/.clp`, and logs
  every merge to `<SESS>.psame.<group>.<tag>.tsv`.
- **g6 fix_linked -> GT** (ARI / purity / completeness): fiber-xcorr-merge 0.577 / 0.758 / 0.598; this stage
  0.603-0.668 / 0.723 / 0.701-0.769 over two runs; merging by the GT itself 0.692 / 0.759 / 0.764.  A
  completeness step at a purity cost.  Held-out-sort pair AUC 0.961 (roll-cos alone 0.910).
- **`--geo 1`** adds `fiber_srcgeom`'s shared-geometry z with its own calibrated weights (off by default).
- **Limits, in docs/methods.md "P(same cell)"**: a cell cut by a hard amplitude threshold scores dbic < 0;
  the per-spike version of the model does not clean clusters; the weights are session-calibrated.
- **test_psame**: EWMA state against fiber_adapt.ewma; a bursting cell cut in time is merged; a tonic cell
  with the same average waveform (roll-cos as close) is not.

## [0.57.7] — fiber_srcgeom: a 6-source chain fits a unit's footprint to the noise; the shared-geometry test
- **New module `fiber_srcgeom`.**  A unit's raw template is the field of six point current sources on one axis
  through the soma (currents summing to zero); given the geometry the currents are linear, so a fit is a
  projection onto a 5-dim channel subspace and only the geometry (soma position, axis, scale) is searched.
  On g6 0316 it reaches the template noise (median misfit / noise 1.05), where the reconstructed CA1 cable
  cells of fiber-morpho stayed ~60x above it.
- **`shared_geometry_z`**: the cost, in noise units, of fitting two templates with ONE geometry and their own
  currents.  g6 GT pairs: AUC 0.78-0.94 alone, roll-cosine 0.878 -> 0.914 with it (held-out sort).
- **Resolution is coarse** (synthetic: 80 um in depth z 5.1, 40 um 2.2, 20 / 40 um laterally 0.1 / 0.3).
  As a veto in fiber-xcorr-merge it changed nothing, and its soma position in place of fiber-cpos's lost
  merges in fiber-intrachunk / fiber-link -- so it is not wired into a stage.  docs/methods.md
  "Shared source geometry".
- **test_srcgeom**: refit to the noise, one place with different currents (small z), and the negative control
  (80 um apart: large z).

## [0.57.6] — tools/fil_noise_rhythms.py: how does a group's noise vary with theta and gamma?
- **What it reads.**  One group's channels of the `.fil` in 1-s windows, keeping the spike-free samples
  (`fil_noise_states.py`'s mask), in the space the pipeline sorts in (`--variant`).  The LFP is `<base>.lfp`,
  else `<base>.eeg`, or `--lfp`.  Its reference is the median of `--lfp-channels` (default the group's;
  `--theta-channels` for a separate theta reference), band-passed (zero-phase Butterworth, Hilbert) for delta,
  theta and each `--gamma-bands` band (default slow 30-50, mid 55-90, fast 90-150 Hz).
- **Four answers.**
  1. *State*: the noise level in theta windows against non-theta windows, overall and per channel.  Its
     correlation with the theta index and with each band's power, raw and given the detection rate.
  2. *Phase*: noise power per phase bin, each window normalised by its own mean, so slow level changes drop out.
     - Theta is measured in the theta windows; each gamma band in its bursts and in its exclusive bursts
       (every other band below its own threshold).
     - Output: modulation m, preferred phase and max/min, per channel and for the channel mean.
     - The null rotates each window's phase histogram by a random offset.
     - The detection rate per phase bin is the comparison.
  3. *Amplitude*: noise power and detection rate per decile of each gamma band's instantaneous envelope, with a
     block-bootstrap interval.  A joint regression on all the bands' envelopes gives each band's unique slope
     (% noise power per SD, the other bands held fixed) beside its marginal one.
  4. *Coupling*: each gamma band's envelope by theta phase, and the correlation between the bands' envelopes.
- **Windows left out:** flat windows (a channel's robust SD below 1 file unit), dropouts (a channel below
  `--drop-frac` of its median) and `--exclude-minutes` spans.  The report counts them.
- **Outputs.**  `<out>.npz`, `<out>.png`, and a printed report.
- **Why the exclusive bursts and the joint regression.**  A strong gamma burst leaks into the neighbouring band.
  In the test, mid-gamma bursts gave the fast band, which held no fast gamma, a noise phase modulation of 0.098
  at the mid phase and a +0.8%/SD marginal slope.  In its exclusive bursts the modulation drops to 0.010; with
  the mid envelope held fixed, the fast slope is -0.8%/SD.
- test_fil_noise_rhythms, on synthetic 20-kHz sessions with a 1250-Hz LFP:
  - **Modulated file** (theta on channels 0-3 at 180 deg, mid-gamma phase on channels 4-7 at +90 deg, a theta
    state x1.2, theta-coupled slow gamma, a 4-s flat stretch and a 3-s channel dropout):
    - the flat and dropout windows are found;
    - the theta windows, the state ratio, the per-channel theta and mid-gamma modulations and phases;
    - the detections' theta modulation;
    - no slow-gamma relation;
    - the leak into the fast band, and its removal in the exclusive bursts;
    - the joint-regression slopes (+15%/SD on channels 4-7 for mid);
    - the slow-gamma coupling at -90 deg;
    - the stderiv_C5 space also runs.
  - **Stationary control:** no phase modulation, amplitude relation or state change.
  - Three planted bugs are each caught: negated phase, non-exclusive bursts, and the flat test without the
    robust-SD floor.

## [0.57.5] — tools/fil_noise_states.py: is a group's noise stationary, or does it come in states?
- **What it measures.**  It streams one group's channels of the `.fil` in 1-s windows and keeps the
  spike-free samples (at least `--guard-ms` outside every detected spike's window).  It measures them in the
  space the pipeline sorts in: `--variant standard` is the `.fil` itself, a stderiv token goes through
  `fiber_stderiv.space_transform` with the session's sdiffPairs.  Per window:
  - the level: robust SD, dithered, because a MAD of int16 data is quantised;
  - the tails (plain over robust SD);
  - the common share (0 = independent channels, 1 = all common);
  - the spectral tilt and band powers;
  - the detection rate;
  - with an LFP (`<base>.lfp`), the theta index and ripple-band power.
- **Four answers.**
  1. *Structure*: the share of the window-to-window variation of the level that is real.  The measurement
     noise comes from even against odd 25-ms sub-blocks.  The level's autocorrelation is reported at 1 s to
     10 min.
  2. *States*: a Gaussian mixture on (level, tails, common share, tilt), K by BIC.  Per state: the features,
     the detection rate, theta, and dwell times.  A small state of single windows is marked *transient*:
     windows straddling a switch, or holding a ripple.
  3. *Fast structure*: noise power by theta phase in theta windows, and the noise level around ripple events.
  4. *Pipeline impact*: the block-Toeplitz noise covariance over the session mask, for the span, each state
     and each 12-min chunk.  It reports the eigenvalues of `Cs^-1/2 C Cs^-1/2`, i.e. how far from white each
     state's noise is under one span whitener (a common factor = a level change, a spread = a change of shape).
- **Outputs.**  `<out>.npz`, `<out>.png` and `<out>_fast.png`, plus a printed report.
- test_fil_noise_states, on synthetic 20-kHz sessions:
  - **Two-state file:** the real share is high, and the two dominant states match the built ones (99% of
    windows). Further checks:
    - the level ratio, common shares, tail order and dwell times;
    - theta-phase modulation x2.9;
    - a noise rise at the ripples x1.8;
    - the whitener mismatch;
    - the stderiv_C5 space separates the states too.
  - **Stationary control:** real share 0%, one state, no phase modulation.  Without the dither, the
    quantised level splits the stationary control into three spurious states.

## [0.57.4] — fiber-bench hashes the consolidation knobs into fiber-session's tag
- **The bug.**  fiber-bench listed `FK_CONS_` among fiber-consensus's prefixes, so fiber-session's hash left out
  every `FK_CONS_*` knob.  Those are fiber-consolidate's knobs, read by the consolidation pass fiber-session runs
  inside the stage (`FK_SESSION_CONS_MODE`), not by fiber-consensus (`FK_CONSENSUS_*`).  Two variants that
  differed only in a consolidation knob (`FK_CONS_AGG_IQR`, `FK_CONS_PURE_IQR`, `FK_CONS_ITERS`, ...) got the
  same session tag: the first variant ran, every other was skipped as cached and scored with the first's
  output.  Sweeps over `FK_CONS_*` made before 0.57.4 compared one sort with itself; run them again.
- **The fix.**  fiber-consensus owns `FK_CONSENSUS_*` only.  `SHARED_PREFIXES` lists the prefixes several stages
  read: `FK_CONS_*` goes into the hash of fiber-session, fiber-intrachunk and fiber-anchor-link, the stages that
  host the pass.  docs/bench.md's hash table says so.
- test_bench: a consolidation knob changes the session hash and not the flag's; a consensus knob leaves the
  session hash alone; intrachunk's and anchor-link's hashes carry `FK_CONS_*` and fiber-consensus's does not
  (the first and third fail on 0.57.3).

## [0.57.3] — the .clu header is the number of distinct ids
- **The header.**  Every `.clu` and `.clc` fiber-kit writes carries its number of distinct ids as its int32
  header, as Klusters writes it on Save; a `.clp` carries its number of entries (`neuro_io.clu_header`).
  - Until now the header was the highest id + 1.  In a chunk run (`FK_CHUNKS`, 0.56.0) the reserved
    clusters 16777116+chunk made that about 16.8 million, and Klusters, which accepts 0-65536, refused
    every output: "Invalid or missing .clu header (nClusters=16777136)".
  - `write_clu_file` takes the header from the ids.  The `n_clusters=` that callers passed are gone
    (fiber-contam, fiber-defrag, fiber-decollide, fiber-intrachunk, fiber-link, fiber-peel,
    fiber-anchor-link, fiber-backbone-link, fiber-flag-artifacts, fiber-realign's duplicate parking, the
    spike-keep subset); the argument is accepted and ignored.
- **`fiber-clu-header FILE...`** repairs a file written before: it rewrites the 4 header bytes in place and
  leaves the ids alone.  `--check` reports without writing and exits 1 when a header is off.  It skips
  names that are not `<base>.<clu|clc|clp>[.<variant>].<elec>[.<tag>]`, text `.clu` files and odd sizes,
  so a glob over a stage's files (`.res`, `.spk`, `.units.npz`, ...) is safe.
- **No reader takes the header for a count.**
  - fiber-refiber cut its per-spike atom labels to the file's header as if it were the spike count.  Its
    atom templates came from the first N spikes only (N = the number of atoms), and atoms absent from those
    were dropped.  It now requires the atom labels and the raw `.spk` to have the same number of rows.
  - fiber-realign's log counted units from the header; it counts them from the labels.
- test_clu_header:
  - a chunk run's `.clu` gets the distinct-id count, within 0-65536;
  - a `.clc` its distinct atom ids, a `.clp` its entry count;
  - `n_clusters=` is ignored, through `write_clu` too;
  - `fiber-clu-header --check` finds the files a 0.57.2 chunk run left and writes nothing; the repair
    changes only the header bytes and skips the `.res`, `.units.npz` and text `.clu` the same glob caught;
  - negative control: on 0.57.2 the headers are 16777119, 41 and 16777117.
- test_chunks: every `.clu`/`.clc`/`.clp` the chunk run writes has the right header.

## [0.57.2] — fiber-pipeline checks a plan's inputs before step 1
- **Preflight.**  Before running anything, fiber-pipeline checks that every selected step's inputs (`in:`,
  `units:`, `cpos:`, `drift:`) are written by an earlier selected step or are already on disk under any
  method token.  For each that is not, it names the step and the tag, and says where its writer is: a later
  step ("move it before step N"), an unselected step, or none.  It then exits 2 without running step 1;
  `--dry-run` warns and carries on.
  - Until now a plan whose backbone step read `exp_xcm` with the merge step that writes it commented out
    ran its first seven steps and failed at step 8.
  - A `drift:` no step writes was not even an error: fiber-backbone-link printed "--drift ignored" and linked
    without the transform.
- The chunk shield reports a step input that does not exist as a message, not a traceback.
- test_plan_steps:
  - a writer that runs after its reader stops the plan before step 1, naming both steps;
  - a dry run only warns;
  - the same steps in order pass;
  - an unselected writer is named;
  - a tag already on disk passes;
  - an unwritten `drift:` stops the plan.
- test_chunks: the shield's missing-input message.

## [0.57.1] — fiber-rdr counts FK_CHUNKS in the session's chunks
- In a chunk run, fiber-rdr mapped `FK_CHUNKS` onto chunks of its own template length (`--chunk-min`,
  `FK_RDR_CHUNK_MIN`, 12 min) instead of fiber-session's (`FK_SESSION_CHUNK_MIN`).  With 18-minute session
  chunks, `FK_CHUNKS: 12-14` selected minutes 144-180 instead of 216-270: every sorted spike was outside the
  selection, rdr scored nothing and exited 1.  rdr now takes `--chunk-length` (default
  `FK_SESSION_CHUNK_MIN`), as fiber-flag-artifacts, fiber-xcorr-merge, fiber-realign and the chunk shield do.
  `--chunk-min` stays the template bin.
- test_chunks: rdr with `FK_CHUNKS=1` on 30-second session chunks scores chunk 1's clusters.  0.57.0, on the
  same session, scored none ("chunks 1 of 12 min").

## [0.57.0] — chunk runs through every relabelling stage (the chunk shield)
- fiber-pipeline runs fiber-cpos, fiber-intrachunk, fiber-link, fiber-anchor-link, fiber-backbone-link,
  fiber-consensus, fiber-lowmix, fiber-peel and fiber-refit on a chunk run (`FK_CHUNKS`) **shielded**.
  These stages only relabel spikes.
  - **Before the stage:** `fiber_chunks.shield` sets the spikes of the chunks not computed to 0 in the
    step's input .clu (and their atoms to 0 in the .clc).  Every stage leaves cluster 0 out of its units.
    The input is saved as `<file>.chunkshield`.
  - **After the stage:** `unshield` puts those spikes back into their reserved clusters 16777116+chunk in
    the .clu the stage wrote, with one atom per chunk under the reserved id in the .clc/.clp.  It then
    restores the input byte for byte.
  - **On failure:** an EXIT trap restores the input when the stage fails.  The next shield undoes a run
    killed between the two.
  - **Tags:** shield and unshield use the plan's `in:`/`out:` tags.  A `--clu-stage`/`--out-stage`
    override in `params:` is not followed.
- Still refused while FK_CHUNKS is set: fiber-refine (its dedup changes the spike set) and fiber-stochastic
  (its own chunking).
- test_chunks:
  - backbone-link and anchor-link run shielded through the runner;
  - the other chunk comes back in its reserved cluster, and the hierarchy reproduces the .clu;
  - the input is restored byte for byte;
  - a failing stage still restores its input;
  - a killed run's shield is undone.

## [0.56.5] — review fixes to 0.56.1–0.56.4
An independent review of the four previous releases found these; each now has a test.
- **fiber-realign keeps a commit journal.**  Writing the .res last (0.56.1) was not enough for `--shift-spk`.
  A roll is relative, so after an interruption a re-run measured ~0 shift on the already-rolled windows
  and left them shifted beside an uncorrected .res.  A commit now records the corrected times, the
  offsets, the mode and the variants written in `<base>.realign.<group>[.<tag>].pending.npz`.  A run that
  finds the journal first finishes that commit (the remaining variants, the lag-alias .fet, the .res),
  removes the journal, and then runs the alignment it was asked for.  Interrupted then re-run now ends
  byte-identical to an uninterrupted commit (test_atomic_io).
- **fiber-pipeline's 0/1 switches accept on/off.**  With YAML 1.2 loading (0.56.2), `FK_SESSION_LINK: off`
  reached bash as "off", which the `= "0"` tests ignored, so linking ran.  Every switch test now goes
  through `_on`/`_off`: they take 1/on/yes/true and 0/off/no/false in any case, and stop the run on
  anything else.
- **The dedup commit rolls back.**  Every live file is moved aside before the subsets go in.  A rename
  that fails puts all of them back; on Windows that happens when another handle maps the file.
  `stale='error'` now refuses before anything is committed.
- The last spike of an exact-multiple span is in the last chunk in fiber-refine, fiber-stochastic,
  fiber-stats and fiber-position too (`fiber_chunks.chunk_hi`); 0.56.3 fixed only fiber-session.
- A GT .clu of another length is reported ("GT not scored") by fiber-xcorr-merge and fiber-backbone-link,
  not a traceback after their output is written.  fiber-session's `--exclude-clu` length error is a
  message again.
- The deconvolution manifest names the unsorted fiber for spikes the sweep moved there; it used to say 0.
- An integer knob of nan/inf is a KnobError.

## [0.56.4] — docs/config.md generated from the yaml; README and config drift
- **docs/config.md is generated** by the new `tools/gen_config_doc.py` from fiber-kit-exp.yaml's own
  comments: one section per `[X]` banner and `-----` sub-header, each knob with its value in both shipped
  yamls.  run_all runs `--check`.  The handwritten page had drifted:
  - it documented 23 knob names no code reads (e.g. FK_ALIGN_LAG, FK_DIP_ALPHA);
  - it missed about 170 knobs the exp yaml sets (FK_RDR_*, FK_FLAG_*, FK_LOWMIX_*, FK_CONSENSUS_*, ...);
  - it quoted 12 defaults the yaml no longer had (chunk length 18 vs 12, FK_XCM_COS_THR 0.99 vs 0.96, ...).
- The check also fails when the code reads an FK_* knob that neither shipped yaml lists.  Added to
  fiber-kit-exp.yaml with their code defaults: FK_XCM_MERGE_ALIGN(_MIN_SCORE), FK_SESSION_RKK_SPLITS /
  _PENALTY_MIX, FK_CONS_SHED_K / _MIN, FK_ALINK_FEAT_LAG, FK_BBLINK_DRIFT_RECENTER, FK_LOWMIX_SIGNAL.
  FK_SESSION_FINE_METHOD, deleted from the exp yaml by accident in 1f896f1 (its description stayed
  behind as stray comment lines), is restored.
- fiber-session reads FK_SESSION_RKK_SPLITS / _PENALTY_MIX through `config.knob_value`; they were read from
  the environment only, so a yaml value was ignored on a direct call.
- The exp yaml's section banners name the current plan steps; they still described the pre-0.52 plan.
  FK_REALIGN_AFTER / _AFTER_MODE / _FINAL_MODE are marked "canonical mode only" in both yamls: a plan's
  own fiber-realign steps decide, and these three knobs are not read.
- README:
  - the version badge said 0.21.0;
  - the CLI reference now lists all 52 commands (it covered 10, and its fiber-session table gave `--fine-algo`
    a default of gmm where the parser says rkk) and points to the generated stages.md / config.md;
  - the project layout describes the directories instead of 18 of 83 modules.
  test_readme (new) keeps the badge and the command table current.
- Removed `klustakwik_classic.py` (328 lines, referenced nowhere).  `python -m fiber_kit.klustakwik`
  writes `<base>.clu.<method>.<elec>.kk` by default, no longer the canonical `<base>.clu.<elec>`.
  `tools/hier_diag.py --help` prints its usage and exits 0.

## [0.56.3] — no merge without its vetoes, no chaining, and cluster 0 only for derivative outliers
- **fiber-session merges through every veto, with complete linkage.**  The re-split loop's merge (Block C,
  on in the exp plan via `FK_SESSION_RESPLIT_PASSES: 3`) used no veto at all.  Because resplit replaces
  Blocks A/B, the yaml's `FK_SESSION_CFIBER_GATE` and `FK_SESSION_OFF_THR` did nothing.  Blocks A, B and C
  now go through one `_merge_vetoes`: the cfiber shape veto, inter-channel timing, and the refractory gate
  when `--refrac-ms` is set.  The resulting edges are joined by `_complete_link`: best pair first, and a
  group joins only if every cross pair is itself an edge.  Union-find had chained A~B and B~C into A+B+C
  (the reviewer's case had cos(A,C) = 0.978).
- **fiber-xcorr-merge's knn split folds only through the merge tests.**  When a candidate failed the RVU
  test, the knn split folded buckets into neighbours chosen by cosine rank, with no threshold, no
  refractory, timing or band check, and the twin guard off.  A bucket now folds into a reference only if
  the pair passes the roll cosine (complexity-scaled), the refractory gate, the timing gate and the band
  overlap.  Otherwise it becomes a new cluster.  The log counts the refusals.
- **fiber-session never writes an unplaced spike to cluster 0.**  Spikes still in 0 after deconvolution
  that the artifact mask did not put there go to one "unsorted" fiber per chunk, a real cluster with one
  atom.  This covers a chunk too small to sort, a failed fiber geometry and rejections
  `--assign-unsorted` did not cover.  The last spike of a session whose span is an exact multiple of the
  chunk length is now in the last chunk; `res < hi` used to leave it out of every chunk.
- **fiber-realign's duplicate parking** (`FK_REALIGN_PARK_DUPS`, and fiber-session's own commit) puts the
  second detection of a spike in a new duplicates cluster (the next free id) instead of cluster 0.
- **fiber-relink** numbers units from 2.  Until now they started at 1, so the first real unit landed in
  Klusters' noise cluster.  Clusters 0/1 and reserved chunk ids pass through unchanged.
- **fiber-refine** stops when `--in-clu` names a missing file.  It used to refine whatever canonical .clu
  was in the folder, or ran a fresh sort.  It also stops with a message when two stderiv clus make the
  fallback ambiguous.
- fiber-intrachunk / fiber-link log it when the source .clc/.clp cannot be read, before writing a flat sort.
  fiber-session reports a timing veto that has no .spk.standard to work on, and refuses a .spk.standard
  of another length.
- test_merge_rules (new).  On 0.56.2 the knn fold moved all 120 spikes into the dissimilar cluster, and
  the session put 33 spikes in cluster 0.  test_session_no_shed's control, test_session_deconv ("from
  noise" is now "from the unsorted fiber"), test_realign_duplicates and test_relink_ids are updated.

## [0.56.2] — FK_* knobs: one precedence, YAML 1.2 booleans, validated values
- **`off` is off.** PyYAML read a bare `off` as False, which reached the stages as "0" (fiber-pipeline) or
  "False" (a direct call).  So `FK_SESSION_REALIGN: off` ran the realign and rewrote the shared .res/.spk,
  and `FK_INTRA_RECUT: off` ran the `all` recut.  `config.yaml_load` now reads every FK config, plan and
  fiber-bench `--set` value with YAML 1.2 booleans: only true/false are booleans.  fiber-pipeline's config
  reader does the same.
- **Values are validated.**  `config.knob_value` / `coerce_knob` convert a knob and stop the stage with its
  name and source when the value is not one of the choices, not a number, or not an integer.  An off/on
  spelling (False, "0", "no", ...) maps onto the choice that means it.  `check_choice_defaults` does the same
  for every argparse default that comes from a knob, before parsing; argparse itself never checks defaults.
  The cases this stops: `FK_INTRA_GATE=Band` silently ran the cosine gate; `FK_CONS_MAX_DIST=0,4` silently
  became 0.5; an int knob of 6.7 in the yaml was truncated to 6.
- **One precedence: env over yaml.**  Every stage's `_knob_default` delegates to `config.knob_value`
  (CLI > FK_* env > $FK_CONFIG > default).  fiber-xcorr-merge, fiber-rdr, fiber-backbone-link, fiber-session's
  chunk/overlap/off-thr/whiten-source knobs and `fiber_chunks.config_value` read the yaml first, so an
  exported one-off value was ignored whenever the yaml set the key.
- A `$FK_CONFIG` that exists but does not parse is an error.  It used to read as an empty config, which ran
  every stage on built-in defaults.
- test_knobs (new).  On 0.56.1 the dry run showed `--realign 0` for `FK_SESSION_REALIGN: off`, and
  fiber-xcorr-merge read 0.96 from the yaml over 0.995 from the environment.

## [0.56.1] — shared files are never left cut short, half-written or half-deduped
- `neuro_io.atomic_write` (new): every writer goes to a temporary file beside the target and moves it in with
  `os.replace`.  This covers .res/.clu/.fet/.spk/.nst/.wti/.wtl/.eap/.tcl, plus .pca, .cpos, .position,
  .fibers, the fiber-stats and fiber-stochastic tables, and the xcorr-merge drift table.  A crash, a full disk
  or Ctrl-C leaves the previous file, never a truncated one.
- fiber-realign refuses a clu whose length differs from the .res or .spk.  Until now it aligned "the first
  n" spikes and committed them, which cut the shared .res and every .spk/.fet down to the clu's length.
- fiber-realign writes the .res last, after the windows, .fet and clu.  A run that died after the .res left
  shifted times beside the old windows, and the re-run shifted those spikes a second time.
- `apply_spike_keep` (dedup) is two-phase: every subset is written beside its file, and only when all of them
  succeeded are they moved in.  A failed rewrite now raises and leaves the group as it was.  It used to
  truncate the file, quarantine the now empty file as .stalebkp and report success.
- `read_clu_file(n_spikes=)` raises on a .clu of another length instead of truncating it or returning it
  short, so a stale or foreign sort can no longer be row-aligned with the wrong spikes.
- An empty .spk reads as 0 spikes; `np.memmap` used to raise on it.
- test_atomic_io (new).  The negative control on 0.56.0 fails 6 checks and reproduces the quarantined
  .spk.

## [0.56.0] — chunk runs (`FK_CHUNKS`) and reserved cluster ids
- `FK_CHUNKS` (global; `--chunks` on each supporting stage) runs only some chunks, e.g. `14-16`.
  Chunks are fiber-session's own `FK_SESSION_CHUNK_MIN` chunks, 0-based.  Every spike of a chunk that
  was not run goes to that chunk's reserved cluster, 16777116+chunk.  These are the top 100 ids a
  float32 holds exactly (2^24-100 .. 2^24-1), so a partial sort still opens in Klusters (docs/pipeline.md,
  Chunk runs).
- Supported by fiber-flag-artifacts, fiber-session, fiber-xcorr-merge, fiber-realign and fiber-rdr.
  These stages never merge, realign, score or flag a reserved cluster.  fiber-session's .clp gives each
  reserved chunk one atom.  fiber-pipeline refuses every other stage while FK_CHUNKS is set.
- `fiber_chunks` (new): the constants, chunk indexing and `next_free_id`.  fiber-session's unit carry,
  fiber-xcorr-merge and fiber-consolidate allocate below the reserved block.  `refiberize` keeps reserved
  ids fixed and no longer builds a lookup table as large as the largest id.
- fiber-bench: `run --chunks`.  `FK_CHUNKS` and `FK_SESSION_CHUNK_MIN` are in every step's hash.  `score`
  leaves reserved spikes out, drops truth units with no spikes in the chunks run, and records
  `units_scored` and `chunks_left_out`.
- test_shared_contract checks the reserved ids against neurosuite-3's types.h when NS3_ROOT is set
  (neurosuite-3 0889, which makes Klusters' id allocators skip the block).
- test_chunks (new): 22 checks.

## [0.55.0] — fiber-bench: parameter variants under a config hash
- `fiber-bench` (new; docs/bench.md).  `run` executes a config's plan (default: up to fiber-session) for
  every `--set` / `--grid` / `--sweep` variant; each clustering step writes under `<out>_<hash>`, the hash
  covering the FK_* knobs that stage reads, its plan params, the method token and the hash of its input
  step -- so equal parameters reuse the cached output, a downstream knob leaves upstream outputs alone,
  and an upstream knob re-tags everything after it.  Each run gets its own config and log under
  `fk-bench/runs/<hash>/`, runs with FK_* stripped from the environment, and is registered in
  `fk-bench/runs.jsonl` with its overrides, timing and fiber-kit version.  fiber-session's end-of-run
  realign (which rewrites the shared .res/.spk in place) is forced off unless `--keep-realign`.
- `score`: against a curation, per truth unit the best cluster's accuracy, recall, precision, frag80
  and the merges, typed pyramidal / interneuron (rate >= 5 Hz or `--interneurons`); truth-free per
  cluster: size, ISI violations, fiber-rdr score, gain IQR, RVU -- and their AUC for impure clusters.
  On the reference g7 data it reproduces the tuning scorer (accuracy 0.3651 vs 0.3655; 11 of 150,243
  spikes matched differently) and types units 6 and 22 as the interneurons.
- `report`: sortable table, pyramidal accuracy vs false pyramidal merges, per-parameter effects, a
  per-unit accuracy heatmap and the truth-free measures against purity, in one HTML file.
- plans/sweeps/session-first-pass.yaml: a one-knob-at-a-time first pass over fiber-session.

## [0.54.0] — fiber-rdr: lower-envelope line, in the exp plan
- fiber-rdr fits the clean line as the 0.25 QUANTILE REGRESSION of median D² on A² (`--fit quantile`,
  `--fit-quantile`), the lower envelope: contamination only raises D.  The 0.53.0 least-squares fit with a
  symmetric 3-MAD clip (`--fit clip`) followed the bulk, so on a sort where most clusters are dirty or small
  it was dragged up (g7 0312 merged sort of minutes 168–352, clusters down to 15 spikes: excess 0.074/SD
  against 0.041) and stopped flagging (24% of the mixed clusters).  The quantile line gives 0.036–0.046/SD
  on all three g7 sorts and at both size floors; AUC equal or better (0.87 / 0.79 / 0.85).
- `--thr` default 1.7 -> 2.0 (the lower line raises every score): 50–78% of mixed and 9–22% of pure
  clusters flagged on those sorts.
- Knobs `FK_RDR_*` (THR, MIN_AMP, MIN_N, CHUNK_MIN, FIT, FIT_QUANTILE, LINE, ALIGN_MIN_SCORE, CAP) from
  `$FK_CONFIG` / the environment; section [E] of fiber-kit-exp.yaml.
- fiber-pipeline knows the `fiber-rdr` stage (post-sort, not in `all`; writes the table, no `.clu`), and
  fiber-plan-edit lists it.  The embedded exp plan ends with `{stage: fiber-rdr, name: rdr, in: exp_xcm}`.

## [0.53.0] — residual-derivative contamination score; merge-align threshold 0.10
- `fiber-rdr` (new, QC only): per-cluster contamination from the RMS of the residual's temporal
  derivative (aligned spike − per-chunk median template, in units of the noise derivative RMS from the
  `.nst`, or from the windows' lead-in when there is none), scored against the clean line
  median D² = D0² + k·A² fitted on all clusters (3-MAD clip) or fixed with `--line`.  Score = largest
  observed/expected excess over channels ≥ `--min-amp` (4) SD; `--thr` (1.7) flags.  Writes
  `<base>.rdr.<clu-method>.<group>[.<stage>].tsv`.  From the g6 0316 residual-derivative report (dirty
  units 1.77–2.57, clean ≤ 1.65 with per-chunk templates); replicated on g7 0312 minutes 168–204 against
  the old curation: AUC 0.87 for clusters below 90% purity, the same level as the gain IQR (0.86) and
  RVU (0.85), and a line slope of 0.041/SD fitted without labels against g6's 0.039–0.042.
- fiber-xcorr-merge `--merge-align-min-score` (`FK_XCM_MERGE_ALIGN_MIN_SCORE`, default 0.10):
  `--merge-align klusters` used Klusters' realign threshold 0.70.  The template is pre-rolled to
  peakSampleIndex − 1 while raw troughs sit 2–3 samples later, so every spike below 0.70 stayed 2–3
  samples behind the rest of its cluster (g6 0316: 26–94% of the spikes of large or dirty units),
  widening the template.  At 0.10 at most 2.4% stay; `--merge-align-min-score 0.70` restores the old
  behaviour.  The default alignment (`peak`) is unchanged.

## [0.52.0] — no real spike in cluster 0; amplitude-aware merging
- fiber-session `--keep-small-groups` (`FK_SESSION_KEEP_SMALL`, `--keep-small-min`, default 5): a fine
  group below `--fine-min-group`, or whose inclusion core falls below it, becomes a fiber of its own
  instead of going to cluster 0 -- the skips that dropped every cell with < 40 spikes in a chunk.
- fiber-session `--assign-unsorted` (`FK_SESSION_ASSIGN_UNSORTED`): every spike still unsorted at the end
  of a chunk is placed in the chunk's whitened, realigned features: within its nearest fiber's own
  spread (median + inclusion-k x MAD) -> that fiber; outside every fiber's spread -> GMM groups of their
  own; the rest -> the nearest fiber.  The "unsorted spikes by cause" table reports what was kept and
  assigned.  With fiber-flag-artifacts -> `--exclude-clu`, cluster 0 holds exactly the flagged artifacts.
- fiber-xcorr-merge, amplitude-aware gates (all off by default): the roll cosine is scale-invariant, so
  two co-shaped cells of different amplitude merged.
  - `--rvu-thr` / `--rvu-rise` (`FK_XCM_RVU_*`): per-channel residual-variance uniformity within
    +/- `--rvu-win` (12) samples of the peak; any channel failing fails the candidate merge.
  - `--rvu-knn 1`: a failing candidate is first waveform-knn-split (fiber_consolidate.knn_pass, Klusters'
    split-by-knn-vs-references), each of its two clusters against the other and its nearest clusters,
    then retried once; spikes moved this way become new child atoms in the carried `.clc`/`.clp`.
  - `--giqr-thr` / `--giqr-rise`: own-gain IQR veto (Klusters strip gain on the raw waveforms).
  Measured on minutes 168-204 of the reference session's group 7 (29 clean units of the old curation,
  fiber-session fibers, cos 0.96, min-n 15): accuracy 0.293 -> 0.365, merges between truth units 4 -> 0
  with `--rvu-thr 0.5 --rvu-rise 0.05 --rvu-knn 1`; clusters per 80% of a unit 4.3 -> 7.9.
- fiber-xcorr-merge `--merge-align klusters` (`FK_XCM_MERGE_ALIGN`); plans/10-session-xcorr.yaml.
- fiber-kit-exp.yaml: the embedded plan is now fiber-flag-artifacts -> fiber-session -> fiber-xcorr-merge
  -> fiber-realign (the previous plan is kept, commented), with `FK_SESSION_EXCLUDE`, `KEEP_SMALL`,
  `ASSIGN_UNSORTED`, `DECONV_ASSIGN` on and the merge at cos 0.96, min-n 15 with the RVU test + knn split.
  Consensus quarantine and intrachunk, which move clean-unit spikes to cluster 0/1, are out of the plan.

## [0.51.1] — fiber-cpos on large clusters
- `fiber-cpos --max-spikes` (default 20000): a cluster's template and position come from a seeded random
  sample of at most this many spikes.  It loaded every spike as float64 before; one 380k-spike cluster
  (group 7 of the reference session) took several GB and the stage was killed on an 8 GB machine.
  Positions move by < 0.1 um; `n` still reports the full count; `--max-spikes 0` is the old behaviour
  (byte-identical output).
- Clusters are grouped once by a sort instead of a `clu == c` scan of every spike per cluster (three
  places), so the stage scales with the spike count, not spikes x clusters.

## [0.51.0] — realign commits on their target; fiber-session out of the reserve ids
- **fiber-realign cuts committed windows where it aligned them**: the spike time at `peakSampleIndex - 1`,
  neurosuite-3's window and the klusters target (`timestamp_index`).  Until now a `--reextract` commit
  cut at `peakSampleIndex`, one sample late: committed templates peaked one sample after the target, a
  second realign moved nearly every spike by +1 to undo it (96 % on sirotaA-jg-000005-20120316 group 6
  with the exp knobs), Klusters' re-cut and fiber-realign's disagreed, and the `.pca` basis was projected
  onto shifted windows.  `--peak` is the YAML's 1-based value.  **Sessions committed by 0.38.1-0.50 keep
  their old windows until repaired**: docs/pipeline.md, "Repairing a session committed before 0.51".
- **A realign aligns in the geometry the windows on disk were cut with**: with a `.fil` it locates them
  first (`window_index`, logged as `windows`).  On windows an older commit cut at `peakSampleIndex` it
  aligns there and re-cuts at `peakSampleIndex - 1`, so one pass repairs them; aligning at the group's
  index would have left every window one sample early and the next pass moving every spike back.
- fiber-session's end-of-run commit cuts at the group's YAML `peakSampleIndex`, or, without one, at the
  index the windows on disk were cut with (located in the `.fil`; without a `.fil`, the median template
  peak of its largest units).  It used build_masks' 15/32 guess, read as 1-based.
- `fiber-realign --check` (report what a realign would do: the windows' geometry, moves, template peak vs
  target, same-sample pairs, inversions; writes nothing) and `--recut` (re-extract every variant at the
  current `.res`, move nothing).
- `fiber-realign --park-duplicates` (`FK_REALIGN_PARK_DUPS`, on in the shipped profiles and always in
  fiber-session's commit): the later of two spikes of one unit that a commit puts on one sample goes to
  cluster 0 as one child atom under 0 (in fiber-session's `--out-variant` copy too; with `--out-tag`
  only in the clu emitted under the tag); cross-unit pairs and inversions are listed in
  `<base>.realign.<variant>.<group>[.<stage>].tsv`.  No row is removed or reordered.
- **fiber-session numbers fibers and atoms from 2**: its first fiber was cluster 1 and its first atom 1,
  Klusters' noise bin for every later stage, so both were lost downstream.  `.fibers` gains `clu_id` (each
  row's cluster in the written `.clu`) and `meta_n_spikes`; the merge-candidate list gains `clu_a` /
  `clu_b`; fiber-relink and fiber-validate-merges read them (pre-0.51 files: gid + 1, with a warning).
- The deconvolution manifest is written after the hierarchy save and the commit: written cluster ids,
  committed `res`, and a new `res_detect` column (the time the pursuit saw).
- `FiberHierarchy.refiberize` parents the empty children of a vanished fiber to 0 instead of keeping an id
  that can alias a live cluster; `FiberHierarchy.load` keeps a 0 parent for a child that holds spikes.
- `nio.apply_spike_keep` (fiber-refine `--dedup`) subsets the `.clc` with its `.clu` (a `.clc` an earlier
  dedup left at the old count is quarantined on its own, not with its stage), subsets the realign
  offsets, and moves the group's deconvolution manifests and realign pair lists aside as `.stalebkp`.
- fiber-validate-merges on a pre-0.51 candidate list keeps the pairs of fiber 0 (cluster 1 there).
- **Sorts change**: cluster and atom ids shift by one (and are recorded), and a commit's windows move one
  sample earlier.

## [0.50.0] — fiber-session commits its own realign
- At the end of a run fiber-session aligns every spike to its fiber's template, writes the shifted
  times to the shared `.res` (+ `<base>.offsets.<group>.npy`), and rebuilds EVERY waveform variant's
  `.spk` on disk at those times, each with a fresh `.fet` from its `.pca` (`_D` alias `.fet` too).
  `--realign auto|reextract|shift-spk|off` (`FK_SESSION_REALIGN`, default `auto`: re-extract from the
  `.fil` when there is one, else roll the `.spk`); `--realign-algo` (`FK_REALIGN_ALGO`);
  `--realign-fil`.  It runs fiber-realign's own `main()` with the flags of the plan step it replaces
  (`--reextract --refeaturize --variants all` on the `.clu` just written): byte-identical output.
  `--emit-fet` now writes its `.fet` from the committed waveforms and times.
- The shipped plans drop the `realign-session` step (fiber-kit.yaml: 7 steps, fiber-kit-exp.yaml: 10);
  `--emit-plan` drops its `fiber-realign` after fiber-session; a stage-list run skips the stage-2
  fiber-realign when fiber-session committed one.  `--realign off` brings back the old outputs.
- fiber-realign `--reextract`: a variant no `.fil` transform derives (fiber-adlp's `standard_ADLP`,
  a `_C` token without a usable sdiffPairs) is now rolled by the offsets instead of skipped, so its
  `.spk`/`.fet` no longer go stale against the shifted `.res`.  fiber-realign takes `--sdiff-pairs`.

## [0.49.2] — the configuration templates name their steps
- The embedded plans in `fiber-kit.yaml` and `fiber-kit-exp.yaml` give every step a `name:`
  (`session`, `realign-session`, `chunk`, `realign-chunk`, `link`, `realign-final`, ...), so
  `fk run 6 realign-chunk` or `fk run 6 chunk..link` works on them; the `<stage>-<k>` ids still do.
  The commands each step runs are unchanged.
- `fiber-pipeline --emit-plan` and `plans/TEMPLATE-all-stages.yaml` document `name:`, `fk list`
  and `fk run`.  The template's stray fiber-stochastic stub (inside the header) is now a
  generated block like every other stage.

## [0.49.1] — `fk run <elec>` runs the whole plan
- `fk run 6` runs what `fiber-pipeline 6` runs (the plan, or the canonical stages without one);
  steps, `--from` and `--to` narrow it.

## [0.49.0] — unique plan step names; `fk list` / `fk run`
- Every plan step has a unique id, `<stage>-<k>` (its k-th run of that stage in plan order:
  `fiber-realign-1/2/3` in fiber-kit-exp.yaml), and may carry its own `name:` alias.
  `fiber_kit.plan_steps` holds the rule for the runner, `fk` and fiber-plan-lint.
- `fk list` prints the steps of the plan `fiber-pipeline` would run; `fk run <elec> <step>...` runs chosen ones.
- `fiber-pipeline <elec> --step <s>` (repeatable), `--from` / `--to`, and positional step names
  (`fiber-pipeline 5 fiber-realign-2`) run selected plan steps with their plan tags and params.
  A bare stage name keeps its stand-alone meaning; a stage that runs several times is refused
  with its ids.
- fiber-plan-edit keeps `name:`; fiber-plan-lint flags malformed or duplicate names.

## [0.48.0] — whitening without a .fil; `.nst`; amplitude-dependent low-pass
- `fiber-session --whiten-source` (`FK_SESSION_WHITEN_SOURCE`): `nst` = the `.nst.<variant>.<group>`
  neurosuite-3's `ndm_extractspikes` now writes beside the `.spk` (noise in the `.spk`'s own transform,
  spike-free windows, outliers rejected; one whitener per session); `spk` = per chunk, the snippets'
  pre-peak samples (stationary block-Toeplitz; no file, any variant); `fil` as in 0.47.1; `auto` (default)
  = `nst` when the opened `.spk`'s `.nst` exists, else `fil`.  The `.fil` is opened, and the variant's
  `.fil` transform required, only when the `.fil` source is used.
- `neuro_io.read_nst` / `write_nst` (NSTE v1, byte-identical to neurosuite-3); `nst` is a
  MethodSpecific custody type; `contract.yaml` lists `nst` and moves to contract 2 with neurosuite-3;
  `custody_vectors.tsv` is neurosuite-3's new copy.
- `fiber-adlp`: amplitude-dependent zero-phase low-pass of the snippets, `fc = clip(fc_ref (A/A_ref)^gamma,
  fc_min, fc_max)` (filtfilt of a digital Butterworth per spike), written as a new variant with its own
  `.pca` and a per-spike cutoff sidecar; sort it with `--whiten-source spk`.
- `fiber-anchor-link` prints the anchor-vs-null AUC and TPR at 1 % FPR of its own score.

## [0.47.1] — the chunk whitener's noise model follows the .spk
- The chunk whitener built its noise model in the all-pairs stderiv space for every session, so a
  `stderiv_C5` session was whitened in the wrong space (on the reference session 80% of the whitened
  noise and ~93% of each spike's whitened energy sat in 15 common-mode-like directions) and a
  `.spk.standard` session got a derivative noise model.  The model is now built with
  `fiber_stderiv.space_transform` of the `.spk` each stage opens (`neuro_io.spk_variant`): fiber-session,
  fiber-stochastic, fiber-refine, fiber-stats, fiber-position, fiber-view, fiber-raw-vs-stderiv.
  Reference session, minutes 180-210: whiteness assignment 0.856 -> 0.948, coarse fibers 49 -> 35.
  **Sorts of C5 and standard sessions change: rerun fiber-session.**  `stderiv` sessions are unchanged.
- A `stderiv_C<n>` `.spk` needs the group's `sdiffPairs`: from the session YAML, or the new
  `--sdiff-pairs` on those stages.  Without it they refuse up front.  Stage logs name the noise model.
- `fiber_stderiv.space_transform` / `space_label` are the one definition of a variant's waveform
  transform (fiber-envelope / fiber-consensus delegate to it and now accept `_S` tokens);
  `fil_chunk_whitener` requires `transform=`.

## [0.47.0] — fiber-session re-sorts around kept units
- `fiber-session --in-stage <stage> --exclude-units 3,7,12-15`: the listed clusters of the input clu
  (`<base>.clu.<in-method>.<elec>.<stage>`; `--in-method` defaults to `--method`) never enter a chunk.
  `--exclude-as carry` (default) writes each back as one cluster and one atom of its own after the new
  fibers and logs `input -> output` ids; `--exclude-as noise` makes them cluster 0.  Refuses an input
  clu not aligned to the `.res` (a stage with its own `.res`), a malformed list, `--exclude-units`
  without `--in-stage`, and an output that would overwrite the input clu.
- `fiber-pipeline`: a fiber-session plan step's `in:` becomes `--in-stage`; the list rides `params:`.

## [0.46.0] — template deconvolution; burst-signature merges
- `fiber_deconv`: detection-anchored sub-sample template deconvolution -- continuous signal, R = 4
  Fourier phases, VAR(4) noise whitening, amplitude prior with a joint refit after every placement,
  stopping threshold from off-spike noise, agreement and partners judged up to a noise-corrected,
  lag-tolerant template equivalence (0.93).  Synthetic: 0.76-0.87 vs 0.53-0.62 for
  `fiber_collision` on collision-affected detections, timing error 0.3 vs 0.76 samples, 0.23 % of
  pure-noise pseudo-detections placed.
- `fiber-session --deconv` (`FK_SESSION_DECONV`; exp profile on): `deconv_agree`,
  `deconv_partner_frac`, `deconv_n_equiv` in `.fibers` and a `.deconv.<method>.<elec>.<stage>.tsv`
  manifest; changes no label.  `deconv_agree` tracks purity where the contaminant is distinguishable.
  `--deconv-assign` (`FK_SESSION_DECONV_ASSIGN`, off: 70-95 % precision on synthetic sessions) parks
  attributed noise spikes as child atoms.
- `fiber_ccg.burst_evidence` and `fiber-peel --burst-foot-lo` (`FK_PEEL_BURST_FOOT_LO`; exp profile
  0.80): below the footprint floor, merge only on a powered refractory dip AND the complex-spike burst
  signature (smaller fragment follows the larger by 2-15 ms).  Simulated: 10/15 bursting cells,
  0/25 negatives including a monosynaptic pair.
- fiber-session takes the detection peak from `build_masks` when the YAML has none (it was
  `int(None)` on the new path); fiber-stochastic slices `_process_chunk`'s result.
- Neither method has been run on a real session yet: docs/methods.md says what to read first.

## [0.45.0] — reference files shared with neurosuite-3; committed-model readers
- `test/golden/`: small `.wti`/`.wtf`/`.wtl`/`.eap`/`.tcl` files, `golden.*` written by neurosuite-3's
  C++ writers and `fk.*` by `neuro_io`, byte-identical in both repositories (hashes in `contract.yaml`).
  `test/test_golden_formats.py` reads every field of `golden.*`, checks that `write_wtf`, `write_eap`
  and `write_tcl` give the C++ bytes and that the writers still produce `fk.*`; neurosuite-3 reads `fk.*`
  with its C++ readers.
- `neuro_io.read_mti` / `read_mtf`: the committed template model Klusters renders from the lineage
  (`.mti`, `.wti` v2 schema; `.mtf`, `.wtf` schema).
- `doc/` (figures only) is merged into `docs/`.

## [0.44.1] — custody token bound; two int-range fixes
- `neuro_io.parse_variant_token`: an `S`/`C` order longer than 9 digits makes the token opaque, as in
  neurosuite-3 (whose parser overflowed).  `custody_vectors.tsv` is neurosuite-3's updated copy.
- `fiber_template.generate(eap=True)`: a class that is not a column of the `.eap` raises `ValueError`
  (it raised `IndexError`, and a negative class read a column from the end).
- `fiber_decollide.decollide_spikes`: output waveforms saturate to int16 instead of wrapping.
- `contract.yaml`: the contract number is bumped only when the protocol, the session schema or a format
  changes.

## [0.44.0] — contract.yaml; format versions in --ndm-version
- `contract.yaml` (repository root) records the ndm plugin protocol, the session-schema version, the
  version read and written of each version-tagged format (`.pca` 2/2, `.wti` 2/2, `.wtl` 2/1, `.tcl`
  1/1, `.eap` 1/1) and the hashes of the shared tables.  `test/test_contract.py` checks it against the
  code: the constants, what each writer puts on disk, and that each reader refuses the next version.
- `test_shared_contract.py` checks the tables against those hashes on its own; with `NS3_ROOT` it also
  checks that neurosuite-3's `contract.yaml` agrees and that each side reads every version the other
  writes.
- `ndm_fiber-kit --ndm-version` adds `format <name> <read> <write>` lines, which ndmanager uses to warn
  about a mismatch, and reports the source version when the package is not installed.
- `neuro_io`: the version literals are named constants (`WTI_VERSION`, `WTL_VERSION`, `TCL_VERSION`,
  `EAP_VERSION`); an unreachable copy of the old v1 `.wtl` reader is gone.
- Releases are tagged from this version on (`v0.44.0`).

## [0.43.0] — shared stderiv and grammar tables; one token rule
- `test/stderiv_vectors.tsv` (shared with neurosuite-3, where every engine now takes its spatial
  derivative from one header) is run against `fiber_stderiv.apply_stderiv_transform`: all 27 rows
  fiber-kit covers match to the sample, saturation and halves included.
- `test/sdiff_grammar_vectors.tsv` (shared) pins the order-4/5 sdiffPairs grammar.  fiber_stderiv's
  parsers now follow the C++ exactly: positions are ASCII digits only ('1_0', '+0', non-ASCII digits
  refused), a blank token is refused, a repeated set member is refused.  `morpho_eap.parse_sdiff_sets`
  delegates to fiber_stderiv instead of carrying a third copy.
- `parse_variant_token`: ASCII digits only, and a `_D` with lag 0 makes the token opaque, as in the
  three neurosuite-3 mirrors; `test/custody_vectors.tsv` gains `waveform_token` rows (run here) and
  `drops_last` rows (not mirrored -- fiber-kit never drops channels itself).

## [0.42.0] — sdiffPairs expressions; .wtl v2; CI and one version
- `sdiffPairs` can give one arithmetic expression per output channel over `c0`, `c1`, … (`+ - * / ^`,
  parentheses, usual precedence): spatial order 6, method token `stderiv_C6`, PCAE method 9, the
  same grammar and results as neurosuite-3's `channel_expr.hpp`, held in step by
  `test/sdiff_expr_vectors.tsv`.  Older patterns keep their meaning; in a session that mixes the
  forms, the others are evaluated as the expressions they compute.  The order-4 root output and
  cycles now follow the extractor (the root is wired to its predecessor).
- `neuro_io.read_wtl` reads `.wtl` version 2, the version neurosuite-3 writes (running mean/std per
  node), as well as version 1; it returned no nodes for every file Klusters saves.  `write_wtl` still
  writes version 1.
- CI (`.github/workflows/ci.yml`) runs `test/run_all.py` on every push with a neurosuite-3 checkout,
  so the shared vector tables are compared.  `test_shared_contract.py` fails when a table is missing
  on the other side (it skipped).
- `docs/stages.md` and `plans/TEMPLATE-all-stages.yaml` regenerated from the parsers (the drift check
  was failing).
- One version: `fiber_kit.__version__` said 0.25.0 and `PKG-INFO` 0.1.0 while `pyproject.toml` said
  0.41.0.  `PKG-INFO` (generated) is removed, and `test/test_version.py` checks that
  `pyproject.toml`, `__version__` and the newest CHANGELOG heading agree.

## [0.41.0] — theta_bg_topology: one latent map per theta phase bin
- analyze `--map-mode per-bin`: a separate map for each phase bin from that bin's own channel pattern
  (plus that bin's unit counts with --features units/both); per bin TwoNN, H1, continuity against the
  feature-shift null, and PHASE-SPECIFIC unit fields (spikes fired in that bin on that bin's map).
  Across bins: the fraction of a cycle's k nearest neighbours shared between two bins' spaces, real vs
  the second bin feature-shifted -- a code that names the same place at every phase gives high
  overlap, a phase-shifted code lower but above the null.  `--map-mode concat` (default) is the
  previous whole-cycle map.  Writes <out>.json, <out>_maps.npz (one map per bin), <out>.png.
- g6 full session, 3 bins, --smooth 1: per bin 6 PCA dims, TwoNN 8.1-8.2 vs null 8.1-8.3, H1 at the
  null, continuity 0.84-0.90 vs 0.88-0.93; k-NN overlap between bins 0.001 (chance 0.0006, null 0.001);
  the strongest fields are the same rate gradients in every bin (6:3, 6:90, 6:104).

## [0.40.6] — theta_bg_topology figure: the longest theta run is the longest
- The trajectory panel took `run[argmax(bincount(run))]` -- the run id OF the cycle whose index equals
  the longest run's id -- so it drew an arbitrary run (10 cycles on the g6 3-bin session; the longest
  has 157).  Figure only; the JSON metrics were unaffected.

## [0.40.5] — theta HMM fitted on usable cycles only
- The dropout cycles (flat .fil, theta index ~log(1e-30) = -69) were in the HMM fit and became the
  "non-theta" state on the full g6 session (mean -30, sd 33; 99.6 % of cycles called theta).  The HMM
  now fits only cycles that pass the flat/guard and sample checks, winsorised to the 0.5-99.5 % range.
  Full g6 session (3 phase bins): states 0.13 / 1.46 (sd 0.57 / 0.51), dwell 2.8 / 3.1 s, 53 % theta.

## [0.40.4] — fiber-session runs on non-32-sample spike windows (nSamples 64 / peak 33)
- The chunk whitener drew its spike-free baseline as fixed 32-sample snippets, but the feature mask is
  built peak-relative for the session's window (build_masks): with nSamples 64 / peak 33 (Plexon
  session) MASK_FULL is samples 29..43 -> "IndexError: index 32 is out of bounds".  The baseline
  snippets now have the spike window's length (`fil_chunk_whitener` passes nsamp; the other entry
  points size them from the mask).
- cluster_chunk (the coarse pass and the fine/reseed re-cuts in cluster_chunk_fine) was called without
  the session mask and fell back to the 32-sample MASK_FULL (samples 11..25), while the whitener was
  fit on the session mask: the features and the whitener described different samples.  On the 64-sample
  window that left every spike coarse-unassigned (0 fibers); on 42/peak-21 sessions (the reference)
  the coarse features sat 6 samples before the peak-relative span.  All three calls now pass the mask.
- Test session built from the 36-47 min g6 data re-extracted at 64/33: 0 -> 224 global fibers (42/21
  on the same spikes: 225).  42/21 sorts change with the corrected coarse mask (ARI 0.19 vs before --
  fiber-session over-splits, so the partition is not stable to feature changes); re-run fiber-session
  on existing sessions before comparing against earlier fiber_session outputs.  test_spike_window
  (fails on the old code with the reported IndexError).

## [0.40.3] — theta_bg_topology: log theta index and a two-state HMM for theta epochs
- extract: the per-cycle theta index is now log( P[5-12] / (P[0-4] + P[13-15]) ) (band powers of the
  reference LFP, zero-phase Butterworth, 1-s window `--ratio-win-s`, averaged over the cycle); the cycle
  filter band is 5-12 Hz.  Stored as `ratio` with `ratio_kind = log_theta_index`.
- analyze `--theta-state hmm` (default): a two-state Gaussian HMM (Baum-Welch + Viterbi, no extra
  dependency) over the time-ordered per-cycle index; the high-mean state is theta.  Reports both
  states' means/sds, stay probabilities, mean dwell times and the theta fraction.  `--theta-state
  threshold` keeps `--theta-min` (absolute, on the log index) or the `--theta-q` quantile.  Old npz
  files (amplitude ratio) are fed to the HMM as log(ratio).
- 11-min g6 test: states at 0.55 / 1.95 (sd 0.54 / 0.51), dwell 2.7 / 3.8 s, 59 % theta; theta runs
  190 -> 108 (longer, contiguous epochs instead of single cycles over a median cut).  Synthetic
  two-state sequence: 99.6 % of samples labelled correctly.

## [0.40.2] — theta_bg_topology: drop dropout epochs before mapping
- Cycles on flat signal (zero background in the .fil: dropouts or blanked stretches) and within
  `--flat-guard-s` (150 s) of them are excluded; `--exclude-min` drops manual windows.  On the g6
  session (6 phase bins) the detached islands of the latent map (69.9-72 and 78-80.5 min, around the
  three dropouts) carried ALL of the long H1 bars (3.5 vs null 0.6) and the largest H0 gap; with them
  excluded the map is one component, H1 0.29 vs null 0.22, dimension 18.5 vs 18.0, continuity = null.

## [0.40.1] — theta_bg_topology: bins of the Hilbert theta phase
- extract `--phase-mode phase` (new default): `--phase-bins` equal bins of the Hilbert theta phase from
  the trough (6 bins = 60 deg each); theta is asymmetric, so these differ in duration.  `--phase-mode
  time` keeps the previous equal-duration bins.  The mode is recorded in the npz meta.

## [0.40.0] — every package installs by default; theta_bg_topology maps cycles without position
- pyproject: all optional packages are core dependencies (diptest, matplotlib, pyqtgraph, PySide6,
  holoviews, bokeh, openTSNE, networkx, scikit-image) plus umap-learn and gudhi for the topology tool.
  The extras keep their names; only `gpu` (cupy, CUDA-specific) stays optional.  Resolves in a fresh venv.
- tools/theta_bg_topology.py
  - extract: `--phase-bins` (default 8, was 4 fixed quarters); a `--clu-stage` whose `--clu-method`
    file is missing now falls back to the stage's single `.clu.*.<g>.<stage>` (a mistyped
    stderiv_C4_D34 silently dropped every group's unit counts) and errors out when no group resolves.
  - analyze: the map is the point now, not time.  Slow drift is removed per feature (`--detrend-s`,
    60 s running mean) -- the previous Isomap was ordered by session time.  A 2-D LATENT MAP of the
    theta cycles (UMAP, else Isomap) from the phase x channel background pattern; continuity of
    consecutive cycles on it; place-like FIELDS of the sorted units on the latent map (Skaggs
    information vs a 60-s circular-shift null, odd/even-run stability) -- with `--features bg` the map
    is background-only, so a stable unit field means the background pattern predicts that unit.
    Saves `<out>_map.npz` (map coordinates per cycle) and a figure (map by time, occupancy, longest
    theta run as a trajectory, null map, top unit fields).

## [0.39.2] — anchor-link translates fiber-session's anchors into the fragment layer
- fiber-session writes its overlap anchors as pairs of its own .clc ATOM ids.  With `frag-layer: units`
  on exp_chunk (the exp plan) there is no exp_chunk .fibers, so the linker fell back to fiber-session's
  and read those atom ids as exp_chunk UNIT ids: any number that exists in both layers became a pair
  of unrelated units, and seeds bypass the score floor (g8 run: "72 -> 17 usable ... recovers 6% of the
  anchors ... seeded 16/17").  Now each anchor atom maps to the fragment holding >= 0.6 of its spikes
  (`--anchor-stage`, default fiber_session; `--anchor-majority`), pairs inside one fragment or without
  a majority drop out, and with no .clc to resolve against nothing is seeded.  Mini session (overlap
  sort, units layer): the old path read 10 arbitrary pairs; translated, 16 of 17 anchors map to pairs
  already in one unit and 1 has no majority, so nothing is (wrongly) seeded.  test_anchor_link_reserve.

## [0.39.1] — tools/theta_bg_topology.py: theta-cycle background and place-like topology over a session
- `extract` streams the session once: theta cycles (5-11 Hz troughs of the group-median LFP; theta/delta
  ratio, duration, amplitude per cycle), the spike-free background per cycle x theta quarter x channel
  (.fil mean-square >= --guard-ms from any spike of the channel's group), optional unit counts per
  cycle quarter from a clu stage.  `analyze` (iterate on the npz): background pattern / units / both,
  smoothed over consecutive cycles, PCA; TwoNN dimension, revisit structure, persistent homology
  (ripser or gudhi) and an Isomap embedding against a per-feature circular-shift null of the same
  dimension; with --pos (.whl) blocked-CV position decoding and feature-vs-place distance.
- Smoke-tested on the 11-min group-6 window (8 ch, no position): runs end to end in ~2 min; there the
  background pattern is ~10-dimensional and its H1 bars match the null -- too little coverage to judge,
  the full-session multi-group run is the test.

## [0.39.0] — fiber-lowmix: split low-amplitude mixture units after noise whitening
- New stage `fiber-lowmix` (opt-in plan step, POST stage; knobs FK_LOWMIX_* in fiber-kit-exp.yaml [D]).
  Per low-amplitude unit: standard .spk, per-channel Legendre smoothing (deg 20), whitening with the
  covariance of spike-free .fil snippets, GMM split gated by a null that keeps each spike's fitted
  amplitude/jitter/position.  Each group: copy of another unit (cos >= 0.95 and a local-rate refractory
  ALLOW relative to the target's own ratio) -> returned; largest other group keeps the unit; others new
  or novel (novel -> noise as reviewable .clc children).  Writes .clu/.clc/.clp + `<base>.lmx.<g>.<tag>`.
- g6 (fix_final): purity 0.805 -> 0.806, ARI 0.525 -> 0.532, 429 -> 466 units; the cores are not more
  refractory than before.  Measurements and the analysis behind it: docs/methods.md "Low-amplitude
  mixtures".  test_lowmix (two-source split vs a one-source negative control, refractory one vs two
  neurons, verdict rules, spike-free snippets).

## [0.38.6] — backbone-link writes the noise bin's children under parent 1
- backbone-link's .clp mapped every child through the fragment->unit table, which has no entry for the
  reserve bin, so the children of cluster 1 got parent 0 and the .clc/.clp triple disagreed with the
  .clu on every reserved spike (16,975 on g6) -- a later stage carrying the hierarchy fell back to .clu
  only.  Children of 1 now map to 1; verified: the rebuilt triple equals the .clu, .clu unchanged.

## [0.38.5] — xcorr-merge's drift table lands where backbone-link looks for it
- fiber-xcorr-merge wrote its drift transform with `np.savez(<path>)`, which appends `.npz`, while
  fiber-pipeline passes backbone-link the canonical `<base>.fibers.<method>.<group>.<stage>`: the
  `drift: exp_xcm` step printed "--drift ignored: ... not found" and ran without drift.  The table is
  now written through a file handle (as fiber-session writes its .fibers), and backbone-link also
  accepts a `<path>.npz` left by earlier runs.  Verified on the g6 run: drift ON from the canonical path.

## [0.38.4] — fiber-anchor-link holds the reserve bins (0, 1) out of the link
- The quarantine re-entered the merge at the linker.  fiber-consensus (quarantine-only) parks its
  quarantine in cluster 1 and intrachunk keeps it there, but fiber-anchor-link took every `clu > 0`
  as an atom: cluster 1 spans the session, so it (a) fused all chunk intervals into one and forced
  the fixed-bin chunk fallback, (b) was chained onto real cells -- on the g6 reference run 14,190
  quarantined spikes ended in exp_final cell 90 together with three real units (15,933 spikes) --
  and (c) the linker numbered its largest chain 1, which xcorr-merge and backbone-link (units are
  `clu > 1`) then treated as noise: a real 7,513-spike chain sat in cluster 1 through exp_final.
- Now: reserve atoms are neither linked nor chunk-assigned, 0/1 pass through unchanged, cells are
  numbered from 2 (`cells_from_links`), the per-spike consolidation excludes both, and on the atom
  (.clc) layer children whose flat label is 1 are folded back into 1.  Re-run on the uploaded
  exp_chunk: all 14,190 quarantined spikes stay in 1, the chunk partition recovers 9 chunks from
  the atom spans, and the three units form their own cell.  test_anchor_link_reserve.

## [0.38.3] — realign = new .res times + every variant re-extracted from the .fil; custody .spk resolution
- A realign changes ONLY the `.res` times; every waveform variant is then re-extracted from the `.fil`
  at the corrected times and its `.fet` regenerated (`_D` alias `.fet` re-projected from its waveform
  token).  `fiber-realign --variants all` = every `.spk.<token>.<group>` on disk (+ the `--variant`'s
  waveform token); fiber-pipeline passes it for the plan's realign and the post-step re-rolls, and
  fiber-kit-exp.yaml sets `FK_REALIGN_AFTER_MODE: reextract`.
- `_S1/_S2/_S3` (the extractor's plain spatial derivatives: nearest-neighbour / Laplacian / all-pairs)
  are re-derived (`fiber_stderiv.plain_order_spec`; tested against a literal port of computeSDiff)
  instead of skipped -- `stderiv_S3` waveforms were left stale against the realigned `.res`.
- `neuro_io.resolve` / `classify_type` / `waveform_token`: the neurosuite-3 custody contract
  (doc/ndmanager-plugins/formats/naming.md).  `.spk`/`.res` are Shared: exact token -> its WAVEFORM
  token for a `_D<lag><dims>` token (`stderiv_C5_D34` -> `.spk.stderiv_C5`) -> standard -> untagged;
  `.clu/.clc/.fet/.pca/...` strict.  test_custody_token_conformance runs the shared table's `classify`
  and `resolve` rows too (87 checks).  fiber-realign resolves its waveforms through it and refuses raw
  waveforms for a stderiv token; `open_spkD` tries the waveform token before the family walk.  With
  `FK_REALIGN_VARIANT: stderiv_C5_D34` and no `_D34` `.spk` the plan's realign failed ("no .spkD/.spk");
  the misleading "spkD" wording is gone from the error.
- fiber-pipeline: the plan's realign step passes `--no-emit-clu` (it re-emitted a stage-tagged `.clu`
  over the base `.clu.<method>.<elec>`).
- Checked on a copy of the g6 mini-session laid out like the reference session: `.res` moved for 6,040
  spikes; standard / stderiv_C5 / stderiv_S3 `.spk` re-extracted (300/300 sampled spikes equal the `.fil`
  window, resp. its transform, at the new time); `.clu/.clc/.clp/.pca/.yaml` untouched; no `_D` `.spk`.

## [0.38.2] — fiber-kit-exp.yaml: realign threshold 0.10 (the curator's Klusters setting)
- `FK_REALIGN_MIN_SCORE: 0.10`; parity with the compiled Klusters code re-checked at 0.10 on the g6
  sort (every spike, both `.spk` variants); template correlation 0.618 / 0.707 vs 0.592 / 0.652 at
  0.70.  Golden 0.10 shifts added to test_realign_parity.  The code default stays Klusters' 0.70.

## [0.38.1] — fiber-realign: exact parity with Klusters' realign, RMS mode
- `klusters_offsets` follows `KlustersDoc::realignSpikes` step for step: int16-truncated mean template,
  circular pre-roll to `peakSampleIndex - 1` for every variant, float32 scores, cumulative per-pass lag
  search, Klusters' defaults (threshold 0.70, 2 iterations, max shift `(peak-1)//2`).  New
  `--recenter-rms` (Klusters realign mode 2) and `--rmin`; knobs `FK_REALIGN_MIN_SCORE / _ITERS /
  _MAX_SHIFT / _RECENTER_RMS / _RMIN` (fiber-kit-exp.yaml: RMS mode on).
- Against the compiled Klusters code on the g6 36-47 min sort (19,543 spikes): identical shift on every
  spike, standard and stderiv_C5_D34, both modes.  Before: 17.6 % / 51.6 % agreement with mode 2.
- Behaviour change: fewer spikes move (the 0.70 threshold), so units are slightly less tight than with
  the old settings (template correlation 0.592 vs 0.617 on stderiv); set MIN_SCORE 0 / ITERS 4 for the
  old behaviour.  `test/test_realign_parity.py`: golden shifts from the compiled code.

## [0.38.0] — fiber-consensus cleans before merging: quarantine-only mode, plan placement
- `--quarantine-only` / `FK_CONSENSUS_QUARANTINE_ONLY`: kept spikes keep their input labels,
  quarantined spikes go to noise (1); the `.clc/.clp` hierarchy is carried (one new atom -> parent 1;
  the rebuilt `.clu` must match or the triple is not written); island / rim clusters go to a
  `<out>_qreview` stage.  `carry_quarantine` + test.
- fiber-kit-exp.yaml: plan runs fiber-consensus (quarantine-only) right after fiber-session's
  realign, so intrachunk and the linkers merge cleaned fragments (`fiber_session -> fs_clean ->
  exp_chunk ...`); `FK_REALIGN_VARIANT: stderiv_C5_D34` and `FK_CONSENSUS_RES_VARIANT: stderiv_S3`
  for the reference session.
- `.res`: fiber-consensus and fiber-envelope resolve the shared `.res` like every other stage unless
  `--res-variant` / `FK_CONSENSUS_RES_VARIANT` pins one (was a hard-coded `stderiv`).
- docs/pipeline.md: the plan order and the order of operations inside fiber-consensus.
- Not yet measured: whether intrachunk / the linkers produce cleaner units from `fs_clean`.

## [0.37.4] — refractory cannot-link in the envelope / consensus merge
- `fiber_envelope.refractory_cannot_link` (fiber_ccg.refractory_gate, power-aware, (0.6, 1.5] ms by
  default) and `any_cannot_link`; on by default in `fiber-consensus` (`FK_CONSENSUS_REFRAC_MS`,
  `FK_CONSENSUS_REFRAC_MIN_EXP`) and in `fiber-envelope --merge` (`--refrac-ms`, `--refrac-min-exp`).
  Judged without ground truth on the g6 11.7 k-spike neighbourhood: the merge without it built an
  8.5 k-spike cell and raised violating pairs from 2 / 2 (fragments) to 38 / 69 (0.6-1.0 / 1.0-1.5
  ms); with it 13 / 20, the largest cell 3.2 k, 13 of 14 powered pairs vetoed.
- docs/methods.md: GT-free evidence -- the round-1 quarantine is refractory-enriched 3.8x; exp_final
  cell 21 (7.2 k spikes) has no refractory gap (autocorrelogram at chance from 1 ms) and small
  amplitude: multi-unit background, not a fast-spiking interneuron; a region-local Legendre
  oblique-axis library finds no axis that splits it into refractory halves.

## [0.37.3] — fiber-consensus: neighbourhood cap off by default; findings
- `FK_CONSENSUS_MAX_SPIKES` default 4000 -> 0 (off).  On g6 the 11.7 k-spike group d' 12 forms
  holds 9 GT cells that are genuinely confusable; consensus on the whole group gives 7 cells at
  purity 0.872 / completeness 0.81 / ARI 0.59 against 26 cells at 0.874 / 0.58 / 0.17 when the
  0.37.1 cap splits it, with higher quarantine precision (0.73 vs 0.70).  The cap stays as a
  run-time bound.  The lone-cell crash fix of 0.37.1 is unaffected.
- docs/methods.md: measured and not adopted -- quarantine held out of a fiber-session rerun;
  fiber direction-profile / Jacobian neighbourhoods; contamination orthogonal to the fiber
  Jacobian; t-SNE stratification of dense cells with median-waveform collision tests.

## [0.37.2] — fiber-consensus: parallel t-SNE runs
- `--workers` / `FK_CONSENSUS_WORKERS` (default 0 = every core): the independent t-SNE runs and
  watershed maps -- ~93 % of a neighbourhood's time -- run in parallel processes (spawn; BLAS/OMP
  threads pinned to 1 per worker).  Subsets and seeds are drawn up front, so results are identical
  for any worker count (tested).  A 1.5 k-spike neighbourhood takes ~3.5 min serially.

## [0.37.1] — fiber-consensus: neighbourhood cap, lone-cell crash
- `--max-spikes` / `FK_CONSENSUS_MAX_SPIKES` (default 4000): a neighbourhood holding more spikes is
  re-cut on its own linkage at theta*0.8, *0.64, ... until every piece fits.  Average linkage at
  d' 12 chained a whole chunk's low-SNR fragments into one group (g6: 11.7 k spikes; a full chunk
  elsewhere: 180 units / 33.5 k spikes merged into 8 cells), far beyond the 1-3 k spikes per
  neighbourhood the consensus was measured on.
- fiber_envelope: a neighbourhood whose communities merge into ONE cell no longer crashes
  (`partner_dictionary` on an empty dictionary -> `ValueError: need at least one array to stack`);
  a lone unit gets no partners.
- fiber-pipeline banner notes that a `--clu-stage` / `--out-tag` override wins over the default tags.

## [0.37.0] — fiber-consensus: watershed categories, FK_CONSENSUS_* knobs, plannable stage
- `watershed_categories`: each quarantined spike is sorted by t-SNE watershed basins voted over
  `--ws-runs` maps into island / own-rim / other-cell.  On g6, islands (20-38 % of the quarantine) are
  66-98 % foreign or noise; own-rim in a clean cell is mostly the cell's own.  Two quarantine clusters
  per neighbourhood (island, rim); `.qrn` gains `category`, `ws_agree`, `ws_dest` (output ids).
- Chunked processing (`--chunk-min`, default the session chunk) and `--keep-ids` (cells keep the
  majority input unit's id), so the stage runs on a chunk-local or a linked sort; `--window`,
  `--min-spikes`, `--seed`.
- Knobs resolve CLI > `FK_CONSENSUS_*` env > `$FK_CONFIG` > default; documented section [C] in
  fiber-kit-exp.yaml.  `fiber-consensus` is a plannable fiber-pipeline stage and a plan-editor node.
- A second quarantine round was measured (purity -0.002..+0.004) and not implemented.
- `scikit-image` joins the `consensus` extra.  fiber-pipeline `-l` no longer fails on the missing
  fiber-stochastic summary line.

## [0.36.0] — fiber-consensus: perimeter rule, cell-relative envelope
- `perimeter_rank`: a spike's average 2-D radial rank inside its cell.  On g6 the rim of a
  contaminated cell holds its collisions (20-38 % vs 1-2 % in the core) and foreign spikes.
- Quarantine now uses the envelope refitted ON THE CELLS and defaults to collision OR perimeter
  (rank >= 0.8 and residual energy / bump score >= 2 robust z): 6.7-7.8 % of spikes at 0.40-0.85
  precision, purity +0.02-0.05 on four neighbourhoods.  Cell switching is opt-in (`--rules`).
- `Envelope.resid_energy` (per-spike residual energy after a T + b T'); `.qrn` gains perimeter,
  residual / bump z.
- docs/pipeline.md: where fiber-envelope and fiber-consensus sit (within-chunk, before linking).

## [0.35.0] — fiber-consensus: resampled t-SNE consensus + quarantine
- New stage `fiber-consensus` (`fiber_consensus`): R subsampled t-SNE runs (openTSNE) of a
  neighbourhood of similar clusters in the `.fet` space, Louvain communities on the co-neighbour
  probability, communities joined into cells by the envelope merge, and a quarantine cluster for
  spikes that switch cell across runs or carry a near collision, with a `.qrn` reason table.
- On four g6 neighbourhoods the quarantine takes 4.7-10.2 % of spikes at 0.34-0.83 precision
  against base rates of 0.08-0.42, and lifts cell purity by 0.02-0.03.  Across-run distance variance
  was measured and rejected (a density measure; AUC 0.25-0.55).
- Optional extra `consensus = [openTSNE, networkx]`; test_consensus (slow, ~3 min).

## [0.34.0] — fiber-envelope: cell layer, cell-aware merging, cell type; probe top-to-bottom fix
- Cell layer (`fit_envelope(times=...)`, `run(sr=...)`): amplitude-dependent shape S, within-window
  drift D and the fiber_adapt adaptation law per unit; `amp_model()` / `channel_ratio()` measure spread
  beyond what one cell does (single-cell ratio 1.28 -> 1.23 in stderiv_C5; bursting cell 28 2.12 -> 1.45).
- `merge_units` / `fiber-envelope --merge THETA`: average linkage on envelope d' with merge inflation
  as a cannot-link (31/32 vetoed g6 pairs were different cells); keeps GT2 ARI at 0.68-0.70 for
  THETA 7-10 where d' alone falls to 0.66.  Writes a `.envmerge` clu stage + per-pair evidence.
- `type_features` (raw width, burst index, rate) in `.env`; `cell_type_conflict` behind
  `--type-veto` (off: it cost 0.11 ARI on fragments).  Adaptation-coupling and position vetoes and a
  residual-bimodality split were measured and NOT adopted (docs/methods.md).
- `fiber_localize.load_geometry(flip_y=...)` / `FIBER_PROBE_FLIP_Y`: mirror each shank of a .probe
  top to bottom; `morpho_eap.group_geometry` now reads through it.  On g6, 71 CA1 pyramidal
  reconstructions fit the recorded footprints in their canonical orientation only with the mirrored
  geometry (docs/morphology.md).
- test_envelope: 12 checks (adapting synthetic cell, energy-band fragment re-joining, probe mirror,
  type features, CLI merge).

## [0.33.0] — fiber-envelope: amplitude-scaled channel check
- `Envelope.amp_model()`: noise_c + var_a T^2 + var_b T'^2 with one var_a / var_b for every channel,
  so the expected spread widens with each channel's amplitude (a near unit's big channels may vary
  more; a distant unit stays noise-dominated).
- `Envelope.channel_ratio(which)`: observed / amplitude-predicted variance per channel; `kept`
  (default) excludes only near collisions a partner explained, so a merged minority -- flagged by the
  bump test but not explainable -- is still counted.  `.env` gains chan_ratio_max + per channel.
- Real g6: independent of unit SNR (Spearman ~0, vs +0.5 for the noise-relative excess) and a
  better merge detector (10 % look-alike admixture AUC 0.75 vs 0.53-0.55).  docs/methods.md also
  records the expected effect on fiber_session output: the envelope d' merge lifts GT2 ARI
  0.10 -> 0.61-0.70 within a chunk; the collision-robust noise estimate itself adds <= 0.004 at the
  best threshold.
- test_envelope: channel_ratio ~1 for single units of different amplitude, > 1.5 with 10 % merged.

## [0.32.0] — fiber-envelope: collision-robust variance envelope + near-collision partners
The per-unit spread used by band_overlap / dispersion_profile is a plain std over cluster
members, which near collisions (a second spike inside the window, often hidden by the detector's
dead time) inflate.  New module + stage `fiber_envelope` / `fiber-envelope`:
- noise floor from off-spike windows of the continuous trace; per-spike amplitude/jitter fit
  against [T, T']; a time-localised bump score with an empirical noise null; iterative exclusion,
  so the envelope sigma^2 = noise + var_a T^2 + var_b T'^2 + robust excess is not contaminated.
- near-collision partners: extended templates matched at non-circular shifts, gated by the
  bump test and a noise-calibrated gain threshold; written as `.nco` (per spike) and `.env`
  (per unit) tables and, with `--eap`, as `.eap` membership on the analysed stage.
- `separation()` -- envelope-normalised d' merge statistic with the estimation-noise term removed.
- test/test_envelope.py: synthetic recording with known truth (5 checks incl. the CLI).
- `space_transform()`: standard | stderiv | stderiv_C<order> (the session's sdiffPairs, read
  from the yaml), via fiber_stderiv.apply_stderiv_transform -- the .spk's own math.  The CLI's
  space defaults to the --clu-method token (stderiv_C5_D34 -> stderiv_C5).
- Real g6, exp_final cells: the plain std overstates the variance by ~22 % in the standard space
  and ~5-10 % in the stderiv spaces; recovered hidden partners fill 74-92 % of the CCG-predicted
  dead-time hole.  The bias does not change fragment merge decisions in any space (d' AUC equal to
  the third decimal), so no existing gate's default is switched.  Details: docs/methods.md.

## [0.31.0] — reduce global-basis features to the caller's dims before the CEM
Fixes the dimensionality mismatch introduced in 0.29/0.30: the channel-major basis
projection is nCh*nComp wide (14 at nFeatures=2, 28 at nFeatures=4), but it was fed
straight into klustakwik's full-covariance CEM (and the GMM split), which costs O(D^2)
params and needs >= D+2 points per cluster -- so the BIC penalty silently over-merged
(on real g5 data a 1500-spike cluster split into 5 instead of ~12, smallest survivor 148).
- `fiber_pca.cluster_features(..., dims=None)`: when `dims` is given, reduce the projected
  features to their top-`dims` SVD scores (mean-centred) -- exactly as `local_features`
  reduces raw waveforms -- keeping the drift-stable global basis as the feature SPACE while
  handing the splitter the dimensionality it is tuned for.
- Every basis->splitter site now passes its existing target dims, IDENTICALLY in
  fiber-session and fiber-refine (they mirror): gmm_split->pca_k, _aligned_pca->k,
  _rkk_realign->dims, the cluster_chunk_fine legacy rkk->rkk_dims / dip->dip_dim, and on the
  refine side _feats->d and _rkk_realign->dims.
- Result on g5: rkk on basis->6 gives 10 clusters (vs the bug's 5; local SVD-6 gives 12),
  so resolution is restored and nFeatures 2->4 no longer explodes the CEM parameter count.

## [0.30.0] — dipsplit & realign shape features from the global ndm_pca basis
Extends 0.29.0 (requires it) to the remaining `fiber_session` SHAPE featurizers, so the
DipSplit and rkk-realign paths cluster on the global basis like gmm/_feats already do.
- `_aligned_pca(..., basis=None)`: the hub for `_dipsplit_realign` and `_rkk_realign` --
  still realigns each node to its own median by circular xcorr (unchanged), but projects the
  aligned waveforms onto the global basis when given (else local SVD).
- `_dipsplit_realign(..., basis=None)` threads the basis through the recursion;
  `_rkk_realign(..., basis=None)` uses it for the seed AND the per-iteration re-featurize.
- `cluster_chunk_fine`'s legacy (non-realign) rkk and dip branches project onto the basis too,
  so every fine-split feature path is consistent; the basis already arrives via the 0.29.0
  `basis=` thread, so no new main wiring.
- The xcorr realignment itself is NOT changed -- only the feature projection moves to the
  basis. `realign_pca` already took a basis (projection-energy realign) and is untouched.
- Still LEFT LOCAL (different feature spaces, by design): `_energy_band_split` (PC1=energy
  confound gate) and `_variance_split` (channel-residual-profile discriminant).

## [0.29.0] — split-stage shape features from the global ndm_pca basis
Extends 0.28.0 (requires it) from `klustakwik` to the in-pipeline SHAPE splitters, so the
fine/refine splits cluster on the shared global basis instead of a per-call local SVD.
- `fiber_session.gmm_split(..., basis=None)` and `cluster_chunk_fine(..., basis=None)`:
  realign as before, then project onto the global basis when given (else local SVD).
- `fiber_refine`: `Ctx` gains an optional `basis` field (default None); `_feats` and the
  dipsplit re-featurize loop project the per-cluster-aligned waveforms onto it. `refine()`
  / `refine_chunked()` take `basis=` and thread it through both the direct and drift-aware
  paths.
- Both mains load the basis via `read_cluster_basis` (variant = `--method` / `--out-method`)
  and expose `--no-cluster-basis` to force the legacy local-SVD path.
- `_energy_band_split` is LEFT UNTOUCHED: its confound gate needs PC1 to be the local
  energy/drift axis, which a channel-major global projection does not preserve. Energy and
  whitener computations everywhere remain local; only the shape features feeding the
  BIC-GMM / rkk step are swapped.

## [0.28.0] — clustering features from the global ndm_pca basis
- `fiber_pca.cluster_features(spk, basis)`: project (realigned) waveforms onto the
  GLOBAL per-channel `.pca`/`.pcaD` basis (the ndm_pca / `process_pca` basis), returning
  the `.fet` feature space. One shared basis across every chunk/run, so features are
  comparable and a basis change (`session.yaml` `nFeatures` 2→4, `--varimax`) propagates
  into clustering with no code change. The stderiv all-pairwise derivative makes the
  trailing channel dependent, so a basis with one fewer channel than the input drops it
  (matching `process_pca_stderiv`); an unresolvable channel mismatch returns `None`.
- `fiber_pca.read_cluster_basis(base, elec, method)`: resolve+read the basis
  (`standard`|`stderiv`), or `None` when absent.
- `fiber_pca.local_features(...)`: the legacy per-call local-SVD feature path, factored
  out as the explicit fallback (its basis is refit per call, so it is NOT comparable
  across chunks).
- `klustakwik.py` CLI now clusters on the global basis when present (`--method`,
  `--no-global-basis`), falling back to local SVD only when no basis is found or channels
  mismatch. Splitter wiring in `fiber_session`/`fiber_refine` is deferred (each splitter's
  feature semantics — e.g. `_energy_band_split`'s PC1=energy axis — need per-site handling).

## [0.27.0] — pipeline driver: positional electrode + rename to `fiber-pipeline`
- `scripts/run_fiber_pipeline` renamed to `scripts/fiber-pipeline` (installed on PATH
  under the new name). The electrode group is now the FIRST positional argument:
  `fiber-pipeline <elec> [all | <stage> ...]`, e.g. `fiber-pipeline 5 fiber-refine`.
  `FK_ELEC` is retained as a fallback when no leading group is given.
- fiber-refine: `--out-variant` renamed to `--out-stage` (it is the output STAGE tag;
  `--out-variant` is reserved for the feature variant everywhere, as in fiber-realign).
- neuro_io.apply_spike_keep: dedup propagation no longer edits backups (dated snapshots,
  `*_bkp`), byte-split fragments (`.part.*`), or sidecars (`.units.npz`); electrode is
  matched exactly as the first numeric token (fixes a latent cross-group match).

## [0.26.0] — installed pipeline driver (`run_fiber_pipeline`)
- New `scripts/run_fiber_pipeline`, installed on PATH by pip (setup.cfg
  `[options] scripts`). Stage dispatcher: `run_fiber_pipeline [all | <stage> ...]`
  where a stage is a tool name (`fiber-refine`, `fiber-cpos`, ...). Requested
  stages always run in canonical order; unknown names are rejected before any work.
- Config is env-overridable (the installed script is not hand-edited):
  `FK_DIR` (default `$PWD`), `FK_SESS` (default basename), `FK_ELEC` (required),
  plus per-knob `FK_*` for the algorithm/warp thresholds.
- Encodes the verified call conventions (realign in-place; cpos after refine on the
  refine clu, from the raw spk) and wires the optional warp knobs (0083-0086),
  passing them only when set so it still runs on a tree without those patches.

## [0.25.0] — residual split integrated into fiber-refine (opt-out)
- `refine()` runs a final residual-split cleanup by default (`fiber_split`): each
  output fiber is split on the residual to its shared d(r) when that lowers
  held-out residual energy beyond a random split of the same cluster. Disable
  with `--no-residual-split`. Applies in single-pass and per-chunk (chunked) mode.
- Params `residual_split=True`, `residual_margin`; an "rsplit" row is logged.
- On the real g5 chunk it carved envelope-similar siblings out of coarse fibers
  (clu-free) spanning the discriminator types: a sub-sample timing shift
  (diff∥∂/∂t, cos 0.93), a distinct waveform shape (cos 0.26), and amplitude.
- NOTE: acceptance is residual-energy-vs-null only; a refractory-improvement /
  time-stability tiebreaker (esp. for pure-timing-shift splits, the most
  alignment-sensitive) is the recommended next gate.

## [0.24.0] — recursive residual splitting (fiber_split)
- `fiber_split`: refine a fiber by splitting on the residual to its single shared
  d(r) (where an envelope-similar second unit hides). A candidate binary split in
  the residual subspace is ACCEPTED only if it lowers OUT-OF-SAMPLE residual
  energy more than a random split of the same node (per-node null), so it never
  rewards the trivial decrease from adding clusters. `recursive_split`,
  `accept_split`, `total_residual_energy`, `shared_fiber_residual`.
- Validated clu-free on the real g5 chunk: refining a coarse CEM baseline cut
  held-out total residual energy ~32%, while a matched random refinement to the
  same cluster count did not (-2%) -> +33% over random at equal K. The splitter
  refines an existing fiber; it does not bootstrap a full sort from one root.

## [0.23.0] — fiber-realign reads the session YAML
- `fiber-realign` now takes `<session> <group>` (like `fiber-refine`) and reads
  group channels / nchan / nsamp from `<session>.yaml`, with `--channels`,
  `--ntotal`, `--nchan`, `--nsamp`, `--sr` overrides. `--nsamp`/`--nchan` are no
  longer required. Backward compatible: a bare file `base` still works (no-YAML
  fallback), and the old `--nch` is kept as an alias of `--nchan`.
- `--clu` unchanged; pass the refined labels, e.g. `<base>.clu.refine.<group>`.

## [0.22.1] — fix: chunked refine duplicate `min_group`
- `refine_chunked` passed `min_group` both explicitly and inside `refine_kw`,
  raising `TypeError: refine() got multiple values for keyword argument
  'min_group'` on any `--chunk-minutes` run. Pop it from `refine_kw` (the
  explicit arg is authoritative). Single-pass mode was unaffected.

## [0.22.0] — drift-predicted, signature-gated continuity linking
- `fiber-refine --chunk-minutes M --link-continuity`: after the overlap-anchor
  backbone, recover fibers too sparse to share enough overlap spikes by bridging
  global tracks across chunk gaps. Coherent drift Δz(t) is estimated from the
  well-linked multi-chunk globals; a track that *ends* is bridged to one that
  *begins* only if the drift-predicted depth matches AND the template signatures
  agree (cosine >= `--continuity-sig-thr`, default 0.6), within
  `--continuity-depth-gate` per chunk and `--continuity-max-gap` chunks. Bridges
  may refuse, preserving genuine discontinuities.
- API: `fiber_session.link_continuity`; `fiber_refine._chunk_fiber_features`
  (energy-weighted channel centroid + template signature). Opt-in; the overlap
  backbone is unchanged when the flag is off.
- The signature gate is what blocks identity swaps when a different unit appears
  on a vanished unit's drift path (validated on synthetic coherent-drift data:
  sig-gated recovers ground truth with 0 id-mixing; the ablation without the
  gate wrongly merges the swap).

## [0.21.0] — fiber-view: most-interesting projection tour
- `fiber-view-tour <base>.bundles.<group>.npz`: guided projection-pursuit tour
  over selected bundles — pools them into one shared PCA space, scores 3-D
  projections by between-bundle separation + drift (`tr(PᵀCP)`), and animates a
  smooth path through the highest-scoring frames (camera spinning). Writes `.gif`
  (Pillow) or `.mp4` (ffmpeg). GUI gains multi-select + a "tour video" button.
- API: `interesting_tour`, `render_tour`, `_select_bundles`.

## [0.20.0] — projection-mix sliders
- The bundle GUI exposes the projection as an editable `ncomp×3` mixing matrix
  (top-K PC scores → the 3 display axes): a slider grid (identity = PC1/2/3,
  each PC row labelled with its variance %) to rotate higher-PC contributions
  into view. PCA is fit once per bundle; slider moves only re-multiply scores.
- API: `projection_basis`, `default_mix`, `apply_mix`; `bundle_figure(..., mix=)`.

## [0.19.0] — fiber-view: figures + rotatable bundle GUI
- `fiber-view` (matplotlib): per-channel interpolated waveform-template montage,
  local-fiber PCA(3) manifold, ISI/`fiber_shape_stats` panels.
- `fiber-view-gui` (PySide6 + pyqtgraph): a **selectable bundle table** (one row
  per global fiber, sorted by drift score) that renders the chosen bundle in a
  rotatable 3-D view — per-chunk trajectories + transparent lofted drift sheet.
- Bundle data layer (`make_bundle`, `bundle_table`, `bundle_drift_score`,
  `bundle_figure`, `load_bundles_npz`); HDR spike-density isosurfaces helper.
- `fiber-refine --chunk-minutes M --bundles` writes `<base>.bundles.<group>.npz`
  (per-chunk un-whitened template curves, comparable across chunks).
- New `[viz]` extra (matplotlib + pyqtgraph + PySide6).

## [0.18.0] — geometry output as lossless npz
- `<base>.geom.<group>.npz` / `<base>.geomchunk.<group>.npz` replace the lossy
  `%.4g` TSV; long-format, no object arrays. Added `load_geometry`.

## [0.17.0] — drift-aware chunked mode
- `fiber-refine --chunk-minutes M`: window the session (disjoint cores + overlap
  exts), fit a separate whitener and run the full refine loop per quasi-stationary
  window, then link per-window fibers by overlap-anchor (`fs.link_chunks`); final
  label by core window. `--track-geometry` writes the per-window (drift) geometry.

## [0.16.0] — fit/reassign re-seed loop + geometry tracking
- `--reseed N`: each extra pass is split → merge_back → **refit fibers →
  reassign** (`run_from_seeds`), converging instead of drifting.
- `fiber_tracer.fiber_shape_stats`: per-fiber radius/cone/bend/smoothness/bimodality.
- `--track-geometry`: link each final fiber back through iteration snapshots by
  spike overlap and record its geometry time series.

## [0.15.0] — shape-distinctness split gate + merge-back on
- `--split-min-corr`: don't carve off a split piece / peeled bucket whose
  normalised median waveform matches its parent (stops energy-level
  over-fragmentation). `merge_back` on by default; `--reseed` (label re-feed).

## [0.14.0] — fiber-refine convergence + contamination-gated merge-back
- Early-stop once nfib/swBand/enCV hold steady; merge-back gated on the imposed
  refractory band.

## [0.13.0] — fiber-refine
- New tool: dedup at the imposed refractory, then an iterative gated
  split/peel/isolate cascade (rkk/dip/knn-peel), each split gated on
  per-channel residual-variance reduction without worsening <1 ms refractory.

## [0.12.0] — iterated circular-xcorr alignment
- `fiber_lib.align_xcorr`: align each spike to the cluster median by channel-summed
  xcorr with sub-sample Fourier phase shift, making the residual-variance measure
  meaningful.

## [0.11.0] — variance-driven auto-split
- Recursive KMeans bisection on the channel-weighted trajectory residual, with
  per-channel residual-variance reduction as the stop criterion.

## [0.10.0] — per-channel residual-variance membership
- `channel_residual_profile` / `split_meanvar`: tighten membership and gauge
  contamination by per-channel variance of the residual to the energy-local
  template; per-channel outlier cone + variance-margin split acceptance.

## [0.9.0] — fiber-position
- `fiber-position`: drift-independent per-spike normalized position along the
  consolidated fiber manifold (direction-based).

## [0.8.0] — fiber-drift
- `fiber-drift`: probe drift tracking from fiber depth trajectories.
  *(Base commit `d981297` for the current patch series.)*

## [0.7.0] — fiber-localize
- Monopole + dipole physical localization from raw-waveform spatial spread.

## [0.6.0] — performance
- Vectorized realign/predict/template alignment; optional CuPy GPU backend for
  the realign/whiten kernels; chunk-level CPU parallelism (`--jobs`); batched
  EWMA and collision decomposition; optimization verification/benchmark harness.

## [0.5.0] — standardized I/O
- `neuro_io`: variant resolution and `.res`/`.clu`/`.fet`/`.spk` readers + writers;
  all fiber-kit I/O routed through it.

## [0.4.0] — fiber-realign
- Per-spike fiber-template offsets + corrected `.res`.

## [0.3.0] — fiber-relink
- Post-hoc geometry-aware re-bundling / re-linking of an existing sort.

## [0.2.1] — whitener memory cap
- Sample the whitener baseline from the memmap; cap memory at O(n_base).

## [0.2.0] — session YAML
- `SESSION.yaml` autoloading; CLIs take `<session> <group>`.

## [0.1.0] — initial
- Coarse mean-shift fibers, cross-chunk overlap-anchor linking, per-fiber
  geometry + quality / firing / drift statistics; `fiber-session` pipeline.
