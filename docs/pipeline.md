# fiber-kit — the pipeline

fiber-kit turns a raw extracellular recording (already spike-detected into per-group `.spk`/`.res`)
into a drift-stable spike sort by **deliberately over-splitting each recording chunk and then
re-uniting each neuron's fragments across chunks**.  Every stage is a `fiber-*` command that reads a
`.clu` and writes a new `.clu`; nothing is destructive.  See [stages.md](stages.md) for every stage's
parameters and [config.md](config.md) for the `FK_*` knobs.

## The unit of work

A recording group (e.g. an octrode) is cut into overlapping **chunks** of a few minutes.  Within a
chunk a neuron is drift-stable, so clustering is easy but a neuron may be *over-split* into several
fragments.  Across chunks the same neuron drifts, so its waveform changes shape and amplitude — the
linking stages reunite those fragments into one identity.

Files follow the dotted **variant** convention `<session>.<type>.<variant>.<group>[.<stage>]`:

- `<type>` — `spk` (waveforms, int16), `res` (spike times, int64), `clu`/`clc`/`clp` (labels /
  micro-fiber hierarchy), `fet`, `cpos` (per-cluster positions), …
- `<variant>` — the feature space: `standard` (raw waveforms, the curation/localization axis) or
  `stderiv` (the derivative-whitened space the sort is built in).
- `<stage>` — the pipeline stage tag (`fiber_session`, `refine`, `exp_chunk`, `backbone_linked`, …).

So `sirotaA-….clu.stderiv.5.fiber_session` is group 5's over-split labels in stderiv space.  A stage
selects its input with `--clu-method <variant> --clu-stage <stage>` (or `--in-clu <path>`) and writes
`--out-tag <stage>`.  Positions come from a `.cpos` table selected with `--cpos-stage` (defaults to
`--clu-stage`).

## Canonical order

| # | stage | does | in → out (tag) |
|---|---|---|---|
| 1 | `fiber-session` | over-cluster each chunk into clean fragments (deliberately split), then commit its own realign | → `fiber_session` (or the bare `.clu`) |
| 2 | `fiber-realign` | Klusters-style per-spike realignment (in place); skipped here when fiber-session committed its own | in-place |
| 3 | `fiber-refine` | per-chunk split + fold + merge-back | `fiber_session` → `refine` |
| 4 | `fiber-peel` | footprint + refractory consolidation, confidence-ordered | `refine` (in place) |
| 5 | `fiber-cpos` | per-cluster positions/templates from the standard `.spk` | `refine` → `refine` (+`.cpos`) |
| 6 | `fiber-intrachunk` | within-chunk precision merge + emit `.units.npz` | `refine` → `exp_chunk` |
| 7 | `fiber-link` | cross-chunk link across drift (union-find bundles) | `exp_chunk` (+units) → `exp_linked` |

`fiber-realign` and `fiber-cpos` are run again after `fiber-intrachunk` in the full plans (positions
must follow the merged labels).

### fiber-session's end-of-run realign

At the end of its run fiber-session aligns every spike to its fiber's template and commits the result
(`--realign`, `FK_SESSION_REALIGN`, default `auto`): the shifted spike times go to the shared `.res`
(and the sub-sample offsets to `<base>.offsets.<group>.npy`), and every waveform variant with a `.spk`
on disk -- `standard`, `stderiv*`, any other method -- is rebuilt at the shifted times, with a fresh
`.fet` from its `.pca` (`_D` alias `.fet` included; `--emit-fet` writes its own from the committed
set).  It is fiber-realign's own code with the flags of the plan step it replaces,
`fiber-realign --reextract --refeaturize --variants all` on the `.clu` the run just wrote, and gives
byte-identical files (test/test_session_realign_e2e.py).  `reextract` re-reads the `.fil`;
`shift-spk` rolls the existing `.spk`; `auto` re-extracts when the `.fil` exists and rolls otherwise.
A variant no `.fil` transform derives (fiber-adlp's `standard_ADLP`, say) is rolled by the offsets in
either mode rather than left stale.  `--realign-algo` (`FK_REALIGN_ALGO`) and the other
`FK_REALIGN_*` knobs tune the alignment as they do for fiber-realign.  So the shipped plans no longer
run a fiber-realign straight after fiber-session, and a stage-list run (`fiber-pipeline <elec>`
without a plan) skips its stage-2 fiber-realign; `--realign off` restores the old behaviour.

The commit cuts at the group's `peakSampleIndex` from the YAML.  Without one, fiber-session measures
it rather than guessing: where the windows on disk hold the spike time, located in the `.fil` (their
own geometry, which the commit keeps), or without a `.fil` the median template peak of its 30 largest
units.  It parks the second of two spikes it stacks on one sample in cluster 0 (`--park-duplicates`,
see [methods](methods.md#the-commits-window-051)) in the sort and in its `--out-variant` copy, and
the deconvolution manifest is written afterwards, in the written cluster ids and the committed times.

Cluster and atom ids: fiber-session's fibers and atoms start at 2.  0 and 1 are Klusters' artefact and
noise bins, and every later stage keeps them as such (intrachunk, consensus, the linkers,
xcorr-merge); until 0.51 the first fiber was written as cluster 1 and the first atom as 1, and both were
lost to noise downstream.  The hierarchy save compacts ids, so a fiber's cluster is the `.fibers`'
`clu_id` (and `clu_a` / `clu_b` in the merge-candidate list), never `gid + 1`.

### Repairing a session committed before 0.51

Every `fiber-realign --reextract` commit from 0.38.1 to 0.50 (fiber-session's own, and the plans'
realign steps) cut its windows one sample after the alignment target.  Run `fiber-realign <elec>
--check --clu <stage clu>` per group first; with the `.fil` beside the session its `windows` line
says where the windows on disk hold the spike time:

* `the spike time at index 20, the group's geometry` (42/21): nothing to repair;
* `index 21, NOT the group's 20`: an old commit cut them.  `fiber-realign <elec> --recut --refeaturize
  --variants all` re-cuts every variant at the current `.res` and leaves the times and the labels
  alone.  The times are on the peaks after one old commit; after two or more they are one sample
  late, and `--check` then shows the templates one sample before the target and a realign moving them
  back -- run one.  A realign (or a re-sort) with the `.fil` does both in one pass instead: it aligns
  in the windows' own geometry and re-cuts at the group's;
* a session sorted under 0.50 is best re-sorted (fibers from 2, a manifest that matches its `.clu` and
  `.res`).

Without the `.fil` there is no `windows` line.  Templates one sample after the target mean one old
commit (`--recut` once the `.fil` is back); after two or more the windows look right -- the templates
on the target -- and only the `.fil` shows that they were cut one sample late.  A `--shift-spk` realign
keeps whatever geometry the windows have.

### Where `fiber-consensus` fits: clean before merging

Consensus cleans out contaminants, so it runs on the chunk-local fragments **before any merge**: every
later merge decision (intrachunk's shape / warp gates, the linkers' templates and bands) is made on
templates and spreads the contaminants would distort.  In that position it runs with
`--quarantine-only 1`: it still builds its consensus cells internally -- they define what "perimeter"
and "collision relative to the cell" mean -- but it does **not** merge: kept spikes keep their
fiber-session fragment labels, quarantined spikes go to noise (cluster 1), and the `.clc/.clp`
hierarchy is carried (fiber-intrachunk and the linkers consume the triple).  The island / rim
clusters are written to a separate `<out>_qreview` stage for review in Klusters, and the per-spike
record to `<sess>.qrn.<elec>.<out>`.

The embedded plan of `fiber-kit-exp.yaml`, in order:

| # | stage | in -> out | what |
|---|---|---|---|
| 1 | fiber-session | -> fiber_session | per-chunk over-clustering (12-min chunks, 4-min overlap); then its own realign: shifted .res, every variant's .spk / .fet |
| 2 | **fiber-consensus** `quarantine-only: 1` | fiber_session -> fs_clean (+ fs_clean_qreview, .qrn) | quarantine contaminants to noise, hierarchy carried |
| 3 | fiber-cpos | fs_clean | positions / templates from .spk.standard |
| 4 | fiber-intrachunk | fs_clean -> exp_chunk | within-chunk merge |
| 5 | fiber-realign | exp_chunk (in place) | |
| 6 | fiber-cpos | exp_chunk | |
| 7 | fiber-anchor-link (`frag-layer: units`) | exp_chunk -> exp_linked | cross-chunk link |
| 8 | fiber-xcorr-merge | exp_linked -> exp_xcm | roll-shift cosine merge |
| 9 | fiber-backbone-link (`drift: exp_xcm`) | exp_xcm -> exp_final | backbone link with the drift transform |
| 10 | fiber-realign | exp_final (in place) | final re-extraction for curation |
| (opt) | fiber-lowmix | exp_final -> exp_lowmix (+ .lmx) | split low-amplitude mixture units; copies returned on a refractory allow, novel groups to noise (docs/methods.md "Low-amplitude mixtures") |
| (QC) | fiber-rdr | the final stage's clu -> `.rdr.<method>.<group>.<in>.tsv` | per-cluster contamination score from the residual's temporal derivative; flags only, no `.clu` (in the 0.54.0 exp plan as step `rdr`, after realign-final) |

Run it with `FK_CONFIG=fiber-kit-exp.yaml fiber-pipeline <elec>`.  The yaml pins the method token
(`FK_REALIGN_VARIANT: stderiv_C5_D34`, the reference session's lag feature space -- inference never
picks a `_D` alias) and the consensus `.res` (`FK_CONSENSUS_RES_VARIANT: stderiv_S3`; the other stages
resolve the one shared `.res`, so that must be the only copy).  Needs `pip install fiber-kit[consensus]`.

Inside fiber-consensus, per chunk (`--chunk-min`, default the session's 12 min):

1. envelope per input unit (collision-robust, spike-free noise floor) -> envelope d' -> neighbourhoods
   by average linkage at d' < 12 (no size cap by default; neighbourhoods under 300 spikes skipped);
2. per neighbourhood: 40 t-SNE runs of a random 70 % of its z-scored `.fet` rows (parallel workers)
   -> 15 nearest 2-D neighbours per run -> co-neighbour graph (P >= 0.1) -> Louvain communities;
3. envelope per community -> merge into cells at envelope d' < 7 with two cannot-links: merge
   inflation > 0.1, and refractory (trains coincide within (0.6, 1.5] ms at chance, >= 3 expected);
4. envelope refitted on the cells, with partners: collision flag, bump score, residual energy;
   within-cell robust z; perimeter rank over the runs; switch rate;
5. quarantine = collision OR (perimeter rank >= 0.8 AND residual or bump z >= 2) (switch opt-in);
6. watershed categories of the quarantine on 5 full maps: island / own-rim / other-cell;
7. output: quarantine-only -> input labels, quarantine to noise, review stage; merge mode -> each cell
   keeps the input id holding most of its kept spikes, island and rim clusters per neighbourhood.

**Merge mode** (`quarantine-only: 0`, the stage default) stays available for a late review pass on
`.exp_final` (`fiber-pipeline <elec> fiber-consensus`): its cells become the units and both quarantine
clusters are kept in the `.clu`.

`fiber-envelope --merge 7` (envelope d' merge with the inflation and refractory cannot-links) is a
standalone alternative to intrachunk's merge; it is not in the plan.

## Alternative linkers

Instead of `intrachunk → link`, you can link the over-split `fiber_session` output directly:

- **`fiber-backbone-link`** — links fragments across chunks on the *invariant backbone* channels by
  the energy-scaled median±σ **band overlap** (`fiber_geometry.band_overlap`), mutual-NN across
  adjacent chunks, gated by the Omlor–Giese warp veto, **starting from the high-SNR clusters**.  It
  needs only the `fiber_session` `.clu` + the standard `.spk` (no refine / intrachunk / cpos).
- **`fiber-xcorr-merge`** — a confidence-ordered agglomeration that fuses near-identical clusters by
  the Klusters roll-shift cosine (max cosine over circular time shifts), **re-aligning after each
  merge**, gated by the refractory cross-correlogram and (by default) the band-overlap co-gate.  Run
  it after `fiber-backbone-link`.

A ready multi-pass plan chains them: `backbone-link → xcorr-merge → backbone-link → xcorr-merge`
(`plans/09-backbone-xcorr.yaml`).

## Matching methods (what "band matching" means)

The **default shape criterion** across the merge/link stages is the energy-scaled median±σ **band
overlap**: each cluster is summarised by a per-sample `[median ± zσ]` band on the compared channels;
two clusters match when the per-sample interval-IoU of their bands is high.  The band is normalised to
unit energy over the compared window because spike-to-spike variance scales with waveform energy, so a
big waveform does not get an unfairly wide band.  See [methods.md](methods.md).

- `fiber-backbone-link` uses it as its primary criterion, on the **invariant channels** (where it is
  strongest).
- `fiber-intrachunk` uses it as the shape gate (`--gate band`, the default) on the full footprint,
  alongside the inter-channel offset and depth gates.
- `fiber-xcorr-merge` uses it as a co-gate (`--band-thr`, default 0.5) on top of the roll-shift cosine.

Set `--gate cfiber` / `--band-thr 0` (or `FK_INTRA_GATE` / `FK_XCM_BAND_THR`) to fall back to the older
cosine/descriptor criteria.

## Running it

`fiber-pipeline` (a shell wrapper installed on PATH; also runnable as `scripts/fiber-pipeline`) drives
the stages.  `<elec>` is the group; `$FK_DIR`/`$FK_SESS` locate the session (default `$PWD` / basename).

```
fiber-pipeline 5                       # whole pipeline (embedded plan if $FK_CONFIG has one, else canonical stages)
fiber-pipeline 5 all                   # same as bare
fiber-pipeline 5 fiber-session         # ONE stage (overrides the embedded plan)
fiber-pipeline 5 fiber-intrachunk --gate cfiber   # one stage with argument overrides
fiber-pipeline 5 --plan plans/09-backbone-xcorr.yaml   # a custom, reorderable, repeatable step plan
fiber-pipeline 5 --dry-run …           # print the stage commands instead of running them
fiber-pipeline -l                      # list stages
fk list                                # list the plan's steps by their unique names
fk run 5                               # the whole plan (same as fiber-pipeline 5)
fiber-pipeline 5 fiber-realign-2       # run ONE plan step (its plan in/out/params)
fiber-pipeline 5 --from fiber-cpos-2   # the plan from that step on (--to <step> ends a span)
```

**Dispatch precedence:** `--plan <file>` wins; otherwise a `pipeline:` section embedded in
`$FK_CONFIG` runs **only when no stage is named** (bare or `all`); naming stage(s) runs exactly those.

A **plan** is a yaml list of steps, each `{stage, in, out, units?, spk?, params?}`; a stage may repeat
with different tags.  `params:` map to that stage's CLI flags (`key: val` → `--key val`,
`key: true` → bare `--flag`).  Edit plans by hand, from `plans/TEMPLATE-all-stages.yaml` (a menu of
every stage and its parameters), or with the GUI editor `fiber-plan-edit`; lint them with
`fiber-plan-lint`.

### Step names

A plan may run a stage several times (`fiber-kit-exp.yaml` runs `fiber-realign` and `fiber-cpos` twice
each), so every step has a unique id, `<stage>-<k>` -- its k-th run of that stage in plan order:
`fiber-realign-1`, `fiber-realign-2`.  A step may also carry its own `name:` (letters, digits, `.`, `_`,
`-`; unique; not another step's id), a stable alias that survives reordering, which the positional id
does not.  `fk list` prints them:

```
$ fk list                       # the plan fiber-pipeline would run ($FK_CONFIG's, or --plan <file>)
#   step                   name           in             out            flags
1   fiber-session-1        session        -              fiber_session
2   fiber-consensus-1      clean          fiber_session  fs_clean       --quarantine-only 1
3   fiber-cpos-1           cpos-clean     fs_clean       fs_clean       spk=standard
4   fiber-intrachunk-1     chunk          fs_clean       exp_chunk
5   fiber-realign-1        realign-chunk  exp_chunk      -
...
10  fiber-realign-2        realign-final  exp_final      -
```

A step is selected by its id, its name, its 1-based position, or -- when the stage runs once -- the bare
stage name; ranges `A..B` (either end open) and comma lists work too:

```
fiber-pipeline 5 realign-chunk                       # positional step names (plan mode)
fiber-pipeline 5 --step fiber-consensus-1 --step 7   # --step, repeatable
fiber-pipeline 5 --from fiber-anchor-link-1          # to the end of the plan
fk run 5 fiber-cpos-1..fiber-realign-2 --dry-run     # the same through fk
fk run 5                                             # the whole plan, as fiber-pipeline 5
fk run 5 --from fiber-cpos-2 --to fiber-realign-2    # a span
```

Selected steps run in plan order with their plan tags and params; the plan's bookkeeping still sees every
step, so a `fiber-link` run alone reads the units its plan's `fiber-intrachunk` emitted.  The steps come
from `--plan`, else from the `pipeline:` section of `$FK_CONFIG`.  A bare stage name that runs more than
once is refused with its ids; on the command line a bare KNOWN stage name (`fiber-pipeline 5
fiber-realign`) keeps its old meaning -- that stage stand-alone with the pipeline defaults, not a plan step.

## Configuration

Point `$FK_CONFIG` at a `fiber-kit.yaml` (or use the shipped `fiber-kit-exp.yaml`).  It holds flat
`FK_*` knobs (see [config.md](config.md)) and optionally an embedded `pipeline:` plan.  Knob precedence
is **CLI/plan-param > `FK_*` env > `$FK_CONFIG` > default**; stages print their resolved knobs at
startup so you can confirm what took effect.
