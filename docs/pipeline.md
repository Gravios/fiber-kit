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
| 1 | `fiber-session` | over-cluster each chunk into clean fragments (deliberately split) | → `fiber_session` (or the bare `.clu`) |
| 2 | `fiber-realign` | Klusters-style per-spike realignment (in place) | in-place |
| 3 | `fiber-refine` | per-chunk split + fold + merge-back | `fiber_session` → `refine` |
| 4 | `fiber-peel` | footprint + refractory consolidation, confidence-ordered | `refine` (in place) |
| 5 | `fiber-cpos` | per-cluster positions/templates from the standard `.spk` | `refine` → `refine` (+`.cpos`) |
| 6 | `fiber-intrachunk` | within-chunk precision merge + emit `.units.npz` | `refine` → `exp_chunk` |
| 7 | `fiber-link` | cross-chunk link across drift (union-find bundles) | `exp_chunk` (+units) → `exp_linked` |

`fiber-realign` and `fiber-cpos` are run again after `fiber-intrachunk` in the full plans (positions
must follow the merged labels).

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
| 1 | fiber-session | -> fiber_session | per-chunk over-clustering (12-min chunks, 4-min overlap) |
| 2 | fiber-realign | fiber_session (in place) | per-spike align; re-extracts and re-featurizes the .spk / .fet |
| 3 | **fiber-consensus** `quarantine-only: 1` | fiber_session -> fs_clean (+ fs_clean_qreview, .qrn) | quarantine contaminants to noise, hierarchy carried |
| 4 | fiber-cpos | fs_clean | positions / templates from .spk.standard |
| 5 | fiber-intrachunk | fs_clean -> exp_chunk | within-chunk merge |
| 6 | fiber-realign | exp_chunk (in place) | |
| 7 | fiber-cpos | exp_chunk | |
| 8 | fiber-anchor-link (`frag-layer: units`) | exp_chunk -> exp_linked | cross-chunk link |
| 9 | fiber-xcorr-merge | exp_linked -> exp_xcm | roll-shift cosine merge |
| 10 | fiber-backbone-link (`drift: exp_xcm`) | exp_xcm -> exp_final | backbone link with the drift transform |
| 11 | fiber-realign | exp_final (in place) | final re-extraction for curation |

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
```

**Dispatch precedence:** `--plan <file>` wins; otherwise a `pipeline:` section embedded in
`$FK_CONFIG` runs **only when no stage is named** (bare or `all`); naming stage(s) runs exactly those.

A **plan** is a yaml list of steps, each `{stage, in, out, units?, spk?, params?}`; a stage may repeat
with different tags.  `params:` map to that stage's CLI flags (`key: val` → `--key val`,
`key: true` → bare `--flag`).  Edit plans by hand, from `plans/TEMPLATE-all-stages.yaml` (a menu of
every stage and its parameters), or with the GUI editor `fiber-plan-edit`; lint them with
`fiber-plan-lint`.

## Configuration

Point `$FK_CONFIG` at a `fiber-kit.yaml` (or use the shipped `fiber-kit-exp.yaml`).  It holds flat
`FK_*` knobs (see [config.md](config.md)) and optionally an embedded `pipeline:` plan.  Knob precedence
is **CLI/plan-param > `FK_*` env > `$FK_CONFIG` > default**; stages print their resolved knobs at
startup so you can confirm what took effect.
