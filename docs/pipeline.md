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
