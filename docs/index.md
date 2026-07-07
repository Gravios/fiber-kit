# fiber-kit documentation

fiber-kit is a Python pipeline for drift-stable extracellular spike sorting: it over-splits each
recording chunk into clean fragments and re-unites each neuron's fragments across chunks.  It pairs
with the Qt/C++ **neurosuite-3** toolchain (klusters, neuroscope, ndmanager) for curation and viewing.

## Contents

- **[pipeline.md](pipeline.md)** — how the pipeline fits together: the chunk/variant model, the
  canonical stage order, the alternative drift linkers, the matching methods ("band matching"), how to
  run stages and plans with `fiber-pipeline`, and configuration.
- **[stages.md](stages.md)** — the complete **stage & parameter reference**: every `fiber-*` command
  with its positionals, flags, defaults and choices (generated from each stage's argument parser).
- **[config.md](config.md)** — every `FK_*` configuration knob, grouped by stage, with defaults and
  meanings (from `fiber-kit-exp.yaml`).
- **[methods.md](methods.md)** — the core numerical primitives (`fiber_geometry`, `fiber_ccg`): the
  energy-scaled median±σ band overlap, the Omlor–Giese drift-warp veto, waveform complexity, and the
  refractory cross-correlogram gate, with signatures and parameters.

The top-level [README](../README.md) has the narrative overview, install, quick start, per-fiber
statistics, curation/merge-validation, visualization (`fiber-view`), and design notes.

## Quick pointers

| I want to… | see |
|---|---|
| run the whole pipeline / a single stage / a plan | [pipeline.md § Running it](pipeline.md#running-it) |
| know a stage's flags and defaults | [stages.md](stages.md) (or `fiber-<stage> -h`) |
| tune a stage without editing code | [config.md](config.md) — set the `FK_*` knob |
| understand band matching / the warp veto | [methods.md](methods.md) |
| link the over-split output directly (skip refine/intrachunk) | `fiber-backbone-link` in [stages.md](stages.md) |
| fuse near-identical clusters after linking | `fiber-xcorr-merge` in [stages.md](stages.md) |
