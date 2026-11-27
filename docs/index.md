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
- **[morphology.md](morphology.md)** — biophysical **forward modelling** (`fiber-morpho`):
  reconstruction → compartmental simulation → extracellular footprint, the variance decomposition
  (position vs. morphology), back-propagation, and the CA1 afferent topology.  Answers questions the
  sort itself cannot: how much waveform variance is the cell and how much is everything else.
- **[bench.md](bench.md)** — `fiber-bench` and `fiber-tune`: run parameter variants of a plan under a config hash,
  score them (truth and truth-free metrics, pyramidal / interneuron) and compare them in an HTML report.
- **[methods.md](methods.md)** — the core numerical primitives (`fiber_geometry`, `fiber_ccg`): the
  energy-scaled median±σ band overlap, the Omlor–Giese drift-warp veto, waveform complexity, the
  refractory cross-correlogram gate and the burst signature, template deconvolution (`fiber_deconv`),
  with signatures, parameters and validation.

- **[organization-audit.md](organization-audit.md)** — a legibility and organization **audit** of the
  source (2026-11-25): function sizes, the two routes of a knob, the stage lists that disagree, option
  names, the profiles; findings and fixes, nothing changed.

- **[neuropixels-plan.md](neuropixels-plan.md)** — a **design plan** (nothing built yet) for
  high-density probes: tiles with a detection set wider than the sorting set, sizes from the row
  pitch and the cell size, spikes assigned to a tile by their spatial profile, a drift field from
  the signal's depth profile, a transient filtered file and a sparse kept one.

- **[roadmap.md](roadmap.md)** — planned work that is not scheduled yet, each item with its status,
  its design document and when to revisit it.

- **[python-upgrade-plan.md](python-upgrade-plan.md)** — a **design plan** (deferred, 2026-11-27)
  for moving past Python 3.12: what 3.14/3.15 offer fiber-kit and what they don't, the `fork` →
  `forkserver` default that affects three process pools, the stated `>=3.9` floor, and the steps and
  triggers for the move.

- **fiber-view/**, **linking/** — the figures the README and `WORKFLOW.md` show, and the
  synthetic hybrid-linker prototype that made `linking/hybrid_linker.png`.

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
| fuse near-identical clusters after linking | `fiber-xcorr-merge`, or `fiber-residual-merge` (the residual strip in reverse), in [stages.md](stages.md) |
