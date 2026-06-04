"""fiber_kit — drift-stable "fiber" reorganization of spike sorts (neurosuite-3).

A fiber is an energy-direction manifold in whitened feature space: one neuron =
one smooth curve d(r) of direction vs energy/radius.  This package clusters
chunk spikes into fibers, links them across a session, and emits per-fiber
geometry + quality / firing / drift statistics for curation.

Quick start (Python):
    import fiber_kit as fk
    fine, geoms = fk.cluster_chunk_fine(waves, res, W, nmean, coarse_mg=200,
                                        mask=fk.fiber_lib.MASK_FULL, sr=32552,
                                        method="rkk", merge_method="sliding", merge_corr=0.90)

CLI (installed console scripts):
    fiber-session        full-session clustering + linking + .fibers/.clu output
    fiber-validate-merges   full-session evidence for proposed same-neuron merges
    fiber-raw-vs-stderiv    raw .fil vs stderiv discrimination test

See WORKFLOW.md (shipped with the package) for the end-to-end recipe.
"""
__version__ = "0.1.0"

from . import fiber_lib, fiber_tracer, fiber_adapt, fiber_collision, laplacian_link
from .fiber_tracer import trajectory, predict
from .klustakwik import klustakwik
from .fiber_session import (
    cluster_chunk, cluster_chunk_fine, fiber_geom, link_chunks,
    read_res, open_spkD, fil_chunk_whitener,
)

__all__ = [
    "__version__",
    "fiber_lib", "fiber_tracer", "fiber_adapt", "fiber_collision", "laplacian_link",
    "trajectory", "predict", "klustakwik",
    "cluster_chunk", "cluster_chunk_fine", "fiber_geom", "link_chunks",
    "read_res", "open_spkD", "fil_chunk_whitener",
]
