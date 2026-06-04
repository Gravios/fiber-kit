"""fiber_kit — drift-stable "fiber" reorganization of spike sorts (neurosuite-3).

A fiber is an energy-direction manifold in whitened feature space: one neuron =
one smooth curve d(r) of direction vs energy/radius.  This package clusters
chunk spikes into fibers, links them across a session, and emits per-fiber
geometry + quality / firing / drift statistics for curation.

CLI (installed console scripts) — read <session>.yaml automatically:
    fiber-session <session> <group>          clustering + linking + .fibers/.clu
    fiber-validate-merges <session> <group>   full-session evidence for merges
    fiber-raw-vs-stderiv <session> <group>    raw .fil vs stderiv discrimination

Python:
    import fiber_kit as fk
    cfg = fk.resolve_session_params("SESSION", 5)   # reads SESSION.yaml
    fine, geoms = fk.cluster_chunk_fine(waves, res, W, nmean, coarse_mg=200,
                                        mask=fk.fiber_lib.MASK_FULL, sr=cfg["sr"],
                                        method="rkk", merge_method="sliding", merge_corr=0.90)

See WORKFLOW.md (shipped with the package) for the end-to-end recipe.
"""
__version__ = "0.2.0"

from . import fiber_lib, fiber_tracer, fiber_adapt, fiber_collision, laplacian_link
from .fiber_tracer import trajectory, predict
from .klustakwik import klustakwik
from .session_yaml import find_session_yaml, load_session, resolve_session_params
from .fiber_session import (
    cluster_chunk, cluster_chunk_fine, fiber_geom, link_chunks,
    read_res, open_spkD, fil_chunk_whitener,
)

__all__ = [
    "__version__",
    "fiber_lib", "fiber_tracer", "fiber_adapt", "fiber_collision", "laplacian_link",
    "trajectory", "predict", "klustakwik",
    "find_session_yaml", "load_session", "resolve_session_params",
    "cluster_chunk", "cluster_chunk_fine", "fiber_geom", "link_chunks",
    "read_res", "open_spkD", "fil_chunk_whitener",
]
