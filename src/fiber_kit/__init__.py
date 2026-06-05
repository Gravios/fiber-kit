"""fiber_kit — drift-stable "fiber" reorganization of spike sorts (neurosuite-3).

A fiber is an energy-direction manifold in whitened feature space: one neuron =
one smooth curve d(r) of direction vs energy/radius.  This package clusters
chunk spikes into fibers, links them across a session, and emits per-fiber
geometry + quality / firing / drift statistics for curation.

CLI (installed console scripts) — read <session>.yaml automatically:
    fiber-session <session> <group>          clustering + linking + .fibers/.clu
    fiber-validate-merges <session> <group>   full-session evidence for merges
    fiber-raw-vs-stderiv <session> <group>    raw .fil vs stderiv discrimination
    fiber-relink <base>.fibers...npz [--clu]  geometry-aware re-link of a finished run
    fiber-realign <base> <elec> --nsamp --nch  per-spike template offsets + corrected .res

Python:
    import fiber_kit as fk
    cfg = fk.resolve_session_params("SESSION", 5)   # reads SESSION.yaml
    fine, geoms = fk.cluster_chunk_fine(waves, res, W, nmean, coarse_mg=200,
                                        mask=fk.fiber_lib.MASK_FULL, sr=cfg["sr"],
                                        method="rkk", merge_method="sliding", merge_corr=0.90)

See WORKFLOW.md (shipped with the package) for the end-to-end recipe.
"""
__version__ = "0.11.0"

from . import fiber_lib, fiber_tracer, fiber_adapt, fiber_collision, laplacian_link
from . import neuro_io
from . import backend
from .backend import use_gpu, gpu_enabled, backend_name
from .fiber_tracer import trajectory, predict, predict_many, channel_residual_profile, split_meanvar
from .klustakwik import klustakwik
from .session_yaml import find_session_yaml, load_session, resolve_session_params
from .fiber_session import (
    cluster_chunk, cluster_chunk_fine, fiber_geom, link_chunks,
    read_res, open_spkD, fil_chunk_whitener,
)
from .fiber_relink import relink, rewrite_clu, write_report
from .fiber_realign import template_offsets, realign, write_outputs
from .fiber_localize import load_geometry, localize, localize_unit
from .fiber_drift import drift_curve, decentralized_drift, write_drift_table
from .fiber_position import (
    load_manifolds, spike_positions, position_by_direction, curve_arclength,
)
from .neuro_io import (
    resolve_input, prefer_derived, prefer_canonical,
    read_res_file, write_res_file, read_clu_file, write_clu_file,
    read_cluster_res, read_fet_file, open_spk, open_signal,
)

__all__ = [
    "__version__",
    "fiber_lib", "fiber_tracer", "fiber_adapt", "fiber_collision", "laplacian_link",
    "neuro_io", "backend", "use_gpu", "gpu_enabled", "backend_name",
    "trajectory", "predict", "predict_many", "channel_residual_profile", "split_meanvar", "klustakwik",
    "find_session_yaml", "load_session", "resolve_session_params",
    "cluster_chunk", "cluster_chunk_fine", "fiber_geom", "link_chunks",
    "read_res", "open_spkD", "fil_chunk_whitener",
    "relink", "rewrite_clu", "write_report",
    "template_offsets", "realign", "write_outputs",
    "load_geometry", "localize", "localize_unit",
    "drift_curve", "decentralized_drift", "write_drift_table",
    "load_manifolds", "spike_positions", "position_by_direction", "curve_arclength",
    "resolve_input", "prefer_derived", "prefer_canonical",
    "read_res_file", "write_res_file", "read_clu_file", "write_clu_file",
    "read_cluster_res", "read_fet_file", "open_spk", "open_signal",
]
