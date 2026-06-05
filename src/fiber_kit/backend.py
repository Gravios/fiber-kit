# ════════════════════════════════════════════════════════════════════════════
#  backend.py — optional GPU (CuPy) acceleration for fiber-kit's hot kernels.
#
#  fiber-kit is numpy-native.  This module provides a thin array-module shim so
#  the flop-heavy, array-API-clean kernels (per-spike realignment, the whitening
#  transform) can run on an NVIDIA GPU via CuPy when available and enabled, while
#  the default path stays pure numpy and bit-identical.
#
#  Design:
#    - Disabled by default.  Enable with the env var FIBER_KIT_GPU=1 (or
#      use_gpu(True)).  If CuPy / a CUDA device is unavailable, it silently
#      stays on numpy — never an error, never a hard dependency.
#    - Kernels call xp() to get the active array module, move inputs onto the
#      device with asarray(), compute, and return a numpy array via asnumpy().
#      So callers are unchanged and GPU is purely internal acceleration; the
#      numpy code path (xp() is np) is exactly the former behaviour.
#    - GPU results are float; correctness on the GPU path needs verification on
#      real hardware (no CUDA in CI/sandbox).  The numpy path is the validated
#      one and remains the default.
#
#  This mirrors the neurosuite-3 GPU-backend philosophy (kiloklustakwik): a
#  plain on/off switch with runtime auto-detection and a clean CPU fallback.
# ════════════════════════════════════════════════════════════════════════════
import os

import numpy as np

_state = {"want": None, "cp": None, "checked": False}


def _detect():
    """Lazily import CuPy and confirm a usable device.  Caches the result."""
    if _state["checked"]:
        return _state["cp"] is not None
    _state["checked"] = True
    if _state["want"] is None:
        _state["want"] = os.environ.get("FIBER_KIT_GPU", "").lower() in (
            "1", "true", "yes", "on")
    if not _state["want"]:
        _state["cp"] = None
        return False
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()      # raises if no device
        _state["cp"] = cp
        return True
    except Exception:
        _state["cp"] = None                   # no CuPy / no device -> numpy
        return False


def use_gpu(flag=True):
    """Force GPU on/off (overrides FIBER_KIT_GPU).  Returns the resulting state
    (False if a GPU was requested but CuPy/device is unavailable)."""
    _state["want"] = bool(flag)
    _state["checked"] = False
    _state["cp"] = None
    return _detect()


def gpu_enabled():
    """True iff GPU acceleration is active for this process."""
    return _detect()


def backend_name():
    return "cupy" if gpu_enabled() else "numpy"


def xp():
    """The active array module — CuPy if enabled & available, else numpy."""
    return _state["cp"] if gpu_enabled() else np


def asarray(a, dtype=None):
    """Move `a` onto the active device (no-op on the numpy path)."""
    return xp().asarray(a, dtype=dtype) if dtype is not None else xp().asarray(a)


def asnumpy(a):
    """Bring an array back to host numpy (no-op on the numpy path)."""
    if gpu_enabled():
        return _state["cp"].asnumpy(a)
    return np.asarray(a)
