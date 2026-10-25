"""fiber_chunks -- run a stage on selected chunks only; the other chunks' spikes go to reserved clusters.

FK_CHUNKS (global, $FK_CONFIG or the environment; `--chunks` on each supporting stage) selects chunks:
"" or "all" = every chunk, else a list such as "14-16" or "3,7,10-12".  A chunk is the session's own:
chunk k holds the spikes with  (res - res.min()) // (chunk length)  == k, the chunk length being
FK_SESSION_CHUNK_MIN minutes (12) -- the same chunks fiber-session sorts, numbered from 0 as its log
prints them (1-based) minus one.

RESERVED CLUSTER IDS.  The top 100 ids Klusters can represent exactly are reserved:

    RESERVED_LO = 16,777,116  ..  RESERVED_HI = 16,777,215      (2**24 - 100 .. 2**24 - 1)
    chunk k (not computed)  ->  cluster RESERVED_LO + k          (k = 0..99)

Klusters passes cluster ids through a float in places (klustersdoc_edit.cpp), which is exact only up to
2**24, so a block at the int32 top would be rounded together.  Every id allocator skips the block:
next_free_id() here, and Klusters' highestClusterId() (neurosuite-3 0889), so new clusters are always
numbered below it.  The same constants are in neurosuite-3's libklustersshared/types.h
(test_shared_contract compares them when NS3_ROOT is set).

A stage run with chunks writes every spike of a chunk it did not compute to that chunk's reserved
cluster, so its output still has one label per spike of the shared .res and opens in Klusters, where
the reserved clusters show which part of the recording was left out.  Stages treat reserved clusters as
"not sorted here": never merged, realigned, scored or split.
"""
import os

import numpy as np

RESERVED_N = 100
RESERVED_HI = 2 ** 24 - 1                         # 16,777,215: the highest id a float32 holds exactly
RESERVED_LO = RESERVED_HI - RESERVED_N + 1        # 16,777,116
DEFAULT_CHUNK_MIN = 12.0
SUPPORTED_STAGES = ("fiber-flag-artifacts", "fiber-session", "fiber-xcorr-merge", "fiber-realign", "fiber-rdr")


def reserved_id(k):
    """The reserved cluster of chunk k (0-based)."""
    k = np.asarray(k)
    if np.any((k < 0) | (k >= RESERVED_N)):
        raise ValueError(f"chunk index outside 0..{RESERVED_N - 1}: only {RESERVED_N} chunks have a reserved cluster "
                         "(use a longer FK_SESSION_CHUNK_MIN)")
    return RESERVED_LO + k


def is_reserved(ids):
    ids = np.asarray(ids)
    return (ids >= RESERVED_LO) & (ids <= RESERVED_HI)


def next_free_id(labels, floor=2):
    """max(non-reserved labels, floor - 1) + 1 -- the id a new cluster takes.  Raises if it would reach the
    reserved block."""
    lab = np.asarray(labels)
    lab = lab[~is_reserved(lab)] if lab.size else lab
    nxt = max(int(lab.max()) + 1 if lab.size else floor, floor)
    if nxt >= RESERVED_LO:
        raise ValueError(f"cluster id {nxt} would enter the reserved block {RESERVED_LO}..{RESERVED_HI}")
    return nxt


def parse(spec):
    """'' / 'all' / None -> None (every chunk); '14-16' / '3,7,10-12' -> sorted list of chunk indices."""
    if spec is None:
        return None
    s = str(spec).strip()
    if s.lower() in ("", "all", "none"):
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise ValueError(f"chunk range {part!r} runs backwards")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if any(k < 0 for k in out):
        raise ValueError(f"negative chunk index in {spec!r}")
    return sorted(out)


def chunk_index(res, sr, chunk_min=DEFAULT_CHUNK_MIN, origin=None):
    """0-based chunk of every spike: (res - origin) // chunk length; origin defaults to res.min(), the
    origin fiber-session uses."""
    res = np.asarray(res, np.int64)
    if res.size == 0:
        return np.zeros(0, np.int64)
    o = int(res.min()) if origin is None else int(origin)
    L = float(chunk_min * 60.0 * sr)
    n = max(1, int(np.ceil((int(res.max()) - o) / L)))     # fiber-session's chunk count
    return np.minimum(((res - o) // L).astype(np.int64), n - 1)


def active_mask(res, sr, chunks, chunk_min=DEFAULT_CHUNK_MIN):
    """True for the spikes of the selected chunks (all True when chunks is None)."""
    if chunks is None:
        return np.ones(len(res), bool)
    return np.isin(chunk_index(res, sr, chunk_min), np.asarray(chunks, np.int64))


def mark_inactive(labels, res, sr, chunks, chunk_min=DEFAULT_CHUNK_MIN):
    """Copy of `labels` with every spike outside the selected chunks set to its chunk's reserved id."""
    out = np.asarray(labels).astype(np.int64, copy=True)
    if chunks is None:
        return out
    ci = chunk_index(res, sr, chunk_min)
    off = ~np.isin(ci, np.asarray(chunks, np.int64))
    if off.any():
        out[off] = reserved_id(ci[off])
    return out


def config_value(gcfg, name, default=""):
    """$FK_CONFIG value, then the environment, then the default (the precedence of every FK_* knob)."""
    v = (gcfg or {}).get(name) if gcfg is not None else None
    if v in (None, ""):
        v = os.environ.get(name, "")
    return default if v in (None, "") else v


def add_argument(ap, gcfg=None, length=True):
    """--chunks (default FK_CHUNKS) and, unless the stage has its own, --chunk-length (default
    FK_SESSION_CHUNK_MIN) on a stage's parser."""
    ap.add_argument("--chunks", default=str(config_value(gcfg, "FK_CHUNKS", "")),
                    help="FK_CHUNKS: process only these 0-based chunks, e.g. 14-16 or 3,7 ('' = all); the other "
                         f"chunks' spikes go to the reserved clusters {RESERVED_LO}+chunk")
    if not length:
        return
    ap.add_argument("--chunk-length", dest="chunk_length", type=float,
                    default=float(config_value(gcfg, "FK_SESSION_CHUNK_MIN", DEFAULT_CHUNK_MIN)),
                    help="FK_SESSION_CHUNK_MIN: chunk length in minutes that --chunks counts in")


def describe(chunks, chunk_min):
    if chunks is None:
        return "all chunks"
    return (f"chunks {fmt(chunks)} of {chunk_min:g} min (others -> reserved clusters "
            f"{RESERVED_LO}+chunk)")


def fmt(chunks):
    """[3,4,5,9] -> '3-5,9'."""
    if not chunks:
        return ""
    out, run = [], [chunks[0]]
    for k in chunks[1:]:
        if k == run[-1] + 1:
            run.append(k)
        else:
            out.append(run); run = [k]
    out.append(run)
    return ",".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in out)
