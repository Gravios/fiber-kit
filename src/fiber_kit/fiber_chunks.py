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
"not sorted here": never merged, realigned, scored or split.  Stages without native support that only
relabel spikes run under fiber-pipeline's CHUNK SHIELD (shield / unshield below; `python -m
fiber_kit.fiber_chunks shield|unshield`).
"""

import numpy as np

try:
    from . import config as cfgmod
except ImportError:                      # run as a script from src/fiber_kit
    import config as cfgmod

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


def chunk_hi(c, t_min, chunk_len, nchunks):
    """The exclusive end of chunk c's core window.  The last chunk has none: nchunks = ceil(span / length),
    so when the span is an exact multiple of the length the last spike sits ON t_min + nchunks * length,
    and a `res < hi` test left it out of every chunk (chunk_index clamps it into the last chunk the same
    way)."""
    return t_min + (c + 1) * chunk_len if c < nchunks - 1 else np.inf


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
    """The environment, then the $FK_CONFIG values `gcfg`, then the default -- the precedence of every
    FK_* knob (config.knob_value)."""
    return cfgmod.knob_value(name, str, default, gcfg if gcfg is not None else {})


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


# ── the chunk shield: chunk runs for stages that do not know about them ─────────
# A stage that only RELABELS spikes (reads a .clu at one tag, writes a .clu at another; never changes the
# spike set or the .res) can run on a chunk run without knowing about it: fiber-pipeline shields it.
#   shield    every spike of a chunk not computed is set to 0 in the input .clu (and its atom to 0 in the
#             .clc) -- the artifact bin, which every stage leaves out of its units -- with the input saved
#             beside it (<file>.chunkshield) and the reserved id of every such spike in a state file;
#   the stage runs on the shielded input;
#   unshield  in the .clu the stage wrote, every shielded spike goes back to its chunk's reserved cluster
#             (in the .clc/.clp as one atom per chunk under the reserved id), and the input is restored.
# A run that died between the two leaves the state file; the next shield restores the input first.
SHIELD_STAGES = ("fiber-cpos", "fiber-intrachunk", "fiber-link", "fiber-anchor-link", "fiber-backbone-link",
                 "fiber-consensus", "fiber-lowmix", "fiber-peel", "fiber-refit")


def _triple(base, elec, variant, tag):
    from . import neuro_io as nio
    return {k: nio.session_path(base, k, elec, variant=variant, tag=tag) for k in ("clu", "clc", "clp")}


def _stat(path):
    import os
    try:
        st = os.stat(path)
        return [st.st_mtime_ns, st.st_size]
    except OSError:
        return None


def _restore_inputs(st):
    import os
    for path in st["backups"]:
        if os.path.exists(path + ".chunkshield"):
            os.replace(path + ".chunkshield", path)


def shield(session, elec, variant, in_tag, out_tag, state_path, chunks, chunk_min, log=print):
    """Hide the spikes of the chunks not computed from the input of a stage (see above)."""
    import json
    import os
    from . import neuro_io as nio
    from . import session_yaml as sy
    if os.path.exists(state_path):                         # an earlier run died between shield and unshield
        with open(state_path) as fh:
            _restore_inputs(json.load(fh))
        os.remove(state_path)
        log(f"chunk shield: restored the input an interrupted run left shielded ({os.path.basename(state_path)})")
    cfg = sy.resolve_session_params(session, int(elec), require=(), verbose=False)
    base = cfg["base"]
    res = nio.read_res(base, elec)
    tin = _triple(base, elec, variant, in_tag)
    _, lab = nio.read_clu_file(tin["clu"], n_spikes=res.size)
    lab = np.asarray(lab, np.int64)
    ci = chunk_index(res, cfg["sr"], chunk_min)
    off = is_reserved(lab) | ~np.isin(ci, np.asarray(chunks, np.int64))
    rid = np.where(off, np.where(is_reserved(lab), lab, RESERVED_LO + np.minimum(ci, RESERVED_N - 1)), 0)
    if np.any(off & (ci >= RESERVED_N)):
        raise SystemExit(f"chunk shield: chunk {int(ci.max())} has no reserved cluster (only {RESERVED_N}); "
                         "use a longer FK_SESSION_CHUNK_MIN")
    backups = []
    for k in ("clu", "clc", "clp"):
        if os.path.exists(tin[k]):
            with open(tin[k], "rb") as src, nio.atomic_write(tin[k] + ".chunkshield") as dst:
                dst.write(src.read())
            backups.append(tin[k])
    masked = lab.copy(); masked[off] = 0
    nio.write_clu_file(tin["clu"], masked)
    if os.path.exists(tin["clc"]) and os.path.exists(tin["clp"]):
        _, child = nio.read_clu_file(tin["clc"])
        if child.size == lab.size:
            child = np.asarray(child, np.int64).copy()
            gone = np.setdiff1d(np.unique(child[off]), np.unique(child[~off]))   # atoms left with no spike
            child[off] = 0
            _, par = nio.read_clu_file(tin["clp"])
            par = np.asarray(par, np.int64).copy()
            g = gone[(gone >= 1) & (gone <= par.size)]
            par[g - 1] = 0
            nio.write_clu_file(tin["clc"], child)
            nio.write_clu_file(tin["clp"], par)
    tout = _triple(base, elec, variant, out_tag)
    np.save(state_path + ".rid.npy", rid)
    st = dict(base=base, elec=int(elec), variant=variant, in_tag=in_tag, out_tag=out_tag, backups=backups,
              out_before={k: _stat(p) for k, p in tout.items()})
    with nio.atomic_write(state_path, "w") as fh:
        json.dump(st, fh)
    log(f"chunk shield: {int(off.sum()):,} spikes of the chunks not computed hidden from the stage "
        f"(input .{in_tag} saved; they return to clusters {RESERVED_LO}+chunk)")
    return st


def unshield(state_path, log=print):
    """Put the shielded spikes back into their reserved clusters in what the stage wrote; restore the input."""
    import json
    import os
    from . import neuro_io as nio
    from .fiber_refiberize import FiberHierarchy
    with open(state_path) as fh:
        st = json.load(fh)
    rid = np.load(state_path + ".rid.npy")
    off = rid != 0
    base, elec, variant = st["base"], st["elec"], st["variant"]
    tout = _triple(base, elec, variant, st["out_tag"])
    wrote = {k: _stat(p) != st["out_before"][k] and _stat(p) is not None for k, p in tout.items()}
    if wrote["clu"]:
        _, lab = nio.read_clu_file(tout["clu"], n_spikes=rid.size)
        lab = np.asarray(lab, np.int64).copy()
        if is_reserved(lab[~off]).any():
            raise SystemExit(f"chunk shield: {tout['clu']} numbers computed spikes into the reserved block")
        lab[off] = rid[off]
        done = False
        if wrote["clc"] and wrote["clp"]:
            h = FiberHierarchy.load(base, elec, clc=tout["clc"], clp=tout["clp"])
            if h.child.size == rid.size:
                child = np.asarray(h.child, np.int64).copy(); parent = dict(h.parent)
                nxt = max(int(child.max()) if child.size else 0, max(parent, default=0)) + 1
                for r in np.unique(rid[off]):
                    child[rid == r] = nxt; parent[nxt] = int(r); nxt += 1
                clu_b, clc_b, clp_b = FiberHierarchy(child, parent).refiberize(renumber=False)
                if np.array_equal(clu_b, lab):
                    for k, arr in (("clu", clu_b), ("clc", clc_b), ("clp", clp_b)):
                        nio.write_clu_file(tout[k], arr)
                    done = True
        if not done:
            nio.write_clu_file(tout["clu"], lab)
        log(f"chunk shield: {int(off.sum()):,} spikes back in their reserved clusters in .{st['out_tag']}"
            + (" (+ .clc/.clp)" if done else ""))
    same = st["in_tag"] == st["out_tag"]
    for path in st["backups"]:
        bak = path + ".chunkshield"
        if not os.path.exists(bak):
            continue
        kind = next((t for t, p in tout.items() if p == path), None)
        if same and kind is not None and wrote.get(kind, False):
            os.remove(bak)                                 # the stage replaced its own input: keep its output
        else:
            os.replace(bak, path)                          # restore the input exactly
    os.remove(state_path + ".rid.npy")
    os.remove(state_path)


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser(prog="fiber_chunks", description="chunk shield for fiber-pipeline (see module doc)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shield")
    s.add_argument("session"); s.add_argument("elec", type=int); s.add_argument("variant")
    s.add_argument("in_tag"); s.add_argument("out_tag"); s.add_argument("state")
    add_argument(s, None, length=True)
    u = sub.add_parser("unshield")
    u.add_argument("state")
    a = ap.parse_args(argv)
    pre = "▸ chunk-shield · "
    if a.cmd == "shield":
        chunks = parse(a.chunks if a.chunks else os.environ.get("FK_CHUNKS", ""))
        if chunks is None:
            return 0
        shield(a.session, a.elec, a.variant, a.in_tag, a.out_tag, a.state, chunks, a.chunk_length,
               log=lambda m: print(pre + m))
    else:
        unshield(a.state, log=lambda m: print(pre + m))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
