# ════════════════════════════════════════════════════════════════════════════
#  neuro_io.py — standardized neurosuite-3 on-disk I/O for fiber-kit.
#
#  Single source of truth for the NeuroSuite binary formats fiber-kit reads and
#  writes, so the naming convention and variant resolution match the C++
#  toolchain exactly instead of being re-implemented (with hardcoded extension
#  lists) in every script.  All artifacts are binary (see below).
#
#  Mirrors:
#    - src/libneurosuite-core/src/neurosuite/core/neurofileio.{h,cpp}
#        readClu / writeClu / readRes / writeRes / readFetBinary /
#        readClusterRes / resolveInput / preferDerived / preferCanonical
#    - src/kiloklustakwik/src/KK_io.cpp
#        LoadData / LoadClu  (the pickInputPath -> resolveInput .fet/.fetD fallback)
#
#  ── Variant-aware naming (NeuroSuite convention) ──────────────────────────────
#  A per-group typed file (res/clu/fet/spk/pca …) may exist in several
#  representation "variants" — most notably the stderiv-derived features.  The
#  group number is ALWAYS the trailing token; the variant, when present, sits
#  between the type and the group:
#
#      <base>.<type>.<group>            canonical (no variant)        e.g. .spk.5
#      <base>.<type>.<variant>.<group>  dotted variant (preferred)   e.g. .spk.stderiv.5
#      <base>.<type><variant>.<group>   legacy glued form (READ-only) e.g. .spkD.5
#
#  The glued form (a single letter glued onto the type token, as the stderiv
#  pipeline historically wrote .fetD/.spkD/.pcaD) is recognised on READ only,
#  for backward compatibility; new writers emit the dotted form.  resolve_input()
#  walks `prefer_variants` in order and returns the first file that exists; ""
#  denotes the canonical (no-variant) form.  fiber-kit operates in stderiv space
#  so it defaults to prefer_derived()  ({"stderiv","D",""}); the .fet reader
#  defaults to prefer_canonical() to match KlustaKwik's pickInputPath.
#
#  ── All artifacts are binary ──────────────────────────────────────────────────
#  fiber-kit reads and writes .res (int64), .clu (int32 nClusters header + int32
#  ids) and .fet (int32 header + int64 body) as raw binary only.  The legacy text
#  format and the first-byte binary/text heuristic have been removed: that
#  heuristic misclassified a binary file whose header low byte lands in 0x30-0x39
#  as text (e.g. a 1332-cluster .clu -> first header byte 0x34 = '4'), which broke
#  reads outright -- it was never the "non-issue in practice" it was assumed to be.
# ════════════════════════════════════════════════════════════════════════════
import os
from collections import namedtuple

import numpy as np

__all__ = [
    "ResolvedInput", "resolve_input", "prefer_derived", "prefer_canonical",
    "read_res_file", "write_res_file", "read_res", "write_res",
    "read_clu_file", "write_clu_file", "read_clu", "write_clu",
    "read_cluster_res",
    "read_fet_file", "read_fet", "write_fet_file", "write_fet",
    "open_spk_file", "open_spk", "open_spkD", "write_spk_file", "write_spk",
    "open_signal",
    "fibers_path",
]

# ── dtypes (little-endian, NeuroSuite on-disk) ──────────────────────────────
RES_DTYPE = np.dtype("<i8")   # .res    int64 timestamps, no header
CLU_DTYPE = np.dtype("<i4")   # .clu    int32 header + int32 ids
FET_DTYPE = np.dtype("<i8")   # .fet    int32 header + int64 values, row-major
SPK_DTYPE = np.dtype("<i2")   # .spk    int16 sample-major, no header
FET_HDR_DTYPE = np.dtype("<i4")


# ── variant-aware input resolution (mirrors neurofileio::resolveInput) ───────
ResolvedInput = namedtuple("ResolvedInput", ["path", "variant", "dotted", "found"])


def prefer_derived():
    """Preference order for stderiv-space inputs: derived first, then a raw fallback.
    Mirrors neurofileio::preferDerived().  Under the dotted naming convention raw is the
    'standard' variant (<base>.spk.standard.N), so the raw fallback is 'standard' (with the
    legacy canonical <base>.spk.N last)."""
    return ["stderiv", "D", "standard", ""]


def prefer_standard():
    """Preference order for RAW (standard) inputs: <base>.spk.standard.N first, then the
    legacy canonical <base>.spk.N.  Deliberately does NOT fall back to stderiv -- callers
    that need raw amplitudes (localization, position, raw-PCA) must fail rather than
    silently localize on the stderiv transform (which breaks the amplitude-distance law)."""
    return ["standard", ""]


def prefer_canonical():
    """Preference order for canonical inputs first.  Mirrors
    neurofileio::preferCanonical()  ->  {"", "stderiv", "D"}."""
    return ["", "stderiv", "D"]


def resolve_input(base, type_, group, prefer_variants):
    """Return the first existing variant of <base>.<type>[.<variant>].<group>.

    Walks `prefer_variants` in order; "" denotes the canonical (no-variant)
    form.  For each non-empty variant the dotted form (<base>.<type>.<variant>.
    <group>) is probed first, then the legacy glued form (<base>.<type><variant>.
    <group>).  If nothing exists, `found` is False and `path` is the canonical
    path so the caller can emit a sensible "missing input" error.

    Faithful port of neurofileio::resolveInput / KK_io's pickInputPath.
    """
    g = str(group)
    canonical = f"{base}.{type_}.{g}"
    for v in prefer_variants:
        if v == "":
            if os.path.exists(canonical):
                return ResolvedInput(canonical, "", False, True)
            continue
        dotted = f"{base}.{type_}.{v}.{g}"
        if os.path.exists(dotted):
            return ResolvedInput(dotted, v, True, True)
        glued = f"{base}.{type_}{v}.{g}"
        if os.path.exists(glued):
            return ResolvedInput(glued, v, False, True)
    return ResolvedInput(canonical, "", False, False)


# ── .res.N ───────────────────────────────────────────────────────────────────
def read_res_file(path):
    """Read a binary .res file (int64 spike timestamps; mirrors
    neurofileio::writeRes).  Returns an int64 ndarray."""
    return np.fromfile(path, dtype=RES_DTYPE).astype(np.int64)


def write_res_file(path, times):
    """Write spike timestamps as canonical binary .res (little-endian int64,
    no header).  Mirrors neurofileio::writeRes' binary counterpart."""
    np.asarray(times, dtype=RES_DTYPE).tofile(path)
    return path


def read_res(base, elec, prefer=None):
    """Resolve and read <base>.res.<elec> (variant-aware).  Defaults to the
    canonical form first; pass prefer_derived() to prefer a derived .res."""
    r = resolve_input(base, "res", elec, prefer or prefer_canonical())
    if not r.found:
        raise FileNotFoundError(f"no .res for {base} elec {elec}")
    return read_res_file(r.path)


def session_path(base, type_, group, variant="", tag=""):
    """Canonical neurosuite-3 path:  <base>.<type>[.<variant>].<group>[.<tag>].

    `variant` (the feature space / method: standard|stderiv|sdiff|...) precedes the
    group; `tag` (the post-group fiber/pipeline STAGE: refine|realigned|...) follows
    it.  Empty slots are omitted.  The tag axis is method-pinned (built directly,
    never probed by resolve_input), so e.g. the refined stderiv sort of group 5 is
        <base>.clu.stderiv.5.refine
    -- variant `stderiv` before the group, fiber stage `refine` after it."""
    p = f"{base}.{type_}"
    if variant:
        p += f".{variant}"
    p += f".{group}"
    if tag:
        p += f".{str(tag).replace('.', '_')}"   # Klusters parses '.' as a field separator, so a
    return p                                     # multi-part STAGE must join with '_' (refine_linked)


def write_res(base, elec, times, variant="", tag=None):
    """Write <base>.res[.<variant>].<elec>[.<tag>] as binary int64.  `tag` is the
    post-group fiber-stage / curation axis (e.g. 'refine' -> <base>.res.<variant>.<elec>.refine,
    'realigned' -> ...res.<elec>.realigned), a separate axis from `variant` and never
    touched by resolve_input."""
    return write_res_file(session_path(base, "res", elec, variant=variant, tag=tag or ""), times)


# ── .clu.N ───────────────────────────────────────────────────────────────────
def read_clu_file(path, n_spikes=None):
    """Read a binary .clu -> (nClusters, ids:int64 ndarray): int32 nClusters
    header + int32 ids (mirrors neurofileio::writeClu / KK::LoadClu binary path).
    `n_spikes` truncates the id array for symmetry with the C++ reader; the file
    size otherwise determines it."""
    raw = np.fromfile(path, dtype=CLU_DTYPE)
    n_clu = int(raw[0]) if raw.size else 0
    ids = raw[1:].astype(np.int64)
    if n_spikes is not None and ids.size != n_spikes:
        ids = ids[:n_spikes]
    return n_clu, ids


def write_clu_file(path, ids, n_clusters=None):
    """Write canonical binary .clu (int32 nClusters header + int32 ids).
    `n_clusters` defaults to max(id)+1 (NeuroSuite convention: ids are the
    header count, 0 = noise).  Mirrors neurofileio::writeClu's binary form."""
    ids = np.asarray(ids, dtype=CLU_DTYPE)
    if n_clusters is None:
        n_clusters = int(ids.max()) + 1 if ids.size else 0
    with open(path, "wb") as f:
        np.array([n_clusters], CLU_DTYPE).tofile(f)
        ids.tofile(f)
    return path


def read_clu(base, elec, n_spikes=None, prefer=None):
    """Resolve and read <base>.clu.<elec> -> (nClusters, ids)."""
    r = resolve_input(base, "clu", elec, prefer or prefer_canonical())
    if not r.found:
        raise FileNotFoundError(f"no .clu for {base} elec {elec}")
    return read_clu_file(r.path, n_spikes=n_spikes)


def read_clu_at(base, elec, variant="", tag="", n_spikes=None):
    """Read a method-pinned staged .clu directly (no resolve_input probing):
    <base>.clu[.<variant>].<elec>[.<tag>], e.g. variant='stderiv', tag='refine'
    -> <base>.clu.stderiv.<elec>.refine.  Raises with the expected path if absent."""
    path = session_path(base, "clu", elec, variant=variant, tag=tag)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no .clu at {path} (variant={variant!r}, tag={tag!r}); pass --in-clu or fix the method/stage")
    return read_clu_file(path, n_spikes=n_spikes)


def write_clu(base, elec, ids, n_clusters=None, variant="", tag=""):
    """Write <base>.clu[.<variant>].<elec>[.<tag>].  `variant` = method (before group),
    `tag` = fiber stage (after group), e.g. variant='stderiv', tag='refine' ->
    <base>.clu.stderiv.<elec>.refine."""
    return write_clu_file(session_path(base, "clu", elec, variant=variant, tag=tag),
                          ids, n_clusters=n_clusters)


# ── matched .clu + .res pair (mirrors neurofileio::readClusterRes) ───────────
ClusterResData = namedtuple(
    "ClusterResData", ["n_clusters", "ids", "times", "binary", "ok"])


def read_cluster_res(base, elec, prefer=None):
    """Read the matched binary .clu/.res pair.  Returns ClusterResData with ok=False
    on any open/parse/length mismatch (never raises), mirroring the C++ helper."""
    prefer = prefer or prefer_canonical()
    rr = resolve_input(base, "res", elec, prefer)
    rc = resolve_input(base, "clu", elec, prefer)
    if not (rr.found and rc.found):
        return ClusterResData(0, np.empty(0, np.int64), np.empty(0, np.int64), False, False)
    times = read_res_file(rr.path)
    n_clu, ids = read_clu_file(rc.path)
    ok = ids.size == times.size
    return ClusterResData(n_clu, ids, times, True, ok)


# ── .fet.N (binary; mirrors KK::LoadData binary path / readFetBinary) ────────
FetBinaryFile = namedtuple("FetBinaryFile", ["n_features", "n_spikes", "values", "ok"])


def read_fet_file(path):
    """Read a binary .fet: int32 feature-count header, then nSpikes × nFeatures
    int64 values row-major.  Returns FetBinaryFile (values shaped (nSpikes,
    nFeatures)).  Mirrors neurofileio::readFetBinary / KK::LoadData."""
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=FET_HDR_DTYPE, count=1)
        if hdr.size != 1 or int(hdr[0]) < 1:
            return FetBinaryFile(0, 0, np.empty((0, 0), np.int64), False)
        n_feat = int(hdr[0])
        body = np.fromfile(f, dtype=FET_DTYPE)
    if body.size % n_feat != 0:
        return FetBinaryFile(n_feat, 0, np.empty((0, n_feat), np.int64), False)
    n_spk = body.size // n_feat
    return FetBinaryFile(n_feat, n_spk, body.reshape(n_spk, n_feat), True)


def read_fet(base, elec, prefer=None):
    """Resolve and read <base>.fet.<elec> (binary), defaulting to canonical-first
    to match KlustaKwik's pickInputPath (.fet then .fetD)."""
    r = resolve_input(base, "fet", elec, prefer or prefer_canonical())
    if not r.found:
        raise FileNotFoundError(f"no .fet for {base} elec {elec}")
    return read_fet_file(r.path)


def write_fet_file(path, values):
    """Write a binary .fet: int32 feature-count header + nSpikes x nFeatures
    int64 values row-major (inverse of read_fet_file)."""
    values = np.asarray(values, dtype=FET_DTYPE)
    if values.ndim != 2:
        raise ValueError("write_fet_file expects (nSpikes, nFeatures)")
    with open(path, "wb") as f:
        np.array([values.shape[1]], FET_HDR_DTYPE).tofile(f)
        values.tofile(f)
    return path


def write_fet(base, elec, values, variant="", tag=""):
    """Write <base>.fet[.<variant>].<elec>[.<tag>] (binary), mirroring write_res/write_clu.
    `variant` = method (before group), `tag` = fiber stage (after group)."""
    return write_fet_file(session_path(base, "fet", elec, variant=variant, tag=tag), values)


# ── .spk.N / .spkD.N (int16, sample-major, no header) ────────────────────────
def open_spk_file(path, nsamp, nchan, mode="r"):
    """Memmap a .spk/.spkD file as (nSpikes, nsamp, nchan) int16, sample-major.
    Only whole spikes are exposed (trailing partial spike, if any, is dropped)."""
    mm = np.memmap(path, dtype=SPK_DTYPE, mode=mode)
    n = mm.size // (nsamp * nchan)
    return mm[:n * nsamp * nchan].reshape(n, nsamp, nchan)


def open_spk(base, elec, nsamp, nchan, prefer=None, mode="r"):
    """Resolve and memmap waveforms for (base, elec).  Defaults to prefer_derived()
    (stderiv/D before canonical), so a stderiv run picks up .spk.stderiv.N /
    .spk.D.N / legacy .spkD.N before the raw .spk.N — preserving fiber-kit's
    historical '.spkD then .spk' preference while also honouring the dotted form.
    Returns (memmap (n,nsamp,nchan), ResolvedInput)."""
    r = resolve_input(base, "spk", elec, prefer or prefer_derived())
    if not r.found:
        raise FileNotFoundError(f"no .spkD/.spk for {base} elec {elec}")
    return open_spk_file(r.path, nsamp, nchan, mode=mode), r


def open_spkD(base, elec, nsamp, nch, mode="r"):
    """Back-compat shim: same contract as the historical fiber_session.open_spkD
    — returns (memmap, path), preferring the derived (stderiv) representation."""
    mm, r = open_spk(base, elec, nsamp, nch, prefer=prefer_derived(), mode=mode)
    return mm, r.path


def open_spk_raw(base, elec, nsamp, nchan, mode="r"):
    """Resolve and memmap the RAW (standard) waveforms: <base>.spk.standard.N (then legacy
    <base>.spk.N).  For position/amplitude work — never returns the stderiv .spk.  Returns
    (memmap (n,nsamp,nchan), ResolvedInput); raises if the resolved file is a stderiv form."""
    mm, r = open_spk(base, elec, nsamp, nchan, prefer=prefer_standard(), mode=mode)
    if r.variant in ("stderiv", "D"):
        raise FileNotFoundError(f"no raw/standard .spk for {base} elec {elec} "
                                f"(resolved {r.path}; refusing stderiv for amplitude work)")
    return mm, r


def write_spk_file(path, waves):
    """Write a .spk/.spkD: int16, sample-major, no header (inverse of
    open_spk_file).  `waves` is (nSpikes, nsamp, nchan)."""
    np.asarray(waves, dtype=SPK_DTYPE).tofile(path)
    return path


def write_spk(base, elec, waves, variant="", tag=""):
    """Write <base>.spk[.<variant>].<elec>[.<tag>] (int16, sample-major).  `variant` =
    method (before group), `tag` = fiber stage (after group)."""
    return write_spk_file(session_path(base, "spk", elec, variant=variant, tag=tag), waves)


# ── group-wide spike-count edit (dedup propagation) ──────────────────────────
# Stage tokens (the dotted segments AFTER the electrode) that mark a file as NOT a
# live per-spike artefact and so must never be edited by a dedup propagation:
#   * a backup snapshot ('..._bkp')
#   * a byte-split fragment ('.part.aa', ...)
#   * a sidecar with its own row semantics ('.units.npz', ...)
# Dated snapshots ('.06.14.2026.12.35') are caught separately by the all-digit test:
# a frozen backup of the pre-dedup state has exactly n_orig rows, so the row-count
# guard alone CANNOT distinguish it from a live file -- the name must.
_PERSPIKE_SIDECAR = {"npz", "npy", "gz", "zip", "bak", "tmp", "swp", "units"}


def _is_live_perspike(stage):
    """True iff the stage tokens (everything after the electrode) name a live per-spike
    file -- not a dated/explicit backup, byte-split fragment, or non-binary sidecar."""
    for tok in stage:
        if tok.isdigit():                 # dated-snapshot component, e.g. 06.14.2026.12.35
            return False
        if tok == "part":                 # byte-split fragment (.part.aa)
            return False
        if tok.endswith("bkp"):           # explicit backup (refine_linked_bkp)
            return False
        if tok in _PERSPIKE_SIDECAR:      # .units.npz and friends
            return False
    return True


def apply_spike_keep(base, elec, keep, n_orig, nsamp, nchan,
                     types=("res", "clu", "spk", "spkD", "fet", "fetD"), verbose=True, strict=True):
    """Subset EVERY live per-spike file of one spike group to `keep`, in place, so the
    .res / .clu / .spk(/.spkD) / .fet(/.fetD) of the group stay row-aligned after a dedup --
    across all variants (standard|stderiv|...) and post-group stages.

    All per-spike files are BINARY; .clu/.res are read here directly with their fixed binary
    dtype (CLU_DTYPE int32 header+ids, RES_DTYPE int64), never via a text/binary heuristic, so a
    binary file whose header byte happens to be an ASCII digit (e.g. nClusters=1590 -> leading
    byte 0x36 '6') is never mis-parsed as text.

    A live file at the original count `n_orig` is rewritten.  A file already at the post-dedup
    count `nkeep` is left untouched (idempotent re-run).  A live file at ANY OTHER count is
    STALE/misaligned -- its rows do not correspond 1:1 to this spike set, so it cannot be subset
    by `keep`; with strict=True (default) this raises (naming the files) rather than silently
    leaving the group half-deduped, so the stale files can be regenerated or removed and the run
    re-tried.  Pass strict=False to skip them and proceed.  `keep` is a boolean mask or index
    array over the original n_orig spikes.  Returns (rewritten_paths, skipped:[(path, rows)])."""
    import glob
    import os
    keep = np.asarray(keep)
    nkeep = int(keep.sum()) if keep.dtype == bool else len(keep)
    elec = str(elec)
    bname = os.path.basename(base)
    rewritten = []
    done = []                                              # already at the deduped count (idempotent)
    orphans = []                                           # stale / misaligned live files
    excluded = 0
    seen = set()
    for t in types:
        for path in sorted(glob.glob(f"{glob.escape(base)}.{t}.*")):
            tail = os.path.basename(path)[len(bname) + 1:].split(".")  # <type>[.<variant>].<elec>[.<stage>]
            if not tail or tail[0] != t:
                continue
            # the electrode is the first all-digit token (variants are non-numeric); everything
            # after it is the stage.  Require an EXACT group match so group 5 never picks up 15.
            ei = next((i for i in range(1, len(tail)) if tail[i].isdigit()), None)
            if ei is None or tail[ei] != elec:
                continue
            if not _is_live_perspike(tail[ei + 1:]):
                excluded += 1                              # backup / fragment / sidecar -- never touch
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                rows = None
                if t == "res":
                    raw = np.fromfile(path, dtype=RES_DTYPE); rows = raw.size       # binary, fixed dtype
                    if rows == n_orig:
                        write_res_file(path, raw[keep]); rewritten.append(path); continue
                elif t == "clu":
                    raw = np.fromfile(path, dtype=CLU_DTYPE); rows = max(raw.size - 1, 0)  # int32 header + ids
                    if rows == n_orig:
                        write_clu_file(path, raw[1:][keep].astype(np.int64), n_clusters=int(raw[0]))
                        rewritten.append(path); continue
                elif t in ("spk", "spkD"):
                    w = open_spk_file(path, nsamp, nchan); rows = w.shape[0]
                    if rows == n_orig:
                        write_spk_file(path, np.asarray(w[keep])); rewritten.append(path); continue  # materialise before truncate
                elif t in ("fet", "fetD"):
                    f = read_fet_file(path); rows = f.n_spikes if f.ok else "unreadable"
                    if f.ok and rows == n_orig:
                        write_fet_file(path, f.values[keep]); rewritten.append(path); continue
                else:
                    continue
                (done if rows == nkeep else orphans).append((path, rows))
            except Exception as e:  # noqa: BLE001 -- one bad file must not abort the sweep silently
                orphans.append((path, f"error: {e}"))
    if verbose:
        if rewritten:
            print(f"dedup: rewrote {len(rewritten)} group file(s) to {nkeep} spikes: "
                  + ", ".join(os.path.basename(p) for p in rewritten))
        for p, c in done:
            print(f"dedup: already deduped, left {os.path.basename(p)} (rows={c})")
        for p, c in orphans:
            print(f"dedup: STALE/misaligned {os.path.basename(p)} (rows={c}, expected {n_orig} or {nkeep})")
        if excluded:
            print(f"dedup: left {excluded} backup/fragment/sidecar file(s) untouched")
    if orphans and strict:
        names = ", ".join(os.path.basename(p) for p, _ in orphans)
        raise RuntimeError(
            f"dedup: {len(orphans)} live per-spike file(s) of group {elec} are misaligned "
            f"(rows != {n_orig} and != {nkeep}) and cannot be subset by the keep-mask: {names}. "
            f"These are stale from an earlier run -- regenerate or remove them and re-run "
            f"(or pass --no-dedup-strict / strict=False to skip them).")
    return rewritten, done + orphans


# ── .dat / .fil / .lfp (interleaved int16) ───────────────────────────────────
def open_signal(path, nchan, mode="r"):
    """Memmap an interleaved int16 wideband/LFP file as (nSamples, nchan).
    Used for .fil/.dat baseline reads (neurofileio::readDatWindow analogue)."""
    return np.memmap(path, dtype=SPK_DTYPE, mode=mode).reshape(-1, nchan)


# ── .fibers.<method>.<elec> (dotted-variant npz) ─────────────────────────────
def fibers_path(base, method, elec):
    """Path of the per-(chunk,fiber) geometry table.  This is already in the
    canonical dotted-variant form <base>.<type>.<variant>.<group> with
    type='fibers', variant=method (e.g. 'stderiv'), group=elec."""
    return f"{base}.fibers.{method}.{elec}"
