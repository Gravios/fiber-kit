# ════════════════════════════════════════════════════════════════════════════
#  neuro_io.py — standardized neurosuite-3 on-disk I/O for fiber-kit.
#
#  Single source of truth for the NeuroSuite binary formats fiber-kit reads and
#  writes, so the naming convention, variant resolution, and binary/text auto-
#  detection match the C++ toolchain exactly instead of being re-implemented
#  (with hardcoded extension lists) in every script.
#
#  Mirrors:
#    - src/libneurosuite-core/src/neurosuite/core/neurofileio.{h,cpp}
#        readClu / writeClu / readRes / writeRes / readFetBinary /
#        isBinaryClusterRes / readClusterRes / resolveInput / preferDerived /
#        preferCanonical
#    - src/kiloklustakwik/src/KK_io.cpp
#        LoadData / LoadClu  (the binary-vs-text first-byte heuristic, and the
#        pickInputPath → resolveInput .fet/.fetD fallback)
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
#  ── Binary vs text auto-detection ─────────────────────────────────────────────
#  Canonical heuristic (NeuroScope / neurofileio::isBinaryClusterRes, KK_io):
#  a file is binary iff its first byte is not an ASCII digit (0x30–0x39).  For
#  .res the size-multiple-of-8 test is added.  Caveat (documented upstream): a
#  binary file whose leading byte happens to fall in 0x30–0x39 — e.g. a .res
#  whose first timestamp is 48..57 samples, or a binary .fet/.clu header whose
#  low byte lands there — is misclassified as text.  This matches the C++ side
#  exactly; fiber-kit only ever writes binary, so it is a non-issue in practice
#  and is kept identical for cross-tool consistency rather than "improved" here.
# ════════════════════════════════════════════════════════════════════════════
import os
from collections import namedtuple

import numpy as np

__all__ = [
    "ResolvedInput", "resolve_input", "prefer_derived", "prefer_canonical",
    "is_binary_first_byte", "is_binary_cluster_res",
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


# ── binary/text detection (mirrors isBinaryClusterRes / KK_io heuristic) ─────
def is_binary_first_byte(path):
    """True iff the first byte of `path` is not an ASCII digit (the KK_io /
    NeuroScope heuristic).  Used for .clu and .fet standalone detection."""
    with open(path, "rb") as f:
        b = f.read(1)
    if not b:
        return False
    return not (0x30 <= b[0] <= 0x39)


def is_binary_cluster_res(res_path):
    """True iff `res_path` is a binary .res: size a non-zero multiple of 8 AND
    first byte not an ASCII digit.  Exact port of neurofileio::isBinaryClusterRes;
    the matched .clu is then read in the same format (see read_cluster_res)."""
    sz = os.path.getsize(res_path)
    if sz <= 0 or (sz % 8) != 0:
        return False
    return is_binary_first_byte(res_path)


# ── .res.N ───────────────────────────────────────────────────────────────────
def read_res_file(path):
    """Read a .res file (auto-detect binary int64 vs legacy whitespace text).
    Returns an int64 ndarray of spike timestamps."""
    if is_binary_cluster_res(path):
        return np.fromfile(path, dtype=RES_DTYPE).astype(np.int64)
    # legacy text: one timestamp per line / whitespace-separated
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


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
        p += f".{tag}"
    return p


def write_res(base, elec, times, variant="", tag=None):
    """Write <base>.res[.<variant>].<elec>[.<tag>] as binary int64.  `tag` is the
    post-group fiber-stage / curation axis (e.g. 'refine' -> <base>.res.<variant>.<elec>.refine,
    'realigned' -> ...res.<elec>.realigned), a separate axis from `variant` and never
    touched by resolve_input."""
    return write_res_file(session_path(base, "res", elec, variant=variant, tag=tag or ""), times)


# ── .clu.N ───────────────────────────────────────────────────────────────────
def read_clu_file(path, n_spikes=None, binary=None):
    """Read a .clu file -> (nClusters, ids:int64 ndarray).

    Auto-detects binary (int32 nClusters header + int32 ids) vs legacy text
    (count line, then one id per line), mirroring KK::LoadClu.  `binary` forces
    the format; `n_spikes` is accepted for symmetry with the C++ binary reader
    (which needs it) but is not required here since the file size determines it.
    """
    if binary is None:
        binary = is_binary_first_byte(path)
    if binary:
        raw = np.fromfile(path, dtype=CLU_DTYPE)
        n_clu = int(raw[0]) if raw.size else 0
        ids = raw[1:].astype(np.int64)
        if n_spikes is not None and ids.size != n_spikes:
            ids = ids[:n_spikes]
        return n_clu, ids
    with open(path) as f:
        n_clu = int(f.readline().split()[0])
        ids = np.array([int(x) for x in f.read().split()], dtype=np.int64)
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
    """Read the matched .clu/.res pair, detecting binary-vs-text from the .res
    and reading the .clu in the same format.  Returns ClusterResData with ok=False
    on any open/parse/length mismatch (never raises), mirroring the C++ helper."""
    prefer = prefer or prefer_canonical()
    rr = resolve_input(base, "res", elec, prefer)
    rc = resolve_input(base, "clu", elec, prefer)
    if not (rr.found and rc.found):
        return ClusterResData(0, np.empty(0, np.int64), np.empty(0, np.int64), False, False)
    binary = is_binary_cluster_res(rr.path)
    times = read_res_file(rr.path)
    n_clu, ids = read_clu_file(rc.path, binary=binary)
    ok = ids.size == times.size
    return ClusterResData(n_clu, ids, times, binary, ok)


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
