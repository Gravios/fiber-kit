#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  fiber_template.py  —  LINKED per-unit median templates, one .wtf per variant
#
#  A template "like a fiber": not one global median, but a LINKED SERIES of
#  per-bin medians along an axis, so the stored template tracks how a unit's
#  waveform changes:
#    * link="drift"  — median per time CHUNK (the unit's waveform moves as the
#                      probe/tissue drifts; cf. the per-(chunk,fiber) `template`
#                      the .fibers npz already carries).
#    * link="adapt"  — median per ENERGY bin (the waveform shrinks/changes with
#                      firing state — within-burst adaptation).
#
#  Both are generated in EVERY requested variant's .spk space and written
#  .spk-style as <base>.wtf.<variant>.<elec>[.<tag>] (one file per method and
#  stage), ROW-ALIGNED across variants to one shared, method-less index
#  <base>.wti.<elec>[.<tag>] -- the canonical .wti (row, unit, link, bin, a, b,
#  nSpikes, src…), the same contract the shared C++ reader (neurofileio) reads.
# ═══════════════════════════════════════════════════════════════════════════
import argparse

import numpy as np

try:
    from . import neuro_io as nio
except ImportError:
    import neuro_io as nio


def chunk_edges(times, sr, chunk_min=None, n_chunks=None):
    """Time-chunk boundaries in SAMPLE units: fixed duration (chunk_min, minutes)
    or fixed count (n_chunks).  Returns (edges[nC+1], nC)."""
    tmax = float(times.max()) if times.size else 0.0
    if n_chunks:
        nC = max(1, int(n_chunks))
    elif chunk_min:
        nC = max(1, int(np.ceil((tmax / sr / 60.0) / float(chunk_min))))
    else:
        nC = 1
    return np.linspace(0.0, tmax + 1.0, nC + 1), nC


def assign_bins(values, edges):
    """Bin index per value (0-based), clamped to [0, nbins-1]."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def spike_energy(waves):
    """Per-spike energy = L2 norm of the waveform (n,) — an amplitude proxy whose
    quantiles separate rested (high-E) from adapted/within-burst (low-E) spikes."""
    w = np.asarray(waves, np.float64)
    return np.sqrt((w * w).reshape(w.shape[0], -1).sum(1))


def energy_edges(E, n_energy):
    """Equal-occupancy (quantile) energy-bin edges, length n_energy+1."""
    qs = np.quantile(E, np.linspace(0.0, 1.0, n_energy + 1))
    qs[-1] += 1e-6                       # include the max in the last bin
    return qs


def _align_to_res(waves, offsets, nsamp):
    """Roll each (nsamp, nchan) waveform so the class EAP — stored `offset` samples
    from res in the .eap — lands in the res frame: aligned[t] = waves[t + offset],
    zero-filled where t+offset leaves the window.  So a class's offset-shifted
    (collision) members median together with its offset-0 clean members.
    waves: (k, nsamp, nchan); offsets: (k,).  A roll of 0 is the identity, so clean
    members (offset 0) are untouched and eap-mode matches clu-mode on a clean unit."""
    k = waves.shape[0]
    if k == 0:
        return waves
    off = np.asarray(offsets, np.int64)
    src = np.arange(nsamp)[None, :] + off[:, None]          # (k, nsamp) source sample per t
    valid = (src >= 0) & (src < nsamp)
    src_c = np.clip(src, 0, nsamp - 1)
    idx3 = np.broadcast_to(src_c[:, :, None], waves.shape)
    out = np.take_along_axis(waves, idx3, axis=1).astype(np.float32, copy=True)
    out[~valid] = 0.0                                        # zero the samples rolled in from outside
    return out


def per_bin_median(spk, idx, bin_of_idx, nbins, nsamp, nchan, max_per=800, off_of_idx=None):
    """(nbins, nsamp, nchan) median of the waveforms at `idx` split by bin; a bin
    with no spikes is left NaN (reported via nsp=0).  Each bin is strided to
    <= max_per spikes — a median over a few hundred is already stable.  When
    `off_of_idx` (aligned with `idx`) is given, each waveform is aligned to the res
    frame by its EAP offset before medianing (see _align_to_res)."""
    T = np.full((nbins, nsamp, nchan), np.nan, np.float32)
    nsp = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        sel = (bin_of_idx == b)
        ix = idx[sel]
        offs = off_of_idx[sel] if off_of_idx is not None else None
        nsp[b] = ix.size
        if ix.size == 0:
            continue
        if ix.size > max_per:
            keep = np.linspace(0, ix.size - 1, max_per).astype(int)
            ix = ix[keep]
            if offs is not None:
                offs = offs[keep]
        W = np.asarray(spk[ix], np.float32)
        if offs is not None:
            W = _align_to_res(W, offs, nsamp)
        T[b] = np.median(W, axis=0)
    return T, nsp


def generate(base, elec, *, nsamp, nchan, sr, variants, clu_variant, clu_tag="",
             out_tag="", units=None, links=("drift",), n_chunks=6, chunk_min=None,
             n_energy=5, energy_variant=None, max_per=800, drop_empty=True, eap=False):
    """Build LINKED per-template-class median templates (drift and/or adapt series)
    in each variant; write one .wtf per variant + the shared .wti index.  Returns
    (rows, paths).

    Membership source:
      * eap=False (legacy): a class is a .clu cluster; members are clu==id.
      * eap=True: a class is an .eap COLUMN (stable id); members are the spikes
        present in that column, each aligned to res by its stored offset so
        collision members contribute aligned.  `units` are class ids (columns);
        default = the .tcl active classes, else any column with a member.  The
        id stamped into .wti is the class id (column)."""
    times = nio.read_res(base, elec)                        # SHARED detection res
    N = int(times.size)

    cells = None
    if eap:
        E = nio.read_eap(base, elec, tag=clu_tag)
        if not E["ok"]:
            raise ValueError(f"no .eap for group {elec} (stage '{clu_tag}'); "
                             f"run process_initeap first")
        cells = E["cells"]
        if cells.shape[0] != N:
            raise ValueError(f"res/eap length mismatch: res={N} eap rows={cells.shape[0]}")
        if units is None:
            tcl = nio.read_tcl(base, elec)
            if tcl["ok"] and tcl["entries"]:
                units = [e["col"] for e in tcl["entries"] if e["status"] == "active"]
            if not units:                                   # no registry / none active
                units = [j for j in range(cells.shape[1])
                         if bool(np.any(cells[:, j] != nio.EAP_ABSENT))]
    else:
        _, clu = nio.read_clu_at(base, elec, variant=clu_variant, tag=clu_tag)
        if clu.size != N:
            raise ValueError(f"res/clu length mismatch: res={N} clu={clu.size}")
        if units is None:
            units = [int(u) for u in np.unique(clu) if u >= 2]

    edges, nC = chunk_edges(times, sr, chunk_min=chunk_min, n_chunks=n_chunks)
    ch_of_spk = assign_bins(times, edges)

    spks = {v: nio.open_spk_at(base, elec, nsamp, nchan, variant=v, tag="") for v in variants}
    eref = energy_variant or ("standard" if "standard" in variants else variants[0])

    rows, tpl = [], {v: [] for v in variants}
    id_key = "class" if eap else "unit"

    def _emit(uid, link, b, lo, hi, per_v):
        nsp_any = np.max([per_v[v][1] for v in variants], axis=0)
        if drop_empty and nsp_any[b] == 0:
            return
        rd = {id_key: uid, "link": link, "bin": b, "lo": lo, "hi": hi,
              "nspk": int(nsp_any[b]), "src_clu_variant": clu_variant, "src_clu_tag": clu_tag}
        rows.append(rd)
        for v in variants:
            tpl[v].append(np.nan_to_num(per_v[v][0][b]))

    for u in units:
        if eap:
            col = cells[:, u]
            idx = np.flatnonzero(col != nio.EAP_ABSENT)
            off = col[idx].astype(np.int64)                 # per-member EAP offset -> alignment
        else:
            idx = np.flatnonzero(clu == u)
            off = None
        if "drift" in links:
            cof = ch_of_spk[idx]
            per_v = {v: per_bin_median(spks[v], idx, cof, nC, nsamp, nchan, max_per, off_of_idx=off)
                     for v in variants}
            for c in range(nC):
                _emit(u, "drift", c, round(float(edges[c]) / sr, 3),
                      round(float(edges[c + 1]) / sr, 3), per_v)
        if "adapt" in links and idx.size:
            E_en = spike_energy(spks[eref][idx])            # energy in the reference variant
            eedges = energy_edges(E_en, n_energy)
            eb = assign_bins(E_en, eedges)
            per_v = {v: per_bin_median(spks[v], idx, eb, n_energy, nsamp, nchan, max_per, off_of_idx=off)
                     for v in variants}
            for b in range(n_energy):
                _emit(u, "adapt", b, round(float(eedges[b]), 1),
                      round(float(eedges[b + 1]), 1), per_v)

    paths = {}
    for v in variants:
        arr = (np.rint(np.stack(tpl[v])).astype(nio.SPK_DTYPE) if tpl[v]
               else np.zeros((0, nsamp, nchan), nio.SPK_DTYPE))
        paths[v] = nio.write_wtf(base, elec, arr, variant=v, tag=out_tag)
    paths["info"] = nio.write_wti(base, elec, rows, nsamp, nchan, sr=sr, variant="", tag=out_tag)
    return rows, paths


def main():
    ap = argparse.ArgumentParser(
        description="Linked per-unit median templates (drift + adapt), one .wtf per variant.")
    ap.add_argument("base")
    ap.add_argument("elec", type=int)
    ap.add_argument("--variants", nargs="+", required=True,
                    help="spk variants to template, e.g. standard stderiv_C5_D34")
    ap.add_argument("--clu-variant", required=True, help="variant of the units .clu")
    ap.add_argument("--clu-tag", default="", help="tag/stage of the units .clu")
    ap.add_argument("--out-tag", default="", help="stage tag for the written .wtf/.wti")
    ap.add_argument("--links", nargs="+", default=["drift"], choices=["drift", "adapt"],
                    help="template link axes to generate (default: drift)")
    ap.add_argument("--nsamp", type=int, required=True)
    ap.add_argument("--nchan", type=int, required=True)
    ap.add_argument("--sr", type=float, default=32552.0)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--chunk-min", type=float, help="drift chunk duration in minutes")
    g.add_argument("--n-chunks", type=int, help="drift: fixed number of chunks (default 6)")
    ap.add_argument("--n-energy", type=int, default=5, help="adapt: number of energy bins")
    ap.add_argument("--energy-variant", help="variant whose energy defines the adapt bins "
                                             "(default: standard if present, else the first)")
    ap.add_argument("--max-per", type=int, default=800)
    ap.add_argument("--units", type=int, nargs="+",
                    help="restrict to these ids (clu ids, or .eap class columns with --eap)")
    ap.add_argument("--eap", action="store_true",
                    help="membership from the .eap matrix (class = column, offset-aligned), "
                         "not the .clu; the stable class id is stamped into .wti")
    a = ap.parse_args()
    # Chunk edges: explicit flag wins; else fixed 12-min chunks in --eap mode (stable
    # across concatenation), else the legacy 6-chunk default.
    chunk_min, n_chunks = a.chunk_min, a.n_chunks
    if not n_chunks and not chunk_min:
        chunk_min, n_chunks = (12.0, None) if a.eap else (None, 6)
    rows, paths = generate(
        a.base, a.elec, nsamp=a.nsamp, nchan=a.nchan, sr=a.sr, variants=a.variants,
        clu_variant=a.clu_variant, clu_tag=a.clu_tag, out_tag=a.out_tag, links=tuple(a.links),
        units=a.units, chunk_min=chunk_min, n_chunks=n_chunks, n_energy=a.n_energy,
        energy_variant=a.energy_variant, max_per=a.max_per, eap=a.eap)
    nu = len({r.get("class", r.get("unit")) for r in rows})
    by = {}
    for r in rows:
        by[r["link"]] = by.get(r["link"], 0) + 1
    print(f"[fiber-template] {len(rows)} rows ({by}) across {nu} unit(s); wrote:")
    for k, p in paths.items():
        print(f"  {k:10s} {p}")


if __name__ == "__main__":
    main()
