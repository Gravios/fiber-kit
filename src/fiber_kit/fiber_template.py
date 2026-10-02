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


def per_bin_median(spk, idx, bin_of_idx, nbins, nsamp, nchan, max_per=800):
    """(nbins, nsamp, nchan) median of the waveforms at `idx` split by bin; a bin
    with no spikes is left NaN (reported via nsp=0).  Each bin is strided to
    <= max_per spikes — a median over a few hundred is already stable."""
    T = np.full((nbins, nsamp, nchan), np.nan, np.float32)
    nsp = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        ix = idx[bin_of_idx == b]
        nsp[b] = ix.size
        if ix.size == 0:
            continue
        if ix.size > max_per:
            ix = ix[np.linspace(0, ix.size - 1, max_per).astype(int)]
        T[b] = np.median(np.asarray(spk[ix], np.float32), axis=0)
    return T, nsp


def generate(base, elec, *, nsamp, nchan, sr, variants, clu_variant, clu_tag="",
             out_tag="", units=None, links=("drift",), n_chunks=6, chunk_min=None,
             n_energy=5, energy_variant=None, max_per=800, drop_empty=True):
    """Build LINKED per-unit median templates (drift and/or adapt series) in each
    variant; write one .wtf per variant + the shared .wti index.  Returns
    (rows, paths)."""
    times = nio.read_res(base, elec)                        # SHARED detection res
    _, clu = nio.read_clu_at(base, elec, variant=clu_variant, tag=clu_tag)
    if clu.size != times.size:
        raise ValueError(f"res/clu length mismatch: res={times.size} clu={clu.size}")

    edges, nC = chunk_edges(times, sr, chunk_min=chunk_min, n_chunks=n_chunks)
    ch_of_spk = assign_bins(times, edges)
    if units is None:
        units = [int(u) for u in np.unique(clu) if u >= 2]

    spks = {v: nio.open_spk_at(base, elec, nsamp, nchan, variant=v, tag="") for v in variants}
    eref = energy_variant or ("standard" if "standard" in variants else variants[0])

    rows, tpl = [], {v: [] for v in variants}

    def _emit(unit, link, b, lo, hi, per_v):
        nsp_any = np.max([per_v[v][1] for v in variants], axis=0)
        if drop_empty and nsp_any[b] == 0:
            return
        rows.append(dict(unit=unit, link=link, bin=b, lo=lo, hi=hi,
                         nspk=int(nsp_any[b]),
                         src_clu_variant=clu_variant, src_clu_tag=clu_tag))
        for v in variants:
            tpl[v].append(np.nan_to_num(per_v[v][0][b]))

    for u in units:
        idx = np.flatnonzero(clu == u)
        if "drift" in links:
            cof = ch_of_spk[idx]
            per_v = {v: per_bin_median(spks[v], idx, cof, nC, nsamp, nchan, max_per)
                     for v in variants}
            for c in range(nC):
                _emit(u, "drift", c, round(float(edges[c]) / sr, 3),
                      round(float(edges[c + 1]) / sr, 3), per_v)
        if "adapt" in links and idx.size:
            E = spike_energy(spks[eref][idx])               # energy in the reference variant
            eedges = energy_edges(E, n_energy)
            eb = assign_bins(E, eedges)
            per_v = {v: per_bin_median(spks[v], idx, eb, n_energy, nsamp, nchan, max_per)
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
    ap.add_argument("--units", type=int, nargs="+", help="restrict to these unit ids")
    a = ap.parse_args()
    n_chunks = a.n_chunks if (a.n_chunks or a.chunk_min) else 6
    rows, paths = generate(
        a.base, a.elec, nsamp=a.nsamp, nchan=a.nchan, sr=a.sr, variants=a.variants,
        clu_variant=a.clu_variant, clu_tag=a.clu_tag, out_tag=a.out_tag, links=tuple(a.links),
        units=a.units, chunk_min=a.chunk_min, n_chunks=n_chunks, n_energy=a.n_energy,
        energy_variant=a.energy_variant, max_per=a.max_per)
    nu = len({r["unit"] for r in rows})
    by = {}
    for r in rows:
        by[r["link"]] = by.get(r["link"], 0) + 1
    print(f"[fiber-template] {len(rows)} rows ({by}) across {nu} unit(s); wrote:")
    for k, p in paths.items():
        print(f"  {k:10s} {p}")


if __name__ == "__main__":
    main()
