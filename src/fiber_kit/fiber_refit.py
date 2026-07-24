#!/usr/bin/env python3
"""Refit the fiber model to a manually curated .clu.

After a curator edits clusters in Klusters (merging fragments, splitting, removing noise), the per-unit
fiber artifacts -- templates, positions, the per-chunk signatures fiber-link/fiber-drift consume -- are
stale: they describe the old automatic grouping.  fiber-refit rebuilds them from the curated labels,
taking the curator's grouping as FINAL (identity grouping -- it never re-merges or re-splits).

For each curated unit it rebuilds a signature PER CHUNK (so drift is still tracked), reusing the exact
modelling fiber-intrachunk uses (mutual-centred mean template, inter-channel offsets, amplitude, optional
cfiber shape descriptor) and the monopole positions from fiber-cpos.  The result is a refreshed
<...>.units.npz keyed on the curated units, ready for fiber-link / fiber-drift / fiber-qc.

Positions come from an existing fiber-cpos cluster table for the curated stage when present; with
--relocalize (and a raw .spk/.fil) the curated units are re-localized here.  Otherwise positions are left
zero and a notice is printed -- templates, amplitudes and per-chunk structure are still rebuilt, but run
fiber-cpos on the curated .clu for positions.
"""
import argparse

import numpy as np

_LP = "\u25b8 fiber-refit"
def _log(m=""): print(f"{_LP} \u00b7 {m}" if m else _LP)
def _det(k, v, w=10): print(f"{' ' * (len(_LP) + 3)}{k:<{w}} {v}")

try:
    from . import (fiber_intrachunk as ic, fiber_lib as fl, neuro_io as nio,
                   session_yaml as sy, fiber_cpos as cp)
except ImportError:                                              # script / direct execution
    import fiber_intrachunk as ic, fiber_lib as fl, neuro_io as nio, session_yaml as sy, fiber_cpos as cp


def refit_units(spkD, cur_clu, res_s, pos, peak, nsamp, *, chunk_min=12.0, min_n=ic.DEFAULT_MIN_N,
                reserve=(0, 1), feats="cfiber", sig_cap=None):
    """Rebuild per-(curated-unit x chunk) signatures from CURATED labels with NO regrouping.

    spkD     : (nspk, nsamp, nchan) stderiv spike memmap/array, aligned to res
    cur_clu  : curated per-spike labels
    res_s    : per-spike times in seconds
    pos      : {curated_unit_id: (x0, y0, z0, A)} from fiber-cpos (or {} -> zeros)
    Returns (units, sig, stride): units is aggregate_units' table plus 'curated_unit' and 'chunk_idx'
    decoding each per-chunk signature back to the curated unit it came from."""
    cur = np.asarray(cur_clu, np.int64)
    res_s = np.asarray(res_s, float)
    chid = (res_s / 60.0 / chunk_min).astype(int)
    stride = int(chid.max()) + 1 if chid.size else 1
    hi = max(reserve) if reserve else 0
    is_real = cur > hi
    comp = np.where(is_real, cur * stride + chid, cur)          # split real units by chunk; keep reserve as-is
    pos_comp = {int(c): pos.get(int(c) // stride, (0.0, 0.0, 0.0, 0.0)) for c in np.unique(comp[is_real])}
    m = fl.build_masks(nsamp, peak)
    sig = ic.build_signatures(spkD, comp, res_s, pos_comp, chunk_min=chunk_min, min_n=min_n,
                              reserve=tuple(reserve), feats=feats, peak=peak,
                              realign_lohi=(m.realign_lo, m.realign_hi), sig_cap=sig_cap)
    label = np.arange(len(sig["ids"]))                          # identity: each signed (unit,chunk) IS a unit
    units = ic.aggregate_units(sig, label)
    comp_ids = np.asarray(sig["ids"])
    units["curated_unit"] = (comp_ids // stride).astype(int)
    units["chunk_idx"] = (comp_ids % stride).astype(int)
    units["members"] = [np.array([cu], int) for cu in units["curated_unit"]]   # spikes recoverable from the
    return units, sig, stride                                                  # curated .clu (id) + chunk_idx


def _positions_from_cpos(base, elec, method, stage):
    """{curated_unit: (x0,y0,z0,A)} from a fiber-cpos .clusters.npz, or None if absent."""
    import os
    path = nio.session_path(base, "cpos", elec, variant=method, tag=stage) + ".clusters.npz"
    if not os.path.isfile(path):
        return None
    z = np.load(path)
    return {int(c): (float(x), float(y), float(zz), float(A))
            for c, x, y, zz, A in zip(z["clu"], z["x0"], z["y0"], z["z0"], z["A"])}


def main():
    ap = argparse.ArgumentParser(
        description="Refit the fiber model (per-unit, per-chunk templates / positions / signatures) to a "
                    "MANUALLY CURATED .clu, taking the curator's grouping as final (no re-merge/split). "
                    "Writes a refreshed <...>.units.npz for fiber-link / fiber-drift / fiber-qc.")
    sy.add_session_args(ap)
    nio.add_clu_args(ap, stage_default="curated", method_help="feature space of the curated clu", stage_help="stage tag of the curated .clu to refit (e.g. 'curated')", in_clu_help="explicit curated .clu path")
    ap.add_argument("--cpos-method", default=None, help="cpos method for positions (default: --clu-method)")
    ap.add_argument("--cpos-stage", default=None, help="cpos stage for positions (default: --variant)")
    ap.add_argument("--relocalize", action="store_true",
                    help="re-localize the curated units here from raw .spk/.fil instead of reading a cpos table")
    ap.add_argument("--spk-method", default="standard", help="raw .spk method for --relocalize")
    ap.add_argument("--gate", choices=["cfiber", "wave", "none"], default="cfiber",
                    help="shape descriptor to attach to each signature (default cfiber)")
    ap.add_argument("--chunk-minutes", "--chunk-min", type=float, default=12.0)
    ap.add_argument("--min-n", type=int, default=ic.DEFAULT_MIN_N, help="min spikes for a per-chunk signature")
    ap.add_argument("--out-stage", default=None, help="stage tag for the refit units (default '<variant>_refit')")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    nchan, nsamp, peak, sr = cfg["nchan"], cfg["nsamp"], cfg["peak"], cfg["sr"]

    res = nio.read_res(base, elec)
    _, clu = nio.resolve_clu(a, base, elec, n_spikes=len(res))
    spkD, _ = nio.open_spkD(base, elec, nsamp, nchan)
    assert spkD.shape[0] == len(res) == len(clu), \
        ".res %d / .clu %d / .spk %d mismatch" % (len(res), len(clu), spkD.shape[0])
    res_s = res.astype(float) / sr
    cur_units = [int(c) for c in np.unique(clu) if c > 1]
    _log(f"curated clu: {len(res):,} spikes · {len(cur_units):,} units (reserve 0/1 excluded)")

    cpos_method = a.cpos_method or a.clu_method
    cpos_stage = a.cpos_stage or a.variant
    pos = _positions_from_cpos(base, elec, cpos_method, cpos_stage)
    if a.relocalize or pos is None:
        try:
            xy = fl.channel_xy(cfg) if hasattr(fl, "channel_xy") else None
        except Exception:
            xy = None
        if a.relocalize and xy is not None:
            spk_raw, _ = nio.open_spk(base, elec, nsamp, nchan, prefer=[a.spk_method]) \
                if hasattr(nio, "open_spk") else (None, None)
            if spk_raw is not None:
                per = cp.localize_clusters(cp.spk_extractor(spk_raw), clu.astype(np.int64), xy)
                pos = {int(r["clu"]): (r["x0"], r["y0"], r["z0"], r["A"]) for r in per} \
                    if isinstance(per, list) else None
        if pos is None:
            _log(f"note: no cpos positions for stage {cpos_method}/{cpos_stage}; positions left zero")
            _det("fix", f"run  fiber-cpos {a.session} {elec} --clu-stage {a.variant}  then re-run "
                        "(or pass --relocalize with a raw .spk)")
            pos = {}

    feats = None if a.gate == "none" else ("cfiber" if a.gate == "cfiber" else "wave")
    units, sig, stride = refit_units(spkD, clu.astype(np.int64), res_s, pos, peak, nsamp,
                                     chunk_min=a.chunk_minutes, min_n=a.min_n, feats=feats)

    n_sigs = len(units["unit"])
    per_unit_chunks = {}
    for u in units["curated_unit"]:
        per_unit_chunks[int(u)] = per_unit_chunks.get(int(u), 0) + 1
    spanning = sum(1 for v in per_unit_chunks.values() if v > 1)
    _log(f"{len(per_unit_chunks):,} curated units → {n_sigs:,} per-chunk signatures over "
         f"{len(np.unique(units['chunk']))} chunks")
    _det("spanning", f"{spanning:,} units span >1 chunk")
    if pos:
        z = units["z0"]
        _det("depth z0", f"{z.min():.0f} .. {z.max():.0f} um")

    out_stage = a.out_stage or ("%s_refit" % a.variant if a.variant else "refit")
    out_base = nio.session_path(base, "clu", elec, variant=a.clu_method, tag=out_stage)
    upath = out_base + ".units.npz"
    np.savez(upath, **{k: v for k, v in units.items() if k != "members"},
             members=np.array(units["members"], dtype=object))
    _log("wrote")
    _det("units", f"{upath}   ({n_sigs:,} unit-chunk signatures keyed on curated labels)")


if __name__ == "__main__":
    main()
