#!/usr/bin/env python3
"""Per-group quality-control report for a sort.

Computes, for every unit in a group's .clu, the metrics a curator actually triages on -- spike count and
rate, refractory (ISI) violation, waveform SNR and amplitude, and presence across the session -- and
renders them as an interactive HoloViz (Bokeh) report: a rate-vs-SNR scatter coloured by contamination, an
ISI-violation bar, a presence heatmap (units x session time, which also reads as a stability/drift view),
and a sortable metrics table.  Optionally folds in fiber-contam's two-cell flag and a fiber-score
comparison against a ground-truth .clu.

The metric layer (compute_metrics, presence_matrix) is pure-numpy and import-clean; only the report
rendering needs holoviews/bokeh.  A CSV of the metrics is always written, so the tool is useful even
where HoloViz is not installed.
"""
import argparse

import numpy as np

try:
    from . import (neuro_io as nio, session_yaml as sy, fiber_ccg as ccg,
                   fiber_session as fs, fiber_score as fsc)
except ImportError:                                              # script / direct execution
    import neuro_io as nio, session_yaml as sy, fiber_ccg as ccg, fiber_session as fs, fiber_score as fsc


# ───────────────────────────── metric core (pure numpy) ─────────────────────────────
def unit_snr_amp(waves, peak):
    """(snr, amplitude) for a spike stack (n, nsamp, nchan).  amplitude = peak-to-peak of the mean template
    on its strongest channel; snr = that amplitude over a robust noise estimate (residual MAD) there."""
    W = np.asarray(waves, float)
    tmpl = W.mean(0)
    ptp = tmpl.max(0) - tmpl.min(0)                              # per channel
    ch = int(np.argmax(ptp))
    amp = float(ptp[ch])
    resid = W[:, :, ch] - tmpl[:, ch]
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    return amp / (2.0 * noise), amp


def presence_matrix(times_per_unit, t0, t1, bins=120):
    """(matrix, edges): matrix[u, b] = spike count of unit u in time-bin b (for the heatmap)."""
    edges = np.linspace(t0, t1, bins + 1)
    M = np.zeros((len(times_per_unit), bins))
    for u, t in enumerate(times_per_unit):
        if len(t):
            M[u], _ = np.histogram(t, bins=edges)
    return M, edges


def compute_metrics(spkD, clu, res_s, sr, peak, *, refrac_ms=1.5, censor_ms=0.3,
                    presence_bins=120, min_spikes=20):
    """Per-unit QC rows + a presence matrix.  res_s in seconds; spkD aligned to clu/res."""
    refrac = ccg.refrac_samples(refrac_ms, sr); censor = ccg.refrac_samples(censor_ms, sr)
    res_samp = (np.asarray(res_s) * sr)
    t0, t1 = float(res_s.min()), float(res_s.max())
    dur = max(t1 - t0, 1e-9)
    rows, times_per_unit = [], []
    for cid in np.unique(clu):
        if cid <= 1:                                            # 0 artefact, 1 noise/unsorted
            continue
        idx = np.flatnonzero(clu == cid)
        if idx.size < min_spikes:
            continue
        t = np.sort(res_s[idx])
        isi = ccg.isi_violation_fraction(res_samp[idx], refrac, censor)
        snr, amp = unit_snr_amp(spkD[idx], peak)
        edges = np.linspace(t0, t1, presence_bins + 1)
        occ = np.histogram(t, bins=edges)[0]
        rows.append(dict(cluster=int(cid), n=int(idx.size), rate_hz=idx.size / dur,
                         isi_viol=float(isi), snr=float(snr), amplitude=float(amp),
                         presence=float((occ > 0).mean())))
        times_per_unit.append(t)
    M, edges = presence_matrix(times_per_unit, t0, t1, bins=presence_bins)
    return rows, M, edges


def flag_rows(rows, *, isi_thr=0.01, snr_thr=4.0, presence_thr=0.5):
    """Attach a 'flags' string to each row for the obvious triage cases."""
    for r in rows:
        f = []
        if r["isi_viol"] > isi_thr:
            f.append("ISI")
        if r["snr"] < snr_thr:
            f.append("lowSNR")
        if r["presence"] < presence_thr:
            f.append("intermittent")
        r["flags"] = ",".join(f)
    return rows


# ───────────────────────────── HoloViz report ─────────────────────────────
def render_report(rows, M, edges, meta, out_html, gt_summary=None):
    """Render the interactive HoloViz/Bokeh report to a standalone HTML file."""
    try:
        import holoviews as hv
        from holoviews import opts
    except ImportError as e:
        raise RuntimeError("the HoloViz report needs holoviews + bokeh (pip install holoviews bokeh); "
                           "the metrics CSV was still written") from e
    hv.extension("bokeh")
    clusters = [r["cluster"] for r in rows]
    rate = [r["rate_hz"] for r in rows]; snr = [r["snr"] for r in rows]
    isi = [r["isi_viol"] for r in rows]; nsp = [r["n"] for r in rows]

    scatter = hv.Points((rate, snr, isi, clusters, nsp),
                        kdims=["rate (Hz)", "SNR"], vdims=["ISI viol", "cluster", "n"]).opts(
        opts.Points(color="ISI viol", cmap="viridis_r", size=9, colorbar=True, logx=True,
                    tools=["hover"], width=440, height=340, title="rate vs SNR (colour = ISI violation)"))
    order = np.argsort(isi)[::-1]
    bars = hv.Bars([(str(clusters[i]), isi[i]) for i in order], "cluster", "ISI viol").opts(
        opts.Bars(width=440, height=340, xrotation=90, color="ISI viol", cmap="viridis_r",
                  colorbar=False, tools=["hover"], title="ISI-violation fraction by unit"))
    t_min = edges / 60.0
    heat = hv.Image((0.5 * (t_min[:-1] + t_min[1:]), np.arange(len(rows)), M),
                    kdims=["session time (min)", "unit row"], vdims=["spikes"]).opts(
        opts.Image(cmap="magma", colorbar=True, width=900, height=320, tools=["hover"],
                   title="presence / stability — spikes per unit over session time"))
    table = hv.Table([(r["cluster"], r["n"], round(r["rate_hz"], 3), round(r["isi_viol"], 4),
                       round(r["snr"], 1), round(r["amplitude"], 1), round(r["presence"], 2),
                       r.get("flags", "")) for r in rows],
                     ["cluster", "n", "rate", "ISI", "SNR", "amp", "presence", "flags"]).opts(
        opts.Table(width=900, height=320))

    title = "fiber-qc — %s group %s : %d units, %d spikes, %.0f min" % (
        meta["session"], meta["group"], len(rows), meta["n_spikes"], meta["minutes"])
    panels = [hv.Div("<h2>%s</h2>" % title), (scatter + bars).cols(2), heat, table]
    if gt_summary:
        panels.insert(1, hv.Div("<pre>%s</pre>" % gt_summary))
    layout = hv.Layout(panels).cols(1)
    hv.save(layout, out_html, backend="bokeh")
    return out_html


def _write_csv(rows, path):
    cols = ["cluster", "n", "rate_hz", "isi_viol", "snr", "amplitude", "presence", "flags"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Per-group QC report (rate, ISI violation, SNR, amplitude, presence) rendered as an "
                    "interactive HoloViz/Bokeh HTML, with a metrics CSV.")
    sy.add_session_args(ap)
    ap.add_argument("--clu-method", default="stderiv")
    ap.add_argument("--clu-stage", "--variant", dest="variant", default="refine",
                    help="post-fiber stage tag at the end of the .clu name")
    ap.add_argument("--in-clu", default=None, help="explicit .clu path")
    ap.add_argument("--refrac-ms", type=float, default=1.5)
    ap.add_argument("--censor-ms", type=float, default=0.3)
    ap.add_argument("--presence-bins", type=int, default=120)
    ap.add_argument("--min-spikes", type=int, default=20)
    ap.add_argument("--isi-thr", type=float, default=0.01, help="ISI-violation fraction to flag (default 0.01)")
    ap.add_argument("--snr-thr", type=float, default=4.0, help="SNR below this is flagged lowSNR (default 4)")
    ap.add_argument("--presence-thr", type=float, default=0.5,
                    help="presence below this is flagged intermittent (lower for sparse data; default 0.5)")
    ap.add_argument("--gt-clu", default=None, help="ground-truth .clu to score against (fiber-score)")
    ap.add_argument("--gt-res", default=None, help=".res for the ground truth (timestamp alignment)")
    ap.add_argument("--out", default=None, help="output HTML path (default <base>.qc.<elec>.html)")
    a = ap.parse_args()

    cfg = sy.resolve_session_params(a.session, a.group, channels=a.channels, ntotal=a.ntotal,
                                    nchan=a.nchan, nsamp=a.nsamp, sr=a.sr)
    base = cfg["base"]; elec = a.group
    nchan, nsamp, peak, sr = cfg["nchan"], cfg["nsamp"], cfg["peak"], cfg["sr"]
    res = nio.read_res(base, elec)
    if a.in_clu:
        _, clu = nio.read_clu_file(a.in_clu, n_spikes=len(res))
    else:
        _, clu = nio.read_clu_at(base, elec, variant=a.clu_method, tag=a.variant, n_spikes=len(res))
    spkD, _ = nio.open_spkD(base, elec, nsamp, nchan)
    res_s = res.astype(float) / sr

    rows, M, edges = compute_metrics(spkD, clu, res_s, sr, peak, refrac_ms=a.refrac_ms,
                                     censor_ms=a.censor_ms, presence_bins=a.presence_bins,
                                     min_spikes=a.min_spikes)
    flag_rows(rows, isi_thr=a.isi_thr, snr_thr=a.snr_thr, presence_thr=a.presence_thr)
    n_flagged = sum(1 for r in rows if r["flags"])
    print("qc: %d units (%d flagged) over %.0f min" % (len(rows), n_flagged, (res_s.max() - res_s.min()) / 60))

    gt_summary = None
    if a.gt_clu:
        _, gt = nio.read_clu_file(a.gt_clu)
        if gt.size == len(res):
            s = fsc.score(clu, gt)
        elif a.gt_res:
            cb, gl, _ = fsc.align_by_res(clu, res, gt, nio.read_res_file(a.gt_res))
            s = fsc.score(cb, gl)
        else:
            s = None
            print("--gt-clu length differs from .res; pass --gt-res to align")
        if s is not None:
            gt_summary = fsc.format_report(s, top=6)
            print(gt_summary)

    out_csv = (a.out[:-5] if a.out and a.out.endswith(".html") else
               nio.session_path(base, "qc", elec, variant=a.clu_method, tag=a.variant)) + ".csv"
    _write_csv(rows, out_csv)
    print("wrote %s" % out_csv)
    out_html = a.out or (nio.session_path(base, "qc", elec, variant=a.clu_method, tag=a.variant) + ".html")
    meta = dict(session=a.session, group=elec, n_spikes=int(len(res)), minutes=(res_s.max() - res_s.min()) / 60)
    try:
        render_report(rows, M, edges, meta, out_html, gt_summary=gt_summary)
        print("wrote %s  (interactive HoloViz report)" % out_html)
    except RuntimeError as e:
        print("note: %s" % e)


if __name__ == "__main__":
    main()
