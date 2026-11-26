"""fiber-unmix: the spikes outside the kept units, explained on the .fil.

A synthetic 150 s group whose .fil really contains the spikes: three kept units
(A, B, C), one cell D that has no cluster, and a noise cluster built from known
cases --
  one          isolated spikes of A/B/C that the sort left in the noise
  collision    a kept unit with a second kept unit 8-18 samples later that was
               never detected (hidden), or ~22 samples later and detected
  partial      a kept unit with a hidden spike of D 9-14 samples later
  shadow       a second detection 8-10 samples down the flank of a kept spike
  unexplained  D alone
Checked: each case is named as what it is (rates printed; the floors are well
under what the fixture gives), the units named are the planted ones, kept
spikes are never relabelled, --one-as assign moves a spike only into its own
unit, the manifest and the .eap say what the .clu says, and --skip-marked
leaves explained rows alone.

NEGATIVE CONTROL: the same noise rows with the labels of A/B/C permuted onto
detections of pure noise (no spike under them) -- nothing is called `one` or
`collision` beyond 2 %.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from fiber_kit import fiber_deconv as fd          # noqa: E402
from fiber_kit import fiber_unmix as fu           # noqa: E402
from fiber_kit import neuro_io as nio             # noqa: E402
from test_deconv import NSAMP, PEAK, SR, C, noise, place, raw_template   # noqa: E402

A, B, CC, D = 2, 3, 4, 9          # cluster ids of the kept units; D never gets one
SHAPES = {A: (40, 1.0, 420), B: (100, 0.9, 340), CC: (130, 0.55, 300), D: (75, 1.5, 260)}


def build(rng, dur_s=150.0):
    """(fil int16, res, clu, truth) -- truth: dict of per-row arrays kind / k1 / k2."""
    dur = int(dur_s * SR)
    fil = noise(rng, dur, 14.0)
    fil0 = fil.copy()                                 # the same recording WITHOUT the hidden partners
    T = {k: raw_template(*v) for k, v in SHAPES.items()}
    ev = []                                           # (detection time, cluster, kind, k1, k2)
    t = 2000.0
    gap = lambda: rng.uniform(4 * NSAMP, 9 * NSAMP)   # noqa: E731  -- events never share a window
    i = 0
    while t < dur - 3000:
        i += 1
        case = rng.choice(["kept", "one", "hid", "det", "partial", "shadow", "D"],
                          p=[0.62, 0.12, 0.07, 0.04, 0.05, 0.04, 0.06])
        k = int(rng.choice([A, B, CC]))
        tf = t + rng.uniform(0, 1)
        a = rng.normal(1.0, 0.06)
        if case == "kept":
            place(fil, T[k], tf, a); place(fil0, T[k], tf, a); ev.append((tf, k, fu.NONE, k, -1))
        elif case == "one":
            place(fil, T[k], tf, a); place(fil0, T[k], tf, a); ev.append((tf, 1, fu.ONE, k, -1))
        elif case in ("hid", "det"):
            k2 = int(rng.choice([u for u in (A, B, CC) if u != k]))
            lag = rng.uniform(8, 18) if case == "hid" else rng.uniform(21, 24)
            a2 = rng.normal(0.9, 0.05)
            place(fil, T[k], tf, a); place(fil, T[k2], tf + lag, a2)
            place(fil0, T[k], tf, a)
            if case == "det":
                place(fil0, T[k2], tf + lag, a2)
            ev.append((tf, 1, fu.COLLISION, k, k2))
            if case == "det":
                ev.append((tf + lag, k2, fu.NONE, k2, -1))      # the partner is a kept spike
        elif case == "partial":
            place(fil, T[k], tf, a); place(fil, T[D], tf + rng.uniform(9, 14), rng.normal(1.0, 0.05))
            ev.append((tf, 1, fu.PARTIAL, k, -1))
        elif case == "shadow":
            place(fil, T[k], tf, a); ev.append((tf, k, fu.NONE, k, -1))
            ev.append((tf + rng.uniform(8, 10), 1, fu.SHADOW, -1, k))
        else:
            place(fil, T[D], tf, a); ev.append((tf, 1, fu.UNEXPLAINED, -1, -1))
        t += gap()
    ev.sort(key=lambda e: e[0])
    res = np.array([int(round(e[0])) for e in ev], np.int64)
    clu = np.array([e[1] for e in ev], np.int64)
    truth = dict(kind=np.array([e[2] for e in ev]), k1=np.array([e[3] for e in ev]),
                 k2=np.array([e[4] for e in ev]), fil0=fil0)
    return np.clip(np.round(fil), -32768, 32767).astype(np.int16), res, clu, truth


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(3)
    fil, res, clu, truth = build(rng)
    read_raw = fd.memmap_reader(fil, np.arange(C))
    noise_rows = clu == 1
    u = fu.unmix(read_raw, res, clu, NSAMP, PEAK, SR, n_samples=fil.shape[0], block_min=1.0)
    check("three blocks of one minute, each with three templates",
          len(u["blocks"]) == 3 and all(b["units"] == 3 for b in u["blocks"]))
    check("kept spikes are named as their own unit by the pursuit (>= 97 % per block)",
          all(b["agree"] >= 0.97 for b in u["blocks"]))
    check("only the noise rows were pursued", np.array_equal(u["kind"] != fu.NONE, noise_rows))

    def rate(kind, cond=None):
        m = truth["kind"] == kind
        hit = u["kind"][m] == kind
        if cond is not None:
            hit &= cond[m]
        return float(hit.mean()), int(m.sum())

    r, n = rate(fu.ONE, u["k1"] == truth["k1"])
    print(f"      one: {r:.1%} of {n} named `one` with the planted unit")
    check("one-unit spikes: >= 90 % named `one` with their unit", r >= 0.90)
    pair = (u["k1"] == truth["k1"]) & (u["k2"] == truth["k2"])
    r, n = rate(fu.COLLISION, pair)
    print(f"      collision: {r:.1%} of {n} named `collision` with both planted units")
    check("collisions: >= 80 % named `collision` with both units", r >= 0.80)
    m = (truth["kind"] == fu.COLLISION) & (u["kind"] == fu.COLLISION) & pair
    lag_true = np.array([res[i + 1] - res[i] if (i + 1 < len(res) and res[i + 1] - res[i] < 26 and clu[i + 1] != 1)
                         else -1 for i in range(len(res))])
    det = m & (lag_true > 0)
    check("a detected partner is flagged detected, a hidden one hidden",
          det.sum() > 5 and (u["k2_det"][det] == 1).mean() >= 0.95
          and (u["k2_det"][m & (lag_true < 0)] == 0).mean() >= 0.95)
    check("the partner's lag is the planted one within a sample (detected partners)",
          np.median(np.abs((u["dt2"] - u["dt1"])[det] - lag_true[det])) <= 1.0)
    r, n = rate(fu.UNEXPLAINED)
    rp = float(np.isin(u["kind"][truth["kind"] == fu.UNEXPLAINED], (fu.UNEXPLAINED, fu.PARTIAL)).mean())
    print(f"      a cell without a template: {r:.1%} of {n} unexplained, {rp:.1%} unexplained or partial")
    check("a cell without a template stays in the noise (>= 90 % unexplained or partial)", rp >= 0.90)
    r, n = rate(fu.PARTIAL, u["k1"] == truth["k1"])
    print(f"      partial: {r:.1%} of {n} named `partial` with the planted unit")
    check("unit + unknown cell: >= 60 % named `partial` with the unit", r >= 0.60)
    stay = float(np.isin(u["kind"][truth["kind"] == fu.PARTIAL], (fu.PARTIAL, fu.UNEXPLAINED)).mean())
    check("unit + unknown cell: >= 85 % stay in the noise", stay >= 0.85)
    r, n = rate(fu.SHADOW, u["k2"] == truth["k2"])
    print(f"      shadow: {r:.1%} of {n} named `shadow` of the planted neighbour")
    check("second detections of a flank: >= 80 % named `shadow` of that unit", r >= 0.80)

    # residual add-back: a row with every OTHER placement of the pursuit subtracted in the raw signal
    ua = fu.unmix(read_raw, res, clu, NSAMP, PEAK, SR, n_samples=fil.shape[0], block_min=1.0, addback=True)
    check("add-back: the kinds and units are those of the run without it",
          all(np.array_equal(ua[k], u[k]) for k in ("kind", "k1", "k2")))
    ar, aw = ua["ab_rows"], ua["ab_waves"].astype(np.float64)
    check("add-back: rows ascending, one record of the .spk window each",
          len(ar) > 50 and bool((np.diff(ar) > 0).all()) and aw.shape[1:] == (NSAMP, C))
    cut = lambda sig, r: sig[int(res[r]) - PEAK:int(res[r]) - PEAK + NSAMP].astype(np.float64)   # noqa: E731
    # hidden-partner collisions named with both units: the record before the partner is known (fil0)
    lag_any = np.array([res[i + 1] - res[i] if i + 1 < len(res) else 10**9 for i in range(len(res))])
    hid = np.flatnonzero((truth["kind"] == fu.COLLISION) & (u["kind"] == fu.COLLISION)
                         & (u["k1"] == truth["k1"]) & (u["k2"] == truth["k2"]) & (lag_any > 26))
    pos = {int(r): i for i, r in enumerate(ar)}
    have = [r for r in hid if int(r) in pos]
    check("add-back: (nearly) every collision with a hidden partner has a record", len(have) >= 0.95 * len(hid) > 20)
    e_raw = np.array([((cut(fil, r) - cut(truth["fil0"], r)) ** 2).sum() for r in have])
    e_ab = np.array([((aw[pos[int(r)]] - cut(truth["fil0"], r)) ** 2).sum() for r in have])
    print(f"      add-back: error against the record before the partner, as a share of the raw row's: "
          f"median {np.median(e_ab / e_raw):.3f}, worst {np.max(e_ab / e_raw):.3f} ({len(have)} hidden-partner collisions)")
    check("add-back: the partner is gone -- error <= 5 % of the raw row's (median), <= 30 % (worst)",
          np.median(e_ab / e_raw) <= 0.05 and np.max(e_ab / e_raw) <= 0.30)
    one_rows = np.flatnonzero((truth["kind"] == fu.ONE) | ((truth["kind"] == fu.NONE) & (lag_any > 60)
                              & (np.r_[10**9, np.diff(res)] > 60)))
    check("add-back: NEGATIVE CONTROL: a spike with nothing else in its window gets no record",
          np.isin(one_rows, ar).mean() <= 0.02)

    # a .res with rows out of time order (what a realign leaves): the same answers, row for row
    sw = np.arange(len(res)); j = np.arange(50, len(res) - 1, 97)
    sw[j], sw[j + 1] = sw[j + 1].copy(), sw[j].copy()
    us = fu.unmix(read_raw, res[sw], clu[sw], NSAMP, PEAK, SR, n_samples=fil.shape[0], block_min=1.0)
    check("a .res with 190 rows out of order: the same kinds and units, row for row",
          np.any(np.diff(res[sw]) < 0) and all(np.array_equal(us[k], u[k][sw]) for k in ("kind", "k1", "k2")))
    usa = fu.unmix(read_raw, res[sw], clu[sw], NSAMP, PEAK, SR, n_samples=fil.shape[0], block_min=1.0, addback=True)
    inv = np.empty(len(sw), np.int64); inv[sw] = np.arange(len(sw))
    q = np.argsort(inv[ar], kind="stable")
    check("add-back with rows out of order: the same records, under the rows' new numbers",
          bool((np.diff(usa["ab_rows"]) > 0).all()) and np.array_equal(usa["ab_rows"], inv[ar][q])
          and np.array_equal(usa["ab_waves"], ua["ab_waves"][q]))

    # relabel
    new, ids = fu.relabel(clu, u)
    check("relabel: kept spikes keep their label", np.array_equal(new[~noise_rows], clu[~noise_rows]))
    check("relabel: one / collision / shadow get three new ids, the rest stay 1",
          ids["one"] == 5 and set(np.unique(new[noise_rows])) <= {1, 5, 6, 7}
          and np.array_equal(new == 1, np.isin(u["kind"], (fu.PARTIAL, fu.UNEXPLAINED))))
    asg, _ = fu.relabel(clu, u, one_as="assign")
    moved = noise_rows & np.isin(asg, (A, B, CC))
    right = float((asg[moved] == truth["k1"][moved]).mean())
    print(f"      --one-as assign: {int(moved.sum())} moved into a unit, {right:.1%} into their own")
    check("--one-as assign: moves many, and >= 99 % into their own unit", moved.sum() >= 100 and right >= 0.99)
    per, pid = fu.relabel(clu, u, one_as="units")
    check("--one-as units: one new cluster per unit",
          sorted(pid["units"]) == [A, B, CC] and all((u["k1"][per == c] == k).all() for k, c in pid["units"].items()))

    # NEGATIVE CONTROL: detections on pure noise
    fake = np.sort(rng.choice(np.arange(3000, fil.shape[0] - 3000), 400, replace=False))
    fake = fake[np.abs(fake[:, None] - res[None, :]).min(1) > 2 * NSAMP]
    res2 = np.r_[res[~noise_rows], fake]; clu2 = np.r_[clu[~noise_rows], np.ones(len(fake), np.int64)]
    o = np.argsort(res2, kind="stable"); res2, clu2 = res2[o], clu2[o]
    u2 = fu.unmix(read_raw, res2, clu2, NSAMP, PEAK, SR, n_samples=fil.shape[0], block_min=1.0)
    bad = float(np.isin(u2["kind"][clu2 == 1], (fu.ONE, fu.COLLISION, fu.PARTIAL)).mean())
    print(f"      negative control: {bad:.1%} of {int((clu2 == 1).sum())} noise-only detections given a unit")
    check("negative control: detections with no spike under them get a unit <= 2 % of the time",
          (clu2 == 1).sum() >= 100 and bad <= 0.02)

    # end to end: the command, its files, --skip-marked
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "syn")
        fil.tofile(f"{base}.fil"); res.tofile(f"{base}.res.1")
        nio.write_clu(base, 1, clu, variant="stderiv", tag="clean")
        with open(f"{base}.yaml", "w") as f:
            f.write("acquisitionSystem: {nChannels: 8, samplingRate: 32552, nBits: 16}\n"
                    "spikeDetection:\n  channelGroups:\n"
                    "    - channels: [0,1,2,3,4,5,6,7]\n      nSamples: 42\n      peakSampleIndex: 21\n"
                    "      nFeatures: 3\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

        def run(extra):
            return subprocess.run([sys.executable, "-m", "fiber_kit.fiber_unmix", base, "1", "--clu-method", "stderiv",
                                   "--block-min", "1"] + extra, capture_output=True, text=True, env=env, timeout=900)
        r1 = run(["--clu-stage", "clean"])
        if r1.returncode != 0:
            print((r1.stdout + r1.stderr)[-3000:])
        check("fiber-unmix exits 0", r1.returncode == 0)
        _, got = nio.read_clu_file(f"{base}.clu.stderiv.1.clean_unmix")
        asu, _ = fu.relabel(clu, u, collision_as="unit")
        check("the written .clu is relabel() of the same run (collisions into their first unit)",
              np.array_equal(got, asu))
        cu = noise_rows & (u["kind"] == fu.COLLISION) & np.isin(asu, (A, B, CC))
        check("collisions moved into a unit went into the unit on the detection (>= 99 %)",
              cu.sum() >= 500 and (asu[cu] == truth["k1"][cu]).mean() >= 0.99)
        with open(f"{base}.unmix.stderiv.1.clean_unmix.tsv") as fh:
            rows = [ln.rstrip("\n").split("\t") for ln in fh]
        check("manifest: header + one row per pursued detection",
              rows[0] == fu.MANIFEST_COLS and len(rows) - 1 == int(noise_rows.sum()))
        kinds = {int(rw[0]): rw[2] for rw in rows[1:]}
        check("manifest kinds are the result's", all(kinds[i] == fu.KIND_NAMES[int(u["kind"][i])]
                                                     for i in np.flatnonzero(noise_rows)))
        e = nio.read_eap(base, 1, tag="clean_unmix"); tcl = nio.read_tcl(base, 1)
        col = {int(x["provenance_clu"]): int(x["col"]) for x in tcl["entries"] if x["status"] == "active"}
        cells = np.asarray(e["cells"])
        check(".eap: one row per spike, a class per kept unit", e["ok"] and e["nSpikes"] == len(res)
              and sorted(col) == [A, B, CC])
        kept = ~noise_rows
        check(".eap mask: every kept spike holds its own class at offset 0 and nothing else",
              all((cells[kept & (clu == k), c] == 0).all() for k, c in col.items())
              and ((cells[kept] != nio.EAP_ABSENT).sum(1) == 1).all())
        cm = np.flatnonzero(u["kind"] == fu.COLLISION)
        check(".eap: a collision row holds both classes at their offsets",
              all(cells[i, col[int(u["k1"][i])]] == int(np.rint(u["dt1"][i]))
                  and cells[i, col[int(u["k2"][i])]] == int(np.rint(u["dt2"][i])) for i in cm))
        un = u["kind"] == fu.UNEXPLAINED
        check(".eap: an unexplained row holds no class", (cells[un] == nio.EAP_ABSENT).all())
        # the .spkc: with the standard .spk present; the narrower .eap (resolved) gives the same file
        idx = res[:, None] + np.arange(-PEAK, NSAMP - PEAK)[None, :]
        fil[idx].astype(np.int16).tofile(f"{base}.spk.standard.1")
        ra = run(["--clu-stage", "clean", "--out-tag", "ua"])
        rr = run(["--clu-stage", "clean", "--out-tag", "ur", "--eap", "resolved"])
        sa = nio.read_spkc(base, 1, variant="standard", tag="ua"); sr = nio.read_spkc(base, 1, variant="standard", tag="ur")
        kinds = u["kind"][sa["rows"]] if sa["rows"].size else np.zeros(0, int)
        ncoll = int((u["kind"] == fu.COLLISION).sum())
        print(f"      .spkc: {sa['rows'].size} rows corrected -- {(kinds == fu.COLLISION).sum()} of the {ncoll} collisions, "
              f"{(kinds == fu.PARTIAL).sum()} partial rows that also hold a second kept unit")
        check("with the .spk present the stage's .spkc is written: >= 95 % of the collisions are in it",
              ra.returncode == 0 and sa["ok"] and (kinds == fu.COLLISION).sum() >= 0.95 * ncoll)
        check("... and nothing but rows whose .eap row names two units (collisions, and partial rows with a second unit)",
              np.isin(kinds, (fu.COLLISION, fu.PARTIAL)).all())
        check("--eap resolved: a narrower .eap, the SAME .spkc (each unit's own spikes are still named)",
              rr.returncode == 0 and sr["ok"] and np.array_equal(sa["rows"], sr["rows"])
              and np.array_equal(sa["waves"], sr["waves"])
              and nio.read_eap(base, 1, tag="ur")["nClasses"] == nio.read_eap(base, 1, tag="ua")["nClasses"])

        # a second run under the same tag, the clustering it reads renumbered (A and B swapped):
        # the stage's .eap is this run's alone -- no kept row names two classes
        swapped = np.where(clu == A, B, np.where(clu == B, A, clu))
        nio.write_clu(base, 1, swapped, variant="stderiv", tag="clean")
        rs = run(["--clu-stage", "clean"])
        es = np.asarray(nio.read_eap(base, 1, tag="clean_unmix")["cells"])
        check("run again under the same tag after the input changed: every kept spike names one class, its own",
              rs.returncode == 0 and ((es[kept] != nio.EAP_ABSENT).sum(1) == 1).all()
              and all((es[kept & (swapped == k), c] == 0).all() for k, c in col.items()))
        nio.write_clu(base, 1, clu, variant="stderiv", tag="clean")
        run(["--clu-stage", "clean"])                       # back to the first run's files for what follows

        r2 = run(["--clu-stage", "clean_unmix", "--skip-marked", "--eap", "off"])
        check("--skip-marked on the result: exits 0 and pursues only the rows without a class",
              r2.returncode == 0 and f"{int(((asu == 1) & (cells != nio.EAP_ABSENT).any(1)).sum()):,} noise rows already"
              in r2.stderr)
        rab = run(["--clu-stage", "clean", "--out-tag", "ab", "--spkc", "2", "--children", "1"])
        _, flat = nio.read_clu_file(f"{base}.clu.stderiv.1.ab")
        (_, hp), (_, hc) = (nio.read_clu_file(f"{base}.{t}.stderiv.1.ab_h") for t in ("clu", "clc"))
        onerow = u["kind"] == fu.ONE
        check("--children 1: the triple under <out>_h -- one-unit spikes in their unit, as a child of their own",
              os.path.exists(f"{base}.clp.stderiv.1.ab_h") and np.array_equal(hp[onerow], u["k1"][onerow])
              and hc[onerow].min() >= 20000 and np.array_equal(hp[~onerow], flat[~onerow])
              and np.array_equal(hc[~onerow], np.where(flat[~onerow] >= 2, flat[~onerow] + 10000, flat[~onerow])))
        sab = nio.read_spkc(base, 1, variant="standard", tag="ab")
        check("--spkc 2: the add-back records as the stage's standard .spkc, the ones unmix() returned",
              rab.returncode == 0 and sab["ok"] and np.array_equal(sab["rows"], ua["ab_rows"])
              and np.array_equal(sab["waves"], ua["ab_waves"]) and "residual add-back" in rab.stderr)
        check("--spkc 2: no .spkc of the method .spk, and the first run's files are untouched",
              not os.path.exists(f"{base}.spkc.stderiv.1.ab") and os.path.exists(f"{base}.clu.stderiv.1.clean_unmix"))
        os.remove(f"{base}.fil")
        r3 = run(["--clu-stage", "clean"])
        check("without the .fil: refuses, naming the file", r3.returncode != 0 and "syn.fil not found" in r3.stderr)

    print(f"test_unmix: {ok} checks passed")


if __name__ == "__main__":
    main()
