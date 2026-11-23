"""The sparse .spkc and the fit that fills it (fiber_spkc; neurosuite-3's decollide_fit.hpp).

Synthetic 52 x 8 records, two units A and B, noise sd 12:
  * .spkc round trip; the writer and the reader refuse an index that is not strictly ascending, a
    row beyond the .spk, a short file, another version;
  * shifted(): whole shifts move samples exactly and zero-fill; half a sample there and back is
    the template within 2 %;
  * fit_partner(): a partner planted at 11.4 samples and 0.8 of its template is found from a stored
    offset 3 samples too EARLY and from one too LATE (shift within 0.2 sample, amplitude within
    5 %); NEGATIVE CONTROLS: no partner in the row, and a partner at 0.15 of its template, are not
    accepted in 200 rows each;
  * home_clusters / split_row: the cluster a class is and a row's own class, after a renumbering too;
  * build(): 60 collisions planted among 900 rows are exactly the rows corrected, each served far
    closer to the record it had before; rows naming a partner that is not there are left out; the
    fitted offsets are the planted ones whatever the .eap said, and write_offsets returns them;
  * the fiber-spkc command writes the file and, with --write-offsets, the .eap.

PARITY with the C++ was measured outside this suite (fiber_spkc.py's header): identical accept
decisions, offsets and corrected int16 records on 2,000 random cases, and the same .spkc as
Klusters on a session with 40 planted collisions.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fiber_kit import fiber_spkc as fs            # noqa: E402
from fiber_kit import neuro_io as nio             # noqa: E402

NS, NC, PEAK = 52, 8, 26


def templ(depth, amp, width):
    s = np.arange(NS)[:, None]; c = np.arange(NC)[None, :]
    w = -np.exp(-0.5 * ((s - PEAK) / (4.0 * width)) ** 2) + 0.3 * np.exp(-0.5 * ((s - PEAK - 9.0 * width) / (5.0 * width)) ** 2)
    return (amp * w * (np.exp(-0.5 * ((c * 20.0 - depth) / 28.0) ** 2) + 0.05)).astype(np.float32)


def main():
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f"  ok  {name}")

    rng = np.random.default_rng(7)
    A, B = templ(40, 400, 1.0), templ(110, 320, 0.8)
    noise = lambda *shape: rng.normal(0, 12.0, shape).astype(np.float32)   # noqa: E731

    with tempfile.TemporaryDirectory() as td:
        # ── the format ──
        p = os.path.join(td, "a.spkc")
        rows = np.array([3, 17, 999]); wav = (np.arange(3 * 4 * 2).reshape(3, 4, 2) * 7 - 50).astype(np.int16)
        nio.write_spkc_file(p, rows, wav, 1000, group=8)
        g = nio.read_spkc_file(p)
        check(".spkc survives the round trip", g["ok"] and g["nSpikes"] == 1000 and g["group"] == 8
              and np.array_equal(g["rows"], rows) and np.array_equal(g["waves"], wav))
        check(".spkc size: 32 B header + index + records", os.path.getsize(p) == 32 + 3 * 8 + 3 * 8 * 2)
        for bad, why in (([17, 3, 999], "not ascending"), ([3, 3, 999], "a repeated row"), ([3, 17, 1000], "a row beyond the .spk")):
            try:
                nio.write_spkc_file(os.path.join(td, "bad.spkc"), bad, wav, 1000)
                refused = False
            except ValueError:
                refused = True
            check(f"write_spkc refuses an index with {why}", refused)
        raw = bytearray(open(p, "rb").read())
        open(os.path.join(td, "short.spkc"), "wb").write(raw[:-5])
        check("read_spkc rejects a short file", not nio.read_spkc_file(os.path.join(td, "short.spkc"))["ok"])
        raw2 = bytearray(raw); raw2[32:40] = (500).to_bytes(8, "little")
        open(os.path.join(td, "order.spkc"), "wb").write(raw2)
        check("read_spkc rejects an index that is not ascending", not nio.read_spkc_file(os.path.join(td, "order.spkc"))["ok"])
        raw[3] = nio.SPKC_VERSION[0] + 1
        open(os.path.join(td, "ver.spkc"), "wb").write(raw)
        check("read_spkc rejects a newer version", not nio.read_spkc_file(os.path.join(td, "ver.spkc"))["ok"])
        nio.write_spkc_file(p, [], np.zeros((0, 4, 2), np.int16), 10)
        check("an .spkc with no rows is valid", nio.read_spkc_file(p)["ok"] and nio.read_spkc_file(p)["rows"].size == 0)

        # ── shifted ──
        q = fs.shifted(A, 3.0)
        check("shifted: +3 moves every sample 3 later and zero-fills the start", np.array_equal(q[3:], A[:-3]) and not q[:3].any())
        q = fs.shifted(A, -2.0)
        check("shifted: -2 moves earlier and zero-fills the end", np.array_equal(q[:-2], A[2:]) and not q[-2:].any())
        back = fs.shifted(fs.shifted(A, 0.5), -0.5)
        check("shifted: half a sample there and back is the template within 2 %",
              ((back - A) ** 2).sum() < 0.02 ** 2 * (A ** 2).sum())

        # ── fit_partner ──
        own = 1.05 * A + noise(NS, NC)
        w = own + 0.8 * fs.shifted(B, 11.4)
        for stored in (8, 14):
            f = fs.fit_partner(w, A, B, stored)
            check(f"planted partner found from a stored offset of {stored}: shift within 0.2 sample, amplitude within 5 %",
                  f["accepted"] and abs(f["shift"] - 11.4) < 0.2 and f["tau"] == 11 and abs(f["amp"] - 0.8) < 0.04)
        cor, fits = fs.correct_row(w, (0, 0), [(1, 8)], {0: A, 1: B})
        check("correct_row: the corrected record is the own spike (error under a quarter of the noise energy)",
              ((cor - own) ** 2).sum() < 0.25 * 144 * NS * NC and fits[0]["accepted"])
        acc = sum(fs.fit_partner(A + noise(NS, NC), A, B, 10)["accepted"] for _ in range(200))
        check("NEGATIVE CONTROL: no partner in the row -> none accepted in 200 rows", acc == 0)
        qb = fs.shifted(B, 12.0)
        acc = sum(fs.fit_partner(A + 0.15 * qb + noise(NS, NC), A, B, 12)["accepted"] for _ in range(200))
        check("NEGATIVE CONTROL: a partner at 0.15 of its template -> none accepted in 200 rows", acc == 0)
        cor, fits = fs.correct_row(w, (0, 0), [(1, 8)], {0: A})
        check("correct_row: a partner without a template is skipped and the record is unchanged",
              not fits[0]["accepted"] and fits[0]["tau"] == 8 and np.array_equal(cor, w))

        # ── a group: 450 rows of each unit, 60 collisions ──
        n = 900
        clu = np.where(np.arange(n) % 2 == 0, 5, 9)
        base_w = np.stack([(rng.normal(1.0, 0.05) * (A if c == 5 else B) + noise(NS, NC)) for c in clu])
        spk = base_w.copy()
        coll = np.sort(rng.choice(n, 60, replace=False))
        true = rng.integers(9, 15, 60) * rng.choice([-1, 1], 60)
        stored = true + rng.integers(-2, 3, 60)
        for i, k in zip(coll, true):
            spk[i] += 0.9 * fs.shifted(B if clu[i] == 5 else A, float(k))
        spk16 = fs.to_int16(spk)
        ghost = np.setdiff1d(np.arange(n), coll)[:5]              # the .eap names a partner; the record has none
        cells = np.full((n, 4), nio.EAP_ABSENT, np.int8)
        cells[clu == 5, 0] = 0; cells[clu == 9, 2] = 0
        for i, s_ in zip(coll, stored):
            cells[i, 2 if clu[i] == 5 else 0] = s_
        for i in ghost:
            cells[i, 2 if clu[i] == 5 else 0] = 11
        home = fs.home_clusters(cells, clu)
        check("home_clusters: class 0 -> cluster 5, class 2 -> cluster 9, the others none", list(home) == [5, -1, 9, -1])
        ren = np.where(clu == 5, 31, 12)
        check("home_clusters follows a renumbering", list(fs.home_clusters(cells, ren)) == [31, -1, 12, -1])
        i0 = int(coll[0])
        own_m, partners = fs.split_row(cells[i0], int(clu[i0]), home)
        check("split_row: own = the class of the row's cluster, the other its partner",
              own_m == ((0, 0) if clu[i0] == 5 else (2, 0)) and len(partners) == 1 and partners[0][1] == stored[0])
        b = fs.build(spk16, cells, clu, write_offsets=True)
        st = b["stats"]
        check("build: exactly the 60 collisions are corrected; the 5 rows without a partner in them are not",
              np.array_equal(b["rows"], coll) and st["collision_rows"] == 65 and st["corrected"] == 60
              and st["partners_without_template"] == 0)
        e_raw = ((spk16[coll].astype(float) - base_w[coll]) ** 2).sum((1, 2))
        e_cor = ((b["waves"].astype(float) - base_w[coll]) ** 2).sum((1, 2))
        print(f"      corrected rows: error as a share of the raw row's: median {np.median(e_cor / e_raw):.3f}, worst {(e_cor / e_raw).max():.3f}")
        check("build: a corrected row is far closer to the record before the partner (median under 5 % of the raw error)",
              np.median(e_cor / e_raw) < 0.05 and (e_cor / e_raw).max() < 0.3)
        fitted = np.array([b["cells"][i, 2 if clu[i] == 5 else 0] for i in coll])
        check("build: the fitted offsets are the planted ones whatever the .eap said (at most 2 of 60 a sample off)",
              (np.abs(fitted - true) <= 1).all() and (fitted != true).sum() <= 2)
        check("build: without write_offsets the .eap matrix is the input", fs.build(spk16, cells, clu)["cells"] is cells)

        # ── the command ──
        base = os.path.join(td, "syn")
        (np.arange(n, dtype=np.int64) * 300 + 1000).tofile(f"{base}.res.1")
        spk16.tofile(f"{base}.spk.standard.1")
        nio.write_clu(base, 1, clu, variant="standard", tag="st")
        nio.write_eap(base, 1, cells, group=1, tag="st")
        with open(f"{base}.yaml", "w") as f:
            f.write("acquisitionSystem: {nChannels: 8, samplingRate: 32552, nBits: 16}\n"
                    "spikeDetection:\n  channelGroups:\n"
                    "    - channels: [0,1,2,3,4,5,6,7]\n      nSamples: 52\n      peakSampleIndex: 26\n      nFeatures: 3\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("FK_")}
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_spkc", base, "1", "--clu-method", "standard",
                            "--clu-stage", "st", "--write-offsets"], capture_output=True, text=True, env=env, timeout=600)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-2000:])
        g = nio.read_spkc(base, 1, variant="standard", tag="st")
        check("fiber-spkc writes <base>.spkc.standard.1.st with the same rows and records",
              r.returncode == 0 and g["ok"] and g["nSpikes"] == n and np.array_equal(g["rows"], coll)
              and np.array_equal(g["waves"], b["waves"]))
        check("--write-offsets: the stage's .eap holds the fitted offsets",
              np.array_equal(nio.read_eap(base, 1, tag="st")["cells"], b["cells"]))
        check("the .spk is unchanged", np.array_equal(np.fromfile(f"{base}.spk.standard.1", np.int16), spk16.ravel()))

    print(f"test_spkc: {ok} checks passed")


if __name__ == "__main__":
    main()
