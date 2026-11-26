"""fiber-unmix-children: a fiber-unmix stage as the .clu/.clc/.clp triple.

A made-up stage (no recording needed: the command reads the stage's .clu and
its manifest): units 2, 3, 4, noise 1, and `one` rows named for 2 and 3,
`collision` rows, `unexplained` rows.  Checked:
  * the one-unit rows are in their unit in the .clu and a child of their own
    in the .clc; the unit's own spikes are its self child (unit + 10000);
    every other row keeps its cluster and is that cluster's self child;
  * the .clp gives each child its parent at entry child - 1, and the headers
    are the distinct ids (.clu, .clc) and the entry count (.clp);
  * the same triple comes from the --one-as cluster, units and assign forms;
  * a row moved by hand since stays where it is;
  * --collisions: the parked collision rows become a third child, a collision
    already in a unit stays that unit's own spike;
  * the stage's .eap / .spkc appear under the new tag;
  * refusals: an output tag equal to the stage, a parent id of 10000 or more,
    a missing manifest.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fiber_kit import fiber_unmix_children as fuc     # noqa: E402
from fiber_kit import neuro_io as nio                 # noqa: E402

ok = 0


def check(what, cond):
    global ok
    if not cond:
        print(f"FAIL: {what}")
        sys.exit(1)
    ok += 1


def stage(n=1200, seed=3):
    """(base clu with the noise in 1, spike rows, kind, k1) -- 300 noise rows: 120 one (2 or 3),
    60 collisions (first unit 2 or 4), 120 unexplained."""
    rng = np.random.default_rng(seed)
    clu = rng.choice([2, 3, 4], n)
    noise = np.sort(rng.choice(n, 300, replace=False))
    clu[noise] = 1
    kind = np.array(["one"] * 120 + ["collision"] * 60 + ["unexplained"] * 120)
    k1 = np.r_[rng.choice([2, 3], 120), rng.choice([2, 4], 60), np.full(120, -1)]
    p = rng.permutation(300)
    return clu, noise, kind[p], k1[p]


def forms(clu, rows, kind, k1):
    """The stage's .clu in the three --one-as forms; half the collisions go into their unit."""
    one = rows[kind == "one"]; u1 = k1[kind == "one"]
    coll = rows[kind == "collision"]; uc = k1[kind == "collision"]
    out = {}
    for form in ("cluster", "units", "assign"):
        c = clu.copy()
        c[coll[::2]] = uc[::2]                    # confident: into the unit
        c[coll[1::2]] = 6                         # parked: the collision cluster
        if form == "cluster":
            c[one] = 5
        elif form == "units":
            c[one] = np.where(u1 == 2, 8, 9)
        else:
            c[one] = u1
        out[form] = c
    return out


def write(base, tag, clu, rows, kind, k1):
    nio.write_clu_file(f"{base}.clu.m.1.{tag}", clu)
    with open(f"{base}.unmix.m.1.{tag}.tsv", "w") as f:
        f.write("spike\tres\tkind\tk1\ta1\tdt1\tgain\tmargin\tresid\tk2\ta2\tdt2\tk2_detected\n")
        for r, kd, k in zip(rows, kind, k1):
            f.write(f"{r}\t{r * 10}\t{kd}\t{k}\t1\t0\t\t\t\t-1\t\t\t0\n")


def read(base, tag):
    return tuple(nio.read_clu_file(f"{base}.{t}.m.1.{tag}") for t in ("clu", "clc", "clp"))


def _bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def run(base, extra):
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"))
    return subprocess.run([sys.executable, "-m", "fiber_kit.fiber_unmix_children", base, "1", "--method", "m"] + extra,
                          capture_output=True, text=True, env=env, timeout=300)


def main():
    clu, rows, kind, k1 = stage()
    f = forms(clu, rows, kind, k1)
    one = rows[kind == "one"]; u1 = k1[kind == "one"]
    parked = rows[kind == "collision"][1::2]; up = k1[kind == "collision"][1::2]
    in_unit = rows[kind == "collision"][::2]

    parent, child, clp, st = fuc.build(f["cluster"], rows, kind, k1)
    check("the one-unit rows are in their unit", np.array_equal(parent[one], u1) and st["one"] == 120)
    rest = np.setdiff1d(np.arange(clu.size), one)
    check("every other row keeps its cluster", np.array_equal(parent[rest], f["cluster"][rest]))
    check("... and is that cluster's self child (unit + 10000; the noise is 1)",
          np.array_equal(child[rest], np.where(parent[rest] >= 2, parent[rest] + 10000, parent[rest])))
    check("the one-unit rows of a unit are one child, 20000 and up, one per unit",
          set(child[one].tolist()) == {20000, 20001} and len(set(zip(child[one].tolist(), u1.tolist()))) == 2
          and st["units"] == {2: 20000, 3: 20001})
    check("every child lies in one parent", fuc.check_nesting(parent, child))
    check("the .clp: each child's parent at entry child - 1, nothing else set",
          clp.size == 20001 and all(clp[c - 1] == p for c, p in set(zip(child.tolist(), parent.tolist())) if c >= 1)
          and np.count_nonzero(clp) == np.unique(child).size)
    check("the emptied one-unit cluster is gone from the parents", 5 not in parent)

    for form in ("units", "assign"):
        p2, c2, m2, _ = fuc.build(f[form], rows, kind, k1)
        check(f"--one-as {form}: the same triple", np.array_equal(p2, parent) and np.array_equal(c2, child)
              and np.array_equal(m2, clp))

    moved = f["cluster"].copy()
    moved[one[:7]] = 4                            # the curator put seven of them into unit 4
    p3, c3, _, s3 = fuc.build(moved, rows, kind, k1)
    keep = one[:7][u1[:7] != 4]
    check("rows moved by hand stay where they are, as that unit's own spikes",
          np.all(p3[keep] == 4) and np.all(c3[keep] == 10004) and s3["one_left"] == keep.size
          and s3["one"] == 120 - keep.size)

    p4, c4, m4, s4 = fuc.build(f["cluster"], rows, kind, k1, collisions=True)
    check("--collisions: the parked collisions go under their first unit as a third child",
          np.array_equal(p4[parked], up) and c4[parked].min() >= 20002 and s4["coll"] == parked.size
          and not (set(c4[parked].tolist()) & set(c4[one].tolist())) and 6 not in p4)
    check("--collisions: a collision already in its unit stays that unit's own spike",
          np.array_equal(c4[in_unit], p4[in_unit] + 10000) and fuc.check_nesting(p4, c4))
    check("without --collisions the collision cluster is left as it is", np.all(parent[parked] == 6))

    big = f["cluster"].copy(); big[big == 4] = 10004
    try:
        fuc.build(big, rows, kind, k1)
        check("a parent id of 10000 or more is refused", False)
    except ValueError as e:
        check("a parent id of 10000 or more is refused", "10000" in str(e))
    res = np.array([16_777_116 + 3]); r2 = f["cluster"].copy(); r2[r2 == 4] = res[0]
    k1r = np.where(k1 == 4, -1, k1)
    p5, c5, m5, _ = fuc.build(r2, rows, kind, k1r)
    check("a reserved chunk cluster is its own child", np.all(c5[p5 == res[0]] == res[0]) and m5[res[0] - 1] == res[0])

    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "syn")
        write(base, "um", f["units"], rows, kind, k1)
        for t in ("eap.1.um", "spkc.standard.1.um", "spkc.m.1.um"):
            with open(f"{base}.{t}", "wb") as fh:
                fh.write(t.encode())
        r = run(base, ["--stage", "um"])
        if r.returncode != 0:
            print(r.stderr[-2000:])
        check("the command exits 0", r.returncode == 0)
        (hu, gu), (hc, gc), (hp, gp) = read(base, "um_h")
        check("the files are build()'s", np.array_equal(gu, parent) and np.array_equal(gc, child) and np.array_equal(gp, clp))
        check("headers: distinct ids (.clu, .clc), entries (.clp)",
              hu == np.unique(parent).size and hc == np.unique(child).size and hp == clp.size)
        check("the stage's .eap and .spkc files are there under the new tag",
              all(_bytes(f"{base}.{t}_h") == t.encode() for t in ("eap.1.um", "spkc.standard.1.um", "spkc.m.1.um")))
        # ... as files of their own: Klusters rewrites its stage's .eap and .spkc in place
        with open(f"{base}.eap.1.um_h", "r+b") as fh:
            fh.write(b"X")
        check("they are copies: a write in place under the new tag leaves the stage's file as it was",
              _bytes(f"{base}.eap.1.um") == b"eap.1.um"
              and not any(os.path.samefile(f"{base}.{t}", f"{base}.{t}_h") for t in ("eap.1.um", "spkc.standard.1.um")))
        os.remove(f"{base}.eap.1.um_h"); os.link(f"{base}.eap.1.um", f"{base}.eap.1.um_h")
        r = run(base, ["--stage", "um"])
        check("a hard link an earlier version left under the new tag is replaced by a copy",
              r.returncode == 0 and not os.path.samefile(f"{base}.eap.1.um", f"{base}.eap.1.um_h")
              and _bytes(f"{base}.eap.1.um_h") == b"eap.1.um")
        check("the stage's own .clu is untouched", np.array_equal(nio.read_clu_file(f"{base}.clu.m.1.um")[1], f["units"]))
        r = run(base, ["--stage", "um", "--collisions", "--out-tag", "umc", "--no-collision-files"])
        check("--collisions --out-tag: written under that tag, no collision files copied",
              r.returncode == 0 and np.array_equal(read(base, "umc")[1][1], c4) and not os.path.exists(f"{base}.eap.1.umc"))
        r = run(base, ["--stage", "um", "--out-tag", "um"])
        check("an output tag equal to the stage is refused", r.returncode != 0 and "overwritten" in r.stderr)
        os.remove(f"{base}.unmix.m.1.um.tsv")
        r = run(base, ["--stage", "um"])
        check("without the manifest: refuses, naming the file", r.returncode != 0 and "um.tsv not found" in r.stderr)

    print(f"test_unmix_children: {ok} checks passed")


if __name__ == "__main__":
    main()
