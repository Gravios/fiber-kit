"""fiber-config-edit: the Qt-free model of the fiber-kit.yaml form editor.

  * LOSSLESS: every shipped config round-trips byte for byte, and an edit changes only the lines it touched
    (the comment column kept where the padding allows, quotes kept or added only where YAML needs them);
  * typing from the comment lead: switches, choice lists, numbers with a documented default and range,
    placeholders (`"" | path`) left as text;
  * off and gating: an off knob is dimmed and so is every FK_X_* knob under an off FK_X;
  * sections follow the file's banners; the embedded plan's steps toggle by commenting, and toggle back to
    the original line exactly;
  * reset, changes(), diff(), validate() and save() (atomic, then the saved state is the new baseline);
  * reflow keeps aligned tables preformatted and joins prose;
  * CRLF files keep their line endings;
  * --check on a shipped file; the window builds off-screen when PySide6 is present.
"""
import glob, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fiber_kit import fiber_config_editor as ce   # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..", "src", "fiber_kit")
fails = 0
def check(ok, what):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + what); fails += not ok


SAMPLE = '''# ============================================================================
#  sample.yaml -- a note block
#  second line of the note
# ============================================================================
pipeline:
  - {stage: fiber-session,        name: session,         out: fiber_session}
  - {stage: fiber-polish,         name: polish,          in: fiber_session, out: fs_polish}   # a BRANCH
  # an alternative:
  # - {stage: fiber-rdr,            name: rdr,             in: fs_polish}

# ============================================================================
#  [1] FIBER-SESSION -- first section
# ============================================================================
FK_A_CHUNK_MIN:          18       # 12 | 8-20 (min).  Chunk length.
                                  #   continued help line
FK_A_GATE:               0        # 0 | 0/1.  1 = turn the gate on.
FK_A_GATE_THR:           0.35     # 0.3 | 0.2-0.5.  Only read when the gate is on.
# ----- a sub-heading ---------------------------------------------------------
FK_A_SOURCE:             auto     # auto | nst | fil | spk.  Where the noise comes from.
FK_A_CHUNKS:             ""       # "" = every chunk | e.g. 14-16 or 3,7 (0-based).  Which chunks.
FK_A_SIGNAL:             ""       # "" | path.  A file.
FK_A_THR:                0        # 0 | 0 or 2.5-4.  Tail multiplier (0 = term off).
FK_A_MODE:               "off"    # off | off/strip/knn/both.  Shared default.
FK_A_EMIT:               1        # 0 = off, 1 = on.  Write the table.
FK_A_PLAIN:              7
# a free note between knobs
#   col a     col b
#   1         2
FK_A_AFTER:              "x y"  # stages after which to re-run
'''


def main():
    # ── every shipped config round-trips exactly ─────────────────────────────
    shipped = sorted(glob.glob(os.path.join(HERE, "fiber-kit*.yaml")))
    check(len(shipped) >= 5, f"{len(shipped)} shipped configs found")
    for p in shipped:
        raw = open(p, encoding="utf-8", newline="").read()
        d = ce.ConfigDoc(raw)
        n = sum(1 for ln in raw.split("\n") if ln.startswith("FK_"))
        check(d.text() == raw and len(d.knobs) == n and not d.validate(),
              f"{os.path.basename(p)}: {len(d.knobs)} knobs read, text reproduced byte for byte, YAML valid")
    exp = ce.ConfigDoc.load(os.path.join(HERE, "fiber-kit-exp.yaml"))
    check(sum(len(s.steps) for s in exp.sections) > 20 and any(s.title.startswith("[H] FIBER-POLISH") for s in exp.sections),
          "the exp profile: plan steps found, sections titled by their banners")
    kinds = {k.kind for k in exp.knobs.values()}
    check(kinds == {"bool", "enum", "number", "text"}, "all four field kinds occur in the exp profile")
    check(exp.knobs["FK_SESSION_WHITEN_SOURCE"].choices == ("auto", "nst", "fil", "spk")
          and exp.knobs["FK_SESSION_CHUNK_MIN"].default == "12" and exp.knobs["FK_SESSION_CHUNK_MIN"].lo == 8.0
          and exp.knobs["FK_POLISH_ISLAND"].kind == "bool", "exp profile: a choice list, a default with a range, a switch")
    check(exp.gate_of(exp.knobs["FK_PSAME_GEO_W_RC"]) is exp.knobs["FK_PSAME_GEO"], "exp profile: FK_PSAME_GEO off gates FK_PSAME_GEO_W_RC")

    # ── the sample: structure and typing ─────────────────────────────────────
    d = ce.ConfigDoc(SAMPLE)
    k = d.knobs
    check(d.text() == SAMPLE, "sample round-trips")
    check([s.title[:14] for s in d.sections] == ["sample.yaml --", "PIPELINE PLAN ", "[1] FIBER-SESS"], "three sections: note block, plan, [1]")
    steps = d.sections[1].steps
    check([s.enabled for s in steps] == [True, True, False] and steps[1].fields == {"stage": "fiber-polish", "name": "polish", "in": "fiber_session", "out": "fs_polish"},
          "plan: two live steps and one commented, fields read")
    check(k["FK_A_CHUNK_MIN"].kind == "number" and k["FK_A_CHUNK_MIN"].default == "12" and (k["FK_A_CHUNK_MIN"].lo, k["FK_A_CHUNK_MIN"].hi) == (8.0, 20.0)
          and "continued help line" in k["FK_A_CHUNK_MIN"].help and k["FK_A_CHUNK_MIN"].off_default, "number: default, range, continuation help, differs from default")
    check(k["FK_A_GATE"].kind == "bool" and k["FK_A_GATE"].is_off() and k["FK_A_EMIT"].kind == "bool" and not k["FK_A_EMIT"].is_off(), "switches: `0 | 0/1` and `0 = off, 1 = on`")
    check(k["FK_A_SOURCE"].kind == "enum" and k["FK_A_SOURCE"].choices == ("auto", "nst", "fil", "spk"), "choice list")
    check(k["FK_A_MODE"].kind == "enum" and k["FK_A_MODE"].choices == ("off", "strip", "knn", "both") and k["FK_A_MODE"].is_off(), "`off | off/strip/knn/both` is a choice list, and off")
    check(k["FK_A_SIGNAL"].kind == "text" and k["FK_A_CHUNKS"].kind == "text" and k["FK_A_CHUNKS"].default == "" and not k["FK_A_CHUNKS"].is_off(),
          '`"" | path` and `"" = every chunk` stay text; an empty value with no off meaning is not off')
    check(k["FK_A_THR"].is_off() and not k["FK_A_PLAIN"].is_off() and k["FK_A_PLAIN"].help == "" and k["FK_A_PLAIN"].default is None, "0 with `(0 = term off)` is off; a bare knob has no help and no default")
    check(d.gate_of(k["FK_A_GATE_THR"]) is k["FK_A_GATE"] and d.is_dimmed(k["FK_A_GATE_THR"]) and d.gate_of(k["FK_A_CHUNK_MIN"]) is None, "FK_A_GATE off dims FK_A_GATE_THR, not its siblings")
    notes = [i for i in d.sections[2].items if isinstance(i, ce.Note)]
    check(len(notes) == 2 and notes[0].sub and notes[0].text == "a sub-heading" and "col a     col b" in notes[1].text, "a sub-heading and a free note are kept as items")
    check(k["FK_A_AFTER"].value == "x y" and k["FK_A_AFTER"].quote == '"', "a quoted value with a tight comment")

    # ── edits touch one line each, the comment column stays ─────────────────
    before = d.text().split("\n")
    check(d.set("FK_A_CHUNK_MIN", "8") and not d.set("FK_A_CHUNK_MIN", "8"), "set returns whether anything changed")
    d.set("FK_A_GATE", "1"); d.set("FK_A_SOURCE", "nst"); d.set("FK_A_CHUNKS", "14-16"); d.set("FK_A_MODE", "strip"); d.set("FK_A_SIGNAL", "/data/a b.fil")
    after = d.text().split("\n")
    diff = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    check(len(before) == len(after) and len(diff) == 6, f"six edits changed six lines ({len(diff)})")
    L = {x.split(":")[0]: x for x in after if x.startswith("FK_")}
    check(L["FK_A_CHUNK_MIN"] == "FK_A_CHUNK_MIN:          8        # 12 | 8-20 (min).  Chunk length.", "a shorter value: the comment stays in its column")
    check(L["FK_A_CHUNKS"].startswith('FK_A_CHUNKS:             "14-16"  # "" = every chunk'), "a longer value eats padding; the original quotes are kept")
    check(L["FK_A_SOURCE"].startswith("FK_A_SOURCE:             nst      # auto"), "an unquoted word stays unquoted")
    check('FK_A_SIGNAL:             "/data/a b.fil"  # "" | path' in L["FK_A_SIGNAL"], "a wider value keeps two spaces before the comment")
    check(not d.gate_of(k["FK_A_GATE_THR"]) and not d.is_dimmed(k["FK_A_GATE_THR"]), "turning the gate on un-dims what it gated")
    d.set("FK_A_PLAIN", "a: b"); d.set("FK_A_THR", "")
    check(d.lines[k["FK_A_PLAIN"].line] == 'FK_A_PLAIN:              "a: b"' and '""' in d.lines[k["FK_A_THR"].line] and k["FK_A_THR"].value == "",
          "a value YAML would misread is quoted; an emptied value is written as \"\"")
    check(not d.validate(), "the edited text is valid YAML with every knob present")
    d.set("FK_A_CHUNK_MIN", "25")
    check(k["FK_A_CHUNK_MIN"].out_of_range and not ce.ConfigDoc(SAMPLE).knobs["FK_A_CHUNK_MIN"].out_of_range, "25 is outside 8-20; 18 is not")
    names = [c[0] for c in d.changes()]
    check(len(names) == 8 and ("FK_A_CHUNK_MIN", "18", "25") in d.changes(), f"changes() lists the eight edited knobs with old and new values ({len(names)})")
    for n in names:
        d.reset(n)
    check(d.text() == SAMPLE and not d.changes(), "reset of every edit restores the file exactly")

    # ── plan steps ───────────────────────────────────────────────────────────
    d.set_step(steps[1], False); d.set_step(steps[2], True)
    check(d.lines[steps[1].line] == "  # - {stage: fiber-polish,         name: polish,          in: fiber_session, out: fs_polish}   # a BRANCH"
          and d.lines[steps[2].line] == "  - {stage: fiber-rdr,            name: rdr,             in: fs_polish}", "unticking comments a step out (its trailing comment kept); ticking uncomments one")
    check(not d.validate() and "+  # - {stage: fiber-polish" in d.diff(), "the toggled plan is valid YAML and shows in the diff")
    d.set_step(steps[1], True); d.set_step(steps[2], False)
    check(d.text() == SAMPLE, "toggling back restores the original lines exactly")

    # ── save: atomic, then the baseline moves; a broken edit is refused ─────
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c.yaml")
        open(p, "w", newline="").write(SAMPLE)
        d = ce.ConfigDoc.load(p)
        d.set("FK_A_GATE_THR", "0.4"); d.save(p)
        check(open(p, newline="").read() == d.text() and not d.changes() and os.listdir(tmp) == ["c.yaml"]
              and ce.ConfigDoc.load(p).knobs["FK_A_GATE_THR"].value == "0.4", "save writes the text, leaves no temp file, and clears the change list")
        d.lines[d.knobs["FK_A_PLAIN"].line] = "FK_A_PLAIN: [unclosed"
        try:
            d.save(p); ok = False
        except ValueError:
            ok = True
        check(ok and "FK_A_PLAIN:              7" in open(p).read(), "text that is not valid YAML is refused and the file left alone")
        crlf = SAMPLE.replace("\n", "\r\n")
        c = ce.ConfigDoc(crlf); c.set("FK_A_PLAIN", "9")
        check(ce.ConfigDoc(crlf).text() == crlf and c.text().count("\r\n") == crlf.count("\r\n") and "\r\r" not in c.text(), "CRLF line endings survive")
        q = os.path.join(HERE, "fiber-kit.yaml")
        env = dict(os.environ, PYTHONPATH=os.path.join(HERE, ".."))
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_config_editor", "--check", q], capture_output=True, text=True, env=env)
        check(r.returncode == 0 and "round trip exact" in r.stdout, "--check on the shipped fiber-kit.yaml")
        r = subprocess.run([sys.executable, "-m", "fiber_kit.fiber_config_editor", os.path.join(tmp, "nosuch.yaml")], capture_output=True, text=True, env=env)
        check(r.returncode != 0 and "not found" in (r.stdout + r.stderr), "a missing file is refused before Qt is imported")

    # ── reflow ───────────────────────────────────────────────────────────────
    blocks = ce.reflow("first line of prose\ncontinues here.  Two sentences.\n  col a     col b\n  1         2\nafter the table")
    check([b[0] for b in blocks] == ["p", "pre", "p"] and blocks[0][1] == "first line of prose continues here.  Two sentences."
          and blocks[1][1].count("\n") == 1, "reflow joins prose and keeps the aligned rows preformatted")

    # ── the window, off-screen ───────────────────────────────────────────────
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
    except ImportError:
        print("  skip: PySide6 not installed -- the window is not built")
    else:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "c.yaml")
            open(p, "w", newline="").write(SAMPLE)
            w = ce._build_window(p)
            w.show(); app.processEvents()
            dk = w.doc.knobs
            row = w.rows[id(dk["FK_A_GATE_THR"])]
            check(len(w.boxes) == 3 and row.name.property("dim") is True and "FK_A_GATE is off" in row.hint.text(), "window: three sections; a gated knob is dimmed and says why")
            w.rows[id(dk["FK_A_GATE"])].ed.setChecked(True); app.processEvents()
            check(dk["FK_A_GATE"].value == "1" and row.name.property("dim") is False and w.windowTitle().startswith("●")
                  and "1 edited" in w.boxes[2].head.text(), "window: ticking the switch edits the doc, un-dims the gated row, marks the title and the section")
            w.show_help(dk["FK_A_CHUNK_MIN"])
            check("continued help line" in w.help.toPlainText() and "8-20" in w.help.toPlainText(), "window: the help pane shows a knob's full comment and range")
            w.search.setText("tail multiplier"); app.processEvents()
            vis = [r.item.name for r in w.rows.values() if isinstance(r.item, ce.Knob) and not r.isHidden()]
            check(vis == ["FK_A_THR"] and w.boxes[1].isHidden(), "window: the filter searches help text and hides sections with no match")
            w.search.setText(""); w.revert_all(); app.processEvents()
            check(not w.doc.changes() and w.doc.text() == SAMPLE, "window: revert all restores the file")
            w.doc.set("FK_A_PLAIN", "8"); check(w._write(p) and "FK_A_PLAIN:              8" in open(p).read(), "window: save writes the file")
            w.close()

    print(f"test_config_editor: {'PASS' if not fails else 'FAIL'} ({fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
