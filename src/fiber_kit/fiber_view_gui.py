"""fiber-view-gui: a standalone, rotatable bundle viewer.

A selectable TABLE of bundles (one row per global fiber, sortable by drift score)
on the left; selecting a row renders that bundle on the right in a rotatable 3-D
view -- the per-chunk trajectories coloured by time plus the transparent drift
manifold lofted between consecutive chunks.

This is the interactive front-end over the tested data layer in `fiber_view`
(`bundle_table`, `bundle_drift_score`, `load_bundles_npz`).  It needs PySide6 +
pyqtgraph (`pip install 'fiber-kit[viz]'`); the heavy lifting (trajectories,
common-frame projection, drift score) lives in `fiber_view`, so this file is a
thin shell.  Run:  fiber-view-gui <session>.bundles.<group>.npz
"""
import sys
import numpy as np

from . import fiber_view as fv

try:
    from PySide6 import QtCore, QtWidgets
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from sklearn.decomposition import PCA
    _HAVE_GUI = True
except Exception:                                            # pragma: no cover
    _HAVE_GUI = False


def _need_gui():
    if not _HAVE_GUI:
        raise SystemExit("fiber-view-gui needs PySide6 + pyqtgraph: "
                         "pip install 'fiber-kit[viz]'")


def _loft_mesh(M, sub=8):
    """M: (nChunks, NPOS, 3) common-frame curve points -> (verts, faces, row_t)
    for a lofted quad surface, time-densified by `sub` between chunks."""
    nC = M.shape[0]
    tf = np.linspace(0, nC - 1, (nC - 1) * sub + 1)
    Mf = np.empty((len(tf), M.shape[1], 3))
    for s in range(M.shape[1]):
        for d in range(3):
            Mf[:, s, d] = np.interp(tf, np.arange(nC), M[:, s, d])
    R, Cc = Mf.shape[0], Mf.shape[1]
    verts = Mf.reshape(-1, 3)
    faces = []
    for i in range(R - 1):
        for j in range(Cc - 1):
            a = i * Cc + j; b = a + 1; c = a + Cc; d = c + 1
            faces.append([a, b, d]); faces.append([a, d, c])
    return verts, np.asarray(faces, int), np.repeat(tf / max(nC - 1, 1), Cc)


if _HAVE_GUI:
    def _viridis(t):
        return np.array(pg.colormap.get("viridis").map(float(t), mode="float"))

    class _BundleTableModel(QtCore.QAbstractTableModel):
        def __init__(self, rows):
            super().__init__()
            self.cols = list(fv._BUNDLE_COLS)
            self.rows = rows

        def rowCount(self, _=QtCore.QModelIndex()):
            return len(self.rows)

        def columnCount(self, _=QtCore.QModelIndex()):
            return len(self.cols)

        def data(self, idx, role=QtCore.Qt.DisplayRole):
            if role != QtCore.Qt.DisplayRole:
                return None
            v = self.rows[idx.row()][self.cols[idx.column()]]
            return f"{v:.3f}" if isinstance(v, float) else str(v)

        def headerData(self, sec, orient, role=QtCore.Qt.DisplayRole):
            if role == QtCore.Qt.DisplayRole and orient == QtCore.Qt.Horizontal:
                return self.cols[sec]
            return None

    class FiberViewWindow(QtWidgets.QMainWindow):
        def __init__(self, bundles):
            super().__init__()
            self.setWindowTitle("fiber-view — bundles")
            self.bundles = {b["gid"]: b for b in bundles}
            rows = fv.bundle_table(bundles)
            split = QtWidgets.QSplitter()
            self.table = QtWidgets.QTableView()
            self.table.setModel(_BundleTableModel(rows))
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.selectionModel().selectionChanged.connect(self._on_select)
            self.view = gl.GLViewWidget()
            self.view.setCameraPosition(distance=6)
            self.view.addItem(gl.GLAxisItem())
            split.addWidget(self.table); split.addWidget(self.view)
            split.setStretchFactor(1, 1)
            self.setCentralWidget(split)
            self._rows = rows
            if rows:
                self.table.selectRow(0)

        def _clear_view(self):
            for it in list(self.view.items):
                if not isinstance(it, gl.GLAxisItem):
                    self.view.removeItem(it)

        def _on_select(self, *_):
            sel = self.table.selectionModel().selectedRows()
            if not sel:
                return
            gid = self._rows[sel[0].row()]["id"]
            self._plot(self.bundles[gid])

        def _plot(self, bundle):
            self._clear_view()
            cv = bundle["curves"]; nC = len(cv)
            fp = PCA(3).fit(np.vstack(cv))
            M = np.stack([fp.transform(c) for c in cv], 0)
            ctr = M.reshape(-1, 3).mean(0); M = M - ctr
            for w in range(nC):
                col = _viridis(w / max(nC - 1, 1))
                self.view.addItem(gl.GLLinePlotItem(pos=M[w], color=col, width=3, antialias=True))
            if nC >= 2:
                verts, faces, rt = _loft_mesh(M)
                vc = np.array([_viridis(t) for t in rt]); vc[:, 3] = 0.28
                mesh = gl.GLMeshItem(vertexes=verts, faces=faces, vertexColors=vc,
                                     smooth=True, drawEdges=False, glOptions="translucent")
                self.view.addItem(mesh)


def main():
    _need_gui()
    import argparse
    ap = argparse.ArgumentParser(prog="fiber-view-gui",
                                 description="Rotatable bundle viewer: selectable table -> 3-D plot.")
    ap.add_argument("bundles", help="a .bundles.<group>.npz written by fiber-refine --chunk-minutes")
    a = ap.parse_args()
    bundles = fv.load_bundles_npz(a.bundles)
    app = QtWidgets.QApplication(sys.argv)
    win = FiberViewWindow(bundles)
    win.resize(1100, 700); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
